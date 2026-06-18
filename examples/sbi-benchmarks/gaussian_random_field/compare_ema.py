"""Compare a trained GRF-32 checkpoint's samples WITH vs WITHOUT EMA weights.

``pipeline.sample()`` defaults to ``use_ema=True``. The EMA weights of some
runs have been observed to degenerate into white-noise / flat-P(k) samples
even when the raw (non-EMA) weights are good. This standalone, reusable tool
restores a saved checkpoint (it never trains), dumps a raw-vs-EMA comparison
(field grids + radial power spectra), and reports a few degeneracy metrics
(parameter L2 norms, conditional flow-matching loss raw-vs-EMA, low-/high-k
P(k) ratio) so the failure can be diagnosed for any experiment.

The model build, data loaders, normalization, and plotting are replicated
EXACTLY from ``train-grf.py`` (same field size, same dataset config) so the
output plots match the style of ``imgs/grf_fields_conf{N}.png`` /
``imgs/grf_pk_conf{N}.png``.

Outputs (per experiment ``N`` from the config):
    imgs/grf_fields_conf{N}_raw.png   imgs/grf_pk_conf{N}_raw.png
    imgs/grf_fields_conf{N}_ema.png   imgs/grf_pk_conf{N}_ema.png

Usage (conda env ``gensbi``):
    python compare_ema.py --config config/config_5.yaml
    python compare_ema.py --config config/config_5.yaml --nsamples 8 --step_size 0.02
    python compare_ema.py --config config/config_5.yaml --device cpu
"""

import os
import argparse

# Platform setup MUST happen before importing jax. Honor a pre-set
# JAX_PLATFORMS; default to cuda (GPU if available); --device overrides.
if __name__ == "__main__":
    _ap = argparse.ArgumentParser(add_help=False)
    _ap.add_argument("--device", default=None, choices=["cpu", "cuda"])
    _known, _ = _ap.parse_known_args()
    if _known.device is not None:
        os.environ["JAX_PLATFORMS"] = _known.device
    else:
        os.environ.setdefault("JAX_PLATFORMS", "cuda")  # GPU if available; JAX_PLATFORMS=cpu wins
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".95")
else:
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax
from jax import numpy as jnp
import numpy as np
from flax import nnx
import yaml

import matplotlib

matplotlib.use("Agg")  # headless cluster nodes
import matplotlib.pyplot as plt

from gensbi.core import FlowMatchingMethod
from gensbi.experimental.recipes import FieldConditionalPipeline

from sbibm_jax.data import OnlineTaskDataset, TaskDataset

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_NAME = "gaussian_random_field"  # 32x32 field

# training_config keys exposed in the yaml `training:` section; the rest of
# that section (batch sizes, flags, seed) is script-level.
_PIPELINE_KEYS = (
    "nsteps", "max_lr", "min_lr", "warmup_steps", "ema_decay",
    "decay_transition", "val_every", "early_stopping", "multistep",
    "experiment_id", "val_error_ratio",
)


# --------------------------------------------------------------------------
# Helpers copied verbatim from train-grf.py (duplication is fine for example
# scripts; keeps compare_ema.py self-contained).
# --------------------------------------------------------------------------
def swap_obs_cond(batch):
    """Loader yields (theta, x); the field pipeline wants (obs=x, cond=theta)."""
    theta, x = batch
    return x, theta


def resolve_model_section(cfg):
    """Return (model_kind, model_cfg) from the yaml: 'pixeldit' or 'fielddit'."""
    for kind in ("pixeldit", "fielddit"):
        if kind in cfg:
            return kind, cfg[kind]
    raise KeyError("config must have a 'pixeldit:' or 'fielddit:' section")


def build_model(model_kind, model_cfg, seed):
    kw = dict(model_cfg)
    kw["field_shape"] = tuple(kw["field_shape"])
    kw["param_dtype"] = getattr(jnp, kw.pop("param_dtype", "bfloat16"))
    if model_kind == "pixeldit":
        from gensbi.experimental.models import PixelDiT, PixelDiTParams

        return PixelDiT(PixelDiTParams(rngs=nnx.Rngs(seed), **kw))
    else:
        from gensbi.experimental.models import FieldDiT, FieldDiTParams

        kw["encoder_widths"] = tuple(kw["encoder_widths"])
        return FieldDiT(FieldDiTParams(rngs=nnx.Rngs(seed), **kw))


def radial_power_spectrum(field, nbins=40):
    """Isotropic P(k) of a 2D field; k in cycles/pixel (Nyquist = 0.5)."""
    field = np.asarray(field, dtype=np.float64)
    H, W = field.shape
    pk2d = np.abs(np.fft.fft2(field)) ** 2 / (H * W)
    kx, ky = np.meshgrid(np.fft.fftfreq(H), np.fft.fftfreq(W), indexing="ij")
    knorm = np.sqrt(kx**2 + ky**2).ravel()
    kbins = np.geomspace(knorm[knorm > 0].min(), 0.5, nbins + 1)
    counts, _ = np.histogram(knorm, kbins)
    power, _ = np.histogram(knorm, kbins, weights=pk2d.ravel())
    kcent = np.sqrt(kbins[1:] * kbins[:-1])
    good = counts > 0
    return kcent[good], power[good] / counts[good]


def plot_field_grid(truths, samples, thetas, n_show, path):
    """One row per theta: true field | n_show posterior samples."""
    n = len(truths)
    fig, axes = plt.subplots(
        n, n_show + 1, figsize=(3 * (n_show + 1), 3.2 * n), squeeze=False
    )
    for i in range(n):
        vmax = float(np.percentile(np.abs(truths[i]), 99.5))
        axes[i][0].imshow(truths[i], vmin=-vmax, vmax=vmax, cmap="coolwarm")
        axes[i][0].set_title(
            f"truth | log_std={thetas[i, 0]:.2f}, alpha={thetas[i, 1]:.2f}",
            fontsize=9,
        )
        for j in range(n_show):
            axes[i][j + 1].imshow(
                samples[i][j], vmin=-vmax, vmax=vmax, cmap="coolwarm"
            )
            axes[i][j + 1].set_title(f"sample {j + 1}", fontsize=9)
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_power_spectra(truths, samples, thetas, path):
    """Per theta: mean P(k) +/- 1 sigma over samples vs the true field."""
    n = len(truths)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 3.8), squeeze=False)
    for i in range(n):
        ax = axes[0][i]
        k, pk_true = radial_power_spectrum(truths[i])
        pks = np.stack([radial_power_spectrum(s)[1] for s in samples[i]])
        mean, std = pks.mean(axis=0), pks.std(axis=0)
        ax.loglog(k, pk_true, "k-", label="truth")
        ax.loglog(k, mean, "C0-", label="samples (mean)")
        ax.fill_between(
            k, np.maximum(mean - std, 1e-20), mean + std, color="C0", alpha=0.3
        )
        ax.set_title(
            f"log_std={thetas[i, 0]:.2f}, alpha={thetas[i, 1]:.2f}", fontsize=9
        )
        ax.set_xlabel("k [cycles/pixel]")
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel("P(k)")
    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Diagnostics specific to compare_ema.py
# --------------------------------------------------------------------------
def param_l2_norm(model):
    """Global L2 norm over all nnx.Param leaves (computed in float32)."""
    leaves = jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))
    sq = sum(float(jnp.sum(jnp.square(leaf.astype(jnp.float32)))) for leaf in leaves)
    return float(np.sqrt(sq))


def fm_loss_on_batch(pipeline, model, obs, cond, key):
    """Conditional FM loss for `model` on a single (obs, cond) batch.

    Uses the pipeline's own loss path (method.prepare_batch -> FMLoss), so the
    time distribution (uniform / logitnormal per the config's time_sampling
    block), x0 ~ N(0, I), xt = (1-t) x0 + t x1, and target dx_t = x1 - x0 all
    match training exactly. Returns (loss, predict_zero_ref) where the
    predict-zero reference is mean((x1 - x0)^2) on the SAME prepared batch.
    """
    prepared = pipeline.method.prepare_batch(key, obs, pipeline.path)
    model_extras = {
        "cond": cond,
        "obs_ids": pipeline.obs_ids,
        "cond_ids": pipeline.cond_ids,
    }
    loss = float(pipeline.loss_obj(model, prepared, model_extras=model_extras))
    x_0, x_1, _ = prepared
    predict_zero = float(jnp.mean(jnp.square(x_1.astype(jnp.float32)
                                             - x_0.astype(jnp.float32))))
    return loss, predict_zero


def pk_ratio(field, nbins=40):
    """Low-k / high-k P(k) ratio: mean over the lowest third vs highest third
    of the populated radial bins. A flat (white-noise) spectrum -> ratio ~ 1;
    a steep GRF spectrum -> ratio >> 1."""
    _, pk = radial_power_spectrum(field, nbins=nbins)
    n = len(pk)
    third = max(1, n // 3)
    low = float(np.mean(pk[:third]))
    high = float(np.mean(pk[-third:]))
    return low / high if high > 0 else float("inf")


def main(config_path, nsamples_override=None, step_size_override=None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    tcfg = cfg["training"]
    scfg = cfg["sampling"]
    experiment = tcfg["experiment_id"]

    imgs_dir = os.path.join(EXAMPLE_DIR, "imgs")
    os.makedirs(imgs_dir, exist_ok=True)

    nsamples = nsamples_override if nsamples_override is not None else scfg["nsamples"]
    step_size = (
        step_size_override if step_size_override is not None else scfg["step_size"]
    )
    nsamples_grid = min(scfg["nsamples_grid"], nsamples)

    print(f"Backend: {jax.default_backend()} | devices: {jax.devices()}")
    print(f"Config: {config_path} (experiment {experiment})")
    print(f"nsamples={nsamples}, step_size={step_size}, nsamples_grid={nsamples_grid}")

    # --- data (mirrors train-grf.py exactly) ---
    task = OnlineTaskDataset(
        TASK_NAME,
        normalize=True,
        dtype=jnp.bfloat16,
    )
    offline_task = TaskDataset(
        TASK_NAME,
        normalize=True,
        dtype=jnp.bfloat16,
    )
    train_loader = task.get_online_train_loader(tcfg["batch_size"]).map(swap_obs_cond)
    val_loader = offline_task.get_val_loader(tcfg["val_batch_size"]).map(swap_obs_cond)

    # --- model + pipeline (mirrors train-grf.py exactly) ---
    model_kind, model_cfg = resolve_model_section(cfg)
    model = build_model(model_kind, model_cfg, seed=tcfg.get("seed", 0))
    n_params = sum(
        leaf.size
        for leaf in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))
    )
    print(f"{model_kind} parameters: {n_params / 1e6:.1f}M")

    training_config = FieldConditionalPipeline.get_default_training_config()
    training_config.update({k: tcfg[k] for k in _PIPELINE_KEYS if k in tcfg})
    # FieldDiT -> checkpoints/, PixelDiT -> checkpoints_pixeldit/.
    ckpt_dirname = "checkpoints_pixeldit" if model_kind == "pixeldit" else "checkpoints"
    training_config["checkpoint_dir"] = os.path.join(EXAMPLE_DIR, ckpt_dirname)

    ts_cfg = cfg.get("time_sampling", {})
    method = FlowMatchingMethod(
        time_dist=ts_cfg.get("dist", "uniform"),
        logitnorm_mean=ts_cfg.get("logitnorm_mean", 0.0),
        logitnorm_std=ts_cfg.get("logitnorm_std", 1.0),
    )

    pipeline = FieldConditionalPipeline(
        model,
        train_loader,
        val_loader,
        field_shape=tuple(model_cfg["field_shape"]),
        dim_cond=model_cfg["cond_dim"],
        method=method,
        ch_obs=1,
        ch_cond=1,
        training_config=training_config,
    )

    # --- restore (never train) ---
    pipeline.restore_model()  # restores both raw and EMA; calls _wrap_model()
    pipeline._wrap_model()

    # --- sanity: parameter L2 norms ---
    raw_norm = param_l2_norm(pipeline.model)
    ema_norm = param_l2_norm(pipeline.ema_model)
    print("\n=== Param L2 norms ===")
    print(f"  raw model : {raw_norm:.6g}")
    print(f"  ema model : {ema_norm:.6g}")

    # --- degeneracy metric: conditional FM loss raw vs EMA ---
    val_batch = next(iter(val_loader))  # (obs, cond)
    obs, cond = val_batch
    loss_key = jax.random.PRNGKey(tcfg.get("seed", 0))
    raw_loss, predict_zero = fm_loss_on_batch(
        pipeline, pipeline.model, obs, cond, loss_key
    )
    ema_loss, _ = fm_loss_on_batch(
        pipeline, pipeline.ema_model, obs, cond, loss_key
    )
    print("\n=== Conditional FM loss on a val batch (same prepared batch) ===")
    print(f"  raw-loss          : {raw_loss:.6g}")
    print(f"  ema-loss          : {ema_loss:.6g}")
    print(f"  predict-zero ref  : {predict_zero:.6g}")

    # --- posterior samples for a few test thetas (mirrors train-grf.py) ---
    n_thetas = scfg["num_thetas"]
    rows = offline_task.df_test[:n_thetas]
    thetas_raw = np.asarray(rows["thetas"], dtype=np.float32)  # (n, 2)
    truths = np.asarray(rows["xs"], dtype=np.float32)          # (n, H, W)
    theta_norm = np.asarray(task.normalize_theta(thetas_raw[..., None]))  # (n, 2, 1)

    written = []
    pk_summary = {}  # tag -> per-theta (low/high) ratios
    for use_ema, tag in ((False, "raw"), (True, "ema")):
        samples = []
        key = jax.random.PRNGKey(tcfg.get("seed", 0))  # SAME PRNG for both tags
        for i in range(n_thetas):
            key, sub = jax.random.split(key)
            s = pipeline.sample(
                sub,
                jnp.asarray(theta_norm[i : i + 1]),  # (1, 2, 1)
                nsamples=nsamples,
                step_size=step_size,
                use_ema=use_ema,
            )  # (nsamples, H, W, 1), normalized
            s = np.asarray(task.unnormalize_x(s), dtype=np.float32)[..., 0]
            samples.append(s)
            print(f"[{tag}] theta {i}: sampled {s.shape}, finite={np.isfinite(s).all()}")

        fields_path = os.path.join(imgs_dir, f"grf_fields_conf{experiment}_{tag}.png")
        pk_path = os.path.join(imgs_dir, f"grf_pk_conf{experiment}_{tag}.png")
        plot_field_grid(truths, samples, thetas_raw, nsamples_grid, fields_path)
        plot_power_spectra(truths, samples, thetas_raw, pk_path)
        written += [fields_path, pk_path]

        # per-theta low/high P(k) ratio, averaged over samples
        ratios = [float(np.mean([pk_ratio(s) for s in samples[i]]))
                  for i in range(n_thetas)]
        pk_summary[tag] = ratios

    truth_ratios = [pk_ratio(truths[i]) for i in range(n_thetas)]

    print("\n=== Low-k / high-k P(k) ratio (lowest-third vs highest-third bins) ===")
    print("    (flat/white-noise -> ~1; steep GRF -> >> 1)")
    for i in range(n_thetas):
        print(
            f"  theta {i}: truth={truth_ratios[i]:.4g}  "
            f"raw={pk_summary['raw'][i]:.4g}  ema={pk_summary['ema'][i]:.4g}"
        )
    print(
        f"  mean   : truth={np.mean(truth_ratios):.4g}  "
        f"raw={np.mean(pk_summary['raw']):.4g}  ema={np.mean(pk_summary['ema']):.4g}"
    )

    print("\n=== Written plots ===")
    for p in written:
        print(f"  {p}")
    print(f"\nDone (experiment {experiment}; raw and ema).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config_5.yaml")
    parser.add_argument("--nsamples", type=int, default=None,
                        help="override sampling.nsamples (e.g. for CPU speed)")
    parser.add_argument("--step_size", type=float, default=None,
                        help="override sampling.step_size (e.g. 0.02 for CPU)")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"],
                        help="JAX platform; set BEFORE importing jax (see top of file)")
    args = parser.parse_args()
    main(args.config, nsamples_override=args.nsamples,
         step_size_override=args.step_size)
