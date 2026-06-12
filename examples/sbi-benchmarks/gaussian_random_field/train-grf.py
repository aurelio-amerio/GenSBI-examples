"""Train FieldDiT or PixelDiT on gaussian_random_field (32x32), sample the posterior.

Field-level NPE: the model learns p(field | theta) with conditional flow
matching. The 32x32 GRF realization is the generation target (obs); theta =
(log_std, alpha) is the conditioning vector. Outputs: loss curves, a
truth-vs-samples field grid, and radial power-spectrum overlays in imgs/.

The model is chosen by the config's model section: a `fielddit:` key builds
FieldDiT (configs 1-3), a `pixeldit:` key builds PixelDiT (config 1b).

Usage (conda env `gensbi`):
    python train-grf.py --config config/config_1.yaml
    python train-grf.py --config config/config_1b.yaml
"""

import os

if __name__ != "__main__":
    os.environ["JAX_PLATFORMS"] = "cpu"
else:
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"
    os.environ.setdefault("JAX_PLATFORMS", "cuda")  # JAX_PLATFORMS=cpu wins

import argparse

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

# training_config keys exposed in the yaml `training:` section; the rest of
# that section (batch sizes, flags, seed) is script-level.
_PIPELINE_KEYS = (
    "nsteps", "max_lr", "min_lr", "warmup_steps", "ema_decay",
    "decay_transition", "val_every", "early_stopping", "multistep",
    "experiment_id", "val_error_ratio",
)


def swap_obs_cond(batch):
    """Loader yields (theta, x); the field pipeline wants (obs=x, cond=theta).

    Module-level (not a lambda) so it survives pickling if it ever moves
    before the loader's mp_prefetch stage.
    """
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
    from gensbi.experimental.models import FieldDiT, FieldDiTParams

    kw["encoder_widths"] = tuple(kw["encoder_widths"])
    return FieldDiT(FieldDiTParams(rngs=nnx.Rngs(seed), **kw))


def radial_power_spectrum(field, nbins=40):
    """Isotropic P(k) of a 2D field; k in cycles/pixel (Nyquist = 0.5)."""
    # float64 throughout: with float32 weights np.histogram accumulates in
    # float32, and for steep spectra the high-k bins (~1e-5 against a ~1e4
    # running total) round away to exactly zero.
    field = np.asarray(field, dtype=np.float64)
    H, W = field.shape
    pk2d = np.abs(np.fft.fft2(field)) ** 2 / (H * W)
    kx, ky = np.meshgrid(np.fft.fftfreq(H), np.fft.fftfreq(W), indexing="ij")
    knorm = np.sqrt(kx**2 + ky**2).ravel()
    # log-spaced bins with geometric centers: equal width on the loglog plot,
    # so a steep power law isn't distorted by wide-in-log low-k bins.
    kbins = np.geomspace(knorm[knorm > 0].min(), 0.5, nbins + 1)
    counts, _ = np.histogram(knorm, kbins)
    power, _ = np.histogram(knorm, kbins, weights=pk2d.ravel())
    kcent = np.sqrt(kbins[1:] * kbins[:-1])
    good = counts > 0
    return kcent[good], power[good] / counts[good]


def plot_losses(loss_array, val_loss_array, val_every, path):
    # train + val are both recorded once per validation event (every
    # val_every steps), so both share the same step-scaled x-axis.
    loss = np.asarray(loss_array, dtype=np.float32)
    val = np.asarray(val_loss_array, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(loss) + 1) * val_every, loss,
            label="train (smoothed)", alpha=0.5)
    ax.plot(np.arange(1, len(val) + 1) * val_every, val, label="val")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


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


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    tcfg = cfg["training"]
    scfg = cfg["sampling"]
    experiment = tcfg["experiment_id"]

    imgs_dir = os.path.join(EXAMPLE_DIR, "imgs")
    os.makedirs(imgs_dir, exist_ok=True)

    # --- data ---
    task = OnlineTaskDataset(
        "gaussian_random_field",
        normalize=True,
        dtype=jnp.bfloat16,
        # dtype=jnp.float32,
        # use_prefetching=True,
        # max_workers=tcfg.get("max_workers"),  # None -> no prefetch
    )

    offline_task = TaskDataset(
        "gaussian_random_field",
        normalize=True,
        dtype=jnp.bfloat16,
    )
    # NOTE: this .map runs in the main process (mp_prefetch is the loader's
    # last stage). Free for a tuple swap; move it before prefetch if it ever
    # does real per-batch work.
    # train_loader = task.get_train_loader(tcfg["batch_size"]).map(swap_obs_cond)
    train_loader = task.get_online_train_loader(tcfg["batch_size"]).map(swap_obs_cond)
    val_loader = offline_task.get_val_loader(tcfg["val_batch_size"]).map(swap_obs_cond)

    # --- model + pipeline ---
    model_kind, model_cfg = resolve_model_section(cfg)
    model = build_model(model_kind, model_cfg, seed=tcfg.get("seed", 0))
    n_params = sum(
        leaf.size
        for leaf in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))
    )
    print(f"{model_kind} parameters: {n_params / 1e6:.1f}M")

    training_config = FieldConditionalPipeline.get_default_training_config()
    training_config.update({k: tcfg[k] for k in _PIPELINE_KEYS if k in tcfg})
    training_config["checkpoint_dir"] = os.path.join(EXAMPLE_DIR, "checkpoints")

    pipeline = FieldConditionalPipeline(
        model,
        train_loader,
        val_loader,
        field_shape=tuple(model_cfg["field_shape"]),
        dim_cond=model_cfg["cond_dim"],
        method=FlowMatchingMethod(),
        ch_obs=1,
        ch_cond=1,
        training_config=training_config,
    )

    # --- train / restore ---
    if tcfg["train_model"]:
        loss_array, val_loss_array = pipeline.train(nnx.Rngs(0), save_model=True)
        plot_losses(
            loss_array,
            val_loss_array,
            training_config["val_every"],
            os.path.join(imgs_dir, f"grf_loss_conf{experiment}.png"),
        )
    if tcfg["restore_model"]:
        pipeline.restore_model()
    pipeline._wrap_model()

    # --- posterior samples for a few test thetas ---
    n_thetas = scfg["num_thetas"]
    rows = task.df_test[:n_thetas]  # slice first: decodes only these rows
    thetas_raw = np.asarray(rows["thetas"], dtype=np.float32)  # (n, 2)
    truths = np.asarray(rows["xs"], dtype=np.float32)          # (n, 32, 32)
    theta_norm = np.asarray(
        task.normalize_theta(thetas_raw[..., None])            # (n, 2, 1)
    )

    samples = []
    key = jax.random.PRNGKey(tcfg.get("seed", 0))
    for i in range(n_thetas):
        key, sub = jax.random.split(key)
        s = pipeline.sample(
            sub,
            jnp.asarray(theta_norm[i : i + 1]),  # (1, 2, 1)
            nsamples=scfg["nsamples"],
            step_size=scfg["step_size"],
        )  # (nsamples, 32, 32, 1), normalized
        s = np.asarray(task.unnormalize_x(s), dtype=np.float32)[..., 0]
        samples.append(s)
        print(f"theta {i}: sampled {s.shape}, finite={np.isfinite(s).all()}")

    plot_field_grid(
        truths,
        samples,
        thetas_raw,
        scfg["nsamples_grid"],
        os.path.join(imgs_dir, f"grf_fields_conf{experiment}.png"),
    )
    plot_power_spectra(
        truths,
        samples,
        thetas_raw,
        os.path.join(imgs_dir, f"grf_pk_conf{experiment}.png"),
    )
    print(f"Plots written to {imgs_dir} (experiment {experiment})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config_1.yaml")
    main(parser.parse_args().config)
