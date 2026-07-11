"""Fixed-theta diagnostic: can PixelDiT learn ONE GRF's 2-point structure?

Trains PixelDiT on a single held-constant theta (no conditioning generalization)
with logit-normal t-sampling, then overlays the radial power spectrum of samples
vs. simulated truth at that theta. Decisive plot: the P(k) overlay — success =
samples develop low-k power matching truth; failure = flat P(k) (white noise).

NOTE: with theta fixed the cond input is constant, so this does NOT test
conditioning — it isolates generative/2-point capacity. Intentional.

Usage (conda env `gensbi`):
    python train-grf-fixedtheta.py --config config/config_4f.yaml
"""

import os

if __name__ != "__main__":
    os.environ["JAX_PLATFORMS"] = "cpu"
else:
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"
    os.environ.setdefault("JAX_PLATFORMS", "cuda")

import argparse

import jax
from jax import numpy as jnp
import numpy as np
from flax import nnx
import yaml

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gensbi.core import FlowMatchingMethod
from gensbi.experimental.recipes import FieldConditionalPipeline

from sbibm_jax.data import OnlineTaskDataset

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_NAME = "gaussian_random_field"

_PIPELINE_KEYS = (
    "nsteps", "max_lr", "min_lr", "warmup_steps", "ema_decay",
    "decay_transition", "val_every", "early_stopping", "multistep",
    "experiment_id", "val_error_ratio",
)


def swap_obs_cond(batch):
    theta, x = batch
    return x, theta


def build_pixeldit(model_cfg, seed):
    from gensbi.experimental.models import PixelDiT, PixelDiTParams

    kw = dict(model_cfg)
    kw["field_shape"] = tuple(kw["field_shape"])
    kw["param_dtype"] = getattr(jnp, kw.pop("param_dtype", "bfloat16"))
    return PixelDiT(PixelDiTParams(rngs=nnx.Rngs(seed), **kw))


def radial_power_spectrum(field, nbins=40):
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


def plot_losses(loss_array, val_loss_array, val_every, path):
    loss = np.asarray(loss_array, dtype=np.float32)
    val = np.asarray(val_loss_array, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(loss) + 1) * val_every, loss, label="train (smoothed)", alpha=0.5)
    ax.plot(np.arange(1, len(val) + 1) * val_every, val, label="val")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_pk_overlay(truths, samples, theta_raw, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    k = radial_power_spectrum(truths[0])[0]
    pkt = np.stack([radial_power_spectrum(t)[1] for t in truths])
    pks = np.stack([radial_power_spectrum(s)[1] for s in samples])
    for arr, color, lab in ((pkt, "k", "truth"), (pks, "C0", "samples")):
        mean, std = arr.mean(0), arr.std(0)
        ax.loglog(k, mean, color=color, label=f"{lab} (mean)")
        ax.fill_between(k, np.maximum(mean - std, 1e-20), mean + std, color=color, alpha=0.2)
    ax.set_xlabel("k [cycles/pixel]")
    ax.set_ylabel("P(k)")
    ax.set_title(f"fixed theta: log_std={theta_raw[0]:.2f}, alpha={theta_raw[1]:.2f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_field_grid(truths, samples, theta_raw, n_show, path):
    n_show = min(n_show, len(samples))
    fig, axes = plt.subplots(1, n_show + 1, figsize=(3 * (n_show + 1), 3.2), squeeze=False)
    vmax = float(np.percentile(np.abs(truths[0]), 99.5))
    axes[0][0].imshow(truths[0], vmin=-vmax, vmax=vmax, cmap="coolwarm")
    axes[0][0].set_title(f"truth | log_std={theta_raw[0]:.2f}, alpha={theta_raw[1]:.2f}", fontsize=9)
    for j in range(n_show):
        axes[0][j + 1].imshow(samples[j], vmin=-vmax, vmax=vmax, cmap="coolwarm")
        axes[0][j + 1].set_title(f"sample {j + 1}", fontsize=9)
    for ax in axes[0]:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def make_fixed_theta_loader(online_task, batch_size, theta_raw):
    """Monkeypatch the task prior to a constant theta; reuse the full online pipeline."""
    dim_theta = online_task.dim_theta
    theta_const = jnp.asarray(theta_raw, dtype=jnp.float32).reshape(dim_theta)
    orig_get_prior = online_task.task.get_prior

    def fixed_get_prior(key, n):
        template = orig_get_prior(key, n)  # (n, dim_theta), correct dtype/shape
        return jnp.broadcast_to(theta_const.astype(template.dtype), template.shape)

    online_task.task.get_prior = fixed_get_prior
    # num_workers=0 (in-process) on purpose: spawn workers would pickle the
    # task, and the monkeypatched prior closure must not cross that boundary.
    return online_task.get_online_train_loader(batch_size).map(swap_obs_cond)


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    tcfg = cfg["training"]
    scfg = cfg["sampling"]
    experiment = tcfg["experiment_id"]
    model_cfg = cfg["pixeldit"]
    field_shape = tuple(model_cfg["field_shape"])
    theta_raw = np.asarray(cfg["fixed_theta"], dtype=np.float32)  # (dim_theta,)
    H, W = field_shape

    imgs_dir = os.path.join(EXAMPLE_DIR, "imgs")
    os.makedirs(imgs_dir, exist_ok=True)

    # --- data (fixed theta) ---
    task = OnlineTaskDataset(TASK_NAME, normalize=True, dtype=jnp.bfloat16)
    train_loader = make_fixed_theta_loader(task, tcfg["batch_size"], theta_raw)
    val_task = OnlineTaskDataset(TASK_NAME, normalize=True, dtype=jnp.bfloat16, seed=123)
    val_loader = make_fixed_theta_loader(val_task, tcfg["val_batch_size"], theta_raw)

    # --- model + method + pipeline ---
    model = build_pixeldit(model_cfg, seed=tcfg.get("seed", 0))
    n_params = sum(leaf.size for leaf in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))
    print(f"pixeldit parameters: {n_params / 1e6:.1f}M (fixed-theta diagnostic)")

    ts_cfg = cfg.get("time_sampling", {})
    method = FlowMatchingMethod(
        time_dist=ts_cfg.get("dist", "uniform"),
        logitnorm_mean=ts_cfg.get("logitnorm_mean", 0.0),
        logitnorm_std=ts_cfg.get("logitnorm_std", 1.0),
    )

    training_config = FieldConditionalPipeline.get_default_training_config()
    training_config.update({k: tcfg[k] for k in _PIPELINE_KEYS if k in tcfg})
    training_config["checkpoint_dir"] = os.path.join(EXAMPLE_DIR, "checkpoints_pixeldit")

    pipeline = FieldConditionalPipeline(
        model, train_loader, val_loader,
        field_shape=field_shape, dim_cond=model_cfg["cond_dim"],
        method=method, ch_obs=1, ch_cond=1, training_config=training_config,
    )

    # --- train / restore ---
    if tcfg["train_model"]:
        loss_array, val_loss_array = pipeline.train(nnx.Rngs(0), save_model=True)
        plot_losses(loss_array, val_loss_array, training_config["val_every"],
                    os.path.join(imgs_dir, f"grf_fixedtheta_loss_conf{experiment}.png"))
    if tcfg["restore_model"]:
        pipeline.restore_model()
    pipeline._wrap_model()

    # --- truth fields at the fixed theta (simulate) ---
    n_eval = scfg["nsamples"]
    key = jax.random.PRNGKey(tcfg.get("seed", 0))
    key, kt = jax.random.split(key)
    theta_batch = jnp.broadcast_to(jnp.asarray(theta_raw), (n_eval, theta_raw.shape[0]))
    truth_flat = task.simulator(kt, theta_batch)                       # (n_eval, H*W) raw
    truths = np.asarray(truth_flat, dtype=np.float32).reshape(n_eval, H, W)

    # --- posterior samples at the fixed theta (EMA vs raw cross-check) ---
    # sample() defaults to the EMA model, which can degenerate to flat-P(k) noise
    # even when the raw model is good; emit both (_ema / _raw) from the SAME noise.
    theta_norm = np.asarray(task.normalize_theta(theta_raw[None, :, None]))  # (1, dim_theta, 1)
    key, sub = jax.random.split(key)
    for use_ema, tag in ((True, "ema"), (False, "raw")):
        s = pipeline.sample(sub, jnp.asarray(theta_norm), nsamples=n_eval,
                            step_size=scfg["step_size"], use_ema=use_ema)
        s = np.asarray(task.unnormalize_x(s), dtype=np.float32)[..., 0]      # (n_eval, H, W) raw
        print(f"[{tag}] sampled {s.shape}, finite={np.isfinite(s).all()}")
        plot_pk_overlay(truths, s, theta_raw,
                        os.path.join(imgs_dir, f"grf_fixedtheta_pk_conf{experiment}_{tag}.png"))
        plot_field_grid(truths, s, theta_raw, scfg["nsamples_grid"],
                        os.path.join(imgs_dir, f"grf_fixedtheta_fields_conf{experiment}_{tag}.png"))
    print(f"Plots written to {imgs_dir} (experiment {experiment}; _ema and _raw)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config_4f.yaml")
    main(parser.parse_args().config)
