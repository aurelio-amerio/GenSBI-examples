"""Train a conditional TarFlow p(field | theta) on gaussian_random_field_256.

Field-level NLE with a transformer autoregressive normalizing flow: the
256x256 GRF realization is the modeled variable (obs), theta = (log_std,
alpha) the condition. TarFlow uses the image tokenizer (patchify) with 2D
rotary position embeddings (use_rope=True) and the vector-prefix conditioning
strategy: each theta coordinate becomes one prefix token behind the
prefix-LM mask.

Trained by exact max likelihood (ConditionalFlowPipeline, structured_obs=True);
sampling uses the KV-cached autoregressive sampler. Outputs: loss curves, a
truth-vs-samples field grid, radial power-spectrum overlays (simulator mean
+/- 1 sigma and the analytic power law), and the exact held-out NLL in
bits/dim -- a scalar the flow-matching baselines in the parent dir cannot
provide.

Training data comes from the online sampler (fresh prior + simulator draws
each batch) by default; set `training.online: false` to train on the
pre-generated HF train split instead.

Usage (conda env `gensbi`):
    python train_tarflow_grf256.py --config config/config_1.yaml
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

from gensbi.models import TarFlow, TarFlowParams
from gensbi.recipes import ConditionalFlowPipeline

from sbibm_jax.data import OnlineTaskDataset, TaskDataset

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))


def build_flow(rngs, model_cfg):
    """TarFlow over 256x256 fields: image tokenizer + rope, vector-prefix cond.

    cond="vector" and modeled="image" are fixed by design (see the spec);
    the yaml model section only sizes the architecture.
    """
    return TarFlow(TarFlowParams(
        rngs=rngs,
        modeled="image",
        img_size=int(model_cfg["img_size"]),
        patch_size=int(model_cfg["patch_size"]),
        img_channels=int(model_cfg.get("img_channels", 1)),
        cond="vector",
        cond_dim=int(model_cfg["cond_dim"]),
        cond_channels=1,
        use_rope=bool(model_cfg.get("use_rope", True)),
        rope_theta=int(model_cfg.get("rope_theta", 10000)),
        head_dim=int(model_cfg.get("head_dim", 64)),
        num_heads=int(model_cfg.get("num_heads", 8)),
        num_blocks=int(model_cfg.get("num_blocks", 8)),
        layers_per_block=int(model_cfg.get("layers_per_block", 2)),
        permutation=str(model_cfg.get("permutation", "flip")),
    ))


def build_training_config(config, checkpoint_dir):
    """Pipeline defaults overlaid with the yaml optimizer+training sections.

    ConditionalFlowPipeline reads training_config keys eagerly in __init__
    with no merge, so it must be complete. Extra script-level keys
    (batch_size, train_model, ...) are harmless -- the pipeline only reads
    the keys it knows.
    """
    tc = ConditionalFlowPipeline.get_default_training_config()
    tc.update(config.get("optimizer", {}))
    tc.update(config.get("training", {}))
    tc["checkpoint_dir"] = checkpoint_dir
    return tc


def to_obs_cond(batch):
    """Loader yields (theta, x); the flow pipeline wants (obs=x, cond=theta).

    sbibm_jax's collate already tokenizes both: x arrives in native image
    shape (B, 256, 256, 1), theta already carries the trailing channel axis
    the VectorConditioner expects, (B, 2, 1) -- no further reshape needed
    here, just the obs/cond swap.
    Module-level (not a lambda) so it survives pickling if it ever moves
    before a prefetch stage.
    """
    theta, x = batch
    return x, theta


def heldout_bits_per_dim(flow, fields_norm, theta_norm, batch_size=64):
    """Exact mean NLL of held-out fields under the flow, in bits/dim.

    Inputs are in normalized units (the flow's training space); the constant
    Jacobian of the dataset normalization is omitted, so values compare
    across runs of this example, not across normalization schemes.
    fields_norm: (N, H, W, 1); theta_norm: (N, cond_dim, 1). Calls
    flow.log_prob directly -- batched cond is native TarFlow, no pipeline
    single-observation restriction.
    """
    n = fields_norm.shape[0]
    ndim = int(np.prod(fields_norm.shape[1:]))
    lps = []
    for i in range(0, n, batch_size):
        lp = flow.log_prob(jnp.asarray(fields_norm[i:i + batch_size]),
                           jnp.asarray(theta_norm[i:i + batch_size]))
        lps.append(np.asarray(lp, dtype=np.float64))
    return float(-np.concatenate(lps).mean() / (ndim * np.log(2.0)))


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


def theory_power_spectrum(k, log_std, alpha, field_size):
    """Analytic conditional P(k) of the GRF simulator at (log_std, alpha).

    The simulator builds sqrt(P) = (knorm*(|alpha|+1e-7))**(-alpha/2)*exp(log_std)
    with knorm the fftfreq grid (= the plotted k). Propagating that through the
    ifftn + radial_power_spectrum normalization (verified numerically) gives the
    measured spectrum E[P(k)] = exp(2 log_std) * (k*(|alpha|+1e-7))**(-alpha) / N^2.
    """
    return np.exp(2.0 * log_std) * (k * (abs(alpha) + 1e-7)) ** (-alpha) / field_size**2


def plot_power_spectra(sim_fields, samples, thetas, field_size, path):
    """Per theta: model-sample P(k) vs the true conditional P(k).

    Truth reference is the mean P(k) over `sim_fields[i]` fresh simulator maps
    at theta_i (solid black + 1 sigma realization band) -- averaging beats down
    the per-realization cosmic variance a single map shows in the low-k bins --
    with the analytic power law overlaid dashed. Model samples: C0 mean +/- sigma.
    """
    n = len(samples)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 3.8), squeeze=False)
    for i in range(n):
        ax = axes[0][i]
        log_std, alpha = float(thetas[i, 0]), float(thetas[i, 1])

        # truth: mean +/- 1 sigma P(k) over fresh simulator realizations
        k, _ = radial_power_spectrum(sim_fields[i][0])
        pk_sim = np.stack([radial_power_spectrum(f)[1] for f in sim_fields[i]])
        sim_mean, sim_std = pk_sim.mean(axis=0), pk_sim.std(axis=0)
        pk_theory = theory_power_spectrum(k, log_std, alpha, field_size)

        # model samples
        pks = np.stack([radial_power_spectrum(s)[1] for s in samples[i]])
        mean, std = pks.mean(axis=0), pks.std(axis=0)

        ax.loglog(k, sim_mean, "k-", label=f"simulator (mean of {len(sim_fields[i])})")
        ax.fill_between(
            k, np.maximum(sim_mean - sim_std, 1e-20), sim_mean + sim_std,
            color="k", alpha=0.15,
        )
        ax.loglog(k, pk_theory, "k--", label="theory")
        ax.loglog(k, mean, "C0-", label="samples (mean)")
        ax.fill_between(
            k, np.maximum(mean - std, 1e-20), mean + std, color="C0", alpha=0.3
        )
        ax.set_title(f"log_std={log_std:.2f}, alpha={alpha:.2f}", fontsize=9)
        ax.set_xlabel("k [cycles/pixel]")
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel("P(k)")
    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


_NLL_BATCH = 64


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]
    tcfg = cfg["training"]
    scfg = cfg["sampling"]
    experiment = tcfg.get("experiment_id", 1)

    # --- data ---
    # online (default) -> fresh (theta, x) prior+simulator draws every batch;
    # offline -> the pre-generated HF train split. The offline task is always
    # built: it serves the val loader, the df_test rows and the NLL eval.
    # Datasets stay float32 (no dtype arg): the exact-likelihood NLL is the
    # headline metric here, unlike the bfloat16 FM script in the parent dir.
    online = tcfg.get("online", True)
    max_workers = tcfg.get("max_workers")  # None -> in-process / no prefetch

    task = OnlineTaskDataset("gaussian_random_field_256", normalize=True)
    offline_task = TaskDataset(
        "gaussian_random_field_256",
        normalize=True,
        max_workers=None if online else max_workers,
    )
    if online:
        train_loader = task.get_online_train_loader(
            tcfg["batch_size"], num_workers=max_workers or 0
        ).map(to_obs_cond)
    else:
        train_loader = offline_task.get_train_loader(
            tcfg["batch_size"]
        ).map(to_obs_cond)
    val_loader = offline_task.get_val_loader(tcfg["val_batch_size"]).map(to_obs_cond)

    # --- flow + pipeline ---
    flow = build_flow(nnx.Rngs(tcfg.get("seed", 0)), model_cfg)
    n_params = sum(
        leaf.size for leaf in jax.tree_util.tree_leaves(nnx.state(flow, nnx.Param))
    )
    print(f"tarflow parameters: {n_params / 1e6:.1f}M")

    img_size = int(model_cfg["img_size"])
    img_channels = int(model_cfg.get("img_channels", 1))
    training_config = build_training_config(
        cfg, os.path.join(EXAMPLE_DIR, "checkpoints"))
    pipeline = ConditionalFlowPipeline(
        flow, train_loader, val_loader,
        dim_obs=img_size * img_size * img_channels,
        dim_cond=int(model_cfg["cond_dim"]),
        structured_obs=True,
        training_config=training_config,
    )
    # Data is normalized upstream (normalize=True datasets); the flow's
    # standardize buffers stay at identity. Mark standardized to suppress
    # the train-time 'did you fit?' warning.
    pipeline._standardized = True

    imgs_dir = os.path.join(EXAMPLE_DIR, "imgs")
    os.makedirs(imgs_dir, exist_ok=True)

    # --- train / restore ---
    if tcfg.get("restore_model", False):
        print("Restoring model from checkpoint...")
        pipeline.restore_model()
    if tcfg.get("train_model", True):
        loss_array, val_loss_array = pipeline.train(nnx.Rngs(0), save_model=True)
        plot_losses(
            loss_array, val_loss_array, training_config["val_every"],
            os.path.join(imgs_dir, f"grf256_loss_conf{experiment}.png"),
        )

    # --- test rows: raw truths + normalized thetas ---
    n_thetas = scfg["num_thetas"]
    n_nll = scfg.get("nll_num_test", 256)
    rows = offline_task.df_test[:max(n_thetas, n_nll)]  # slice first: decodes only these
    thetas_raw = np.asarray(rows["thetas"], dtype=np.float32)  # (n, 2)
    truths = np.asarray(rows["xs"], dtype=np.float32)          # (n, 256, 256)
    theta_norm = np.asarray(
        task.normalize_theta(thetas_raw[..., None])            # (n, 2, 1)
    )
    field_size = truths.shape[-1]

    # --- exact held-out NLL (bits/dim), raw and EMA weights ---
    n_nll = min(n_nll, len(truths))
    fields_norm = np.asarray(task.normalize_x(truths[:n_nll, ..., None]))
    for use_ema, tag in ((True, "ema"), (False, "raw")):
        m = pipeline.ema_model if use_ema else pipeline.model
        bpd = heldout_bits_per_dim(m, fields_norm, theta_norm[:n_nll], _NLL_BATCH)
        print(f"held-out NLL [{tag}]: {bpd:.4f} bits/dim over {n_nll} test fields")

    # --- fresh simulator realizations per theta: P(k) truth reference ---
    # Mean over these (+/- 1 sigma) beats down the per-realization cosmic
    # variance a single stored map shows in the low-k bins. Raw field units,
    # matching unnormalize_x(samples). Separate PRNG stream from sampling.
    n_sim_pk = scfg.get("nsim_pk", 64)
    simulator = task.task.get_simulator(jax.random.PRNGKey(tcfg.get("seed", 0)))
    sim_fields = []
    sim_key = jax.random.PRNGKey(tcfg.get("seed", 0) + 1)
    for i in range(n_thetas):
        sim_key, sk = jax.random.split(sim_key)
        thetas_M = jnp.broadcast_to(jnp.asarray(thetas_raw[i]), (n_sim_pk, 2))
        f = np.asarray(simulator(sk, thetas_M), dtype=np.float32)  # (M, N*N)
        sim_fields.append(f.reshape(n_sim_pk, field_size, field_size))

    # --- sample p(field | theta), raw and EMA weights (identical PRNG) ---
    # EMA-degeneration cross-check inherited from the FM script: emit both.
    for use_ema, tag in ((True, "ema"), (False, "raw")):
        samples = []
        key = jax.random.PRNGKey(tcfg.get("seed", 0))
        for i in range(n_thetas):
            key, sub = jax.random.split(key)
            s = pipeline.sample(
                sub,
                jnp.asarray(theta_norm[i:i + 1]),  # (1, 2, 1)
                nsamples=scfg["nsamples"],
                use_ema=use_ema,
            )  # (nsamples, 256, 256, 1), normalized; KV-cached sampler
            s = np.asarray(task.unnormalize_x(s), dtype=np.float32)[..., 0]
            samples.append(s)
            print(f"[{tag}] theta {i}: sampled {s.shape}, "
                  f"finite={np.isfinite(s).all()}")

        plot_field_grid(
            truths[:n_thetas], samples, thetas_raw, scfg["nsamples_grid"],
            os.path.join(imgs_dir, f"grf256_fields_conf{experiment}_{tag}.png"),
        )
        plot_power_spectra(
            sim_fields, samples, thetas_raw, field_size,
            os.path.join(imgs_dir, f"grf256_pk_conf{experiment}_{tag}.png"),
        )
    print(f"Plots written to {imgs_dir} (experiment {experiment}; _ema and _raw)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=os.path.join(EXAMPLE_DIR, "config", "config_1.yaml"))
    main(parser.parse_args().config)
