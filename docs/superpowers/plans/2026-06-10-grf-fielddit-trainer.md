# GRF-256 FieldDiT Trainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A cluster-submittable script that trains FieldDiT on `gaussian_random_field_256` via `FieldConditionalPipeline` (10k steps), samples the posterior for a few test thetas, and saves sanity plots (loss curves, field grids, radial power spectra).

**Architecture:** Single script `train-grf.py` (lensing-trainer style: `main()` guarded by `__name__`, YAML config) + two configs (real run, CPU smoke). Data comes from sbibm-jax `TaskDataset` (repo-HEAD `x_kind/x_shape` schema); a grain `.map(swap_obs_cond)` turns `(theta, x)` batches into the `(obs, cond)` the field pipeline expects.

**Tech Stack:** GenSBI branch `FieldDiT` (editable install in conda env `gensbi`), sbibm-jax @ HEAD, JAX/flax-nnx, grain, matplotlib (Agg).

**Spec:** `docs/superpowers/specs/2026-06-10-grf-fielddit-trainer-design.md`

**Working dir:** `/lhome/ific/a/aamerio/data/github/GenSBI-examples` (git branch `gaussian_random_field`)

**Interpreter:** `PY=/lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python` (the conda `gensbi` env — has gensbi editable from the FieldDiT branch checkout)

**Note on testing:** this is an example training script, not library code — there are no unit tests. Verification is (a) `--help` import check and (b) an end-to-end CPU smoke run with a tiny model config (Task 4). Library behavior is already covered by GenSBI's 666-test suite.

---

### Task 1: Config files

**Files:**
- Create: `examples/sbi-benchmarks/gaussian_random_field/config/config_1.yaml`
- Create: `examples/sbi-benchmarks/gaussian_random_field/config/config_smoke.yaml`

- [ ] **Step 1: Write `config/config_1.yaml`** (the real 10k-step run; model = the verified `test_realistic_256_config_smoke` config)

```yaml
# GRF-256 FieldDiT — experiment 1: 10k-step sanity run (verified smoke config)
fielddit:
  in_channels: 1
  field_shape: [256, 256]
  encoder_widths: [64, 128, 256, 256]   # D=3 -> 32x32 meeting grid
  patch_size: 2                          # -> 16x16 = 256 tokens
  cond_dim: 2                            # theta = (log_std, alpha)
  cond_in_channels: 1
  param_dtype: bfloat16
  # defaults kept: num_heads 12, axes_dim [16,24,24] -> hidden 768,
  # depth 2, depth_single_blocks 2, res_blocks 2+2

training:
  batch_size: 128          # ~40 GB GPU; halve on OOM
  val_batch_size: 128
  max_workers: 4           # grain mp_prefetch workers (null -> no prefetch)
  nsteps: 10000
  max_lr: 1.0e-4
  val_every: 100
  early_stopping: false    # 10k steps IS the budget for this sanity run
  multistep: 1
  experiment_id: 1
  train_model: true
  restore_model: false
  seed: 0

sampling:
  num_thetas: 3            # test rows used for plots
  nsamples: 16             # posterior samples per theta (P(k) statistics)
  nsamples_grid: 3         # samples shown per row in the field grid
  step_size: 0.01          # Euler ODE step (100 steps)
```

- [ ] **Step 2: Write `config/config_smoke.yaml`** (tiny model, 2 steps, CPU-runnable end-to-end wiring check; field_shape must stay 256 — the data is 256x256)

```yaml
# Wiring smoke test: tiny model, 2 steps, CPU. NOT a learning check.
fielddit:
  in_channels: 1
  field_shape: [256, 256]
  encoder_widths: [8, 16, 16, 16]       # D=3 -> 32x32 meeting grid
  patch_size: 2                          # -> 256 tokens
  cond_dim: 2
  cond_in_channels: 1
  num_heads: 2
  axes_dim: [4, 6, 6]                    # hidden 32
  depth: 1
  depth_single_blocks: 1
  res_blocks_down: 1
  res_blocks_up: 1
  param_dtype: float32

training:
  batch_size: 2
  val_batch_size: 2
  max_workers: null        # no mp_prefetch on the login node
  nsteps: 2
  warmup_steps: 1
  max_lr: 1.0e-4
  val_every: 1
  early_stopping: false
  multistep: 1
  experiment_id: 99        # separate checkpoint slot from the real run
  train_model: true
  restore_model: false
  seed: 0

sampling:
  num_thetas: 2
  nsamples: 2
  nsamples_grid: 2
  step_size: 0.5           # 2 Euler steps — wiring only
```

- [ ] **Step 3: Commit**

```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI-examples
git add examples/sbi-benchmarks/gaussian_random_field/config/
git commit -m "feat(grf): configs for FieldDiT trainer (real + smoke)"
```

---

### Task 2: `train-grf.py`

**Files:**
- Create: `examples/sbi-benchmarks/gaussian_random_field/train-grf.py`

- [ ] **Step 1: Write the script** (complete content)

```python
"""Train FieldDiT on gaussian_random_field_256 and sample the posterior.

Field-level NPE: the model learns p(field | theta) with conditional flow
matching. The 256x256 GRF realization is the generation target (obs); theta =
(log_std, alpha) is the conditioning vector. Outputs: loss curves, a
truth-vs-samples field grid, and radial power-spectrum overlays in imgs/.

Usage (conda env `gensbi`):
    python train-grf.py --config config/config_1.yaml
"""

import os

if __name__ != "__main__":
    os.environ["JAX_PLATFORMS"] = "cpu"
else:
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"
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
from gensbi.experimental.models import FieldDiT, FieldDiTParams
from gensbi.experimental.recipes import FieldConditionalPipeline

from sbibm_jax.data import TaskDataset

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


def build_model(model_cfg, seed):
    kw = dict(model_cfg)
    kw["field_shape"] = tuple(kw["field_shape"])
    kw["encoder_widths"] = tuple(kw["encoder_widths"])
    kw["param_dtype"] = getattr(jnp, kw.get("param_dtype", "bfloat16"))
    return FieldDiT(FieldDiTParams(rngs=nnx.Rngs(seed), **kw))


def radial_power_spectrum(field, nbins=40):
    """Isotropic P(k) of a 2D field; k in cycles/pixel (Nyquist = 0.5)."""
    H, W = field.shape
    pk2d = np.abs(np.fft.fft2(field)) ** 2 / (H * W)
    kx, ky = np.meshgrid(np.fft.fftfreq(H), np.fft.fftfreq(W), indexing="ij")
    knorm = np.sqrt(kx**2 + ky**2).ravel()
    kbins = np.linspace(knorm[knorm > 0].min(), 0.5, nbins + 1)
    counts, _ = np.histogram(knorm, kbins)
    power, _ = np.histogram(knorm, kbins, weights=pk2d.ravel())
    kcent = 0.5 * (kbins[1:] + kbins[:-1])
    good = counts > 0
    return kcent[good], power[good] / counts[good]


def plot_losses(loss_array, val_loss_array, val_every, path):
    loss = np.asarray(loss_array, dtype=np.float32)
    val = np.asarray(val_loss_array, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(loss) + 1), loss, label="train", alpha=0.5)
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
    task = TaskDataset(
        "gaussian_random_field_256",
        normalize=True,
        dtype=jnp.bfloat16,
        use_prefetching=True,
        max_workers=tcfg.get("max_workers"),  # None -> no prefetch
    )
    # NOTE: this .map runs in the main process (mp_prefetch is the loader's
    # last stage). Free for a tuple swap; move it before prefetch if it ever
    # does real per-batch work.
    train_loader = task.get_train_loader(tcfg["batch_size"]).map(swap_obs_cond)
    val_loader = task.get_val_loader(tcfg["val_batch_size"]).map(swap_obs_cond)

    # --- model + pipeline ---
    model = build_model(cfg["fielddit"], seed=tcfg.get("seed", 0))
    n_params = sum(
        leaf.size
        for leaf in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))
    )
    print(f"FieldDiT parameters: {n_params / 1e6:.1f}M")

    training_config = FieldConditionalPipeline.get_default_training_config()
    training_config.update({k: tcfg[k] for k in _PIPELINE_KEYS if k in tcfg})
    training_config["checkpoint_dir"] = os.path.join(EXAMPLE_DIR, "checkpoints")

    pipeline = FieldConditionalPipeline(
        model,
        train_loader,
        val_loader,
        field_shape=tuple(cfg["fielddit"]["field_shape"]),
        dim_cond=cfg["fielddit"]["cond_dim"],
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
    truths = np.asarray(rows["xs"], dtype=np.float32)          # (n, 256, 256)
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
        )  # (nsamples, 256, 256, 1), normalized
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
```

- [ ] **Step 2: Import/arg-parse check** (catches syntax errors and broken imports without touching data or GPU)

```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI-examples/examples/sbi-benchmarks/gaussian_random_field
JAX_PLATFORMS=cpu /lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python train-grf.py --help
```

Expected: usage text printing the `--config` option, exit 0. (gensbi/sbibm_jax import at module level, so this also proves the env resolves them.)

- [ ] **Step 3: Commit**

```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI-examples
git add examples/sbi-benchmarks/gaussian_random_field/train-grf.py
git commit -m "feat(grf): FieldDiT trainer for gaussian_random_field_256"
```

---

### Task 3: Environment — sbibm_jax @ HEAD in the `gensbi` env

The conda env currently has an old sbibm_jax (site-packages) that reads the
old `data_kind/data_shape` hub schema. The script targets repo HEAD
(`x_kind/x_shape`), matching the republished hub metadata.

- [ ] **Step 1: Check whether the installed version is already current**

```bash
JAX_PLATFORMS=cpu /lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python -c "
import inspect
from sbibm_jax.data.dataset import TaskDataset
src = inspect.getsource(TaskDataset.__init__)
print('NEW schema' if 'x_kind' in src else 'OLD schema')
"
```

Expected: `NEW schema` (skip Step 2) or `OLD schema` (do Step 2).

- [ ] **Step 2 (only if OLD): editable-install the local repo into the env**

```bash
/lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python -m pip install -e /lhome/ific/a/aamerio/data/github/sbibm-jax --no-deps
```

`--no-deps` so pip does not touch jax/grain pins in the env. Re-run Step 1's
check; expected `NEW schema`.

- [ ] **Step 3: Verify the republished hub metadata loads** (GATE: Aurelio is
  republishing the test repo — if this fails with KeyError on `x_kind`, the
  republish hasn't landed yet; STOP and report, do not work around)

```bash
JAX_PLATFORMS=cpu /lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python -c "
from sbibm_jax.data import TaskDataset
import numpy as np
t = TaskDataset('gaussian_random_field_256', normalize=True)
print('x_shape', t.x_shape, 'theta_shape', t.theta_shape)
assert t.x_shape == (256, 256) and t.theta_shape == (2,)
print('stats ok:', t.x_std is not None)
"
```

Expected: `x_shape (256, 256) theta_shape (2,)` and `stats ok: True`. Note:
the constructor also triggers the HF dataset download (~GBs, first time only,
shared cache) — let it run.

---

### Task 4: End-to-end CPU smoke run

- [ ] **Step 1: Run the smoke config** (tiny model, 2 train steps, 2 ODE steps; minutes on CPU, dominated by data download/decode)

```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI-examples/examples/sbi-benchmarks/gaussian_random_field
JAX_PLATFORMS=cpu /lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python train-grf.py --config config/config_smoke.yaml
```

Expected: parameter count printed, training progress for 2 steps, two
`theta i: sampled (2, 256, 256) finite=True` lines, and
`Plots written to .../imgs (experiment 99)`. Exit 0.

- [ ] **Step 2: Verify outputs exist**

```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI-examples
ls -la examples/sbi-benchmarks/gaussian_random_field/imgs/grf_loss_conf99.png \
       examples/sbi-benchmarks/gaussian_random_field/imgs/grf_fields_conf99.png \
       examples/sbi-benchmarks/gaussian_random_field/imgs/grf_pk_conf99.png
ls examples/sbi-benchmarks/gaussian_random_field/checkpoints/
```

Expected: all three PNGs exist (nonzero size); a checkpoint for experiment 99.

- [ ] **Step 3: Read the three PNGs** (Read tool) — sanity: loss plot has two
  curves; field grid is 2 rows x 3 panels with GRF-looking truth panels
  (samples will be noise after 2 steps — that is fine); P(k) plot has truth
  curves + sample bands on log-log axes.

- [ ] **Step 4: Commit any fixes made during the smoke run**

```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI-examples
git add -u examples/sbi-benchmarks/gaussian_random_field/
git commit -m "fix(grf): smoke-run fixes for FieldDiT trainer"
```

(Skip the commit if the run passed with no changes. Do NOT commit smoke
artifacts: `imgs/*conf99*` and the experiment-99 checkpoint should be removed:
`rm -f examples/sbi-benchmarks/gaussian_random_field/imgs/*conf99*` and remove
the experiment-99 checkpoint directory.)

---

### Task 5: Wrap-up

- [ ] **Step 1: Self-review the diff against the spec**

```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI-examples
git log --oneline main..HEAD 2>/dev/null || git log --oneline -5
git diff HEAD~3 --stat
```

Check: config values match the spec (batch 128, nsteps 10000, early stopping
off, verified model config); swap via `.map`; three plot outputs; no TARP/LC2ST
code (out of scope).

- [ ] **Step 2: Report the cluster command to Aurelio**

The deliverable command (he submits with his own templates):

```bash
conda activate gensbi
cd /lhome/ific/a/aamerio/data/github/GenSBI-examples/examples/sbi-benchmarks/gaussian_random_field
python train-grf.py --config config/config_1.yaml
```

First GPU run notes: HF dataset already cached by the smoke run; watch the
first ~100 steps for OOM (fallback: `batch_size: 64` in the yaml); checkpoints
land in `checkpoints/` under experiment 1; plots in `imgs/*conf1*`.
