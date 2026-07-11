# TarFlow GRF-32 Field-Level Example Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A field-level example training a conditional TarFlow p(field | theta) on the 32×32 `gaussian_random_field` task, with RoPE positional encoding and vector-prefix theta conditioning, producing the same diagnostics as the flow-matching baseline plus an exact held-out NLL.

**Architecture:** One standalone script `train_tarflow_grf.py` (yaml-driven, import-safe) built on `TarFlow(modeled="image", use_rope=True, cond="vector")` and `ConditionalFlowPipeline(structured_obs=True)`. Diagnostics helpers are copied from the sibling FM scripts (importing them would force `JAX_PLATFORMS=cpu`). A pytest file in `tests/` smoke-tests every module-level helper on CPU, mirroring `tests/test_train_maf_npe.py`.

**Tech Stack:** JAX / flax.nnx, GenSBI (editable checkout at `/lustre/ific.uv.es/ml/ific088/github/GenSBI`, branch with TarFlow RoPE + KV-cache), sbibm_jax datasets, pytest.

**Spec:** `docs/superpowers/specs/2026-07-11-tarflow-grf-design.md`

## Global Constraints

- Python for all commands: `/lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python` (the `gensbi` conda env; it has the editable GenSBI with `use_rope`, sbibm_jax, pytest 9).
- All tests and smoke runs on CPU: the script sets `JAX_PLATFORMS=cpu` when imported; tests set it defensively too.
- No GenSBI library changes — RoPE and KV-cached sampling are already merged.
- `cond="vector"` and `modeled="image"` are hardcoded in `build_flow` (design decision); the yaml only sizes the architecture. No `bias` or no-rope config ships.
- Diagnostics helpers are copied verbatim from `examples/sbi-benchmarks/gaussian_random_field_256/train-grf.py` (the richer P(k) diagnostics: simulator-mean reference + analytic power law, per the spec).
- Repo working branch: `field-level-inference`. Commit prefixes follow repo convention (`feat:`, `test:`, `spec:`).
- All new-file paths are relative to the repo root `/lustre/ific.uv.es/ml/ific088/github/GenSBI-examples`.

---

### Task 1: Script skeleton — config, build_flow, build_training_config

**Files:**
- Create: `examples/sbi-benchmarks/gaussian_random_field/tarflow/train_tarflow_grf.py`
- Create: `examples/sbi-benchmarks/gaussian_random_field/tarflow/config/config_1.yaml`
- Test: `tests/test_train_tarflow_grf.py`

**Interfaces:**
- Consumes: `TarFlow`, `TarFlowParams` from `gensbi.models`; `ConditionalFlowPipeline.get_default_training_config()` from `gensbi.recipes`.
- Produces: `build_flow(rngs: nnx.Rngs, model_cfg: dict) -> TarFlow` and `build_training_config(config: dict, checkpoint_dir: str) -> dict`, used by `main` (Task 4). The test module defines `_load_script_module()` and `_TINY_MODEL_CFG`, reused by all later test tasks.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_train_tarflow_grf.py`:

```python
import os
os.environ.setdefault("MPLBACKEND", "Agg")   # headless-safe before any pyplot import
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import importlib.util
import pathlib

import numpy as np
import pytest

# --- load the example script as a module (it lives outside the package) ---
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SCRIPT = (_REPO_ROOT /
           "examples/sbi-benchmarks/gaussian_random_field/tarflow/train_tarflow_grf.py")
_CONFIG = (_REPO_ROOT /
           "examples/sbi-benchmarks/gaussian_random_field/tarflow/config/config_1.yaml")

# Tiny rope model: 8x8 image, patch 4 -> T=4 tokens of F=16; head_dim 8 (%4==0).
_TINY_MODEL_CFG = {
    "img_size": 8, "patch_size": 4, "img_channels": 1, "cond_dim": 2,
    "use_rope": True, "rope_theta": 10000, "head_dim": 8, "num_heads": 2,
    "num_blocks": 2, "layers_per_block": 1, "permutation": "flip",
}


def _load_script_module():
    spec = importlib.util.spec_from_file_location("train_tarflow_grf", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_config_yaml_has_required_sections():
    import yaml
    with open(_CONFIG) as f:
        cfg = yaml.safe_load(f)
    for section in ("model", "optimizer", "training", "sampling"):
        assert section in cfg, f"missing section {section!r}"
    m = cfg["model"]
    assert m["use_rope"] is True                 # rope-only example by design
    assert m["head_dim"] % 4 == 0                # TarFlowParams rope requirement
    assert m["img_size"] == 32
    assert m["img_size"] % m["patch_size"] == 0  # ImageTokenizer requirement
    assert m["cond_dim"] == 2                    # theta = (log_std, alpha)


def test_build_flow_rope_log_prob_and_sample_shapes():
    mod = _load_script_module()
    from flax import nnx
    import jax
    import jax.numpy as jnp
    flow = mod.build_flow(nnx.Rngs(0), _TINY_MODEL_CFG)
    x = jax.random.normal(jax.random.key(0), (3, 8, 8, 1))
    cond = jax.random.normal(jax.random.key(1), (3, 2, 1))
    lp = flow.log_prob(x, cond)
    assert lp.shape == (3,) and bool(jnp.all(jnp.isfinite(lp)))
    s = flow.sample(jax.random.key(2), cond=cond[:2])
    assert s.shape == (2, 8, 8, 1) and bool(jnp.all(jnp.isfinite(s)))


def test_build_flow_uses_rope_and_vector_prefix():
    mod = _load_script_module()
    from flax import nnx
    flow = mod.build_flow(nnx.Rngs(0), _TINY_MODEL_CFG)
    blk = flow.blocks[0]
    assert blk.pos_embed is None          # rope replaces the learned table
    assert blk.freqs_cis is not None
    assert type(blk.conditioner).__name__ == "VectorConditioner"


def test_build_training_config_merges_defaults_and_overrides(tmp_path):
    mod = _load_script_module()
    from gensbi.recipes import ConditionalFlowPipeline
    cfg = {
        "optimizer": {"max_lr": 4.0e-4},
        "training": {"nsteps": 123, "experiment_id": 7},
    }
    tc = mod.build_training_config(cfg, checkpoint_dir=str(tmp_path / "ckpt"))
    # every key the pipeline reads must be present (defaults filled in)
    for key in ConditionalFlowPipeline.get_default_training_config():
        assert key in tc, f"missing required training_config key {key!r}"
    assert tc["nsteps"] == 123
    assert tc["max_lr"] == 4.0e-4
    assert tc["experiment_id"] == 7
    assert tc["checkpoint_dir"] == str(tmp_path / "ckpt")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd /lustre/ific.uv.es/ml/ific088/github/GenSBI-examples
/lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python -m pytest tests/test_train_tarflow_grf.py -v
```
Expected: all 4 tests FAIL (`FileNotFoundError` from `_load_script_module` / missing config).

- [ ] **Step 3: Create the config**

Create `examples/sbi-benchmarks/gaussian_random_field/tarflow/config/config_1.yaml`:

```yaml
# GRF-32 TarFlow field-level NLE — experiment 1: rope + vector-prefix cond
model:
  img_size: 32
  patch_size: 4            # -> T = 64 tokens, F = 16 per token
  img_channels: 1
  cond_dim: 2              # theta = (log_std, alpha) -> 2 prefix tokens
  use_rope: true           # 2D rotary PE; learned pos_embed dropped
  rope_theta: 10000
  head_dim: 32             # % 4 == 0 required by rope
  num_heads: 8             # channels = 256
  num_blocks: 8
  layers_per_block: 2
  permutation: flip

optimizer:
  max_lr: 1.0e-4
  min_lr: 1.0e-6
  warmup_steps: 500
  decay_transition: 0.80

training:
  batch_size: 256
  val_batch_size: 256
  max_workers: 4           # online sim workers (grain prefetch if offline)
  online: true             # fresh prior+simulator draws every batch
  nsteps: 10000
  ema_decay: 0.999
  val_every: 100
  early_stopping: false    # 10k steps IS the budget for this baseline
  multistep: 1
  experiment_id: 1
  train_model: true
  restore_model: false
  seed: 0

sampling:
  num_thetas: 3            # test rows used for plots
  nsamples: 16             # fields sampled per theta (P(k) statistics)
  nsamples_grid: 3         # samples shown per row in the field grid
  nsim_pk: 64              # fresh simulator maps per theta for the P(k) reference
  nll_num_test: 256        # held-out fields for the bits/dim NLL
```

- [ ] **Step 4: Create the script skeleton**

Create `examples/sbi-benchmarks/gaussian_random_field/tarflow/train_tarflow_grf.py`:

```python
"""Train a conditional TarFlow p(field | theta) on gaussian_random_field (32x32).

Field-level NLE with a transformer autoregressive normalizing flow: the 32x32
GRF realization is the modeled variable (obs), theta = (log_std, alpha) the
condition. TarFlow uses the image tokenizer (patchify) with 2D rotary position
embeddings (use_rope=True) and the vector-prefix conditioning strategy: each
theta coordinate becomes one prefix token behind the prefix-LM mask.

Trained by exact max likelihood (ConditionalFlowPipeline, structured_obs=True);
sampling uses the KV-cached autoregressive sampler. Outputs: loss curves, a
truth-vs-samples field grid, radial power-spectrum overlays (simulator mean
+/- 1 sigma and the analytic power law), and the exact held-out NLL in
bits/dim -- a scalar the flow-matching baselines cannot provide.

Training data comes from the online sampler (fresh prior + simulator draws
each batch) by default; set `training.online: false` to train on the
pre-generated HF train split instead.

Usage (conda env `gensbi`):
    python train_tarflow_grf.py --config config/config_1.yaml
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
    """TarFlow over 32x32 fields: image tokenizer + rope, vector-prefix cond.

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
        head_dim=int(model_cfg.get("head_dim", 32)),
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run:
```bash
/lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python -m pytest tests/test_train_tarflow_grf.py -v
```
Expected: 4 PASSED.

- [ ] **Step 6: Commit**

```bash
git add examples/sbi-benchmarks/gaussian_random_field/tarflow tests/test_train_tarflow_grf.py
git commit -m "feat(grf tarflow): script skeleton — rope + vector-prefix flow builder, config"
```

---

### Task 2: Batch map and held-out bits/dim NLL helper

**Files:**
- Modify: `examples/sbi-benchmarks/gaussian_random_field/tarflow/train_tarflow_grf.py` (append after `build_training_config`)
- Test: `tests/test_train_tarflow_grf.py` (append)

**Interfaces:**
- Consumes: `build_flow` (Task 1) in the test; `TarFlow.log_prob(x, cond) -> (B,)`.
- Produces: `to_obs_cond(batch: tuple) -> tuple` (loader `.map` callback) and `heldout_bits_per_dim(flow, fields_norm, theta_norm, batch_size=64) -> float`, both used by `main` (Task 4).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_train_tarflow_grf.py`:

```python
def test_to_obs_cond_swaps_and_adds_channel():
    mod = _load_script_module()
    theta = np.zeros((4, 2), dtype=np.float32)
    x = np.zeros((4, 8, 8, 1), dtype=np.float32)
    obs, cond = mod.to_obs_cond((theta, x))
    assert obs.shape == (4, 8, 8, 1)     # obs = the field, native shape
    assert cond.shape == (4, 2, 1)       # cond = theta, channel-carrying


def test_heldout_bits_per_dim_matches_gaussian_at_identity_init():
    # zero_init=True (TarFlowParams default) makes every MetaBlock the
    # identity, so the untrained flow is exactly N(0, I): bits/dim of
    # N(0,1) data must equal 0.5*log2(2*pi*e) ~= 2.047 up to MC error.
    mod = _load_script_module()
    from flax import nnx
    import jax
    flow = mod.build_flow(nnx.Rngs(0), _TINY_MODEL_CFG)
    k1, k2 = jax.random.split(jax.random.key(0))
    fields = np.asarray(jax.random.normal(k1, (256, 8, 8, 1)))
    theta = np.asarray(jax.random.normal(k2, (256, 2, 1)))
    got = mod.heldout_bits_per_dim(flow, fields, theta, batch_size=128)
    expected = 0.5 * np.log2(2 * np.pi * np.e)
    assert abs(got - expected) < 0.05
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
/lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python -m pytest tests/test_train_tarflow_grf.py -v -k "to_obs_cond or bits_per_dim"
```
Expected: 2 FAILED with `AttributeError: module 'train_tarflow_grf' has no attribute ...`.

- [ ] **Step 3: Implement the helpers**

Append to `train_tarflow_grf.py`:

```python
def to_obs_cond(batch):
    """Loader yields (theta, x); the flow pipeline wants (obs=x, cond=theta).

    x already arrives in native image shape (B, 32, 32, 1); theta (B, 2)
    gains the channel axis the VectorConditioner expects: (B, 2, 1).
    Module-level (not a lambda) so it survives pickling if it ever moves
    before a prefetch stage.
    """
    theta, x = batch
    return x, theta[..., None]


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
/lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python -m pytest tests/test_train_tarflow_grf.py -v
```
Expected: 6 PASSED.

- [ ] **Step 5: Commit**

```bash
git add examples/sbi-benchmarks/gaussian_random_field/tarflow/train_tarflow_grf.py tests/test_train_tarflow_grf.py
git commit -m "feat(grf tarflow): batch map + exact held-out bits/dim NLL"
```

---

### Task 3: Diagnostics helpers (copied from the FM 256 script)

**Files:**
- Modify: `examples/sbi-benchmarks/gaussian_random_field/tarflow/train_tarflow_grf.py` (append after `heldout_bits_per_dim`)
- Test: `tests/test_train_tarflow_grf.py` (append)
- Reference (copy source, do not import): `examples/sbi-benchmarks/gaussian_random_field_256/train-grf.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `radial_power_spectrum(field, nbins=40) -> (k, pk)`, `theory_power_spectrum(k, log_std, alpha, field_size) -> pk`, `plot_losses(loss_array, val_loss_array, val_every, path)`, `plot_field_grid(truths, samples, thetas, n_show, path)`, `plot_power_spectra(sim_fields, samples, thetas, field_size, path)` — all used by `main` (Task 4).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_train_tarflow_grf.py`:

```python
def test_radial_power_spectrum_white_noise_level_and_range():
    mod = _load_script_module()
    rng = np.random.default_rng(0)
    field = rng.normal(size=(128, 128))
    k, pk = mod.radial_power_spectrum(field)
    assert k.shape == pk.shape and len(k) > 10
    assert k.min() > 0 and k.max() <= 0.5      # cycles/pixel, Nyquist bound
    assert np.all(pk > 0)
    # white noise: E[P(k)] = sigma^2 = 1; high-k bins average many modes
    high = k > 0.1
    assert abs(pk[high].mean() - 1.0) < 0.1


def test_plot_helpers_write_files(tmp_path):
    mod = _load_script_module()
    rng = np.random.default_rng(0)
    truths = rng.normal(size=(2, 16, 16))
    samples = [rng.normal(size=(3, 16, 16)) for _ in range(2)]
    sim_fields = [rng.normal(size=(4, 16, 16)) for _ in range(2)]
    thetas = np.array([[0.0, 2.0], [0.5, 3.0]])
    p_grid = tmp_path / "grid.png"
    p_pk = tmp_path / "pk.png"
    p_loss = tmp_path / "loss.png"
    mod.plot_field_grid(truths, samples, thetas, n_show=3, path=str(p_grid))
    mod.plot_power_spectra(sim_fields, samples, thetas, field_size=16,
                           path=str(p_pk))
    mod.plot_losses(np.ones(10), np.ones(10), val_every=100, path=str(p_loss))
    assert p_grid.exists() and p_pk.exists() and p_loss.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
/lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python -m pytest tests/test_train_tarflow_grf.py -v -k "radial or plot_helpers"
```
Expected: 2 FAILED with `AttributeError`.

- [ ] **Step 3: Copy the five helpers**

Copy the following functions **verbatim** from
`examples/sbi-benchmarks/gaussian_random_field_256/train-grf.py` into
`train_tarflow_grf.py` (after `heldout_bits_per_dim`): `radial_power_spectrum`,
`plot_losses`, `plot_field_grid`, `theory_power_spectrum`, `plot_power_spectra`.
For reference, the exact code being copied:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
/lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python -m pytest tests/test_train_tarflow_grf.py -v
```
Expected: 8 PASSED.

- [ ] **Step 5: Commit**

```bash
git add examples/sbi-benchmarks/gaussian_random_field/tarflow/train_tarflow_grf.py tests/test_train_tarflow_grf.py
git commit -m "feat(grf tarflow): P(k) + plotting diagnostics copied from FM 256 script"
```

---

### Task 4: main() wiring + smoke config

**Files:**
- Modify: `examples/sbi-benchmarks/gaussian_random_field/tarflow/train_tarflow_grf.py` (append `main` + `__main__` block at end of file)
- Create: `examples/sbi-benchmarks/gaussian_random_field/tarflow/config/config_smoke.yaml`
- Test: `tests/test_train_tarflow_grf.py` (append)

**Interfaces:**
- Consumes: everything from Tasks 1–3 (`build_flow`, `build_training_config`, `to_obs_cond`, `heldout_bits_per_dim`, the five diagnostics helpers); `ConditionalFlowPipeline(flow, train, val, dim_obs, dim_cond, structured_obs=True, training_config=...)`; `pipeline.train(rngs, save_model=True) -> (loss_array, val_loss_array)`; `pipeline.sample(key, x_o, nsamples, use_ema) -> (nsamples, 32, 32, 1)`; `pipeline.restore_model()`; `OnlineTaskDataset` / `TaskDataset` from sbibm_jax.
- Produces: `main(config_path: str)` — the script entry point.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_train_tarflow_grf.py`:

```python
def test_main_exists_and_smoke_config_is_tiny():
    import yaml
    mod = _load_script_module()
    assert callable(mod.main)
    smoke = (_REPO_ROOT / "examples/sbi-benchmarks/gaussian_random_field"
             / "tarflow/config/config_smoke.yaml")
    with open(smoke) as f:
        cfg = yaml.safe_load(f)
    assert cfg["training"]["nsteps"] <= 50          # cheap enough for CPU
    assert cfg["model"]["head_dim"] % 4 == 0        # rope requirement holds
    assert cfg["sampling"]["nsamples"] <= 4
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
/lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python -m pytest tests/test_train_tarflow_grf.py -v -k smoke
```
Expected: FAIL (`AttributeError: ... no attribute 'main'`).

- [ ] **Step 3: Write the smoke config**

Create `examples/sbi-benchmarks/gaussian_random_field/tarflow/config/config_smoke.yaml`:

```yaml
# CPU smoke config: verifies the full train -> NLL -> sample -> plot path.
# Not a science run. Requires the gaussian_random_field HF dataset cache
# (val loader + df_test come from the offline split).
model:
  img_size: 32
  patch_size: 8            # -> T = 16 tokens (fast CPU sampling)
  img_channels: 1
  cond_dim: 2
  use_rope: true
  rope_theta: 10000
  head_dim: 16
  num_heads: 2             # channels = 32
  num_blocks: 2
  layers_per_block: 1
  permutation: flip

optimizer:
  max_lr: 1.0e-4
  min_lr: 1.0e-6
  warmup_steps: 5
  decay_transition: 0.80

training:
  batch_size: 32
  val_batch_size: 32
  max_workers: null
  online: true
  nsteps: 20
  ema_decay: 0.999
  val_every: 10
  early_stopping: false
  multistep: 1
  experiment_id: 99
  train_model: true
  restore_model: false
  seed: 0

sampling:
  num_thetas: 1
  nsamples: 2
  nsamples_grid: 2
  nsim_pk: 4
  nll_num_test: 16
```

- [ ] **Step 4: Implement main()**

Append to `train_tarflow_grf.py`:

```python
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
    online = tcfg.get("online", True)
    max_workers = tcfg.get("max_workers")  # None -> in-process / no prefetch

    task = OnlineTaskDataset("gaussian_random_field", normalize=True)
    offline_task = TaskDataset(
        "gaussian_random_field",
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
            os.path.join(imgs_dir, f"grf_loss_conf{experiment}.png"),
        )

    # --- test rows: raw truths + normalized thetas ---
    n_thetas = scfg["num_thetas"]
    n_nll = scfg.get("nll_num_test", 256)
    rows = offline_task.df_test[:max(n_thetas, n_nll)]  # slice first: decodes only these
    thetas_raw = np.asarray(rows["thetas"], dtype=np.float32)  # (n, 2)
    truths = np.asarray(rows["xs"], dtype=np.float32)          # (n, 32, 32)
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
            )  # (nsamples, 32, 32, 1), normalized; KV-cached sampler
            s = np.asarray(task.unnormalize_x(s), dtype=np.float32)[..., 0]
            samples.append(s)
            print(f"[{tag}] theta {i}: sampled {s.shape}, "
                  f"finite={np.isfinite(s).all()}")

        plot_field_grid(
            truths[:n_thetas], samples, thetas_raw, scfg["nsamples_grid"],
            os.path.join(imgs_dir, f"grf_fields_conf{experiment}_{tag}.png"),
        )
        plot_power_spectra(
            sim_fields, samples, thetas_raw, field_size,
            os.path.join(imgs_dir, f"grf_pk_conf{experiment}_{tag}.png"),
        )
    print(f"Plots written to {imgs_dir} (experiment {experiment}; _ema and _raw)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=os.path.join(EXAMPLE_DIR, "config", "config_1.yaml"))
    main(parser.parse_args().config)
```

- [ ] **Step 5: Run the full test file**

Run:
```bash
/lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python -m pytest tests/test_train_tarflow_grf.py -v
```
Expected: 9 PASSED.

- [ ] **Step 6: End-to-end CPU smoke run**

Run (needs the `gaussian_random_field` HF cache, present on this cluster):
```bash
cd examples/sbi-benchmarks/gaussian_random_field/tarflow
JAX_PLATFORMS=cpu /lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python \
    train_tarflow_grf.py --config config/config_smoke.yaml
cd -
```
Expected output: parameter count print, 20 training steps, two
`held-out NLL [...] bits/dim` lines (finite values), `[ema] theta 0: sampled
(2, 32, 32), finite=True` and the `[raw]` twin, and five PNGs under
`examples/sbi-benchmarks/gaussian_random_field/tarflow/imgs/`
(`grf_loss_conf99.png`, `grf_fields_conf99_{ema,raw}.png`,
`grf_pk_conf99_{ema,raw}.png`). Sample quality is irrelevant at 20 steps —
only that every stage runs and stays finite.

If the smoke run fails inside pipeline/dataset code, fix the script's usage
(shapes, argument names), not the library. Re-run until it completes.

- [ ] **Step 7: Commit**

```bash
git add examples/sbi-benchmarks/gaussian_random_field/tarflow tests/test_train_tarflow_grf.py
git commit -m "feat(grf tarflow): main() — train, held-out NLL, sample, P(k) diagnostics"
```

---

### Task 5: Condor submit file + final verification

**Files:**
- Create: `sub/train_model_tarflow_grf.sub`
- Reference (pattern source): `sub/train_model_tarflow_npe.sub`

**Interfaces:**
- Consumes: the finished `train_tarflow_grf.py` + `config/config_1.yaml` (Tasks 1–4).
- Produces: nothing consumed downstream; cluster submission entry point.

- [ ] **Step 1: Create the submit file**

Create `sub/train_model_tarflow_grf.sub` (mirrors `sub/train_model_tarflow_npe.sub`; note the `workdir` uses the `/lhome/.../data/github` path alias like every other sub file):

```
experiment_name = tarflow_grf
version = 1a
workdir = /lhome/ific/a/aamerio/data/github/GenSBI-examples/examples/sbi-benchmarks/gaussian_random_field/tarflow
script_path = train_tarflow_grf.py

universe = vanilla

request_memory = 64 GB
request_cpus = 8

executable = train_model.sh
arguments = "$(workdir) $(script_path) --config config/config_1.yaml"
getenv = True
request_gpus = 1
requirements = (Machine != "mlwn11.ific.uv.es")
+UseNvidiaA100 = True


log                     = condor_logs/logs_$(experiment_name)_$(version).log
output                  = condor_logs/outfile_$(experiment_name)_$(version).out
error                   = condor_logs/errors_$(experiment_name)_$(version).err

#########

queue
```

- [ ] **Step 2: Run the full test suite one last time**

Run:
```bash
/lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python -m pytest tests/test_train_tarflow_grf.py -v
```
Expected: 9 PASSED.

- [ ] **Step 3: Commit**

```bash
git add sub/train_model_tarflow_grf.sub
git commit -m "feat(grf tarflow): condor submit file"
```

- [ ] **Step 4: Report cluster handoff**

The GPU science run is submitted by the user, not this plan:
`condor_submit sub/train_model_tarflow_grf.sub` from `sub/`. Note this in the
final task report; do not submit it.
