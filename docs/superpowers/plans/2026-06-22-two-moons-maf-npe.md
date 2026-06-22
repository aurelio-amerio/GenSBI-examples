# Two-Moons NPE (RQSpline MAF) Example — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained, YAML-driven example that trains a conditional MAF (RQSpline transformer) on the `two_moons` SBI benchmark via NPE and produces a posterior contour plot for one observation.

**Architecture:** A single example script (`train_maf_npe.py`) reads a YAML config, builds a `Flow` via `make_maf`, wraps it in `ConditionalFlowPipeline` (the flow *is* the density model; trained by max-likelihood NLL), standardizes θ from precomputed stats (loader stays unstandardized), trains, then evaluates `q(θ|x_o)` on a grid and plots it with `plot_2d_dist_contour`. Pure-logic helpers are module-level functions so they can be unit-tested offline (no HF download, no GPU); `main()` does orchestration only.

**Tech Stack:** Python 3.12, JAX + Flax NNX, `gensbi` (editable lib), `gensbi_examples` (tasks), `grain` datasets, `matplotlib`, `pyyaml`, `pytest` (+ pytest-env sets `JAX_PLATFORMS=cpu`).

**Environment / commands:**
- Interpreter: `/lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python` (call it `PY`).
- Repo root: `/lhome/ific/a/aamerio/data/github/GenSBI-examples`.
- Run tests (CPU, offline, xdist disabled for clean output):
  `cd /lhome/ific/a/aamerio/data/github/GenSBI-examples && JAX_PLATFORMS=cpu $PY -m pytest tests/test_train_maf_npe.py -v -n0`
- This login node has **no GPU and constrained HF network**. All unit tests below are offline/CPU. The full end-to-end training run (Task 6) needs a GPU compute node with HF access.

---

## File Structure

- **Create** `examples/sbi-benchmarks/two_moons/maf/config/config_maf_npe.yaml` — experiment config (task, model, optimizer, training, evaluation sections).
- **Create** `examples/sbi-benchmarks/two_moons/maf/train_maf_npe.py` — the example: module-level helpers + `main()` guarded by `if __name__ == "__main__"`.
- **Create** `tests/test_train_maf_npe.py` — offline unit tests for the helpers (built up incrementally across Tasks 1–5).

Naming note: the `npe` suffix is deliberate — the planned NLE sibling will be `train_maf_nle.py` / `config_maf_nle.yaml` in the same `maf/` dir.

---

## Task 1: Config YAML + config-load test

**Files:**
- Create: `examples/sbi-benchmarks/two_moons/maf/config/config_maf_npe.yaml`
- Create (new): `tests/test_train_maf_npe.py`

- [ ] **Step 1: Write the config YAML**

Create `examples/sbi-benchmarks/two_moons/maf/config/config_maf_npe.yaml`:

```yaml
task_name: two_moons

strategy:
  method: npe
  model: maf

model:
  n_layers: 8
  transformer: rqspline      # affine | rqspline
  num_bins: 8                # used only when transformer == rqspline
  nn_width: 64
  nn_depth: 2
  permutation: reverse       # reverse | random
  standardize: true
  zero_init: true

optimizer:
  warmup_steps: 500
  decay_transition: 0.80
  max_lr: 4.0e-4
  min_lr: 4.0e-6

training:
  batch_size: 4096
  nsamples: 90000            # training subset; MUST be < task.max_samples
  nsteps: 20000
  ema_decay: 0.999
  val_every: 100
  early_stopping: true
  experiment_id: 1
  restore_model: false
  train_model: true

evaluation:
  observation_idx: 1
  grid_size: 200
  padding: 0.5
  n_ref_overlay: 2000
```

- [ ] **Step 2: Write the failing test (config loads + required structure)**

Create `tests/test_train_maf_npe.py`:

```python
import os
os.environ.setdefault("MPLBACKEND", "Agg")  # headless-safe before any pyplot import

import importlib.util
import pathlib

import numpy as np
import pytest

# --- load the example script as a module (it lives outside the package) ---
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "examples/sbi-benchmarks/two_moons/maf/train_maf_npe.py"
_CONFIG = _REPO_ROOT / "examples/sbi-benchmarks/two_moons/maf/config/config_maf_npe.yaml"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("train_maf_npe", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_config_yaml_has_required_sections():
    import yaml
    with open(_CONFIG) as f:
        cfg = yaml.safe_load(f)
    assert cfg["task_name"] == "two_moons"
    for section in ("model", "optimizer", "training", "evaluation"):
        assert section in cfg, f"missing section {section!r}"
    assert cfg["model"]["transformer"] in ("affine", "rqspline")
    assert cfg["training"]["nsamples"] < 100_000  # default get_train_dataset cap
    assert cfg["evaluation"]["grid_size"] > 0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /lhome/ific/a/aamerio/data/github/GenSBI-examples && JAX_PLATFORMS=cpu $PY -m pytest tests/test_train_maf_npe.py::test_config_yaml_has_required_sections -v -n0`
Expected: PASS (the YAML exists from Step 1; this test depends only on the YAML, not the script). If the YAML is missing/malformed it FAILs at `open`/`safe_load`.

> Note: this first test validates config-as-data and intentionally does not import the script (which doesn't exist yet). The `_load_script_module` helper is defined now for use in later tasks.

- [ ] **Step 4: Commit**

```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI-examples
git add examples/sbi-benchmarks/two_moons/maf/config/config_maf_npe.yaml tests/test_train_maf_npe.py
git commit -m "feat(maf): two-moons NPE config + config-load test"
```

---

## Task 2: Script skeleton + transformer/flow builders

**Files:**
- Create: `examples/sbi-benchmarks/two_moons/maf/train_maf_npe.py`
- Modify: `tests/test_train_maf_npe.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_train_maf_npe.py`:

```python
def test_build_transformer_selects_type():
    mod = _load_script_module()
    from gensbi.normalizing_flows import Affine, RQSpline
    assert isinstance(mod.build_transformer({"transformer": "affine"}), Affine)
    rq = mod.build_transformer({"transformer": "rqspline", "num_bins": 6})
    assert isinstance(rq, RQSpline)
    assert rq.num_bins == 6
    with pytest.raises(ValueError):
        mod.build_transformer({"transformer": "nope"})


def test_build_flow_runs_log_prob():
    mod = _load_script_module()
    from flax import nnx
    import jax.numpy as jnp
    flow = mod.build_flow(nnx.Rngs(0), dim_obs=2, dim_cond=2,
                          model_cfg={"n_layers": 2, "transformer": "affine"})
    x = jnp.zeros((4, 2))
    cond = jnp.zeros((4, 2))
    lp = flow.log_prob(x, cond)
    assert lp.shape == (4,)
    assert bool(jnp.all(jnp.isfinite(lp)))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /lhome/ific/a/aamerio/data/github/GenSBI-examples && JAX_PLATFORMS=cpu $PY -m pytest tests/test_train_maf_npe.py -k "build_transformer or build_flow" -v -n0`
Expected: FAIL — `FileNotFoundError`/`exec_module` error because `train_maf_npe.py` does not exist yet.

- [ ] **Step 3: Create the script skeleton + builders**

Create `examples/sbi-benchmarks/two_moons/maf/train_maf_npe.py`:

```python
"""Two-moons NPE example: train a conditional MAF (RQSpline) and plot q(theta|x_o).

Run (on a GPU node with HF access):
    python train_maf_npe.py --config config/config_maf_npe.yaml

The flow IS the density model (ConditionalFlowPipeline, max-likelihood NPE).
Helpers are module-level and import-safe so they can be unit-tested on CPU.
"""

import os

# Import-safe (tests / module import): default to CPU. When run as the main
# training script we leave JAX_PLATFORMS unset so it can use a GPU.
if __name__ != "__main__":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse

import numpy as np
import yaml
import jax
import jax.numpy as jnp
from flax import nnx
import matplotlib.pyplot as plt

from gensbi.normalizing_flows import make_maf, Affine, RQSpline
from gensbi.recipes import ConditionalFlowPipeline
from gensbi.utils.plotting import plot_2d_dist_contour
from gensbi_examples.tasks import get_task, _load_precomputed_stats


def build_transformer(model_cfg):
    """Return the elementwise transformer named in the model config."""
    name = str(model_cfg.get("transformer", "rqspline")).lower()
    if name == "affine":
        return Affine()
    if name == "rqspline":
        return RQSpline(num_bins=int(model_cfg.get("num_bins", 8)))
    raise ValueError(f"unknown transformer {name!r} (expected 'affine' or 'rqspline')")


def build_flow(rngs, dim_obs, dim_cond, model_cfg):
    """Build the MAF Flow from the model config section."""
    transformer = build_transformer(model_cfg)
    return make_maf(
        rngs,
        dim=dim_obs,
        cond_dim=dim_cond,
        n_layers=int(model_cfg.get("n_layers", 8)),
        transformer=transformer,
        nn_width=int(model_cfg.get("nn_width", 64)),
        nn_depth=int(model_cfg.get("nn_depth", 2)),
        permutation=str(model_cfg.get("permutation", "reverse")),
        standardize=bool(model_cfg.get("standardize", True)),
        zero_init=bool(model_cfg.get("zero_init", True)),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /lhome/ific/a/aamerio/data/github/GenSBI-examples && JAX_PLATFORMS=cpu $PY -m pytest tests/test_train_maf_npe.py -k "build_transformer or build_flow" -v -n0`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI-examples
git add examples/sbi-benchmarks/two_moons/maf/train_maf_npe.py tests/test_train_maf_npe.py
git commit -m "feat(maf): script skeleton + transformer/flow builders"
```

---

## Task 3: training_config builder + obs-stats loader

**Files:**
- Modify: `examples/sbi-benchmarks/two_moons/maf/train_maf_npe.py`
- Modify: `tests/test_train_maf_npe.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_train_maf_npe.py`:

```python
def test_build_training_config_merges_defaults_and_overrides(tmp_path):
    mod = _load_script_module()
    from gensbi.recipes import ConditionalFlowPipeline
    cfg = {
        "optimizer": {"max_lr": 4.0e-4, "min_lr": 4.0e-6, "warmup_steps": 500,
                      "decay_transition": 0.80},
        "training": {"nsteps": 123, "ema_decay": 0.99, "val_every": 50,
                     "early_stopping": True, "experiment_id": 7},
    }
    tc = mod.build_training_config(cfg, checkpoint_dir=str(tmp_path / "ckpt"))
    # every key the pipeline reads must be present (defaults filled in)
    for key in ConditionalFlowPipeline.get_default_training_config():
        assert key in tc, f"missing required training_config key {key!r}"
    # overrides applied
    assert tc["nsteps"] == 123
    assert tc["max_lr"] == 4.0e-4
    assert tc["experiment_id"] == 7
    assert tc["checkpoint_dir"] == str(tmp_path / "ckpt")


def test_load_obs_stats_shape_two_moons():
    mod = _load_script_module()
    import jax.numpy as jnp
    mean, std = mod.load_obs_stats("two_moons", dim_obs=2)
    assert mean.shape == (2,)
    assert std.shape == (2,)
    assert bool(jnp.all(std > 0))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /lhome/ific/a/aamerio/data/github/GenSBI-examples && JAX_PLATFORMS=cpu $PY -m pytest tests/test_train_maf_npe.py -k "training_config or obs_stats" -v -n0`
Expected: FAIL — `AttributeError: module 'train_maf_npe' has no attribute 'build_training_config'`.

- [ ] **Step 3: Add the helpers**

Append to `examples/sbi-benchmarks/two_moons/maf/train_maf_npe.py` (after `build_flow`):

```python
def build_training_config(config, checkpoint_dir):
    """Start from the pipeline defaults, overlay YAML optimizer+training, set ckpt dir.

    ConditionalFlowPipeline reads training_config keys eagerly in __init__ with no
    merge, so it must be complete. Extra keys (batch_size, nsamples, restore_model,
    train_model) are harmless — the pipeline only reads the keys it knows.
    """
    tc = ConditionalFlowPipeline.get_default_training_config()
    tc.update(config.get("optimizer", {}))
    tc.update(config.get("training", {}))
    tc["checkpoint_dir"] = checkpoint_dir
    return tc


def load_obs_stats(task_name, dim_obs):
    """Precomputed θ (obs) mean/std as shape (dim_obs,) — no fitting, no HF download.

    Stats ship as (1, dim_obs, 1) in gensbi_examples/stats/stats_<task>.npz and are
    loaded by the same internal helper the Task uses.
    """
    stats = _load_precomputed_stats(task_name)
    if stats is None:
        raise FileNotFoundError(f"no precomputed stats for task {task_name!r}")
    mean = jnp.asarray(stats["obs_mean"]).reshape(dim_obs)
    std = jnp.asarray(stats["obs_std"]).reshape(dim_obs)
    return mean, std
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /lhome/ific/a/aamerio/data/github/GenSBI-examples && JAX_PLATFORMS=cpu $PY -m pytest tests/test_train_maf_npe.py -k "training_config or obs_stats" -v -n0`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI-examples
git add examples/sbi-benchmarks/two_moons/maf/train_maf_npe.py tests/test_train_maf_npe.py
git commit -m "feat(maf): training_config builder + precomputed obs-stats loader"
```

---

## Task 4: standardization, grid, density, and plot helpers

**Files:**
- Modify: `examples/sbi-benchmarks/two_moons/maf/train_maf_npe.py`
- Modify: `tests/test_train_maf_npe.py`

- [ ] **Step 1: Write the failing tests (with a shared tiny-pipeline fixture)**

Append to `tests/test_train_maf_npe.py`:

```python
@pytest.fixture
def tiny_pipeline(tmp_path):
    """A minimal ConditionalFlowPipeline on a tiny CPU flow (no HF, no training)."""
    mod = _load_script_module()
    from flax import nnx
    from gensbi.recipes import ConditionalFlowPipeline
    import jax.numpy as jnp

    flow = mod.build_flow(nnx.Rngs(0), dim_obs=2, dim_cond=2,
                          model_cfg={"n_layers": 2, "transformer": "affine"})
    obs = jnp.zeros((8, 2, 1))   # (B, dim_obs, 1) — theta
    cond = jnp.zeros((8, 2, 1))  # (B, dim_cond, 1) — x
    dummy = [(obs, cond)]
    tc = ConditionalFlowPipeline.get_default_training_config()
    tc["checkpoint_dir"] = str(tmp_path / "ckpt")
    pipe = ConditionalFlowPipeline(flow, dummy, dummy, dim_obs=2, dim_cond=2,
                                   training_config=tc)
    return mod, pipe


def test_apply_standardization_sets_buffers(tiny_pipeline):
    mod, pipe = tiny_pipeline
    import jax.numpy as jnp
    mean = jnp.array([1.0, -2.0])
    std = jnp.array([3.0, 4.0])
    mod.apply_standardization(pipe, mean, std)
    assert pipe._standardized is True
    # both model and EMA must carry the buffer (EMA averages only Params)
    for m in (pipe.model, pipe.ema_model):
        std_bijection = [b for b in m.chain.bijections
                         if b.__class__.__name__ == "Standardize"][0]
        assert bool(jnp.allclose(std_bijection.mean.value, mean))
        assert bool(jnp.allclose(std_bijection.std.value, std))


def test_make_density_grid_shapes_and_bounds():
    mod = _load_script_module()
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(500, 2)) * 0.3
    xx, yy, grid_pts = mod.make_density_grid(ref, grid_size=20, padding=0.5)
    assert xx.shape == (20, 20)
    assert yy.shape == (20, 20)
    assert grid_pts.shape == (400, 2)
    assert grid_pts[:, 0].min() <= ref[:, 0].min()
    assert grid_pts[:, 0].max() >= ref[:, 0].max()


def test_posterior_density_shape_and_finite(tiny_pipeline):
    mod, pipe = tiny_pipeline
    import jax.numpy as jnp
    mod.apply_standardization(pipe, jnp.zeros(2), jnp.ones(2))
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(500, 2)) * 0.3
    xx, yy, grid_pts = mod.make_density_grid(ref, grid_size=15, padding=0.5)
    Z = mod.posterior_density(pipe, grid_pts, obs=np.array([0.1, 0.2]), grid_size=15)
    assert Z.shape == (15, 15)
    assert np.all(np.isfinite(Z))
    assert np.all(Z >= 0.0)


def test_plot_posterior_contour_returns_figure(tiny_pipeline):
    mod, pipe = tiny_pipeline
    import jax.numpy as jnp
    import matplotlib
    mod.apply_standardization(pipe, jnp.zeros(2), jnp.ones(2))
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(500, 2)) * 0.3
    xx, yy, grid_pts = mod.make_density_grid(ref, grid_size=15, padding=0.5)
    Z = mod.posterior_density(pipe, grid_pts, obs=np.array([0.0, 0.0]), grid_size=15)
    fig, ax = mod.plot_posterior_contour(xx, yy, Z, true_param=np.array([0.0, 0.0]),
                                         ref_samples=ref, n_ref_overlay=100)
    assert isinstance(fig, matplotlib.figure.Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /lhome/ific/a/aamerio/data/github/GenSBI-examples && JAX_PLATFORMS=cpu $PY -m pytest tests/test_train_maf_npe.py -k "standardization or density_grid or posterior_density or plot_posterior" -v -n0`
Expected: FAIL — `AttributeError: module 'train_maf_npe' has no attribute 'apply_standardization'`.

- [ ] **Step 3: Add the helpers**

Append to `examples/sbi-benchmarks/two_moons/maf/train_maf_npe.py` (after `load_obs_stats`):

```python
def apply_standardization(pipeline, mean, std):
    """Set the θ Standardize buffers on both model and EMA from precomputed stats.

    EMA averages only Params, so its non-Param Standardize buffer must be set too.
    Marks the pipeline standardized to suppress the train-time 'did you fit?' warning.
    """
    pipeline.model.set_standardization(mean, std)
    pipeline.ema_model.set_standardization(mean, std)
    pipeline._standardized = True


def make_density_grid(ref_samples, grid_size, padding=0.5):
    """Build a 2D θ grid framing the reference samples (+padding).

    Returns (xx, yy, grid_pts): xx, yy are (G, G) meshgrids (indexing='xy');
    grid_pts is (G*G, 2) row-aligned with xx.ravel()/yy.ravel() (C order).
    """
    ref = np.asarray(ref_samples).reshape(-1, 2)
    lo = ref.min(axis=0) - padding
    hi = ref.max(axis=0) + padding
    xs = np.linspace(lo[0], hi[0], grid_size)
    ys = np.linspace(lo[1], hi[1], grid_size)
    xx, yy = np.meshgrid(xs, ys)
    grid_pts = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    return xx, yy, grid_pts


def posterior_density(pipeline, grid_pts, obs, grid_size, use_ema=True):
    """Evaluate q(θ|obs) on grid_pts and reshape to (G, G) aligned with the meshgrid."""
    grid_pts = jnp.asarray(grid_pts)                       # (G*G, 2)
    logp = pipeline.log_prob(grid_pts, obs, use_ema=use_ema)  # (G*G,)
    Z = np.asarray(jnp.exp(logp)).reshape(grid_size, grid_size)
    return Z


def plot_posterior_contour(xx, yy, Z, true_param, ref_samples=None, n_ref_overlay=2000):
    """Contour plot of the posterior with the true θ marked and a light ref-sample overlay."""
    fig, ax = plot_2d_dist_contour(xx, yy, Z, true_param=np.asarray(true_param).reshape(-1))
    if ref_samples is not None and n_ref_overlay > 0:
        ref = np.asarray(ref_samples).reshape(-1, 2)[:n_ref_overlay]
        ax.scatter(ref[:, 0], ref[:, 1], s=2, alpha=0.15, color="k", zorder=5)
    return fig, ax
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /lhome/ific/a/aamerio/data/github/GenSBI-examples && JAX_PLATFORMS=cpu $PY -m pytest tests/test_train_maf_npe.py -v -n0`
Expected: PASS (all tests so far — config, builders, training_config, stats, standardization, grid, density, plot).

- [ ] **Step 5: Commit**

```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI-examples
git add examples/sbi-benchmarks/two_moons/maf/train_maf_npe.py tests/test_train_maf_npe.py
git commit -m "feat(maf): standardization + grid + density + contour-plot helpers"
```

---

## Task 5: `main()` orchestration

**Files:**
- Modify: `examples/sbi-benchmarks/two_moons/maf/train_maf_npe.py`

No new unit test — `main()` is orchestration over already-tested helpers + HF data + (optional) GPU training. It is exercised in Task 6's manual verification. The only automated check here is that the module still imports cleanly (the existing suite re-imports it).

- [ ] **Step 1: Add `main()` and the entrypoint guard**

Append to `examples/sbi-benchmarks/two_moons/maf/train_maf_npe.py`:

```python
def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(here, "config", "config_maf_npe.yaml")
    parser = argparse.ArgumentParser(description="Two-moons NPE (MAF) training/eval")
    parser.add_argument("--config", type=str, default=default_config,
                        help="Path to the YAML config.")
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    exp_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(exp_dir, "checkpoints")
    img_dir = os.path.join(exp_dir, "imgs")
    os.makedirs(img_dir, exist_ok=True)

    task_name = config["task_name"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    eval_cfg = config["evaluation"]

    # --- task / data (UNSTANDARDIZED loader: normalize_data=False) ---
    task = get_task(task_name, kind="conditional", normalize_data=False)
    dim_obs, dim_cond = task.dim_obs, task.dim_cond
    train_ds = task.get_train_dataset(int(train_cfg["batch_size"]),
                                      nsamples=int(train_cfg["nsamples"]))
    val_ds = task.get_val_dataset(512)

    # --- flow + pipeline ---
    flow = build_flow(nnx.Rngs(0), dim_obs, dim_cond, model_cfg)
    training_config = build_training_config(config, checkpoint_dir)
    pipeline = ConditionalFlowPipeline(flow, train_ds, val_ds, dim_obs, dim_cond,
                                       training_config=training_config)

    # --- standardize θ from precomputed stats (no fitting; loader stays raw) ---
    mean, std = load_obs_stats(task_name, dim_obs)
    apply_standardization(pipeline, mean, std)

    # --- train / restore ---
    if train_cfg.get("restore_model", False):
        print("Restoring model from checkpoint...")
        pipeline.restore_model()
    if train_cfg.get("train_model", True):
        print("Starting training...")
        pipeline.train(nnx.Rngs(0))
        print("Training complete.")

    # --- evaluate posterior + contour plot for one observation ---
    idx = int(eval_cfg["observation_idx"])
    obs, ref_samples = task.get_reference(idx)
    true_param = np.asarray(task.get_true_parameters(idx)).reshape(-1)

    grid_size = int(eval_cfg["grid_size"])
    xx, yy, grid_pts = make_density_grid(ref_samples, grid_size,
                                         padding=float(eval_cfg.get("padding", 0.5)))
    Z = posterior_density(pipeline, grid_pts, obs, grid_size, use_ema=True)
    fig, ax = plot_posterior_contour(xx, yy, Z, true_param, ref_samples=ref_samples,
                                     n_ref_overlay=int(eval_cfg.get("n_ref_overlay", 2000)))
    out_path = os.path.join(img_dir, f"posterior_contour_obs{idx}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved posterior contour to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the module still imports and the full suite passes**

Run: `cd /lhome/ific/a/aamerio/data/github/GenSBI-examples && JAX_PLATFORMS=cpu $PY -m pytest tests/test_train_maf_npe.py -v -n0`
Expected: PASS (all tests; importing the now-complete module must not error).

- [ ] **Step 3: Sanity-check the script parses without running training**

Run: `cd /lhome/ific/a/aamerio/data/github/GenSBI-examples/examples/sbi-benchmarks/two_moons/maf && JAX_PLATFORMS=cpu $PY -c "import importlib.util as u; s=u.spec_from_file_location('m','train_maf_npe.py'); m=u.module_from_spec(s); s.loader.exec_module(m); print('import OK; has main:', hasattr(m,'main'))"`
Expected: prints `import OK; has main: True` (no training is triggered on import).

- [ ] **Step 4: Commit**

```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI-examples
git add examples/sbi-benchmarks/two_moons/maf/train_maf_npe.py
git commit -m "feat(maf): main() orchestration for two-moons NPE train+plot"
```

---

## Task 6: End-to-end run + visual verification

**Files:** none changed (produces `checkpoints/` and `imgs/posterior_contour_obs1.png`).

This task needs a **GPU compute node with Hugging Face network access** (the login node has neither). If only CPU is available, do a reduced smoke run first (Step 1) to confirm the pipeline executes end-to-end and writes a plot, then the full run (Step 2) for quality.

- [ ] **Step 1 (optional CPU smoke): run a few steps to confirm it executes**

Make a temporary low-cost config copy and run it:

```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI-examples/examples/sbi-benchmarks/two_moons/maf
cp config/config_maf_npe.yaml config/_smoke.yaml
$PY - <<'PY'
import yaml
p = "config/_smoke.yaml"
c = yaml.safe_load(open(p))
c["training"].update({"nsteps": 100, "batch_size": 1024, "val_every": 50,
                      "early_stopping": False})
c["evaluation"]["grid_size"] = 50
yaml.safe_dump(c, open(p, "w"))
print("smoke config written")
PY
JAX_PLATFORMS=cpu $PY train_maf_npe.py --config config/_smoke.yaml
```
Expected: training logs print, then `Saved posterior contour to .../imgs/posterior_contour_obs1.png`. Delete the smoke artifacts afterward: `rm config/_smoke.yaml`. (Plot quality is meaningless at 100 steps — this only proves the wiring.)

- [ ] **Step 2: Full run on a GPU node**

```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI-examples/examples/sbi-benchmarks/two_moons/maf
$PY train_maf_npe.py --config config/config_maf_npe.yaml
```
Expected: training to ~20k steps (or early-stopped), then `Saved posterior contour to .../imgs/posterior_contour_obs1.png`.

- [ ] **Step 3: Visually verify the contour**

Open `imgs/posterior_contour_obs1.png`. Expected: two crescent-shaped high-density regions (the two-moons posterior), the overlaid reference samples sitting on top of the filled contours, and the true-θ marker (square + crosshair) inside a high-density region. If the contour is a blurry single blob, increase `n_layers`/`num_bins`/`nsteps` in the YAML (bump `experiment_id`) and re-run.

- [ ] **Step 4: Commit the produced figure (optional, matches repo convention of shipping example imgs)**

```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI-examples
git add examples/sbi-benchmarks/two_moons/maf/imgs/posterior_contour_obs1.png
git commit -m "docs(maf): two-moons NPE posterior contour figure"
```

> Note: do **not** commit `checkpoints/` (large binaries); add a `.gitignore` entry if the repo doesn't already ignore them.

---

## Self-Review

**Spec coverage:**
- Self-contained, YAML-driven, NPE via flow → Tasks 1–5.
- RQSpline default with affine selectable → `build_transformer` (Task 2), config (Task 1).
- Unstandardized loader (`normalize_data=False`) → `main()` (Task 5).
- Standardization from precomputed stats, both model+EMA, no fitting → `load_obs_stats` + `apply_standardization` (Tasks 3–4).
- Complete training_config (defaults + overlay) → `build_training_config` (Task 3).
- Grid over θ from reference extent, `log_prob`→exp→`plot_2d_dist_contour`, ref overlay, true_param → `make_density_grid`/`posterior_density`/`plot_posterior_contour` (Task 4), wired in `main()` (Task 5).
- Outputs under script dir; `npe` filename suffix → Task 5 / File Structure.
- Out-of-scope items (diagnostics, model card, NLE) → not present. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code; every command has expected output. ✓

**Type/name consistency:** Helper names used identically across tasks and `main()` — `build_transformer`, `build_flow`, `build_training_config`, `load_obs_stats`, `apply_standardization`, `make_density_grid`, `posterior_density`, `plot_posterior_contour`. `make_density_grid` returns `(xx, yy, grid_pts)`; `posterior_density(pipeline, grid_pts, obs, grid_size, use_ema)` returns `Z`; `plot_posterior_contour(xx, yy, Z, true_param, ref_samples, n_ref_overlay)` returns `(fig, ax)`. Consistent. ✓
