# Two-Moons MAF NLE + MCMC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a two-moons example that trains a MAF likelihood `q(x|theta)` (NLE) and draws posterior samples with NUTS via `gensbi.inference.NLEPosterior`, plotting the posterior contour with the true parameter marked.

**Architecture:** A near-mirror of the existing NPE script (`maf_NPE/train_maf_npe.py`), reusing `make_maf`, `ConditionalFlowPipeline`, and `TaskDataset`. The only structural change is the obs/cond flip — NLE trains `obs=x, cond=theta` (NPE is the reverse) — applied by mapping the loader's `(theta, x)` batches to `(x, theta)`. Inference combines the trained likelihood with the task prior under NUTS; the task prior must be made *validated* so it bounds the sampler.

**Tech Stack:** Python, JAX, flax.nnx, numpyro (NUTS), grain (data loading), gensbi, sbibm-jax, matplotlib/seaborn.

## Global Constraints

- **Run/verify environment:** the conda `gensbi` env, exactly as the cluster uses it (`sub/train_model.sh` runs `conda activate /lhome/ific/a/aamerio/miniforge3/envs/gensbi`). Interpreter for ALL commands: `/lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python`.
- **This repo is NOT a Python package** and has no committed test suite for examples. Do NOT add pytest files. Verify with inline CPU smoke checks and a final reduced end-to-end run. (The stale `tests/test_train_maf_npe.py` points at a non-existent `maf/` path — ignore it, do not touch it.)
- **Smoke-check harness rule:** helper-only smokes may use a heredoc (`python - <<'PY'`). Any smoke that constructs a `TaskDataset` loader runs `mp_prefetch` spawn-workers that re-import `__main__` by file path — those MUST be run as a real `.py` file under `__main__`, never a heredoc.
- **NLE convention:** `obs = x` (dim `task.dim_x`, the standardized autoregressive target), `cond = theta` (dim `task.dim_theta`).
- **Mirror the NPE script's style:** module-level, import-safe helpers; `os.environ.setdefault("JAX_PLATFORMS", "cpu")` only when imported (not `__main__`), so smokes run on CPU and the real run can use a GPU.
- **No edits to the GenSBI or sbibm-jax repos.** Use `NLEPosterior` as-is, single chain.
- **Prior MUST be validated** (`validate_args=True`) before use, or NUTS escapes the prior support and the posterior is wrong (verified). See Task 2.
- **`max_workers` cap:** shared node — use `max_workers=2` (as the NPE script does).
- **MCMC:** single chain, `num_warmup=1000`, `num_samples=50000`, `num_chains=1`.

---

### Task 1: New directory + config

**Files:**
- Create: `examples/sbi-benchmarks/two_moons/maf_NLE/config/config_maf_nle.yaml`

**Interfaces:**
- Produces: a YAML config with sections `task_name`, `strategy`, `model`, `optimizer`, `training`, `mcmc`, `evaluation` — consumed by `main()` in Task 3.

- [ ] **Step 1: Create the directory and write the config**

```bash
mkdir -p examples/sbi-benchmarks/two_moons/maf_NLE/config
```

Write `examples/sbi-benchmarks/two_moons/maf_NLE/config/config_maf_nle.yaml`:

```yaml
task_name: two_moons

strategy:
  method: nle
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
  max_lr: 2.0e-4
  min_lr: 2.0e-6

training:
  batch_size: 1024
  nsamples: 100000            # training subset; MUST be < task.max_samples
  nsteps: 50000
  ema_decay: 0.999
  val_every: 100
  early_stopping: true
  experiment_id: 1
  restore_model: false
  train_model: true

mcmc:
  num_warmup: 1000
  num_samples: 50000
  num_chains: 1

evaluation:
  observation_idx: 8
```

- [ ] **Step 2: Verify the config loads and has every required section**

Run:
```bash
/lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python - <<'PY'
import yaml
cfg = yaml.safe_load(open("examples/sbi-benchmarks/two_moons/maf_NLE/config/config_maf_nle.yaml"))
assert cfg["task_name"] == "two_moons"
for s in ("strategy", "model", "optimizer", "training", "mcmc", "evaluation"):
    assert s in cfg, f"missing section {s!r}"
assert cfg["strategy"]["method"] == "nle"
assert cfg["model"]["transformer"] in ("affine", "rqspline")
assert cfg["training"]["nsamples"] < 1_000_000
assert cfg["mcmc"]["num_chains"] == 1
assert cfg["evaluation"]["observation_idx"] >= 1
print("CONFIG OK")
PY
```
Expected: prints `CONFIG OK`, exit 0.

- [ ] **Step 3: Commit**

```bash
git add examples/sbi-benchmarks/two_moons/maf_NLE/config/config_maf_nle.yaml
git commit -m "feat(maf_NLE): add two-moons NLE config"
```

---

### Task 2: The NLE script (helpers + main)

**Files:**
- Create: `examples/sbi-benchmarks/two_moons/maf_NLE/train_maf_nle.py`

**Interfaces:**
- Consumes: `config_maf_nle.yaml` (Task 1); `make_maf`, `Affine`, `RQSpline` (`gensbi.normalizing_flows`); `ConditionalFlowPipeline` (`gensbi.recipes`); `NLEPosterior` (`gensbi.inference`); `plot_marginals` (`gensbi.utils.plotting`); `TaskDataset` (`sbibm_jax.data`); `get_task` (`sbibm_jax.tasks`).
- Produces (module-level callables used by `main` and by the smokes):
  - `build_transformer(model_cfg) -> Affine | RQSpline`
  - `build_flow(rngs, dim_obs, dim_cond, model_cfg) -> Flow`
  - `build_training_config(config, checkpoint_dir) -> dict`
  - `swap_obs_cond(batch) -> (x, theta)` (input `(theta, x)`)
  - `load_x_stats(task, dim_obs) -> (mean, std)` each shape `(dim_obs,)`
  - `apply_standardization(pipeline, mean, std) -> None`
  - `build_prior(task_name, validate_args=True) -> numpyro Distribution`
  - `build_posterior(pipeline, prior, mcmc_cfg, use_ema=True) -> NLEPosterior`
  - `parse_args()`, `main()`

- [ ] **Step 1: Write the full script**

Write `examples/sbi-benchmarks/two_moons/maf_NLE/train_maf_nle.py`:

```python
"""Two-moons NLE example: train a conditional MAF likelihood q(x|theta) and
sample the posterior with NUTS (MCMC).

Run (on a GPU node with HF access, via the gensbi conda env):
    python train_maf_nle.py --config config/config_maf_nle.yaml

NLE convention (mirror of the NPE script): obs = x, cond = theta, so the flow
models q(x | theta). The posterior is recovered at inference time by combining the
learned likelihood with the task prior under NUTS via gensbi NLEPosterior. Helpers
are module-level and import-safe so they can be smoke-tested on CPU.
"""

import os

# Import-safe (module import / CPU smoke checks): default to CPU. When run as the
# main training script we leave JAX_PLATFORMS unset so it can use a GPU.
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
from gensbi.inference import NLEPosterior
from gensbi.utils.plotting import plot_marginals
from sbibm_jax.data import TaskDataset
from sbibm_jax.tasks import get_task


def build_transformer(model_cfg):
    """Return the elementwise transformer named in the model config."""
    name = str(model_cfg.get("transformer", "rqspline")).lower()
    if name == "affine":
        return Affine()
    if name == "rqspline":
        return RQSpline(num_bins=int(model_cfg.get("num_bins", 8)))
    raise ValueError(f"unknown transformer {name!r} (expected 'affine' or 'rqspline')")


def build_flow(rngs, dim_obs, dim_cond, model_cfg):
    """Build the MAF Flow from the model config section.

    For NLE, dim_obs == task.dim_x (autoregressive target) and dim_cond == task.dim_theta.
    """
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


def build_training_config(config, checkpoint_dir):
    """Pipeline defaults overlaid with YAML optimizer+training, with ckpt dir set.

    ConditionalFlowPipeline reads training_config keys eagerly in __init__, so it must
    be complete. Extra keys are harmless — the pipeline only reads the keys it knows.
    """
    tc = ConditionalFlowPipeline.get_default_training_config()
    tc.update(config.get("optimizer", {}))
    tc.update(config.get("training", {}))
    tc["checkpoint_dir"] = checkpoint_dir
    return tc


def swap_obs_cond(batch):
    """Map a conditional batch (theta, x) -> (x, theta) for NLE.

    TaskDataset's conditional collate yields (theta, x); the pipeline reads
    obs, cond = batch. NLE models q(x | theta), so obs must be x and cond theta.
    """
    theta, x = batch
    return x, theta


def load_x_stats(task, dim_obs):
    """Precomputed x mean/std as shape (dim_obs,), read straight off the TaskDataset.

    NLE analog of the NPE script's load_obs_stats: x is the autoregressive target
    ("obs") for NLE, so its stats drive the in-flow Standardize bijection. The stats
    ship as shape (1, dim_obs) regardless of the loader's normalize flag.
    """
    if task.x_mean is None or task.x_std is None:
        raise ValueError(f"task {getattr(task, 'name', task)!r} has no x stats")
    mean = jnp.asarray(task.x_mean).reshape(dim_obs)
    std = jnp.asarray(task.x_std).reshape(dim_obs)
    return mean, std


def apply_standardization(pipeline, mean, std):
    """Set the obs (=x) Standardize buffers on both model and EMA from precomputed stats.

    EMA averages only Params, so its non-Param Standardize buffer must be set too.
    Marks the pipeline standardized to suppress the train-time 'did you fit?' warning.
    """
    pipeline.model.set_standardization(mean, std)
    pipeline.ema_model.set_standardization(mean, std)
    pipeline._standardized = True


def build_prior(task_name, validate_args=True):
    """Return the task's numpyro prior over theta, with out-of-support log_prob = -inf.

    get_task(...).get_prior_dist() ships with validate_args=False, so its log_prob
    returns the in-support constant *everywhere* and does NOT bound the NLE potential —
    NUTS then wanders outside the prior support and the posterior is wrong. Re-enabling
    validation makes log_prob = -inf outside the support, confining NUTS to the box.
    """
    prior = get_task(task_name).get_prior_dist()
    prior._validate_args = validate_args
    if hasattr(prior, "base_dist"):          # Independent wraps a base distribution
        prior.base_dist._validate_args = validate_args
    return prior


def build_posterior(pipeline, prior, mcmc_cfg, use_ema=True):
    """Wrap the trained likelihood flow + validated prior in an NLE NUTS posterior."""
    flow = pipeline.ema_model if use_ema else pipeline.model
    return NLEPosterior(
        flow,
        prior,
        num_warmup=int(mcmc_cfg.get("num_warmup", 1000)),
        num_samples=int(mcmc_cfg.get("num_samples", 50000)),
        num_chains=int(mcmc_cfg.get("num_chains", 1)),
    )


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(here, "config", "config_maf_nle.yaml")
    parser = argparse.ArgumentParser(description="Two-moons NLE (MAF) training/eval")
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
    mcmc_cfg = config["mcmc"]
    eval_cfg = config["evaluation"]

    # --- task / data (raw loader: normalize=False; x is standardized in-flow) ---
    task = TaskDataset(task_name, kind="conditional", normalize=False,
                       use_prefetching=True, max_workers=2)
    # NLE: obs = x, cond = theta (mirror of NPE). Swap the (theta, x) batches.
    dim_obs, dim_cond = task.dim_x, task.dim_theta
    train_ds = task.get_train_loader(
        int(train_cfg["batch_size"]),
        num_samples=int(train_cfg["nsamples"]),
    ).map(swap_obs_cond)
    val_ds = task.get_val_loader(512).map(swap_obs_cond)

    # --- flow + pipeline ---
    flow = build_flow(nnx.Rngs(0), dim_obs, dim_cond, model_cfg)
    training_config = build_training_config(config, checkpoint_dir)
    pipeline = ConditionalFlowPipeline(flow, train_ds, val_ds, dim_obs, dim_cond,
                                       training_config=training_config)

    # --- standardize x from dataset stats (no fitting; loader stays raw) ---
    mean, std = load_x_stats(task, dim_obs)
    apply_standardization(pipeline, mean, std)

    # --- train / restore ---
    if train_cfg.get("restore_model", False):
        print("Restoring model from checkpoint...")
        pipeline.restore_model()
    if train_cfg.get("train_model", True):
        print("Starting training...")
        pipeline.train(nnx.Rngs(0))
        print("Training complete.")

    # --- NLE posterior via NUTS for one observation ---
    idx = int(eval_cfg["observation_idx"])
    obs, _ = task.get_reference(idx)
    true_param = np.asarray(task.get_true_parameters(idx)).reshape(-1)

    prior = build_prior(task_name)
    posterior = build_posterior(pipeline, prior, mcmc_cfg, use_ema=True)
    print(f"Sampling posterior with NUTS "
          f"(warmup={mcmc_cfg['num_warmup']}, samples={mcmc_cfg['num_samples']})...")
    samples = posterior.sample(jax.random.PRNGKey(0), obs)   # (n, dim_theta, 1)

    # --- contour plot of the posterior samples with the true value marked ---
    plot_marginals(np.asarray(samples[..., 0]), plot_levels=False, backend="seaborn",
                   gridsize=50, true_param=true_param)
    out_path = os.path.join(img_dir, f"posterior_marginals_obs{idx}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close("all")
    print(f"Saved posterior marginals to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-check the pure / flow helpers (heredoc OK — no loader)**

Run:
```bash
JAX_PLATFORMS=cpu /lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python - <<'PY'
import importlib.util, pathlib, types
import numpy as np, jax.numpy as jnp
from flax import nnx
from gensbi.normalizing_flows import Affine, RQSpline
from gensbi.recipes import ConditionalFlowPipeline

p = "examples/sbi-benchmarks/two_moons/maf_NLE/train_maf_nle.py"
spec = importlib.util.spec_from_file_location("train_maf_nle", p)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

# build_transformer
assert isinstance(m.build_transformer({"transformer": "affine"}), Affine)
assert isinstance(m.build_transformer({"transformer": "rqspline", "num_bins": 6}), RQSpline)

# build_flow: NLE dims obs=x(2), cond=theta(2); log_prob finite
flow = m.build_flow(nnx.Rngs(0), dim_obs=2, dim_cond=2,
                    model_cfg={"n_layers": 2, "transformer": "affine"})
lp = flow.log_prob(jnp.zeros((4, 2)), jnp.zeros((4, 2)))
assert lp.shape == (4,) and bool(jnp.all(jnp.isfinite(lp)))

# swap_obs_cond: (theta, x) -> (x, theta)
theta = jnp.zeros((3, 2, 1)); x = jnp.ones((3, 2, 1))
o, c = m.swap_obs_cond((theta, x))
assert bool(jnp.all(o == 1)) and bool(jnp.all(c == 0))

# load_x_stats: reshapes (1, dim) stats; raises without stats
task = types.SimpleNamespace(x_mean=np.array([[1.0, -2.0]]),
                             x_std=np.array([[3.0, 4.0]]), name="two_moons")
mean, std = m.load_x_stats(task, dim_obs=2)
assert mean.shape == (2,) and std.shape == (2,)
assert bool(jnp.allclose(mean, jnp.array([1.0, -2.0])))
try:
    m.load_x_stats(types.SimpleNamespace(x_mean=None, x_std=None, name="t"), 2)
    raise AssertionError("expected ValueError")
except ValueError:
    pass

# build_training_config: all default keys present + overrides applied
tc = m.build_training_config(
    {"optimizer": {"max_lr": 4e-4}, "training": {"nsteps": 123, "experiment_id": 7}},
    checkpoint_dir="/tmp/ckpt_nle")
for k in ConditionalFlowPipeline.get_default_training_config():
    assert k in tc, f"missing {k}"
assert tc["nsteps"] == 123 and tc["max_lr"] == 4e-4 and tc["checkpoint_dir"] == "/tmp/ckpt_nle"
print("HELPERS OK")
PY
```
Expected: prints `HELPERS OK`, exit 0.

- [ ] **Step 3: Smoke-check the validated prior (heredoc OK)**

Run:
```bash
JAX_PLATFORMS=cpu /lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python - <<'PY'
import importlib.util, warnings, jax.numpy as jnp
spec = importlib.util.spec_from_file_location(
    "train_maf_nle", "examples/sbi-benchmarks/two_moons/maf_NLE/train_maf_nle.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
prior = m.build_prior("two_moons")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    inside = float(prior.log_prob(jnp.zeros(2)))
    outside = float(prior.log_prob(jnp.array([2.0, 2.0])))
assert jnp.isfinite(jnp.array(inside)), inside
assert outside == float("-inf"), outside     # the whole point: bounded support
print("PRIOR OK  inside=", inside, " outside=", outside)
PY
```
Expected: prints `PRIOR OK  inside= -1.386...  outside= -inf`, exit 0.

- [ ] **Step 4: Smoke-check the inference path end-to-end on a tiny flow (heredoc OK — no loader)**

Run:
```bash
JAX_PLATFORMS=cpu /lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python - <<'PY'
import importlib.util, numpy as np, jax, jax.numpy as jnp
from flax import nnx
spec = importlib.util.spec_from_file_location(
    "train_maf_nle", "examples/sbi-benchmarks/two_moons/maf_NLE/train_maf_nle.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

# tiny NLE flow + identity standardization, wrapped like the real pipeline.ema_model
flow = m.build_flow(nnx.Rngs(0), dim_obs=2, dim_cond=2,
                    model_cfg={"n_layers": 3, "transformer": "affine", "zero_init": False})
flow.set_standardization(jnp.zeros(2), jnp.ones(2))
import types
fake_pipe = types.SimpleNamespace(model=flow, ema_model=flow)
prior = m.build_prior("two_moons")
post = m.build_posterior(fake_pipe, prior, {"num_warmup": 100, "num_samples": 300}, use_ema=True)
s = np.asarray(post.sample(jax.random.PRNGKey(0), jnp.array([0.1, -0.2]))[..., 0])
assert s.shape == (300, 2)
assert bool(np.all(np.isfinite(s))), "non-finite samples"
inbox = float(np.mean(np.all(np.abs(s) <= 1.0, axis=1)))
assert inbox == 1.0, f"samples escaped the prior box: in-box frac={inbox}"
print("INFERENCE OK  in-box=", inbox, " shape", s.shape)
PY
```
Expected: prints `INFERENCE OK  in-box= 1.0  shape (300, 2)`, exit 0. (This is the check that catches the bounded-prior bug — `in-box` must be `1.0`.)

- [ ] **Step 5: Commit**

```bash
git add examples/sbi-benchmarks/two_moons/maf_NLE/train_maf_nle.py
git commit -m "feat(maf_NLE): add two-moons NLE training + NUTS posterior script"
```

---

### Task 3: End-to-end integration run (HF data) + production launch

**Files:**
- Run-only (no source edits). Produces `examples/sbi-benchmarks/two_moons/maf_NLE/imgs/posterior_marginals_obs8.png` and `.../checkpoints/`.
- Uses a throwaway reduced config in the scratchpad (not committed).

**Interfaces:**
- Consumes: `train_maf_nle.py` (Task 2), `config_maf_nle.yaml` (Task 1). Exercises the swapped loader, standardization, the training loop, NUTS, and plotting together (these need the two-moons HF dataset — requires network access on the node).

- [ ] **Step 1: Write a reduced config for a fast wiring check**

Write `/tmp/aamerio/claude-6356/-lustre-ific-uv-es-ml-ific088-github-GenSBI-examples/2694ae16-0028-46eb-b5cb-55b49415e32c/scratchpad/config_maf_nle_smoke.yaml` (a copy of the real config with these overrides — tiny so it finishes in ~1-2 min on CPU; quality is irrelevant here, only that the pipeline runs and a figure is written):

```yaml
task_name: two_moons
strategy: {method: nle, model: maf}
model:
  n_layers: 4
  transformer: affine
  nn_width: 32
  nn_depth: 2
  permutation: reverse
  standardize: true
  zero_init: true
optimizer: {warmup_steps: 50, decay_transition: 0.80, max_lr: 2.0e-4, min_lr: 2.0e-6}
training:
  batch_size: 256
  nsamples: 2000
  nsteps: 200
  ema_decay: 0.99
  val_every: 50
  early_stopping: false
  experiment_id: 99
  restore_model: false
  train_model: true
mcmc: {num_warmup: 50, num_samples: 300, num_chains: 1}
evaluation: {observation_idx: 8}
```

- [ ] **Step 2: Run the script end-to-end with the reduced config (real file, CPU)**

Run (from the repo root; the script must run as a file so `mp_prefetch` spawn-workers can re-import it):
```bash
cd examples/sbi-benchmarks/two_moons/maf_NLE && \
JAX_PLATFORMS=cpu /lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python train_maf_nle.py \
  --config /tmp/aamerio/claude-6356/-lustre-ific-uv-es-ml-ific088-github-GenSBI-examples/2694ae16-0028-46eb-b5cb-55b49415e32c/scratchpad/config_maf_nle_smoke.yaml
```
Expected: prints `Starting training...`, `Training complete.`, `Sampling posterior with NUTS ...`, and `Saved posterior marginals to .../imgs/posterior_marginals_obs8.png`; exit 0. (If it fails with a network/HF error, the node lacks dataset access — rerun on a node with HF access; the wiring is already covered by Task 2's smokes.)

- [ ] **Step 3: Verify the figure was produced**

Run:
```bash
ls -l examples/sbi-benchmarks/two_moons/maf_NLE/imgs/posterior_marginals_obs8.png
```
Expected: the PNG exists and is non-empty (> 1 KB).

- [ ] **Step 4: Clean the throwaway smoke checkpoints (keep the dir tidy before the real run)**

Run:
```bash
rm -rf examples/sbi-benchmarks/two_moons/maf_NLE/checkpoints/99 \
       examples/sbi-benchmarks/two_moons/maf_NLE/checkpoints/ema/99
```
Expected: exit 0 (no error if the paths are absent).

- [ ] **Step 5: Launch the full production run (real config, GPU)**

This is the actual deliverable (50k training steps + 50k NUTS samples) and is long-running — launch it on a GPU node, either directly or via the existing condor submit flow (`sub/train_model.sub` + `sub/train_model.sh`, pointing `workdir` at `.../two_moons/maf_NLE` and `script_path` at `train_maf_nle.py`). Direct form on a GPU node:
```bash
cd examples/sbi-benchmarks/two_moons/maf_NLE && \
/lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python train_maf_nle.py \
  --config config/config_maf_nle.yaml
```
Expected: training runs to completion, then `Saved posterior marginals to .../imgs/posterior_marginals_obs8.png` with the full-quality posterior.

- [ ] **Step 6: Commit the produced figure (optional artifact)**

```bash
git add examples/sbi-benchmarks/two_moons/maf_NLE/imgs/posterior_marginals_obs8.png
git commit -m "feat(maf_NLE): add two-moons NLE posterior figure (obs 8)"
```

---

## Self-Review

**Spec coverage:**
- New `maf_NLE/` dir + `train_maf_nle.py` + `config/config_maf_nle.yaml` → Tasks 1, 2.
- obs/cond swap (`obs=x, cond=theta`) via `swap_obs_cond` loader-map → Task 2 (`main`), verified Step 2/Task 3.
- Standardize x from `task.x_mean/x_std` → `load_x_stats` + `apply_standardization`, Task 2.
- Prior from `get_task(...).get_prior_dist()` **validated** → `build_prior`, Task 2 Step 3.
- Single-chain NUTS 1000/50000 via `NLEPosterior` → `build_posterior` + config `mcmc`, Tasks 1/2.
- Eval = `plot_marginals` contour + true value (style of `scripts/train_sbi_model.py`) → `main`, Task 2.
- Bounded-prior correctness fix → `build_prior` validation, verified Task 2 Step 4 (`in-box == 1.0`).
- Run in the gensbi conda env; no pytest suite; no gensbi/sbibm-jax edits → Global Constraints, all commands.

**Placeholder scan:** No TBD/TODO; every code and command step is complete and runnable.

**Type consistency:** `swap_obs_cond` returns `(x, theta)` consistently; `build_prior`/`build_posterior`/`load_x_stats`/`apply_standardization` signatures match between the Interfaces block, the script, and the smokes; `dim_obs=task.dim_x`, `dim_cond=task.dim_theta` used consistently.
