# Two-Moons MAF NLE + MCMC — Design

**Date:** 2026-06-26
**Status:** Approved (pending spec review)

## Goal

Add a training/eval script that performs **Neural Likelihood Estimation (NLE)** for
the two-moons task using a MAF, then draws posterior samples with **NUTS (MCMC)** via
the existing `gensbi.inference.NLEPosterior`, and plots the resulting posterior
contour with the true parameter marked.

This is the mirror of the existing NPE example
(`examples/sbi-benchmarks/two_moons/maf_NPE/train_maf_npe.py`): NPE trains the
posterior `q(theta | x)`; NLE trains the likelihood `q(x | theta)` and recovers the
posterior at inference time by combining the learned likelihood with the prior under
MCMC.

## Decisions (locked)

- **Location:** new directory `examples/sbi-benchmarks/two_moons/maf_NLE/` parallel to
  `maf_NPE/`, with its own `train_maf_nle.py`, `config/config_maf_nle.yaml`, and
  runtime `checkpoints/` + `imgs/`.
- **Eval output:** a single posterior figure — MCMC samples rendered as a 2D
  contour/corner with the true parameter marked, using
  `gensbi.utils.plotting.plot_marginals` in the style of
  `scripts/train_sbi_model.py`.
- **MCMC:** single chain, `num_warmup=1000`, `num_samples=50000`, `num_chains=1`,
  using `NLEPosterior` **as-is** (no edits to the GenSBI repo).
- **Prior:** obtained from the task via `sbibm_jax.tasks.get_task(task_name).get_prior_dist()`.

## Architecture

The script reuses the same building blocks as the NPE script — `make_maf`,
`ConditionalFlowPipeline`, `TaskDataset` — and differs only in the obs/cond
assignment and the inference step.

### The one structural difference from NPE: obs/cond swap

| | obs (autoregressive target, standardized) | cond |
|---|---|---|
| NPE | `theta` | `x` |
| **NLE** | **`x`** | **`theta`** |

Concretely:

- `dim_obs = task.dim_x`, `dim_cond = task.dim_theta` (both 2 for two-moons, kept
  general).
- `TaskDataset`'s `kind="conditional"` collate yields batches `(theta, x)`. The
  pipeline reads `obs, cond = batch`, so a module-level
  `swap_obs_cond(batch) -> (batch[1], batch[0])` is applied via `loader.map(...)` to
  both the train and val loaders, turning each batch into `(x, theta)`.
- Standardization is applied to the **obs = x** target, using `task.x_mean` /
  `task.x_std` from the `TaskDataset` metadata (the NLE analog of the NPE script's
  `theta` stats; no data pass needed). `cond = theta` stays raw — `theta in [-1, 1]`
  is already well-scaled, matching the NPE convention of leaving the conditioner
  input unstandardized.

This matches the canonical pattern in `GenSBI/scripts/maf_nle_recovery.py`, which
builds data with `x` first, calls `fit_standardization(x)`, then
`NLEPosterior(pipe.ema_model, prior)`.

### Getting the prior (with a correctness fix)

The offline `TaskDataset` does not expose the underlying task object, so the prior is
fetched directly from the task registry:

```python
from sbibm_jax.tasks import get_task
prior = get_task(task_name).get_prior_dist()   # numpyro Independent(Uniform([-1,1]^2), 1)
```

**Critical subtlety found during verification:** the task's prior is constructed with
`validate_args=False`, so its `log_prob` returns the in-support constant
(`-log(2)` per dim) *everywhere* — it never returns `-inf` outside `[-1,1]`. Fed to
`NLEPosterior` as-is, the prior therefore does **not** bound the NUTS potential, and
the chain wanders far outside the prior support (a CPU smoke produced samples from
−41 to +25, only 0.2% inside the box). The posterior would be wrong.

The fix lives entirely in `build_prior`: re-enable validation on the returned prior so
out-of-support `log_prob` becomes `-inf`, which confines the potential (hence NUTS) to
the prior box. This works on the task's own prior object, needs no gensbi edit, and
keeps `NLEPosterior` exactly as the user chose:

```python
def build_prior(task_name, validate_args=True):
    prior = get_task(task_name).get_prior_dist()
    prior._validate_args = validate_args          # out-of-support log_prob -> -inf
    if hasattr(prior, "base_dist"):               # Independent wraps a base dist
        prior.base_dist._validate_args = validate_args
    return prior
```

Verified on CPU: with the toggle, `log_prob([2,2]) = -inf`, `log_prob([0,0]) = -1.386`,
and NUTS via `NLEPosterior` stays finite and **100% inside `[-1,1]`** (matching the
numpyro constrained-model route). The prior is used both for `prior.log_prob` in the
NLE potential and `prior.sample` for the NUTS init.

### Inference + plot

```python
post = NLEPosterior(pipeline.ema_model, prior,
                    num_warmup=mcmc_cfg["num_warmup"],
                    num_samples=mcmc_cfg["num_samples"],
                    num_chains=mcmc_cfg["num_chains"])
obs, _ = task.get_reference(idx)               # raw observation x_o
true_param = task.get_true_parameters(idx)
samples = post.sample(key, obs)                # (n, dim_theta, 1)
plot_marginals(samples[..., 0], plot_levels=False, backend="seaborn",
               gridsize=50, true_param=np.asarray(true_param).reshape(-1))
```

Saved to `imgs/posterior_marginals_obs{idx}.png`.

## Components (module-level, import-safe)

Following the NPE script, helpers are module-level and import-safe (default
`JAX_PLATFORMS=cpu` on import, GPU when run as `__main__`) so they can be unit-tested
on CPU.

Reused unchanged from the NPE script (logic-identical):
- `build_transformer(model_cfg)` — affine | rqspline.
- `build_flow(rngs, dim_obs, dim_cond, model_cfg)` — `make_maf(...)`.
- `build_training_config(config, checkpoint_dir)` — pipeline defaults + YAML overlay.
- `apply_standardization(pipeline, mean, std)` — set Standardize buffers on model + EMA.

New for NLE:
- `swap_obs_cond(batch)` — `(theta, x) -> (x, theta)`.
- `load_x_stats(task, dim_obs)` — read `task.x_mean` / `task.x_std`, reshape to
  `(dim_obs,)`. The NLE analog of the NPE `load_obs_stats`.
- `build_prior(task_name, validate_args=True)` — `get_task(task_name).get_prior_dist()`
  with validation re-enabled so out-of-support `log_prob` is `-inf` (see "Getting the
  prior" — required for correct NLE posteriors).
- `build_posterior(pipeline, prior, mcmc_cfg)` — construct `NLEPosterior`.

`main()` wires it together: load config -> `TaskDataset` -> swapped loaders -> flow +
pipeline -> standardize x -> train/restore -> `NLEPosterior` -> sample one observation
-> `plot_marginals` -> save figure.

## Config (`config/config_maf_nle.yaml`)

Same `model` / `optimizer` / `training` sections as the NPE config, with
`strategy.method: nle` and a new `mcmc` block:

```yaml
task_name: two_moons

strategy:
  method: nle
  model: maf

model:
  n_layers: 8
  transformer: rqspline
  num_bins: 8
  nn_width: 64
  nn_depth: 2
  permutation: reverse
  standardize: true
  zero_init: true

optimizer:
  warmup_steps: 500
  decay_transition: 0.80
  max_lr: 2.0e-4
  min_lr: 2.0e-6

training:
  batch_size: 1024
  nsamples: 100000
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

## Data flow

1. `TaskDataset(task_name, kind="conditional", normalize=False, ...)`.
2. `train/val` loaders -> `.map(swap_obs_cond)` -> batches `(x, theta)` of shape
   `(B, dim, 1)`.
3. `make_maf(dim=dim_x, cond_dim=dim_theta, standardize=True)` -> `ConditionalFlowPipeline`.
4. `apply_standardization(pipeline, *load_x_stats(task, dim_obs))`.
5. `pipeline.train(...)` (or `restore_model()`), max-likelihood `-mean(log q(x|theta))`.
6. `NLEPosterior(pipeline.ema_model, prior).sample(key, x_o)` -> `(n, dim_theta, 1)`.
7. `plot_marginals(samples[..., 0], true_param=...)` -> save.

## Error handling / edge cases

- `load_x_stats` raises if `task.x_mean` / `task.x_std` are absent (metadata without
  stats), mirroring the NPE `load_obs_stats` guard.
- `observation_idx` is 1-indexed (`TaskDataset.get_reference` enforces
  `1 <= idx <= num_observations`).
- The pipeline warns if `train()` is called without standardization having been set;
  `apply_standardization` sets `_standardized = True` to suppress it.

## Bounded-prior handling (resolved)

Every existing NLE test/script uses an **unbounded Gaussian** prior. two-moons has a
**bounded Uniform** prior. Verification (CPU smoke) showed the task's prior is built
`validate_args=False`, so it does not bound the NUTS potential and the chain escapes
`[-1, 1]` — see "Getting the prior". The fix is the `validate_args=True` toggle in
`build_prior`, which confines NUTS to the support (verified 100% in-box, finite). No
reparametrization and no gensbi edit are needed: NUTS on the validated potential
rejects out-of-box proposals (the in-box gradients stay finite), and for two-moons the
posterior mass sits well inside the box. This is the standard NLEPosterior route the
user chose, made correct by bounding the prior.

## Out of scope

- Editing `GenSBI` (`NLEPosterior`, vectorized/parallel chains).
- C2ST / TARP / SBC / LC2ST diagnostics (the NPE script doesn't run them either).
- Multi-observation or batched posterior evaluation.
