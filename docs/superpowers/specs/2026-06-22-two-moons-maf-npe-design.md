# Two-Moons NPE example — RQSpline MAF

**Date:** 2026-06-22
**Status:** Approved (brainstorming)
**Repo:** GenSBI-examples (branch `maf`)
**Depends on:** `gensbi.recipes.ConditionalFlowPipeline`, `gensbi.normalizing_flows.make_maf` / `RQSpline` / `Affine`

## Goal

A small, self-contained, YAML-driven example that trains a conditional
normalizing flow (MAF) on the `two_moons` SBI benchmark via **NPE**
(Neural Posterior Estimation — the flow models `q(theta | x)` by
maximum likelihood) and produces a posterior **contour plot** for a single
observation. No calibration diagnostics. This is the first of a planned pair;
an **NLE** sibling (`*_nle.py`) is expected next, hence the `npe` suffix in the
filenames.

## Why this shape

- `ConditionalFlowPipeline.init_pipeline_from_config` deliberately raises
  `NotImplementedError` (the flow is built directly via `make_maf`), so the
  script constructs the flow itself rather than going through the central
  `scripts/train_sbi_model.py` path.
- YAML config is retained (not hardcoded constants) so experiments are
  trackable and tunable if the first run underperforms.
- The two-moons posterior for a fixed `x` is two narrow crescents (bimodal,
  curved). A plain affine MAF tends to over-smooth it; an RQ-spline transformer
  (RQ-NSF, still masked-autoregressive) captures it cleanly. Default =
  `rqspline`, with `affine` selectable in the config.

## Files

- `examples/sbi-benchmarks/two_moons/maf/config/config_maf_npe.yaml`
- `examples/sbi-benchmarks/two_moons/maf/train_maf_npe.py`
- Outputs (created at run time, under the script's own dir):
  - `checkpoints/` (model + EMA, via the pipeline)
  - `imgs/posterior_contour_obs{idx}.png`

## Config schema (`config_maf_npe.yaml`)

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
  nsamples: 100000           # training subset size; must be < task.max_samples
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
  n_ref_overlay: 2000        # reference posterior samples overlaid on the contour
```

## Script flow (`train_maf_npe.py`)

1. **Parse config.** `argparse --config`, defaulting to the sibling
   `config/config_maf_npe.yaml`. Resolve `EXP_DIR` = the script's own
   directory; `checkpoint_dir = EXP_DIR/checkpoints`, `img_dir = EXP_DIR/imgs`.

2. **Task / data (unstandardized).**
   `task = get_task("two_moons", kind="conditional", normalize_data=False)`.
   This yields **raw** `(obs, cond)` batches of shape `(B, dim, 1)` where
   `obs = theta` (NPE target) and `cond = x` (conditioning). Read
   `dim_obs = task.dim_obs` (= 2) and `dim_cond = task.dim_cond` (= 2).
   `train_ds = task.get_train_dataset(batch_size, nsamples)`,
   `val_ds = task.get_val_dataset(512)`.

3. **Build flow from `model` config.**
   `transformer = RQSpline(num_bins=...)` when `transformer == rqspline`, else
   `Affine()`. Then
   `flow = make_maf(rngs, dim=dim_obs, cond_dim=dim_cond, n_layers=..., transformer=transformer, nn_width=..., nn_depth=..., permutation=..., standardize=..., zero_init=...)`.

4. **Build pipeline with a complete training_config.**
   `ConditionalFlowPipeline` reads `training_config` keys eagerly in
   `__init__` (no merge with defaults), so start from
   `ConditionalFlowPipeline.get_default_training_config()` and overlay the YAML
   `optimizer` + `training` keys, then set `checkpoint_dir`. Construct
   `ConditionalFlowPipeline(flow, train_ds, val_ds, dim_obs, dim_cond, training_config=training_config)`.

5. **Standardization from precomputed stats (no fitting).**
   `stats = _load_precomputed_stats("two_moons")` (module function in
   `gensbi_examples.tasks`; `task.obs_mean` is `None` when
   `normalize_data=False`). Reshape `obs_mean`/`obs_std` from `(1, 2, 1)` to
   `(2,)`. Call `pipeline.model.set_standardization(obs_mean, obs_std)` **and**
   `pipeline.ema_model.set_standardization(obs_mean, obs_std)` (EMA averages
   only `Param`s, so its non-Param `Standardize` buffer must be set explicitly).
   Set `pipeline._standardized = True` to suppress the train-time warning.
   Only `theta` (obs) is standardized inside the flow; `x` (cond) is fed raw to
   the conditioner MLP — acceptable since two-moons `x` is O(1).

6. **Train / restore.** If `restore_model`: `pipeline.restore_model()`.
   If `train_model`: `pipeline.train(nnx.Rngs(0))`. Sampling and `log_prob` use
   the EMA model (`use_ema=True`).

7. **Evaluate + contour plot.**
   `obs, ref_samples = task.get_reference(observation_idx)`;
   `true_param = task.get_true_parameters(observation_idx)`.
   Build a `grid_size × grid_size` grid over theta-space from the
   `ref_samples` extent (+ padding) so the posterior is always framed.
   `logp = pipeline.log_prob(grid_pts, obs, use_ema=True)`, reshape to
   `(G, G)`, `Z = exp(logp)`. `fig, ax = plot_2d_dist_contour(xx, yy, Z, true_param=true_param)`.
   Overlay `n_ref_overlay` reference samples as a light scatter on `ax` for an
   eyeball check (no formal calibration). Save to
   `imgs/posterior_contour_obs{idx}.png`.

## Out of scope

C2ST, TARP, SBC, LC2ST, marginal coverage, model-card generation, and the
NLE / MCMC path (NLE is a planned follow-up, separate script).

## Open risks

- If the affine/spline MAF still under-fits after a first 20k-step run, the
  YAML makes it cheap to bump `n_layers`, `num_bins`, `nsteps`, or `nn_width`
  and re-run under a new `experiment_id`.
- `x` (cond) is fed unstandardized; if a future task has large-magnitude `x`
  this would need revisiting (not a concern for two-moons).
