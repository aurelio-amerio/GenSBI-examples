# TarFlow field-level example on gaussian_random_field (32×32)

**Date:** 2026-07-11
**Status:** approved design, pending implementation plan

## Goal

A field-level example where a TarFlow autoregressive normalizing flow models
p(field | theta) for the `gaussian_random_field` task (32×32): the condition
is the unstructured 2-vector theta = (log_std, alpha), the generated output is
the GRF realization itself. Counterpart to
`examples/sbi-benchmarks/gaussian_random_field/train-grf.py` (FieldDiT /
PixelDiT flow matching), producing the same diagnostics so results are
directly comparable across model families.

This is the first end-to-end exercise of TarFlow's image route with the new
RoPE support (GenSBI `use_rope`/`rope_theta`, `VisionRotaryEmbedding`,
KV-cached sampling — merged 2026-07-11, tested in
`tests/models/tarflow/test_structured_integration.py::test_field_nle_train_smoke_and_mclmc`).

## Decisions

- **Resolution:** 32×32 (`gaussian_random_field`) first; port to 256×256 only
  after this works.
- **Positional encoding:** RoPE only (`use_rope: true`). No learned-pos_embed
  baseline config — RoPE is the right choice for image generation here.
- **Theta conditioning:** prefix strategy only (`cond: vector`): each theta
  coordinate becomes one prefix token behind the prefix-LM mask. Prefix tokens
  sit at the identity rotation, composing cleanly with RoPE. The `bias` seam
  remains available in the model but ships no config.
- **Script structure:** standalone directory, helpers copied not imported
  (importing `train-grf.py` would execute its `JAX_PLATFORMS=cpu` side
  effect). Duplication between examples is the repo norm.

## Layout

```
examples/sbi-benchmarks/gaussian_random_field/tarflow/
  train_tarflow_grf.py
  config/config_1.yaml
  imgs/          (runtime)
  checkpoints/   (runtime)
```

Script skeleton follows `slcp/tarflow_NPE/train_tarflow_npe.py`: import-safe
(`JAX_PLATFORMS=cpu` when imported, CUDA when `__main__`), yaml-driven, build
model → pipeline → train/restore → sample → plot. The four GRF diagnostics
helpers (`radial_power_spectrum`, `theory_power_spectrum`, `plot_field_grid`,
`plot_power_spectra`) are copied verbatim from the sibling `train-grf.py`.

## Model

Built from the yaml `model:` section mapped 1:1 onto `TarFlowParams`:

```yaml
model:
  modeled: image        # ImageTokenizer: (B, 32, 32, 1) -> 64 tokens of F=16
  img_size: 32
  patch_size: 4
  img_channels: 1
  use_rope: true        # 2D VisionRotaryEmbedding; pos_embed dropped
  rope_theta: 10000
  cond: vector          # theta -> 2 prefix tokens (prefix-LM mask)
  cond_dim: 2
  head_dim: 32          # % 4 == 0 required by rope; >= 32 recommended
  num_heads: 8          # channels = 256
  num_blocks: 8
  layers_per_block: 2
  permutation: flip
  standardize: true     # buffers stay identity (data normalized upstream)
```

~13M parameters, comparable to the FM baselines.

## Data

Mirrors `train-grf.py`:

- Train: `OnlineTaskDataset("gaussian_random_field", normalize=True)` — fresh
  prior + simulator draws per batch; `training.online: false` falls back to
  the offline HF train split. `training.max_workers` forwarded as in the FM
  script.
- Val + test rows: offline `TaskDataset` (always built).
- Batch map: loader yields `(theta, x)` with `x` already `(B, 32, 32, 1)`;
  map to `(x, theta[..., None])` so cond is channel-carrying `(B, 2, 1)`.
  Module-level function (pickling safety), like `swap_obs_cond`.
- Pipeline: `ConditionalFlowPipeline(flow, train, val, dim_obs=1024,
  dim_cond=2, structured_obs=True, training_config=...)`.
- Standardization: dataset `normalize=True` handles it; flow buffers stay at
  identity and `pipeline._standardized = True` is set with a comment (same
  pattern as the SLCP tarflow script).

## Training

`ConditionalFlowPipeline.get_default_training_config()` overlaid with the
yaml `training:` section (nsteps, max_lr/min_lr, warmup_steps, ema_decay,
decay_transition, val_every, early_stopping, multistep, experiment_id) — the
key-passthrough idiom of the SLCP script. `train_model` / `restore_model`
flags as in the other examples. Loss curves (train smoothed + val, log y)
written to `imgs/`.

## Sampling & evaluation

Same outputs as `train-grf.py`, for `sampling.num_thetas` test thetas from
`offline_task.df_test`:

1. Truth-vs-samples field grid (`sampling.nsamples_grid` columns).
2. Radial P(k) overlays: model samples (mean ± σ) vs the mean P(k) of
   `sampling.nsim_pk` fresh simulator realizations (± σ band) and the
   analytic power law.
3. Both `_ema` and `_raw` weights, identical PRNG (EMA-degeneration
   cross-check inherited from the FM script).

Sampling: test thetas come raw from `df_test` and are normalized with
`task.normalize_theta(thetas[..., None])` (as in the FM script) before
`pipeline.sample(key, theta_norm[i:i+1], nsamples,
use_ema=...)` → native `(nsamples, 32, 32, 1)`; `task.unnormalize_x` before
plotting. Uses the KV-cached sampler automatically (64 token steps).

Flow-specific addition: exact held-out likelihood — mean
`pipeline.log_prob` over the offline test fields, reported in bits/dim,
printed for both raw and EMA weights. A scalar quality gate the FM models
cannot provide.

## Error handling & verification

- Config validation is delegated to `TarFlowParams.__post_init__` (raises on
  rope/head_dim/img_size violations) — no duplicate checks in the script.
- CPU smoke test before any cluster run: import the module, build a tiny
  variant (small head_dim/num_blocks, few steps), run a couple of training
  steps and a 2-sample draw.
- Optional: `sub/train_model_tarflow_grf.sub` condor file mirroring the
  existing ones.

## Out of scope

- 256×256 port (follow-up once 32×32 validates).
- `cond: bias` config, no-rope baseline config.
- Any GenSBI library changes (RoPE/KV-cache already merged).
