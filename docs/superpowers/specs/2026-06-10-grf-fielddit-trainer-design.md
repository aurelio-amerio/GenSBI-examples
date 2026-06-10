# GRF-256 FieldDiT trainer — design

**Date:** 2026-06-10
**Status:** approved (brainstorm with Aurelio)
**Scope:** `examples/sbi-benchmarks/gaussian_random_field/` in GenSBI-examples

## Goal

A cluster-submittable script that trains the experimental FieldDiT model
(GenSBI branch `FieldDiT`) on the `gaussian_random_field_256` task via
`FieldConditionalPipeline`, then draws posterior samples and saves sanity
plots. Success for this phase is judged visually: fields that look like GRFs
with the right power law after ~10k steps. TARP / L-C2ST diagnostics are
explicitly **out of scope** until the visual check passes.

## Task

`gaussian_random_field_256` (sbibm-jax): theta = (log_std, alpha) controls a
power-law power spectrum; x is a 256x256 GRF realization. The **field is the
inference target**: the model learns p(field | theta), so for FieldDiT

- obs  = field, `(B, 256, 256, 1)`
- cond = theta, `(B, 2, 1)`

## Files

```
examples/sbi-benchmarks/gaussian_random_field/
  train-grf.py          # train + sample + plot, main() guarded by __name__
  config/config_1.yaml  # fielddit + training sections
  checkpoints/          # orbax checkpoints (exists)
  imgs/                 # output plots (created)
```

## Data

- `TaskDataset("gaussian_random_field_256", normalize=True,
  dtype=jnp.bfloat16, use_prefetching=True, max_workers=4)` — targets the
  **sbibm-jax repo HEAD** interface (`x_kind`/`x_shape` metadata schema, stats
  in hub metadata). The hub test repo (`aurelio-amerio/SBI-benchmarks-test`)
  is being republished with this schema; the conda `gensbi` env must have the
  updated sbibm_jax installed (the old site-packages version reads the old
  `data_kind` schema and will KeyError on the new metadata, and vice versa).
- Loaders yield `(theta, x)`; the pipeline wants `(obs, cond)`. Append a
  `.map` to the returned grain dataset:

  ```python
  def swap_obs_cond(batch):  # module-level: survives pickling into workers
      theta, x = batch
      return x, theta

  train_loader = task.get_train_loader(batch_size).map(swap_obs_cond)
  ```

  Note: `get_train_loader` applies `mp_prefetch` last, so this map runs in
  the main process after the workers. Fine for a free tuple swap; if the map
  ever grows real per-batch work, it must move before the prefetch stage.
- Train batch 128, val batch 128 (yaml knobs; ~40 GB GPU — reduce if OOM).
- 100k train / 10k val / 10k test rows on the hub; 10k steps x 128 ≈ 13
  epochs.

## Model

Verified smoke-test config (`test_realistic_256_config_smoke`), all values in
yaml under `fielddit:` and passed as `FieldDiTParams(**cfg)` plus
`rngs=nnx.Rngs(seed)`:

```yaml
fielddit:
  in_channels: 1
  field_shape: [256, 256]
  encoder_widths: [64, 128, 256, 256]   # D=3 -> 32x32 meeting grid
  patch_size: 2                          # -> 16x16 = 256 tokens
  cond_dim: 2
  cond_in_channels: 1
  # num_heads 12, axes_dim [16,24,24] -> hidden 768; depth 2 + 2 (defaults)
  param_dtype: bfloat16                  # mapped to jnp dtype in the script
```

## Pipeline + training

```python
FieldConditionalPipeline(
    model, train_loader, val_loader,
    field_shape=(256, 256), dim_cond=2,
    method=FlowMatchingMethod(), ch_obs=1, ch_cond=1,
    training_config=training_config,
)
```

`training:` yaml section (defaults for this run): `batch_size: 128`,
`nsteps: 10_000`, `max_lr: 1.0e-4`, `val_every: 100`,
`early_stopping: false` (10k steps IS the budget; don't let the ratio-based
early stop kill a noisy sanity run), `multistep: 1`, `experiment_id: 1`,
`train_model: true`, `restore_model: false`. `checkpoint_dir` is set in the
script to `<example dir>/checkpoints`.

Flow: build everything → `pipeline.train(nnx.Rngs(0), save_model=True)` if
`train_model`, `pipeline.restore_model()` if `restore_model` →
`pipeline._wrap_model()` (matches the integration test) → sample + plot.

## Outputs (all `imgs/grf_*_conf{experiment_id}.png`)

1. **Loss curves** — train + val arrays returned by `pipeline.train()`,
   log-scale y.
2. **Field grid** — for 3 test thetas: true test field | 3 posterior samples,
   one row per theta, shared color scale per row, theta values
   (unnormalized) in the row title.
3. **Power spectra** — per theta: mean radial P(k) over 16 samples with a
   min/max (or ±1σ) band vs. the true field's P(k), log-log axes. Computed
   with `jnp.fft.fft2` + radial binning on **unnormalized** fields
   (`task.unnormalize_x`). This is the direct "did it learn the power law"
   check, since theta sets (amplitude, slope).

Sampling: `pipeline.sample(key, x_o=theta_norm[None] -> (1, 2, 1),
nsamples=16)` per theta on the EMA model; samples come back
`(16, 256, 256, 1)`.

## Risks / notes

- **Env:** run under the conda `gensbi` env (gensbi installed editable from
  the local FieldDiT branch checkout). sbibm_jax must be the updated version
  matching the republished hub metadata (see Data section).
- **Memory:** batch 128 through the conv encoder at full 256x256 resolution
  is the main pressure point (~1 GB per stage-1 activation tensor in
  bfloat16). First fallback: halve batch_size in yaml.
- **No sbatch file:** the script is a plain `python train-grf.py`; Aurelio
  submits with his own templates.
- Phase 2 (after visual sign-off): TARP + L-C2ST on flattened fields,
  following the lensing example's diagnostics section.
