# TarFlow GRF-256 Field-Level Example — Design

**Date:** 2026-07-11
**Status:** Approved (user, 2026-07-11)
**Predecessor:** `2026-07-11-tarflow-grf-design.md` (the 32×32 example, implemented as
`examples/sbi-benchmarks/gaussian_random_field/tarflow/`, commits c08dc3db..b02d06d4)

## Goal

A field-level example training a conditional TarFlow p(field | theta) on the 256×256
`gaussian_random_field_256` task — the harder, higher-resolution sibling of the 32×32
example — with the same diagnostics (loss curves, truth-vs-samples field grid, radial
P(k) overlays with simulator-mean + analytic references, exact held-out bits/dim NLL),
trained for 50k steps on an A100 via HTCondor.

## Decisions (user-approved)

1. **Architecture:** patch 16 → T = 256 tokens of F = 256; head_dim 64 × 8 heads
   (channels 512); 8 blocks × 2 layers/block; `use_rope: true` (head_dim % 4 == 0,
   ≥ 32 as recommended for images); flip permutation; `cond="vector"` +
   `modeled="image"` hardcoded in `build_flow` as in the 32 example. 53.6M params
   (measured). The 256-token budget matches both FM models in the sibling
   `gaussian_random_field_256` dir (FieldDiT: 32×32 grid / patch 2; PixelDiT: 256
   patch tokens).
2. **Layout:** new standalone dir
   `examples/sbi-benchmarks/gaussian_random_field_256/tarflow/` with
   `train_tarflow_grf256.py` — a copy of the 32 script with task name, config and
   naming changes. Repo convention: every benchmark dir is standalone; FM dirs
   already duplicate the helpers. No shared module, no import of sibling scripts
   (importing them forces `JAX_PLATFORMS=cpu`).
3. **Training (config_1.yaml):** 50k steps (user requirement — harder problem gets
   more budget); batch_size 64 = val_batch_size (FM PixelDiT precedent at this
   resolution; halve on OOM); online simulation, max_workers 4; max_lr 1.0e-4,
   min_lr 1.0e-6, warmup 500, decay_transition 0.80; ema_decay 0.999; val_every 100;
   early_stopping false; multistep 1; experiment_id 1; seed 0.
4. **Datasets stay float32** (deviation from the FM 256 script's `dtype=jnp.bfloat16`):
   the exact-likelihood NLL is the example's headline metric and wants float32; at
   batch 64 the data memory is trivial. Do not pass a dtype argument.

## Script structure (mirrors the 32 example exactly)

`train_tarflow_grf256.py`, yaml-driven, import-safe (`JAX_PLATFORMS=cpu` when
imported, cuda default under `__main__`):

- `build_flow(rngs, model_cfg) -> TarFlow` — hardcoded `modeled="image"`,
  `cond="vector"`, `cond_channels=1`; yaml sizes the architecture only.
- `build_training_config(config, checkpoint_dir)` — pipeline defaults overlaid with
  yaml `optimizer` + `training` sections.
- `to_obs_cond(batch)` — returns `(x, theta)` unchanged: the sbibm_jax collate
  already tokenizes theta to (B, 2, 1) (lesson learned in the 32 example; the
  df_test path bypasses the collate and needs the manual `[..., None]`).
- `heldout_bits_per_dim(flow, fields_norm, theta_norm, batch_size=64)` — exact mean
  NLL in bits/dim, float64 accumulation, `_NLL_BATCH = 64`.
- Five diagnostics helpers copied VERBATIM from
  `examples/sbi-benchmarks/gaussian_random_field_256/train-grf.py`:
  `radial_power_spectrum`, `plot_losses`, `plot_field_grid`,
  `theory_power_spectrum`, `plot_power_spectra`. (These are byte-identical to the
  copies in the 32 tarflow script; the copy source is the FM script in this same
  dir. Same no-import rationale.)
- `main(config_path)` — identical flow to the 32 example with
  `OnlineTaskDataset("gaussian_random_field_256", normalize=True)` /
  `TaskDataset("gaussian_random_field_256", ...)`: online train loader (offline
  fallback via `training.online: false`), val loader from the offline split,
  parameter-count print, train + loss plot, held-out NLL for ema and raw weights,
  fresh simulator realizations per theta for the P(k) reference (`nsim_pk`),
  KV-cached sampling for ema and raw with identical PRNG, field grid + P(k) plots
  per weight set. Plot filenames `grf256_*_conf{experiment}[_tag].png`.

## Configs

- `config/config_1.yaml` (production): model section per Decision 1; optimizer +
  training per Decision 3; sampling: num_thetas 3, nsamples 16, nsamples_grid 3,
  nsim_pk 64, nll_num_test 256.
- `config/config_smoke.yaml` (CPU smoke, not a science run): full 256×256 field,
  patch_size 32 → 64 tokens, head_dim 16 × 2 heads, 2 blocks × 1 layer; nsteps 20,
  batch 8, warmup 5, val_every 10, experiment_id 99, max_workers null; sampling:
  num_thetas 1, nsamples 2, nsamples_grid 2, nsim_pk 4, nll_num_test 8. Requires
  the gaussian_random_field_256 HF cache (present — the FM example trained here).

## Tests

`tests/test_train_tarflow_grf256.py` — mirrors the 9 tests of
`tests/test_train_tarflow_grf.py` (loader via `importlib`, tiny 8×8 rope model for
the unit tests — size-independent), with config assertions updated: img_size 256,
img_size % patch_size == 0, head_dim % 4 == 0, cond_dim 2, use_rope true; smoke
config nsteps <= 50, nsamples <= 4. The `to_obs_cond` test uses the collate-shaped
fake input (theta (4, 2, 1)) and the corrected name/behavior from the 32 example
(pass-through, no extra axis). All tests run on CPU.

## Job + operations

- `sub/train_model_tarflow_grf256.sub` — mirrors `sub/train_model_tarflow_grf.sub`:
  experiment_name tarflow_grf256, version 1a, workdir the new tarflow dir (via the
  `/lhome/.../data/github` alias), 1 A100 GPU, 8 CPUs, 64 GB.
- After implementation + CPU smoke pass: submit with `condor_submit`, monitor
  periodically, and send the conf1 EMA figures (fields, P(k), loss) to the user on
  completion. Expected runtime ~5–9 h (16× tokens, 2× width, 5× steps vs the
  32-case's 21 min; step-rate estimate to be confirmed from the first progress
  lines).

## Error handling / risks

- OOM at batch 64: halve batch_size (note in config comment). Forward-only NLL at
  batch 64 is lighter than a training step and should fit.
- If the smoke run fails inside pipeline/dataset code: fix the script's usage,
  never the libraries (GenSBI, sbibm_jax) — same rule as the 32 plan.
- KV-cached AR sampling at T=256 is 4× the 32-case's sequential steps; with
  nsamples 16 × 3 thetas × 2 weight sets it remains a small fraction of runtime.

## Out of scope

- No GenSBI/sbibm_jax library changes.
- No bias-conditioner or no-rope config variants (same rejection as the 32 spec).
- No MCMC/posterior inference — this is the p(field | theta) NLE example.
