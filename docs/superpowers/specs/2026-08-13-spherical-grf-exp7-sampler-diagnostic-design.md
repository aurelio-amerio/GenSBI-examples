# Spherical GRF exp-7 sampler diagnostic ("7diag") — design

**Date:** 2026-08-13
**Status:** approved design, pre-implementation
**Depends on:** exp-7 checkpoint (`spherical_grf_fm_healpix_7`, commit c607a1f5, results 8bb7d991)

## Motivation

exp-7 (8x isometry augmentation + 300k steps) lowered val loss 0.124 → 0.116
but left mean C2ST flat (0.790 → 0.808, within the ±0.05 per-obs run-to-run
scatter). The dispersion defect flipped sign: the logA over-dispersion that
motivated the reserved exp-8 capacity bump is gone (flow std 0.016–0.017 vs
ref 0.015), but the other two parameters are now ~25% **under-dispersed**
across all three evaluated observations (e.g. 0.018–0.020 vs ref 0.026;
0.033–0.038 vs ref 0.044–0.048). TARP was disabled in exp-7, so there is no
calibration curve for this new overconfident regime. The train-val gap
(~0.030 absolute) did not move under 8x augmentation, pointing at per-sample
memorization of the 100k-sim training set rather than orientation
memorization.

Before designing exp-8 (a 23–45h training run), spend ~4 short GPU jobs
answering: **is the under-dispersion a property of the model or of the
sampler?** All work is eval-only on the frozen exp-7 EMA checkpoint; no
training, no checkpoint writes.

Attribution logic:

- Euler 0.01 vs Euler 0.002 vs Dopri5 agree → integration is converged; the
  under-dispersion belongs to the model → exp-8 must fix it on the training
  side (dataset size / regularization / capacity — gate re-evaluated after
  this job).
- Finer/adaptive stepping changes the posterior std → part of the defect was
  an ODE discretization artifact → free fix (eval-side).
- SDE samplers recover the reference dispersion while the ODE stays narrow →
  the marginal law of the learned flow is fine but the deterministic sampler
  under-spreads; stochastic sampling is a legitimate cheap mitigation, and
  exp-8 can target the residual mean bias instead.
- TARP (ODE vs SDE) shows which calibration regime each sampler lives in.

## Deliverables

1. `eval-spherical-grf-diag.py` — v2-script-style standalone (copy of
   `train-spherical-grf-v2.py` with training stripped), same config-path CLI.
2. Four small mode configs: `config/config_healpix_7diag_{ode,sde,tarp_ode,tarp_sde}.yaml`.
3. `sub/spherical_grf_7diag.sub` — one submit file, four `queue` blocks
   (condor re-reads `config` between blocks); all jobs A100.
4. Results under the existing naming convention:
   `spherical_grf_fm_healpix_7diag_<mode>_results.txt`, TARP + overlay pngs
   in `imgs/`.

## Script design

- Reuses the v2 building blocks verbatim: `SphericalGRFModel`, `make_datasets`
  (loaders are needed only to construct the pipeline; `augment` irrelevant),
  `prep_x`, config parsing, results logging. `training.train_model: false`,
  `training.restore_model: true`, `experiment_id: 7` restores the exp-7 EMA
  checkpoint via `pipeline.restore_model()` (path proven by the exp-3/4
  eval-only recovery).
- New YAML section `diagnostic:` with `mode: ode | sde | tarp_ode | tarp_sde`
  and the sampler matrix parameters (below). Everything else inherited from
  `config_healpix_7.yaml` values.
- Sampler variants are passed through the existing pipeline API:
  `pipeline.sample(..., step_size=s, method=m, solver=(cls, {"alpha": a}))` —
  `ConditionalPipeline.get_sampler` forwards `**sampler_kwargs` to
  `FlowMatchingMethod.build_sampler_fn`, which accepts `method="Dopri5"` and
  SDE solver tuples (`ZeroEndsSolver`, `NonSingularSolver`,
  `gensbi.flow_matching.solver`). FM SDE solvers derive the score from the
  velocity field internally and require the Gaussian prior (satisfied).
- QUICK env smoke mode preserved (tiny sample counts, CPU-runnable) for a
  pre-submit sanity check.

## Sampler matrix

All matrix configs: observations 1–10, 10 000 posterior samples each,
`sample_chunk_size` 1000, C2ST against the task's 10k reference posterior
samples per observation. Log per config × obs: flow mean/std vs ref
mean/std, C2ST, wall time.

| job | config | sampler | purpose |
|---|---|---|---|
| ode | 1 | Euler, step 0.01 | baseline; must reproduce exp-7 obs 1–3 within C2ST scatter (validates restore) |
| ode | 2 | Euler, step 0.002 | step-size sensitivity |
| ode | 3 | Dopri5 (adaptive) | converged-ODE reference |
| sde | 4 | NonSingular SDE, α=0.5, step 0.01 | mild churn |
| sde | 5 | NonSingular SDE, α=1.0, step 0.01 | strong churn |
| sde | 6 | ZeroEnds SDE, α=1.0, step 0.01 | alternative churn schedule (diffusion → 0 at both ends) |

**C2ST scatter estimate** (ode job only): for config 1, obs 1–3, three
additional independent 10k draws with independent C2ST seeds → per-obs
C2ST mean ± std. This prices the exp-6↔exp-7 per-obs shuffling and sets the
significance threshold for all other comparisons and for exp-8.

**Overlay corner plots**: every variant, obs 1–3 only (18 pngs total; the
sde job has no in-job baseline to measure divergence against, so "most
divergent only" logic is dropped for uniformity).

## TARP jobs

200 fresh prior pairs × 1000 posterior samples (chunk 20), normalized theta
space — identical protocol to exp-6's TARP:

- `tarp_ode`: Euler, step 0.01 — the regime exp-7 actually shipped.
- `tarp_sde`: NonSingular SDE, α=1.0, step 0.01 — does stochastic sampling
  restore calibration?

Fresh simulator pairs use the same `SBIBM_JAX_USE_TEST=1` repo as training
(exported inside `run_spherical_grf.sh`).

## Submission plan

Single `spherical_grf_7diag.sub`, four jobs:

| job | GPU | est. wall |
|---|---|---|
| ode | A100 (`+UseNvidiaA100`) | ~4 h (Euler 0.002 dominates: 5× baseline × 10 obs) |
| sde | A100 | ~2.5 h (SDE ≈ 2 model evals/step) |
| tarp_ode | A100 | ~1 h |
| tarp_sde | A100 | ~2.5 h |

All four jobs on A100 (H100s are reserved for higher-priority work and have
a slower queue for us). A100 occupancy at design time (2026-08-13): 13/14
slots claimed, 1 free — jobs will start as slots free up; all four are
independent, so partial progress is fine. 8 CPU / 32 GB / 1 GPU per job as
in the existing sub files. Eval memory is bounded by the sample chunk
(~3 GiB), far under 40 GB.

## Success criteria

1. Config 1 reproduces exp-7's obs 1–3 C2ST within the measured scatter
   (else the restore or sampler wiring is wrong — stop and debug).
2. Every matrix cell reports mean/std/C2ST for all 10 obs; both TARP curves
   produced.
3. The results answer, with the scatter estimate as the yardstick: (a) does
   integration accuracy change dispersion? (b) does SDE sampling change
   dispersion/C2ST? (c) is the ODE posterior over-confident in TARP, and
   does the SDE fix it?

## Out of scope (deferred to exp-8 brainstorm, after results)

- Any training-side change (dataset size increase, regularization, capacity
  bump). The old "capacity bump if logA over-dispersed" gate is retired; the
  new decision takes these diagnostic results as input.
- Encoder re-encoding per ODE step (caching) — known perf issue, unchanged.
- Residual mean bias along the logA–α degeneracy — documented, not addressed
  by sampler choice.
