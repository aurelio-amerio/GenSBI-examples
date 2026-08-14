# Spherical GRF exp-8 ("v8"): probes + amplitude-pathway training runs — Design

**Date:** 2026-08-14
**Status:** approved for autonomous execution (user away ~3 days; execute via the
companion implementation plan in a fresh session)
**Predecessors:** exp-7 (`config/config_healpix_7.yaml`, checkpoint
`checkpoints/spherical_grf_fm_healpix_7/`, orbax step 7), 7diag
(`docs/superpowers/specs/2026-08-13-spherical-grf-exp7-sampler-diagnostic-design.md`,
results committed 2026-08-13/14).

## Motivation: what 7diag established

All eval-side hypotheses are closed:

- Gate passed: `euler_p01` reproduced exp-7's shipped C2ST (0.855±.002 /
  0.841±.002 / 0.728±.001 vs 0.852/0.842/0.729).
- Converged ODE (`euler_p002` ≈ `dopri5`) beats Euler 0.01 by 0.02–0.05 on all
  10 obs → **dopri5 (or euler_p002) is the standard eval sampler from now on**.
- FM-SDE samplers REFUTED (α=1.0 catastrophic 0.96–0.99; α=0.5 net worse).
- Raw-vs-EMA REFUTED: raw ≈ EMA (raw slightly worse); no EMA wash-out; the
  float32 fix holds.
- TARP (ODE): well calibrated (66.95/90.40/99.43% at nominal 68.27/95.45/99.73).
- exp-7 val loss still falling at 300k (0.1184→0.1135 over last 20%); train–val
  gap widened 20→33% (per user: NOT a concern unless val increases; no dropout,
  no weight-decay changes — AdamW's decay is lr-scaled and effectively nil, and
  in this LN-heavy network it would act as LR control, not regularization).

Residual defects, with two separate mechanistic hypotheses:

1. **n/α under-dispersion (12–20%) + posterior means displaced 0.7–1.7 σ_ref
   toward the truth** ("truth-regression"), sampler-independent.
   **Hypothesis H-INFO**: the reference posterior conditions on Ĉ_ℓ for
   ℓ ≤ 2·nside = 128 only (deliberate: anafast aliasing bias above 2·nside —
   see sbibm-jax `reference_posterior.py` docstring), while the network sees
   the full pixel map carrying power to ℓ = 191. Mode count: ~16.6k modes at
   ℓ=2–128 vs ~20.2k at ℓ=129–191. A network extracting high-ℓ information
   yields a narrower, closer-to-truth posterior that is C2ST-separable from
   the reference while remaining TARP-calibrated — exactly what is observed.
   P1 (grid validation, DONE, see below) already rules out sampler error.
2. **logA over-dispersion (~30% at converged ODE)**.
   **Hypothesis H-AMP**: the architecture has no linear amplitude channel.
   Code trace (heal_swin_nnx + script + gensbi flux1): dataset normalization
   uses global scalars (per-map amplitude survives input); PatchEmbed is
   linear (exp-7 has no patch-embed LN); Swin-V2 res-post-norm blocks see the
   raw stream in-branch but LayerNorm every branch output; cosine attention +
   QK-norm make all routing amplitude-blind; PatchMerging LNs before mixing;
   the **encoder's final per-token LayerNorm destroys absolute amplitude**;
   GlobalTokenBlocks (pre-LN) and Flux1 (QKNorm, t-driven AdaLN) cannot
   recover it. logA information survives only as a saturating nonlinear code
   (bias-vs-signal ratios inside LNs, GELU curvature) — consistent with
   inflated logA uncertainty.

## Probes (P1–P3): settle both hypotheses before/alongside v8

### P1 — grid validation of the reference sampler (DONE 2026-08-14, PASSED)

`v8_probes/p1_grid_check.py`: the ℓ≤128 likelihood is separable in θ
(e^{-logA} factors; Σw·lnC linear), so the exact 3-param grid posterior is
cheap. All 10 obs: MCLMC-vs-grid mean offsets ≤0.05σ, stds within ±2.5%,
correlations match (`v8_probes/p1_grid_check_results.txt`). **The shipped
references are exactly right for the ℓ≤128 likelihood.** Any flow-vs-reference
gap is about information content, not sampler error.

### P2 — extended-ℓ corrected reference + C2ST re-comparison (tests H-INFO)

1. **Transfer function**: per obs i, simulate M=200 maps at θ_true(i) with the
   task's own healpy path (`synfast`, lmax=191, new=True, no pixwin), measure
   w_ℓ(i) = ⟨anafast(map)_ℓ⟩ / C_ℓ(θ_true(i)) for ℓ=2..191. Also check w_ℓ is
   ~θ-independent (compare across obs; expect geometric quadrature effect
   ≈1 below ℓ≈128, up to ~7% suppression above).
2. **Corrected reference**: re-run the sbibm-jax MCLMC (reuse
   `reference_posterior.py` machinery with `make_logdensity` modified to
   D_ℓ = w_ℓ·C_ℓ(θ), ℓ ≤ 191) for obs 1–3 minimum, all 10 if cheap.
   r̂ < 1.01 required. CPU is fine (3 params).
3. **C2ST**: exp-7 flow samples (10k, dopri5 — dumped by P3's GPU job) vs the
   corrected reference. Also compare moments (flow vs corrected-ref mean/std).

**Interpretation:** if C2ST(flow vs corrected-ref) drops substantially below
C2ST(flow vs shipped-ref) (e.g. 0.82→≤0.65) and the mean displacement shrinks,
H-INFO is confirmed: much of the "model defect" is a benchmark information
mismatch. **Do NOT modify the canonical references in sbibm-jax** — that is
the user's decision on return; v8 evaluation simply reports against both.

### P3 — amplitude-equivariance probe on the frozen exp-7 checkpoint (tests H-AMP)

Scaling a map by e^{δ/2} is *exactly* a logA shift of +δ (ln C_ℓ = logA + …).
On GPU (one condor job): for obs 1–3 and δ ∈ {−0.4,−0.2,−0.1,0,+0.1,+0.2,+0.4},
sample N=2000 (dopri5, EMA weights) from the flow given map·e^{δ/2}; record
posterior mean/std of all 3 params. Fit slope of mean-logA vs δ.

- slope ≈ 1 and n/α means flat: no amplitude compression end-to-end → H-AMP
  weakened; v8-B's lever switches to capacity (see gate below).
- slope < ~0.9 or curvature (saturation at large |δ|): H-AMP confirmed,
  quantified.

Same GPU job dumps 10k dopri5 samples for obs 1–3 (unnormalized θ) to
`v8_probes/exp7_dopri5_samples_obs{1,2,3}.npz` for P2's C2ST.

## v8 training runs

Two parallel 600k-step A100 runs (exp-7 did 300k in ~23h → ~45h each; both
finish within the user's 3-day absence if slots are free):

- **Run A ("v8a", control):** exp-7 architecture unchanged; only
  `nsteps: 600000` (warmup/decay_transition fractions unchanged →
  decay_transition 0.6 stays fractional-equivalent). Answers "was exp-7 just
  under-trained?" and provides the clean baseline for B.
- **Run B ("v8b", the bet):** Run A + the **raw-map power pass-through**
  (H-AMP fix, "Option 2"), OR — only if P3 shows slope ≈ 1 (no compression) —
  Run A + the reserved capacity bump (embed_dim 96, depths [2,2,6,4],
  global_depth 4) instead.

### The raw-map power pass-through (Option 2)

The GRF sufficient statistic is quadratic in the map; give the network a
parallel path that squares and pools the raw input, entirely inside
`SphericalGRFModel` in the training script (no heal-swin-nnx/gensbi changes):

1. From the NEST-ordered input map x (npix = 49152, already dataset-
   normalized), compute per-superpixel mean of x² at nside 16, 8, 4 —
   NEST ordering makes this a reshape-mean: (12·16², 16)-mean, etc.
2. For each scale s ∈ {16, 8, 4}: take log(P_s + eps) of the pooled power
   P_s, then reshape-mean the log values down to the 192-cell nside-4 grid.
   Result: 3 channels on the nside-4 grid, aligned with the 192 bottleneck
   tokens (both NEST).
3. Standardize each channel with fixed constants measured on one training
   batch at init (or just a fixed scalar shift/scale — keep simple, log-space
   values are O(1) across the prior).
4. Project with a **zero-initialized** nnx.Linear(3 → COND_FEATURES) and add
   to the encoder output tokens (after the global blocks, before cond_in), so
   v8b ≡ v8a exactly at init. (Zero-init of a linear feeding an add has
   nonzero gradients — not the AdaLN-zero trap.)

logA enters these features linearly (f_s ≈ logA + shape terms + sampling
noise); the three scales also carry band-power ratios for n/α. Params: ~3·512
+ bias ≈ 2k. Raw-map-only constraint respected (internal architectural
computation; no pseudo-Cℓ inputs, no vec_in features).

### Naming and configs

- Configs: `config/config_healpix_8a.yaml`, `config/config_healpix_8b.yaml`
  (standalone, exp-7 style — not the 7diag overlay mechanism, since the
  training script `train-spherical-grf-v2.py` has no merge support; copy
  config_healpix_7 and edit).
- Script: extend `train-spherical-grf-v2.py` with the pass-through behind a
  YAML switch `encoder.power_passthrough: true` (default false → v8a and all
  earlier configs untouched). run_name: `spherical_grf_fm_healpix_8a` / `_8b`.
- Submit: `sub/spherical_grf_v8.sub`, two queue blocks, A100, 8 CPU/32 GB,
  same runner contract. NEVER H100.
- Eval within the training script stays exp-7-style for the in-run smoke eval;
  the real post-run eval uses the 7diag machinery (below).

### Post-run evaluation (auto, when each run finishes)

Reuse `eval-spherical-grf-diag.py` (7diag) pointed at the v8 checkpoints via
small configs: `mode: ode` matrix with variants euler_p01 (baseline,
scatter ×3 on obs 1–3) + dopri5, all 10 obs; plus `tarp_ode` (200×1000).
Compare against: exp-7 EMA numbers (above), AND the P2 corrected reference
(re-run the C2ST offline against corrected samples for obs 1–3).

Success criteria:
- v8a vs exp-7: any C2ST/val improvement = pure convergence gain.
- v8b vs v8a: logA std ratio (flow/ref) moving 1.3 → ~1.0 and C2ST drop
  beyond v8a's = H-AMP fix pays off. TARP must stay calibrated.
- Against corrected reference (if P2 confirms H-INFO): report both; the
  benchmark-definition question is deferred to the user.

## Constraints (binding, carried from user decisions)

- Dataset frozen at 100k train sims; 8× isometry augmentation ON (as exp-7).
- No dropout; no weight-decay change; EMA unchanged (decay as exp-7).
- Raw map stays the only input; no pseudo-Cℓ/vec_in feature engineering.
- Eval-only reads of exp-7 checkpoint; v8 writes its own checkpoint dirs.
- `SBIBM_JAX_USE_TEST=1` for anything touching data (in run script already;
  export explicitly for local commands). Local validation: `JAX_PLATFORMS=cpu`,
  gensbi env.
- SMOKE/QUICK env-mode test convention; QUICK debris never committed.
- Do not modify sbibm-jax canonical reference files. P2 artifacts live in
  GenSBI-examples under `examples/sbi-benchmarks/spherical_grf/v8_probes/`.
- A100 only.

## Execution order (fresh session)

1. P3 GPU job (submit first — it queues while P2 runs on CPU) and P2 locally.
2. Gate: pick v8b's lever from P3 (power pass-through vs capacity bump).
3. Implement v8 (script switch + configs + submit file), SMOKE/QUICK, commit.
4. Submit both runs; verify startup; background-watch.
5. On completion: auto-eval, write results + comparison summary, update memory.
6. Report: probes verdicts (H-INFO/H-AMP), v8a/v8b outcomes, recommendation
   for the benchmark-definition question — all committed for the user's return.
