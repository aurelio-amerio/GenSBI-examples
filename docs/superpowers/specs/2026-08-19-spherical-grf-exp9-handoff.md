# Spherical GRF exp-9 — design-discussion handoff

**Date:** 2026-08-19 (discussion after exp-8 completion; execute in a fresh session)
**Status:** direction agreed in conversation, NOT yet a spec/plan — write the spec first, then a plan.
**Predecessor:** exp-8, fully done and committed — read `examples/sbi-benchmarks/spherical_grf/v8_probes/v8_final_report.md` first (commit `abcec88f`, fixes in `6de1d141`).

## Where exp-8 left the science

The residual defect is a **per-observation posterior mean displacement of 0.7–1.7 σ_ref**
("truth-regression": the flow sits near the prior-averaged response instead of tracking the
observation's noise realization), plus logA ~30% over-dispersion; TARP stays roughly calibrated.
Ruled out by direct experiment: sampler error (P1/7diag), SDE samplers, EMA artifacts,
reference ℓ≤128 information mismatch (P2: corrected ℓ≤191 refs move flow C2ST ≤0.04),
amplitude blindness (P3: logA response slope ≈ 1.000 on every checkpoint), encoder capacity
(8b: bigger encoder = worse on 7–8/10 obs), and most of under-training (8a: 500k vs 300k bought
only 0.02–0.05 on obs 1–2, nothing on obs 3; dopri5 0.805/0.763/0.715 vs exp-7 0.822/0.814/0.700).
Still standing: **data volume** (100k sims — the one axis never varied), **objective/head**
(flow matching's velocity-regression signal), architecture *class* (not size).

## Exp-9 direction agreed (two complementary prongs)

### Prong 1 — head swap: TarFlow NPE head on the unchanged HealSwin encoder

Hypothesis: θ is 3-d and near-Gaussian; the hard part is the encoder. The 40M-param Flux1
flow-matching head is overkill AND trains the encoder only through a high-variance velocity
regression (irreducible conditional-vs-marginal velocity gap; poor SNR for a tight posterior).
An exact-likelihood MLE head gives clean per-sample placement gradients — and "placement" is
exactly the failure signature.

Decisions already made in discussion:
- **TarFlow, not MAF** — exact-MLE property is the same, but TarFlow has proven in-repo
  plumbing (`tarflow_npe_slcp` / `tarflow_nle_slcp` examples; channel support). Keep it
  deliberately SMALL (few blocks, modest width) so this tests head *class*, not capacity.
- **No separate Gaussian-head arm** — a 1-layer affine autoregressive transform conditioned
  on x IS a conditional Gaussian (Cholesky parameterization), so the small flow nests it and
  starts there at identity init. Recover the Gaussian diagnostic post-hoc for free: fit a
  moment-matched Gaussian to the trained flow's per-obs samples and C2ST it vs the reference —
  if it matches the flow's own score, the posterior was Gaussian-sufficient.
- **Conditioning = the full 192×512 encoder tokens as context tokens, NOT pooled.**
  AR invertibility only constrains attention *among the transformed θ tokens* (triangular
  Jacobian); conditioning is exogenous, contributes nothing to the log-det, and may be fully
  visible to every θ position in every block and both directions. Identical conditioning
  interface to Flux1 ⇒ clean attribution (only head + objective change).
  Implementation gotchas: (a) TarFlow alternates AR direction between blocks — a naive causal
  mask over a concatenated sequence would hide cond tokens in reversed blocks; the mask must be
  causal among θ tokens only, cond columns always visible (or cross-attention — check how the
  in-repo SLCP NPE tarflow does it); (b) cond tokens must be excluded from the transform and
  the log-det.
- **Training-noise gotcha:** the TarFlow recipe adds Gaussian noise to targets. Reference
  posterior stds are TINY (0.015–0.05 in raw θ). Noise must be ≪ the smallest posterior std in
  *normalized* θ space (or annealed/zero), else it manufactures fake over-dispersion — one of
  the very symptoms under study. Verify learned widths aren't noise-floored.
- **Optional cheap attribution probe first:** train the small head on 8a's FROZEN encoder
  features (hours). Fixes displacement ⇒ features had the info, FM head wasted it. Doesn't ⇒
  encoder/data limit. Then retrain end-to-end for the fair headline comparison. Control = 8a.
- Bonus: MLE head kills the eval-cost problem (no ODE, no re-encoding per step; sampling =
  3 sequential AR steps).

Interpretation matrix: TarFlow fixes displacement → FM/Flux1 objective was the culprit.
Same displacement → head exonerated, encoder/data confirmed → prong 2 discriminates further.

### Prong 2 — data-scaling probe

Dataset (100k train sims, frozen through exp-5..8) is the last untouched axis, and
truth-regression is the classic signature of an information/data limit (hedging toward the
conditional-mean behavior learnable from finite pairs). A scaling curve (e.g. 25k/50k/100k,
or extend if new sims are cheap) on the fixed 8a architecture discriminates the data
hypothesis independently of prong 1.

## Operational constraints carried forward (all learned the hard way in exp-8)

- **48h wall-time hold** (hold = vacate for vanilla jobs; verified exit-143 kills at exactly
  48.0h). No job may exceed 48h; size schedules accordingly (~3.0–4.3 it/s observed on A100
  for the v8 models; 500k ≈ 42h for 8a-size, 8b missed at 494k/500k).
- **`training.save_every` intermediate checkpoints** now first-class in gensbi
  (f9bf543 + 0f5494b + 8653d33 — parser-exposed; wip/ dirs, max_to_keep=2, stale-clear on
  resubmit, `restore_intermediate`; promotion recipe = restore_intermediate + save_model,
  proven on 8b). Use it in every long run (25k cadence worked). Dormant caveat: save_every is
  not multistep-scaled (fine while `multistep: 1` everywhere).
- **Env scrub before every condor_submit with getenv=True**: `env -u VIRTUAL_ENV
  -u VIRTUAL_ENV_PROMPT`, PATH stripped of stray venvs (tng-filaments pollution bit twice);
  also grep env for `POWER_PASSTHROUGH*` (now SMOKE-gated in code) and `JAX_PLATFORMS`.
- **Orbax step key = `training.experiment_id`**, which v8 configs inherit as 7 → checkpoints
  live at `checkpoints/spherical_grf_fm_healpix_8{a,b}/7/` (+ `ema/7/`). Bump experiment_id
  for exp-9 or keep the convention knowingly.
- `SBIBM_JAX_USE_TEST=1` on every data-touching command; A100 only; SMOKE/QUICK test
  convention; dataset stats banner x_mean=-1.75e-06 confirms the test repo.
- TarFlow image examples live partly on unmerged branch `field-level-inference`; the SLCP
  NPE/NLE tarflow examples are the closer template — locate them before writing the spec.

## Repo state at handoff

- GenSBI-examples: 12 exp-8 commits `58b525d7..6de1d141` on main, **unpushed**.
  GenSBI: `f9bf543`, `0f5494b`, `8653d33`, **unpushed**. Push = user's call.
- Reusable eval machinery: `eval-spherical-grf-diag.py` overlay configs (pattern:
  `config_healpix_8a_diag_ode.yaml`), `v8_probes/p2_c2st_compare.py` takes `FLOW_PREFIX`,
  `v8_probes/p3_amplitude_probe.py` runs on any diag config. exp-8 checkpoints stay read-only.
- Deletable when convenient (not done): `checkpoints/spherical_grf_fm_healpix_8{a,b}/wip*/`
  (redundant; 8b's 475000 promoted), uncommitted `v8_probes/dopri5_samples_*.npz` dumps.
- Deferred to exp-9 from the final review: log `power_passthrough` in real run headers before
  any passthrough run; auto-resume-from-wip as save_every's companion; generalize the p3
  probe docstring ("frozen exp-7 checkpoint" is stale).
