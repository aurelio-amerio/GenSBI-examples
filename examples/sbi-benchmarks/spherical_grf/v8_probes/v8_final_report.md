# Spherical GRF exp-8 (v8) — final report

Plan: `docs/superpowers/plans/2026-08-14-spherical-grf-v8.md`. Spec:
`docs/superpowers/specs/2026-08-14-spherical-grf-exp8-design.md`. Ledger:
`.superpowers/sdd/2026-08-14-spherical-grf-v8/progress.md`.

exp-8 asked two questions against the exp-7 baseline (300k steps, C2ST ~0.70-0.90
across obs, val loss still falling at end of training): (a) does more training
(500k-600k steps) converge the residual C2ST gap (obs-specific mean displacement
vs the reference posterior), and (b) does more model capacity help. Two
architectures were trained: **8a** = exp-7 architecture (embed_dim 64, depths
(2,2,6,2), 2x8h global) run longer; **8b** = capacity bump (embed_dim 96, depths
(2,2,6,4), 4x8h global, cond 192x768), selected by the P3 gate (see §2).

## 1. Run summary & the 48h failure

The first attempt trained both runs for 600,000 steps on cluster `1103473`
(8a=.0, 8b=.1). At the observed throughput (2.4-4.4 it/s) 600k steps needs
~48.8h (8a) to ~56h (8b) of wall time. The cluster enforces a hard 48.0h
wall-time hold (verified against 3 prior jobs that died at exactly 48.0h, exit
143), and the gensbi training pipeline at the time saved checkpoints **only**
at the end of the training loop — no intermediate saves. Both conditions were
identified in the ledger; the failure was detected at ~47h elapsed via a user
question, not by monitoring. A GDB in-process salvage of the live JAX state was
staged and offered but **the user declined it**; jobs were `condor_rm`'ed at
47h11m with 8a at ~583k/600k steps and 8b at ~524k/600k steps. Result: **~94
A100-hours lost, zero checkpoints, nothing recoverable.**

Root causes: (1) 600k steps does not fit the cluster's 48h wall limit at
observed throughput; (2) no intermediate checkpointing in the training loop.

Remedy: `training.save_every` periodic checkpointing was added to the gensbi
pipeline (GenSBI commits `f9bf543` + `0f5494b`, plus the companion
GenSBI-examples commit `7120ab84`; a `wip` orbax CheckpointManager,
`max_to_keep=2`, step-keyed, with a `restore_intermediate` recovery path;
reviewed and fix-rounded before use). Configs were also cut from 600k to 500k
steps per user directive (commit `e859d3e7`, `save_every: 25000`).

The re-run (cluster `1103529`) submitted both jobs at 500k steps with
`save_every=25000`:
- **8a completed 500k steps cleanly in 42.2h** (inside the 48h limit). Exit 0.
  Final train loss 0.0729, final val loss 0.1356.
- **8b hit the 48h hold at step 494,444/500,000 (98.9%)** — one save short of
  the finish line. No final checkpoint was written, but the new `wip`
  mechanism had a checkpoint at **step 475,000 (95% of the 500k schedule)** on
  disk. This was promoted to the final checkpoint layout
  (`checkpoints/spherical_grf_fm_healpix_8b/{7,ema/7}`) via
  `restore_intermediate` + `save_model`; restore+sample sanity checks were
  finite and the layout matched 8a's. The subsequent eval job restored this
  promoted checkpoint cleanly (correct 96/768/4-global architecture banner,
  no tracebacks) — the recovery path is validated end-to-end.

**All 8b numbers below are from the step-475,000 checkpoint, not step 500,000**
— the promoted checkpoint is 475,000/500,000 = 95% of 8a's full schedule (a
5% step deficit vs 8a), and is itself ~4% short of the point where the job
was actually killed (494,444 steps), the gap between kill point and promoted
checkpoint being an artifact of the 25k-step wip save cadence. This is on top
of 8b already needing more steps to converge as a larger model.

## 2. Probe verdicts (from the 7diag baseline, re-checked on v8a/v8b)

- **P1** (sampler check, grid-vs-MCLMC reference at l<=128): passed for all 10
  obs (|delta_mean|<0.05 sigma, |std_ratio-1|<0.03). The sampler/reference
  pipeline is exonerated as a driver of the C2ST gap.

- **P2 / H-INFO** (does the shipped l<=128 truncated reference, vs a corrected
  l<=191 reference, explain the gap?): P2b built corrected references (all
  r-hat < 1.01, N_eff sufficient) via a w_l-corrected likelihood; std shrinks
  by roughly 10% (logA), ~41% (n), and ~37% (alpha) vs the shipped grid. P2c
  compared flow-vs-reference C2ST under both references:
  - exp-7 (300k): shipped→corrected C2ST 0.820→0.781, 0.815→0.775,
    0.704→0.703 (drops 0.039/0.040/0.001). **H-INFO REFUTED** — criterion
    (>=0.10 drop on >=2/3 obs) not met.
  - v8a re-check: 0.807→0.791, 0.758→0.740, 0.723→0.700 (drops
    0.016/0.018/0.023). Confirms the exp-7 finding.
  - v8b re-check: 0.870→0.799, 0.777→0.696, 0.724→0.722 (drops
    0.071/0.081/0.002). Larger sensitivity than 8a/exp-7 on obs1-2, but still
    under the 0.10 criterion — H-INFO stays refuted.

- **P3 / H-AMP** (amplitude-equivariance defect — does the model's response to
  a synthetic logA shift depart from the truth slope 1.0?): exp-7 slopes
  0.9972 / 1.0037 / 1.0092 (mean 1.0034, all n/alpha drifts < 0.02).
  **H-AMP REFUTED**. This result set the P3 gate to **V8B_LEVER = capacity**
  (mean slope >= 0.93 and drifts < 0.02 both hold, ruling out amplitude
  passthrough as the lever). v8 re-checks: 8a 0.9948 / 1.0135 / 0.9975; 8b
  1.0095 / 1.0366 / 0.9877 (n/alpha drifts all ≤ 0.037). Amplitude equivariance
  is intact in both v8 runs — H-AMP stays refuted after more training and more
  capacity.

## 3. v8a vs exp-7 (does more training converge the gap?)

| obs | euler_p01 8a | euler_p01 exp-7 | dopri5 8a | dopri5 exp-7 |
|---|---|---|---|---|
| 1 | 0.830 | 0.855 | 0.805 | 0.822 |
| 2 | 0.779 | 0.841 | 0.763 | 0.814 |
| 3 | 0.762 | 0.728 | 0.715 | 0.700 |

Full 10-obs euler_p01 (8a): 0.830, 0.779, 0.762, 0.799, 0.810, 0.688, 0.874,
0.920, 0.743, 0.711.
Full 10-obs euler_p01 (exp-7): 0.855, 0.841, 0.728, 0.797, 0.845, 0.742, 0.895,
0.902, 0.701, 0.688.

Final val loss: 8a 0.1356 vs exp-7 0.1184. Val loss is noisy under this
training setup (early-stopping counter oscillates on val ratio spikes); during
the killed first (600k) attempt, the same 8a architecture had already reached
val ~0.1075 at ~240k steps (40% through that 600k schedule) — below exp-7's
final 0.1184 — indicating the completed 500k run's final-val readout of 0.1356
is a noisy single sample rather than a plateau estimate, though it is not a
like-for-like comparison since it comes from the earlier, killed run.

TARP (`imgs/spherical_grf_fm_healpix_8adiag_tarp_ode_tarp.png`, dopri5,
200x1000 draws): the credibility-coverage curve tracks the diagonal closely
through low-to-mid credibility and drifts into the overconfident side
(curve below the diagonal) through the mid-to-upper range before coming back
toward it near the top. Confidence-panel readout (as plotted): 62.2% empirical
at nominal 68.27%, 86.3% at nominal 95.45%, 99.1% at nominal 99.73% — mildly
overconfident at 1- and 2-sigma, essentially calibrated at 3-sigma.

**Verdict**: more training buys a modest, inconsistent gain. obs1-2 improve
(-0.025, -0.062 euler_p01; -0.017, -0.051 dopri5) but obs3 regresses slightly
(+0.034 euler_p01, +0.015 dopri5). Under-training explains part of exp-7's gap
but clearly not all of it — even at 500k steps with a lower best-seen val loss,
C2ST sits well above the 0.5 ideal on every observation.

## 4. v8b vs v8a (does more capacity help?)

| obs | euler_p01 8b | euler_p01 8a | dopri5 8b | dopri5 8a |
|---|---|---|---|---|
| 1 | 0.880 | 0.830 | 0.870 | 0.805 |
| 2 | 0.802 | 0.779 | 0.779 | 0.763 |
| 3 | 0.746 | 0.762 | 0.721 | 0.715 |

Full 10-obs euler_p01 (8b): 0.880, 0.802, 0.746, 0.849, 0.787, 0.674, 0.888,
0.952, 0.852, 0.766. (8a list above for comparison.)

TARP (`imgs/spherical_grf_fm_healpix_8bdiag_tarp_ode_tarp.png`): visually very
similar shape to 8a — close to the diagonal at low credibility, mildly
overconfident through the middle, converging back near the top. Confidence
readout: 65.4% at nominal 68.27%, 87.2% at nominal 95.45%, 99.2% at nominal
99.73% — marginally better-calibrated than 8a at each checkpoint, but the same
qualitative pattern.

**Verdict**: **the capacity lever did not pay off.** Across the full 10-obs
sets, 8b is worse than 8a on 7/10 obs under euler_p01 (obs1,2,4,7,8,9,10;
better on obs3,5,6 by small margins) and 8/10 obs under dopri5
(obs1,2,3,4,7,8,9,10; better only on obs5,6) — including all of obs1-2's
large margins in both samplers (obs1 +0.050 euler_p01/+0.065 dopri5, obs2
+0.023 euler_p01/+0.016 dopri5). obs3 is a small, sampler-dependent wash
(-0.016 euler_p01 improvement vs +0.006 dopri5 regression). The 475k-vs-500k
step deficit (caveat from §1) is a plausible partial explanation but is
unlikely to account for an effect as large as obs1's +0.065 given how flat
the loss/C2ST trajectory looks near convergence in both runs. As a secondary
observation, 8b's P2c
shipped→corrected sensitivity is notably larger than 8a's on obs1-2 (drops of
0.071/0.081 vs 8a's 0.016/0.018) — a curiosity, not something that changes
either H-INFO verdict, but possibly indicative of the larger model encoding
more high-l information into its posterior shape even though its overall C2ST
is worse.

## 5. Recommendation for exp-9

Evidence collected across the P1-P3 probes and the two 500k runs now excludes
or substantially weakens every driver tested so far as *the* explanation for
the residual C2ST gap (roughly 0.7-0.8 vs the 0.5 ideal, observation-dependent):

- **Sampler**: excluded (P1 passed; grid and MCLMC references agree to
  <0.05 sigma).
- **Reference truncation (l<=128 vs l<=191)**: excluded as a major driver
  (H-INFO refuted three times — exp-7, 8a, 8b — max observed drop 0.081, well
  under the 0.10 bar). It does materially shrink the reference posterior's own
  std (roughly 10% for logA, ~37-41% for n/alpha), so it matters for the
  benchmark's *definition* (see below) but not for explaining why the flow
  disagrees with either reference.
- **Amplitude-channel defect**: excluded (H-AMP refuted before and after more
  training/capacity; slopes stay in the 0.988-1.037 band with small drifts).
- **Capacity**: tested directly (8b) and it made things worse, not better —
  ruled out as a lever, at least along the specific axis tried (wider
  embed_dim, deeper global blocks).
- **Under-training**: mostly excluded — 8a's extra 200k steps (300k→500k)
  bought only a small, observation-inconsistent improvement, and the best-seen
  val loss during 8a's run was already close to what more steps delivered at
  the end.

What remains untouched: **data volume** (the training set has been frozen at
100k simulations across exp-6 through exp-8), the truth-regression mechanism
itself (whether the objective/parameterization systematically pulls posterior
means toward the truth in a way C2ST penalizes), the training schedule/
optimizer/regularization choices beyond what capacity scaling covers, and
architecture classes other than width/depth scaling (e.g. different attention
patterns, conditioning mechanisms).

**Suggested next step (for the user's decision): a data-scaling probe first.**
It is the one axis in this list that has never been varied across exp-6/7/8,
it is cheap to test at small scale (e.g. train on 2x or 4x simulations, or
even just check the loss/C2ST trend as a function of dataset fraction under
the existing 100k set) before committing another 40+ A100-hour full run, and
if data volume is *not* the driver either, the schedule/optimization and
truth-regression-mechanism hypotheses become the natural remaining candidates.

**Separately, a benchmark-definition question is now ready for the user to
decide**: should the project's reference posteriors switch from the shipped
l<=128 truncation to the corrected l<=191 references? P2 evidence: the
corrected references are themselves much tighter (std shrinks roughly 10% for
logA and ~37-41% for n/alpha) and are the "more correct" target, but switching does not
materially change any flow's measured C2ST (drops of 0.00-0.08, all under the
0.10 "matters" bar tested). This is a correctness/reproducibility choice more
than a performance one.

## 6. Operational lessons

- **48h wall-time policy**: this cluster holds (effectively vacates) vanilla
  jobs at exactly 48.0h wall time, confirmed against multiple jobs dying at
  that exact boundary with exit 143. Any run whose ETA is close to or over 48h
  needs either a step budget with real margin, or checkpointing that survives
  a hold — do not rely on the visible ETA alone; 8a's actual 42.2h for 500k
  steps was already close to the ceiling.
- **`getenv=True` env-pollution gotcha**: HTCondor's `getenv=True` inherits the
  *submitting shell's* full environment, including stale `VIRTUAL_ENV`/`PATH`
  from unrelated projects (proven twice this experiment: job 1103457 crashed
  on a stale `VIRTUAL_ENV`, and the Task 8 controller shell was independently
  found polluted with a stale `VIRTUAL_ENV` — a `tng-filaments` venv — right
  before the v8 training submission and had to be scrubbed via
  `env -u VIRTUAL_ENV -u VIRTUAL_ENV_PROMPT` plus a cleaned `PATH`; a
  `POWER_PASSTHROUGH` env leak was separately identified as a live risk but
  never actually observed). Always scrub `VIRTUAL_ENV`/`VIRTUAL_ENV_PROMPT`/
  `PATH` and grep for feature-flag env vars immediately before `condor_submit`.
- **`training.save_every` should be a default, not an opt-in**, for any run
  expected to approach the wall-time limit. It was added reactively after a
  ~94 A100-hour loss; a reasonable default cadence (this experiment used
  25,000 steps, giving at most ~2h of exposure at observed throughput) should
  ship with any long-running training config going forward, with `wip`
  checkpoints understood as a first-class recovery path (`restore_intermediate`),
  not just a safety net.
- **Orbax step key = `experiment_id`, not the training step count.** Both 8a
  and 8b configs were copied from the exp-7 config and inherited
  `experiment_id: 7`; their final checkpoints therefore live at
  `checkpoints/spherical_grf_fm_healpix_8{a,b}/7/` (not a step-8 or step-500000
  directory). This is consistent with how the eval configs restore checkpoints
  (same experiment_id via config merge) so nothing broke, but it means the
  on-disk step label does not describe the run and should not be used to
  infer training progress — check the `wip/` directory listing or the run log
  instead.
