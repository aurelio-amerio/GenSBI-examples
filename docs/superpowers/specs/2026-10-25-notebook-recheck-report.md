# Example notebooks re-check — report

Date: 2026-10-25. Design: `2026-10-25-notebook-recheck-design.md`.

All 26 tracked notebooks were re-checked (GenSBI 0.5.1, `MPLBACKEND=Agg`, from the notebook's own directory, via jupytext py:percent working copies).

Runs were done on two machines:

- The first two notebooks (`my_first_model`, `methods_and_samplers`) were run partly on a GPU box with `/home/aure/github/GenSBI/.venv`.
- Everything else on a CPU-only node: JAX forced to CPU, 6 pinned cores, with run-time "shrink" overrides of sample / grid / diagnostic sizes applied to an in-memory copy (the working copies kept the real sizes, and so do the committed notebooks).

Diagnostics from shrunk CPU runs are not comparable to the README numbers. Full-size behaviour (e.g. GPU memory) was not exercised after the move to the CPU node. The working copies, the shrink files and the runner scripts were deleted at the end of the branch (since removed; `scripts/recheck/`, all `*.wip.py`).

## Per notebook

| Notebook | Status | Notes |
|---|---|---|
| examples/getting_started/my_first_model | fixed | Regenerated from the newer `.md`. Fixed: SBC cell used undefined `test_data` (was assigned to `test_data_`); missing import of `compute_marginal_coverage`/`plot_marginal_coverage`; L-C2ST drew one sample per condition with `pipeline.sample` on a batch of 10k conditions, which current GenSBI rejects → `sample_batched(key, xs_, nsamples=1)[0]`; log-prob grid (132k points) OOMed on a 12 GB GPU (19.45 GiB in one solve) → evaluated in 10k-point chunks; Colab `%cd` path had a stray `/GenSBI-examples`; `$\theta$` rendered as a tab (3×); "fomulation"; TARP expansion ("Tests of Accuracy with Random Points"); `# (3,)` shape comment; 5 `savefig` lines labelled "uncomment to save" were active → commented out (the marginals one was already commented on `main`); the log-prob sentence ("This method requires an ODE formulation, and is currently implemented for Flow Matching ... and for Score Matching models using the probability flow ODE formulation") was restored from `main`'s `.ipynb`, as the newer `.md` had the outdated "currently only implemented for Flow Matching". Diagnostics: SBC KS p-values 0.18/0.70/0.37, TARP ATC −0.21 with KS p = 1.0. |
| examples/methods_and_samplers/methods_and_samplers | fixed | Regenerated from `.md`. Fixed: exact FM log-prob on the ~132k-point grid OOMed on a 12 GB GPU -> evaluated in 10k-point chunks with `jnp.concatenate`; missing period in an EDM comment. All FM, EDM and SM cells ran clean (checkpoints flow/edm/sm restored, exit=0) on a shrunk CPU run (5k samples, small log-prob grids, ~30 min); no diagnostics computed. |
| examples/sbi-benchmarks/gaussian_linear/flow_flux | fixed (text only) | Ran clean unshrunk on CPU (5.3 min, exit=0, restore + one 10k-sample posterior). Text fixes only (wording/typo fixes from the generic template; see Template fixes). The notebook does not compute C2ST; the stored text value (0.5087 ± 0.0061) matches the example README. |
| examples/sbi-benchmarks/gaussian_linear/flow_flux1joint | fixed (text only) | Ran clean unshrunk on CPU (13.3 min, exit=0, restore + one 10k-sample posterior). Text fixes only (wording/typo fixes from the generic template; see Template fixes). The notebook does not compute C2ST; the stored text value (0.5066 ± 0.0040) matches the example README. |
| examples/sbi-benchmarks/bernoulli_glm/flow_flux | fixed (text only) | Ran clean unshrunk on CPU (8.6 min, exit=0, restore + one 10k-sample posterior). Text fixes only (wording/typo fixes from the generic template; see Template fixes). The notebook does not compute C2ST; the stored text value (0.5594 ± 0.0143) matches the example README. |
| examples/sbi-benchmarks/bernoulli_glm/flow_flux1joint | fixed (text only) | Ran clean unshrunk on CPU (13.2 min, exit=0, restore + one 10k-sample posterior). Text fixes only (wording/typo fixes from the generic template; see Template fixes). The notebook does not compute C2ST; the stored text value (0.5574 ± 0.0197) matches the example README. |
| examples/sbi-benchmarks/gaussian_mixture/flow_flux | fixed (text only) | Ran clean unshrunk on CPU (2.6 min, exit=0, restore + one 10k-sample posterior). Text fixes only (wording/typo fixes from the generic template; see Template fixes). The notebook does not compute C2ST; the stored text value (0.5087 ± 0.0072) matches the example README. |
| examples/sbi-benchmarks/gaussian_mixture/flow_flux1joint | fixed (text only) | Ran clean unshrunk on CPU (3.1 min, exit=0, restore + one 10k-sample posterior). Text fixes only (wording/typo fixes from the generic template; see Template fixes). The notebook does not compute C2ST; the stored text value (0.5014 ± 0.0038) matches the example README. |
| examples/sbi-benchmarks/slcp/flow_flux | fixed (text only) | Ran clean unshrunk on CPU (9.4 min, exit=0, restore + one 10k-sample posterior). Text fixes only (wording/typo fixes from the generic template; see Template fixes). The notebook does not compute C2ST; the stored text value (0.6886 ± 0.0472) matches the example README. |
| examples/sbi-benchmarks/slcp/flow_flux1joint | fixed (text only) | Ran clean unshrunk on CPU (14.5 min, exit=0, restore + one 10k-sample posterior). Text fixes only (wording/typo fixes from the generic template; see Template fixes). The notebook does not compute C2ST; the stored text value (0.5490 ± 0.0187) matches the example README. |
| examples/sbi-benchmarks/slcp/flow_simformer | fixed (text only) | Ran clean unshrunk on CPU (2.2 min, exit=0, restore + one 10k-sample posterior). Text fixes only (wording/typo fixes from the generic template; see Template fixes). The notebook does not compute C2ST; the stored text value (0.5515 ± 0.0254) matches the example README. |
| examples/sbi-benchmarks/two_moons/flow_flux | fixed | Ran clean on CPU (15.8 min, exit=0). L-C2ST cell `pipeline.sample` on batch -> `sample_batched(...,nsamples=1)[0]`; text fixes; stale shape comment (100, 10000, 2) -> (100, 10000, 2, 1). Shrunk CPU run (100 conditions x 300 posterior samples, L-C2ST 2000, 2000-sample draws); notebook prints no C2ST/p-values (plots only), README C2ST 0.507 not recomputed; all diagnostics cells ran to completion, no gross failure visible. |
| examples/sbi-benchmarks/two_moons/flow_flux1joint | fixed | Ran clean on CPU (~15 min, exit=0). Same fixes as flow_flux (L-C2ST `sample_batched`, shape comment, text fixes). Library UserWarning `x_o has batch dimension 2 > 1` from Flux1Joint `sample()` (false positive: x_o is (dim_cond=2, ch)); results unaffected. Shrunk CPU run (100 conditions x 300 posterior samples, L-C2ST 2000, 2000-sample draws); notebook prints no C2ST/p-values (plots only), README C2ST 0.504 not recomputed; all diagnostics cells ran to completion, no gross failure visible. |
| examples/sbi-benchmarks/two_moons/flow_simformer | fixed | Ran clean on CPU (~7 min, exit=0). Same fixes as flow_flux; same x_o warning from Simformer `sample()`. Notebook writes untracked `animated_plot_samples.gif` while tracked gifs are named `*_simformer.gif` (pre-existing naming quirk, reverted/deleted after the run). Shrunk CPU run (100 conditions x 300 posterior samples, L-C2ST 2000, 2000-sample draws); notebook prints no C2ST/p-values (plots only), README C2ST 0.510 not recomputed; all diagnostics cells ran to completion, no gross failure visible. |
| examples/sbi-benchmarks/two_moons/diffusion_flux | fixed | Ran clean on CPU (9.5 min, exit=0). Same fixes as flow_flux plus: `time_grid=` is ignored by score-matching samplers and the run crashed (slice TypeError in the 5.1 sample cell); now `nsteps=100, return_intermediates=True` via new placeholder `{intermediates_kwargs}`. Shrunk CPU run (100 conditions x 300 posterior samples, L-C2ST 2000, 2000-sample draws); notebook prints no C2ST/p-values (plots only), README C2ST 0.544 not recomputed; all diagnostics cells ran to completion, no gross failure visible. |
| examples/sbi-benchmarks/two_moons/diffusion_flux1joint | fixed | Ran clean on CPU (10 min, exit=0). Same fixes as flow_flux incl. intermediates fix; x_o warning as above. Shrunk CPU run (100 conditions x 300 posterior samples, L-C2ST 2000, 2000-sample draws); notebook prints no C2ST/p-values (plots only), README C2ST 0.571 not recomputed; all diagnostics cells ran to completion, no gross failure visible. |
| examples/sbi-benchmarks/two_moons/diffusion_simformer | fixed | Ran clean on CPU (4 min, exit=0). Same fixes as flow_flux incl. intermediates fix; x_o warning as above. Shrunk CPU run (100 conditions x 300 posterior samples, L-C2ST 2000, 2000-sample draws); notebook prints no C2ST/p-values (plots only), README C2ST 0.617 not recomputed; all diagnostics cells ran to completion, no gross failure visible. |
| examples/NDE/flow_matching_2d_unconditional_flux1joint | ran clean (flags flipped: restore_model=True, train_model=False) | Checkpoint restore worked (`Restored model from checkpoint`, 185201 params); sampling ran. Text: 'velocity field model (an MLP)' -> Flux1Joint. Shrunk run (nsamples 100k->20k), CPU. |
| examples/NDE/diffusion_EDM_2d_unconditional | ran clean | Shrunk sampling nsamples 500k->20k; 3 PNG links added; PNGs reverted. |
| examples/NDE/diffusion_SM_2d_unconditional | ran clean | Shrunk sampling nsamples 500k->20k (nsteps 251/500 kept); 3 PNG links added. |
| examples/NDE/flow_matching_2d_unconditional | ran clean | Shrunk sampling nsamples 500k->20k; 3 PNG links added. |
| examples/NDE/gensbi_text | fixed | Stale API: `UnconditionalFlowPipeline` no longer exists -> `UnconditionalPipeline` + `FlowMatchingMethod()` (method=...); dropped unused Flux1Joint import. Shrunk nsamples 500k->20k; gensbi_animation.gif reverted. |
| examples/sbi-benchmarks/two_moons/maf_NLE | ran clean | Unshrunk CPU run, 6.6 min (SMC 20000 particles, NS num_live 2000). SMC log_evidence -0.685 (text says ~-0.70), NS -0.573 +- 0.048 (text: -0.57 +- 0.05), unique 10000/10000. No text issues. |
| examples/sbi-benchmarks/slcp/tarflow_NLE | ran clean (shrunk) | Unshrunk SMC (20000 particles, 20 MCMC steps) did not finish in >30 min on 6 cores; shrunk CPU run (num_particles 20000->500, num_mcmc_steps 20->8, literal override), 7.6 min. Typos fixed: Excercise/excercise, 'is in generally slower'. Markdown C2ST 0.518 not recomputed. |
| examples/sbi-benchmarks/lensing | fixed | savefig `img/` -> `imgs/`; stale API: pipeline sampler no longer forwards `encoder_key` kwarg to the model (TraceContextError) -> 2026-10-26: the key is passed explicitly as `model_extras={"encoder_key": PRNGKey(1234)}` on every sampling call (needs GenSBI branch `sampler-kwargs-fixes`, > 0.5.1, where `sample_batched` forwards `model_extras`); no fallback in LensingModel. Typos: excercise, 'the the', drawan, `$\\theta$`. Shrunk 100k->5k samples; 4.3 min. lensing.png reverted. |
| examples/sbi-benchmarks/gravitational_waves | fixed | Config path `gw_config_6c_1.yaml` -> `gw_config_6c.yaml`; same encoder_key fix as lensing (GWModel); the cell introducing the pipeline now says `ConditionalPipeline` (with `FlowMatchingMethod`) instead of the removed `ConditionalFlowPipeline`; missing imports run_sbc/sbc_rank_plot/run_tarp/plot_tarp added; run_tarp now returns TARPResult (`tarp_results = run_tarp(...)`, `plot_tarp(tarp_results)`); typo 'is note'. Shrunk 100k->2k samples, diagnostics 200 conds x 1000 samples -> 20 x 60 (diagnostic sampling is very slow on CPU); 18 min. Diagnostics from a shrunk CPU run are not meaningful. gw_example.png reverted. |

## Template fixes (generic)

Applied identically to `scripts/notebook_template.md` and the 9 wips (all nine notebooks). Each regenerated notebook was verified identical to the working copy that ran (9 of 9).

1. TOC row 5: "Visualize loss, sample posterior, compute log prob" -> "Sample the posterior and compare it with the reference samples".
2. Reference plot comment: "# plot the 2D posterior" -> "# plot the reference posterior".
3. Section 3 intro: removed the Simformer-only sentence "Edge masks and marginalization functions ...".
4. Section 5 intro: "trained Simformer model" -> "trained model".
5. Typo: "ocmpute" -> "compute".
6. Typo: "excercise" -> "exercise".
7. Section 5.2 intro: "as a 2D histogram" -> "as 1D and 2D marginal histograms" (the cell calls `plot_marginals`).
8. Whitespace: non-breaking space in the "# alternatively use corner" comment replaced by a plain space.

New placeholders `{use_ema_default}` (gaussian_mixture/flow_flux1joint: True, others False) and `{plot_marginals_extra}` (gaussian_mixture/flow_flux1joint: `, true_param=..., range=...`, others empty).

## Template fixes (two_moons)

Applied to `scripts/two_moons_template.md` and the 6 working copies; each regenerated notebook was verified identical to the working copy that ran (6 of 6). Same text fixes 1, 3-7 as the generic template (TOC row 5, Section 3 intro, Section 5 intro, `ocmpute`, `excercise`, 5.2 intro). The Section 5 intro also dropped the two sentences about computing and visualizing the unnormalized log probability over a grid (the two_moons notebooks do not do that). Plus: L-C2ST `pipeline.sample_batched(..., nsamples=1)[0]`; stale `samples.shape` comment; new placeholder `{intermediates_kwargs}` (flow: `time_grid=jnp.linspace(0,1,100)`, diffusion: `nsteps=100, return_intermediates=True`).

## Flagged results

None. Every notebook ran to completion (all rows are "ran clean" or "fixed"). Caveats: the gw diagnostics come from a heavily shrunk run and are not meaningful (20 conditions x 60 samples); `slcp/tarflow_NLE` was run with a much smaller SMC (500 particles, 8 MCMC steps) because the real setting did not finish in >30 min on 6 cores; the two_moons notebooks print no C2ST, so README numbers were not recomputed.

## Skipped

None. No notebook was skipped.

## Library findings (GenSBI, not changed here)

1. `JointPipeline.sample` (`recipes/joint_pipeline.py:446-454`) warns "x_o has batch dimension N > 1" for a 1-D `(dim_cond,)` `x_o`. This is a false positive (the shape is expanded to `(1, dim_cond, 1)` afterwards); results are unaffected.
2. `sample_batched` replaces the caller's `model_extras` per condition (`recipes/pipeline.py:1390-1403`) and `build_sampler_fn` silently accepts unknown kwargs (`flow_matching.py:219`), so `encoder_key` is dropped and the model raises a TraceContextError. **2026-10-26 update:** fixed in GenSBI branch `sampler-kwargs-fixes` (`sample_batched` forwards `model_extras`; unknown sampler kwargs raise `TypeError`). The gw/lensing notebooks and `train-gw.py`/`train-lensing.py` now pass `model_extras={"encoder_key": jax.random.PRNGKey(1234)}` to every `sample`/`sample_batched` call; the wrapper-default and `inference_encoder_seed` workarounds were removed.

## Not done / for the maintainer

- Fixed in a follow-up (see `2026-10-26-notebook-recheck-handout.md`): Colab badges for `methods_and_samplers` and the 6 `two_moons` notebooks; 3 notebooks added to the README; EDM/SM "velocity field model" wording; `gensbi_text` title/intro; gw `ConditionalFlowPipeline` comment; lensing `savefig` commented out; `train-gw.py`/`train-lensing.py`/`train_sbi_model.py`/`first_model.py` patched like the notebooks.
- `two_moons/flow_simformer` writes `animated_plot_samples.gif`, while the tracked files are named `*_simformer.gif` (pre-existing).
- Diffusion `two_moons` notebooks now sample with `nsteps=100` (library default 18), forced by the ignored `time_grid=` crash.
- The gw/lensing wrappers now differ from the training-script model classes (fixed encoder noise if copied into training).
- The L-C2ST cells (`my_first_model` and the 6 `two_moons` notebooks) now draw via `sample_batched(..., nsamples=1)`, which loops over the 10 000 conditions one device call at a time. Full-size GPU runtime is untested (shrunk runs used 2 000).
- The gw notebook's pipeline cell wording (`ConditionalFlowPipeline` -> `ConditionalPipeline` with `FlowMatchingMethod`) was fixed after the main re-check (see the gw row).

### Re-run at full size on a GPU

- `methods_and_samplers`: the GPU run stopped at the score-matching cell; the final run was shrunk on CPU.
- The 5 NDE notebooks (500k samples; shrunk to 20k on CPU).
- `gw` and `lensing` (100k samples; shrunk to 2k / 5k).
- `gw` diagnostics (200 conditions x 1000 samples; ran 20 x 60).
- `slcp/tarflow_NLE` (SMC with 20 000 particles never finished on CPU).
- The L-C2ST cells above.

## Source-of-truth files changed on this branch

Non-docs files, from `git diff --stat main...HEAD`:

- `README.md`
- `scripts/make_notebook.py`, `scripts/make_notebook_two_moons.py`, `scripts/notebook_template.md`, `scripts/two_moons_template.md`
- `examples/getting_started/my_first_model.md`, `examples/methods_and_samplers/methods_and_samplers.md`
- the 26 regenerated `.ipynb` files (outputs stripped, no checkpoints touched)
- removed: `examples/methods_and_samplers/methods_and_samplers.py` (stale copy)

## Final tree checks

```
$ ls scripts/recheck            -> No such file or directory
$ git ls-files '*.wip.py' | wc -l -> 0
$ git diff main --stat -- '**/checkpoints/**'  -> (empty)
$ outputs/execution_count over all 26 tracked .ipynb -> (no output)
$ git ls-files -o --exclude-standard -> (empty)
```

## Spot checks

Four regenerated notebooks (the real `.ipynb`, not the wip) were executed via a kernel runner script (since removed) on CPU (6 cores) with a per-notebook shrink file (since removed) applied, not via `nbconvert --execute` on GPU (this node has no GPU). Model/training config and checkpoints were not shrunk. Executed copies went to the logs dir (`<stem>.spot.executed.ipynb`); nothing was left in the repo (overwritten tracked images/GIFs were reverted after the run; committed `.ipynb` unchanged).

| Notebook | Result | Run time | Shrink |
|---|---|---|---|
| examples/sbi-benchmarks/gaussian_linear/flow_flux/gaussian_linear_flow_flux.ipynb | exit=0, all cells ran | 4m53 | none |
| examples/sbi-benchmarks/two_moons/flow_flux/two_moons_flow_flux.ipynb | exit=0, all cells ran | 15m30 | a shrink file (calibration/posterior/L-C2ST sizes) |
| examples/getting_started/my_first_model.ipynb | exit=0, all cells ran | 35m21 (shrunk; well over the ~20 min target) | a shrink file (100k->5k samples, 50x51x52 grid -> 20x21x22, 200->100 SBC/TARP conditions with 1000->300 samples, 10k->2k L-C2ST data/samples) |
| examples/NDE/flow_matching_2d_unconditional.ipynb | exit=0, all cells ran | 40 s | `nsamples=500_000` -> 20_000 |

The `KeyboardInterrupt` in the my_first_model log tail is grain worker shutdown at kernel teardown, not a cell error.
