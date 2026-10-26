# Example notebooks re-check — report

Date: 2026-10-25. Design: `2026-10-25-notebook-recheck-design.md`.

All notebooks run with `/home/aure/github/GenSBI/.venv` (GenSBI 0.5.1), `MPLBACKEND=Agg`, from the notebook's own directory, via jupytext py:percent working copies.

## Per notebook

| Notebook | Status | Notes |
|---|---|---|
| getting_started/my_first_model | fixed | Regenerated from the newer `.md`. Fixed: SBC cell used undefined `test_data` (was assigned to `test_data_`); missing import of `compute_marginal_coverage`/`plot_marginal_coverage`; L-C2ST drew one sample per condition with `pipeline.sample` on a batch of 10k conditions, which current GenSBI rejects → `sample_batched(key, xs_, nsamples=1)[0]`; log-prob grid (132k points) OOMed on a 12 GB GPU (19.45 GiB in one solve) → evaluated in 10k-point chunks; Colab `%cd` path had a stray `/GenSBI-examples`; `$\theta$` rendered as a tab (3×); "fomulation"; TARP expansion ("Tests of Accuracy with Random Points"); `# (3,)` shape comment; 6 `savefig` lines labelled "uncomment to save" were active → commented out. Diagnostics: SBC KS p-values 0.18/0.70/0.37, TARP ATC −0.21 with KS p = 1.0. |
| methods_and_samplers | fixed | Regenerated from `.md`. Fixed: exact FM log-prob on the ~132k-point grid OOMed on a 12 GB GPU -> evaluated in 10k-point chunks with `jnp.concatenate`; missing period in an EDM comment. All FM, EDM and SM cells ran clean (checkpoints flow/edm/sm restored, exit=0) on a shrunk CPU run (5k samples, small log-prob grids, ~30 min); no diagnostics computed. |
| sbi-benchmarks/gaussian_linear/flow_flux | fixed (text only) | Ran clean unshrunk on CPU (5.3 min, exit=0, restore + one 10k-sample posterior). Text fixes only (see Task 4 report). The notebook does not compute C2ST; the stored text value (0.5087 ± 0.0061) matches the example README. |
| sbi-benchmarks/gaussian_linear/flow_flux1joint | fixed (text only) | Ran clean unshrunk on CPU (13.3 min, exit=0, restore + one 10k-sample posterior). Text fixes only (see Task 4 report). The notebook does not compute C2ST; the stored text value (0.5066 ± 0.0040) matches the example README. |
| sbi-benchmarks/bernoulli_glm/flow_flux | fixed (text only) | Ran clean unshrunk on CPU (8.6 min, exit=0, restore + one 10k-sample posterior). Text fixes only (see Task 4 report). The notebook does not compute C2ST; the stored text value (0.5594 ± 0.0143) matches the example README. |
| sbi-benchmarks/bernoulli_glm/flow_flux1joint | fixed (text only) | Ran clean unshrunk on CPU (13.2 min, exit=0, restore + one 10k-sample posterior). Text fixes only (see Task 4 report). The notebook does not compute C2ST; the stored text value (0.5574 ± 0.0197) matches the example README. |
| sbi-benchmarks/gaussian_mixture/flow_flux | fixed (text only) | Ran clean unshrunk on CPU (2.6 min, exit=0, restore + one 10k-sample posterior). Text fixes only (see Task 4 report). The notebook does not compute C2ST; the stored text value (0.5087 ± 0.0072) matches the example README. |
| sbi-benchmarks/gaussian_mixture/flow_flux1joint | fixed (text only) | Ran clean unshrunk on CPU (3.1 min, exit=0, restore + one 10k-sample posterior). Text fixes only (see Task 4 report). The notebook does not compute C2ST; the stored text value (0.5014 ± 0.0038) matches the example README. |
| sbi-benchmarks/slcp/flow_flux | fixed (text only) | Ran clean unshrunk on CPU (9.4 min, exit=0, restore + one 10k-sample posterior). Text fixes only (see Task 4 report). The notebook does not compute C2ST; the stored text value (0.6886 ± 0.0472) matches the example README. |
| sbi-benchmarks/slcp/flow_flux1joint | fixed (text only) | Ran clean unshrunk on CPU (14.5 min, exit=0, restore + one 10k-sample posterior). Text fixes only (see Task 4 report). The notebook does not compute C2ST; the stored text value (0.5490 ± 0.0187) matches the example README. |
| sbi-benchmarks/slcp/flow_simformer | fixed (text only) | Ran clean unshrunk on CPU (2.2 min, exit=0, restore + one 10k-sample posterior). Text fixes only (see Task 4 report). The notebook does not compute C2ST; the stored text value (0.5515 ± 0.0254) matches the example README. |
| sbi-benchmarks/two_moons/flow_flux | fixed | Ran clean on CPU (15.8 min, exit=0). L-C2ST cell `pipeline.sample` on batch -> `sample_batched(...,nsamples=1)[0]`; text fixes; stale shape comment (100, 10000, 2) -> (100, 10000, 2, 1). Shrunk CPU run (100 conditions x 300 posterior samples, L-C2ST 2000, 2000-sample draws); notebook prints no C2ST/p-values (plots only), README C2ST 0.507 not recomputed; all diagnostics cells ran to completion, no gross failure visible. |
| sbi-benchmarks/two_moons/flow_flux1joint | fixed | Ran clean on CPU (~15 min, exit=0). Same fixes. Library UserWarning `x_o has batch dimension 2 > 1` from Flux1Joint `sample()` (false positive: x_o is (dim_cond=2, ch)); results unaffected. Shrunk CPU run (100 conditions x 300 posterior samples, L-C2ST 2000, 2000-sample draws); notebook prints no C2ST/p-values (plots only), README C2ST 0.504 not recomputed; all diagnostics cells ran to completion, no gross failure visible. |
| sbi-benchmarks/two_moons/flow_simformer | fixed | Ran clean on CPU (~7 min, exit=0). Same fixes; same x_o warning from Simformer `sample()`. Notebook writes untracked `animated_plot_samples.gif` while tracked gifs are named `*_simformer.gif` (pre-existing naming quirk, deleted in P2). Shrunk CPU run (100 conditions x 300 posterior samples, L-C2ST 2000, 2000-sample draws); notebook prints no C2ST/p-values (plots only), README C2ST 0.510 not recomputed; all diagnostics cells ran to completion, no gross failure visible. |
| sbi-benchmarks/two_moons/diffusion_flux | fixed | Ran clean on CPU (9.5 min, exit=0). Same fixes plus: `time_grid=` is ignored by score-matching samplers and the run crashed (slice TypeError in the 5.1 sample cell); now `nsteps=100, return_intermediates=True` via new placeholder `{intermediates_kwargs}`. Shrunk CPU run (100 conditions x 300 posterior samples, L-C2ST 2000, 2000-sample draws); notebook prints no C2ST/p-values (plots only), README C2ST 0.544 not recomputed; all diagnostics cells ran to completion, no gross failure visible. |
| sbi-benchmarks/two_moons/diffusion_flux1joint | fixed | Ran clean on CPU (10 min, exit=0). Same fixes incl. intermediates fix; x_o warning as above. Shrunk CPU run (100 conditions x 300 posterior samples, L-C2ST 2000, 2000-sample draws); notebook prints no C2ST/p-values (plots only), README C2ST 0.571 not recomputed; all diagnostics cells ran to completion, no gross failure visible. |
| sbi-benchmarks/two_moons/diffusion_simformer | fixed | Ran clean on CPU (4 min, exit=0). Same fixes incl. intermediates fix; x_o warning as above. Shrunk CPU run (100 conditions x 300 posterior samples, L-C2ST 2000, 2000-sample draws); notebook prints no C2ST/p-values (plots only), README C2ST 0.617 not recomputed; all diagnostics cells ran to completion, no gross failure visible. |
| examples/NDE/flow_matching_2d_unconditional_flux1joint | ran clean (flags flipped: restore_model=True, train_model=False) | Checkpoint restore worked (`Restored model from checkpoint`, 185201 params); sampling ran. Text: 'velocity field model (an MLP)' -> Flux1Joint. Shrunk run (nsamples 100k->20k), CPU. |
| examples/NDE/diffusion_EDM_2d_unconditional | ran clean | Shrunk sampling nsamples 500k->20k; 3 PNG links added; PNGs reverted. |
| examples/NDE/diffusion_SM_2d_unconditional | ran clean | Shrunk sampling nsamples 500k->20k (nsteps 251/500 kept); 3 PNG links added. |
| examples/NDE/flow_matching_2d_unconditional | ran clean | Shrunk sampling nsamples 500k->20k; 3 PNG links added. |
| examples/NDE/gensbi_text | fixed | Stale API: `UnconditionalFlowPipeline` no longer exists -> `UnconditionalPipeline` + `FlowMatchingMethod()` (method=...); dropped unused Flux1Joint import. Shrunk nsamples 500k->20k; gensbi_animation.gif reverted. |
| examples/sbi-benchmarks/two_moons/maf_NLE | ran clean | Unshrunk CPU run, 6.6 min (SMC 20000 particles, NS num_live 2000). SMC log_evidence -0.685 (text says ~-0.70), NS -0.573 +- 0.048 (text: -0.57 +- 0.05), unique 10000/10000. No text issues. |
| examples/sbi-benchmarks/slcp/tarflow_NLE | ran clean (shrunk) | Unshrunk SMC (20000 particles, 20 MCMC steps) did not finish in >30 min on 6 cores; shrunk CPU run (num_particles 20000->500, num_mcmc_steps 20->8, literal override), 7.6 min. Typos fixed: Excercise/excercise, 'is in generally slower'. Markdown C2ST 0.518 not recomputed. |
| examples/sbi-benchmarks/lensing | fixed | savefig `img/` -> `imgs/`; stale API: pipeline sampler no longer forwards `encoder_key` kwarg to the model (TraceContextError) -> LensingModel falls back to fixed PRNGKey(1234) when encoder_key is None, kwarg dropped from sample_batched call. Typos: excercise, 'the the', drawan, `$\\theta$`. Shrunk 100k->5k samples; 4.3 min. lensing.png reverted. |
| examples/sbi-benchmarks/gravitational_waves | fixed | Config path `gw_config_6c_1.yaml` -> `gw_config_6c.yaml`; same encoder_key fix as lensing (GWModel); missing imports run_sbc/sbc_rank_plot/run_tarp/plot_tarp added; run_tarp now returns TARPResult (`tarp_results = run_tarp(...)`, `plot_tarp(tarp_results)`); typo 'is note'. Shrunk 100k->2k samples, diagnostics 200 conds x 1000 samples -> 20 x 60 (diagnostic sampling is very slow on CPU); 18 min. Diagnostics from a shrunk CPU run are not meaningful. gw_example.png reverted. |

## Template fixes (generic)

Applied identically to `scripts/notebook_template.md` and the 9 wips (all tasks). Regenerated notebooks match the wips (`verify_regen.py`: 9 x OK).

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

Applied to `scripts/two_moons_template.md` and the 6 wips; regenerated notebooks match the wips (6 x OK). Same text fixes 1, 3-7 as the generic template (TOC row 5, Section 3 intro, Section 5 intro, `ocmpute`, `excercise`, 5.2 intro). Plus: L-C2ST `pipeline.sample_batched(..., nsamples=1)[0]`; stale `samples.shape` comment; new placeholder `{intermediates_kwargs}` (flow: `time_grid=jnp.linspace(0,1,100)`, diffusion: `nsteps=100, return_intermediates=True`).

## Flagged results

## Skipped
