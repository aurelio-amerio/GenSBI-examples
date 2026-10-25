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

## Flagged results

## Skipped
