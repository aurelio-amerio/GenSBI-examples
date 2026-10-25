# Example notebooks re-check — report

Date: 2026-10-25. Design: `2026-10-25-notebook-recheck-design.md`.

All notebooks run with `/home/aure/github/GenSBI/.venv` (GenSBI 0.5.1), `MPLBACKEND=Agg`, from the notebook's own directory, via jupytext py:percent working copies.

## Per notebook

| Notebook | Status | Notes |
|---|---|---|
| getting_started/my_first_model | fixed | Regenerated from the newer `.md`. Fixed: SBC cell used undefined `test_data` (was assigned to `test_data_`); missing import of `compute_marginal_coverage`/`plot_marginal_coverage`; L-C2ST drew one sample per condition with `pipeline.sample` on a batch of 10k conditions, which current GenSBI rejects → `sample_batched(key, xs_, nsamples=1)[0]`; log-prob grid (132k points) OOMed on a 12 GB GPU (19.45 GiB in one solve) → evaluated in 10k-point chunks; Colab `%cd` path had a stray `/GenSBI-examples`; `$\theta$` rendered as a tab (3×); "fomulation"; TARP expansion ("Tests of Accuracy with Random Points"); `# (3,)` shape comment; 6 `savefig` lines labelled "uncomment to save" were active → commented out. Diagnostics: SBC KS p-values 0.18/0.70/0.37, TARP ATC −0.21 with KS p = 1.0. |
| methods_and_samplers | fixed | Regenerated from `.md`. Fixed: exact FM log-prob on the ~132k-point grid OOMed on a 12 GB GPU -> evaluated in 10k-point chunks with `jnp.concatenate`; missing period in an EDM comment. All FM, EDM and SM cells ran clean (checkpoints flow/edm/sm restored, exit=0) on a shrunk CPU run (5k samples, small log-prob grids, ~30 min); no diagnostics computed. |

## Flagged results

## Skipped
