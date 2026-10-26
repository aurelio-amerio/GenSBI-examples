# Notebook re-check: handout (2026-10-26)

Branch `notebook-recheck` (HEAD `ef962142`, 12 commits ahead of `origin/notebook-recheck`, **not pushed, not merged**).
Full per-notebook detail: `docs/superpowers/specs/2026-10-25-notebook-recheck-report.md`.
The earlier pause handout (`docs/superpowers/plans/2026-10-25-notebook-recheck-handout.md`) is superseded.

## 1. Issues solved

All 26 tracked notebooks run end to end against GenSBI 0.5.1. Fixes are in each notebook's source of truth, and every notebook was regenerated without outputs. Checkpoints are byte-identical to `main`.

**Code fixes (notebooks failed without them)**

| Where | Problem | Fix |
|---|---|---|
| `my_first_model`, 6 × `two_moons` (L-C2ST cell) | `pipeline.sample` on a batch of conditions now raises | `pipeline.sample_batched(key, xs, nsamples=1)[0]` |
| `my_first_model`, `methods_and_samplers` | exact log-prob on a ~132k-point grid OOMs on a 12 GB GPU (19.45 GiB in one solve) | evaluate in 10k-point chunks with `jnp.concatenate` |
| `my_first_model` | SBC used undefined `test_data`; missing `compute_marginal_coverage`/`plot_marginal_coverage` import; wrong Colab `%cd` path | fixed in the `.md` |
| 3 × `two_moons/diffusion_*` | `time_grid=` is silently ignored by the diffusion sampler → slice `TypeError` in the 5.1 cell | `nsteps=100, return_intermediates=True` via a new template placeholder `{intermediates_kwargs}` |
| `NDE/gensbi_text` | `UnconditionalFlowPipeline` no longer exists | `UnconditionalPipeline(..., method=FlowMatchingMethod())` |
| `gw_example`, `lensing_example` | `encoder_key` no longer reaches the model → flax `TraceContextError` | model wrappers default `encoder_key` to `PRNGKey(1234)` (same key the training scripts use at inference, so results are unchanged) |
| `gw_example` | config path pointed at deleted `gw_config_6c_1.yaml`; SBC/TARP functions not imported; `run_tarp` now returns a `TARPResult` | `gw_config_6c.yaml`; imports added; `plot_tarp(tarp_results)` |
| `lensing_example` | `savefig("img/...")` into a non-existent folder | `imgs/` |
| `NDE/flow_matching_2d_unconditional_flux1joint` | trained from scratch by default | `restore_model=True`, `train_model=False`; committed checkpoint restores |

**Templates / generators**

- `scripts/notebook_template.md` + `make_notebook.py`: the hand-patched `gaussian_mixture/flow_flux1joint` is now expressed with placeholders `{use_ema_default}` and `{plot_marginals_extra}`, so all 9 generic notebooks regenerate exactly.
- Text fixes in both templates (all variants): TOC row 5 no longer promises log-prob plots; Simformer-only wording removed from Flux1 notebooks; "2D posterior" / "2D histogram" wording for >2D tasks; typos `ocmpute`, `excercise`.

**Other**

- NDE notebooks: outputs stripped; the existing PNGs are now linked as figures (raw GitHub URLs).
- Typos and stale names fixed across notebooks (e.g. `fomulation`, `$\theta$` rendered as a tab, `drawan`, `the the`, `ConditionalFlowPipeline` in the gw markdown).
- README: "CUDA 12" → "CUDA 13".
- Removed: stale `examples/methods_and_samplers/methods_and_samplers.py`.

## 2. Not done yet

**Re-run at full size on a GPU.** After the move to the CPU-only node, runs used reduced sizes (at run time only; the committed notebooks keep the real sizes). These have never completed at full size on current GenSBI:

- `methods_and_samplers`: the GPU run stopped at the score-matching cell.
- The 5 NDE notebooks (500k samples; ran at 20k).
- `gw` and `lensing` posterior draws (100k; ran at 2k / 5k).
- `gw` diagnostics (200 conditions × 1000 samples; ran at 20 × 60, so the CPU results are not meaningful).
- `slcp/tarflow_NLE` (SMC with 20 000 particles, 20 MCMC steps; did not finish in >30 min on 6 CPU cores; ran at 500 / 8).
- The L-C2ST cells (`my_first_model` and the 6 `two_moons`): runtime of the new per-condition `sample_batched` loop over 10 000 conditions is untested (see GenSBI issue 3).

No C2ST was recomputed. The generic-template notebooks only show stored values, and the two_moons notebooks only plot.

**Left out on purpose (new content, outside this branch's scope)**

- 7 notebooks have no Colab badge: `methods_and_samplers` and the 6 `two_moons` (also missing on `main`).
- The top-level `README.md` notebook list omits some notebooks (e.g. `NDE/gensbi_text`).

**Small leftovers**

- `gw_example.ipynb` code comment (cell around source line 393) still says "the ConditionalFlowPipeline supports an arbitrary model".
- EDM and SM NDE notebooks call the model a "velocity field model" (flow-matching wording).
- `gensbi_text` title and intro are copied from the flow-matching notebook.
- `two_moons/flow_simformer` writes `animated_plot_samples.gif`, but the tracked files are `*_simformer.gif`.
- `lensing_example` now really writes `imgs/lensing.png` on every run, overwriting the tracked image (as the NDE notebooks do).
- `train-gw.py` / `train-lensing.py` still pass `encoder_key=` to `sample_batched` (see GenSBI issue 2); their evaluation sections will fail.

**Branch**

- Not pushed or merged. Next step: push and open a PR, or merge into `main` locally.

## 3. Potential issues found in GenSBI

Paths are relative to `GenSBI/src/gensbi` (v0.5.1). Nothing was changed in the library.

1. **Silent kwarg swallowing in `build_sampler_fn`.** `core/flow_matching.py:217-219`, `core/diffusion_edm.py:166-170` and `core/score_matching.py:191-200` all end in `**kwargs` that are dropped. A wrong or unsupported argument (e.g. `time_grid=` on a diffusion pipeline, `encoder_key=` passed to `sample_batched`) is ignored without a warning. The two_moons diffusion crash and the gw/lensing `TraceContextError` both trace back to this. Suggestion: raise or warn on unknown kwargs.

2. **`sample_batched` drops the caller's model extras.** `recipes/pipeline.py:1390-1403` rebuilds `model_extras` per condition as `{cond, obs_ids, cond_ids}`, so anything the user wants forwarded to the model (e.g. `encoder_key` for the VAE-embedded gw/lensing models) is lost. `get_sampler(model_extras=...)` supports it for `sample`, but not for `sample_batched`. Suggestion: merge user extras into `extras_i`.

3. **`sample_batched` loops over conditions in Python.** The same loop (`pipeline.py:1390ff`) does one device call per condition. Workflows that need one sample for each of many conditions (L-C2ST: 10 000 conditions × 1 sample) used to be one batched call through `sample`; that path now raises and the replacement is a 10 000-iteration loop. Suggestion: a vectorized path (e.g. `vmap` over conditions, or chunks of conditions) when `nsamples` is small.

4. **False-positive batch warning in `JointPipeline.sample`.** `recipes/joint_pipeline.py:446-454` takes `x_o.shape[0]` as the batch size. For a single 1-D observation of shape `(dim_cond,)` (what sbibm `get_reference` returns), it warns "x_o has batch dimension 2 > 1" even though `_expand_dims` correctly turns it into one condition `(1, dim_cond, 1)`. Results are correct; the warning is noise in every Flux1Joint/Simformer two_moons notebook. Suggestion: only warn when `x_o.ndim` ≥ the expected batched rank.

5. **API breaks without a deprecation path** (worth a changelog entry or aliases): `UnconditionalFlowPipeline` removed (now `UnconditionalPipeline` + `FlowMatchingMethod()`); `run_tarp` returns a `TARPResult` instead of a tuple; `pipeline.sample` on a batch of conditions now raises (use `sample_batched`). Any user code written against the older API fails the same way the notebooks did.
