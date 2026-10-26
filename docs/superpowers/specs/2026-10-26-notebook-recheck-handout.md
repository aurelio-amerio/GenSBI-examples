# Notebook re-check: handout (2026-10-26)

Branch `notebook-recheck` (ahead of `origin/notebook-recheck`, **not pushed, not merged**).
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
| `gw_example`, `lensing_example` | `encoder_key` no longer reaches the model → flax `TraceContextError` | **2026-10-26:** the key is passed explicitly as `model_extras={"encoder_key": PRNGKey(1234)}` on every `sample`/`sample_batched` call, with no fallback in the model wrappers (needs GenSBI branch `sampler-kwargs-fixes`, > 0.5.1, which makes `sample_batched` forward `model_extras`). The earlier wrapper-default workaround was removed. |
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

`scripts/train_sbi_model.py` and `examples/getting_started/first_model.py` got the same L-C2ST `sample_batched` fix as the notebooks but were not run here (the identical call ran in the notebooks).

**Done after the first version of this handout** (follow-up commit)

- Colab badges added to the 7 notebooks that had none: `methods_and_samplers` (in the `.md`) and the 6 `two_moons` (in `scripts/two_moons_template.md`, then regenerated).
- `README.md` now lists `methods_and_samplers`, `NDE/gensbi_text` and `two_moons/maf_NLE`.
- Leftovers fixed: the gw code comment now says `ConditionalPipeline`; the EDM/SM NDE notebooks say "network" instead of "velocity field model" and have distinct titles; `gensbi_text` has its own title, intro and Section 2 text; the lensing `savefig("imgs/lensing.png")` is commented out ("uncomment to save the figure"), so runs no longer overwrite the tracked image.
- Training scripts patched to match the notebooks:
  - `train-gw.py`, `train-lensing.py`: **2026-10-26:** same as the notebooks — `model_extras={"encoder_key": jax.random.PRNGKey(1234)}` on every sampling call; the interim `inference_encoder_seed` attribute was removed. Training noise is unchanged (the VAE still draws from its own RNG state when no key is passed).
  - `scripts/train_sbi_model.py`, `examples/getting_started/first_model.py`: L-C2ST `pipeline.sample` on 10 000 conditions → `sample_batched(..., nsamples=1)[0]`.
- Not changed, on purpose: `two_moons/flow_simformer` writes `animated_plot_samples.gif` while the tracked GIFs are `*_simformer.gif`. Those are hand-made assets linked from GenSBI's README, not run output.

**Branch**

- Not pushed or merged. Next step: push and open a PR, or merge into `main` locally.

## 3. Potential issues found in GenSBI

Paths are relative to `GenSBI/src/gensbi` (v0.5.1). Status as of 2026-10-26: 1, 2, 4 and 5 addressed on GenSBI branch `sampler-kwargs-fixes`; 3 has a design handout. The gw/lensing examples now require that branch (or the next GenSBI release).

1. **Silent kwarg swallowing in `build_sampler_fn`.** `core/flow_matching.py:217-219`, `core/diffusion_edm.py:166-170` and `core/score_matching.py:191-200` all end in `**kwargs` that are dropped. A wrong or unsupported argument (e.g. `time_grid=` on a diffusion pipeline, `encoder_key=` passed to `sample_batched`) is ignored without a warning. The two_moons diffusion crash and the gw/lensing `TraceContextError` both trace back to this. Suggestion: raise or warn on unknown kwargs. **FIXED 2026-10-26** on GenSBI branch `sampler-kwargs-fixes` (b95dd165): the concrete builders enumerate their arguments; an unknown keyword raises `TypeError`.

2. **`sample_batched` drops the caller's model extras.** `recipes/pipeline.py:1390-1403` rebuilds `model_extras` per condition as `{cond, obs_ids, cond_ids}`, so anything the user wants forwarded to the model (e.g. `encoder_key` for the VAE-embedded gw/lensing models) is lost. `get_sampler(model_extras=...)` supports it for `sample`, but not for `sample_batched`. Suggestion: merge user extras into `extras_i`. **FIXED 2026-10-26** on the same branch: `model_extras` is merged into every per-condition call.

3. **`sample_batched` loops over conditions in Python.** The same loop (`pipeline.py:1390ff`) does one device call per condition. Workflows that need one sample for each of many conditions (L-C2ST: 10 000 conditions × 1 sample) used to be one batched call through `sample`; that path now raises and the replacement is a 10 000-iteration loop. Suggestion: a vectorized path (e.g. `vmap` over conditions, or chunks of conditions) when `nsamples` is small. **OPEN — design handout written:** `GenSBI/docs/superpowers/specs/2026-10-26-sample-batched-vectorized-handout.md`.

4. **False-positive batch warning in `JointPipeline.sample`.** `recipes/joint_pipeline.py:446-454` takes `x_o.shape[0]` as the batch size. For a single 1-D observation of shape `(dim_cond,)` (what sbibm `get_reference` returns), it warns "x_o has batch dimension 2 > 1" even though `_expand_dims` correctly turns it into one condition `(1, dim_cond, 1)`. Results are correct; the warning is noise in every Flux1Joint/Simformer two_moons notebook. Suggestion: only warn when `x_o.ndim` ≥ the expected batched rank. **FIXED 2026-10-26** on the same branch: the check runs on the canonical `(batch, dim, ch)` shape.

5. **API breaks without a deprecation path** (worth a changelog entry or aliases): `UnconditionalFlowPipeline` removed (now `UnconditionalPipeline` + `FlowMatchingMethod()`); `run_tarp` returns a `TARPResult` instead of a tuple; `pipeline.sample` on a batch of conditions now raises (use `sample_batched`). Any user code written against the older API fails the same way the notebooks did. **2026-10-26:** `GenSBI/CHANGELOG.md` added with a migration table; no aliases on purpose (all three breaks shipped in 0.1–0.4, the new interfaces are the supported ones).
