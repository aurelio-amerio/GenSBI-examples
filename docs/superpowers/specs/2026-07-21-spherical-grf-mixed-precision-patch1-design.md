# Spherical GRF: mixed-precision + patch_size=1 run (experiment 2)

**Date:** 2026-07-21
**Status:** approved (option A)
**Files:** `examples/sbi-benchmarks/spherical_grf/{config/config_healpix.yaml, train-spherical-grf.py}`

## Motivation

Experiment 1 (48 cond tokens, fp32 everywhere) produced an underconfident
posterior. To make the model more expressive, both upstream repos gained new
features, already committed on their local `main`s:

- **GenSBI** — mixed-precision Flux1: fp32 master weights (`param_dtype`),
  bf16 compute (`dtype`), fp32 islands (norms, softmax, embeddings, final
  projection), fp32 EMA/optimizer. The YAML `model:` section's
  `param_dtype`/`dtype` keys are parsed by `parse_flux1_params`.
- **HEAL-SWIN-nnx** — same compute-dtype contract threaded through HealSwin
  (defaults now `param_dtype="float32"`, `dtype="bfloat16"`), plus
  `patch_size=1` support (no pixel regrouping at the patch embed).

Experiment 2 uses both: a patch-1, embed-64, window-64 encoder giving a
16x-denser conditioning (768 tokens x 512 features vs 48 x 512), and a deeper
Flux1 (`depth_single_blocks` 4 -> 8), trained bf16-compute/fp32-weights.

## Problems found in the current working tree

1. **Inverted precision knob** — `model.param_dtype: "bfloat16"` in
   `config_healpix.yaml` stores master weights in bf16: it recreates the
   EMA-underflow bug the mixed-precision branch exists to kill and trips the
   pipeline's `_warn_if_not_fp32_master_weights` guard.
2. **`encoder.patch_size: 1` silently ignored** — `make_encoder_params()`
   never passes `patch_size` (nor dtypes), so `HealSwinParams` defaults to
   `patch_size=4`. The script's hardcoded bottleneck formula
   `NSIDE // (2 * 2**(len(DEPTHS)-1))` happens to agree with the patch-4
   encoder (both give 192 tokens), so the run trains without error — but not
   the configured model. With patch 1 the true bottleneck is nside 8 -> 768
   tokens.
3. **Stale eval-recovery flags** — `train_model: false, restore_model: true`
   with the new `run_name: ..._2` would try to restore a nonexistent
   checkpoint and crash without training.

Already fine, no change needed: the dataset collate casts to float32
unconditionally (`sbibm_jax/data/process.py`), and Flux1 casts inputs to fp32
at the door. The explicit `dtype=np.float32` below is self-documentation only.

## Design (approach A: generalize the formula in the script)

Chosen over deriving geometry from a constructed `HealSwinParams` (option B):
the script already derives `COND_FEATURES` from the same config block, the fix
is one line, and `HealSwinParams.__post_init__` validates divisibility anyway,
so an inconsistent config still fails loudly.

### Config changes — `config/config_healpix.yaml`

1. `model.param_dtype: "float32"`; add explicit `model.dtype: "bfloat16"`.
2. Add `encoder.param_dtype: "float32"` and `encoder.dtype: "bfloat16"`
   (matches heal-swin defaults; explicit and greppable).
3. `training.train_model: true`, `training.restore_model: false`; drop the
   stale eval-recovery comments (fresh training run, experiment 2).

### Script changes — `train-spherical-grf.py`

4. `PATCH_SIZE = _ENC.get("patch_size", 4)` (default keeps
   `config_pos1d.yaml` and the archived config working) and
   `NSIDE_BOTTLENECK = NSIDE // (isqrt(PATCH_SIZE) * 2**(len(DEPTHS)-1))`
   -> 768 cond tokens, nside-8 `HealpixRope`, correct derived `theta`.
   Update the docstring's "48 bottleneck tokens" and the `(B, 48, 512)`
   comment: geometry is config-dependent.
5. `make_encoder_params()` additionally passes
   `patch_size=PATCH_SIZE`,
   `dtype=_ENC.get("dtype", "bfloat16")`,
   `param_dtype=_ENC.get("param_dtype", "float32")`.
6. `TaskDataset(..., dtype=np.float32)` explicit.
7. The startup log line gains `patch=...` and both dtype pairs so run logs
   record the precision scheme.

### Out of scope

No changes to GenSBI, HEAL-SWIN-nnx, or sbibm-jax; no changes to
`config_pos1d.yaml` beyond continuing to work via the defaults. The untracked
`config_healpix_1.yaml` is committed as-is as the experiment-1 archive.

## Verification

- `SMOKE=1 JAX_PLATFORMS=cpu python train-spherical-grf.py` — forward-shape
  check must pass with the new geometry (768 cond tokens).
- `QUICK=1 JAX_PLATFORMS=cpu python train-spherical-grf.py` — tiny end-to-end
  run (train, sample, TARP) must complete.
- No fp32-master-weights warning from the pipeline at startup.

## Final step: submit the training job

After implementation and verification, submit the GPU run (the sub file
already targets `config_healpix.yaml` on an A100). HTCondor may resolve
relative paths in the sub file against the submission directory, so first
verify path resolution with a dry run, submitting from the sub file's own
directory:

    cd examples/sbi-benchmarks/spherical_grf/sub
    condor_submit -dry-run /dev/stdout spherical_grf.sub  # inspect resolved
                                                          # log/output/error
                                                          # and argument paths
    condor_submit spherical_grf.sub

If the dry run shows the `condor_logs/...` or executable paths resolving
wrong, fix the submission directory (or the sub file) before the real
submission.
