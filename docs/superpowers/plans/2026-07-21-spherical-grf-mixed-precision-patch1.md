# Spherical GRF Mixed-Precision + patch_size=1 (Experiment 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the experiment-2 config/script so the configured model (patch-1 HealSwin encoder, 768×512 conditioning, fp32-master-weight/bf16-compute mixed precision) is actually what trains, then submit the HTCondor GPU job.

**Architecture:** Config-and-script change only. `config_healpix.yaml` gets the correct precision knobs and fresh-run flags; `train-spherical-grf.py` threads `patch_size`/`dtype`/`param_dtype` into `HealSwinParams`, generalizes the bottleneck-geometry formula with a `sqrt(patch_size)` factor, and makes the dataset float32 cast explicit. No changes to GenSBI, HEAL-SWIN-nnx, or sbibm-jax (their features are already merged on local `main`s).

**Tech Stack:** JAX / Flax NNX, gensbi (local main), heal-swin-nnx (local main), sbibm-jax, HTCondor.

**Spec:** `docs/superpowers/specs/2026-07-21-spherical-grf-mixed-precision-patch1-design.md`

## Global Constraints

- Repo root: `/lustre/ific.uv.es/ml/ific088/github/GenSBI-examples`; example dir: `examples/sbi-benchmarks/spherical_grf/` (all paths below relative to repo root unless absolute).
- Python env: `conda activate /lhome/ific/a/aamerio/miniforge3/envs/gensbi` (has editable installs of the local gensbi + heal-swin-nnx mains).
- Precision contract everywhere: `param_dtype="float32"` (master weights), `dtype="bfloat16"` (compute). Never the reverse.
- `config_pos1d.yaml` must keep working unmodified → every new config key read in the script uses `.get(key, default)` with defaults `patch_size=4`, `dtype="bfloat16"`, `param_dtype="float32"`.
- CPU-only for all verification runs: prefix commands with `JAX_PLATFORMS=cpu`.
- Do not modify the upstream repos (`/lhome/ific/a/aamerio/data/github/GenSBI`, `.../HEAL-SWIN-nnx`, `.../sbibm-jax`).
- The working tree has pre-existing uncommitted changes from the user (config, script, `sub/run_spherical_grf.sh`). Do not revert them; the tasks below commit them piecewise.
- Commit messages end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Environment sanity check + archive the experiment-1 config

**Files:**
- Commit as-is (untracked): `examples/sbi-benchmarks/spherical_grf/config/config_healpix_1.yaml`

**Interfaces:**
- Produces: confirmation that the env exposes the new upstream features (`HealSwinParams(patch_size=1, dtype=..., param_dtype=...)`, `parse_flux1_params` returning a `dtype` key). Every later task assumes this.

- [ ] **Step 1: Verify the env has the new upstream features**

```bash
cd /lustre/ific.uv.es/ml/ific088/github/GenSBI-examples
JAX_PLATFORMS=cpu python -c "
from gensbi.models import HealSwinParams
p = HealSwinParams(nside=64, in_channels=1, out_channels=1, embed_dim=64,
                   depths=(2, 2, 6, 2), num_heads=(4, 8, 16, 16),
                   window_size=64, patch_size=1,
                   dtype='bfloat16', param_dtype='float32')
print('healswin ok:', p.patch_size, p.dtype, p.param_dtype)
import inspect
from gensbi.recipes.flux1 import parse_flux1_params
src = inspect.getsource(parse_flux1_params)
assert 'dtype=getattr' in src, 'flux1 recipe lacks the dtype knob'
print('flux1 recipe ok')
"
```

Expected output includes `healswin ok: 1 bfloat16 float32` and `flux1 recipe ok`. (The dtype fields are canonicalized by `HealSwinParams.__post_init__`; printing them may show `<class 'ml_dtypes.bfloat16'>`-style reprs — any bfloat16/float32 spelling is fine.)

**If this fails**, the env doesn't have the new local mains installed — STOP and report; installing the upstream packages is out of scope.

- [ ] **Step 2: Commit the archived experiment-1 config verbatim**

`config/config_healpix_1.yaml` is the user's archive of the run-1 config. Do not edit it.

```bash
git add examples/sbi-benchmarks/spherical_grf/config/config_healpix_1.yaml
git commit -m "chore: archive experiment-1 healpix config as config_healpix_1.yaml

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Fix `config_healpix.yaml` (precision knobs + fresh-run flags)

**Files:**
- Modify: `examples/sbi-benchmarks/spherical_grf/config/config_healpix.yaml`

**Interfaces:**
- Produces: YAML keys read by Task 3 — `encoder.patch_size` (int), `encoder.dtype` / `encoder.param_dtype` (strings), `model.dtype` / `model.param_dtype` (strings), `training.train_model: true`, `training.restore_model: false`.

- [ ] **Step 1: Edit the encoder section**

Replace:

```yaml
# HealSwin encoder
encoder:
  embed_dim: 64
  depths: [2, 2, 6, 2]
  num_heads: [4, 8, 16, 16]
  window_size: 64
  patch_size: 1
```

with:

```yaml
# HealSwin encoder: patch_size 1 keeps the nside-64 grid, then 3 mergings
# 64 -> 32 -> 16 -> 8 (768 bottleneck tokens x 512 features).
encoder:
  embed_dim: 64
  depths: [2, 2, 6, 2]
  num_heads: [4, 8, 16, 16]
  window_size: 64
  patch_size: 1
  param_dtype: "float32"   # fp32 master weights
  dtype: "bfloat16"        # bf16 compute
```

- [ ] **Step 2: Fix the model precision knobs**

Replace:

```yaml
  qkv_bias: true
  param_dtype: "bfloat16"
```

with:

```yaml
  qkv_bias: true
  param_dtype: "float32"   # fp32 master weights
  dtype: "bfloat16"        # bf16 compute
```

- [ ] **Step 3: Flip the fresh-run flags and drop the stale eval-recovery comments**

Replace:

```yaml
  train_model: false        # eval-only recovery: reload the saved step-1 checkpoint
  restore_model: true       # (100k-sample eval OOM'd; retrain not needed)
```

with:

```yaml
  train_model: true
  restore_model: false
```

- [ ] **Step 4: Verify the config parses with the intended values**

```bash
cd /lustre/ific.uv.es/ml/ific088/github/GenSBI-examples/examples/sbi-benchmarks/spherical_grf
python -c "
import yaml
c = yaml.safe_load(open('config/config_healpix.yaml'))
assert c['encoder']['patch_size'] == 1
assert c['encoder']['param_dtype'] == 'float32'
assert c['encoder']['dtype'] == 'bfloat16'
assert c['model']['param_dtype'] == 'float32'
assert c['model']['dtype'] == 'bfloat16'
assert c['training']['train_model'] is True
assert c['training']['restore_model'] is False
print('config ok')
"
```

Expected: `config ok`

- [ ] **Step 5: Commit**

```bash
cd /lustre/ific.uv.es/ml/ific088/github/GenSBI-examples
git add examples/sbi-benchmarks/spherical_grf/config/config_healpix.yaml
git commit -m "fix(spherical_grf): exp-2 config — fp32 master weights, explicit bf16 compute, fresh-run flags

param_dtype bfloat16 was inverted (recreates the EMA-underflow bug the
mixed-precision branch fixes); train/restore flags were stale eval-recovery
leftovers pointing at a nonexistent _2 checkpoint.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Thread patch_size + dtypes through the training script

**Files:**
- Modify: `examples/sbi-benchmarks/spherical_grf/train-spherical-grf.py`

**Interfaces:**
- Consumes: YAML keys from Task 2 (`encoder.patch_size`, `encoder.dtype`, `encoder.param_dtype`).
- Produces: module constants `PATCH_SIZE` (int), corrected `NSIDE_BOTTLENECK`/`COND_TOKENS` (= 8 / 768 for config_healpix.yaml), `make_encoder_params()` passing `patch_size`, `dtype`, `param_dtype` to `HealSwinParams`. Task 4 relies on the smoke block printing `cond tokens: 768 features: 512`.

- [ ] **Step 1: Add the isqrt import**

Replace:

```python
import time

import jax
```

with:

```python
import time
from math import isqrt

import jax
```

- [ ] **Step 2: Read patch_size and generalize the bottleneck formula**

Replace:

```python
# HealSwin encoder geometry (bottleneck derived, never written twice)
_ENC = CONFIG["encoder"]
EMBED_DIM = _ENC["embed_dim"]
DEPTHS = tuple(_ENC["depths"])
ENC_NUM_HEADS = tuple(_ENC["num_heads"])
WINDOW_SIZE = _ENC["window_size"]
NSIDE_BOTTLENECK = NSIDE // (2 * 2 ** (len(DEPTHS) - 1))  # patch /2, then mergings
COND_TOKENS = 12 * NSIDE_BOTTLENECK ** 2
COND_FEATURES = EMBED_DIM * 2 ** (len(DEPTHS) - 1)
```

with:

```python
# HealSwin encoder geometry (bottleneck derived, never written twice)
_ENC = CONFIG["encoder"]
EMBED_DIM = _ENC["embed_dim"]
DEPTHS = tuple(_ENC["depths"])
ENC_NUM_HEADS = tuple(_ENC["num_heads"])
WINDOW_SIZE = _ENC["window_size"]
PATCH_SIZE = _ENC.get("patch_size", 4)  # power of four; 1 = no regrouping
# patch embed divides nside by sqrt(patch_size), each merging halves it
NSIDE_BOTTLENECK = NSIDE // (isqrt(PATCH_SIZE) * 2 ** (len(DEPTHS) - 1))
COND_TOKENS = 12 * NSIDE_BOTTLENECK ** 2
COND_FEATURES = EMBED_DIM * 2 ** (len(DEPTHS) - 1)
```

- [ ] **Step 3: Pass patch_size and the precision knobs to HealSwinParams**

Replace:

```python
def make_encoder_params() -> HealSwinParams:
    return HealSwinParams(
        nside=NSIDE,
        in_channels=1,
        out_channels=1,  # required by the dataclass; unused by the encoder
        embed_dim=EMBED_DIM,
        depths=DEPTHS,
        num_heads=ENC_NUM_HEADS,
        window_size=WINDOW_SIZE,
    )
```

with:

```python
def make_encoder_params() -> HealSwinParams:
    return HealSwinParams(
        nside=NSIDE,
        in_channels=1,
        out_channels=1,  # required by the dataclass; unused by the encoder
        embed_dim=EMBED_DIM,
        depths=DEPTHS,
        num_heads=ENC_NUM_HEADS,
        window_size=WINDOW_SIZE,
        patch_size=PATCH_SIZE,
        dtype=_ENC.get("dtype", "bfloat16"),
        param_dtype=_ENC.get("param_dtype", "float32"),
    )
```

- [ ] **Step 4: Make the dataset float32 cast explicit**

Replace:

```python
    ds = TaskDataset(
        "spherical_grf", ordering="nest", normalize=True,
        seed=SEED, max_workers=NUM_WORKERS,
    )
```

with:

```python
    ds = TaskDataset(
        "spherical_grf", ordering="nest", normalize=True, dtype=np.float32,
        seed=SEED, max_workers=NUM_WORKERS,
    )
```

- [ ] **Step 5: Update the stale geometry comments and docstring**

5a. In the module docstring, replace:

```python
A HEALPix-native Swin encoder compresses each nside-64 spherical map to 48
bottleneck tokens (nside 2, 512 features), which condition a gensbi Flux1
```

with:

```python
A HEALPix-native Swin encoder compresses each nside-64 spherical map to a
bottleneck token grid (config-derived; 768 tokens x 512 features at nside 8
for config_healpix.yaml), which conditions a gensbi Flux1
```

5b. In `SphericalGRFModel.__call__`, replace:

```python
        tokens, _skips = self.encoder(cond)  # (B, NPIX, 1) -> (B, 48, 512)
```

with:

```python
        tokens, _skips = self.encoder(cond)  # (B, NPIX, 1) -> (B, COND_TOKENS, COND_FEATURES)
```

5c. In the SMOKE block at the bottom, replace:

```python
    if COND_ID_KIND == "healpix-rope":
        cond_ids, _ = COND_STRATEGY.build(COND_TOKENS)  # (1, 48, 3) float32
    else:
        cond_ids, _ = init_ids_1d(COND_TOKENS, 1)  # (1, 48, 2)
```

with:

```python
    print("cond tokens:", COND_TOKENS, "features:", COND_FEATURES)
    if COND_ID_KIND == "healpix-rope":
        cond_ids, _ = COND_STRATEGY.build(COND_TOKENS)  # (1, COND_TOKENS, 3) float32
    else:
        cond_ids, _ = init_ids_1d(COND_TOKENS, 1)  # (1, COND_TOKENS, 2)
```

- [ ] **Step 6: Record patch + precision scheme in the startup log line**

Replace:

```python
    log(f"quick={QUICK} batch={BATCH_SIZE} nsteps={NSTEPS} workers={NUM_WORKERS} "
        f"nside={NSIDE} embed_dim={EMBED_DIM} depths={DEPTHS} window={WINDOW_SIZE} "
        f"cond={COND_TOKENS}x{COND_FEATURES} "
        f"flux={FLUX_PARAMS_DICT['depth']}d+{FLUX_PARAMS_DICT['depth_single_blocks']}s "
        f"heads={FLUX_PARAMS_DICT['num_heads']} "
        f"ids={FLUX_PARAMS_DICT['id_embedding_strategy']} cond_ids={COND_ID_KIND}")
```

with:

```python
    log(f"quick={QUICK} batch={BATCH_SIZE} nsteps={NSTEPS} workers={NUM_WORKERS} "
        f"nside={NSIDE} embed_dim={EMBED_DIM} depths={DEPTHS} window={WINDOW_SIZE} "
        f"patch={PATCH_SIZE} cond={COND_TOKENS}x{COND_FEATURES} "
        f"enc_dtype={_ENC.get('dtype', 'bfloat16')}/{_ENC.get('param_dtype', 'float32')} "
        f"flux_dtype={FLUX_PARAMS_DICT['dtype'].__name__}"
        f"/{FLUX_PARAMS_DICT['param_dtype'].__name__} "
        f"flux={FLUX_PARAMS_DICT['depth']}d+{FLUX_PARAMS_DICT['depth_single_blocks']}s "
        f"heads={FLUX_PARAMS_DICT['num_heads']} "
        f"ids={FLUX_PARAMS_DICT['id_embedding_strategy']} cond_ids={COND_ID_KIND}")
```

(`FLUX_PARAMS_DICT['dtype']` is a scalar type like `jnp.bfloat16` from `getattr(jnp, ...)`; `.__name__` yields `"bfloat16"` / `"float32"`. dtype pairs log as compute/param.)

- [ ] **Step 7: Run the SMOKE forward-shape check**

```bash
cd /lustre/ific.uv.es/ml/ific088/github/GenSBI-examples/examples/sbi-benchmarks/spherical_grf
SMOKE=1 JAX_PLATFORMS=cpu python train-spherical-grf.py --config config/config_healpix.yaml
```

Expected output (order matters, timings vary):

```
cond tokens: 768 features: 512
vector field shape: (2, 3, 1)
forward smoke check OK
```

If `cond tokens` is not 768 or the assert trips, fix before proceeding.

- [ ] **Step 8: Regression-check the pos1d config still parses to the old geometry**

`config_pos1d.yaml` has no `patch_size` key, so it must default to 4 and reproduce the pre-change geometry:

```bash
cd /lustre/ific.uv.es/ml/ific088/github/GenSBI-examples/examples/sbi-benchmarks/spherical_grf
SMOKE=1 JAX_PLATFORMS=cpu python train-spherical-grf.py --config config/config_pos1d.yaml
```

Expected: `cond tokens: 48 features: 512` (nside-2 bottleneck), then `forward smoke check OK`.

(If `config_pos1d.yaml` still has 5-stage depths like the archived config, 48 = `12 * (64 // (2 * 2**4))**2`. Whatever the number, it must equal the value the *unmodified* formula would have produced: `12 * (NSIDE // (2 * 2**(len(depths)-1)))**2`.)

- [ ] **Step 9: Commit**

```bash
cd /lustre/ific.uv.es/ml/ific088/github/GenSBI-examples
git add examples/sbi-benchmarks/spherical_grf/train-spherical-grf.py
git commit -m "fix(spherical_grf): thread patch_size + mixed-precision dtypes into the encoder

encoder.patch_size was silently ignored (HealSwinParams defaulted to 4) and
the bottleneck formula hardcoded the patch /2, so the configured patch-1
model was never built. Formula gains the sqrt(patch_size) factor; encoder
gets explicit dtype/param_dtype from the YAML; dataset float32 cast made
explicit; startup log records the precision scheme.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: QUICK end-to-end verification on CPU

**Files:**
- No source changes (fix-forward only if it fails).
- Generated (throwaway, gitignored or removable): `spherical_grf_fm_healpix_2_quick_results.txt`, `..._quick_losses.npz`, `imgs/spherical_grf_fm_healpix_2_quick_*.png`, `checkpoints/spherical_grf_fm_healpix_2_quick/`

**Interfaces:**
- Consumes: the committed config + script from Tasks 2–3.
- Produces: green light for GPU submission (Task 5).

- [ ] **Step 1: Run the QUICK end-to-end debug mode**

Needs the HF spherical_grf dataset cache (present from experiment 1). The patch-1 encoder is ~8x the FLOPs of the old one — expect several minutes to ~30 min on CPU.

```bash
cd /lustre/ific.uv.es/ml/ific088/github/GenSBI-examples/examples/sbi-benchmarks/spherical_grf
QUICK=1 JAX_PLATFORMS=cpu python train-spherical-grf.py --config config/config_healpix.yaml 2>&1 | tee /tmp/aamerio/claude-6356/-lustre-ific-uv-es-ml-ific088-github-GenSBI-examples/d3a16b75-bcb4-4731-882b-f5da3014f7e6/scratchpad/quick_run.log
```

Expected: completes without traceback; the startup line contains `patch=1 cond=768x512 enc_dtype=bfloat16/float32 flux_dtype=bfloat16/float32`; a `training: ... steps` line; `obs 1: ... samples`; a `TARP: ...` line.

- [ ] **Step 2: Assert no fp32-master-weights warning fired**

The pipeline's `_warn_if_not_fp32_master_weights` guard must be silent now that param_dtype is fp32:

```bash
grep -i "master" /tmp/aamerio/claude-6356/-lustre-ific-uv-es-ml-ific088-github-GenSBI-examples/d3a16b75-bcb4-4731-882b-f5da3014f7e6/scratchpad/quick_run.log || echo "no master-weight warning: ok"
```

Expected: `no master-weight warning: ok`

- [ ] **Step 3: Clean up QUICK artifacts**

```bash
cd /lustre/ific.uv.es/ml/ific088/github/GenSBI-examples/examples/sbi-benchmarks/spherical_grf
rm -rf checkpoints/spherical_grf_fm_healpix_2_quick
rm -f spherical_grf_fm_healpix_2_quick_results.txt spherical_grf_fm_healpix_2_quick_losses.npz
rm -f imgs/spherical_grf_fm_healpix_2_quick_*.png
git status --short   # must show no new untracked files from the QUICK run
```

---

### Task 5: Commit the run script, condor dry run, then submit

**Files:**
- Commit (pre-existing user change): `examples/sbi-benchmarks/spherical_grf/sub/run_spherical_grf.sh`
- Read-only: `examples/sbi-benchmarks/spherical_grf/sub/spherical_grf.sub`

**Interfaces:**
- Consumes: green light from Task 4.
- Produces: a queued HTCondor job (cluster id reported to the user).

- [ ] **Step 1: Commit the user's run-script change (conda env activation)**

The working tree already contains the user's edit adding `conda activate .../envs/gensbi` to `sub/run_spherical_grf.sh` — required for the condor worker to find the env. Commit it verbatim; do not edit.

```bash
cd /lustre/ific.uv.es/ml/ific088/github/GenSBI-examples
git add examples/sbi-benchmarks/spherical_grf/sub/run_spherical_grf.sh
git commit -m "fix(spherical_grf): activate the gensbi conda env in the condor run script

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 2: Dry-run the submission from the sub directory**

HTCondor resolves the relative `log/output/error` paths against the submit-time working directory unless `initialdir` covers them — the sub file sets `initialdir = $(example_dir)/sub`, but verify rather than trust:

```bash
cd /lustre/ific.uv.es/ml/ific088/github/GenSBI-examples/examples/sbi-benchmarks/spherical_grf/sub
condor_submit -dry-run /dev/stdout spherical_grf.sub | grep -Ei "^(Iwd|UserLog|Out|Err|Cmd|Args|Arguments)" 
```

Expected (paths must all be absolute and correct):

- `Iwd = "/lustre/ific.uv.es/ml/ific088/github/GenSBI-examples/examples/sbi-benchmarks/spherical_grf/sub"`
- `Cmd` = `.../sub/run_spherical_grf.sh`
- `UserLog`/`Out`/`Err` under `.../sub/condor_logs/` with the `config_healpix` stem
- `Args`/`Arguments` containing the example dir and `config_healpix.yaml`

If the grep matches nothing, dump the full dry-run output and inspect those attributes by eye. If any path resolves outside `.../spherical_grf/sub/condor_logs`, STOP and report — fixing the sub file is a user decision.

- [ ] **Step 3: Submit and confirm queued**

```bash
cd /lustre/ific.uv.es/ml/ific088/github/GenSBI-examples/examples/sbi-benchmarks/spherical_grf/sub
condor_submit spherical_grf.sub
condor_q
```

Expected: `1 job(s) submitted to cluster <N>.` and the job visible in `condor_q` (idle or running). Report the cluster id, the log file paths, and remind the user the run writes to `checkpoints/spherical_grf_fm_healpix_2/`, `spherical_grf_fm_healpix_2_results.txt`, and `imgs/spherical_grf_fm_healpix_2_*.png`.
