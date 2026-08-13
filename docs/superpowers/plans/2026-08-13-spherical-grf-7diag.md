# Spherical GRF exp-7 Sampler Diagnostic ("7diag") Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eval-only sampler diagnostic on the frozen exp-7 checkpoint: an ODE step-size/Dopri5 matrix and FM SDE solvers over all 10 observations, a C2ST scatter estimate, and TARP in both ODE and SDE regimes, submitted as four A100 condor jobs.

**Architecture:** One standalone script `eval-spherical-grf-diag.py` (copy of `train-spherical-grf-v2.py` with training stripped) restores the exp-7 EMA checkpoint and dispatches on a new `diagnostic:` YAML section. Four tiny mode configs deep-merge over `config_healpix_7.yaml` at load time. One submit file with four `queue` blocks.

**Tech Stack:** JAX/flax-nnx, gensbi (`ConditionalPipeline`, `FlowMatchingMethod`, `NonSingularSolver`, `ZeroEndsSolver`), sbibm-jax `TaskDataset`, orbax (read-only restore), HTCondor.

**Spec:** `docs/superpowers/specs/2026-08-13-spherical-grf-exp7-sampler-diagnostic-design.md`

## Global Constraints

- Working dir for all commands: `examples/sbi-benchmarks/spherical_grf/` (all paths below relative to it unless they start with `docs/`).
- Every command that touches data MUST run with `SBIBM_JAX_USE_TEST=1` (else a different 24 GB dataset repo silently downloads). The condor path exports it inside `sub/run_spherical_grf.sh` already; **local test commands must export it explicitly**.
- Local validation runs use `JAX_PLATFORMS=cpu` (no GPU on the login node) in the `gensbi` conda env (`conda activate /lhome/ific/a/aamerio/miniforge3/envs/gensbi`).
- Eval-only: the exp-7 checkpoint (`checkpoints/spherical_grf_fm_healpix_7/`, orbax step key `7`) is restored read-only; nothing in this plan trains or writes checkpoints.
- Output naming stem: `spherical_grf_fm_healpix_7diag_<mode>` (`<mode>` ∈ `ode`, `sde`, `tarp_ode`, `tarp_sde`); QUICK runs append `_quick`. QUICK artifacts (`*_quick_results.txt`, `imgs/*_quick_*.png`) are test debris — delete before committing, never commit them.
- All condor jobs request A100 (`+UseNvidiaA100 = True`), 8 CPU / 32 GB / 1 GPU. Never target H100.
- This repo's test convention for these scripts is the built-in `SMOKE=1` (CPU shape check, no data) and `QUICK=1` (tiny end-to-end run) env modes — there is no pytest harness. Each task's test cycle uses those modes; "failing test first" maps to "run the mode before the feature exists and see it fail, then after and see it pass".

---

### Task 1: Diagnostic script skeleton + config merge + `ode` mode config

**Files:**
- Create: `eval-spherical-grf-diag.py` (from `train-spherical-grf-v2.py`)
- Create: `config/config_healpix_7diag_ode.yaml`

**Interfaces:**
- Consumes: `train-spherical-grf-v2.py` (copy source), `config/config_healpix_7.yaml` (merge base), gensbi `parse_flux1_params(path)` / `parse_training_config(path)` (both re-read the YAML from a *path*, hence the merged-temp-file trick below).
- Produces: module-level constants later tasks rely on — `MODE: str`, `IS_TARP: bool`, `VARIANTS: tuple[dict, ...]`, `SCATTER_REPEATS: int`, `SCATTER_OBS: tuple[int, ...]`, `EXPERIMENT_ID: str` (the 7diag output stem), `CHECKPOINT_DIR` (exp-7's own, no `_quick` suffix). Everything else (model classes, `make_datasets`, `prep_x`, `make_pipeline`, `evaluate`, `tarp_diagnostic`, `main`, SMOKE block) is carried over unchanged from v2 by the copy; Tasks 2–3 edit those carried-over functions in place.

- [ ] **Step 1: Copy the v2 script**

```bash
cp train-spherical-grf-v2.py eval-spherical-grf-diag.py
```

- [ ] **Step 2: Write the ode mode config**

Create `config/config_healpix_7diag_ode.yaml` with exactly:

```yaml
# exp-7 sampler diagnostic — ODE matrix job. Design:
# docs/superpowers/specs/2026-08-13-spherical-grf-exp7-sampler-diagnostic-design.md
#
# Deep-merged over base_config at load time by eval-spherical-grf-diag.py:
# only overrides and the diagnostic section live here. Restores the exp-7
# EMA checkpoint read-only; never trains.
base_config: config/config_healpix_7.yaml

data:
  augment: false            # eval-only; training-stream augmentation is irrelevant

training:
  train_model: false
  restore_model: true

evaluation:
  observations: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  tarp_pairs: 0             # TARP runs in the dedicated tarp_* jobs

diagnostic:
  mode: ode                 # output stem: spherical_grf_fm_healpix_7diag_ode
  scatter_repeats: 3        # extra draws for baseline-flagged variants (obs below)
  scatter_observations: [1, 2, 3]
  variants:
    - name: euler_p01       # exp-7's shipped sampler — must reproduce its C2STs
      kind: euler
      step_size: 0.01
      baseline: true
    - name: euler_p002      # 5x finer: step-size sensitivity
      kind: euler
      step_size: 0.002
    - name: dopri5          # adaptive (library defaults atol=rtol=1e-5)
      kind: dopri5
```

- [ ] **Step 3: Run the not-yet-adapted script to see it fail**

```bash
cd examples/sbi-benchmarks/spherical_grf
SMOKE=1 JAX_PLATFORMS=cpu python eval-spherical-grf-diag.py --config config/config_healpix_7diag_ode.yaml
```

Expected: FAIL with `KeyError` (the copied script reads `CONFIG["data"]["nside"]` etc. from the tiny overlay file, which has no such keys — the merge doesn't exist yet).

- [ ] **Step 4: Add the base-config deep merge**

In `eval-spherical-grf-diag.py`, directly after the existing lines

```python
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)
```

insert:

```python
def _deep_merge(base, overlay):
    """Overlay wins; dicts merge recursively, everything else replaces."""
    out = dict(base)
    for k, v in overlay.items():
        out[k] = (_deep_merge(out[k], v)
                  if isinstance(v, dict) and isinstance(out.get(k), dict)
                  else v)
    return out


if "base_config" in CONFIG:
    _base_rel = CONFIG.pop("base_config")
    _base_path = (_base_rel if os.path.isabs(_base_rel)
                  else os.path.join(BASE_DIR, _base_rel))
    with open(_base_path) as f:
        CONFIG = _deep_merge(yaml.safe_load(f), CONFIG)
    # parse_flux1_params / parse_training_config re-read the YAML from a
    # path; point them at the merged view via a temp file.
    import tempfile
    _merged = tempfile.NamedTemporaryFile(
        "w", suffix=".yaml", prefix="7diag_merged_", delete=False)
    yaml.safe_dump(CONFIG, _merged)
    _merged.close()
    CONFIG_PATH = _merged.name
```

- [ ] **Step 5: Add diagnostic constants and fix the naming/checkpoint split**

Still in the configuration block: the copied script ends its constants section with (v2 lines 196–199):

```python
EXPERIMENT_ID = _TRAIN["run_name"] + ("_quick" if QUICK else "")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints", EXPERIMENT_ID)
IMGS_DIR = os.path.join(BASE_DIR, "imgs")
RESULTS_FILE = os.path.join(BASE_DIR, f"{EXPERIMENT_ID}_results.txt")
```

Replace those four lines with:

```python
# diagnostic mode
_DIAG = CONFIG["diagnostic"]
MODE = _DIAG["mode"]                       # ode | sde | tarp_ode | tarp_sde
IS_TARP = MODE.startswith("tarp")
VARIANTS = tuple(_DIAG["variants"])        # dicts: name/kind[/step_size/alpha/baseline]
SCATTER_REPEATS = 0 if QUICK else int(_DIAG.get("scatter_repeats", 0))
SCATTER_OBS = tuple(_DIAG.get("scatter_observations", ()))

# Outputs get the 7diag stem; the checkpoint dir is exp-7's own (read-only
# restore) and must NOT get the _quick suffix, or QUICK finds no checkpoint.
EXPERIMENT_ID = _TRAIN["run_name"] + "diag_" + MODE + ("_quick" if QUICK else "")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints", _TRAIN["run_name"])
IMGS_DIR = os.path.join(BASE_DIR, "imgs")
RESULTS_FILE = os.path.join(BASE_DIR, f"{EXPERIMENT_ID}_results.txt")
```

- [ ] **Step 6: Strip training-only code**

All in `eval-spherical-grf-diag.py` (function boundaries as in the v2 copy):

1. Delete the whole functions `_isometry_perms` and `_augmented_batches` (v2 lines 300–333).
2. Delete the whole function `save_loss_history` (v2 lines 483–510).
3. In `make_datasets`, delete the augmentation branch so the body reads:

```python
    ds = TaskDataset(
        "spherical_grf", ordering="nest", normalize=True, dtype=np.float32,
        seed=SEED, max_workers=NUM_WORKERS,
    )
    train_loader = ds.get_train_loader(BATCH_SIZE)
    # The pipeline draws one fixed val batch at train start.
    val_loader = ds.get_val_loader(VAL_BATCH_SIZE)
    return ds, train_loader, val_loader
```

4. Delete the module-level `AUGMENT = bool(CONFIG["data"].get("augment", False))` line and its comment block (v2 lines 117–122).
5. In `main()`, replace

```python
    if TRAIN_MODEL:
        t0 = time.time()
        losses, val_losses = pipeline.train(nnx.Rngs(SEED + 2), save_model=True)
        log(f"training: {len(losses)} steps in {time.time() - t0:.0f}s, "
            f"final train loss {float(losses[-1]):.4f}, "
            f"final val loss {float(val_losses[-1]):.4f}")
        save_loss_history(losses, val_losses, log)
    if RESTORE_MODEL:
        pipeline.restore_model()
        pipeline._wrap_model()
```

with:

```python
    assert RESTORE_MODEL and not TRAIN_MODEL, (
        "7diag is eval-only: set training.train_model false / restore_model true")
    t0 = time.time()
    pipeline.restore_model()
    pipeline._wrap_model()
    log(f"restored exp-7 EMA checkpoint from {CHECKPOINT_DIR} "
        f"(step {make_training_config()['experiment_id']}) in {time.time() - t0:.0f}s")
```

6. Update the module docstring: replace the v2 title/first paragraph with a short header stating this is the eval-only exp-7 sampler diagnostic (spec path above), that it restores the exp-7 checkpoint and runs the sampler-variant matrix or TARP per `diagnostic.mode`, and keep the SMOKE/QUICK usage lines with the script name updated to `eval-spherical-grf-diag.py`.

Keep everything else — model classes, `prep_x`, `make_training_config`, `make_pipeline`, `evaluate`, `tarp_diagnostic`, the SMOKE block — untouched in this task.

- [ ] **Step 7: SMOKE check passes**

```bash
SMOKE=1 JAX_PLATFORMS=cpu python eval-spherical-grf-diag.py --config config/config_healpix_7diag_ode.yaml
```

Expected: PASS — prints `cond tokens: 192 features: 512 global blocks: 2` and the velocity-output shape line, exit 0. This proves the merge produced a complete config (module-level constants, `parse_flux1_params` on the merged temp file) and the model builds.

- [ ] **Step 8: Commit**

```bash
git add eval-spherical-grf-diag.py config/config_healpix_7diag_ode.yaml
git commit -m "feat(spherical_grf): 7diag eval-only script skeleton + ode config

Copy of train-spherical-grf-v2.py with training stripped: base-config
deep-merge (temp-file trick for path-based gensbi parsers), diagnostic
constants, 7diag output stem split from the exp-7 checkpoint dir."
```

---

### Task 2: Sampler-variant matrix (variants loop, C2ST scatter, overlays) + `sde` config

**Files:**
- Modify: `eval-spherical-grf-diag.py` (imports; add `sampler_kwargs`; replace `evaluate`; dispatch in `main`)
- Create: `config/config_healpix_7diag_sde.yaml`

**Interfaces:**
- Consumes: Task 1's constants (`VARIANTS`, `SCATTER_REPEATS`, `SCATTER_OBS`, `IS_TARP`, `EXPERIMENT_ID`); gensbi API `pipeline.sample(key, x_o, nsamples, chunk_size=..., step_size=..., method=..., solver=...)` — `**sampler_kwargs` forwards through `get_sampler` to `FlowMatchingMethod.build_sampler_fn(..., step_size=0.01, method="Euler", time_grid=None, solver=None)`; SDE solvers are passed as `solver=(SolverClass, {"alpha": ...})` and require the default Gaussian prior (satisfied — `FlowMatchingMethod()` default).
- Produces: `sampler_kwargs(v: dict) -> dict` (also used by Task 3's TARP), `evaluate_matrix(pipeline, ds, log)`.

- [ ] **Step 1: Write the sde mode config**

Create `config/config_healpix_7diag_sde.yaml` with exactly:

```yaml
# exp-7 sampler diagnostic — SDE matrix job. Design:
# docs/superpowers/specs/2026-08-13-spherical-grf-exp7-sampler-diagnostic-design.md
#
# Stochastic FM samplers (arXiv:2410.02217; score derived from the velocity
# field): does churn recover the reference dispersion that the ODE
# under-covers? Merged over base_config; eval-only.
base_config: config/config_healpix_7.yaml

data:
  augment: false

training:
  train_model: false
  restore_model: true

evaluation:
  observations: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  tarp_pairs: 0

diagnostic:
  mode: sde                 # output stem: spherical_grf_fm_healpix_7diag_sde
  variants:
    - name: nonsingular_a05
      kind: nonsingular
      alpha: 0.5
      step_size: 0.01
    - name: nonsingular_a10
      kind: nonsingular
      alpha: 1.0
      step_size: 0.01
    - name: zeroends_a10    # diffusion vanishes at both time ends
      kind: zeroends
      alpha: 1.0
      step_size: 0.01
```

- [ ] **Step 2: Run QUICK to see the matrix machinery is missing**

```bash
SBIBM_JAX_USE_TEST=1 QUICK=1 JAX_PLATFORMS=cpu python eval-spherical-grf-diag.py --config config/config_healpix_7diag_ode.yaml
```

Expected: FAIL — the carried-over v2 `evaluate` runs the single hard-coded Euler sampler and knows nothing about `VARIANTS`; there is no per-variant output. (It may even complete without error — the *failure* is the results file `spherical_grf_fm_healpix_7diag_ode_quick_results.txt` containing no `variant` lines. Check it and delete it: `rm -f spherical_grf_fm_healpix_7diag_ode_quick_results.txt`.)

- [ ] **Step 3: Implement the variant machinery**

In `eval-spherical-grf-diag.py`:

1. Extend the gensbi imports (next to the existing `from gensbi.core import FlowMatchingMethod`):

```python
from gensbi.flow_matching.solver import NonSingularSolver, ZeroEndsSolver
```

2. Add above `evaluate`:

```python
def sampler_kwargs(v):
    """YAML variant dict -> kwargs for pipeline.sample / sample_batched.

    Forwarded through get_sampler to FlowMatchingMethod.build_sampler_fn.
    QUICK coarsens fixed steps to 0.25 (4 steps) like the training script.
    """
    kind = v["kind"]
    step = v.get("step_size")
    if QUICK and step is not None:
        step = 0.25
    if kind == "euler":
        return {"step_size": step, "method": "Euler"}
    if kind == "dopri5":
        return {"step_size": None, "method": "Dopri5"}  # adaptive; atol=rtol=1e-5
    if kind == "nonsingular":
        return {"step_size": step,
                "solver": (NonSingularSolver, {"alpha": float(v["alpha"])})}
    if kind == "zeroends":
        return {"step_size": step,
                "solver": (ZeroEndsSolver, {"alpha": float(v["alpha"])})}
    raise ValueError(f"unknown diagnostic variant kind: {kind!r}")
```

3. Replace the whole `evaluate` function with:

```python
def evaluate_matrix(pipeline, ds, log):
    """Sampler-variant matrix: posterior vs reference for every observation.

    Per variant x obs: mean/std vs reference, C2ST. Baseline-flagged
    variants get SCATTER_REPEATS extra independent draws on SCATTER_OBS
    (C2ST mean +/- std prices the run-to-run scatter). Overlay corners for
    obs 1-3 only, every variant.
    """
    key = jax.random.PRNGKey(SEED + 7)
    labels = list(THETA_LABELS)

    for v in VARIANTS:
        kw = sampler_kwargs(v)
        vname = v["name"]
        log(f"variant {vname}: {kw}")
        for i in EVAL_OBSERVATIONS:
            x_raw, ref = ds.get_reference(i)                 # RING map, (S, 3)
            x_o = prep_x(np.asarray(x_raw).reshape(1, -1), ds)
            theta_true = np.asarray(ds.get_true_parameters(i)).reshape(-1)
            ref = np.asarray(ref)

            ndraws = 1 + (SCATTER_REPEATS
                          if v.get("baseline") and i in SCATTER_OBS else 0)
            c2sts, first_flow, n_c2st = [], None, 0
            for rep in range(ndraws):
                t0 = time.time()
                key, sk = jax.random.split(key)
                samples = np.asarray(pipeline.sample(
                    sk, x_o, NUM_POSTERIOR_SAMPLES,
                    chunk_size=SAMPLE_CHUNK, **kw))
                flow = np.asarray(ds.unnormalize_theta(samples))[:, :, 0]
                n_c2st = int(min(flow.shape[0], ref.shape[0]))
                _rng = np.random.default_rng(SEED + 1000 * rep + i)
                f_c = flow[_rng.choice(flow.shape[0], n_c2st, replace=False)]
                r_c = ref[_rng.choice(ref.shape[0], n_c2st, replace=False)]
                c2sts.append(float(c2st(jnp.asarray(f_c), jnp.asarray(r_c),
                                        seed=SEED + rep)))
                if rep == 0:
                    first_flow = flow
                    log(f"{vname} obs {i}: {flow.shape[0]} samples in "
                        f"{time.time() - t0:.0f}s | "
                        f"true {np.array2string(theta_true, precision=3)} | "
                        f"flow mean {np.array2string(flow.mean(0), precision=3)} "
                        f"std {np.array2string(flow.std(0), precision=3)} | "
                        f"ref mean {np.array2string(ref.mean(0), precision=3)} "
                        f"std {np.array2string(ref.std(0), precision=3)}")

            if ndraws > 1:
                log(f"{vname} obs {i}: C2ST = {np.mean(c2sts):.3f} +/- "
                    f"{np.std(c2sts):.3f} over {ndraws} draws "
                    f"{np.array2string(np.asarray(c2sts), precision=3)}")
            else:
                log(f"{vname} obs {i}: C2ST(flow vs ref) = {c2sts[0]:.3f} "
                    f"over {n_c2st} samples (0.5=indistinguishable, "
                    f"1.0=separable)")

            if i in (1, 2, 3):
                fig = corner(ref, labels=labels, truths=list(theta_true),
                             color="C0", hist_kwargs={"density": True},
                             plot_contours=not QUICK, plot_density=not QUICK)
                corner(first_flow, fig=fig, color="C1",
                       hist_kwargs={"density": True},
                       plot_contours=not QUICK, plot_density=not QUICK)
                fig.suptitle(f"obs {i} [{vname}]: reference (blue) vs "
                             f"flow (orange)")
                fig.savefig(os.path.join(
                    IMGS_DIR, f"{EXPERIMENT_ID}_{vname}_overlay_obs{i}.png"),
                    dpi=100, bbox_inches="tight")
                plt.close(fig)
```

Note this drops v2's separate per-side `plot_marginals` pngs and its end-of-function TARP call — TARP dispatch moves to `main` (Task 3); the deleted `evaluate` was the only caller of nothing else.

4. In `main()`, replace the line `evaluate(pipeline, ds, log)` with:

```python
    if IS_TARP:
        if TARP_PAIRS:
            tarp_diagnostic(pipeline, ds, log, jax.random.PRNGKey(SEED + 7))
        else:
            log("TARP: skipped (evaluation.tarp_pairs = 0)")
    else:
        evaluate_matrix(pipeline, ds, log)
```

(`tarp_diagnostic` is still the carried-over v2 version at this point; Task 3 adapts it. Its signature `(pipeline, ds, log, key)` already matches.)

- [ ] **Step 4: QUICK end-to-end, both matrix modes**

```bash
SBIBM_JAX_USE_TEST=1 QUICK=1 JAX_PLATFORMS=cpu python eval-spherical-grf-diag.py --config config/config_healpix_7diag_ode.yaml
SBIBM_JAX_USE_TEST=1 QUICK=1 JAX_PLATFORMS=cpu python eval-spherical-grf-diag.py --config config/config_healpix_7diag_sde.yaml
```

Expected: both PASS (exit 0). QUICK restores the real exp-7 checkpoint, uses obs `(1,)`, 64 samples, step 0.25. Verify per run:
- results file `spherical_grf_fm_healpix_7diag_{ode,sde}_quick_results.txt` contains a `restored exp-7 EMA checkpoint` line and one `C2ST` line per variant (3 each);
- the sde run exercises `NonSingularSolver`/`ZeroEndsSolver` (this is the riskiest wiring — a failure here means the `solver=(cls, kwargs)` tuple isn't reaching `build_sampler_fn`; debug there, do not submit);
- the ode run's `dopri5` variant completes (adaptive on CPU is slower — minutes, not hours; be patient before assuming a hang).

CPU note: each run is roughly comparable to a v2 QUICK eval per variant; expect ~5–20 min total each.

- [ ] **Step 5: Delete QUICK debris and commit**

```bash
rm -f spherical_grf_fm_healpix_7diag_*_quick_results.txt
rm -f imgs/spherical_grf_fm_healpix_7diag_*_quick_*.png
git add eval-spherical-grf-diag.py config/config_healpix_7diag_sde.yaml
git commit -m "feat(spherical_grf): 7diag sampler-variant matrix + sde config

Variant loop over Euler/Dopri5/FM-SDE samplers (solver kwargs forwarded
through pipeline.sample), per-obs C2ST, scatter repeats on baseline
variants, overlay corners for obs 1-3."
```

---

### Task 3: TARP modes + their configs

**Files:**
- Modify: `eval-spherical-grf-diag.py` (adapt `tarp_diagnostic`)
- Create: `config/config_healpix_7diag_tarp_ode.yaml`
- Create: `config/config_healpix_7diag_tarp_sde.yaml`

**Interfaces:**
- Consumes: `sampler_kwargs(v)` from Task 2; `VARIANTS[0]` as the single TARP variant; `pipeline.sample_batched(key, x_tok, nsamples, chunk_size=..., **sampler_kwargs)` (forwards kwargs to `sample` per condition); the `main` dispatch added in Task 2 already routes `IS_TARP` modes here.
- Produces: mode configs `tarp_ode` / `tarp_sde`; TARP png `imgs/spherical_grf_fm_healpix_7diag_tarp_{ode,sde}_tarp.png`.

- [ ] **Step 1: Write both TARP configs**

`config/config_healpix_7diag_tarp_ode.yaml`:

```yaml
# exp-7 sampler diagnostic — TARP, ODE regime (the sampler exp-7 shipped).
# Design: docs/superpowers/specs/2026-08-13-spherical-grf-exp7-sampler-diagnostic-design.md
base_config: config/config_healpix_7.yaml

data:
  augment: false

training:
  train_model: false
  restore_model: true

evaluation:
  observations: []          # matrix skipped in tarp modes
  tarp_pairs: 200
  tarp_posterior_samples: 1000

diagnostic:
  mode: tarp_ode            # output stem: spherical_grf_fm_healpix_7diag_tarp_ode
  variants:
    - name: euler_p01
      kind: euler
      step_size: 0.01
```

`config/config_healpix_7diag_tarp_sde.yaml`:

```yaml
# exp-7 sampler diagnostic — TARP, SDE regime: does stochastic sampling
# restore calibration where the ODE is over-confident?
# Design: docs/superpowers/specs/2026-08-13-spherical-grf-exp7-sampler-diagnostic-design.md
base_config: config/config_healpix_7.yaml

data:
  augment: false

training:
  train_model: false
  restore_model: true

evaluation:
  observations: []
  tarp_pairs: 200
  tarp_posterior_samples: 1000

diagnostic:
  mode: tarp_sde            # output stem: spherical_grf_fm_healpix_7diag_tarp_sde
  variants:
    - name: nonsingular_a10
      kind: nonsingular
      alpha: 1.0
      step_size: 0.01
```

- [ ] **Step 2: Run QUICK tarp_ode to see the stale sampler call fail the intent**

```bash
SBIBM_JAX_USE_TEST=1 QUICK=1 JAX_PLATFORMS=cpu python eval-spherical-grf-diag.py --config config/config_healpix_7diag_tarp_ode.yaml
```

Expected: runs (QUICK: 2 pairs × 8 samples) but the carried-over v2 `tarp_diagnostic` still calls `sample_batched(..., step_size=SAMPLE_STEP_SIZE, ...)` — it ignores the variant. The tarp_sde config would silently produce ODE TARP. That silent wrongness is what this task fixes. Delete debris: `rm -f spherical_grf_fm_healpix_7diag_tarp_ode_quick_results.txt imgs/spherical_grf_fm_healpix_7diag_tarp_ode_quick_tarp.png`.

- [ ] **Step 3: Adapt `tarp_diagnostic` to the variant**

In the carried-over `tarp_diagnostic`, make exactly two edits:

1. At the top of the function body (after the docstring), add:

```python
    v = VARIANTS[0]
    kw = sampler_kwargs(v)
    log(f"TARP variant {v['name']}: {kw}")
```

2. In its `pipeline.sample_batched(...)` call, replace the argument `step_size=SAMPLE_STEP_SIZE` with `**kw`.

- [ ] **Step 4: QUICK both TARP modes**

```bash
SBIBM_JAX_USE_TEST=1 QUICK=1 JAX_PLATFORMS=cpu python eval-spherical-grf-diag.py --config config/config_healpix_7diag_tarp_ode.yaml
SBIBM_JAX_USE_TEST=1 QUICK=1 JAX_PLATFORMS=cpu python eval-spherical-grf-diag.py --config config/config_healpix_7diag_tarp_sde.yaml
```

Expected: both PASS; each results file contains the `TARP variant ...` line (with `solver` in the kwargs for tarp_sde) and the `TARP: 2 pairs x 8 samples ... -> ..._tarp.png` line; the png exists under `imgs/`.

- [ ] **Step 5: Delete QUICK debris and commit**

```bash
rm -f spherical_grf_fm_healpix_7diag_tarp_*_quick_results.txt
rm -f imgs/spherical_grf_fm_healpix_7diag_tarp_*_quick_*.png
git add eval-spherical-grf-diag.py config/config_healpix_7diag_tarp_ode.yaml config/config_healpix_7diag_tarp_sde.yaml
git commit -m "feat(spherical_grf): 7diag TARP modes (ODE vs NonSingular-SDE regime)"
```

---

### Task 4: Submit file + dry-run

**Files:**
- Create: `sub/spherical_grf_7diag.sub`

**Interfaces:**
- Consumes: `sub/run_spherical_grf.sh` contract — `$1` example dir, `$2` config file *name* (resolved as `config/$2`), `$3` script name; it exports `SBIBM_JAX_USE_TEST=1` and activates the gensbi env itself.
- Produces: the four-job submission unit for Task 5.

- [ ] **Step 1: Write the submit file**

Create `sub/spherical_grf_7diag.sub` with exactly:

```
# HTCondor submit file — exp-7 sampler diagnostic (7diag): four eval-only
# jobs on the frozen exp-7 checkpoint. Design:
# docs/superpowers/specs/2026-08-13-spherical-grf-exp7-sampler-diagnostic-design.md
#
#     condor_submit sub/spherical_grf_7diag.sub
#
# ode  (~4 h): Euler 0.01 baseline + Euler 0.002 + Dopri5, 10 obs, C2ST scatter
# sde  (~2.5 h): NonSingular a=0.5/1.0 + ZeroEnds a=1.0, 10 obs
# tarp_ode (~1 h) / tarp_sde (~2.5 h): 200 pairs x 1000 samples
# All A100 (H100s reserved for higher-priority work). Eval memory ~3 GiB
# (chunked sampling), far under the 40 GB card.
repo_root   = /lustre/ific.uv.es/ml/ific088/github/GenSBI-examples
example_dir = $(repo_root)/examples/sbi-benchmarks/spherical_grf
script      = eval-spherical-grf-diag.py

universe   = vanilla
initialdir = $(example_dir)/sub
executable = $(example_dir)/sub/run_spherical_grf.sh
arguments  = "$(example_dir) $(config) $(script)"
getenv     = True

request_cpus   = 8
request_memory = 32 GB
request_gpus   = 1

+UseNvidiaA100 = True

log    = condor_logs/spherical_grf_$Fn(config)_a100.log
output = condor_logs/spherical_grf_$Fn(config)_a100.out
error  = condor_logs/spherical_grf_$Fn(config)_a100.err

config = config_healpix_7diag_ode.yaml
queue
config = config_healpix_7diag_sde.yaml
queue
config = config_healpix_7diag_tarp_ode.yaml
queue
config = config_healpix_7diag_tarp_sde.yaml
queue
```

- [ ] **Step 2: Dry-run the submit file**

```bash
cd sub && condor_submit -dry-run /dev/stdout spherical_grf_7diag.sub | grep -E "^(Arguments|Args)" ; cd ..
```

Expected: four job ads, whose arguments name the four different diag configs and `eval-spherical-grf-diag.py`. (If `-dry-run` output has no `Arguments` lines on this condor version, inspect the full dump for the four `config_healpix_7diag_*.yaml` names instead.)

- [ ] **Step 3: Commit**

```bash
git add sub/spherical_grf_7diag.sub
git commit -m "feat(spherical_grf): 7diag submit file — four A100 eval jobs"
```

---

### Task 5: Submit and verify

**Files:**
- None created; cluster state only.

**Interfaces:**
- Consumes: everything above, committed; the exp-7 checkpoint on lustre.
- Produces: a running/queued condor cluster; its ID reported to the user.

- [ ] **Step 1: Pre-submission sanity**

```bash
git status --short           # expect: clean
ls checkpoints/spherical_grf_fm_healpix_7/   # expect: orbax step dir "7"
```

- [ ] **Step 2: Submit**

```bash
condor_submit sub/spherical_grf_7diag.sub
```

Expected: `4 job(s) submitted to cluster <N>`. Record `<N>`.

- [ ] **Step 3: Verify queue state**

```bash
condor_q <N> -nobatch
```

Expected: 4 jobs, mix of `R`/`I` (13/14 A100 slots were claimed at design time — idle jobs are normal and will start as slots free).

- [ ] **Step 4: Verify the first running job restores correctly**

Once any job shows `R`, wait ~5 min, then:

```bash
grep -l "restored exp-7 EMA checkpoint" sub/condor_logs/spherical_grf_config_healpix_7diag_*_a100.out
```

Expected: at least one hit; also check the matching `.err` for tracebacks (`tail sub/condor_logs/...err`). A missing restore line after the data-load line (~7 s) plus a traceback means stop: `condor_rm <N>`, debug, resubmit.

- [ ] **Step 5: Report**

Report to the user: cluster ID, queue state, and the post-run analysis gate — when the `ode` job finishes, its `euler_p01` C2ST on obs 1–3 must match exp-7's shipped 0.852 / 0.842 / 0.729 within the scatter its own repeats measure; if not, the restore or sampler wiring is suspect and the other jobs' numbers should not be trusted until explained.

---

## Post-plan notes (not tasks)

- Results interpretation and the exp-8 decision (dataset size / regularization / capacity vs eval-side SDE fix) happen in a follow-up session once the four results files exist; the decision tree is in the spec's Motivation section.
- The known encoder-re-encoding-per-ODE-step inefficiency (~100x redundant encoder work during sampling) is untouched; Dopri5 and SDE runtimes inherit it. If the `ode` job blows past ~6 h on Dopri5 alone, that inefficiency is why — not a hang.
