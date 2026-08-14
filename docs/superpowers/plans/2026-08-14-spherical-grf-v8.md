# Spherical GRF v8 (probes P2/P3 + exp-8 a/b runs) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the two remaining diagnostics (P2 information-mismatch reference, P3 amplitude-equivariance), gate on their outcome, then train and auto-evaluate two 600k-step v8 runs (control + amplitude-pass-through) on A100s.

**Architecture:** Probes are standalone scripts in `v8_probes/` (P2 CPU-local, P3 one GPU condor job). v8 extends `train-spherical-grf-v2.py` with a YAML-switched raw-map power pass-through inside `SphericalGRFModel`; two standalone configs (`config_healpix_8a/8b`) and one submit file with two queue blocks. Post-run eval reuses the 7diag eval script (which also gains the pass-through switch so it can restore v8b).

**Tech Stack:** JAX/flax-nnx, gensbi, heal-swin-nnx, sbibm-jax, healpy, blackjax (adjusted MCLMC), HTCondor.

**Spec:** `docs/superpowers/specs/2026-08-14-spherical-grf-exp8-design.md` (read it first; it carries the motivation, the H-INFO/H-AMP hypotheses, and the P1 result — P1 is DONE and PASSED, do not redo it).

## Global Constraints

- Working dir for all commands: `examples/sbi-benchmarks/spherical_grf/` (paths below relative to it unless they start with `docs/`).
- Every command that touches data MUST run with `SBIBM_JAX_USE_TEST=1` (the condor runner `sub/run_spherical_grf.sh` exports it; local commands must export it explicitly).
- Local validation: `JAX_PLATFORMS=cpu`, python `/lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python` (has gensbi, sbibm_jax, healpy 1.19, blackjax).
- Condor runner contract: `arguments = "$(example_dir) $(config) $(script)"` → it cds to example_dir and runs `python $3 --config config/$2`. Script paths with subdirectories (e.g. `v8_probes/p3_amplitude_probe.py`) are valid for `$3`.
- All condor jobs: A100 (`+UseNvidiaA100 = True`), 8 CPU / 32 GB / 1 GPU. NEVER H100.
- exp-7 checkpoint (`checkpoints/spherical_grf_fm_healpix_7/`, orbax step 7) is read-only. v8 runs write `checkpoints/spherical_grf_fm_healpix_8a/` and `_8b/`.
- Dataset frozen (100k train sims); augmentation stays ON for v8 training; no dropout; no weight-decay change; EMA unchanged; raw map is the only model input.
- Do NOT modify anything under `/lhome/ific/a/aamerio/data/github/sbibm-jax` (canonical references included). P2 writes only inside `v8_probes/`.
- Test convention: no pytest. Scripts get a `QUICK=1` env mode (tiny CPU run); "failing test first" maps to "run before the feature exists → observe the documented failure; run after → observe the documented pass".
- QUICK/scratch artifacts must not be committed; every task lists exactly what to commit.
- The eval sampler standard is dopri5 (adaptive, atol=rtol=1e-5) with euler_p01 kept only for cross-experiment comparability.
- The user is away: never block on user input. Decision points in this plan carry explicit machine-checkable criteria; record every gate decision in the SDD ledger.

---

### Task 1: P3 amplitude-equivariance probe script + GPU submission

**Files:**
- Create: `v8_probes/p3_amplitude_probe.py`
- Create: `sub/spherical_grf_p3.sub`

**Interfaces:**
- Consumes: `eval-spherical-grf-diag.py` loaded as a module via importlib (gives `make_datasets`, `SphericalGRFModel`, `make_pipeline`, `prep_x`, `SEED`, `nnx`); config `config/config_healpix_7diag_ode.yaml` (EMA weights, exp-7 checkpoint); `ds.get_observation` is NOT in TaskDataset — observations come via `ds.get_reference(i)` which returns `(x_raw, ref_samples)`.
- Produces (filenames carry the config's EXPERIMENT_ID; for this config that is `spherical_grf_fm_healpix_7diag_ode`): `v8_probes/p3_results_spherical_grf_fm_healpix_7diag_ode.{txt,npz}` (slope fits, per-obs per-delta moments) and `v8_probes/dopri5_samples_spherical_grf_fm_healpix_7diag_ode_obs{1,2,3}.npz` (10k unnormalized-θ samples each, key `samples`) consumed by Task 4. The same script re-run with an 8a/8b diag config (Task 9) produces the analogous files for v8.

- [ ] **Step 1: Write the probe script**

Create `v8_probes/p3_amplitude_probe.py`:

```python
"""P3: amplitude-equivariance probe on the frozen exp-7 checkpoint.

Scaling a map by e^{delta/2} is exactly a logA shift of +delta
(ln C_l = logA + n x + a x^2/2). For obs 1-3 and a delta ladder, sample the
flow posterior from the scaled map and fit slope(mean logA vs delta).
Slope ~1: no end-to-end amplitude compression (H-AMP weakened).
Slope <0.9 or saturation at |delta|=0.4: H-AMP confirmed.

Also dumps 10k dopri5 samples per obs at delta=0 for P2's C2ST.

QUICK=1 (CPU shape check): obs (1,), deltas (0, 0.4), 16 samples, Euler 0.25,
no sample dump. Full mode is meant for the GPU condor job (sub/spherical_grf_p3.sub).
"""

import os
import sys
import importlib.util

QUICK = bool(os.environ.get("QUICK"))

# Load the diag eval script as a module (it parses --config from sys.argv).
spec = importlib.util.spec_from_file_location("diag", "eval-spherical-grf-diag.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

import jax
import numpy as np

DELTAS = (0.0, 0.4) if QUICK else (-0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4)
OBS = (1,) if QUICK else (1, 2, 3)
NSAMP = 16 if QUICK else 2000
NDUMP = 0 if QUICK else 10000
KW = ({"step_size": 0.25, "method": "Euler"} if QUICK
      else {"step_size": None, "method": "Dopri5"})

ds, tl, vl = m.make_datasets()
model = m.SphericalGRFModel(rngs=m.nnx.Rngs(m.SEED))
pipe = m.make_pipeline(model, tl, vl)
pipe.restore_model()
pipe._wrap_model()

lines = []
means = np.zeros((len(OBS), len(DELTAS), 3))
stds = np.zeros_like(means)
key = jax.random.PRNGKey(m.SEED + 42)

for oi, i in enumerate(OBS):
    x_raw, _ = ds.get_reference(i)
    x_raw = np.asarray(x_raw).reshape(1, -1)
    for di, d in enumerate(DELTAS):
        x_o = m.prep_x(x_raw * np.exp(d / 2.0), ds)
        key, sk = jax.random.split(key)
        s = np.asarray(pipe.sample(sk, x_o, NSAMP, chunk_size=1000,
                                   use_ema=True, **KW))
        th = np.asarray(ds.unnormalize_theta(s))[:, :, 0]
        means[oi, di] = th.mean(0)
        stds[oi, di] = th.std(0)
        lines.append(f"obs {i} delta {d:+.2f}: mean "
                     f"{np.array2string(th.mean(0), precision=4)} std "
                     f"{np.array2string(th.std(0), precision=4)}")
        print(lines[-1], flush=True)
    dd = np.asarray(DELTAS)
    slope = np.polyfit(dd, means[oi, :, 0], 1)[0]
    drift_n = means[oi, :, 1].max() - means[oi, :, 1].min()
    drift_a = means[oi, :, 2].max() - means[oi, :, 2].min()
    lines.append(f"obs {i}: logA slope = {slope:.4f} (target 1.0) | "
                 f"n drift {drift_n:.4f}, alpha drift {drift_a:.4f}")
    print(lines[-1], flush=True)

    if NDUMP:
        key, sk = jax.random.split(key)
        x_o = m.prep_x(x_raw, ds)
        s = np.asarray(pipe.sample(sk, x_o, NDUMP, chunk_size=1000,
                                   use_ema=True, **KW))
        th = np.asarray(ds.unnormalize_theta(s))[:, :, 0]
        np.savez(f"v8_probes/dopri5_samples_{m.EXPERIMENT_ID}_obs{i}.npz",
                 samples=th)
        print(f"dumped {th.shape} -> dopri5_samples_{m.EXPERIMENT_ID}_obs{i}.npz",
              flush=True)

np.savez(f"v8_probes/p3_results_{m.EXPERIMENT_ID}.npz",
         deltas=np.asarray(DELTAS), obs=np.asarray(OBS),
         means=means, stds=stds)
with open(f"v8_probes/p3_results_{m.EXPERIMENT_ID}.txt", "w") as f:
    f.write("P3 amplitude equivariance (exact truth: logA slope 1.0, zero n/alpha drift)\n"
            + "\n".join(lines) + "\n")
print("P3 done", flush=True)
```

- [ ] **Step 2: QUICK run before submitting (the local test)**

```bash
cd examples/sbi-benchmarks/spherical_grf
SBIBM_JAX_USE_TEST=1 QUICK=1 JAX_PLATFORMS=cpu \
  /lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python v8_probes/p3_amplitude_probe.py
```

Expected: PASS — prints two `obs 1 delta ...` lines and one `logA slope` line, exits 0, writes `v8_probes/p3_results_spherical_grf_fm_healpix_7diag_ode_quick.{txt,npz}` (QUICK appends `_quick` to EXPERIMENT_ID) and no sample dumps. The QUICK slope at 16 samples/4 Euler steps is noisy — do not interpret it; the run only proves the wiring. Delete the QUICK outputs: `rm -f v8_probes/p3_results_*_quick.*`.

- [ ] **Step 3: Write the submit file**

Create `sub/spherical_grf_p3.sub` with exactly:

```
# HTCondor submit file — v8 probe P3: amplitude equivariance on the frozen
# exp-7 checkpoint + dopri5 sample dumps for P2's C2ST. ~1 h on A100.
# Design: docs/superpowers/specs/2026-08-14-spherical-grf-exp8-design.md
repo_root   = /lustre/ific.uv.es/ml/ific088/github/GenSBI-examples
example_dir = $(repo_root)/examples/sbi-benchmarks/spherical_grf
script      = v8_probes/p3_amplitude_probe.py

universe   = vanilla
initialdir = $(example_dir)/sub
executable = $(example_dir)/sub/run_spherical_grf.sh
arguments  = "$(example_dir) $(config) $(script)"
getenv     = True

request_cpus   = 8
request_memory = 32 GB
request_gpus   = 1

+UseNvidiaA100 = True

log    = condor_logs/spherical_grf_p3_probe_a100.log
output = condor_logs/spherical_grf_p3_probe_a100.out
error  = condor_logs/spherical_grf_p3_probe_a100.err

config = config_healpix_7diag_ode.yaml
queue
```

- [ ] **Step 4: Dry-run then submit**

```bash
cd sub && condor_submit -dry-run /dev/stdout spherical_grf_p3.sub | grep -E "^(Arguments|Args)"; cd ..
condor_submit sub/spherical_grf_p3.sub
```

Expected: dry-run shows one job with `v8_probes/p3_amplitude_probe.py` and `config_healpix_7diag_ode.yaml`; submit prints `1 job(s) submitted to cluster <N>`. Record `<N>`. The job runs while Tasks 2–3 proceed on CPU.

- [ ] **Step 5: Commit**

```bash
git add v8_probes/p3_amplitude_probe.py sub/spherical_grf_p3.sub v8_probes/p1_grid_check.py v8_probes/p1_grid_check_results.txt docs/superpowers/specs/2026-08-14-spherical-grf-exp8-design.md docs/superpowers/plans/2026-08-14-spherical-grf-v8.md
git commit -m "feat(spherical_grf): v8 probes — P1 grid validation (passed) + P3 amplitude probe

P1: shipped MCLMC references match exact grid posterior (<=0.05 sigma,
stds within 2.5%) for all 10 obs — sampler exonerated. P3: amplitude-
equivariance probe on frozen exp-7 ckpt, one A100 job; dumps dopri5
samples for P2's C2ST."
```

---

### Task 2: P2a — anafast transfer function

**Files:**
- Create: `v8_probes/p2_transfer_function.py`

**Interfaces:**
- Consumes: canonical observations (`/lhome/ific/a/aamerio/data/github/sbibm-jax/src/sbibm_jax/tasks/spherical_grf/files/nside_64/observations.npz`, keys `observations` (10, npix) and `true_parameters` (10, 3)); healpy.
- Produces: `v8_probes/p2_transfer_wl.npz` — keys `wl` (10, 192) [per-obs transfer for ℓ=0..191, wl[:, :2] set to 1], `ell` (192,) — consumed by Task 3.

- [ ] **Step 1: Write the script**

Create `v8_probes/p2_transfer_function.py`:

```python
"""P2a: measure the anafast transfer function w_l on simulations.

For each canonical obs i: simulate M maps at theta_true(i) exactly the way
the task does (healpy synfast, lmax = 3*nside-1 = 191, new=True, no pixwin),
and estimate w_l(i) = <anafast(map)_l> / C_l(theta_true(i)).

Expectation (spec): w_l ~ 1 below l~2*nside, dipping up to ~7% for
2*nside < l <= 191, and approximately theta-independent (geometric).
CPU, ~4 min for 10 obs x 200 sims.
"""

import numpy as np
import healpy as hp

FILES = ("/lhome/ific/a/aamerio/data/github/sbibm-jax/src/sbibm_jax/tasks/"
         "spherical_grf/files/nside_64")
NSIDE, ELL0, LMAX, M = 64, 64.0, 191, 200

obs = np.load(f"{FILES}/observations.npz")
truths = obs["true_parameters"].reshape(10, 3)

def cl_target(theta):
    la, n, a = theta
    ell = np.arange(LMAX + 1)
    x = np.log(np.maximum(ell, 1) / ELL0)
    cl = np.exp(la + n * x + 0.5 * a * x * x)
    cl[:2] = 0.0
    return cl

rng = np.random.default_rng(20260814)
wl = np.ones((10, LMAX + 1))
for i in range(10):
    cl = cl_target(truths[i])
    acc = np.zeros(LMAX + 1)
    for j in range(M):
        np.random.seed(rng.integers(0, 2**31, size=4))
        mp = hp.synfast(cl, NSIDE, lmax=LMAX, new=True)
        acc += hp.anafast(mp, lmax=LMAX)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = acc / M / cl
    w[:2] = 1.0
    wl[i] = w
    print(f"obs {i+1}: w_l median(2..128)={np.median(w[2:129]):.4f} "
          f"median(129..191)={np.median(w[129:]):.4f} min={w[2:].min():.4f}",
          flush=True)

np.savez("v8_probes/p2_transfer_wl.npz", wl=wl, ell=np.arange(LMAX + 1))
spread = wl[:, 2:].std(0) / wl[:, 2:].mean(0)
print(f"cross-obs relative spread: median {np.median(spread):.4f} "
      f"max {spread.max():.4f}  (theta-independence check; expect <~0.03)")
print("P2a done -> v8_probes/p2_transfer_wl.npz")
```

- [ ] **Step 2: Run it (this is the test — expectations are printed)**

```bash
SBIBM_JAX_USE_TEST=1 /lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python v8_probes/p2_transfer_function.py
```

Expected: PASS — per-obs `median(2..128)` ≈ 1.00 (±0.01), `median(129..191)` in ~0.90–1.00 (the documented ≤7% suppression is an upper bound, not a target), cross-obs spread median < 0.03. If `median(2..128)` deviates from 1 by >3%, STOP — the simulate path doesn't match the task (check synfast args against `task.py:_simulate_one_np`) — fix before Task 3.

- [ ] **Step 3: Commit**

```bash
git add v8_probes/p2_transfer_function.py v8_probes/p2_transfer_wl.npz
git commit -m "feat(spherical_grf): v8 probe P2a — anafast transfer function w_l per obs"
```

---

### Task 3: P2b — corrected ℓ≤191 reference posteriors

**Files:**
- Create: `v8_probes/p2_corrected_reference.py`

**Interfaces:**
- Consumes: `v8_probes/p2_transfer_wl.npz` (Task 2); sbibm-jax reference machinery imported from the installed package: `from sbibm_jax.tasks.spherical_grf.reference_posterior import compute_cl_hat, theta_from_z, z_from_theta, _run_chain` and `from sbibm_jax.tasks.spherical_grf.task import cl_target`; canonical observations npz (path as Task 2).
- Produces: `v8_probes/p2_corrected_reference_obs{i}.npz` (key `samples`, (10000, 3)) for i in 1..10, and `v8_probes/p2b_results.txt` — consumed by Task 4.

- [ ] **Step 1: Write the script**

Create `v8_probes/p2_corrected_reference.py`:

```python
"""P2b: exact-likelihood reference with the full band limit l<=191, using the
measured anafast transfer function: D_l = w_l * C_l(theta).

Reuses sbibm-jax's own MCLMC machinery so the only difference from the
shipped reference is (lmax 128 -> 191, w_l correction). 4 chains x 2500,
rhat<1.01 required per obs. CPU, minutes per obs.
Writes nothing outside v8_probes/ (canonical files untouched).
"""

import jax
import jax.numpy as jnp
import numpy as np
import blackjax.diagnostics as bj_diag
from sbibm_jax.tasks.spherical_grf.reference_posterior import (
    compute_cl_hat, theta_from_z, z_from_theta, _run_chain)
from sbibm_jax.tasks.spherical_grf.task import cl_target

FILES = ("/lhome/ific/a/aamerio/data/github/sbibm-jax/src/sbibm_jax/tasks/"
         "spherical_grf/files/nside_64")
NSIDE, ELL0, LMAX = 64, 64.0, 191
LOW = jnp.array([-2.0, -3.0, -0.5])
HIGH = jnp.array([2.0, 0.0, 0.5])
NUM, CHAINS, TUNE = 10000, 4, 5000

obs = np.load(f"{FILES}/observations.npz")
maps = obs["observations"]
wl_all = np.load("v8_probes/p2_transfer_wl.npz")["wl"]

lines = []
for i in range(10):
    cl_hat = jnp.asarray(compute_cl_hat(maps[i], LMAX))[2:]
    w_ell = jnp.asarray(wl_all[i])[2:]
    ell = jnp.arange(2, LMAX + 1)
    wdeg = 2.0 * ell + 1.0

    def logdensity(z):
        theta = theta_from_z(z, LOW, HIGH)
        log_jac = jnp.sum(jnp.log(HIGH - LOW) + jax.nn.log_sigmoid(z)
                          + jax.nn.log_sigmoid(-z))
        d = w_ell * cl_target(theta, LMAX, ELL0)[2:]
        loglik = -0.5 * jnp.sum(wdeg * (cl_hat / d + jnp.log(d)))
        return loglik + log_jac

    per_chain = -(-NUM // CHAINS)
    keys = jax.random.split(jax.random.PRNGKey(20260814 + i), CHAINS + 1)
    zs = []
    for c in range(CHAINS):
        u = jax.random.uniform(jax.random.fold_in(keys[0], c), (3,),
                               minval=0.05, maxval=0.95)
        z0 = z_from_theta(LOW + (HIGH - LOW) * u, LOW, HIGH)
        z, _ = _run_chain(keys[c + 1], logdensity, z0, per_chain, TUNE, 0.9)
        zs.append(z)
    th = theta_from_z(jnp.stack(zs), LOW, HIGH)
    rhat = np.asarray(bj_diag.potential_scale_reduction(th))
    samples = np.asarray(th.reshape(-1, 3)[:NUM])
    np.savez(f"v8_probes/p2_corrected_reference_obs{i+1}.npz", samples=samples)
    lines.append(f"obs {i+1}: rhat {np.array2string(rhat, precision=4)} "
                 f"mean {np.array2string(samples.mean(0), precision=4)} "
                 f"std {np.array2string(samples.std(0), precision=4)}")
    print(lines[-1], flush=True)
    assert rhat.max() < 1.01, f"obs {i+1} rhat {rhat.max():.4f} >= 1.01"

with open("v8_probes/p2b_results.txt", "w") as f:
    f.write("P2b corrected reference (lmax 191, w_l-corrected likelihood)\n"
            + "\n".join(lines) + "\n")
print("P2b done")
```

- [ ] **Step 2: Run it**

```bash
SBIBM_JAX_USE_TEST=1 JAX_PLATFORMS=cpu \
  /lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python v8_probes/p2_corrected_reference.py
```

Expected: PASS — 10 obs, every r̂ < 1.01 (the script asserts), stds strictly SMALLER than the shipped reference stds (more modes → tighter; compare against Task-1-committed `p1_grid_check_results.txt` grid stds: corrected n/α stds should shrink by roughly 20–40%). If any r̂ fails, re-run that obs with `TUNE = 20000` (edit the constant) before escalating.

- [ ] **Step 3: Commit**

```bash
git add v8_probes/p2_corrected_reference.py v8_probes/p2b_results.txt v8_probes/p2_corrected_reference_obs*.npz
git commit -m "feat(spherical_grf): v8 probe P2b — corrected l<=191 reference posteriors (rhat<1.01)"
```

---

### Task 4: P2c — C2ST against the corrected reference (needs Task 1's GPU job)

**Files:**
- Create: `v8_probes/p2_c2st_compare.py`

**Interfaces:**
- Consumes: `v8_probes/dopri5_samples_spherical_grf_fm_healpix_7diag_ode_obs{1,2,3}.npz` (Task 1 GPU job — wait for it: `condor_q <N>`; on completion check `sub/condor_logs/spherical_grf_p3_probe_a100.err` for tracebacks), `v8_probes/p2_corrected_reference_obs{1,2,3}.npz` (Task 3), shipped references (`files/nside_64/reference_posterior_samples.npz`), `from gensbi.diagnostics.metrics import c2st`.
- Produces: `v8_probes/p2c_results_spherical_grf_fm_healpix_7diag_ode.txt` — the H-INFO verdict, consumed by Task 5's gate. (The script takes `FLOW_PREFIX` from the environment so Task 9 can reuse it on v8 samples.)

- [ ] **Step 1: Write the comparison script**

Create `v8_probes/p2_c2st_compare.py`:

```python
"""P2c: is the flow's C2ST gap explained by the reference's l<=128 cutoff?

For obs 1-3: C2ST(exp-7 dopri5 flow samples vs shipped l<=128 reference)
vs C2ST(same flow samples vs corrected l<=191 reference), plus moment table.
H-INFO confirmed if C2ST drops substantially (e.g. 0.82 -> <=0.65) and the
flow-vs-reference mean displacement shrinks.
"""

import os
import numpy as np
import jax.numpy as jnp
from gensbi.diagnostics.metrics import c2st

# FLOW_PREFIX selects whose samples to compare: exp-7 by default; Task 9
# re-runs this with FLOW_PREFIX=spherical_grf_fm_healpix_8adiag_ode etc.
PREFIX = os.environ.get("FLOW_PREFIX", "spherical_grf_fm_healpix_7diag_ode")

FILES = ("/lhome/ific/a/aamerio/data/github/sbibm-jax/src/sbibm_jax/tasks/"
         "spherical_grf/files/nside_64")
shipped = np.load(f"{FILES}/reference_posterior_samples.npz")["samples"]

lines = []
for i in (1, 2, 3):
    flow = np.load(f"v8_probes/dopri5_samples_{PREFIX}_obs{i}.npz")["samples"]
    corr = np.load(f"v8_probes/p2_corrected_reference_obs{i}.npz")["samples"]
    ship = shipped[i - 1]
    n = min(len(flow), len(ship), len(corr))
    rng = np.random.default_rng(1000 + i)
    f = flow[rng.choice(len(flow), n, replace=False)]
    c_old = float(c2st(jnp.asarray(f), jnp.asarray(
        ship[rng.choice(len(ship), n, replace=False)]), seed=7))
    c_new = float(c2st(jnp.asarray(f), jnp.asarray(
        corr[rng.choice(len(corr), n, replace=False)]), seed=7))
    disp_old = (flow.mean(0) - ship.mean(0)) / ship.std(0)
    disp_new = (flow.mean(0) - corr.mean(0)) / corr.std(0)
    lines.append(
        f"obs {i}: C2ST shipped {c_old:.3f} -> corrected {c_new:.3f}\n"
        f"  flow mean {np.array2string(flow.mean(0), precision=4)} "
        f"std {np.array2string(flow.std(0), precision=4)}\n"
        f"  shipped   {np.array2string(ship.mean(0), precision=4)} "
        f"std {np.array2string(ship.std(0), precision=4)} "
        f"disp/sigma {np.array2string(disp_old, precision=2)}\n"
        f"  corrected {np.array2string(corr.mean(0), precision=4)} "
        f"std {np.array2string(corr.std(0), precision=4)} "
        f"disp/sigma {np.array2string(disp_new, precision=2)}")
    print(lines[-1], flush=True)

with open(f"v8_probes/p2c_results_{PREFIX}.txt", "w") as f:
    f.write(f"P2c ({PREFIX}): flow-vs-reference C2ST, shipped (l<=128) vs corrected (l<=191)\n"
            "H-INFO confirmed if corrected C2ST drops >=0.10 below shipped on >=2 of 3 obs\n\n"
            + "\n".join(lines) + "\n")
print("P2c done")
```

- [ ] **Step 2: Wait for Task 1's GPU job, then run**

```bash
# poll until the P3 cluster finishes (check every few minutes; ~1 h runtime)
condor_q <N> -nobatch
tail -5 sub/condor_logs/spherical_grf_p3_probe_a100.err   # no tracebacks
SBIBM_JAX_USE_TEST=1 JAX_PLATFORMS=cpu \
  /lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python v8_probes/p2_c2st_compare.py
```

Expected: PASS — three `C2ST shipped ... -> corrected ...` lines. Record the verdict per the criterion in the file header. (Either direction is a valid scientific result; the task fails only on crash/missing inputs.)

- [ ] **Step 3: Commit (probe outputs + P3's results)**

```bash
git add v8_probes/p2_c2st_compare.py v8_probes/p2c_results_*.txt v8_probes/p3_results_*.txt v8_probes/p3_results_*.npz
git commit -m "results(spherical_grf): v8 probes P2c + P3 — H-INFO/H-AMP verdicts"
```

(The 10k-sample npz dumps stay uncommitted — large and reproducible.)

---

### Task 5: Gate — pick v8b's lever, record the decision

**Files:**
- None (ledger + commit-message record only).

**Interfaces:**
- Consumes: `v8_probes/p3_results_spherical_grf_fm_healpix_7diag_ode.txt` (slope), `v8_probes/p2c_results_spherical_grf_fm_healpix_7diag_ode.txt` (H-INFO verdict).
- Produces: the decision `V8B_LEVER ∈ {passthrough, capacity}` used verbatim by Tasks 6–7.

- [ ] **Step 1: Apply the gate criterion**

Read `v8_probes/p3_results_spherical_grf_fm_healpix_7diag_ode.txt`. Compute the mean logA slope over obs 1–3.

- Mean slope < 0.93 → `V8B_LEVER = passthrough` (H-AMP confirmed; spec Option 2).
- Mean slope ≥ 0.93 AND every |n drift|, |alpha drift| < 0.02 → `V8B_LEVER = capacity` (H-AMP weakened; v8b uses the reserved capacity bump instead: `embed_dim: 96`, `depths: [2, 2, 6, 4]`, `global_depth: 4`, `num_heads: [4, 8, 16, 16]` unchanged, `global_axes_dim: [22, 22, 20]` unchanged — note embed_dim 96 → num_features 768 and features/heads 96, so `global_axes_dim` must become `[32, 32, 32]` to sum to 96).
- Mean slope ≥ 0.93 but drifts ≥ 0.02 (equivariance broken in shape, not scale): treat as `passthrough` (the pass-through also carries multiscale shape information) and note the anomaly in the ledger.

- [ ] **Step 2: Record**

Append to the SDD ledger: `Gate: V8B_LEVER=<value> (P3 slopes <numbers>; P2c verdict <one line>)`. The P2c verdict does NOT change v8 — it changes the final report's interpretation only (spec: canonical references are the user's decision).

---

### Task 6: v8 training-script change — power pass-through switch (both scripts)

**Files:**
- Modify: `train-spherical-grf-v2.py` (SphericalGRFModel + config constant)
- Modify: `eval-spherical-grf-diag.py` (same change, so post-run eval can restore v8b)

**Interfaces:**
- Consumes: `encoder.power_passthrough` YAML bool (default false); NEST ordering of the input map (`cond` argument, shape (B, 49152, 1), dataset-normalized).
- Produces: `SphericalGRFModel` optionally containing `self.power_proj: nnx.Linear(3, COND_FEATURES)` zero-initialized, and `_power_tokens(cond) -> (B, 192, 3)`. Checkpoint compatibility: with the switch OFF the param tree is byte-identical to exp-7's.

**Do this task even if `V8B_LEVER = capacity`** — the switch defaults to false everywhere, v8b's config then simply doesn't set it, and the code is exercised by SMOKE. (If `V8B_LEVER = capacity`, `config_healpix_8b.yaml` in Task 7 uses the capacity values from Task 5 instead of `power_passthrough: true`.)

- [ ] **Step 1: The failing test**

```bash
SBIBM_JAX_USE_TEST=1 SMOKE=1 JAX_PLATFORMS=cpu POWER_PASSTHROUGH_SMOKE=1 \
  /lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python train-spherical-grf-v2.py --config config/config_healpix_7.yaml
```

Expected now: identical to a plain SMOKE run (the env var does nothing yet) — that IS the failure: no pass-through code exists. After Step 2 the same command must print a `power passthrough: ON` line.

- [ ] **Step 2: Implement in `train-spherical-grf-v2.py`**

(a) In the encoder-config constants block (near `GLOBAL_DEPTH = ...`), add:

```python
POWER_PASSTHROUGH = bool(CONFIG["encoder"].get("power_passthrough", False)
                         or os.environ.get("POWER_PASSTHROUGH_SMOKE"))
```

(b) In `SphericalGRFModel.__init__`, after `self.global_blocks = nnx.List(blocks)`, add:

```python
        # Raw-map power pass-through (exp-8b): the GRF sufficient statistic
        # is quadratic in the map; hand Flux1 local log-band-powers at three
        # scales so amplitude (logA) has a linear channel past the LayerNorms.
        # Zero-init => exactly the exp-7 function at init.
        self.power_proj = None
        if POWER_PASSTHROUGH:
            self.power_proj = nnx.Linear(
                3, COND_FEATURES,
                kernel_init=nnx.initializers.zeros_init(),
                dtype=ENC_DTYPE, param_dtype=ENC_PARAM_DTYPE, rngs=rngs)
```

(c) Add a method to `SphericalGRFModel` (above `__call__`):

```python
    def _power_tokens(self, cond):
        """(B, NPIX, 1) NEST map -> (B, 192, 3) log local power at
        nside 16/8/4, on the nside-4 (bottleneck-token) grid."""
        x2 = jnp.square(cond[..., 0].astype(jnp.float32))    # (B, NPIX)
        b = x2.shape[0]
        feats = []
        for s in (16, 8, 4):                 # children per nside-4 cell: 16,4,1
            p = x2.reshape(b, 12 * s * s, -1).mean(-1)       # (B, 12 s^2)
            f = jnp.log(p + 1e-12)
            feats.append(f.reshape(b, 192, -1).mean(-1))     # (B, 192)
        return jnp.stack(feats, axis=-1)                     # (B, 192, 3)
```

(d) In `SphericalGRFModel.__call__`, after the global-blocks loop, insert:

```python
        if self.power_proj is not None:
            tokens = tokens + self.power_proj(
                self._power_tokens(cond).astype(ENC_DTYPE))
```

(e) In the SMOKE block (the `if SMOKE:` section near the bottom), after the existing config print, add:

```python
    print(f"power passthrough: {'ON' if POWER_PASSTHROUGH else 'off'}")
```

- [ ] **Step 3: Run the test again — both switch states**

```bash
SBIBM_JAX_USE_TEST=1 SMOKE=1 JAX_PLATFORMS=cpu POWER_PASSTHROUGH_SMOKE=1 \
  /lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python train-spherical-grf-v2.py --config config/config_healpix_7.yaml
SBIBM_JAX_USE_TEST=1 SMOKE=1 JAX_PLATFORMS=cpu \
  /lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python train-spherical-grf-v2.py --config config/config_healpix_7.yaml
```

Expected: first prints `power passthrough: ON` and passes the forward smoke check (zero-init ⇒ output unchanged); second prints `power passthrough: off` and passes identically to before the change.

- [ ] **Step 4: Mirror the change in `eval-spherical-grf-diag.py`**

Apply exactly the same four edits (a)–(d) to `eval-spherical-grf-diag.py` (its constants block reads the merged CONFIG, so `encoder.power_passthrough` arrives via `base_config` merge). Then:

```bash
SBIBM_JAX_USE_TEST=1 SMOKE=1 JAX_PLATFORMS=cpu \
  /lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python eval-spherical-grf-diag.py --config config/config_healpix_7diag_ode.yaml
```

Expected: PASS unchanged (`cond tokens: 192 ...`, exit 0) — the switch is off for all existing configs.

- [ ] **Step 5: Commit**

```bash
git add train-spherical-grf-v2.py eval-spherical-grf-diag.py
git commit -m "feat(spherical_grf): raw-map power pass-through behind encoder.power_passthrough

3-scale log local power (nside 16/8/4, NEST reshape-means) on the 192-token
grid, zero-init Linear(3->512) added to the bottleneck tokens: a linear
amplitude channel past the LayerNorm chokepoints. Off by default; exp-7
configs and checkpoints unaffected."
```

---

### Task 7: v8 configs + submit file

**Files:**
- Create: `config/config_healpix_8a.yaml`
- Create: `config/config_healpix_8b.yaml`
- Create: `sub/spherical_grf_v8.sub`

**Interfaces:**
- Consumes: `config/config_healpix_7.yaml` (copy base), Task 5's `V8B_LEVER`, Task 6's `encoder.power_passthrough` switch.
- Produces: run_names `spherical_grf_fm_healpix_8a` / `_8b`; checkpoints under `checkpoints/<run_name>/`; the submission unit for Task 8.

- [ ] **Step 1: Create both configs**

```bash
cp config/config_healpix_7.yaml config/config_healpix_8a.yaml
cp config/config_healpix_7.yaml config/config_healpix_8b.yaml
```

Edit `config_healpix_8a.yaml`: (1) replace the header comment with `# exp-8a: exp-7 architecture, schedule extended to 600k (convergence control). Design: docs/superpowers/specs/2026-08-14-spherical-grf-exp8-design.md`; (2) `training.run_name: spherical_grf_fm_healpix_8a`; (3) `training.nsteps: 600000`. Everything else identical to config 7 (augment stays true, EMA/decay_transition/max_lr unchanged).

Edit `config_healpix_8b.yaml`: same three edits with run_name `spherical_grf_fm_healpix_8b` and header `# exp-8b: 8a + <lever>. Design: ...`, plus the lever:
- If `V8B_LEVER = passthrough`: add `power_passthrough: true` under `encoder:`.
- If `V8B_LEVER = capacity`: under `encoder:` set `embed_dim: 96`, `depths: [2, 2, 6, 4]`, `global_depth: 4`, `global_axes_dim: [32, 32, 32]`.

Verify the exact key names against the config 7 you copied (e.g. `run_name` and `nsteps` live under `training:`); if a name differs, match the existing file — it is the source of truth.

- [ ] **Step 2: SMOKE both configs**

```bash
SBIBM_JAX_USE_TEST=1 SMOKE=1 JAX_PLATFORMS=cpu \
  /lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python train-spherical-grf-v2.py --config config/config_healpix_8a.yaml
SBIBM_JAX_USE_TEST=1 SMOKE=1 JAX_PLATFORMS=cpu \
  /lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python train-spherical-grf-v2.py --config config/config_healpix_8b.yaml
```

Expected: both pass; 8b prints `power passthrough: ON` (or, capacity branch, `cond tokens: 192 features: 768 global blocks: 4`).

- [ ] **Step 3: QUICK 8b end-to-end (train path, tiny)**

```bash
SBIBM_JAX_USE_TEST=1 QUICK=1 JAX_PLATFORMS=cpu \
  /lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python train-spherical-grf-v2.py --config config/config_healpix_8b.yaml
```

Expected: PASS (few-step train + restore + tiny eval; ~10–20 min CPU). This is the only end-to-end check the new code path gets before GPU — do not skip. Delete QUICK debris:

```bash
rm -rf checkpoints/spherical_grf_fm_healpix_8b_quick
rm -f spherical_grf_fm_healpix_8b_quick_results.txt imgs/spherical_grf_fm_healpix_8b_quick_*.png
```

- [ ] **Step 4: Write the submit file**

Create `sub/spherical_grf_v8.sub` with exactly:

```
# HTCondor submit file — exp-8: two 600k-step training runs on A100.
# 8a = convergence control (exp-7 arch), 8b = 8a + v8b lever (see configs).
# ~45 h each (exp-7: 300k in ~23 h). Design:
# docs/superpowers/specs/2026-08-14-spherical-grf-exp8-design.md
repo_root   = /lustre/ific.uv.es/ml/ific088/github/GenSBI-examples
example_dir = $(repo_root)/examples/sbi-benchmarks/spherical_grf
script      = train-spherical-grf-v2.py

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

config = config_healpix_8a.yaml
queue
config = config_healpix_8b.yaml
queue
```

- [ ] **Step 5: Dry-run and commit**

```bash
cd sub && condor_submit -dry-run /dev/stdout spherical_grf_v8.sub | grep -E "^(Arguments|Args)"; cd ..
git add config/config_healpix_8a.yaml config/config_healpix_8b.yaml sub/spherical_grf_v8.sub
git commit -m "feat(spherical_grf): exp-8 configs (600k control + <lever>) and submit file"
```

(Substitute `<lever>` in the commit message with the actual Task 5 decision: `power pass-through` or `capacity bump`.)

Expected dry-run: two job ads naming the two configs and `train-spherical-grf-v2.py`.

---

### Task 8: Submit v8 and verify startup

**Files:** none (cluster state).

- [ ] **Step 1: Pre-submission sanity**

```bash
git status --short        # expect: clean (only untracked v8_probes/*.npz dumps allowed)
ls checkpoints/           # expect: no spherical_grf_fm_healpix_8* dirs yet
```

- [ ] **Step 2: Submit and verify**

```bash
condor_submit sub/spherical_grf_v8.sub     # expect: 2 job(s) submitted to cluster <M>
condor_q <M> -nobatch
```

Once a job runs, wait ~10 min then check its `.out` for the config banner and the first training-step lines, and `.err` for tracebacks:

```bash
tail -20 sub/condor_logs/spherical_grf_config_healpix_8a_a100.out
tail -5  sub/condor_logs/spherical_grf_config_healpix_8a_a100.err
```

A traceback before the first training step → `condor_rm <M>`, debug, resubmit. Idle jobs are normal on a busy queue — do NOT remove idle jobs.

- [ ] **Step 3: Record**

Ledger: `v8 submitted — cluster <M> (8a, 8b), startup verified/pending`.

---

### Task 9: Monitor, auto-eval, final report

**Files:**
- Create: `config/config_healpix_8a_diag_ode.yaml`, `config/config_healpix_8a_diag_tarp.yaml`, `config/config_healpix_8b_diag_ode.yaml`, `config/config_healpix_8b_diag_tarp.yaml`
- Create: `sub/spherical_grf_v8_eval.sub`
- Create: `v8_probes/v8_final_report.md`

**Interfaces:**
- Consumes: finished v8 checkpoints (orbax step 8 under `checkpoints/spherical_grf_fm_healpix_8{a,b}/` — the step key equals `training.experiment_id` in the config; verify with `ls`); the 7diag eval script (Task 6 version).
- Produces: `spherical_grf_fm_healpix_8{a,b}diag_{ode,tarp_ode}_results.txt`, TARP pngs, and the final report.

- [ ] **Step 1: Write the four eval configs (while training runs)**

`config/config_healpix_8a_diag_ode.yaml` (the other three follow the same pattern):

```yaml
# exp-8a post-run eval: 7diag matrix on the 8a checkpoint.
# Design: docs/superpowers/specs/2026-08-14-spherical-grf-exp8-design.md
base_config: config/config_healpix_8a.yaml

data:
  augment: false

training:
  train_model: false
  restore_model: true

evaluation:
  observations: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  tarp_pairs: 0

diagnostic:
  mode: ode
  scatter_repeats: 3
  scatter_observations: [1, 2, 3]
  variants:
    - name: euler_p01
      kind: euler
      step_size: 0.01
      baseline: true
    - name: dopri5
      kind: dopri5
```

`config_healpix_8a_diag_tarp.yaml`: same header/base_config/data/training, then

```yaml
evaluation:
  observations: []
  tarp_pairs: 200
  tarp_posterior_samples: 1000

diagnostic:
  mode: tarp_ode
  variants:
    - name: dopri5
      kind: dopri5
```

`8b` variants: identical with `base_config: config/config_healpix_8b.yaml`. Output stems come out as `spherical_grf_fm_healpix_8adiag_ode` etc. (run_name + "diag_" + mode) — distinct per run, no collisions.

`sub/spherical_grf_v8_eval.sub`: copy `sub/spherical_grf_7diag.sub` and replace the four `config = ...` lines with the four new eval configs (script stays `eval-spherical-grf-diag.py`).

Commit these now (they're inert until submitted):

```bash
git add config/config_healpix_8*_diag_*.yaml sub/spherical_grf_v8_eval.sub
git commit -m "feat(spherical_grf): exp-8 post-run eval configs + submit file"
```

- [ ] **Step 2: Watch the training cluster**

Use a harness-tracked background loop (NOT nohup) polling `condor_q <M>` every ~30 min; on each wake also `grep -c "loss" sub/condor_logs/spherical_grf_config_healpix_8a_a100.out` style progress checks. When BOTH jobs leave the queue: check exit codes (`condor_history <M> -af:j ExitCode` — both 0), confirm `checkpoints/spherical_grf_fm_healpix_8a/` and `_8b/` contain the orbax step dir, then submit the eval:

```bash
condor_submit sub/spherical_grf_v8_eval.sub
```

If a training job exits nonzero: collect the last 50 lines of its `.err`/`.out` into the ledger, do NOT resubmit blindly; if the other run is healthy, continue with its eval alone and flag the failure prominently in the final report.

- [ ] **Step 3: v8 sample dumps + amplitude re-probe (one condor job per run)**

Copy `sub/spherical_grf_p3.sub` to `sub/spherical_grf_p3_v8.sub`, change the `config = ...` line to `config_healpix_8a_diag_ode.yaml` and add a second `config = config_healpix_8b_diag_ode.yaml` / `queue` pair (log/output/error already use `$Fn(config)`? No — the p3 sub file uses fixed log names: change the three log lines to the `$Fn(config)` pattern from `sub/spherical_grf_v8.sub` first). Submit it. Each job re-runs `v8_probes/p3_amplitude_probe.py` against a v8 checkpoint, writing `p3_results_spherical_grf_fm_healpix_8{a,b}diag_ode.{txt,npz}` and `dopri5_samples_spherical_grf_fm_healpix_8{a,b}diag_ode_obs{1,2,3}.npz`. When done, run the corrected-reference comparison for both:

```bash
for P in spherical_grf_fm_healpix_8adiag_ode spherical_grf_fm_healpix_8bdiag_ode; do
  SBIBM_JAX_USE_TEST=1 JAX_PLATFORMS=cpu FLOW_PREFIX=$P \
    /lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python v8_probes/p2_c2st_compare.py
done
```

Key readouts: v8b's logA slope (pass-through branch: expect → 1.0; exp-7's slope is the Task 1 baseline) and both runs' C2ST against shipped AND corrected references.

- [ ] **Step 4: Collect and compare**

When the eval cluster finishes (exit 0 ×4), extract the numbers:

```bash
grep -E "C2ST" spherical_grf_fm_healpix_8adiag_ode_results.txt
grep -E "C2ST" spherical_grf_fm_healpix_8bdiag_ode_results.txt
grep -E "TARP" spherical_grf_fm_healpix_8*diag_tarp_ode_results.txt
```

Write `v8_probes/v8_final_report.md` with these sections (fill every one):
1. **Probe verdicts** — P1 (passed, from its results file), P2 (w_ℓ summary; corrected-reference stds vs shipped; C2ST shipped→corrected per obs; H-INFO verdict per the ≥0.10-drop criterion), P3 (slopes; H-AMP verdict; gate decision taken).
2. **v8a vs exp-7** — table: val loss (from each run's `..._results.txt` training line), euler_p01 and dopri5 C2ST obs 1–3 (exp-7 EMA baselines: euler_p01 0.855/0.841/0.728, dopri5 0.822/0.814/0.700), TARP 3 percentiles (exp-7: 66.95/90.40/99.43). Verdict: convergence gain yes/no.
3. **v8b vs v8a** — same table between the two runs; for the passthrough branch additionally the logA std ratio flow/reference on obs 1–3 (exp-7 was ~1.3; target ~1.0). Verdict: lever paid off yes/no.
4. **Recommendation** — what exp-9 should be, and the benchmark-definition question (shipped ℓ≤128 vs corrected ℓ≤191 references) stated as a decision for the user with the P2 evidence summarized. Do not change sbibm-jax.

Commit results + report:

```bash
git add spherical_grf_fm_healpix_8*diag_*_results.txt imgs/spherical_grf_fm_healpix_8*diag_* v8_probes/v8_final_report.md
git commit -m "results(spherical_grf): exp-8 a/b eval + final report"
```

- [ ] **Step 5: Update memory**

Update `/lhome/ific/a/aamerio/.claude/projects/-lustre-ific-uv-es-ml-ific088-github-GenSBI-examples/memory/spherical-grf-exp6-global-attention.md` (and its MEMORY.md index line): append an exp-8 section with the probe verdicts, run outcomes, and the open benchmark question. Keep it under ~15 lines.
