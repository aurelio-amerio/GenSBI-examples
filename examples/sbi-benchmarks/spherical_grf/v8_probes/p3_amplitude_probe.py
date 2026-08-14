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

# eval-spherical-grf-diag.py's `if __name__ != "__main__": os.environ[
# "JAX_PLATFORMS"] = "cpu"` guard (meant to keep grain worker subprocesses
# that re-import the script off the GPU) fires unconditionally when we load
# it here via importlib, since its __name__ is "diag", never "__main__" —
# it would silently force this GPU probe onto CPU. JAX reads JAX_PLATFORMS
# only at first backend initialization and caches the result, so lock the
# backend here, before that module is loaded, and any later overwrite of
# the env var is inert.
os.environ.setdefault("JAX_PLATFORMS", "cpu" if QUICK else "cuda")
import jax
jax.devices()  # force backend init now, before the diag module's guard runs

# Load the diag eval script as a module (it parses --config from sys.argv).
spec = importlib.util.spec_from_file_location("diag", "eval-spherical-grf-diag.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

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
