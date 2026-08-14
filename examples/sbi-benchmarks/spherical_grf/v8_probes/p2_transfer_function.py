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
