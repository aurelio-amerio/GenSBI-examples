"""P1: validate the shipped MCLMC reference posteriors by exact grid integration.

The spherical GRF reference likelihood (sbibm-jax reference_posterior.py) is
-2 lnL = sum_{l=2..128} (2l+1) [Chat_l/C_l(theta) + ln C_l(theta)] on the
uniform prior box. With ln C_l = logA + n x_l + 0.5 a x_l^2 the likelihood
separates: e^{-lnC} = e^{-logA} e^{-(n x + a x^2/2)} and sum w_l lnC is
linear in theta, so the full 3D grid posterior needs only a (G,G) inner
computation. Exact — no sampler. Compare mean/std/corr against the shipped
10k MCLMC samples for all 10 canonical observations.

Pure-NumPy/healpy, CPU, ~minutes. Output: v8_probes/p1_grid_check_results.txt
"""

import numpy as np
import healpy as hp

FILES = ("/lhome/ific/a/aamerio/data/github/sbibm-jax/src/sbibm_jax/tasks/"
         "spherical_grf/files/nside_64")
NSIDE, ELL0 = 64, 64.0
LMAX = 2 * NSIDE           # reference likelihood band limit (not 3*nside-1)
LOW = np.array([-2.0, -3.0, -0.5])
HIGH = np.array([2.0, 0.0, 0.5])
NGRID = 200                # per axis

obs = np.load(f"{FILES}/observations.npz")
ref = np.load(f"{FILES}/reference_posterior_samples.npz")["samples"]  # (10,10000,3)
maps = obs["observations"]                                            # (10, npix)
truths = obs["true_parameters"]                                       # (10, 3)

ell = np.arange(2, LMAX + 1)
w = 2.0 * ell + 1.0
x = np.log(np.maximum(ell, 1) / ELL0)          # (L,)
W0, W1, W2 = w.sum(), w @ x, w @ x**2

ax = [np.linspace(LOW[i], HIGH[i], NGRID) for i in range(3)]
gA = ax[0][:, None, None]                       # logA axis broadcast
shape_n = ax[1][:, None]                        # (G,1)
shape_a = ax[2][None, :]                        # (1,G)

lines = []
for i in range(10):
    cl_hat = hp.anafast(maps[i].astype(np.float64), lmax=LMAX)[2:]     # (L,)
    # S1(n,a) = sum_l w_l chat_l exp(-(n x_l + a x_l^2 / 2))   (G,G)
    expo = -(shape_n[..., None] * x + 0.5 * shape_a[..., None] * x**2)  # (G,G,L)
    S1 = np.einsum("l,ijl->ij", w * cl_hat, np.exp(expo))
    ll = -0.5 * (np.exp(-ax[0])[:, None, None] * S1[None]
                 + W0 * gA + W1 * shape_n[None] + 0.5 * W2 * shape_a[None])
    ll -= ll.max()
    p = np.exp(ll)
    p /= p.sum()

    pA, pn, pa = p.sum((1, 2)), p.sum((0, 2)), p.sum((0, 1))
    mean = np.array([pA @ ax[0], pn @ ax[1], pa @ ax[2]])
    var = np.array([pA @ ax[0]**2, pn @ ax[1]**2, pa @ ax[2]**2]) - mean**2
    std = np.sqrt(var)
    cA = ax[0] - mean[0]
    cn = ax[1] - mean[1]
    ca = ax[2] - mean[2]
    cov01 = np.einsum("ijk,i,j->", p, cA, cn)
    cov02 = np.einsum("ijk,i,k->", p, cA, ca)
    cov12 = np.einsum("ijk,j,k->", p, cn, ca)
    corr = np.array([cov01 / (std[0] * std[1]), cov02 / (std[0] * std[2]),
                     cov12 / (std[1] * std[2])])

    r = ref[i]
    rm, rs = r.mean(0), r.std(0)
    rc = np.corrcoef(r.T)
    rcorr = np.array([rc[0, 1], rc[0, 2], rc[1, 2]])
    dmean_sigma = (rm - mean) / std
    dstd_rel = rs / std - 1.0

    lines.append(
        f"obs {i+1}: grid mean {np.array2string(mean, precision=4)} "
        f"std {np.array2string(std, precision=4)} corr {np.array2string(corr, precision=3)}\n"
        f"        mclmc mean {np.array2string(rm, precision=4)} "
        f"std {np.array2string(rs, precision=4)} corr {np.array2string(rcorr, precision=3)}\n"
        f"        delta_mean/sigma {np.array2string(dmean_sigma, precision=3)} "
        f"std_ratio-1 {np.array2string(dstd_rel, precision=3)} "
        f"| truth {np.array2string(truths[i].ravel(), precision=4)}")
    print(lines[-1], flush=True)

with open("v8_probes/p1_grid_check_results.txt", "w") as f:
    f.write("P1 grid-vs-MCLMC reference validation (lmax=128 likelihood)\n")
    f.write("PASS criterion: |delta_mean|<0.05 sigma, |std_ratio-1|<0.03, corr match\n\n")
    f.write("\n".join(lines) + "\n")
print("done -> v8_probes/p1_grid_check_results.txt")
