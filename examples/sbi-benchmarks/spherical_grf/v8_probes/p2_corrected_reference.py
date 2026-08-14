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
