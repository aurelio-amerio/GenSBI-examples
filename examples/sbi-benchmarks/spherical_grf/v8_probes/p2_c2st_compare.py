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
