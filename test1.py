# %%
import os

# select device

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp

import pytest

import numpy as np

from gensbi_examples.tasks import get_task

# ("two_moons", "conditional"),
# ("bernoulli_glm", "conditional"),
# ("gaussian_linear", "conditional"),
# ("gaussian_linear_uniform", "conditional"),
# ("gaussian_mixture", "conditional"),
# ("slcp", "conditional"),
tasks = [
    "two_moons",
    "bernoulli_glm",
    "gaussian_linear",
    "gaussian_linear_uniform",
    "gaussian_mixture",
    "slcp",
]
# %%
task_name = "two_moons"
kind = "conditional"
task = get_task(task_name, kind, use_multiprocessing=False, normalize_data=True)
# %%
# write to a file the mean and std of the data
with open("mean_std_two_moons.txt", "w") as f:
    f.write("obs_mean, obs_std, cond_mean, cond_std")
    f.write(f"{task.obs_mean}, {task.obs_std}, {task.cond_mean}, {task.cond_std}")

data_dict = {
    "obs_mean": task.obs_mean,
    "obs_std": task.obs_std,
    "cond_mean": task.cond_mean,
    "cond_std": task.cond_std,
}
np.savez("mean_std_two_moons.npz", **data_dict)
# %%
