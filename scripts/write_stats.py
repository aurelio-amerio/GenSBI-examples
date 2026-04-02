# %%
import os

# select device

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp

import pytest

import numpy as np

from gensbi_examples.tasks import get_task

tasks = [
    "two_moons",
    "bernoulli_glm",
    "gaussian_linear",
    "gaussian_linear_uniform",
    "gaussian_mixture",
    "slcp",
]
# %%
kind = "conditional"


def dump_stats(task_name):
    task = get_task(task_name, kind, use_prefetching=False, normalize_data=True)
    # write to a file the mean and std of the data

    data_dict = {
        "obs_mean": task.obs_mean,
        "obs_std": task.obs_std,
        "cond_mean": task.cond_mean,
        "cond_std": task.cond_std,
    }
    dir_ = "/lhome/ific/a/aamerio/data/github/GenSBI-examples/src/gensbi_examples/stats"
    np.savez(f"{dir_}/stats_{task_name}.npz", **data_dict)
    return


# %%
def dump_stats_gw():
    task = get_task("gravitational_waves")
    # write to a file the mean and std of the data
    cond = np.array(task.df_train["xs"][:])
    obs = np.array(task.df_train["thetas"][:])

    cond_mean = np.mean(cond, axis=0, keepdims=True)
    cond_std = np.std(cond, axis=0, keepdims=True)
    obs_mean = np.mean(obs, axis=0, keepdims=True)[..., None]
    obs_std = np.std(obs, axis=0, keepdims=True)[..., None]

    data_dict = {
        "obs_mean": obs_mean,
        "obs_std": obs_std,
        "cond_mean": cond_mean,
        "cond_std": cond_std,
    }
    dir_ = "/lhome/ific/a/aamerio/data/github/GenSBI-examples/src/gensbi_examples/stats"
    np.savez(f"{dir_}/stats_gravitational_waves.npz", **data_dict)
    return


def dump_stats_gl():
    task = get_task("gravitational_lensing")
    # write to a file the mean and std of the data
    cond = np.array(task.df_train["xs"][:])
    obs = np.array(task.df_train["thetas"][:])

    cond_mean = np.mean(cond, axis=0, keepdims=True)
    cond_std = np.std(cond, axis=0, keepdims=True)
    obs_mean = np.mean(obs, axis=0, keepdims=True)[..., None]
    obs_std = np.std(obs, axis=0, keepdims=True)[..., None]

    data_dict = {
        "obs_mean": obs_mean,
        "obs_std": obs_std,
        "cond_mean": cond_mean,
        "cond_std": cond_std,
    }
    dir_ = "/lhome/ific/a/aamerio/data/github/GenSBI-examples/src/gensbi_examples/stats"
    np.savez(f"{dir_}/stats_gravitational_lensing.npz", **data_dict)
    return


# %%
for task_name in tasks:
    print(task_name)
    dump_stats(task_name)
# %%
print("dumping gw stats")
dump_stats_gw()
# %%
print("dumping gl stats")
dump_stats_gl()