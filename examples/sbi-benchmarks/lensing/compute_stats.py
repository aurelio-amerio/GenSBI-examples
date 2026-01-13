# %% using VAE encoder
# still need to train this
import os

os.environ["JAX_PLATFORMS"] = "cpu"

# if __name__ != "__main__":
#     os.environ["JAX_PLATFORMS"] = "cpu"
# else:
#     os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"  # use 90% of GPU memory
#     os.environ["JAX_PLATFORMS"] = "cuda"  # change to 'cpu' if no GPU is available

import gc

from datasets import load_dataset

import grain

import jax
from jax import numpy as jnp

import yaml

import numpy as np

from flax import nnx

from gensbi.experimental.models.autoencoders import (
    AutoEncoder2D,
    AutoEncoderParams,
)

from gensbi.experimental.recipes.vae_pipeline import parse_autoencoder_params
from gensbi.recipes.flux1 import parse_flux1_params, parse_training_config

from gensbi.utils.plotting import plot_marginals

from jax import Array

from tqdm import tqdm

import matplotlib.pyplot as plt

from gensbi.models import Flux1Params, Flux1
from gensbi.recipes import ConditionalFlowPipeline

# imports
from gensbi.diagnostics import run_tarp, plot_tarp
from gensbi.diagnostics import run_sbc, sbc_rank_plot
from gensbi.diagnostics import LC2ST, plot_lc2st


config_path = "./config/config_1a.yaml"


def normalize(batch, mean, std):
    mean = jnp.asarray(mean, dtype=batch.dtype)
    std = jnp.asarray(std, dtype=batch.dtype)
    return (batch - mean) / std


def unnormalize(batch, mean, std):
    mean = jnp.asarray(mean, dtype=batch.dtype)
    std = jnp.asarray(std, dtype=batch.dtype)
    return batch * std + mean


class GWModel(nnx.Module):
    def __init__(self, vae, sbi_model):
        self.vae = vae
        self.sbi_model = sbi_model

    def __call__(
        self,
        t: Array,
        obs: Array,
        obs_ids: Array,
        cond: Array,
        cond_ids: Array,
        conditioned: bool | Array = True,
        guidance: Array | None = None,
        encoder_key=None,
    ):

        # first we encode the conditioning data
        cond_latent = self.vae.encode(cond, encoder_key)

        # then we pass to the sbi model
        return self.sbi_model(
            t=t,
            obs=obs,
            obs_ids=obs_ids,
            cond=cond_latent,
            cond_ids=cond_ids,
            conditioned=conditioned,
            guidance=guidance,
        )


# %%
repo_name = "aurelio-amerio/SBI-benchmarks"

task_name = "lensing"

dataset = load_dataset(repo_name, task_name).with_format("numpy")

# %%
df_train = dataset["train"]
df_val = dataset["validation"]
df_test = dataset["test"]
# %%

# compute the mean of xs and thetas
xs_mean = np.mean(df_train["xs"], axis=(0, 1, 2), keepdims=False)
thetas_mean = np.mean(df_train["thetas"], axis=(0), keepdims=False)

xs_std = np.std(df_train["xs"], axis=(0, 1, 2), keepdims=False)
thetas_std = np.std(df_train["thetas"], axis=(0), keepdims=False)

# %%
print(xs_mean)
print(thetas_mean)
print(xs_std)
print(thetas_std)

# %%
# write the stats to a file
# data = {}
# data["xs_mean"] = xs_mean
# data["thetas_mean"] = thetas_mean
# data["xs_std"] = xs_std
# data["thetas_std"] = thetas_std

# np.savez("lensing_stats.npz", **data)
#%%
data = np.load("lensing_stats.npz") 
print(data["xs_mean"])
print(data["thetas_mean"])
print(data["xs_std"])
print(data["thetas_std"])


# %%
