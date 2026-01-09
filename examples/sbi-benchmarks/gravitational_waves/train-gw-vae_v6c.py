# %% using VAE encoder
# still need to train this
import os


if __name__ != "__main__":
    os.environ["JAX_PLATFORMS"] = "cpu"
else:
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"  # use 90% of GPU memory
    os.environ["JAX_PLATFORMS"] = "cuda"  # change to 'cpu' if no GPU is available

import gc

from datasets import load_dataset

import grain

import jax
from jax import numpy as jnp

import yaml

import numpy as np

from flax import nnx

from gensbi.experimental.models.autoencoders import (
    AutoEncoder1D,
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
from gensbi_validation import PosteriorWrapper
from sbi.diagnostics import run_tarp
from sbi.analysis.plot import plot_tarp

import torch

config_path = "./config/gw_config_6c.yaml"


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


dim_obs = 2  # dimension of the observation (theta)
dim_cond = 8192  # not used since we use a VAE for the conditionals
ch_obs = 1  # we have 1 channel for the observation (theta)
ch_cond = 2  # not used since we use a VAE for the conditionals


def main():
    repo_name = "aurelio-amerio/SBI-benchmarks"

    task_name = "gravitational_waves"

    # dataset = load_dataset(repo_name, task_name).with_format("numpy")
    dataset = load_dataset(
        repo_name, task_name, cache_dir="/data/users/.cache"
    ).with_format("numpy")

    # %%
    df_train = dataset["train"]
    df_val = dataset["validation"]
    df_test = dataset["test"]

    # compute the mean of xs and thetas
    # xs_mean = np.mean(df_train["xs"], axis=(0, 1), keepdims=True)
    # thetas_mean = np.mean(df_train["thetas"], axis=0, keepdims=True)

    # xs_std = np.std(df_train["xs"], axis=(0, 1), keepdims=True)
    # thetas_std = np.std(df_train["thetas"], axis=0, keepdims=True)
    xs_mean = jnp.array([[[0.00051776, -0.00040733]]], dtype=jnp.bfloat16)
    thetas_mean = jnp.array([[44.826576, 45.070328]], dtype=jnp.bfloat16)

    xs_std = jnp.array([[[60.80799, 59.33193]]], dtype=jnp.bfloat16)
    thetas_std = jnp.array([[20.189356, 20.16127]], dtype=jnp.bfloat16)

    params_dict = parse_autoencoder_params(config_path)

    ae_params = AutoEncoderParams(
        rngs=nnx.Rngs(0),
        **params_dict,
    )

    # define the vae model
    vae_model = AutoEncoder1D(ae_params)

    # for the sake of the NPE, we delete the decoder model as it is not needed
    vae_model.Decoder1D = None
    # run the garbage collector to free up memory
    gc.collect()

    # now we define the NPE pipeline
    # get the latent dimensions from the autoencoder
    dim_cond_latent = vae_model.latent_shape[1]
    z_ch = vae_model.latent_shape[2]

    params_dict_flux = parse_flux1_params(config_path)

    params_flux = Flux1Params(
        rngs=nnx.Rngs(0),
        dim_obs=dim_obs,
        dim_cond=dim_cond_latent,
        **params_dict_flux,
    )

    model_sbi = Flux1(params_flux)

    # full model with VAE encoding the conditionals
    model = GWModel(vae_model, model_sbi)
    
    training_config = parse_training_config(config_path)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        batch_size = config["training"]['batch_size']
        nsteps = config["training"]['nsteps']
        multistep = config["training"]['multistep']
        experiment = config["training"]['experiment_id']

    def split_data(batch):
        obs = jnp.array(batch["thetas"], dtype=jnp.bfloat16)
        obs = normalize(obs, thetas_mean, thetas_std)
        obs = obs.reshape(obs.shape[0], dim_obs, ch_obs)
        cond = jnp.array(batch["xs"], dtype=jnp.bfloat16)
        cond = normalize(cond, xs_mean, xs_std)
        return obs, cond

    train_dataset_npe = (
        grain.MapDataset.source(df_train).shuffle(42).repeat().to_iter_dataset()
    )

    performance_config = grain.experimental.pick_performance_config(
        ds=train_dataset_npe,
        ram_budget_mb=1024 * 8,
        max_workers=None,
        max_buffer_size=None,
    )

    train_dataset_npe = (
        train_dataset_npe.batch(batch_size)
        .map(split_data)
        .mp_prefetch(performance_config.multiprocessing_options)
    )

    val_dataset_npe = (
        grain.MapDataset.source(df_val)
        .shuffle(42)
        .repeat()
        .to_iter_dataset()
        .batch(512)
        .map(split_data)
    )

    training_config["checkpoint_dir"] = (
        "/home/zaldivar/symlinks/aure/Github/GenSBI-examples/tests/gw_npe_v6c/checkpoints"
    )


    pipeline_latent = ConditionalFlowPipeline(
        model,
        train_dataset_npe,
        val_dataset_npe,
        dim_obs=dim_obs,
        dim_cond=dim_cond_latent,  # we are workin in the latent space of the vae
        ch_obs=ch_obs,
        ch_cond=z_ch,  # conditioning is now in the latent space
        training_config=training_config,
    )

    pipeline_latent.train(nnx.Rngs(0), nsteps * multistep, save_model=True)
    # pipeline_latent.restore_model()

    # plot the results

    x_o = df_test["xs"][0][None, ...]
    x_o = normalize(jnp.array(x_o, dtype=jnp.bfloat16), xs_mean, xs_std)

    theta_true = df_test["thetas"][0]  # already unnormalized

    samples = pipeline_latent.sample_batched(
        nnx.Rngs(0).sample(),
        x_o,
        100_000,
        chunk_size=10_000,
        encoder_key=jax.random.PRNGKey(1234),
    )
    print("Samples shape:", samples.shape)
    res = samples[:, 0, :, 0]  # shape (num_samples, 1, 2, 1) -> (num_samples, 2)
    print("Res shape:", res.shape)
    # unnormalize the results for plotting
    res_unnorm = unnormalize(res, thetas_mean, thetas_std)

    # these are degrees, we should compute the modulo 360 for better visualization
    res_unnorm = jnp.mod(res_unnorm, 360.0)

    # plot_marginals(res_unnorm, true_param=theta_true, range=[(0,120),(0,120)], gridsize=20)
    plot_marginals(
        res_unnorm, true_param=theta_true, range=[(25, 75), (25, 75)], gridsize=30
    )
    plt.savefig(f"gw_samples_v6c_conf{experiment}.png", dpi=100, bbox_inches="tight")
    plt.show()

    # run tarp
    posterior = PosteriorWrapper(
        pipeline_latent,
        rngs=nnx.Rngs(1234),
        theta_shape=(2, 1),
        x_shape=(8192, 2),
        encoder_key=jax.random.PRNGKey(1234),
    )

    # split in thetas and xs
    thetas = np.array(df_test["thetas"])[:200]
    xs = np.array(df_test["xs"])[:200]

    thetas = normalize(jnp.array(thetas, dtype=jnp.bfloat16), thetas_mean, thetas_std)
    xs = normalize(jnp.array(xs, dtype=jnp.bfloat16), xs_mean, xs_std)

    thetas_ = posterior._ravel(thetas)
    xs_ = posterior._ravel(xs)

    thetas_torch = torch.Tensor(np.asarray(thetas_, dtype=np.float32))
    xs_torch = torch.Tensor(np.asarray(xs_, dtype=np.float32))

    ecp, alpha = run_tarp(
        thetas_torch,
        xs_torch,
        posterior,
        references=None,  # will be calculated automatically.
        num_posterior_samples=1000,  # reduce this number to 1000 if you go OOM
    )

    plot_tarp(ecp, alpha)
    plt.savefig(
        f"gw_tarp_v6c_conf{experiment}.png", dpi=100, bbox_inches="tight"
    )  # uncomment to save the figure
    plt.show()


if __name__ == "__main__":
    main()

# %%
