# %% using VAE encoder
import os

experiment = 1

if __name__ != "__main__":
    os.environ["JAX_PLATFORMS"] = "cpu"
else:
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90" # use 90% of GPU memory
    os.environ["JAX_PLATFORMS"] = "cuda"  # change to 'cpu' if no GPU is available

import gc

from datasets import load_dataset

import grain

import jax
from jax import numpy as jnp

import numpy as np

from flax import nnx

from gensbi.experimental.models.autoencoders import (
    AutoEncoder1D,
    AutoEncoderParams,
    vae_loss_fn,
)
from gensbi.experimental.models.autoencoders.commons import Loss

from gensbi.utils.plotting import plot_marginals


import optax

from jax import Array

from tqdm import tqdm

import matplotlib.pyplot as plt

from gensbi.models import Flux1Params, Flux1
from gensbi.recipes import ConditionalFlowPipeline

# imports
from gensbi_validation import PosteriorWrapper
from sbi.analysis.plot import sbc_rank_plot
from sbi.diagnostics import check_sbc, check_tarp, run_sbc, run_tarp
from sbi.analysis.plot import plot_tarp

import torch


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
        
    def __call__(self, 
                    t: Array,
                    obs: Array,
                    obs_ids: Array,
                    cond: Array,
                    cond_ids: Array,
                    conditioned: bool | Array = True,
                    guidance: Array | None = None,
                    encoder_key = None):
        
        # first we encode the conditioning data
        cond1 = self.vae.encode(cond[..., 0:1], encoder_key)  # (B, 100)
        cond2 = self.vae.encode(cond[..., 1:2], encoder_key)  # (B, 100)
        cond_latent = jnp.concatenate([cond1, cond2], axis=1)  # (B, 2, 100)
        
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

dim_obs = 2 # dimension of the observation (theta)
dim_cond = 8192  # not used since we use a VAE for the conditionals
ch_obs = 1 # we have 1 channel for the observation (theta)
ch_cond = 1  # not used since we use a VAE for the conditionals

z_ch = 128

effective_batch_size = 256
batch_size = 256

max_lr = 1e-4
min_lr = 1e-6

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
    xs_mean = jnp.array([[[ 0.00051776, -0.00040733]]], dtype=jnp.bfloat16) 
    thetas_mean = jnp.array([[44.826576, 45.070328]], dtype=jnp.bfloat16) 

    xs_std = jnp.array([[[60.80799, 59.33193]]], dtype=jnp.bfloat16) 
    thetas_std = jnp.array([[20.189356, 20.16127 ]], dtype=jnp.bfloat16) 


    ae_params = AutoEncoderParams(
        resolution=dim_cond,
        in_channels=ch_cond,
        ch=32,
        out_ch=ch_cond,
        ch_mult=[
            1,  # 8192
            2,  # 4096
            4,  # 2048
            8,  # 1024
            16,  # 512
            16,  # 256
            16,  # 128
            16,  # 64
            16, # 32
            16, # 16
            # 16, # 8
            # 16, # 4
        ],
        num_res_blocks=1,
        z_channels=z_ch,
        # scale_factor=0.3611,
        # shift_factor=0.1159,
        scale_factor=1.0,
        shift_factor=0.0,
        rngs=nnx.Rngs(42),
        param_dtype=jnp.bfloat16,
    )

    # define the vae model
    vae_model = AutoEncoder1D(ae_params)

    # for the sake of the NPE, we delete the decoder model as it is not needed
    vae_model.Decoder1D = None
    # run the garbage collector to free up memory
    gc.collect()

    # now we define the NPE pipeline
    # get the latent dimensions from the autoencoder
    dim_cond_latent = vae_model.latent_shape[1]*2 # we have 2 latent vectors (one for each channel)

    # dim_joint = dim_obs + dim_cond_latent  # not used for this script

    params_flux = Flux1Params(
        in_channels=ch_obs,
        vec_in_dim=None,
        context_in_dim=z_ch,
        mlp_ratio=4,
        num_heads=4,
        depth=4,
        depth_single_blocks=8,
        axes_dim=[
            20,
        ],
        dim_obs=dim_obs,
        dim_cond=dim_cond_latent,
        theta = 10*dim_cond_latent,
        qkv_bias=True,
        guidance_embed=False,
        id_embedding_kind=("absolute", "pos1dd"),
        rngs=nnx.Rngs(0),
        param_dtype=jnp.bfloat16,
    )

    model_sbi = Flux1(params_flux)

    # full model with VAE encoding the conditionals
    model = GWModel(vae_model, model_sbi)

    def split_data(batch):
        obs = jnp.array(batch["thetas"], dtype=jnp.bfloat16)
        obs = normalize(obs, thetas_mean, thetas_std)
        obs = obs.reshape(obs.shape[0], dim_obs, ch_obs)
        cond = jnp.array(batch["xs"], dtype=jnp.bfloat16)
        cond = normalize(cond, xs_mean, xs_std)
        return obs, cond

    multistep = effective_batch_size // batch_size

    train_dataset_npe = (
        grain.MapDataset.source(df_train)
        .shuffle(42)
        .repeat()
        .to_iter_dataset()
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

    training_config = ConditionalFlowPipeline._get_default_training_config()
    training_config["checkpoint_dir"] = (
        "/home/zaldivar/symlinks/aure/Github/GenSBI-examples/tests/gw_npe_v7/checkpoints"
    )
    training_config["experiment_id"] = experiment
    training_config["multistep"] = multistep
    training_config["val_every"] = 100*multistep  # validate every 100 effective steps
    training_config["max_lr"] = max_lr
    training_config["min_lr"] = min_lr
    training_config["early_stopping"] = True

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

    pipeline_latent.train(nnx.Rngs(0), 50_000*multistep, save_model=True)
    # pipeline_latent.restore_model()

    # plot the results

    x_o = df_test["xs"][0][None, ...]
    x_o = normalize(jnp.array(x_o, dtype=jnp.bfloat16), xs_mean, xs_std)

    theta_true = df_test["thetas"][0]  # already unnormalized

    samples = pipeline_latent.sample(
        nnx.Rngs(0).sample(), x_o, 10_000, encoder_key=jax.random.PRNGKey(1234)
    )
    print("Samples shape:", samples.shape)
    res = samples[:, :, 0]  # shape (num_samples, 2, 1) -> (num_samples, 2)
    print("Res shape:", res.shape)
    # unnormalize the results for plotting
    res_unnorm = unnormalize(res, thetas_mean, thetas_std)
    
    # these are degrees, we should compute the modulo 360 for better visualization
    res_unnorm = jnp.mod(res_unnorm, 360.0)

    # plot_marginals(res_unnorm, true_param=theta_true, range=[(0,120),(0,120)], gridsize=20)
    plot_marginals(res_unnorm, true_param=theta_true, range=[(25,75),(25,75)], gridsize=30)
    plt.savefig(f"gw_samples_v7_conf{experiment}.png", dpi=100, bbox_inches="tight")
    plt.show()
    
    
    # run tarp
    posterior = PosteriorWrapper(pipeline_latent, rngs=nnx.Rngs(1234), theta_shape=(2,1), x_shape=(8192,2), encoder_key=jax.random.PRNGKey(1234))


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
        num_posterior_samples=1000, # reduce this number to 1000 if you go OOM
    )
    
    plot_tarp(ecp, alpha)
    plt.savefig(f"gw_tarp_v7_conf{experiment}.png", dpi=100, bbox_inches="tight") # uncomment to save the figure
    plt.show()


if __name__ == "__main__":
    main()

# %%
