# %%
import os

experiment=2

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



# we define a CNN to embed the data
class ConvEmbed(nnx.Module):
    def __init__(self, dim_cond, ch_cond, dout, *, rngs):
        features = 16
        padding = "SAME"
        self.activation = jax.nn.gelu
        
        dlin = dim_cond
        conv1 = nnx.Conv(ch_cond, features, kernel_size=(9,), strides=2, padding=padding, rngs=rngs, param_dtype=jnp.bfloat16) # 4096
        dlin = dlin//2
        bn1 = nnx.BatchNorm(features, rngs=rngs, param_dtype=jnp.bfloat16)
        
        conv2 = nnx.Conv(features, features*2, kernel_size=(6,), strides=2, padding=padding, rngs=rngs, param_dtype=jnp.bfloat16) # 2048
        dlin = dlin//2
        features *= 2 # 32
        bn2 = nnx.BatchNorm(features, rngs=rngs, param_dtype=jnp.bfloat16)
        
        conv3 = nnx.Conv(features, features*2, kernel_size=(3,), strides=2, padding=padding, rngs=rngs, param_dtype=jnp.bfloat16) # 1024
        dlin = dlin//2
        features *= 2 # 64
        bn3 = nnx.BatchNorm(features, rngs=rngs, param_dtype=jnp.bfloat16)
        
        conv4 = nnx.Conv(features, features*2, kernel_size=(3,), strides=2, padding=padding, rngs=rngs, param_dtype=jnp.bfloat16) # 512
        dlin = dlin//2
        features *= 2 # 128
        bn4 = nnx.BatchNorm(features, rngs=rngs, param_dtype=jnp.bfloat16)
        
        conv5 = nnx.Conv(features, features*2, kernel_size=(3,), strides=2, padding=padding, rngs=rngs, param_dtype=jnp.bfloat16) # 256
        dlin = dlin//2
        features *= 2 # 256
        bn5 = nnx.BatchNorm(features, rngs=rngs, param_dtype=jnp.bfloat16)
        
        conv6 = nnx.Conv(features, features*2, kernel_size=(3,), strides=2, padding=padding, rngs=rngs, param_dtype=jnp.bfloat16) # 128
        dlin = dlin//2
        features *= 2 # 512
        bn6 = nnx.BatchNorm(features, rngs=rngs, param_dtype=jnp.bfloat16)
        
        conv7 = nnx.Conv(features, features, kernel_size=(3,), strides=2, padding=padding, rngs=rngs, param_dtype=jnp.bfloat16) # 64
        dlin = dlin//2
        bn7 = nnx.BatchNorm(features, rngs=rngs, param_dtype=jnp.bfloat16)
        
        conv8 = nnx.Conv(features, features, kernel_size=(3,), strides=2, padding=padding, rngs=rngs, param_dtype=jnp.bfloat16) # 32
        dlin = dlin//2
        bn8 = nnx.BatchNorm(features, rngs=rngs, param_dtype=jnp.bfloat16)
        
        conv9 = nnx.Conv(features, features, kernel_size=(3,), strides=2, padding=padding, rngs=rngs, param_dtype=jnp.bfloat16) # 16
        dlin = dlin//2
        bn9 = nnx.BatchNorm(features, rngs=rngs, param_dtype=jnp.bfloat16)
        
        conv10 = nnx.Conv(features, features, kernel_size=(3,), strides=2, padding=padding, rngs=rngs, param_dtype=jnp.bfloat16) # 8
        dlin = dlin//2
        bn10 = nnx.BatchNorm(features, rngs=rngs, param_dtype=jnp.bfloat16)
        
        conv11 = nnx.Conv(features, features, kernel_size=(3,), strides=2, padding=padding, rngs=rngs, param_dtype=jnp.bfloat16) # 4
        dlin = dlin//2
        bn11 = nnx.BatchNorm(features, rngs=rngs, param_dtype=jnp.bfloat16)

        self.conv_layers = nnx.List([conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11])
        self.bn_layers = nnx.List([bn1, bn2, bn3, bn4, bn5, bn6, bn7, bn8, bn9, bn10, bn11])

        self.linear = nnx.Linear(int(dlin)*features, dout, rngs=rngs)
    
    def __call__(self, x):
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x)
            x = self.activation(x)
            x = self.bn_layers[i](x)
        
        #flatten x
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        
        return x



class GWModel(nnx.Module):
    def __init__(self, encoder, sbi_model):
        self.encoder = encoder
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
    ):
        # first we encode the conditioning data
        # cond: (B, N, 2)
        # per-channel encoding: (B, N, 1) -> (B, 100); stack channels on axis=1 -> (B, 2, 100)
        # cond_latent = jax.vmap(
        #     lambda x: self.encoder(x[..., None]),
        #     in_axes=1, # this is the axis of the channels we are mapping over
        #     out_axes=1, # we stack the outputs on axis=1
        # )(cond)
        
        cond1 = self.encoder(cond[..., 0:1])  # (B, 100)
        cond2 = self.encoder(cond[..., 1:2])  # (B, 100)
        cond_latent = jnp.stack([cond1, cond2], axis=1)  # (B, 2, 100)
        
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
    # compute the mean of xs and thetas
    xs_mean = jnp.array([[[ 0.00051776, -0.00040733]]], dtype=jnp.bfloat16) 
    thetas_mean = jnp.array([[44.826576, 45.070328]], dtype=jnp.bfloat16) 

    xs_std = jnp.array([[[60.80799, 59.33193]]], dtype=jnp.bfloat16) 
    thetas_std = jnp.array([[20.189356, 20.16127 ]], dtype=jnp.bfloat16) 

    
    # now we define the NPE pipeline
    dim_obs = 2 # dimension of the observation (theta)
    dim_cond = 8192  # dimension of the condition (xs)
    ch_obs = 1 # we have 1 channel for the observation (theta)
    ch_cond = 2  # we have 2 channels for the condition (xs), that is two GW detectors

    # define the vae model
    encoder = ConvEmbed(dim_cond, 1, dout=100, rngs=nnx.Rngs(0))
    
    # get the latent dimensions from the autoencoder
    dim_cond_latent = 2
    z_ch = 100

    dim_joint = dim_obs + dim_cond_latent  # used to set theta for rope

    params_flux = Flux1Params(
        in_channels=ch_obs,
        vec_in_dim=None,
        context_in_dim=z_ch,
        mlp_ratio=3,
        num_heads=4,
        depth=4,
        depth_single_blocks=8,
        axes_dim=[
            4,
        ],
        dim_obs=dim_obs,
        dim_cond=dim_cond_latent,
        theta = 10*dim_joint,
        qkv_bias=True,
        guidance_embed=False,
        rngs=nnx.Rngs(0),
        param_dtype=jnp.bfloat16,
    )

    model_sbi = Flux1(params_flux)

    # full model with VAE encoding the conditionals
    model = GWModel(encoder, model_sbi)

    def split_data(batch):
        obs = jnp.array(batch["thetas"], dtype=jnp.bfloat16)
        obs = normalize(obs, thetas_mean, thetas_std)
        obs = obs.reshape(obs.shape[0], dim_obs, ch_obs)
        cond = jnp.array(batch["xs"], dtype=jnp.bfloat16)
        cond = normalize(cond, xs_mean, xs_std)
        return obs, cond

    effective_batch_size = 512
    batch_size = 512
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
        .batch(batch_size)
        .map(split_data)
    )

    training_config = ConditionalFlowPipeline._get_default_training_config()
    training_config["checkpoint_dir"] = (
        "/home/zaldivar/symlinks/aure/Github/GenSBI-examples/tests/gw_npe_v3/checkpoints"
    )
    training_config["experiment_id"] = experiment
    training_config["multistep"] = multistep
    training_config["val_every"] = 100*multistep  # validate every 100 effective steps

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

    # pipeline_latent.train(nnx.Rngs(0), 30_000*multistep, save_model=True)
    pipeline_latent.restore_model()

    # plot the results

    x_o = df_test["xs"][0][None, ...]
    x_o = normalize(jnp.array(x_o, dtype=jnp.bfloat16), xs_mean, xs_std)

    theta_true = df_test["thetas"][0]  # already unnormalized

    samples = pipeline_latent.sample(
        nnx.Rngs(0).sample(), x_o, 10_000#, key=jax.random.PRNGKey(1234)
    )

    res = samples[:, :, 0]  # shape (num_samples, dim_obs, 1) -> (num_samples, dim_obs)
    
    # unnormalize the results for plotting
    res_unnorm = unnormalize(res, thetas_mean, thetas_std)
    
    # these are degrees, we should compute the modulo 360 for better visualization
    res_unnorm = jnp.mod(res_unnorm, 360.0)

    plot_marginals(res_unnorm, true_param=theta_true, range=[(0,120),(0,120)], gridsize=20)
    plt.savefig("gw_samples_v3.png", dpi=100, bbox_inches="tight")
    plt.show()
    
    
    # run tarp
    posterior = PosteriorWrapper(pipeline_latent, rngs=nnx.Rngs(1234), theta_shape=(2,1), x_shape=(8192,2))
    
    # key = jax.random.PRNGKey(1234)

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
    plt.savefig("gw_tarp_v3.png", dpi=100, bbox_inches="tight") # uncomment to save the figure
    plt.show()


if __name__ == "__main__":
    main()

# %%
