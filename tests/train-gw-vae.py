# %%
import os

if __name__ != "__main__":
    os.environ["JAX_PLATFORMS"] = "cpu"
else:
    os.environ["JAX_PLATFORMS"] = "cuda"  # change to 'cpu' if no GPU is available

import gc

from datasets import load_dataset

import grain

import jax
from jax import numpy as jnp

import numpy as np

from flax import nnx

from gensbi.experimental.models.autoencoders import AutoEncoder1D, AutoEncoderParams, vae_loss_fn
from gensbi.experimental.models.autoencoders.commons import Loss

from gensbi.utils.plotting import plot_marginals

from gensbi.recipes import VAE1DPipeline

import optax

from tqdm import tqdm

import matplotlib.pyplot as plt

from gensbi.models import Flux1Params
from gensbi.recipes import Flux1FlowPipeline


def normalize(batch, mean, std):
    mean = jnp.asarray(mean, dtype=batch.dtype)
    std = jnp.asarray(std, dtype=batch.dtype)
    return (batch - mean) / std

def unnormalize(batch, mean, std):
    mean = jnp.asarray(mean, dtype=batch.dtype)
    std = jnp.asarray(std, dtype=batch.dtype)
    return batch * std + mean


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
    xs_mean = np.mean(df_train["xs"], axis=(0,1), keepdims=True)
    thetas_mean = np.mean(df_train["thetas"], axis=0, keepdims=True)

    xs_std = np.std(df_train["xs"], axis=(0,1), keepdims=True)
    thetas_std = np.std(df_train["thetas"], axis=0, keepdims=True)

    # %%
    df_train.shape

    # %%
    def get_obs(batch):
        xs = jnp.array(batch["xs"],dtype=jnp.bfloat16)
        return normalize(xs, xs_mean, xs_std)

    # %%
    batch_size = 128

    train_dataset_grain = (
        grain.MapDataset.source(df_train).shuffle(42).repeat().to_iter_dataset()
    )
    performance_config = grain.experimental.pick_performance_config(
        ds=train_dataset_grain,
        ram_budget_mb=1024 * 8,
        max_workers=None,
        max_buffer_size=None,
    )

    train_dataset = (
        train_dataset_grain.batch(batch_size)
        .map(get_obs)
        .mp_prefetch(performance_config.multiprocessing_options)
    )

    val_dataset = (
        grain.MapDataset.source(df_val)
        .shuffle(42)
        .repeat()
        .to_iter_dataset()
        .batch(batch_size)
        .map(get_obs)
    )

    # %%
    ae_params = AutoEncoderParams(
        resolution=8192,
        in_channels=2,
        ch=32,
        out_ch=2,
        ch_mult=[
            1,  # 8192
            2,  # 4096
            4,  # 2048
            6,  # 1024
            16,  # 512
            16, # 256
            16, # 128
            16, # 64
            # 16, # 32
            # 16, # 16
            # 16, # 8
            # 16, # 4
        ],
        num_res_blocks=1,
        z_channels=512,
        scale_factor=0.3611,
        shift_factor=0.1159,
        rngs=nnx.Rngs(42),
        param_dtype=jnp.bfloat16,
    )
    
    training_config = VAE1DPipeline._get_default_training_config()
    training_config["checkpoint_dir"] = "/home/zaldivar/symlinks/aure/Github/GenSBI-examples/tests/gw_vae/checkpoints"
    training_config["experiment_id"] = 3
    
    # %%
    pipeline = VAE1DPipeline(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        params=ae_params,
        training_config=training_config,
    )

    # %%
    pipeline.train(nnx.Rngs(0), nsteps=20000, save_model=True)
    # pipeline.restore_model()

    # an observation for testing
    x_o = df_test["xs"][0][None,...]
    x_o = normalize(jnp.array(x_o,dtype=jnp.bfloat16), xs_mean, xs_std)

    # %%
    pred = pipeline.model(x_o)
    # we need to unnormalize the prediction and observation
    pred_unnorm = unnormalize(pred, xs_mean, xs_std)
    x_o_unnorm = unnormalize(x_o, xs_mean, xs_std)

    # %%
    plt.plot(x_o_unnorm[0, :, 0], label="Original Signal")
    plt.plot(pred_unnorm[0, :, 0], label="Reconstructed Signal")
    plt.legend()
    plt.savefig(
        "gw_test.png", dpi=300, bbox_inches="tight"
    )  # uncomment to save the figure
    plt.show()
    
    
    # for the sake of the NPE, we delete the decoder model as it is not needed
    pipeline.model.Decoder1D = None
    # run the garbage collector to free up memory
    gc.collect()
    
    # now we define the NPE pipeline
    dim_obs = 1
    dim_cond = 8192 # not used since we use a VAE for the conditionals
    ch_obs = 2
    ch_cond = 2 # not used since we use a VAE for the conditionals
    
    # get the latent dimensions from the autoencoder
    dim_cond_latent = pipeline.model.latent_shape[1]
    z_ch = pipeline.model.latent_shape[2]
    
    dim_joint = dim_obs + dim_cond # not used for this script
    
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
        qkv_bias=True,
        guidance_embed=False,
        rngs=nnx.Rngs(0),
        param_dtype=jnp.bfloat16,
    )
    
    def split_data(batch):
        obs = jnp.array(batch["thetas"],dtype=jnp.bfloat16)
        obs = obs[:,None,:]
        obs = normalize(obs, thetas_mean, thetas_std)
        cond = jnp.array(batch["xs"],dtype=jnp.bfloat16)
        cond = normalize(cond, xs_mean, xs_std)
        return obs, cond
    
    batch_size = 1024
    
    train_dataset_npe = (
            grain.MapDataset.source(df_train)
            .shuffle(42)
            .repeat()
            .to_iter_dataset()
            # .batch(batch_size)
            # .map(split_data)
            # .mp_prefetch()
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
    
    training_config = Flux1FlowPipeline._get_default_training_config()
    training_config["checkpoint_dir"] = "/home/zaldivar/symlinks/aure/Github/GenSBI-examples/tests/gw_npe/checkpoints"
    training_config["experiment_id"] = 3
    
    pipeline_latent = Flux1FlowPipeline(
                train_dataset_npe,
                val_dataset_npe,
                dim_obs,
                dim_cond,
                vae_obs=None,
                vae_cond=pipeline.model,
                params=params_flux,
                training_config=training_config,
            )
    
    pipeline_latent.train(nnx.Rngs(0), save_model=True)
    
    # plot the results
    
    x_o = df_test["xs"][0][None,...]
    x_o = normalize(jnp.array(x_o,dtype=jnp.bfloat16), xs_mean, xs_std)
    
    theta_true = df_test["thetas"][0] # already unnormalized
    
    samples = pipeline_latent.sample(nnx.Rngs(0).sample(), x_o, 10_000)
    res = samples[:,0,:] # shape (num_samples, 1, ch_obs) -> (num_samples, ch_obs)
    
    # unnormalize the results for plotting
    res_unnorm = unnormalize(res, thetas_mean, thetas_std)
    
    plot_marginals(res_unnorm, true_param=theta_true)
    plt.savefig("gw_samples.png", dpi=100, bbox_inches="tight")
    plt.show()



if __name__ == "__main__":
    main()
