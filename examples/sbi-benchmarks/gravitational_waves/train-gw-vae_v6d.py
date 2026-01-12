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
from gensbi.diagnostics import run_tarp, plot_tarp
from gensbi.diagnostics import run_sbc, sbc_rank_plot
from gensbi.diagnostics import LC2ST, plot_lc2st


config_path = "./config/gw_config_6d.yaml"


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
    def __init__(self, dim_cond, ch_cond, z_ch, *, rngs):
        features = 16
        padding = "SAME"
        self.activation = jax.nn.gelu

        dlin = dim_cond
        conv1 = nnx.Conv(
            ch_cond,
            features,
            kernel_size=(9,),
            strides=2,
            padding=padding,
            rngs=rngs,
            param_dtype=jnp.bfloat16,
        )  # 4096
        dlin = dlin // 2
        bn1 = nnx.BatchNorm(features, rngs=rngs, param_dtype=jnp.bfloat16)

        conv2 = nnx.Conv(
            features,
            features * 2,
            kernel_size=(6,),
            strides=2,
            padding=padding,
            rngs=rngs,
            param_dtype=jnp.bfloat16,
        )  # 2048
        dlin = dlin // 2
        features *= 2  # 32
        bn2 = nnx.BatchNorm(features, rngs=rngs, param_dtype=jnp.bfloat16)

        conv3 = nnx.Conv(
            features,
            features * 2,
            kernel_size=(3,),
            strides=2,
            padding=padding,
            rngs=rngs,
            param_dtype=jnp.bfloat16,
        )  # 1024
        dlin = dlin // 2
        features *= 2  # 64
        bn3 = nnx.BatchNorm(features, rngs=rngs, param_dtype=jnp.bfloat16)

        conv4 = nnx.Conv(
            features,
            features * 2,
            kernel_size=(3,),
            strides=2,
            padding=padding,
            rngs=rngs,
            param_dtype=jnp.bfloat16,
        )  # 512
        dlin = dlin // 2
        features *= 2  # 128
        bn4 = nnx.BatchNorm(features, rngs=rngs, param_dtype=jnp.bfloat16)

        conv5 = nnx.Conv(
            features,
            features * 2,
            kernel_size=(3,),
            strides=2,
            padding=padding,
            rngs=rngs,
            param_dtype=jnp.bfloat16,
        )  # 256
        dlin = dlin // 2
        features *= 2  # 256
        bn5 = nnx.BatchNorm(features, rngs=rngs, param_dtype=jnp.bfloat16)

        conv6 = nnx.Conv(
            features,
            features * 2,
            kernel_size=(3,),
            strides=2,
            padding=padding,
            rngs=rngs,
            param_dtype=jnp.bfloat16,
        )  # 128
        dlin = dlin // 2
        features *= 2  # 512
        bn6 = nnx.BatchNorm(features, rngs=rngs, param_dtype=jnp.bfloat16)

        conv7 = nnx.Conv(
            features,
            features,
            kernel_size=(3,),
            strides=2,
            padding=padding,
            rngs=rngs,
            param_dtype=jnp.bfloat16,
        )  # 64
        dlin = dlin // 2
        bn7 = nnx.BatchNorm(features, rngs=rngs, param_dtype=jnp.bfloat16)

        conv8 = nnx.Conv(
            features,
            features,
            kernel_size=(3,),
            strides=2,
            padding=padding,
            rngs=rngs,
            param_dtype=jnp.bfloat16,
        )  # 32
        dlin = dlin // 2
        bn8 = nnx.BatchNorm(features, rngs=rngs, param_dtype=jnp.bfloat16)

        conv9 = nnx.Conv(
            features,
            features,
            kernel_size=(3,),
            strides=2,
            padding=padding,
            rngs=rngs,
            param_dtype=jnp.bfloat16,
        )  # 16
        dlin = dlin // 2
        bn9 = nnx.BatchNorm(features, rngs=rngs, param_dtype=jnp.bfloat16)

        conv10 = nnx.Conv(
            features,
            features,
            kernel_size=(3,),
            strides=2,
            padding=padding,
            rngs=rngs,
            param_dtype=jnp.bfloat16,
        )  # 8
        dlin = dlin // 2
        bn10 = nnx.BatchNorm(features, rngs=rngs, param_dtype=jnp.bfloat16)

        conv11 = nnx.Conv(
            features,
            z_ch,
            kernel_size=(3,),
            strides=2,
            padding=padding,
            rngs=rngs,
            param_dtype=jnp.bfloat16,
        )  # 4
        dlin = dlin // 2
        bn11 = nnx.BatchNorm(z_ch, rngs=rngs, param_dtype=jnp.bfloat16)

        self.latent_shape = (1, dlin, z_ch)

        self.conv_layers = nnx.List(
            [
                conv1,
                conv2,
                conv3,
                conv4,
                conv5,
                conv6,
                conv7,
                conv8,
                conv9,
                conv10,
                conv11,
            ]
        )
        self.bn_layers = nnx.List(
            [bn1, bn2, bn3, bn4, bn5, bn6, bn7, bn8, bn9, bn10, bn11]
        )

        # self.linear = nnx.Linear(int(dlin), dout, rngs=rngs)

    def __call__(self, x):
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x)
            x = self.activation(x)
            x = self.bn_layers[i](x)

        # #flatten x
        # x = x.reshape(x.shape[0], -1)
        # x = self.linear(x)

        # return x[..., None,:]
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
        encoder_key=None,
    ):

        # first we encode the conditioning data
        cond_latent = self.encoder(cond)

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
z_ch = 128  # dimension of the latent space

def main(): 
    repo_name = "aurelio-amerio/SBI-benchmarks"

    task_name = "gravitational_waves"

    # dataset = load_dataset(repo_name, task_name).with_format("numpy")
    # dataset = load_dataset(
    #     repo_name, task_name, cache_dir="/data/users/.cache"
    # ).with_format("numpy")
    dataset = load_dataset(
        repo_name, task_name
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

    encoder = ConvEmbed(dim_cond, ch_cond, z_ch, rngs=nnx.Rngs(0))


    # now we define the NPE pipeline
    # get the latent dimensions from the autoencoder
    dim_cond_latent = encoder.latent_shape[1]
    assert z_ch == encoder.latent_shape[2], "Latent dimensions do not match"

    params_dict_flux = parse_flux1_params(config_path)

    params_flux = Flux1Params(
        rngs=nnx.Rngs(0),
        dim_obs=dim_obs,
        dim_cond=dim_cond_latent,
        **params_dict_flux,
    )

    model_sbi = Flux1(params_flux)

    # full model with CNN encoding the conditionals
    model = GWModel(encoder, model_sbi)

    training_config = parse_training_config(config_path)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        batch_size = config["training"]["batch_size"]
        nsteps = config["training"]["nsteps"]
        multistep = config["training"]["multistep"]
        experiment = config["training"]["experiment_id"]

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
        "/lhome/ific/a/aamerio/data/github/GenSBI-examples/examples/sbi-benchmarks/gravitational_waves/gw_npe_v6d/checkpoints"
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

    # pipeline_latent.train(nnx.Rngs(0), nsteps * multistep, save_model=True)
    pipeline_latent.restore_model()

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
    # plot_marginals(
    #     res_unnorm, true_param=theta_true, range=[(25, 75), (25, 75)], gridsize=30
    # )
    plot_marginals(
        res_unnorm, true_param=theta_true, gridsize=30
    )
    plt.savefig(f"gw_samples_v6d_conf{experiment}.png", dpi=100, bbox_inches="tight")
    plt.show()

    # split in thetas and xs
    thetas_ = np.array(df_test["thetas"])[:200]
    xs_ = np.array(df_test["xs"])[:200]

    thetas_ = normalize(jnp.array(thetas_, dtype=jnp.bfloat16), thetas_mean, thetas_std)
    xs_ = normalize(jnp.array(xs_, dtype=jnp.bfloat16), xs_mean, xs_std)

    num_posterior_samples = 1000

    posterior_samples_ = pipeline_latent.sample_batched(
        jax.random.PRNGKey(42),
        xs_,
        num_posterior_samples,
        chunk_size=20,
        encoder_key=jax.random.PRNGKey(1234),
    )

    thetas = thetas_.reshape(thetas_.shape[0], -1)
    xs = xs_.reshape(xs_.shape[0], -1)

    posterior_samples = posterior_samples_.reshape(
        posterior_samples_.shape[0], posterior_samples_.shape[1], -1
    )

    ecp, alpha = run_tarp(
        thetas,
        posterior_samples,
        references=None,  # will be calculated automatically.
    )

    plot_tarp(ecp, alpha)
    plt.savefig(
        f"gw_tarp_v6d_conf{experiment}.png", dpi=100, bbox_inches="tight"
    )  # uncomment to save the figure
    plt.show()
    
    ranks, dap_samples = run_sbc(thetas, xs, posterior_samples)

    f, ax = sbc_rank_plot(ranks, num_posterior_samples, plot_type="hist", num_bins=20)
    plt.savefig(f"gw_sbc_v6d_conf{experiment}.png", dpi=100, bbox_inches="tight") # uncomment to save the figure
    plt.show()

    # LC2ST diagnostic
    thetas_ = np.array(df_test["thetas"])[:10_000]
    xs_ = np.array(df_test["xs"])[:10_000]

    thetas_ = normalize(jnp.array(thetas_, dtype=jnp.bfloat16), thetas_mean, thetas_std)
    xs_ = normalize(jnp.array(xs_, dtype=jnp.bfloat16), xs_mean, xs_std)

    num_posterior_samples = 1

    posterior_samples_ = pipeline_latent.sample(jax.random.PRNGKey(42), x_o=xs_, nsamples=xs_.shape[0])

    thetas = thetas_.reshape(thetas_.shape[0], -1)  # (10_000, 3)
    xs = xs_.reshape(xs_.shape[0], -1)  # (10_000, 3)
    posterior_samples = posterior_samples_.reshape(posterior_samples_.shape[0], -1)  # (10_000, 3)

    # Train the L-C2ST classifier.
    lc2st = LC2ST(
        thetas=thetas[:-1],
        xs=xs[:-1],
        posterior_samples=posterior_samples[:-1],
        classifier="mlp",
        num_ensemble=1,
    )

    _ = lc2st.train_under_null_hypothesis()
    _ = lc2st.train_on_observed_data()

    x_o = xs_[-1 : ]  # Take the last observation as observed data.
    theta_o = thetas_[-1 : ]  # True parameter for the observed data.

    post_samples_star = pipeline_latent.sample(jax.random.PRNGKey(42), x_o, nsamples=10_000) 

    x_o = x_o.reshape(1,-1)  
    post_samples_star = np.array(post_samples_star.reshape(post_samples_star.shape[0], -1))  

    fig,ax = plot_lc2st(
        lc2st,
        post_samples_star,
        x_o,
    )
    plt.savefig(f"gw_lc2st_v6d_conf{experiment}.png", dpi=100, bbox_inches="tight") # uncomment to save the figure
    plt.show()


if __name__ == "__main__":
    main()

# %%
