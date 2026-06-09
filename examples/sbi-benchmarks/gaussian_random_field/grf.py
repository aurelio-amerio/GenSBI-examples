#%%
import os

from gensbi_examples.tasks import GravitationalLensing 
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import jax 
from jax import numpy as jnp

from sbibm_jax.data import TaskDataset

import matplotlib.pyplot as plt
#%%
task = TaskDataset("gaussian_random_field_256", normalize=True, dtype=jnp.bfloat16, use_prefetching=False)
# %%
train_dataset = task.get_train_loader(16)
# %%
data = next(iter(train_dataset))
x = np.asarray(data[1],dtype=np.float32)
# %%
data[0].shape, data[1].shape
# %%
plt.clf()
plt.imshow(x[1,:,:,0], vmin=-1, vmax=1, cmap="coolwarm")
plt.show()
# %%
# %% using VAE encoder
# still need to train this
import os

# if __name__ != "__main__":
#     os.environ["JAX_PLATFORMS"] = "cpu"
# else:
#     os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"  # use 90% of GPU memory
#     os.environ["JAX_PLATFORMS"] = "cuda"  # change to 'cpu' if no GPU is available

import gensbi

# base libraries
import jax
from jax import Array
from jax import numpy as jnp
import numpy as np
from flax import nnx

from tqdm import tqdm
import gc

# data loading
import grain
from datasets import load_dataset
import yaml

# plotting
import matplotlib.pyplot as plt

# gensbi
from gensbi.recipes import ConditionalPipeline
from gensbi.core import FlowMatchingMethod
from gensbi.recipes.flux1 import parse_flux1_params, parse_training_config
from gensbi.recipes.utils import patchify_2d, depatchify_2d

# from gensbi.experimental.models.autoencoders import AutoEncoder2D, AutoEncoderParams
# from gensbi.experimental.recipes.vae_pipeline import parse_autoencoder_params
from gensbi.models import Flux1Params, Flux1

from gensbi.utils.plotting import plot_marginals

from gensbi.diagnostics import LC2ST, plot_lc2st
from gensbi.diagnostics import run_sbc, sbc_rank_plot
from gensbi.diagnostics import run_tarp, plot_tarp
from gensbi.diagnostics.marginal_coverage import (
    compute_marginal_coverage,
    plot_marginal_coverage,
)



### end of imports ###

config_path = "./config/config_1a.yaml"


def normalize(batch, mean, std):
    mean = jnp.asarray(mean, dtype=batch.dtype)
    std = jnp.asarray(std, dtype=batch.dtype)
    return (batch - mean) / std


def unnormalize(batch, mean, std):
    mean = jnp.asarray(mean, dtype=batch.dtype)
    std = jnp.asarray(std, dtype=batch.dtype)
    return batch * std + mean

#%%
# class LensingModel(nnx.Module):
#     def __init__(self, vae, sbi_model):
#         self.vae = vae
#         self.sbi_model = sbi_model

#     def __call__(
#         self,
#         t: Array,
#         obs: Array,
#         obs_ids: Array,
#         cond: Array,
#         cond_ids: Array,
#         conditioned: bool | Array = True,
#         guidance: Array | None = None,
#         encoder_key=None,
#     ):

#         # first we encode the conditioning data
#         cond_latent = self.vae.encode(cond, encoder_key)
#         # patchify the cond_latent for the transformer
#         cond_latent = patchify_2d(cond_latent)

#         # then we pass to the sbi model
#         return self.sbi_model(
#             t=t,
#             obs=obs,
#             obs_ids=obs_ids,
#             cond=cond_latent,
#             cond_ids=cond_ids,
#             conditioned=conditioned,
#             guidance=guidance,
#         )


params_flux = Flux1Params(
    in_channels=64,
    vec_in_dim=None,
    context_in_dim=1,
    mlp_ratio= 4,
    num_heads= 4,
    depth= 8,
    depth_single_blocks= 16,
    axes_dim= [4, 4, 2],
    qkv_bias=True,
    dim_obs=64,  # (64 // 8) * (64 // 8) = 64 patches with size=8
    dim_cond=2,
    id_embedding_strategy=("rope2d", "absolute"),
    theta=10_000,
    rngs=nnx.Rngs(default=42),
    param_dtype=jnp.bfloat16,
)



model = Flux1(params_flux)


#%%



# model = LensingModel(vae_model, model_sbi)

# training_config = parse_training_config(config_path)

# with open(config_path, "r") as f:
#     config = yaml.safe_load(f)
#     batch_size = config["training"]["batch_size"]
#     nsteps = config["training"]["nsteps"]
#     multistep = config["training"]["multistep"]
#     experiment = config["training"]["experiment_id"]

# def split_data(batch):
#     obs = jnp.array(batch["thetas"], dtype=jnp.bfloat16)
#     obs = normalize(obs, thetas_mean, thetas_std)
#     obs = obs.reshape(obs.shape[0], dim_obs, ch_obs)
#     cond = jnp.array(batch["xs"], dtype=jnp.bfloat16)
#     cond = normalize(cond, xs_mean, xs_std)
#     cond = cond[..., None]
#     return obs, cond

# train_dataset_npe = (
#     grain.MapDataset.source(df_train).shuffle(42).repeat().to_iter_dataset()
# )

# performance_config = grain.experimental.pick_performance_config(
#     ds=train_dataset_npe,
#     ram_budget_mb=1024 * 8,
#     max_workers=None,
#     max_buffer_size=None,
# )

# train_dataset_npe = (
#     train_dataset_npe.batch(batch_size)
#     .map(split_data)
#     .mp_prefetch(performance_config.multiprocessing_options)
# )

# val_dataset_npe = (
#     grain.MapDataset.source(df_val)
#     .shuffle(42)
#     .repeat()
#     .to_iter_dataset()
#     .batch(256)
#     .map(split_data)
# )

# current_dir = os.getcwd()
# training_config["checkpoint_dir"] = os.path.join(current_dir, "checkpoints")


train_dataset = task.get_train_loader(batch_size=16)
val_dataset = task.get_val_loader(batch_size=256)

pipeline = ConditionalPipeline(
    model,
    train_dataset,
    val_dataset,
    dim_obs=(
        64,
        64,
    ),
    dim_cond=2,  # we are workin in the latent space of the vae
    ch_obs=1,
    ch_cond=64,  # conditioning is now in the latent space
    method=FlowMatchingMethod(),
    id_embedding_strategy=("rope2d", "absolute"),
    size=8,  # patch size 8 -> obs_ids.shape = (1, 64, 3) i.e. (64//8)*(64//8)=64 tokens
)
#%%
pipeline.obs_ids.shape
#%%

# pipeline.obs_ids
# pipeline.cond_ids
obs = patchify_2d(data[1],size=8)
cond = data[0]
model(0.5, obs, pipeline.obs_ids, cond, pipeline.cond_ids)
#%%
obs.shape

#%%

if train_model:
    pipeline_latent.train(nnx.Rngs(0), save_model=True)

if restore_model:
    pipeline_latent.restore_model()

# plot the results

x_o = df_test["xs"][0][None, ...]
x_o = normalize(jnp.array(x_o, dtype=jnp.bfloat16), xs_mean, xs_std)
x_o = x_o[..., None]

theta_true = df_test["thetas"][0]  # already unnormalized

samples = pipeline_latent.sample_batched(
    nnx.Rngs(0).sample(),
    x_o,
    100_000,
    chunk_size=10_000,
    encoder_key=jax.random.PRNGKey(1234),
)
# print("Samples shape:", samples.shape)
res = samples[:, 0, :, 0]  # shape (num_samples, 1, 2, 1) -> (num_samples, 2)
# print("Res shape:", res.shape)
# unnormalize the results for plotting
res_unnorm = unnormalize(res, thetas_mean, thetas_std)

plot_marginals(res_unnorm, true_param=theta_true, gridsize=30)
plt.savefig(
    f"imgs/grf_samples_conf{experiment}.png", dpi=100, bbox_inches="tight"
)
plt.show()

# # split in thetas and xs
thetas_ = np.array(df_test["thetas"])[:200]
xs_ = np.array(df_test["xs"])[:200]

thetas_ = normalize(jnp.array(thetas_, dtype=jnp.bfloat16), thetas_mean, thetas_std)
xs_ = normalize(jnp.array(xs_, dtype=jnp.bfloat16), xs_mean, xs_std)
xs_ = xs_[..., None]

num_posterior_samples = 10000

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

tarp_result = run_tarp(
    thetas,
    posterior_samples,
    references=None,  # will be calculated automatically.
    bootstrap=False,
)

plot_tarp(tarp_result, mode="both")
plt.savefig(
    f"imgs/grf_tarp_conf{experiment}.png", dpi=100, bbox_inches="tight"
)  # uncomment to save the figure
plt.show()

# Marginal Coverage
print("Running Marginal Coverage diagnostic...")
alpha_marginal = compute_marginal_coverage(
    thetas, posterior_samples, method="histogram"
)
plot_marginal_coverage(alpha_marginal)
plt.savefig(
    f"imgs/grf_marginal_coverage_conf{experiment}.png",
    dpi=100,
    bbox_inches="tight",
)
plt.show()

ranks, dap_samples = run_sbc(thetas, xs, posterior_samples)

f, ax = sbc_rank_plot(ranks, num_posterior_samples, plot_type="hist", num_bins=20)
plt.savefig(
    f"imgs/grf_sbc_conf{experiment}.png", dpi=100, bbox_inches="tight"
)  # uncomment to save the figure
plt.show()

# LC2ST diagnostic
thetas_ = np.array(df_test["thetas"])[:10_000]
xs_ = np.array(df_test["xs"])[:10_000]

thetas_ = normalize(jnp.array(thetas_, dtype=jnp.bfloat16), thetas_mean, thetas_std)
xs_ = normalize(jnp.array(xs_, dtype=jnp.bfloat16), xs_mean, xs_std)
xs_ = xs_[..., None]

num_posterior_samples = 1

posterior_samples_ = pipeline_latent.sample(
    jax.random.PRNGKey(42),
    x_o=xs_,
    nsamples=xs_.shape[0],
    encoder_key=jax.random.PRNGKey(1234),
)

thetas = thetas_.reshape(thetas_.shape[0], -1)
xs = xs_.reshape(xs_.shape[0], -1)
posterior_samples = posterior_samples_.reshape(posterior_samples_.shape[0], -1)

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

x_o = xs_[-1:]  # Take the last observation as observed data.
theta_o = thetas_[-1:]  # True parameter for the observed data.

post_samples_star = pipeline_latent.sample(
    jax.random.PRNGKey(42),
    x_o,
    nsamples=10_000,
    encoder_key=jax.random.PRNGKey(1234),
)

x_o = x_o.reshape(1, -1)
post_samples_star = np.array(
    post_samples_star.reshape(post_samples_star.shape[0], -1)
)

fig, ax = plot_lc2st(
    lc2st,
    post_samples_star,
    x_o,
)
plt.savefig(
    f"imgs/grf_lc2st_conf{experiment}.png", dpi=100, bbox_inches="tight"
)  # uncomment to save the figure
plt.show()


# if __name__ == "__main__":
#     main()

# %%
