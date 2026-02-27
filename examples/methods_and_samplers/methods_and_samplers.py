# %% [markdown]
# # Methods and Samplers in GenSBI
#
# GenSBI supports three generative methods for simulation-based inference:
#
# 1. **Flow Matching** — learns a velocity field and integrates an ODE from noise to data.
# 2. **EDM Diffusion** — learns a denoiser in σ-space (Karras et al., 2022).
# 3. **Score Matching** — learns the score function ∇ log p_t(x) (Song et al., 2021).
#
# Each method has a default solver and one or more alternatives that can be swapped
# at **sample time** without retraining. This example trains one model per method and
# demonstrates all available sampling strategies.
#
# We use the unified `ConditionalPipeline` API throughout, which is model-agnostic
# and parameterized by a `GenerativeMethod` object.

# %% Imports
import os

# Set JAX backend (use 'cuda' for GPU, 'cpu' otherwise)
os.environ["JAX_PLATFORMS"] = "cuda"

import grain
import numpy as np
import jax
from jax import numpy as jnp
from numpyro import distributions as dist
from flax import nnx

# Unified pipeline and generative methods
from gensbi.recipes import ConditionalPipeline
from gensbi.core import FlowMatchingMethod, DiffusionEDMMethod, ScoreMatchingMethod

# Model
from gensbi.models import Flux1, Flux1Params

# Plotting
from gensbi.utils.plotting import plot_marginals
import matplotlib.pyplot as plt

# %% [markdown]
# ## Shared Setup
#
# We use a simple 3D toy problem throughout: the simulator draws parameters θ from a
# uniform prior and produces observations x = θ + 1 + noise. This is identical to the
# `conditional_pipeline.py` example, so we can focus on the methods and samplers.

# %% Define the simulator and prior
theta_prior = dist.Uniform(
    low=jnp.array([-2.0, -2.0, -2.0]), high=jnp.array([2.0, 2.0, 2.0])
)

dim_obs = 3
dim_cond = 3
dim_joint = dim_obs + dim_cond


def simulator(key, nsamples):
    theta_key, sample_key = jax.random.split(key, 2)
    thetas = theta_prior.sample(theta_key, (nsamples,))
    xs = thetas + 1 + jax.random.normal(sample_key, thetas.shape) * 0.1

    thetas = thetas[..., None]
    xs = xs[..., None]

    # For the conditional pipeline, thetas (observations) come first
    data = jnp.concatenate([thetas, xs], axis=1)
    return data


# %% Generate training and validation data
train_data = simulator(jax.random.PRNGKey(0), 100_000)
val_data = simulator(jax.random.PRNGKey(1), 2000)

# %% Normalize the dataset
# Normalizing the data to zero mean and unit variance is important for stable training.
means = jnp.mean(train_data, axis=0)
stds = jnp.std(train_data, axis=0)


def normalize(data, means, stds):
    return (data - means) / stds


def unnormalize(data, means, stds):
    return data * stds + means


# %% Prepare the data for the conditional pipeline
# The conditional pipeline expects each batch to be a tuple of (observations, conditions).
def split_obs_cond(data):
    data = normalize(data, means, stds)
    return (
        data[:, :dim_obs],
        data[:, dim_obs:],
    )


# %% Create the input pipeline using Grain
batch_size = 256

train_dataset_grain = (
    grain.MapDataset.source(np.array(train_data))
    .shuffle(42)
    .repeat()
    .to_iter_dataset()
    .batch(batch_size)
    .map(split_obs_cond)
)

val_dataset_grain = (
    grain.MapDataset.source(np.array(val_data))
    .shuffle(42)
    .repeat()
    .to_iter_dataset()
    .batch(batch_size)
    .map(split_obs_cond)
)

# %% Prepare a test observation for sampling
# We generate one observation from the simulator and extract the true parameters
# and the conditioning data x_o. This will be reused across all methods.
new_sample = simulator(jax.random.PRNGKey(20), 1)
true_theta = new_sample[:, :dim_obs, :]

new_sample_norm = normalize(new_sample, means, stds)
x_o = new_sample_norm[:, dim_obs:, :]

# Plotting range (same for all methods)
plot_range = [(1, 3), (1, 3), (-0.6, 0.5)]

# %% [markdown]
# ---
# ## Section 1: Flow Matching
#
# **Flow Matching** learns a velocity field $v_\theta(t, x)$ that transports samples
# from a simple prior (Gaussian noise at $t=0$) to the data distribution (at $t=1$)
# via an ordinary differential equation (ODE).
#
# - **Default solver**: `ODESolver` — deterministic ODE integration (Euler or Dopri5).
# - **Alternative solvers** (SDE-based, stochastic):
#   - `ZeroEndsSolver` — diffusion vanishes at both time endpoints (arXiv:2410.02217).
#   - `NonSingularSolver` — non-singular diffusion coefficient (arXiv:2410.02217).
#
# The SDE solvers can sometimes improve sample diversity at the cost of
# additional stochasticity. They require prior statistics (`mu0`, `sigma0`) and a
# diffusion strength parameter `alpha`.

# %% Define the Flow Matching model and pipeline
params_fm = Flux1Params(
    in_channels=1,
    vec_in_dim=None,
    context_in_dim=1,
    mlp_ratio=3,
    num_heads=2,
    depth=4,
    depth_single_blocks=8,
    axes_dim=[10],
    qkv_bias=True,
    dim_obs=dim_obs,
    dim_cond=dim_cond,
    id_embedding_strategy=("absolute", "absolute"),
    theta=10 * dim_joint,
    rngs=nnx.Rngs(default=42),
    param_dtype=jnp.float32,
)

model_fm = Flux1(params_fm)

method_fm = FlowMatchingMethod()

training_config_fm = ConditionalPipeline.get_default_training_config()
training_config_fm["nsteps"] = 10000
training_config_fm["checkpoint_dir"] = os.path.join(os.getcwd(), "checkpoints", "flow")

pipeline_fm = ConditionalPipeline(
    model_fm,
    train_dataset_grain,
    val_dataset_grain,
    dim_obs=dim_obs,
    dim_cond=dim_cond,
    method=method_fm,
    training_config=training_config_fm,
)

# %% Train the Flow Matching model
# Uncomment the following lines to train the model.
# Once trained, the model is saved to checkpoints/flow and can be restored below.
rngs = nnx.Rngs(42)
pipeline_fm.train(rngs, save_model=True)

# %% Restore the trained Flow Matching model
# pipeline_fm.restore_model()

# %% Sample with the default ODE solver
# The default solver for flow matching is the ODESolver, which performs deterministic
# ODE integration from noise (t=0) to data (t=1).
samples_fm = pipeline_fm.sample(rngs.sample(), x_o, nsamples=100_000)
samples_fm = unnormalize(samples_fm, means[:dim_obs], stds[:dim_obs])

# %% Plot Flow Matching samples (ODE solver)
plot_marginals(
    np.array(samples_fm[..., 0]),
    gridsize=30,
    true_param=np.array(true_theta[0, :, 0]),
    range=plot_range,
)
plt.suptitle("Flow Matching — ODE Solver (default)", y=1.02)
plt.savefig("fm_ode_marginals.png", dpi=100, bbox_inches="tight")
plt.show()

# %% Alternative: ZeroEndsSolver (SDE-based)
# The ZeroEndsSolver adds stochastic noise during sampling. The diffusion coefficient
# vanishes at both t=0 and t=1, ensuring clean endpoints.
# Required kwargs: mu0 (prior mean), sigma0 (prior std), alpha (diffusion strength).
from gensbi.flow_matching.solver import ZeroEndsSolver

solver_kwargs_ze = {
    "mu0": jnp.zeros((dim_obs, 1)),  # prior mean (data is normalized)
    "sigma0": jnp.ones((dim_obs, 1)),  # prior std
    "alpha": 0.2,  # diffusion strength
}

samples_fm_ze = pipeline_fm.sample(
    rngs.sample(),
    x_o,
    nsamples=100_000,
    solver=(ZeroEndsSolver, solver_kwargs_ze),
)
samples_fm_ze = unnormalize(samples_fm_ze, means[:dim_obs], stds[:dim_obs])

# %% Plot Flow Matching samples (ZeroEndsSolver)
plot_marginals(
    np.array(samples_fm_ze[..., 0]),
    gridsize=30,
    true_param=np.array(true_theta[0, :, 0]),
    range=plot_range,
)
plt.suptitle("Flow Matching — ZeroEndsSolver (SDE)", y=1.02)
plt.savefig("fm_zeroends_marginals.png", dpi=100, bbox_inches="tight")
plt.show()

# %% Alternative: NonSingularSolver (SDE-based)
# The NonSingularSolver uses a non-singular diffusion coefficient, which can provide
# different sample quality characteristics compared to ZeroEndsSolver.
# It takes the same kwargs as ZeroEndsSolver.
from gensbi.flow_matching.solver import NonSingularSolver

solver_kwargs_ns = {
    "mu0": jnp.zeros((dim_obs, 1)),
    "sigma0": jnp.ones((dim_obs, 1)),
    "alpha": 0.2,
}

samples_fm_ns = pipeline_fm.sample(
    rngs.sample(),
    x_o,
    nsamples=100_000,
    solver=(NonSingularSolver, solver_kwargs_ns),
)
samples_fm_ns = unnormalize(samples_fm_ns, means[:dim_obs], stds[:dim_obs])

# %% Plot Flow Matching samples (NonSingularSolver)
plot_marginals(
    np.array(samples_fm_ns[..., 0]),
    gridsize=30,
    true_param=np.array(true_theta[0, :, 0]),
    range=plot_range,
)
plt.suptitle("Flow Matching — NonSingularSolver (SDE)", y=1.02)
plt.savefig("fm_nonsingular_marginals.png", dpi=100, bbox_inches="tight")
plt.show()

# %% Cleanup Flow Matching model to free memory
del model_fm, pipeline_fm

# %% [markdown]
# ---
# ## Section 2: EDM Diffusion
#
# **EDM Diffusion** (Karras et al., 2022) learns a denoiser $D_\theta(x; \sigma)$ in
# $\sigma$-space. The training noise schedule can use one of three prescriptions:
#
# - `DiffusionEDMMethod()` — default EDM scheduler (**recommended**)
# - `DiffusionEDMMethod(sde="VP")` — Variance Preserving scheduler
# - `DiffusionEDMMethod(sde="VE")` — Variance Exploding scheduler
#
# The model can be trained with any of these three prescriptions, and then sampled
# using any of the three as well. However, the **EDM scheduler is recommended** for
# both training and sampling. The scheduler variants are training-time choices that
# affect the noise schedule used during the diffusion process.
#
# **Solver**: `EDMSolver` is the only available solver for EDM. It implements the
# stochastic denoising sampler from Karras et al., 2022.

# %% Define the EDM Diffusion model and pipeline
params_edm = Flux1Params(
    in_channels=1,
    vec_in_dim=None,
    context_in_dim=1,
    mlp_ratio=3,
    num_heads=2,
    depth=4,
    depth_single_blocks=8,
    axes_dim=[10],
    qkv_bias=True,
    dim_obs=dim_obs,
    dim_cond=dim_cond,
    id_embedding_strategy=("absolute", "absolute"),
    theta=10 * dim_joint,
    rngs=nnx.Rngs(default=42),
    param_dtype=jnp.float32,
)

model_edm = Flux1(params_edm)

# Default EDM scheduler (recommended for both training and sampling)
method_edm = DiffusionEDMMethod()
# Alternative training schedulers (uncomment to use):
# method_edm = DiffusionEDMMethod(sde="VP")  # Variance Preserving
# method_edm = DiffusionEDMMethod(sde="VE")  # Variance Exploding

training_config_edm = ConditionalPipeline.get_default_training_config()
training_config_edm["nsteps"] = 10000
training_config_edm["checkpoint_dir"] = os.path.join(os.getcwd(), "checkpoints", "edm")

pipeline_edm = ConditionalPipeline(
    model_edm,
    train_dataset_grain,
    val_dataset_grain,
    dim_obs=dim_obs,
    dim_cond=dim_cond,
    method=method_edm,
    training_config=training_config_edm,
)

# %% Train the EDM Diffusion model
# Uncomment the following lines to train the model.
# Once trained, the model is saved to checkpoints/edm and can be restored below.
rngs = nnx.Rngs(42)
pipeline_edm.train(rngs, save_model=True)

# %% Restore the trained EDM Diffusion model
# pipeline_edm.restore_model()

# %% Sample with the default EDMSolver
# The EDMSolver implements the stochastic denoising sampler from Karras et al., 2022.
# It progressively denoises samples following a noise schedule from high to low sigma.
samples_edm = pipeline_edm.sample(rngs.sample(), x_o, nsamples=100_000)
samples_edm = unnormalize(samples_edm, means[:dim_obs], stds[:dim_obs])

# %% Plot EDM Diffusion samples
plot_marginals(
    np.array(samples_edm[..., 0]),
    gridsize=30,
    true_param=np.array(true_theta[0, :, 0]),
    range=plot_range,
)
plt.suptitle("EDM Diffusion — EDMSolver (default)", y=1.02)
plt.savefig("edm_marginals.png", dpi=100, bbox_inches="tight")
plt.show()

# %% Alternative: sample with VE scheduler
# You can override the noise schedule at sample time without retraining.
# Here we use the VE (Variance Exploding) scheduler instead of the default EDM one.
# This changes the sigma discretization used by the EDMSolver.
from gensbi.diffusion.path.scheduler import VEEdmScheduler

ve_scheduler = VEEdmScheduler()

samples_edm_ve = pipeline_edm.sample(
    rngs.sample(),
    x_o,
    nsamples=100_000,
    solver_scheduler=ve_scheduler,
)
samples_edm_ve = unnormalize(samples_edm_ve, means[:dim_obs], stds[:dim_obs])

# %% Plot EDM samples with VE scheduler
plot_marginals(
    np.array(samples_edm_ve[..., 0]),
    gridsize=30,
    true_param=np.array(true_theta[0, :, 0]),
    range=plot_range,
)
plt.suptitle("EDM Diffusion — VE scheduler (sample-time override)", y=1.02)
plt.savefig("edm_ve_marginals.png", dpi=100, bbox_inches="tight")
plt.show()

# %% Alternative: sample with VP scheduler
# Similarly, we can use the VP (Variance Preserving) scheduler at sample time.
from gensbi.diffusion.path.scheduler import VPEdmScheduler

vp_scheduler = VPEdmScheduler()

samples_edm_vp = pipeline_edm.sample(
    rngs.sample(),
    x_o,
    nsamples=100_000,
    solver_scheduler=vp_scheduler,
)
samples_edm_vp = unnormalize(samples_edm_vp, means[:dim_obs], stds[:dim_obs])

# %% Plot EDM samples with VP scheduler
plot_marginals(
    np.array(samples_edm_vp[..., 0]),
    gridsize=30,
    true_param=np.array(true_theta[0, :, 0]),
    range=plot_range,
)
plt.suptitle("EDM Diffusion — VP scheduler (sample-time override)", y=1.02)
plt.savefig("edm_vp_marginals.png", dpi=100, bbox_inches="tight")
plt.show()

# %% Cleanup EDM Diffusion model to free memory
del model_edm, pipeline_edm

# %% [markdown]
# ---
# ## Section 3: Score Matching
#
# **Score Matching** (Song et al., 2021) learns the score function
# $\nabla \log p_t(x)$, which points toward regions of higher data density.
# Samples are generated by running a **reverse-time SDE** from noise back to data.
#
# The SDE formulation can be either:
# - `ScoreMatchingMethod()` — Variance Preserving (VP) SDE (**default**)
# - `ScoreMatchingMethod(sde_type="VE")` — Variance Exploding (VE) SDE
#
# **Solvers**:
# - `SMSolver` (**default**) — reverse-time SDE, generates stochastic samples.
# - `SMPFSolver` — probability flow ODE, generates deterministic samples from the
#   same learned score function. Useful when reproducibility or lower variance is
#   desired.

# %% Define the Score Matching model and pipeline
params_sm = Flux1Params(
    in_channels=1,
    vec_in_dim=None,
    context_in_dim=1,
    mlp_ratio=3,
    num_heads=2,
    depth=4,
    depth_single_blocks=8,
    axes_dim=[10],
    qkv_bias=True,
    dim_obs=dim_obs,
    dim_cond=dim_cond,
    id_embedding_strategy=("absolute", "absolute"),
    theta=10 * dim_joint,
    rngs=nnx.Rngs(default=42),
    param_dtype=jnp.float32,
)

model_sm = Flux1(params_sm)

# Default: Variance Preserving (VP) SDE
method_sm = ScoreMatchingMethod()
# Alternative: Variance Exploding (VE) SDE (uncomment to use)
# method_sm = ScoreMatchingMethod(sde_type="VE")

training_config_sm = ConditionalPipeline.get_default_training_config()
training_config_sm["nsteps"] = 50000
training_config_sm["checkpoint_dir"] = os.path.join(os.getcwd(), "checkpoints", "sm")

pipeline_sm = ConditionalPipeline(
    model_sm,
    train_dataset_grain,
    val_dataset_grain,
    dim_obs=dim_obs,
    dim_cond=dim_cond,
    method=method_sm,
    training_config=training_config_sm,
)

# %% Train the Score Matching model
# Uncomment the following lines to train the model.
# Once trained, the model is saved to checkpoints/sm and can be restored below.
rngs = nnx.Rngs(42)
pipeline_sm.train(rngs, save_model=True)

# %% Restore the trained Score Matching model
# pipeline_sm.restore_model()

# %% Sample with the default SMSolver (reverse SDE)
# The default solver for score matching generates stochastic samples by running the
# reverse-time SDE. Each call with a different key produces a different set of samples.
samples_sm = pipeline_sm.sample(rngs.sample(), x_o, nsamples=100_000)
samples_sm = unnormalize(samples_sm, means[:dim_obs], stds[:dim_obs])

# %% Plot Score Matching samples (SMSolver)
plot_marginals(
    np.array(samples_sm[..., 0]),
    gridsize=30,
    true_param=np.array(true_theta[0, :, 0]),
    range=plot_range,
)
plt.suptitle("Score Matching — SMSolver (default, reverse SDE)", y=1.02)
plt.savefig("sm_sde_marginals.png", dpi=100, bbox_inches="tight")
plt.show()

# %% Alternative: SMPFSolver (probability flow ODE)
# The probability flow ODE produces deterministic samples from the same learned score
# function. This can be useful when you want reproducible results or lower variance.
from gensbi.diffusion.solver import SMPFSolver

samples_sm_pf = pipeline_sm.sample(
    rngs.sample(),
    x_o,
    nsamples=100_000,
    solver=(SMPFSolver, {}),
)
samples_sm_pf = unnormalize(samples_sm_pf, means[:dim_obs], stds[:dim_obs])

# %% Plot Score Matching samples (SMPFSolver)
plot_marginals(
    np.array(samples_sm_pf[..., 0]),
    gridsize=30,
    true_param=np.array(true_theta[0, :, 0]),
    range=plot_range,
)
plt.suptitle("Score Matching — SMPFSolver (probability flow ODE)", y=1.02)
plt.savefig("sm_pf_marginals.png", dpi=100, bbox_inches="tight")
plt.show()

# %% Cleanup Score Matching model to free memory
del model_sm, pipeline_sm

# %%
