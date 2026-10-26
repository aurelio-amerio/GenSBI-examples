# ---
# jupyter:
#   jupytext:
#     default_lexer: ipython3
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: gensbi
#     language: python
#     name: python3
# ---

# %% [markdown]
# # SLCP TarFlow NLE Example
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-amerio/GenSBI-examples/blob/main/examples/sbi-benchmarks/slcp/tarflow_NLE/slcp_tarflow_nle.ipynb)
# > Notice: If you find any errors, please [open an issue](https://github.com/aurelio-amerio/GenSBI-examples/issues) on the GenSBI-examples GitHub repository.

# %% [markdown]
# ---
# This notebook demonstrates **Neural Likelihood Estimation (NLE)** on the SLCP task using a **TarFlow** (transformer autoregressive normalizing flow) in JAX and Flax.
#
# **NLE in one paragraph.** Instead of learning the posterior $p(\theta \mid x)$ directly (as in NPE), NLE learns the *likelihood* $q(x \mid \theta)$ with a conditional density estimator. At inference time the posterior is recovered by combining the learned likelihood with the prior, $p(\theta \mid x_o) \propto q(x_o \mid \theta)\, p(\theta)$, and drawing samples from this (unnormalized) density with a sampler. This decouples *learning* the density from *sampling* the posterior, and lets a single trained likelihood serve any observation.
#
# **Why SLCP + TarFlow + tempered SMC.** SLCP ("Simple Likelihood, Complex Posterior") has a genuinely simple likelihood — the 8-D observation $x$ is four i.i.d. 2-D Gaussian draws — but a *complex, multimodal* posterior: the scale parameters enter the likelihood squared and the correlation through `tanh`, producing sign-symmetric modes over the 5-D parameter $\theta$. A modest TarFlow captures the likelihood, but a single MCMC chain would collapse onto one mode, so we sample the posterior with **Tempered SMC** — a particle cloud walked from the prior to the posterior that populates every mode.
#
# > **NLE convention (mirror of NPE):** `obs = x`, `cond = theta`, so the flow models $q(x \mid \theta)$. This is the *only* substantive difference from an NPE example; the flow and pipeline are otherwise identical.
#
# ## Table of Contents
# | Section | Description |
# |---|---|
# | 1. [Introduction & Setup](#introduction-setup) | Overview, environment, device |
# | 2. [Task & Data Preparation](#task-data-preparation) | Define task, swap obs/cond for NLE, build datasets |
# | 3. [Model Configuration & Definition](#model-configuration-definition) | Load config, build the TarFlow with the params class, standardize |
# | 4. [Training](#training) | Train (commented out) or restore the checkpoint |
# | 5. [NLE Posterior Sampling](#nle-posterior-sampling) | Prior + learned likelihood, tempered SMC, visualize posterior |
# | 6. [Performance (C2ST)](#performance-c2st) | Reported C2ST accuracy of the trained model |
#
# ---

# %% [markdown]
# ## 1. Introduction & Setup
# ---
# In this section, we set up the computational environment, import the required libraries, and configure JAX for CPU or GPU usage. Compatibility with Google Colab is also ensured.

# %%
# Check if running on Colab and install dependencies if needed
try:
    import google.colab
    colab = True
except ImportError:
    colab = False

if colab:
    # Install required packages and clone the repository
    # !uv pip install --quiet "gensbi[cuda13,examples]" "sbibm-jax[loader]"
    # !git clone --depth 1 https://github.com/aurelio-amerio/GenSBI-examples
    # %cd GenSBI-examples/examples/sbi-benchmarks/slcp/tarflow_NLE

# %%
import os
# select device

os.environ["JAX_PLATFORMS"] = "cuda" 
# os.environ["JAX_PLATFORMS"] = "cpu" 

# %%
import yaml
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from flax import nnx

from gensbi.models import TarFlow, TarFlowParams
from gensbi.recipes import ConditionalFlowPipeline
from gensbi.inference import NLEPosterior, TemperedSMC
from gensbi.utils.plotting import plot_marginals
from sbibm_jax.data import TaskDataset
from sbibm_jax.tasks import get_task

# %% [markdown]
# ## 2. Task & Data Preparation
# ---
# In this section, we define the SLCP task, visualize the reference posterior, and build the training/validation datasets. The key NLE-specific step is **swapping `obs` and `cond`** so the flow learns $q(x \mid \theta)$.

# %%
# By default we restore the released checkpoint instead of training from scratch.
restore_model = True
train_model = False

# %%
# Resolve paths relative to this notebook. The checkpoint and config ship in the repo.
notebook_path = os.getcwd()
checkpoint_dir = os.path.join(notebook_path, "checkpoints")
config_path = os.path.join(notebook_path, "config", "config_tarflow_nle.yaml")

with open(config_path) as f:
    config = yaml.safe_load(f)

task_name = config["task_name"]   # "slcp"

# %%
# Conditional task. We keep the raw loader (normalize=False): x is standardized
# *inside* the flow (see Section 3), not by the data loader.
task = TaskDataset(task_name, kind="conditional", normalize=False, use_prefetching=False)

# NLE: the autoregressive target is the observation x, conditioned on theta.
dim_obs, dim_cond = task.dim_x, task.dim_theta
print(f"dim_obs (x) = {dim_obs}, dim_cond (theta) = {dim_cond}")

# %%
# reference posterior samples (theta) for one observation
obs, reference_samples = task.get_reference(num_observation=1)

# %%
# plot the reference posterior over the 5-D theta
plot_marginals(np.asarray(reference_samples, dtype=np.float32), gridsize=50, plot_levels=False, backend="seaborn")
plt.show()


# %%
def swap_obs_cond(batch):
    """Map a conditional batch (theta, x) -> (x, theta) for NLE.

    TaskDataset's conditional collate yields (theta, x); the pipeline reads
    `obs, cond = batch`. NLE models q(x | theta), so obs must be x and cond theta.
    """
    theta, x = batch
    return x, theta


# %%
# Build the datasets and apply the NLE obs/cond swap. nsamples must be < task.max_samples.
train_cfg = config["training"]
batch_size = int(train_cfg["batch_size"])
nsamples = int(train_cfg["nsamples"])

train_ds = task.get_train_loader(batch_size, num_samples=nsamples).map(swap_obs_cond)
val_ds = task.get_val_loader(512).map(swap_obs_cond)

# %% [markdown]
# ## 3. Model Configuration & Definition
# ---
# In this section, we build the TarFlow from its **parameter class** `TarFlowParams`, wrap it in a `ConditionalFlowPipeline`, and set the in-flow standardization from the dataset statistics.
#
# `TarFlowParams` is the single source of truth for the architecture: the head sizing follows the Flux1 convention (specify `head_dim` and `num_heads`; the total width `channels = head_dim * num_heads` is derived), and `cond` selects how `theta` enters the flow. Here `cond="vector"` turns each of the 5 `theta` coordinates into its own prefix token (`VectorConditioner`) — richer conditioning than the default per-token additive bias (`cond="bias"`).

# %%
model_cfg = config["model"]

def build_flow(rngs, dim_obs, dim_cond, model_cfg):
    """Instantiate the TarFlow from the model config via the TarFlowParams class.

    For NLE, dim == dim_obs (the autoregressive target x) and cond_dim == dim_cond (theta).
    """
    return TarFlow(TarFlowParams(
        rngs=rngs,
        dim=dim_obs,
        cond_dim=dim_cond,
        cond=str(model_cfg.get("cond", "bias")),            # bias | vector | image
        cond_channels=int(model_cfg.get("cond_channels", 1)),
        num_blocks=int(model_cfg.get("num_blocks", 8)),
        layers_per_block=int(model_cfg.get("layers_per_block", 2)),
        head_dim=int(model_cfg.get("head_dim", 16)),
        num_heads=int(model_cfg.get("num_heads", 4)),        # width = head_dim * num_heads
        block_size=int(model_cfg.get("block_size", 1)),
        permutation=str(model_cfg.get("permutation", "flip")),
        standardize=bool(model_cfg.get("standardize", True)),
        zero_init=bool(model_cfg.get("zero_init", True)),
    ))

flow = build_flow(nnx.Rngs(0), dim_obs, dim_cond, model_cfg)

# %%
# Training config: pipeline defaults overlaid with the YAML optimizer + training sections.
# The pipeline reads these eagerly, and uses `experiment_id` + `checkpoint_dir` to locate
# the checkpoint when restoring.
training_config = ConditionalFlowPipeline.get_default_training_config()
training_config.update(config.get("optimizer", {}))
training_config.update(config.get("training", {}))
training_config["checkpoint_dir"] = checkpoint_dir

pipeline = ConditionalFlowPipeline(
    flow, train_ds, val_ds, dim_obs, dim_cond, training_config=training_config
)

# %%
# In-flow standardization: x (the obs/target) is standardized inside the flow using the
# precomputed dataset statistics, so we don't fit them on the fly. The EMA copy averages
# only Params, so its non-Param Standardize buffer must be set explicitly too.
mean = jnp.asarray(task.x_mean).reshape(dim_obs)
std = jnp.asarray(task.x_std).reshape(dim_obs)

pipeline.model.set_standardization(mean, std)
pipeline.ema_model.set_standardization(mean, std)
pipeline._standardized = True  # suppress the train-time "did you fit?" warning

# %% [markdown]
# ## 4. Training
# ---
# Training is shown but **commented out** by default; we restore the released checkpoint instead. Flip `restore_model` / `train_model` in Section 2 (or uncomment below) to train from scratch — on a GPU this takes a while for the configured `nsteps`.

# %%
# if train_model:
#     pipeline.train(nnx.Rngs(0))

# %%
if restore_model:
    pipeline.restore_model()


# %% [markdown]
# ## 5. NLE Posterior Sampling
# ---
# NLE recovers the posterior as `learned likelihood × prior`, then samples it. Two pieces matter:
#
# 1. **The prior must validate its support.** `get_prior_dist()` ships with `validate_args=False`, so its `log_prob` returns the in-support constant *everywhere* and would not bound the potential — the sampler would then wander outside the prior box and the posterior would be wrong. Re-enabling validation makes `log_prob = -inf` outside the support (here the uniform prior on $[-3, 3]^5$), confining the sampler to the box.
# 2. **The sampler must be multimodal.** We use `TemperedSMC`, which walks a particle cloud along $p(\theta)\, q(x_o \mid \theta)^\beta$ for $\beta: 0 \to 1$, populating every mode. `num_particles` is the number of posterior samples returned.

# %%
def build_prior(task_name, validate_args=True):
    """Task prior over theta with out-of-support log_prob = -inf (bounds the NLE potential)."""
    prior = get_task(task_name).get_prior_dist()
    prior._validate_args = validate_args
    if hasattr(prior, "base_dist"):          # Independent wraps a base distribution
        prior.base_dist._validate_args = validate_args
    return prior

prior = build_prior(task_name)

# %%
# NLEPosterior holds the (EMA) likelihood flow + prior; the sampler is supplied at sample time.
posterior = NLEPosterior(pipeline.ema_model, prior)

sampler_cfg = config["sampler"]
sampler = TemperedSMC(
    num_particles=int(sampler_cfg.get("num_particles", 20000)),
    target_ess=float(sampler_cfg.get("target_ess", 0.9)),
    num_mcmc_steps=int(sampler_cfg.get("num_mcmc_steps", 10)),
    inner_kernel=str(sampler_cfg.get("inner_kernel", "mclmc")),  # mclmc | nuts
    inner_step_size=float(sampler_cfg.get("inner_step_size", 0.1)),
    inner_num_integration_steps=int(sampler_cfg.get("inner_num_integration_steps", 5)),
)

# %%
# Pick an observation, sample its posterior, and keep the true parameter for reference.
idx = int(config["evaluation"]["observation_idx"])
obs, _ = task.get_reference(num_observation=idx)
true_param = np.asarray(task.get_true_parameters(idx)).reshape(-1)

samples = posterior.sample(jax.random.PRNGKey(0), obs, sampler=sampler)  # (num_particles, dim_theta, 1)

# %%
# Posterior marginals (5-D theta) with the true parameter marked.
plot_marginals(np.asarray(samples[..., 0]), plot_levels=False, backend="seaborn",
               gridsize=50, true_param=true_param)
plt.show()

# %% [markdown]
# Example of the sampled posterior (the multimodal, sign-symmetric SLCP structure is clearly recovered):
#
# <img src="https://raw.githubusercontent.com/aurelio-amerio/GenSBI-examples/refs/heads/main/examples/sbi-benchmarks/slcp/tarflow_NLE/imgs/posterior_marginals_obs1_MCMC.png" width=800>
#
# The exact configuration used to train this checkpoint is in [`config/config_tarflow_nle.yaml`](https://github.com/aurelio-amerio/GenSBI-examples/blob/main/examples/sbi-benchmarks/slcp/tarflow_NLE/config/config_tarflow_nle.yaml).

# %% [markdown]
# ## 6. Performance (C2ST)

# %% [markdown]
# We summarize performance with the **Classifier 2-Sample Test (C2ST)**: a classifier is trained to tell the sampled posterior from the reference posterior, and its accuracy is reported. **0.5 is ideal** (the two are indistinguishable); **1.0 means** they are trivially separable. The number below is the mean ± std over the task's 10 reference observations, using the EMA likelihood flow.
#
# This notebook intentionally **omits the full posterior calibration suite** (marginal coverage, TARP, SBC, L-C2ST). For a worked example of those checks, see the Two Moons notebooks.

# %% [markdown]
# Average C2ST accuracy (SMC): 0.5180 +- 0.0130

# %% [markdown]
# ### Exercise: Try a different sampler

# %% [markdown]
# As an exercise, you can also check how to sample from the posterior using a different sampler. 
# In this example, we used a tempered SMC sampler, since the posterior is expected to be multimodal. What changes if we use an HMC sampler instead? 
# Furthermore, you can try implementing a nested sampler, which is an entirely different technique particularly suited for multimodal distributions.
# Check [this script](https://github.com/aurelio-amerio/GenSBI-examples/blob/main/examples/sbi-benchmarks/slcp/tarflow_NLE/train_tarflow_nle.py) for an example on how to use nested sampling for this task. Sampling with NS is generally slower, but it can be more robust than HMC for multimodal distributions.
# In this case, we achieve a similar c2st:
#
# Average C2ST accuracy (NS) : 0.5192 +- 0.0140
