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
# # Gaussian Mixture Flux1 Flow Example
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-amerio/GenSBI-examples/blob/main/examples/sbi-benchmarks/gaussian_mixture/flow_flux/gaussian_mixture_flow_flux.ipynb)
# > Notice: This notebook has been automatically generated. If you find any errors, please [open an issue](https://github.com/aurelio-amerio/GenSBI-examples/issues) on the GenSBI-examples GitHub repository.

# %% [markdown]
# ---
# This notebook demonstrates conditional Flow Matching on the Gaussian Mixture task using JAX and Flax. 
#
# ## Table of Contents
# | Section | Description |
# |---|---|
# | 1. [Introduction & Setup](#introduction-setup) | Overview, environment, device, autoreload |
# | 2. [Task & Data Preparation](#task-data-preparation) | Define task, visualize data, create datasets |
# | 3. [Model Configuration & Definition](#model-configuration-definition) | Load config, set parameters, instantiate model |
# | 4. [Training](#training) | Train or restore model, manage checkpoints |
# | 5. [Evaluation & Visualization](#evaluation-visualization) | Sample the posterior and compare it with the reference samples |
# | 6. [Posterior Calibration Checks](#posterior-calibration-checks) | Marginal coverage, TARP, SBC, L-C2ST |
#
# ---

# %% [markdown]
# ## 1. Introduction & Setup
# ---
# In this section, we introduce the problem, set up the computational environment, import required libraries, configure JAX for CPU or GPU usage, and enable autoreload for iterative development. Compatibility with Google Colab is also ensured.

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
    # %cd GenSBI-examples/examples/sbi-benchmarks/gaussian_mixture/flow_flux

# %%
import os
# select device

os.environ["JAX_PLATFORMS"] = "cuda" 
# os.environ["JAX_PLATFORMS"] = "cpu" 

# %% [markdown]
# ## 2. Task & Data Preparation
# ---
# In this section, we define the Gaussian Mixture task, visualize reference samples, and create the training and validation datasets required for model learning. Batch size and sample count are set for reproducibility and performance.

# %%
restore_model=True
train_model=False

# %%
import orbax.checkpoint as ocp
# get the current notebook path
notebook_path = os.getcwd()
checkpoint_dir = os.path.join(notebook_path, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

# %%
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from flax import nnx

from numpyro import distributions as dist
import numpy as np

# %%
from gensbi.utils.plotting import plot_marginals

# %%
from sbibm_jax.data import TaskDataset
task = TaskDataset("gaussian_mixture", kind="conditional", use_prefetching=False)

# %%
# reference posterior for an observation
obs, reference_samples = task.get_reference(num_observation=8)

# %%
# plot the reference posterior
plot_marginals(np.asarray(reference_samples, dtype=np.float32), gridsize=50, plot_levels=False, backend="seaborn")
plt.show()

# %%
# make a dataset
nsamples = int(1e5)

# %%
# Set batch size for training. Larger batch sizes help prevent overfitting, but are limited by available GPU memory.
batch_size = 4096
# Create training and validation datasets using the Gaussian Mixture task object.
train_dataset = task.get_train_loader(batch_size, num_samples=nsamples)
val_dataset = task.get_val_loader(batch_size)

# Create iterators for the training and validation datasets.
dataset_iter = iter(train_dataset)
val_dataset_iter = iter(val_dataset)

# %% [markdown]
# ## 3. Model Configuration & Definition
# ---
# In this section, we load the model and optimizer configuration, set all relevant parameters, and instantiate the Flux1 model.

# %%
from gensbi.recipes import Flux1FlowPipeline

# %%
import yaml

# Path to the configuration file.
config_path = f"{notebook_path}/config/config_flow_flux.yaml"

# %%
# Extract dimensionality information from the task object.
dim_obs = task.dim_theta  # Number of parameters to infer
dim_cond = task.dim_x    # Number of observed data dimensions

dim_joint = task.dim_joint  # Joint dimension (for model input)

# %%
pipeline = Flux1FlowPipeline.init_pipeline_from_config(
        train_dataset,
        val_dataset,
        dim_obs,
        dim_cond,
        config_path,
        checkpoint_dir,
    )

# %% [markdown]
# ## 4. Training
# ---
# In this section, we train the model or restore a checkpoint.

# %%
# pipeline.train(nnx.Rngs(0), save_model=False)

# %%
pipeline.restore_model()


# %% [markdown]
# ## 5. Evaluation & Visualization
# ---
# In this section, we evaluate the trained model by sampling from the posterior, and comparing results to reference data.

# %% [markdown]
# ### Section 5.1: Posterior Sampling
# ---
# In this section, we sample from the posterior distribution using the trained model and visualize the results. Posterior samples are generated for a selected observation and compared to reference samples to assess model accuracy.

# %%
# we want to do conditional inference. We need an observation for which we want to compute the posterior
def get_samples(idx, nsamples=10_000, use_ema=False, key=None):
    observation, reference_samples = task.get_reference(idx)
    true_param = jnp.array(task.get_true_parameters(idx))

    if key is None:
        key = jax.random.PRNGKey(42)

    time_grid = jnp.linspace(0,1,100)

    samples = pipeline.sample(key, observation, nsamples, use_ema=use_ema, time_grid=time_grid)
    return samples, true_param, reference_samples


# %%
samples, true_param, reference_samples =  get_samples(8)

# %% [markdown]
# ### Section 5.2: Visualize Posterior Samples
# ---
# In this section, we plot the posterior samples as 1D and 2D marginal histograms to visualize the learned distribution and compare it to the ground truth.

# %%
from gensbi.utils.plotting import plot_marginals, plot_2d_dist_contour

# %%
plot_marginals(samples[-1,...,0], backend="seaborn", gridsize=50)
plt.show()

# alternatively use "corner" to plot containment levels too
# plot_marginals(samples[-1,...,0], backend="corner", gridsize=20)
# plt.show()

# %% [markdown]
# <img src="https://raw.githubusercontent.com/aurelio-amerio/GenSBI-examples/refs/heads/main/examples/sbi-benchmarks/gaussian_mixture/flow_flux/imgs/marginals_ema_1.png" width=800>

# %% [markdown]
# ## 6. Posterior Calibration Checks

# %% [markdown]
# We report here the results of the posterior calibration tests. As an exercise, you can implement the tests as in the Two Moons example and compare the results.

# %% [markdown]
# **Average C2ST**: 0.5087 ± 0.0072

# %% [markdown]
# **Marginal Coverage:** <br><br>
# <img src="https://raw.githubusercontent.com/aurelio-amerio/GenSBI-examples/refs/heads/main/examples/sbi-benchmarks/gaussian_mixture/flow_flux/imgs/marginal_coverage_1.png" width=800>

# %% [markdown]
# **TARP:** <br><br>
# <img src="https://raw.githubusercontent.com/aurelio-amerio/GenSBI-examples/refs/heads/main/examples/sbi-benchmarks/gaussian_mixture/flow_flux/imgs/tarp_1.png" width=800>

# %% [markdown]
# **SBC** <br><br>
#
# <img src="https://raw.githubusercontent.com/aurelio-amerio/GenSBI-examples/refs/heads/main/examples/sbi-benchmarks/gaussian_mixture/flow_flux/imgs/sbc_1.png" width=800>

# %% [markdown]
# **L-C2ST**<br><br>
# <img src="https://raw.githubusercontent.com/aurelio-amerio/GenSBI-examples/refs/heads/main/examples/sbi-benchmarks/gaussian_mixture/flow_flux/imgs/lc2st_1.png" width=500>
