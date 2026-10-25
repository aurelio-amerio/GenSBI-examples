# ---
# jupyter:
#   jupytext:
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

# %% [markdown] id="8D7OxAlV-wLC"
# # Flow Matching 2D Unconditional Example
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-amerio/GenSBI-examples/blob/main/examples/NDE/gensbi_text.ipynb)
#
# This notebook demonstrates how to train and sample from a flow-matching model on a 2D toy dataset using JAX and Flax. We will cover data generation, model definition, training, sampling, and density estimation using the pipeline utility.

# %% [markdown] id="mMB4DXA6-wLD"
# ## 1. Environment Setup
#
# In this section, we set up the notebook environment, import required libraries, and configure JAX for CPU or GPU usage.

# %% id="zZffZjCK-wLE"
try: #check if we are using colab, if so install all the required software
    import google.colab
    colab=True
except:
    colab=False

# %% id="bcNfj6Xr-wLE"
if colab: # you may have to restart the runtime after installing the packages
    # !uv pip install --quiet "gensbi[cuda13,examples]"
    # !git clone --depth 1 https://github.com/aurelio-amerio/GenSBI-examples
    # %cd GenSBI-examples/examples/NDE

# %% id="iQKjU1CQ-wLF"
# Set training and model restoration flags
overwrite_model = False
restore_model = True  # Use pretrained model if available
train_model = False  # Set to True to train from scratch

normalize_data = False

# %% [markdown] id="3bCU0VbE-wLF"
# ### Library Imports and JAX Backend Selection

# %% id="bUX0jfol-wLF"
# Import libraries and set JAX backend
import os
os.environ['JAX_PLATFORMS']="cuda" # select cpu instead if no gpu is available
# os.environ['JAX_PLATFORMS']="cpu"

from flax import nnx
import jax
import jax.numpy as jnp
import optax
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt
from matplotlib import cm

from gensbi_dist import get_gensbi_samples

# %% id="hxY0Uhc2-wLF"
# Specify the checkpoint directory for saving/restoring models
import orbax.checkpoint as ocp
checkpoint_dir = f"{os.getcwd()}/checkpoints/gensbi_text"

import os
os.makedirs(checkpoint_dir, exist_ok=True)

if overwrite_model:
    checkpoint_dir = ocp.test_utils.erase_and_create_empty(checkpoint_dir)

# %% [markdown] id="usuQfer--wLG"
# ## 2. Data Generation
#
# We generate a synthetic 2D dataset using JAX. This section defines the data generation functions and visualizes the data distribution.

# %% id="AxbGRGLw-wLG"
# Define a function to generate 2D box data using JAX
import jax
import jax.numpy as jnp
from jax import random
from functools import partial
import grain

from gensbi.utils.plotting import transparent_cmap, cmap_lcontour

# %%
batch_size = 1024
factor = np.sqrt(batch_size/256)

data = get_gensbi_samples(n_samples=500_000)

mean = data.mean(axis=0)
std = data.std(axis=0)

def normalize(x):
    return (x - mean) / std

def unnormalize(x):
    return x * std + mean


# %% id="BtRmaPV0-wLG"
# # Infinite data generator for training batches
# @partial(jax.jit, static_argnums=[1])  # type: ignore
# def inf_train_gen(key, batch_size: int = 200):
#     x = make_boxes_jax(key, batch_size)

#     return x

if normalize_data:
    data = normalize(data)

train_dataset_grain = (
    grain.MapDataset.source(np.array(data)[...,None])
    .shuffle(42)
    .repeat()
    .to_iter_dataset()
)

performance_config = grain.experimental.pick_performance_config(
            ds=train_dataset_grain,
            ram_budget_mb=1024 * 4,
            max_workers=None,
            max_buffer_size=None,
        )

train_dataset_batched = train_dataset_grain.batch(batch_size).mp_prefetch(
            performance_config.multiprocessing_options
        )

train_iter = iter(train_dataset_batched)

data_val = get_gensbi_samples(n_samples=1_000)
if normalize_data:
    data_val = normalize(data_val)


val_dataset_batched = (
    grain.MapDataset.source(np.array(data_val)[...,None])
    .shuffle(42)
    .repeat()
    .to_iter_dataset()
    .batch(512)
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 430} id="H9eFA3Ok-wLG" outputId="4ddd9a77-1680-4266-e66a-d616b039cdd8"
# # Visualize the generated data distribution
if normalize_data:
    samples = np.array(unnormalize(data))
else:
    samples = np.array(data)

def plot_samples(ax, samples):

    ax.hexbin( 
        samples[:, 0],
        samples[:, 1],
        gridsize=(200, 100),
        cmap=transparent_cmap,
        extent=[-2, 2, -1, 1],
    )

    ax.set_xlim(-2, 2)
    ax.set_ylim(-1, 1)
    #hide axes
    ax.grid(None)
    ax.set_xticks([])
    ax.set_yticks([])

    return ax

fig, ax = plt.subplots(figsize=(10, 5))
plot_samples(ax, samples)
plt.show()

# %% [markdown] id="Wzw6YPzT-wLH"
# ## 3. Model and Loss Definition
#
# We define the velocity field model (an MLP), the loss function, and the optimizer for training the flow-matching model.

# %% id="opragETv-wLH"
# Import flow matching components and utilities
from gensbi.recipes import UnconditionalFlowPipeline
from gensbi.models import Flux1Joint, Flux1JointParams


# %% id="j7Af9XBo-wLH"
# Define the MLP velocity field model
class MLP(nnx.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 128, *, rngs: nnx.Rngs, n_layers=8):

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        din = input_dim + 1

        self.obs_in= nnx.Linear(din, self.hidden_dim, rngs=rngs)

        self.middle_layers = nnx.List([nnx.Linear(self.hidden_dim, self.hidden_dim, rngs=rngs) for i in range(n_layers-2)])

        self.out = nnx.Linear(self.hidden_dim, self.input_dim, rngs=rngs)
     

    def __call__(self, t: jax.Array, obs: jax.Array,  **kwargs):
        assert obs.ndim == 3, f"Input obs must have shape (batch_size, input_dim, 1), got {obs.shape}"
        t = jnp.atleast_1d(t)
        x = jnp.squeeze(obs, axis=-1)
        if t.ndim<2:
            t = t[..., None]

        t = jnp.broadcast_to(t, (x.shape[0], t.shape[-1]))
        h = jnp.concatenate([x, t], axis=-1)

        x = self.obs_in(h)
        x = jax.nn.gelu(x)

        for i in range(len(self.middle_layers)):
            x = self.middle_layers[i](x)
            x = jax.nn.gelu(x)

        x = self.out(x)

        return x[...,None]

# %%
# Initialize the velocity field model
hidden_dim = 512

# velocity field model init
model = MLP(input_dim=2, hidden_dim=hidden_dim, rngs=nnx.Rngs(0), n_layers=12)

# %% id="N2WekuU6-wLH"
training_config = UnconditionalFlowPipeline.get_default_training_config()
training_config["checkpoint_dir"] = checkpoint_dir
training_config["nsteps"] = 500_000
training_config["max_lr"] = 1e-4 * factor
training_config["min_lr"] = 1e-6 * factor


pipeline = UnconditionalFlowPipeline(model,
    train_dataset_batched,
    val_dataset_batched,
    2,
    training_config=training_config)

# %% id="4jnodGVs-wLH"
# Restore the model from checkpoint if requested
if restore_model:
    pipeline.restore_model()

# %%
model_params = nnx.state(pipeline.model, nnx.Param)
total_params  = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(model_params))
print(f"Total model parameters: {total_params}")

# %% [markdown] id="QHRYClq5-wLI"
# ## 4. Training Loop
#
# This section defines the training and validation steps, and runs the training loop if enabled. Early stopping and learning rate scheduling are used for efficient training.

# %%
if train_model:
    # Train the model
    pipeline.train(nnx.Rngs(0))

# %% [markdown] id="mm5Cg6mI-wLJ"
# ## 5. Sampling from the Model
#
# In this section, we sample trajectories from the trained flow-matching model and visualize the results at different time steps.

# %% [markdown] id="N4AFN-07-wLJ"
# ### sample the model

# %% id="w4bKvrXV-wLJ"
key = jax.random.PRNGKey(42)
T = jnp.linspace(0,1,100)  # sample times
sol = pipeline.sample(key, nsamples=500_000, time_grid=T)

# %%
if normalize_data:
    samples = unnormalize(sol[-1,...,0])
else:
    samples = sol[-1,...,0]

# %% colab={"base_uri": "https://localhost:8080/", "height": 242} id="RQUHGNtB-wLJ" outputId="6bb74743-353f-4c77-b171-5c4f97cd1e35"
fig, ax = plt.subplots(figsize=(10, 5))
plot_samples(ax, np.array(samples))
plt.show()

# %% [markdown]
# Now we make an animation of the sampling

# %%
import imageio.v3 as imageio
import io
from tqdm import tqdm

# %%
# samples

images = []

for i in tqdm(range(len(sol))):
    fig, ax = plt.subplots(figsize=(10, 5))
    samples = np.array(sol[i,...,0])
    plot_samples(ax, samples)

    # ax.text(0, 1.03, f"t = {(i+1)/len(sol):.2f}", transform=plt.gca().transAxes)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    image = imageio.imread(buf)
    buf.close()
    if i == 0:
        images = []
    images.append(image)
    plt.close()

# %%
# repeat the last frame 10 times to make the gif pause at the end
# images += [images[-1]] * 20
images2 = images + [images[-1]] * 20

# %%
imageio.imwrite(
    'gensbi_animation.gif', 
    images2, 
    duration=3000/len(images2), 
    loop=0  # 0 means loop indefinitely
)

# %% [markdown]
# <img src="https://raw.githubusercontent.com/aurelio-amerio/GenSBI-examples/refs/heads/main/examples/NDE/gensbi_animation_small.gif" width=600>
