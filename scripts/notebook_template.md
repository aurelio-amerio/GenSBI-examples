---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  display_name: gensbi
  language: python
  name: python3
---

# {TASKNAME} {model_architecture} {technique} Example
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-amerio/GenSBI-examples/blob/main/examples/sbi-benchmarks/{task_name_gensbi}/{model_name}/{task_name_gensbi}_{model_name}.ipynb)
> Notice: This notebook has been automatically generated. If you find any errors, please [open an issue](https://github.com/aurelio-amerio/GenSBI-examples/issues) on the GenSBI-examples GitHub repository.

+++


---
This notebook demonstrates {kind} {matching_technique} on the {TASKNAME} task using JAX and Flax. 

## Table of Contents
| Section | Description |
|---|---|
| 1. [Introduction & Setup](#introduction-setup) | Overview, environment, device, autoreload |
| 2. [Task & Data Preparation](#task-data-preparation) | Define task, visualize data, create datasets |
| 3. [Model Configuration & Definition](#model-configuration-definition) | Load config, set parameters, instantiate model |
| 4. [Training](#training) | Train or restore model, manage checkpoints |
| 5. [Evaluation & Visualization](#evaluation-visualization) | Visualize loss, sample posterior, compute log prob |
| 6. [Diagnostics](#diagnostics) | Run diagnostics (TARP, SBC, L-C2ST) |

---

+++



## 1. Introduction & Setup
---
In this section, we introduce the problem, set up the computational environment, import required libraries, configure JAX for CPU or GPU usage, and enable autoreload for iterative development. Compatibility with Google Colab is also ensured.

```{code-cell} ipython3
# Check if running on Colab and install dependencies if needed
try:
    import google.colab
    colab = True
except ImportError:
    colab = False

if colab:
    # Install required packages and clone the repository
    %pip install --quiet "gensbi[cuda12] @ git+https://github.com/aurelio-amerio/GenSBI"
    %pip install --quiet "gensbi-examples @ git+https://github.com/aurelio-amerio/GenSBI-examples"
    !git clone --depth 1 https://github.com/aurelio-amerio/GenSBI-examples
    %cd GenSBI-examples/examples/sbi-benchmarks/{task_name_gensbi}/{model_name}
```

```{code-cell} ipython3
import os
# select device

os.environ["JAX_PLATFORMS"] = "cuda" 
# os.environ["JAX_PLATFORMS"] = "cpu" 
```

## 2. Task & Data Preparation
---
In this section, we define the {TASKNAME} task, visualize reference samples, and create the training and validation datasets required for model learning. Batch size and sample count are set for reproducibility and performance.

```{code-cell} ipython3
restore_model=True
train_model=False
```

```{code-cell} ipython3
import orbax.checkpoint as ocp
# get the current notebook path
notebook_path = os.getcwd()
checkpoint_dir = os.path.join(notebook_path, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)
```

```{code-cell} ipython3
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from flax import nnx

from numpyro import distributions as dist
import numpy as np
```

```{code-cell} ipython3
from gensbi.utils.plotting import plot_marginals
```

```{code-cell} ipython3
from gensbi_examples.tasks import get_task
task = get_task("{task_name_gensbi}", kind="{kind}", use_multiprocessing=False)
```

```{code-cell} ipython3
# reference posterior for an observation
obs, reference_samples = task.get_reference(num_observation=8)
```

```{code-cell} ipython3
# plot the 2D posterior 
plot_marginals(np.asarray(reference_samples, dtype=np.float32), gridsize=50, plot_levels=False, backend="seaborn")
plt.show()
```

```{code-cell} ipython3
# make a dataset
nsamples = int(1e5)
```

```{code-cell} ipython3
# Set batch size for training. Larger batch sizes help prevent overfitting, but are limited by available GPU memory.
batch_size = 4096
# Create training and validation datasets using the {TASKNAME} task object.
train_dataset = task.get_train_dataset(batch_size)
val_dataset = task.get_val_dataset(batch_size)

# Create iterators for the training and validation datasets.
dataset_iter = iter(train_dataset)
val_dataset_iter = iter(val_dataset)
```

## 3. Model Configuration & Definition
---
In this section, we load the model and optimizer configuration, set all relevant parameters, and instantiate the {model_architecture} model. Edge masks and marginalization functions are used for flexible inference and posterior estimation.

```{code-cell} ipython3
from gensbi.recipes import {model_architecture}{technique}Pipeline
```

```{code-cell} ipython3
import yaml

# Path to the configuration file.
config_path = f"{notebook_path}/config/config_{model_name}.yaml"
```

```{code-cell} ipython3
# Extract dimensionality information from the task object.
dim_obs = task.dim_obs  # Number of parameters to infer
dim_cond = task.dim_cond    # Number of observed data dimensions

dim_joint = task.dim_joint  # Joint dimension (for model input)
```

```{code-cell} ipython3
pipeline = {model_architecture}{technique}Pipeline.init_pipeline_from_config(
        train_dataset,
        val_dataset,
        dim_obs,
        dim_cond,
        config_path,
        checkpoint_dir,
    )
```

## 4. Training
---
In this section, we train the model or restore a checkpoint.

```{code-cell} ipython3
# pipeline.train(nnx.Rngs(0), save_model=False)
```

```{code-cell} ipython3
pipeline.restore_model()
```

## 5. Evaluation & Visualization
---
In this section, we evaluate the trained Simformer model by sampling from the posterior, and comparing results to reference data.

+++

### Section 5.1: Posterior Sampling
---
In this section, we sample from the posterior distribution using the trained model and visualize the results. Posterior samples are generated for a selected observation and compared to reference samples to assess model accuracy.

```{code-cell} ipython3
# we want to do {kind} inference. We need an observation for which we want to ocmpute the posterior
def get_samples(idx, nsamples=10_000, use_ema=False, key=None):
    observation, reference_samples = task.get_reference(idx)
    true_param = jnp.array(task.get_true_parameters(idx))

    if key is None:
        key = jax.random.PRNGKey(42)

    time_grid = jnp.linspace(0,1,100)

    samples = pipeline.sample(key, observation, nsamples, use_ema=use_ema, time_grid=time_grid)
    return samples, true_param, reference_samples
```

```{code-cell} ipython3
samples, true_param, reference_samples =  get_samples(8)
```

### Section 5.2: Visualize Posterior Samples
---
In this section, we plot the posterior samples as a 2D histogram to visualize the learned distribution and compare it to the ground truth.

```{code-cell} ipython3
from gensbi.utils.plotting import plot_marginals, plot_2d_dist_contour
```

```{code-cell} ipython3
plot_marginals(samples[-1,...,0], backend="seaborn", gridsize=50)
plt.show()

# alternatively use "corner" to plot containment levels too
# plot_marginals(samples[-1,...,0], backend="corner", gridsize=20)
# plt.show()
```

<img src="https://raw.githubusercontent.com/aurelio-amerio/GenSBI-examples/refs/heads/main/examples/sbi-benchmarks/{task_name_gensbi}/{model_name}/imgs/marginals_ema.png" width=800>

+++

## 6. Diagnostics

+++

We report here the results of the posterior calibration tests. As an excercise, you can implement the tests as in the Two Moons example and compare the results. 

+++

**Average C2ST**: {C2ST_ACCURACY} ± {C2ST_STD}

+++

**TARP:** <br><br>
<img src="https://raw.githubusercontent.com/aurelio-amerio/GenSBI-examples/refs/heads/main/examples/sbi-benchmarks/{task_name_gensbi}/{model_name}/imgs/tarp.png" width=400>

+++

**SBC** <br><br>

<img src="https://raw.githubusercontent.com/aurelio-amerio/GenSBI-examples/refs/heads/main/examples/sbi-benchmarks/{task_name_gensbi}/{model_name}/imgs/sbc.png" width=800>

+++

**L-C2ST**<br><br>
<img src="https://raw.githubusercontent.com/aurelio-amerio/GenSBI-examples/refs/heads/main/examples/sbi-benchmarks/{task_name_gensbi}/{model_name}/imgs/lc2st.png" width=400>
