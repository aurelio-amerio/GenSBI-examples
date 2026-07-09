# GenSBI Examples
<!-- [![Build](https://github.com/aurelio-amerio/GenSBI-examples/actions/workflows/python-app.yml/badge.svg)](https://github.com/aurelio-amerio/GenSBI-examples/actions/workflows/python-app.yml)
![Coverage](https://raw.githubusercontent.com/aurelio-amerio/GenSBI-examples/refs/heads/main/img/badges/coverage.svg) -->
<!-- [![Downloads](https://pepy.tech/badge/gensbi)](https://pepy.tech/project/gensbi)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20407738.svg)](https://doi.org/10.5281/zenodo.20407738) -->

This repository contains a collection of examples, tutorials, and recipes for **GenSBI**, a JAX-based library for Simulation-Based Inference using generative methods.

These examples demonstrate how to use GenSBI for various tasks, including:

- Defining and running inference pipelines.
- Using different embedding networks (MLP, ResNet, etc.).
- Handling various data types (1D signals, 2D images).

## Installation

To run these examples you need **GenSBI** installed with its `examples` extra, which pulls in the additional dependencies used by the notebooks and scripts (plotting, dataset loading, etc.). The example notebooks and training scripts themselves live in this repository — clone it as described below.

### Using uv (recommended)

```bash
uv add gensbi[examples]
# or, for a standalone install:
uv pip install gensbi[examples]
```

For GPU support (CUDA 12):

```bash
uv add gensbi[cuda12,examples]
# or
uv pip install gensbi[cuda12,examples]
```

### Using pip

```bash
pip install gensbi[examples]
# with GPU support (CUDA 12):
pip install gensbi[cuda12,examples]
```

For more installation options, including how to install `uv`, see the [Installation Guide](https://aurelio-amerio.github.io/GenSBI/getting_started/installation.html).

### Download the example notebooks and scripts

The notebooks and training scripts live in this repository. Task data is loaded via [`sbibm-jax`](https://github.com/aurelio-amerio/sbibm-jax) (`pip install "sbibm-jax[loader]"`). To get the examples, clone the repo:

```bash
git clone https://github.com/aurelio-amerio/GenSBI-examples.git
```

## Structure

- `examples/`: Contains standalone example scripts and notebooks.

## Getting Started

- **My First Model**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-amerio/GenSBI-examples/blob/main/examples/getting_started/my_first_model.ipynb)

## Training Models

Models can be trained from the command line by passing a YAML configuration file to one of the provided training scripts. Each configuration file specifies the task, model architecture, training methodology (flow matching, diffusion, or score matching), optimizer settings, and training hyperparameters. Example configuration files can be found alongside each benchmark task under the `config/` subdirectory (e.g., `examples/sbi-benchmarks/two_moons/flow_flux/config/config_flow_flux.yaml`).

**General SBI tasks.** For the standard SBI benchmark tasks (Two Moons, Gaussian Linear, SLCP, etc.), use the general-purpose training script:

```bash
python scripts/train_sbi_model.py --config <path_to_config.yaml>
```

This script trains the model, runs sampling, and computes diagnostic metrics (C2ST, TARP, SBC, LC2ST) automatically.

**SBIBM benchmarks (budget/model/methodology scans).** To reproduce the results reported in the main paper—including scans over training budget, model type, and methodology—use the dedicated SBIBM training script:

```bash
python scripts/train_sbi_model_sbibm.py --config <path_to_config.yaml>
```

This script additionally accepts a `--dsize` flag to control the training dataset size (default: 100,000), which is used for the budget scan experiments:

```bash
python scripts/train_sbi_model_sbibm.py --config <path_to_config.yaml> --dsize 10000
```

**Advanced tasks.** For more complex tasks such as gravitational waves and strong lensing, dedicated training scripts are provided in the corresponding example directories (e.g., `examples/sbi-benchmarks/gravitational_waves/train-gw.py` and `examples/sbi-benchmarks/lensing/train-lensing.py`).

## Neural Density Estimators (NDE)

These examples demonstrate the usage of Neural Density Estimators for unconditional density estimation tasks.

- **Diffusion EDM 2D Unconditional**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-amerio/GenSBI-examples/blob/main/examples/NDE/diffusion_EDM_2d_unconditional.ipynb)
- **Diffusion SM 2D Unconditional**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-amerio/GenSBI-examples/blob/main/examples/NDE/diffusion_SM_2d_unconditional.ipynb)
- **Flow Matching 2D Unconditional**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-amerio/GenSBI-examples/blob/main/examples/NDE/flow_matching_2d_unconditional.ipynb)
- **Flow Matching 2D Unconditional (Flux1Joint)**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-amerio/GenSBI-examples/blob/main/examples/NDE/flow_matching_2d_unconditional_flux1joint.ipynb)

## SBI Benchmark Examples

This repository includes a comprehensive suite of Simulation-Based Inference (SBI) benchmarks. These examples cover a range of standard tasks used to evaluate SBI methods, including simple distributions, physical systems, and complex toy problems. For each task, we provide implementations using various generative methods available in GenSBI, such as Flow Matching and Diffusion models with different architectures (Flux, SimFormer).

### Two Moons

- **Diffusion Flux**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-amerio/GenSBI-examples/blob/main/examples/sbi-benchmarks/two_moons/diffusion_flux/two_moons_diffusion_flux.ipynb)
- **Diffusion Flux1Joint**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-amerio/GenSBI-examples/blob/main/examples/sbi-benchmarks/two_moons/diffusion_flux1joint/two_moons_diffusion_flux1joint.ipynb)
- **Diffusion SimFormer**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-amerio/GenSBI-examples/blob/main/examples/sbi-benchmarks/two_moons/diffusion_simformer/two_moons_diffusion_simformer.ipynb)
- **Flow Flux**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-amerio/GenSBI-examples/blob/main/examples/sbi-benchmarks/two_moons/flow_flux/two_moons_flow_flux.ipynb)
- **Flow Flux1Joint**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-amerio/GenSBI-examples/blob/main/examples/sbi-benchmarks/two_moons/flow_flux1joint/two_moons_flow_flux1joint.ipynb)
- **Flow SimFormer**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-amerio/GenSBI-examples/blob/main/examples/sbi-benchmarks/two_moons/flow_simformer/two_moons_flow_simformer.ipynb)

### Bernoulli GLM

- **Flow Flux**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-amerio/GenSBI-examples/blob/main/examples/sbi-benchmarks/bernoulli_glm/flow_flux/bernoulli_glm_flow_flux.ipynb)
- **Flow Flux1Joint**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-amerio/GenSBI-examples/blob/main/examples/sbi-benchmarks/bernoulli_glm/flow_flux1joint/bernoulli_glm_flow_flux1joint.ipynb)

### Gaussian Linear

- **Flow Flux**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-amerio/GenSBI-examples/blob/main/examples/sbi-benchmarks/gaussian_linear/flow_flux/gaussian_linear_flow_flux.ipynb)
- **Flow Flux1Joint**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-amerio/GenSBI-examples/blob/main/examples/sbi-benchmarks/gaussian_linear/flow_flux1joint/gaussian_linear_flow_flux1joint.ipynb)

### Gaussian Mixture

- **Flow Flux**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-amerio/GenSBI-examples/blob/main/examples/sbi-benchmarks/gaussian_mixture/flow_flux/gaussian_mixture_flow_flux.ipynb)
- **Flow Flux1Joint**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-amerio/GenSBI-examples/blob/main/examples/sbi-benchmarks/gaussian_mixture/flow_flux1joint/gaussian_mixture_flow_flux1joint.ipynb)

### Gravitational Waves

- **GW Example**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-amerio/GenSBI-examples/blob/main/examples/sbi-benchmarks/gravitational_waves/gw_example.ipynb)

### Lensing

- **Lensing Example**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-amerio/GenSBI-examples/blob/main/examples/sbi-benchmarks/lensing/lensing_example.ipynb)

### SLCP

- **Flow Flux**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-amerio/GenSBI-examples/blob/main/examples/sbi-benchmarks/slcp/flow_flux/slcp_flow_flux.ipynb)
- **Flow Flux1Joint**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-amerio/GenSBI-examples/blob/main/examples/sbi-benchmarks/slcp/flow_flux1joint/slcp_flow_flux1joint.ipynb)
- **Flow SimFormer**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-amerio/GenSBI-examples/blob/main/examples/sbi-benchmarks/slcp/flow_simformer/slcp_flow_simformer.ipynb)
- **TarFlow NLE**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-amerio/GenSBI-examples/blob/main/examples/sbi-benchmarks/slcp/tarflow_NLE/slcp_tarflow_nle.ipynb)

