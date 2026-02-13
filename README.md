# GenSBI Examples
[![Build](https://github.com/aurelio-amerio/GenSBI-examples/actions/workflows/python-app.yml/badge.svg)](https://github.com/aurelio-amerio/GenSBI-examples/actions/workflows/python-app.yml)
![Coverage](https://raw.githubusercontent.com/aurelio-amerio/GenSBI-examples/refs/heads/main/img/badges/coverage.svg)
[![Version](https://img.shields.io/pypi/v/gensbi-examples.svg?maxAge=3600)](https://pypi.org/project/gensbi-examples/)
[![Downloads](https://pepy.tech/badge/gensbi-examples)](https://pepy.tech/project/gensbi-examples)

This repository contains a collection of examples, tutorials, and recipes for **GenSBI**, a JAX-based library for Simulation-Based Inference using generative models.

These examples demonstrate how to use GenSBI for various tasks, including:

- Defining and running inference pipelines.
- Using different embedding networks (MLP, ResNet, etc.).
- Handling various data types (1D signals, 2D images).

## Installation

### Prerequisites

You need to have **GenSBI** and the examples package installed.

**With CUDA 12 support (Recommended):**

```bash
pip install gensbi[cuda12, examples]
```

**CPU-only:**

```bash
pip install gensbi
```

### Install Examples Package

To download these examples, clone the github repository:

```bash
git clone https://github.com/aurelio-amerio/GenSBI-examples.git
```

## Structure

- `examples/`: Contains standalone example scripts and notebooks.
- `src/gensbi_examples`: Helper utilities for the examples.

## Getting Started

- **My First Model**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-amerio/GenSBI-examples/blob/main/examples/getting_started/my_first_model.ipynb)

## Neural Density Estimators (NDE)

These examples demonstrate the usage of Neural Density Estimators for unconditional density estimation tasks.

- **Diffusion 2D Unconditional**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-amerio/GenSBI-examples/blob/main/examples/NDE/diffusion_2d_unconditional.ipynb)
- **Flow Matching 2D Unconditional**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-amerio/GenSBI-examples/blob/main/examples/NDE/flow_matching_2d_unconditional.ipynb)
- **Flow Matching 2D Unconditional (Flux1Joint)**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-amerio/GenSBI-examples/blob/main/examples/NDE/flow_matching_2d_unconditional_flux1joint.ipynb)

## SBI Benchmark Examples

This repository includes a comprehensive suite of Simulation-Based Inference (SBI) benchmarks. These examples cover a range of standard tasks used to evaluate SBI methods, including simple distributions, physical systems, and complex toy problems. For each task, we provide implementations using various generative models available in GenSBI, such as Flow Matching and Diffusion models with different architectures (Flux, SimFormer).

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