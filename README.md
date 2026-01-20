# GenSBI Examples

This repository contains a collection of examples, tutorials, and recipes for **GenSBI**, a JAX-based library for Simulation-Based Inference using generative models.

These examples demonstrate how to use GenSBI for various tasks, including:

- Defining and running inference pipelines.
- Using different embedding networks (MLP, ResNet, etc.).
- Handling various data types (1D signals, 2D images).

## Installation

### Prerequisites

You need to have **GenSBI** installed.

**With CUDA 12 support (Recommended):**

```bash
pip install gensbi[cuda12]
```

**CPU-only:**

```bash
pip install gensbi
```

### Install Examples Package

To run the examples and ensure all dependencies are met, install this package:

```bash
pip install gensbi-examples
```

## Structure

- `examples/`: Contains standalone example scripts and notebooks.
- `src/gensbi_examples`: Helper utilities for the examples.
