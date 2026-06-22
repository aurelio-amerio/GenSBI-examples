"""Two-moons NPE example: train a conditional MAF (RQSpline) and plot q(theta|x_o).

Run (on a GPU node with HF access):
    python train_maf_npe.py --config config/config_maf_npe.yaml

The flow IS the density model (ConditionalFlowPipeline, max-likelihood NPE).
Helpers are module-level and import-safe so they can be unit-tested on CPU.
"""

import os

# Import-safe (tests / module import): default to CPU. When run as the main
# training script we leave JAX_PLATFORMS unset so it can use a GPU.
if __name__ != "__main__":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse

import numpy as np
import yaml
import jax
import jax.numpy as jnp
from flax import nnx
import matplotlib.pyplot as plt

from gensbi.normalizing_flows import make_maf, Affine, RQSpline
from gensbi.recipes import ConditionalFlowPipeline
from gensbi.utils.plotting import plot_2d_dist_contour
from gensbi_examples.tasks import get_task, _load_precomputed_stats


def build_transformer(model_cfg):
    """Return the elementwise transformer named in the model config."""
    name = str(model_cfg.get("transformer", "rqspline")).lower()
    if name == "affine":
        return Affine()
    if name == "rqspline":
        return RQSpline(num_bins=int(model_cfg.get("num_bins", 8)))
    raise ValueError(f"unknown transformer {name!r} (expected 'affine' or 'rqspline')")


def build_flow(rngs, dim_obs, dim_cond, model_cfg):
    """Build the MAF Flow from the model config section."""
    transformer = build_transformer(model_cfg)
    return make_maf(
        rngs,
        dim=dim_obs,
        cond_dim=dim_cond,
        n_layers=int(model_cfg.get("n_layers", 8)),
        transformer=transformer,
        nn_width=int(model_cfg.get("nn_width", 64)),
        nn_depth=int(model_cfg.get("nn_depth", 2)),
        permutation=str(model_cfg.get("permutation", "reverse")),
        standardize=bool(model_cfg.get("standardize", True)),
        zero_init=bool(model_cfg.get("zero_init", True)),
    )


def build_training_config(config, checkpoint_dir):
    """Start from the pipeline defaults, overlay YAML optimizer+training, set ckpt dir.

    ConditionalFlowPipeline reads training_config keys eagerly in __init__ with no
    merge, so it must be complete. Extra keys (batch_size, nsamples, restore_model,
    train_model) are harmless — the pipeline only reads the keys it knows.
    """
    tc = ConditionalFlowPipeline.get_default_training_config()
    tc.update(config.get("optimizer", {}))
    tc.update(config.get("training", {}))
    tc["checkpoint_dir"] = checkpoint_dir
    return tc


def load_obs_stats(task_name, dim_obs):
    """Precomputed θ (obs) mean/std as shape (dim_obs,) — no fitting, no HF download.

    Stats ship as (1, dim_obs, 1) in gensbi_examples/stats/stats_<task>.npz and are
    loaded by the same internal helper the Task uses.
    """
    stats = _load_precomputed_stats(task_name)
    if stats is None:
        raise FileNotFoundError(f"no precomputed stats for task {task_name!r}")
    mean = jnp.asarray(stats["obs_mean"]).reshape(dim_obs)
    std = jnp.asarray(stats["obs_std"]).reshape(dim_obs)
    return mean, std
