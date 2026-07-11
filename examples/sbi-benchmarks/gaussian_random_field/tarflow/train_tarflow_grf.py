"""Train a conditional TarFlow p(field | theta) on gaussian_random_field (32x32).

Field-level NLE with a transformer autoregressive normalizing flow: the 32x32
GRF realization is the modeled variable (obs), theta = (log_std, alpha) the
condition. TarFlow uses the image tokenizer (patchify) with 2D rotary position
embeddings (use_rope=True) and the vector-prefix conditioning strategy: each
theta coordinate becomes one prefix token behind the prefix-LM mask.

Trained by exact max likelihood (ConditionalFlowPipeline, structured_obs=True);
sampling uses the KV-cached autoregressive sampler. Outputs: loss curves, a
truth-vs-samples field grid, radial power-spectrum overlays (simulator mean
+/- 1 sigma and the analytic power law), and the exact held-out NLL in
bits/dim -- a scalar the flow-matching baselines cannot provide.

Training data comes from the online sampler (fresh prior + simulator draws
each batch) by default; set `training.online: false` to train on the
pre-generated HF train split instead.

Usage (conda env `gensbi`):
    python train_tarflow_grf.py --config config/config_1.yaml
"""

import os

if __name__ != "__main__":
    os.environ["JAX_PLATFORMS"] = "cpu"
else:
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"
    os.environ.setdefault("JAX_PLATFORMS", "cuda")  # JAX_PLATFORMS=cpu wins

import argparse

import jax
from jax import numpy as jnp
import numpy as np
from flax import nnx
import yaml

import matplotlib

matplotlib.use("Agg")  # headless cluster nodes
import matplotlib.pyplot as plt

from gensbi.models import TarFlow, TarFlowParams
from gensbi.recipes import ConditionalFlowPipeline

from sbibm_jax.data import OnlineTaskDataset, TaskDataset

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))


def build_flow(rngs, model_cfg):
    """TarFlow over 32x32 fields: image tokenizer + rope, vector-prefix cond.

    cond="vector" and modeled="image" are fixed by design (see the spec);
    the yaml model section only sizes the architecture.
    """
    return TarFlow(TarFlowParams(
        rngs=rngs,
        modeled="image",
        img_size=int(model_cfg["img_size"]),
        patch_size=int(model_cfg["patch_size"]),
        img_channels=int(model_cfg.get("img_channels", 1)),
        cond="vector",
        cond_dim=int(model_cfg["cond_dim"]),
        cond_channels=1,
        use_rope=bool(model_cfg.get("use_rope", True)),
        rope_theta=int(model_cfg.get("rope_theta", 10000)),
        head_dim=int(model_cfg.get("head_dim", 32)),
        num_heads=int(model_cfg.get("num_heads", 8)),
        num_blocks=int(model_cfg.get("num_blocks", 8)),
        layers_per_block=int(model_cfg.get("layers_per_block", 2)),
        permutation=str(model_cfg.get("permutation", "flip")),
    ))


def build_training_config(config, checkpoint_dir):
    """Pipeline defaults overlaid with the yaml optimizer+training sections.

    ConditionalFlowPipeline reads training_config keys eagerly in __init__
    with no merge, so it must be complete. Extra script-level keys
    (batch_size, train_model, ...) are harmless -- the pipeline only reads
    the keys it knows.
    """
    tc = ConditionalFlowPipeline.get_default_training_config()
    tc.update(config.get("optimizer", {}))
    tc.update(config.get("training", {}))
    tc["checkpoint_dir"] = checkpoint_dir
    return tc


def to_obs_cond(batch):
    """Loader yields (theta, x); the flow pipeline wants (obs=x, cond=theta).

    x already arrives in native image shape (B, 32, 32, 1); theta (B, 2)
    gains the channel axis the VectorConditioner expects: (B, 2, 1).
    Module-level (not a lambda) so it survives pickling if it ever moves
    before a prefetch stage.
    """
    theta, x = batch
    return x, theta[..., None]


def heldout_bits_per_dim(flow, fields_norm, theta_norm, batch_size=64):
    """Exact mean NLL of held-out fields under the flow, in bits/dim.

    Inputs are in normalized units (the flow's training space); the constant
    Jacobian of the dataset normalization is omitted, so values compare
    across runs of this example, not across normalization schemes.
    fields_norm: (N, H, W, 1); theta_norm: (N, cond_dim, 1). Calls
    flow.log_prob directly -- batched cond is native TarFlow, no pipeline
    single-observation restriction.
    """
    n = fields_norm.shape[0]
    ndim = int(np.prod(fields_norm.shape[1:]))
    lps = []
    for i in range(0, n, batch_size):
        lp = flow.log_prob(jnp.asarray(fields_norm[i:i + batch_size]),
                           jnp.asarray(theta_norm[i:i + batch_size]))
        lps.append(np.asarray(lp, dtype=np.float64))
    return float(-np.concatenate(lps).mean() / (ndim * np.log(2.0)))
