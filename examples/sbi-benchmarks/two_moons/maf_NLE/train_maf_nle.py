"""Two-moons NLE example: train a conditional MAF likelihood q(x|theta) and
sample the posterior with NUTS (MCMC).

Run (on a GPU node with HF access, via the gensbi conda env):
    python train_maf_nle.py --config config/config_maf_nle.yaml

NLE convention (mirror of the NPE script): obs = x, cond = theta, so the flow
models q(x | theta). The posterior is recovered at inference time by combining the
learned likelihood with the task prior under NUTS via gensbi NLEPosterior. Helpers
are module-level and import-safe so they can be smoke-tested on CPU.
"""

import os

# Import-safe (module import / CPU smoke checks): default to CPU. When run as the
# main training script we leave JAX_PLATFORMS unset so it can use a GPU.
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
from gensbi.inference import NLEPosterior
from gensbi.utils.plotting import plot_marginals
from sbibm_jax.data import TaskDataset
from sbibm_jax.tasks import get_task


def build_transformer(model_cfg):
    """Return the elementwise transformer named in the model config."""
    name = str(model_cfg.get("transformer", "rqspline")).lower()
    if name == "affine":
        return Affine()
    if name == "rqspline":
        return RQSpline(num_bins=int(model_cfg.get("num_bins", 8)))
    raise ValueError(f"unknown transformer {name!r} (expected 'affine' or 'rqspline')")


def build_flow(rngs, dim_obs, dim_cond, model_cfg):
    """Build the MAF Flow from the model config section.

    For NLE, dim_obs == task.dim_x (autoregressive target) and dim_cond == task.dim_theta.
    """
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
    """Pipeline defaults overlaid with YAML optimizer+training, with ckpt dir set.

    ConditionalFlowPipeline reads training_config keys eagerly in __init__, so it must
    be complete. Extra keys are harmless — the pipeline only reads the keys it knows.
    """
    tc = ConditionalFlowPipeline.get_default_training_config()
    tc.update(config.get("optimizer", {}))
    tc.update(config.get("training", {}))
    tc["checkpoint_dir"] = checkpoint_dir
    return tc


def swap_obs_cond(batch):
    """Map a conditional batch (theta, x) -> (x, theta) for NLE.

    TaskDataset's conditional collate yields (theta, x); the pipeline reads
    obs, cond = batch. NLE models q(x | theta), so obs must be x and cond theta.
    """
    theta, x = batch
    return x, theta


def load_x_stats(task, dim_obs):
    """Precomputed x mean/std as shape (dim_obs,), read straight off the TaskDataset.

    NLE analog of the NPE script's load_obs_stats: x is the autoregressive target
    ("obs") for NLE, so its stats drive the in-flow Standardize bijection. The stats
    ship as shape (1, dim_obs) regardless of the loader's normalize flag.
    """
    if task.x_mean is None or task.x_std is None:
        raise ValueError(f"task {getattr(task, 'name', task)!r} has no x stats")
    mean = jnp.asarray(task.x_mean).reshape(dim_obs)
    std = jnp.asarray(task.x_std).reshape(dim_obs)
    return mean, std


def apply_standardization(pipeline, mean, std):
    """Set the obs (=x) Standardize buffers on both model and EMA from precomputed stats.

    EMA averages only Params, so its non-Param Standardize buffer must be set too.
    Marks the pipeline standardized to suppress the train-time 'did you fit?' warning.
    """
    pipeline.model.set_standardization(mean, std)
    pipeline.ema_model.set_standardization(mean, std)
    pipeline._standardized = True


def build_prior(task_name, validate_args=True):
    """Return the task's numpyro prior over theta, with out-of-support log_prob = -inf.

    get_task(...).get_prior_dist() ships with validate_args=False, so its log_prob
    returns the in-support constant *everywhere* and does NOT bound the NLE potential —
    NUTS then wanders outside the prior support and the posterior is wrong. Re-enabling
    validation makes log_prob = -inf outside the support, confining NUTS to the box.
    """
    prior = get_task(task_name).get_prior_dist()
    prior._validate_args = validate_args
    if hasattr(prior, "base_dist"):          # Independent wraps a base distribution
        prior.base_dist._validate_args = validate_args
    return prior


def build_posterior(pipeline, prior, mcmc_cfg, use_ema=True):
    """Wrap the trained likelihood flow + validated prior in an NLE NUTS posterior."""
    flow = pipeline.ema_model if use_ema else pipeline.model
    return NLEPosterior(
        flow,
        prior,
        num_warmup=int(mcmc_cfg.get("num_warmup", 1000)),
        num_samples=int(mcmc_cfg.get("num_samples", 50000)),
        num_chains=int(mcmc_cfg.get("num_chains", 1)),
    )


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(here, "config", "config_maf_nle.yaml")
    parser = argparse.ArgumentParser(description="Two-moons NLE (MAF) training/eval")
    parser.add_argument("--config", type=str, default=default_config,
                        help="Path to the YAML config.")
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    exp_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(exp_dir, "checkpoints")
    img_dir = os.path.join(exp_dir, "imgs")
    os.makedirs(img_dir, exist_ok=True)

    task_name = config["task_name"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    mcmc_cfg = config["mcmc"]
    eval_cfg = config["evaluation"]

    # --- task / data (raw loader: normalize=False; x is standardized in-flow) ---
    task = TaskDataset(task_name, kind="conditional", normalize=False,
                       use_prefetching=True, max_workers=2)
    # NLE: obs = x, cond = theta (mirror of NPE). Swap the (theta, x) batches.
    dim_obs, dim_cond = task.dim_x, task.dim_theta
    train_ds = task.get_train_loader(
        int(train_cfg["batch_size"]),
        num_samples=int(train_cfg["nsamples"]),
    ).map(swap_obs_cond)
    val_ds = task.get_val_loader(512).map(swap_obs_cond)

    # --- flow + pipeline ---
    flow = build_flow(nnx.Rngs(0), dim_obs, dim_cond, model_cfg)
    training_config = build_training_config(config, checkpoint_dir)
    pipeline = ConditionalFlowPipeline(flow, train_ds, val_ds, dim_obs, dim_cond,
                                       training_config=training_config)

    # --- standardize x from dataset stats (no fitting; loader stays raw) ---
    mean, std = load_x_stats(task, dim_obs)
    apply_standardization(pipeline, mean, std)

    # --- train / restore ---
    if train_cfg.get("restore_model", False):
        print("Restoring model from checkpoint...")
        pipeline.restore_model()
    if train_cfg.get("train_model", True):
        print("Starting training...")
        pipeline.train(nnx.Rngs(0))
        print("Training complete.")

    # --- NLE posterior via NUTS for one observation ---
    idx = int(eval_cfg["observation_idx"])
    obs, _ = task.get_reference(idx)
    true_param = np.asarray(task.get_true_parameters(idx)).reshape(-1)

    prior = build_prior(task_name)
    posterior = build_posterior(pipeline, prior, mcmc_cfg, use_ema=True)
    print(f"Sampling posterior with NUTS "
          f"(warmup={mcmc_cfg['num_warmup']}, samples={mcmc_cfg['num_samples']})...")
    samples = posterior.sample(jax.random.PRNGKey(0), obs)   # (n, dim_theta, 1)

    # --- contour plot of the posterior samples with the true value marked ---
    plot_marginals(np.asarray(samples[..., 0]), plot_levels=False, backend="seaborn",
                   gridsize=50, true_param=true_param)
    out_path = os.path.join(img_dir, f"posterior_marginals_obs{idx}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close("all")
    print(f"Saved posterior marginals to {out_path}")


if __name__ == "__main__":
    main()
