"""Two-moons NPE example: train a conditional TarFlow (TransformerFlow) and plot q(theta|x_o).

Run (on a GPU node with HF access):
    python train_tarflow_npe.py --config config/config_tarflow_npe.yaml

The flow IS the density model (ConditionalFlowPipeline, max-likelihood NPE).
TransformerFlow is a transformer autoregressive normalizing flow (adapted from
apple/ml-tarflow); the pipeline is flow-agnostic, so this mirrors the MAF example
verbatim apart from the model factory and its config section.
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

from gensbi.models import TarFlow, TarFlowParams
from gensbi.recipes import ConditionalFlowPipeline
from gensbi.utils.plotting import plot_2d_dist_contour
from sbibm_jax.data import TaskDataset


def build_flow(rngs, dim_obs, dim_cond, model_cfg):
    """Build the TarFlow (transformer autoregressive flow) stack from the model config.

    TarFlowParams uses the head_dim/num_heads convention and derives the total width
    channels = head_dim * num_heads.
    """
    head_dim = int(model_cfg.get("head_dim", 16))
    num_heads = int(model_cfg.get("num_heads", 4))
    return TarFlow(TarFlowParams(
        rngs=rngs,
        dim=dim_obs,
        cond_dim=dim_cond,
        num_blocks=int(model_cfg.get("num_blocks", 8)),
        layers_per_block=int(model_cfg.get("layers_per_block", 2)),
        head_dim=head_dim,
        num_heads=num_heads,
        block_size=int(model_cfg.get("block_size", 1)),
        permutation=str(model_cfg.get("permutation", "flip")),
        standardize=bool(model_cfg.get("standardize", True)),
        zero_init=bool(model_cfg.get("zero_init", True)),
    ))


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


def load_obs_stats(task, dim_obs):
    """Precomputed θ mean/std as shape (dim_obs,) — read straight off the TaskDataset.

    ``TaskDataset`` exposes ``theta_mean``/``theta_std`` from the dataset metadata
    (shape ``(1, dim_obs)``) regardless of its ``normalize`` flag, so no fitting and
    no extra data pass is needed. θ is the autoregressive target ("obs") for NPE.
    """
    if task.theta_mean is None or task.theta_std is None:
        raise ValueError(f"task {getattr(task, 'name', task)!r} has no theta stats")
    mean = jnp.asarray(task.theta_mean).reshape(dim_obs)
    std = jnp.asarray(task.theta_std).reshape(dim_obs)
    return mean, std


def apply_standardization(pipeline, mean, std):
    """Set the θ standardization buffers on both model and EMA from precomputed stats.

    EMA averages only Params, so its non-Param standardization buffer must be set too.
    Marks the pipeline standardized to suppress the train-time 'did you fit?' warning.
    """
    pipeline.model.set_standardization(mean, std)
    pipeline.ema_model.set_standardization(mean, std)
    pipeline._standardized = True


def make_density_grid(ref_samples, grid_size, padding=0.5):
    """Build a 2D θ grid framing the reference samples (+padding).

    Returns (xx, yy, grid_pts): xx, yy are (G, G) meshgrids (indexing='xy');
    grid_pts is (G*G, 2) row-aligned with xx.ravel()/yy.ravel() (C order).
    """
    ref = np.asarray(ref_samples).reshape(-1, 2)
    lo = ref.min(axis=0) - padding
    hi = ref.max(axis=0) + padding
    xs = np.linspace(lo[0], hi[0], grid_size)
    ys = np.linspace(lo[1], hi[1], grid_size)
    xx, yy = np.meshgrid(xs, ys)
    grid_pts = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    return xx, yy, grid_pts


def posterior_density(pipeline, grid_pts, obs, grid_size, use_ema=True):
    """Evaluate q(θ|obs) on grid_pts and reshape to (G, G) aligned with the meshgrid."""
    grid_pts = jnp.asarray(grid_pts)[..., None]            # (G*G, 2) -> (G*G, 2, 1)
    obs = jnp.asarray(obs)
    if obs.ndim == 1:
        obs = obs[None, :, None]                            # (dim,) -> (1, dim, 1)
    logp = pipeline.log_prob(grid_pts, obs, use_ema=use_ema)  # (G*G,)
    Z = np.asarray(jnp.exp(logp)).reshape(grid_size, grid_size)
    return Z


def plot_posterior_contour(xx, yy, Z, true_param, ref_samples=None, n_ref_overlay=2000):
    """Contour plot of the posterior with the true θ marked and a light ref-sample overlay."""
    fig, ax = plot_2d_dist_contour(xx, yy, Z, true_param=np.asarray(true_param).reshape(-1))
    if ref_samples is not None and n_ref_overlay > 0:
        ref = np.asarray(ref_samples).reshape(-1, 2)[:n_ref_overlay]
        ax.scatter(ref[:, 0], ref[:, 1], s=2, alpha=0.15, color="k", zorder=5)
    return fig, ax


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(here, "config", "config_tarflow_npe.yaml")
    parser = argparse.ArgumentParser(description="Two-moons NPE (TarFlow) training/eval")
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
    eval_cfg = config["evaluation"]

    # --- task / data (raw loader: normalize=False; θ is standardized in-flow) ---
    task = TaskDataset(task_name, kind="conditional", normalize=False,
                       use_prefetching=True, max_workers=2)
    dim_obs, dim_cond = task.dim_theta, task.dim_x
    train_ds = task.get_train_loader(int(train_cfg["batch_size"]),
                                     num_samples=int(train_cfg["nsamples"]))
    val_ds = task.get_val_loader(512)

    # --- flow + pipeline ---
    flow = build_flow(nnx.Rngs(0), dim_obs, dim_cond, model_cfg)
    training_config = build_training_config(config, checkpoint_dir)
    pipeline = ConditionalFlowPipeline(flow, train_ds, val_ds, dim_obs, dim_cond,
                                       training_config=training_config)

    # --- standardize θ from dataset stats (no fitting; loader stays raw) ---
    mean, std = load_obs_stats(task, dim_obs)
    apply_standardization(pipeline, mean, std)

    # --- train / restore ---
    if train_cfg.get("restore_model", False):
        print("Restoring model from checkpoint...")
        pipeline.restore_model()
    if train_cfg.get("train_model", True):
        print("Starting training...")
        pipeline.train(nnx.Rngs(0))
        print("Training complete.")

    # --- evaluate posterior + contour plot for one observation ---
    idx = int(eval_cfg["observation_idx"])
    obs, ref_samples = task.get_reference(idx)
    true_param = np.asarray(task.get_true_parameters(idx)).reshape(-1)

    grid_size = int(eval_cfg["grid_size"])
    xx, yy, grid_pts = make_density_grid(ref_samples, grid_size,
                                         padding=float(eval_cfg.get("padding", 0.5)))
    Z = posterior_density(pipeline, grid_pts, obs, grid_size, use_ema=True)
    # fig, ax = plot_posterior_contour(xx, yy, Z, true_param, ref_samples=ref_samples,
    #                                  n_ref_overlay=int(eval_cfg.get("n_ref_overlay", 2000)))
    fig, ax = plot_2d_dist_contour(xx, yy, Z, true_param=np.asarray(true_param).reshape(-1))

    out_path = os.path.join(img_dir, f"posterior_contour_obs{idx}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved posterior contour to {out_path}")


if __name__ == "__main__":
    main()
