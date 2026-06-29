"""SLCP NPE example: train a conditional TarFlow (TransformerFlow) q(theta|x) and
plot the (5-D, multimodal) posterior marginals for one observation.

Run (on a GPU node with HF access, via the gensbi conda env):
    python train_tarflow_npe.py --config config/config_tarflow_npe.yaml

The flow IS the density model (ConditionalFlowPipeline, max-likelihood NPE): obs =
theta, cond = x, so it models q(theta | x). For SLCP, theta is the 5-D parameter and
x the 8-D observation (4 i.i.d. 2-D points). NPE needs no separate inference step --
the trained flow is already the (amortized) posterior -- so we just draw samples from
q(theta | x_o) and corner-plot their marginals. SLCP posteriors are multimodal (the
scale params enter the likelihood squared and the correlation through tanh, giving
sign-symmetric modes); the conditional flow captures this directly, no MCMC/SMC needed.
TransformerFlow is a transformer autoregressive normalizing flow (adapted from
apple/ml-tarflow); the pipeline is flow-agnostic, so this mirrors the MAF NPE example
verbatim apart from the model factory and its config section. Helpers are module-level
and import-safe so they can be smoke-tested on CPU.
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

from gensbi.models import TarFlow, TarFlowParams
from gensbi.recipes import ConditionalFlowPipeline
from gensbi.diagnostics.metrics import c2st
from gensbi.utils.plotting import plot_marginals
from sbibm_jax.data import TaskDataset


def build_flow(rngs, dim_obs, dim_cond, model_cfg):
    """Build the TarFlow (transformer autoregressive flow) stack from the model config.

    TarFlowParams uses the head_dim/num_heads convention and derives the total width
    channels = head_dim * num_heads.

    For NPE, dim_obs == task.dim_theta (autoregressive target) and dim_cond == task.dim_x.
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
    train_model) are harmless -- the pipeline only reads the keys it knows.
    """
    tc = ConditionalFlowPipeline.get_default_training_config()
    tc.update(config.get("optimizer", {}))
    tc.update(config.get("training", {}))
    tc["checkpoint_dir"] = checkpoint_dir
    return tc


def load_theta_stats(task, dim_obs):
    """Precomputed theta mean/std as shape (dim_obs,) -- read straight off the TaskDataset.

    TaskDataset exposes theta_mean/theta_std from the dataset metadata (shape
    (1, dim_obs)) regardless of its normalize flag, so no fitting and no extra data
    pass is needed. theta is the autoregressive target ("obs") for NPE.
    """
    if task.theta_mean is None or task.theta_std is None:
        raise ValueError(f"task {getattr(task, 'name', task)!r} has no theta stats")
    mean = jnp.asarray(task.theta_mean).reshape(dim_obs)
    std = jnp.asarray(task.theta_std).reshape(dim_obs)
    return mean, std


def apply_standardization(pipeline, mean, std):
    """Set the obs (=theta) Standardize buffers on both model and EMA from precomputed stats.

    EMA averages only Params, so its non-Param Standardize buffer must be set too.
    Marks the pipeline standardized to suppress the train-time 'did you fit?' warning.
    """
    pipeline.model.set_standardization(mean, std)
    pipeline.ema_model.set_standardization(mean, std)
    pipeline._standardized = True


def compute_c2st(pipeline, task, key=None):
    """C2ST accuracy per observation for the raw and EMA posteriors.

    For each of the task's reference observations, draw posterior samples from the
    amortized flow q(theta | x_o) and score them against the reference posterior samples
    with a classifier two-sample test. Accuracy ~0.5 means the learned posterior is
    indistinguishable from the reference; ~1.0 means they are easily told apart. The
    posterior sample count is matched to the reference count so the two c2st classes
    stay balanced (an imbalance would shift the chance accuracy away from 0.5). NPE
    sampling is amortized and cheap, so both the raw and EMA flows are evaluated.
    Returns {"raw": [...], "ema": [...]} of per-observation accuracies.
    """
    if key is None:
        key = jax.random.PRNGKey(42)
    results = {"raw": [], "ema": []}
    for tag, use_ema in (("raw", False), ("ema", True)):
        for idx in range(1, task.num_observations + 1):
            obs, reference_samples = task.get_reference(idx)
            n = reference_samples.shape[0]
            key, subkey = jax.random.split(key)
            samples = pipeline.sample(subkey, obs, nsamples=n, use_ema=use_ema)
            acc = float(c2st(reference_samples, samples[..., 0]))
            results[tag].append(acc)
            print(f"C2ST [{tag}] observation={idx}: {acc:.4f}")
    return results


def save_c2st_results(c2st_dir, accuracies, tag, experiment_id, method, model):
    """Write per-observation c2st accuracies and their mean +- std to a txt file.

    Mirrors scripts/train_sbi_model.py. tag is "raw" or "ema" and selects both the
    filename suffix and the in-file labels. Returns the path written.
    """
    suffix = "_ema" if tag == "ema" else ""
    label = "EMA " if tag == "ema" else ""
    path = os.path.join(
        c2st_dir, f"c2st_results{suffix}_{experiment_id}_{method}_{model}.txt"
    )
    with open(path, "w") as f:
        for idx, acc in enumerate(accuracies, start=1):
            f.write(f"C2ST accuracy {label}for observation={idx}: {acc:.4f}\n")
        f.write(
            f"Average C2ST accuracy {label}: "
            f"{np.mean(accuracies):.4f} +- {np.std(accuracies):.4f}\n"
        )
    return path


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(here, "config", "config_tarflow_npe.yaml")
    parser = argparse.ArgumentParser(description="SLCP NPE (TarFlow) training/eval")
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

    # --- task / data (raw loader: normalize=False; theta is standardized in-flow) ---
    task = TaskDataset(task_name, kind="conditional", normalize=False,
                       use_prefetching=True, max_workers=2)
    # NPE: obs = theta, cond = x. TaskDataset's conditional collate yields (theta, x)
    # and the pipeline reads obs, cond = batch, so no swap is needed.
    dim_obs, dim_cond = task.dim_theta, task.dim_x
    train_ds = task.get_train_loader(int(train_cfg["batch_size"]),
                                     num_samples=int(train_cfg["nsamples"]))
    val_ds = task.get_val_loader(512)

    # --- flow + pipeline ---
    flow = build_flow(nnx.Rngs(0), dim_obs, dim_cond, model_cfg)
    training_config = build_training_config(config, checkpoint_dir)
    pipeline = ConditionalFlowPipeline(flow, train_ds, val_ds, dim_obs, dim_cond,
                                       training_config=training_config)

    # --- standardize theta from dataset stats (no fitting; loader stays raw) ---
    mean, std = load_theta_stats(task, dim_obs)
    apply_standardization(pipeline, mean, std)

    # --- train / restore ---
    if train_cfg.get("restore_model", False):
        print("Restoring model from checkpoint...")
        pipeline.restore_model()
    if train_cfg.get("train_model", True):
        print("Starting training...")
        pipeline.train(nnx.Rngs(0))
        print("Training complete.")

    # --- amortized NPE posterior: sample q(theta|x_o) directly, no inference step ---
    idx = int(eval_cfg["observation_idx"])
    obs, _ = task.get_reference(idx)
    true_param = np.asarray(task.get_true_parameters(idx)).reshape(-1)

    nsamples = int(eval_cfg.get("nsamples_posterior", 20000))
    print(f"Sampling amortized NPE posterior (nsamples={nsamples})...")
    samples = pipeline.sample(jax.random.PRNGKey(0), obs, nsamples=nsamples,
                              use_ema=True)  # (nsamples, dim_theta, 1)

    # --- corner plot of the posterior samples with the true value marked ---
    plot_marginals(np.asarray(samples[..., 0]), plot_levels=False, backend="seaborn",
                   gridsize=50, true_param=true_param)
    out_path = os.path.join(img_dir, f"posterior_marginals_obs{idx}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close("all")
    print(f"Saved posterior marginals to {out_path}")

    # --- C2ST: score the amortized posterior against the reference posterior ---
    strategy = config.get("strategy", {})
    method = strategy.get("method", "npe")
    model = strategy.get("model", "tarflow")
    experiment_id = train_cfg.get("experiment_id", 1)

    c2st_dir = os.path.join(exp_dir, "c2st_results")
    os.makedirs(c2st_dir, exist_ok=True)

    print(f"Running C2ST over {task.num_observations} observations...")
    c2st_results = compute_c2st(pipeline, task)
    for tag in ("raw", "ema"):
        accs = c2st_results[tag]
        label = "EMA" if tag == "ema" else "raw"
        print(f"Average C2ST accuracy [{label}]: "
              f"{np.mean(accs):.4f} +- {np.std(accs):.4f}")
        path = save_c2st_results(c2st_dir, accs, tag, experiment_id, method, model)
        print(f"Saved C2ST results to {path}")
    print("C2ST complete.")


if __name__ == "__main__":
    main()
