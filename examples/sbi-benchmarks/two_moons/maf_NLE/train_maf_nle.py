"""Two-moons NLE example: train a conditional MAF likelihood q(x|theta) and
sample the (multimodal) posterior two ways -- tempered SMC and nested sampling.

Run (on a GPU node with HF access, via the gensbi conda env):
    python train_maf_nle.py --config config/config_maf_nle.yaml

NLE convention (mirror of the NPE script): obs = x, cond = theta, so the flow
models q(x | theta). The posterior is recovered at inference time by combining the
learned likelihood with the task prior via gensbi NLEPosterior. Two-moons posteriors
are multimodal, so we sample with TemperedSMC (a particle cloud walked from prior to
posterior) instead of a single MCMC/MCLMC chain, which would capture only one mode.
As a cross-check we also sample with NestedSampler (blackjax nested slice sampling),
which handles the multimodal posterior without tempering and returns a log-evidence
estimate; the two posteriors are saved to separate corner plots (the PNG filenames
carry an "MCMC" or "NS" suffix). Helpers are module-level and import-safe so they can
be smoke-tested on CPU.
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

from gensbi.normalizing_flows import Affine, RQSpline
from gensbi.models import MAFlow, MAFlowParams
from gensbi.recipes import ConditionalFlowPipeline
from gensbi.inference import NLEPosterior, TemperedSMC, NestedSampler
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
    return MAFlow(MAFlowParams(
        rngs=rngs,
        dim=dim_obs,
        cond_dim=dim_cond,
        n_layers=int(model_cfg.get("n_layers", 8)),
        transformer=transformer,
        nn_width=int(model_cfg.get("nn_width", 64)),
        nn_depth=int(model_cfg.get("nn_depth", 2)),
        permutation=str(model_cfg.get("permutation", "reverse")),
        standardize=bool(model_cfg.get("standardize", True)),
        zero_init=bool(model_cfg.get("zero_init", True)),
    ))


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
    the sampler then wanders outside the prior support and the posterior is wrong.
    Re-enabling validation makes log_prob = -inf outside the support, which both bounds
    the SMC tempering target and confines any inner MCMC/MCLMC moves to the box.
    """
    prior = get_task(task_name).get_prior_dist()
    prior._validate_args = validate_args
    if hasattr(prior, "base_dist"):          # Independent wraps a base distribution
        prior.base_dist._validate_args = validate_args
    return prior


def build_posterior(pipeline, prior, use_ema=True):
    """Wrap the trained likelihood flow + validated prior in an NLE posterior.

    The sampler (and its config) is supplied separately at sample() time; NLEPosterior
    only holds the flow + prior and builds the log-densities for a given observation.
    """
    flow = pipeline.ema_model if use_ema else pipeline.model
    return NLEPosterior(flow, prior)


def build_sampler(sampler_cfg):
    """Tempered SMC sampler for the multimodal two-moons posterior.

    SMC walks a particle cloud along p(theta) * q(x_o|theta)^beta for beta: 0 -> 1,
    populating every mode of the posterior (a single MCMC/MCLMC chain captures only
    one). num_particles is the number of posterior samples returned. The inner
    rejuvenation kernel defaults to adjusted MCLMC ("mclmc"); "nuts" is the fallback.
    """
    return TemperedSMC(
        num_particles=int(sampler_cfg.get("num_particles", 20000)),
        target_ess=float(sampler_cfg.get("target_ess", 0.5)),
        num_mcmc_steps=int(sampler_cfg.get("num_mcmc_steps", 10)),
        inner_kernel=str(sampler_cfg.get("inner_kernel", "mclmc")),
        inner_step_size=float(sampler_cfg.get("inner_step_size", 0.1)),
        inner_num_integration_steps=int(sampler_cfg.get("inner_num_integration_steps", 5)),
    )


def build_nested_sampler(ns_cfg):
    """Nested slice sampler for the multimodal two-moons posterior.

    An alternative to tempered SMC: blackjax nested slice sampling walks live points
    from the prior inward, handling the multimodal posterior without tempering and also
    returning a log-evidence estimate. num_samples is the number of equal-weight
    posterior draws returned. num_delete and num_inner_steps resolve from num_live and
    the target dim when left unset. num_rejuvenation_steps > 0 applies posterior-invariant
    slice moves per equal-weight draw after resampling, breaking the duplicate rows that
    with-replacement resampling produces when num_samples ~ run ESS.
    """
    kwargs = dict(
        num_live=int(ns_cfg.get("num_live", 2000)),
        num_samples=int(ns_cfg.get("num_samples", 20000)),
        dlogz=float(ns_cfg.get("dlogz", -5.0)),
        max_iterations=int(ns_cfg.get("max_iterations", 100_000)),
        num_rejuvenation_steps=int(ns_cfg.get("num_rejuvenation_steps", 10)),
    )
    if ns_cfg.get("num_delete") is not None:
        kwargs["num_delete"] = int(ns_cfg["num_delete"])
    if ns_cfg.get("num_inner_steps") is not None:
        kwargs["num_inner_steps"] = int(ns_cfg["num_inner_steps"])
    return NestedSampler(**kwargs)


def save_diagnostics(diag_dir, idx, lines):
    """Write the per-sampler diagnostics collected for one observation to a txt file.

    Persists the same numbers printed to stdout -- most importantly each sampler's
    log-evidence (SMC and NS estimate the same log Z, so the two are a mutual cross-check)
    -- so results survive past the console. Returns the path written.
    """
    path = os.path.join(diag_dir, f"diagnostics_obs{idx}.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


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
    diag_dir = os.path.join(exp_dir, "diagnostics")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(diag_dir, exist_ok=True)

    task_name = config["task_name"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    sampler_cfg = config["sampler"]
    ns_cfg = config.get("nested_sampler", {})
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

    # --- NLE posterior for one observation, sampled two ways (SMC + nested sampling) ---
    idx = int(eval_cfg["observation_idx"])
    obs, _ = task.get_reference(idx)
    true_param = np.asarray(task.get_true_parameters(idx)).reshape(-1)

    prior = build_prior(task_name)
    posterior = build_posterior(pipeline, prior, use_ema=True)

    def _plot_marginals(post_samples, sampler_tag, range_=None):
        """Corner plot of one sampler's posterior draws with the true value marked.

        The sampler tag ("MCMC" or "NS") is appended to the filename so the tempered-SMC
        and nested-sampling posteriors are saved to distinct PNGs.
        """
        plot_marginals(np.asarray(post_samples[..., 0]), plot_levels=False,
                       backend="seaborn", gridsize=50, true_param=true_param,
                       range=range_)
        out_path = os.path.join(img_dir,
                                f"posterior_marginals_obs{idx}_{sampler_tag}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close("all")
        print(f"Saved {sampler_tag} posterior marginals to {out_path}")

    # Diagnostic lines are both printed and collected here, then written to a txt file
    # at the end so the numbers (especially the log-evidence) survive past the console.
    diag_lines = [f"NLE posterior diagnostics for observation={idx}",
                  f"true_param: {np.array2string(true_param, precision=4)}"]

    def _report(line):
        print(line)
        diag_lines.append(line)

    # --- tempered SMC ("MCMC"): evidence + corner plot ---
    sampler = build_sampler(sampler_cfg)
    print(f"Sampling multimodal posterior with tempered SMC "
          f"(num_particles={sampler.num_particles}, inner_kernel={sampler.inner_kernel})...")
    smc_samples, smc_info = posterior.sample(jax.random.PRNGKey(0), obs,
                                             sampler=sampler, return_info=True)  # (n, dim_theta, 1)
    _report(f"[MCMC] num_temperature_steps={smc_info.num_temperature_steps}, "
            f"log_evidence={smc_info.log_evidence:.4f}")
    _plot_marginals(smc_samples, "MCMC", range_=((-0.9, 0.4), (-0.4, 0.9)))

    # --- nested sampling ("NS"): evidence + ess/unique-draw count + corner plot.
    # ess << num_samples means the equal-weight resampling is duplicating draws; with
    # num_rejuvenation_steps > 0 the duplicates are broken and unique should be num_samples. ---
    ns_sampler = build_nested_sampler(ns_cfg)
    print(f"Sampling with nested sampling (num_live={ns_sampler.num_live}, "
          f"num_samples={ns_sampler.num_samples}, dlogz={ns_sampler.dlogz}, "
          f"num_rejuvenation_steps={ns_sampler.num_rejuvenation_steps})...")
    ns_samples, ns_info = posterior.sample(jax.random.PRNGKey(0), obs,
                                           sampler=ns_sampler, return_info=True)
    ns_post = np.asarray(ns_samples[..., 0])
    n_uniq = int(np.unique(ns_post, axis=0).shape[0])
    _report(f"[NS] log_evidence={ns_info.log_evidence:.4f} +- {ns_info.log_evidence_err:.4f}, "
            f"ess={ns_info.ess:.0f}, num_dead={ns_info.num_dead}, "
            f"unique={n_uniq}/{ns_post.shape[0]}")
    _plot_marginals(ns_samples, "NS", range_=((-0.9, -0.4), (0.4, 0.9)))

    diag_path = save_diagnostics(diag_dir, idx, diag_lines)
    print(f"Saved diagnostics to {diag_path}")


if __name__ == "__main__":
    main()
