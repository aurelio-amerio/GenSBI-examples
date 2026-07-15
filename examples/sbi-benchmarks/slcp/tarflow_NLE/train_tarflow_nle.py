"""SLCP NLE example: train a conditional TarFlow likelihood q(x|theta) and sample the
(complex, multimodal) posterior with tempered SMC.

Run (on a GPU node with HF access, via the gensbi conda env):
    python train_tarflow_nle.py --config config/config_tarflow_nle.yaml

NLE convention (mirror of the NPE script): obs = x, cond = theta, so the flow
models q(x | theta). For SLCP, x is the 8-D observation (4 i.i.d. 2-D points) and
theta is the 5-D parameter. The posterior is recovered at inference time by combining
the learned likelihood with the task prior (uniform on [-3, 3]^5) via gensbi
NLEPosterior. SLCP posteriors are multimodal -- the scale parameters enter the
likelihood squared and the correlation through tanh, producing sign-symmetric modes --
so we sample with TemperedSMC (a particle cloud walked from prior to posterior) instead
of a single MCMC/MCLMC chain, which would capture only one mode. TransformerFlow is a
transformer autoregressive normalizing flow (adapted from apple/ml-tarflow); the
pipeline is flow-agnostic, so this mirrors the MAF NLE example verbatim apart from the
model factory and its config section. Helpers are module-level and import-safe so they
can be smoke-tested on CPU.
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
from gensbi.inference import NLEPosterior, TemperedSMC, NestedSampler
from gensbi.diagnostics.metrics import c2st
from gensbi.utils.plotting import plot_marginals
from sbibm_jax.data import TaskDataset
from sbibm_jax.tasks import get_task


def build_flow(rngs, dim_obs, dim_cond, model_cfg):
    """Build the TarFlow (transformer autoregressive flow) stack from the model config.

    TarFlowParams uses the head_dim/num_heads convention and derives the total width
    channels = head_dim * num_heads.

    For NLE, dim_obs == task.dim_x (autoregressive target) and dim_cond == task.dim_theta.

    The conditioner is selected by ``model.cond`` (default ``"bias"``): ``"bias"`` embeds
    theta as a single per-token additive bias (AdditiveBiasConditioner), while ``"vector"``
    embeds each theta coordinate as its own prefix token (VectorConditioner) -- richer
    conditioning at the cost of M = dim_cond extra tokens. The pipeline already feeds the
    condition as (B, dim_cond, cond_channels), so both conditioners are drop-in; for the
    tabular SLCP theta, cond_channels stays 1.
    """
    head_dim = int(model_cfg.get("head_dim", 16))
    num_heads = int(model_cfg.get("num_heads", 4))
    return TarFlow(TarFlowParams(
        rngs=rngs,
        dim=dim_obs,
        cond_dim=dim_cond,
        cond=str(model_cfg.get("cond", "bias")),
        cond_channels=int(model_cfg.get("cond_channels", 1)),
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
    """Pipeline defaults overlaid with YAML optimizer+training, with ckpt dir set.

    ConditionalFlowPipeline reads training_config keys eagerly in __init__, so it must
    be complete. Extra keys are harmless -- the pipeline only reads the keys it knows.
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

    NLE analog of the NPE script's load_theta_stats: x is the autoregressive target
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
    returns the in-support constant *everywhere* and does NOT bound the NLE potential --
    the sampler then wanders outside the prior support and the posterior is wrong.
    Re-enabling validation makes log_prob = -inf outside the support, which both bounds
    the SMC tempering target and confines any inner MCMC/MCLMC moves to the box. For
    SLCP this box is the uniform prior on [-3, 3]^5.
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
    """Tempered SMC sampler for the complex, multimodal SLCP posterior.

    SMC walks a particle cloud along p(theta) * q(x_o|theta)^beta for beta: 0 -> 1,
    populating every mode of the posterior (a single MCMC/MCLMC chain captures only
    one). num_particles is the number of posterior samples returned. The inner
    rejuvenation kernel defaults to adjusted MCLMC ("mclmc"); "nuts" is the fallback.
    """
    return TemperedSMC(
        num_particles=int(sampler_cfg.get("num_particles", 20000)),
        target_ess=float(sampler_cfg.get("target_ess", 0.9)),
        num_mcmc_steps=int(sampler_cfg.get("num_mcmc_steps", 10)),
        inner_kernel=str(sampler_cfg.get("inner_kernel", "mclmc")),
        inner_step_size=float(sampler_cfg.get("inner_step_size", 0.1)),
        inner_num_integration_steps=int(sampler_cfg.get("inner_num_integration_steps", 5)),
    )


def build_nested_sampler(ns_cfg):
    """Nested slice sampler for the complex, multimodal SLCP posterior.

    An alternative to tempered SMC: blackjax nested slice sampling walks live points
    from the prior inward, handling multimodal posteriors without tempering and also
    returning a log-evidence estimate. num_samples is the number of equal-weight
    posterior draws returned (matched by default to the reference-posterior count so the
    c2st classes stay balanced). num_delete and num_inner_steps resolve from num_live and
    the target dim when left unset.
    """
    kwargs = dict(
        num_live=int(ns_cfg.get("num_live", 500)),
        num_samples=int(ns_cfg.get("num_samples", 10000)),
        dlogz=float(ns_cfg.get("dlogz", -3.0)),
        max_iterations=int(ns_cfg.get("max_iterations", 100_000)),
    )
    if ns_cfg.get("num_delete") is not None:
        kwargs["num_delete"] = int(ns_cfg["num_delete"])
    if ns_cfg.get("num_inner_steps") is not None:
        kwargs["num_inner_steps"] = int(ns_cfg["num_inner_steps"])
    return NestedSampler(**kwargs)


def compute_c2st(pipeline, task, prior, make_sampler, key=None):
    """C2ST accuracy per observation for the raw and EMA NLE posteriors.

    For each of the task's reference observations, rebuild the NLE posterior (likelihood
    flow + prior) and draw posterior samples with the sampler built by ``make_sampler``
    (a zero-arg factory returning a fresh sampler -- tempered SMC or nested sampling),
    then score them against the reference posterior samples with a classifier two-sample
    test. Accuracy ~0.5 means the recovered posterior is indistinguishable from the
    reference; ~1.0 means they are easily told apart. Each posterior sample is subsampled
    to the reference count so the two c2st classes stay balanced (an imbalance would shift
    the chance accuracy away from 0.5). The sampler is run once per observation for both
    the raw and EMA flows. Returns ``(results, evidence)``, each a
    ``{"raw": [...], "ema": [...]}`` of per-observation C2ST accuracies and sampler
    log-evidences respectively.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    results = {"raw": [], "ema": []}
    evidence = {"raw": [], "ema": []}
    for tag, use_ema in (("raw", False), ("ema", True)):
        posterior = build_posterior(pipeline, prior, use_ema=use_ema)
        for idx in range(1, task.num_observations + 1):
            obs, reference_samples = task.get_reference(idx)
            sampler = make_sampler()
            key, subkey = jax.random.split(key)
            samples, info = posterior.sample(subkey, obs, sampler=sampler,
                                             return_info=True)
            post = samples[..., 0]
            n = min(reference_samples.shape[0], post.shape[0])
            acc = float(c2st(reference_samples[:n], post[:n]))
            results[tag].append(acc)
            # Both samplers report a log-evidence (SmcInfo / NestedSamplerInfo); print it
            # alongside the C2ST. For NS also surface ess + unique-draw count -- the direct
            # read on whether equal-weight resampling is duplicating draws (ess << num_samples
            # inflates C2ST regardless of how well the modes are covered).
            logZ = float(getattr(info, "log_evidence", float("nan")))
            evidence[tag].append(logZ)
            extra = f" logZ={logZ:.3f}"
            if hasattr(info, "ess"):
                n_uniq = int(np.unique(np.asarray(post), axis=0).shape[0])
                extra += (f" (err={info.log_evidence_err:.3f}) ess={info.ess:.0f}"
                          f" unique={n_uniq}/{post.shape[0]}")
            print(f"C2ST [{tag}] observation={idx}: {acc:.4f}{extra}")
    return results, evidence


def save_c2st_results(c2st_dir, accuracies, tag, experiment_id, method, model,
                      sampler_name):
    """Write per-observation c2st accuracies and their mean +- std to a txt file.

    Mirrors scripts/train_sbi_model.py. tag is "raw" or "ema" and selects both the
    filename suffix and the in-file labels. sampler_name ("MCMC" or "NS") is appended to
    the filename so results from the tempered-SMC and nested-sampling posteriors do not
    collide. Returns the path written.
    """
    suffix = "_ema" if tag == "ema" else ""
    label = "EMA " if tag == "ema" else ""
    path = os.path.join(
        c2st_dir,
        f"c2st_results{suffix}_{experiment_id}_{method}_{model}_{sampler_name}.txt",
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
    default_config = os.path.join(here, "config", "config_tarflow_nle.yaml")
    parser = argparse.ArgumentParser(description="SLCP NLE (TarFlow) training/eval")
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

    # --- NLE posterior via tempered SMC for one observation ---
    idx = int(eval_cfg["observation_idx"])
    obs, _ = task.get_reference(idx)
    true_param = np.asarray(task.get_true_parameters(idx)).reshape(-1)

    prior = build_prior(task_name)
    posterior = build_posterior(pipeline, prior, use_ema=True)

    def _plot_marginals(post_samples, sampler_tag):
        """Corner plot of one sampler's posterior draws with the true value marked."""
        plot_marginals(np.asarray(post_samples[..., 0]), plot_levels=False,
                       backend="seaborn", gridsize=50, true_param=true_param)
        out_path = os.path.join(img_dir,
                                f"posterior_marginals_obs{idx}_{sampler_tag}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close("all")
        print(f"Saved {sampler_tag} posterior marginals to {out_path}")

    # --- tempered SMC ("MCMC") single-observation preview: evidence + corner plot ---
    sampler = build_sampler(sampler_cfg)
    print(f"Sampling multimodal posterior with tempered SMC "
          f"(num_particles={sampler.num_particles}, inner_kernel={sampler.inner_kernel})...")
    smc_samples, smc_info = posterior.sample(jax.random.PRNGKey(0), obs,
                                             sampler=sampler, return_info=True)
    print(f"SMC done: {smc_info.num_temperature_steps} temperature steps, "
          f"log_evidence={smc_info.log_evidence:.3f}")
    _plot_marginals(smc_samples, "MCMC")

    # --- nested sampling single-observation preview: evidence + ess/unique + corner plot.
    # A fast read (before the slow full C2ST) on whether NS recovers the modes and how many
    # independent draws it actually supports: ess << num_samples means the equal-weight
    # resampling is duplicating draws (which inflates C2ST regardless of mode coverage). ---
    ns_sampler = build_nested_sampler(ns_cfg)
    print(f"Sampling with nested sampling (num_live={ns_sampler.num_live}, "
          f"num_samples={ns_sampler.num_samples}, dlogz={ns_sampler.dlogz})...")
    ns_samples, ns_info = posterior.sample(jax.random.PRNGKey(0), obs,
                                           sampler=ns_sampler, return_info=True)
    ns_post = np.asarray(ns_samples[..., 0])
    n_uniq = int(np.unique(ns_post, axis=0).shape[0])
    print(f"NS done: log_evidence={ns_info.log_evidence:.3f} +- {ns_info.log_evidence_err:.3f}, "
          f"ess={ns_info.ess:.0f}, num_dead={ns_info.num_dead}, "
          f"unique={n_uniq}/{ns_post.shape[0]}")
    _plot_marginals(ns_samples, "NS")

    # --- C2ST: score the SMC posterior against the reference posterior ---
    strategy = config.get("strategy", {})
    method = strategy.get("method", "nle")
    model = strategy.get("model", "tarflow")
    experiment_id = train_cfg.get("experiment_id", 1)

    c2st_dir = os.path.join(exp_dir, "c2st_results")
    os.makedirs(c2st_dir, exist_ok=True)

    # Score both posterior samplers. NS is run first (it is the one under scrutiny, so
    # surface its C2ST before the slower SMC pass), then the tempered-SMC ("MCMC").
    # Results are written to separate files (see save_c2st_results).
    methods = (
        ("NS", lambda: build_nested_sampler(ns_cfg)),
        ("MCMC", lambda: build_sampler(sampler_cfg)),
    )
    print(f"Running C2ST over {task.num_observations} observations "
          f"(x2 for raw+EMA, x2 for NS+MCMC; this is slow)...")
    for sampler_name, make_sampler in methods:
        print(f"--- C2ST with {sampler_name} sampler ---")
        c2st_results, evidence = compute_c2st(pipeline, task, prior, make_sampler)
        for tag in ("raw", "ema"):
            accs = c2st_results[tag]
            label = "EMA" if tag == "ema" else "raw"
            print(f"Average C2ST accuracy [{label}, {sampler_name}]: "
                  f"{np.mean(accs):.4f} +- {np.std(accs):.4f}"
                  f"  |  mean logZ={np.nanmean(evidence[tag]):.3f}")
            path = save_c2st_results(c2st_dir, accs, tag, experiment_id, method,
                                     model, sampler_name)
            print(f"Saved C2ST results to {path}")
    print("C2ST complete.")


if __name__ == "__main__":
    main()
