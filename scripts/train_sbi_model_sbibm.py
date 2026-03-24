# this uses EMA for the model weights
# %%

import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, module="google.protobuf.runtime_version"
)

import os

if __name__ != "__main__":
    os.environ["JAX_PLATFORMS"] = "cpu"
else:
    pass  # use the default platform
    # os.environ["JAX_PLATFORMS"] = "cuda"  # change to 'cpu' if no GPU is available

# Set JAX backend (use 'cuda' for GPU, 'cpu' otherwise)
# os.environ["JAX_PLATFORMS"] = "cuda"

import argparse
import yaml
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from gensbi_examples.tasks import get_task
from gensbi.diagnostics.metrics import c2st

from gensbi.diagnostics import run_tarp, plot_tarp
from gensbi.diagnostics import run_sbc, sbc_rank_plot
from gensbi.diagnostics import LC2ST, plot_lc2st
from gensbi.diagnostics.marginal_coverage import (
    compute_marginal_coverage,
    plot_marginal_coverage,
)

from gensbi.models import Flux1JointParams, Flux1Params
from gensbi.recipes import (
    Flux1JointFlowPipeline,
    Flux1JointDiffusionPipeline,
    Flux1JointSMPipeline,
    Flux1FlowPipeline,
    Flux1DiffusionPipeline,
    Flux1SMPipeline,
    SimformerFlowPipeline,
    SimformerDiffusionPipeline,
    SimformerSMPipeline,
)


from gensbi.utils.plotting import plot_marginals
import matplotlib.pyplot as plt


# %%
def main():
    # --- Global override: set to True to force training, False to skip & restore, None to use config ---
    TRAINING_OVERRIDE = None

    # Argument parser for config file
    parser = argparse.ArgumentParser(description="Simformer Training Script")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g. .../two_moons/flow_simformer/config/config_flow_simformer.yaml)",
    )
    parser.add_argument(
        "--dsize",
        type=int,
        default=100_000,
        dest="dataset_size",
        help="Size of the training dataset (default: 100000)",
    )
    args, _ = parser.parse_known_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # methodology
    strategy = config.get("strategy", {})
    method = strategy.get("method")
    model_type = strategy.get("model")

    # training
    train_params = config.get("training", {})
    batch_size = train_params.get("batch_size", 256)
    restore_model = train_params.get("restore_model", False)
    train_model = train_params.get("train_model", True)

    # Override from variable if set
    if TRAINING_OVERRIDE is not None:
        train_model = TRAINING_OVERRIDE
        restore_model = not TRAINING_OVERRIDE
    experiment_id = train_params.get("experiment_id", 1)
    dataset_size = args.dataset_size

    # task_name and variant
    task_name = config.get("task_name")
    variant = f"{method}_{model_type}"

    # Set experiment directory to new structure
    experiment_directory = (
        f"examples/sbi-benchmarks/{task_name}/{variant}/sbibm/{dataset_size}"
    )
    os.makedirs(experiment_directory, exist_ok=True)
    os.chdir(experiment_directory)

    assert model_type in [
        "simformer",
        "flux1joint",
        "flux",
    ], f"Model type must be 'simformer' or 'flux', got {model_type}."
    assert method in [
        "flow",
        "diffusion",
        "score_matching",
    ], f"Method must be 'flow', 'diffusion', or 'score_matching', got {method}."

    # define the appropriate pipeline
    if model_type == "simformer" and method == "flow":
        PipelineClass = SimformerFlowPipeline
        kind = "joint"
    elif model_type == "simformer" and method == "diffusion":
        PipelineClass = SimformerDiffusionPipeline
        kind = "joint"
    elif model_type == "simformer" and method == "score_matching":
        PipelineClass = SimformerSMPipeline
        kind = "joint"
    elif model_type == "flux1joint" and method == "flow":
        PipelineClass = Flux1JointFlowPipeline
        kind = "joint"
    elif model_type == "flux1joint" and method == "diffusion":
        PipelineClass = Flux1JointDiffusionPipeline
        kind = "joint"
    elif model_type == "flux1joint" and method == "score_matching":
        PipelineClass = Flux1JointSMPipeline
        kind = "joint"
    elif model_type == "flux" and method == "flow":
        PipelineClass = Flux1FlowPipeline
        kind = "conditional"
    elif model_type == "flux" and method == "diffusion":
        PipelineClass = Flux1DiffusionPipeline
        kind = "conditional"
    elif model_type == "flux" and method == "score_matching":
        PipelineClass = Flux1SMPipeline
        kind = "conditional"
    else:
        raise ValueError(
            f"Invalid combination of model_type {model_type} and method {method}"
        )

    # Task and dataset setup
    task = get_task(
        task_name, kind=kind, normalize_data=True, use_prefetching=True, max_workers=2
    )

    # Set checkpoint directory (new structure)
    checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")

    train_dataset = task.get_train_dataset(batch_size, nsamples=dataset_size)
    val_dataset = task.get_val_dataset(
        512
    )  # we are using the mean loss, so batch size does not matter

    dim_obs = task.dim_obs
    dim_cond = task.dim_cond
    dim_joint = task.dim_joint

    # Model parameters from config
    model_params = config.get("model", {})

    kwargs = {}
    if method == "score_matching":
        kwargs["sde_type"] = "VE"

    pipeline = PipelineClass.init_pipeline_from_config(
        train_dataset,
        val_dataset,
        dim_obs,
        dim_cond,
        config_path=args.config,
        checkpoint_dir=checkpoint_dir,
        **kwargs,
    )

    # current training config

    if restore_model:
        print("Restoring model from checkpoint...")
        pipeline.restore_model()

    if train_model:
        print("Starting training...")
        pipeline.train(nnx.Rngs(0))
        print("Training complete.")

    # --------- Define sampling function ----------
    def get_samples(idx, nsamples=10_000, use_ema=True, key=None):
        observation, reference_samples = task.get_reference(idx)
        true_param = jnp.array(task.get_true_parameters(idx))

        if key is None:
            key = jax.random.PRNGKey(42)

        # Reshape to (1, D, 1) so normalization broadcasts correctly
        obs_for_model = jnp.array(observation).reshape(1, -1, 1)
        # Normalize observation before feeding to the model
        obs_for_model = task.normalize_cond(obs_for_model)

        sampler_kwargs = {}
        if method == "diffusion":
            sampler_kwargs["solver_params"] = {"S_churn": 30, "S_noise": 1.0}

        samples = pipeline.sample(key, obs_for_model, nsamples, use_ema=use_ema, **sampler_kwargs)

        # Unnormalize model output back to physical space
        samples = task.unnormalize_obs(samples)

        return samples, true_param, reference_samples

    # make plots
    img_dir = os.path.join(os.getcwd(), "imgs")
    os.makedirs(img_dir, exist_ok=True)

    # --------- Sampling ----------
    samples, true_param, _ = get_samples(8, nsamples=100_000, use_ema=True)

    plot_marginals(samples[..., 0], plot_levels=False, backend="seaborn", gridsize=50)
    plt.savefig(
        f"{img_dir}/marginals_ema_{experiment_id}.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

    # --------- C2ST TEST ---------

    # Run C2ST
    print("Running C2ST tests...")

    # Set c2st_results directory (new structure)
    c2st_dir = os.path.join(os.getcwd(), "c2st_results")
    os.makedirs(c2st_dir, exist_ok=True)

    c2st_accuracies = []
    for idx in range(1, 11):
        samples, true_param, reference_samples = get_samples(
            idx, nsamples=10_000, use_ema=False
        )
        c2st_accuracy = c2st(reference_samples, samples[..., 0])
        c2st_accuracies.append(c2st_accuracy)
        print(f"C2ST accuracy for observation={idx}: {c2st_accuracy:.4f}\n")

    print(
        f"Average C2ST accuracy: {np.mean(c2st_accuracies):.4f} +- {np.std(c2st_accuracies):.4f}"
    )
    # Save C2ST results in a txt file
    c2st_results_file = (
        f"{c2st_dir}/c2st_results_{experiment_id}_{method}_{model_type}.txt"
    )
    with open(c2st_results_file, "w") as f:
        for idx, accuracy in enumerate(c2st_accuracies, start=1):
            f.write(f"C2ST accuracy for observation={idx}: {accuracy:.4f}\n")

        # print mean and std accuracy
        f.write(
            f"Average C2ST accuracy: {np.mean(c2st_accuracies):.4f} +- {np.std(c2st_accuracies):.4f}\n"
        )

    # repeat for the ema model
    c2st_accuracies_ema = []
    for idx in range(1, 11):
        samples, true_param, reference_samples = get_samples(
            idx, nsamples=10_000, use_ema=True
        )
        c2st_accuracy = c2st(reference_samples, samples[..., 0])
        c2st_accuracies_ema.append(c2st_accuracy)
        print(f"C2ST accuracy EMA for observation={idx}: {c2st_accuracy:.4f}\n")
    print(
        f"Average C2ST accuracy EMA: {np.mean(c2st_accuracies_ema):.4f} +- {np.std(c2st_accuracies_ema):.4f}"
    )
    # Save C2ST results in a txt file
    c2st_results_file_ema = (
        f"{c2st_dir}/c2st_results_ema_{experiment_id}_{method}_{model_type}.txt"
    )
    with open(c2st_results_file_ema, "w") as f:
        for idx, accuracy in enumerate(c2st_accuracies_ema, start=1):
            f.write(f"C2ST accuracy EMA for observation={idx}: {accuracy:.4f}\n")

        # print mean and std accuracy
        f.write(
            f"Average C2ST accuracy EMA: {np.mean(c2st_accuracies_ema):.4f} +- {np.std(c2st_accuracies_ema):.4f}\n"
        )
    print("C2ST tests complete.")


if __name__ == "__main__":
    main()
