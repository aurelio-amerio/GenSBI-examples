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

from gensbi.models import SimformerParams, Flux1JointParams, Flux1Params
from gensbi.recipes import (
    SimformerFlowPipeline,
    SimformerDiffusionPipeline,
    Flux1JointFlowPipeline,
    Flux1JointDiffusionPipeline,
    Flux1FlowPipeline,
    Flux1DiffusionPipeline,
)


from gensbi.utils.plotting import plot_marginals
import matplotlib.pyplot as plt


# %%
def main():
    # Argument parser for config file
    parser = argparse.ArgumentParser(description="Simformer Training Script")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g. .../two_moons/flow_simformer/config/config_flow_simformer.yaml)",
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
    batch_size = train_params.get("batch_size", 4096)
    restore_model = train_params.get("restore_model", False)
    train_model = train_params.get("train_model", True)
    experiment_id = train_params.get("experiment_id", 1)

    # task_name and variant
    task_name = config.get("task_name")
    variant = f"{method}_{model_type}"

    # Set experiment directory to new structure
    experiment_directory = f"examples/sbi-benchmarks/{task_name}/{variant}"
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
    ], f"Method must be 'flow' or 'diffusion', got {method}."

    # define the appropriate pipeline
    if model_type == "simformer" and method == "flow":
        PipelineClass = SimformerFlowPipeline
        kind = "joint"
    elif model_type == "simformer" and method == "diffusion":
        PipelineClass = SimformerDiffusionPipeline
        kind = "joint"
    elif model_type == "flux1joint" and method == "flow":
        PipelineClass = Flux1JointFlowPipeline
        kind = "joint"
    elif model_type == "flux1joint" and method == "diffusion":
        PipelineClass = Flux1JointDiffusionPipeline
        kind = "joint"
    elif model_type == "flux" and method == "flow":
        PipelineClass = Flux1FlowPipeline
        kind = "conditional"
    elif model_type == "flux" and method == "diffusion":
        PipelineClass = Flux1DiffusionPipeline
        kind = "conditional"
    else:
        raise ValueError(
            f"Invalid combination of model_type {model_type} and method {method}"
        )

    # Task and dataset setup
    task = get_task(task_name, kind=kind)

    # Set checkpoint directory (new structure)
    checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")

    train_dataset = task.get_train_dataset(batch_size)
    val_dataset = task.get_val_dataset(
        512
    )  # we are using the mean loss, so batch size does not matter

    dim_obs = task.dim_obs
    dim_cond = task.dim_cond
    dim_joint = task.dim_joint

    # Model parameters from config
    model_params = config.get("model", {})

    pipeline = PipelineClass.init_pipeline_from_config(
        train_dataset,
        val_dataset,
        dim_obs,
        dim_cond,
        config_path=args.config,
        checkpoint_dir=checkpoint_dir,
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

        samples = pipeline.sample(key, observation, nsamples, use_ema=use_ema)
        return samples, true_param, reference_samples

    # make plots
    img_dir = os.path.join(os.getcwd(), "imgs")
    os.makedirs(img_dir, exist_ok=True)

    # --------- Sampling ----------
    samples, true_param, _ = get_samples(8, nsamples=100_000, use_ema=True)

    plot_marginals(samples[..., 0], plot_levels=False, backend="seaborn", gridsize=50)
    plt.savefig(f"{img_dir}/marginals_ema_{experiment_id}.png", dpi=300, bbox_inches="tight")
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

    print("Generating model card...")
    # now we call the script to generate the model card
    from generate_model_card import create_markdown_content, parse_config, parse_results

    # parse config and results
    config_data = parse_config(args.config)
    results_data = {
        "mean_accuracy": float(np.mean(c2st_accuracies_ema)),
        "std_dev": float(np.std(c2st_accuracies_ema)),
    }

    markdown = create_markdown_content(config_data, results_data)

    # save the model card

    with open("README.md", "w") as f:
        f.write(markdown)

    print("Model card generated as README.md")

    print("Running TARP diagnostic...")

    data = task.dataset["test"].with_format("jax")[:500]
    xs = jnp.asarray(data["xs"][:], dtype=jnp.bfloat16)
    thetas = jnp.asarray(data["thetas"][:], dtype=jnp.bfloat16)

    num_posterior_samples = 1_000

    posterior_samples = pipeline.sample_batched(
        jax.random.PRNGKey(12345), xs, num_posterior_samples, use_ema=True
    )

    # reshape
    xs = xs.reshape((xs.shape[0], -1))
    thetas = thetas.reshape((thetas.shape[0], -1))
    posterior_samples = posterior_samples.reshape(
        (posterior_samples.shape[0], posterior_samples.shape[1], -1)
    )

    ecp, alpha = run_tarp(
        thetas,
        posterior_samples,
        references=None,  # will be calculated automatically.
    )

    plot_tarp(ecp, alpha)
    plt.savefig(f"{img_dir}/tarp_{experiment_id}.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("TARP diagnostic complete.")
    print("Running SBC diagnostic...")

    ranks, dap_samples = run_sbc(thetas, xs, posterior_samples)

    f, ax = sbc_rank_plot(ranks, num_posterior_samples, plot_type="hist", num_bins=20)
    plt.savefig(
        f"{img_dir}/sbc_{experiment_id}.png", dpi=100, bbox_inches="tight"
    )  # uncomment to save the figure
    plt.show()

    # LC2ST diagnostic
    data = task.dataset["test"].with_format("jax")[:10_00]
    xs_ = jnp.asarray(data["xs"][:], dtype=jnp.bfloat16)
    thetas_ = jnp.asarray(data["thetas"][:], dtype=jnp.bfloat16)

    num_posterior_samples = 1

    posterior_samples_ = pipeline.sample(
        jax.random.PRNGKey(42), x_o=xs_, nsamples=xs_.shape[0]
    )

    thetas = thetas_.reshape(thetas_.shape[0], -1)
    xs = xs_.reshape(xs_.shape[0], -1)
    posterior_samples = posterior_samples_.reshape(posterior_samples_.shape[0], -1)

    # Train the L-C2ST classifier.
    lc2st = LC2ST(
        thetas=thetas[:-1],
        xs=xs[:-1],
        posterior_samples=posterior_samples[:-1],
        classifier="mlp",
        num_ensemble=1,
    )

    _ = lc2st.train_under_null_hypothesis()
    _ = lc2st.train_on_observed_data()

    x_o = xs_[-1:]  # Take the last observation as observed data.
    theta_o = thetas_[-1:]  # True parameter for the observed data.

    post_samples_star = pipeline.sample(jax.random.PRNGKey(42), x_o, nsamples=10_000)

    x_o = x_o.reshape(1, -1)
    post_samples_star = np.array(
        post_samples_star.reshape(post_samples_star.shape[0], -1)
    )

    fig, ax = plot_lc2st(
        lc2st,
        post_samples_star,
        x_o,
    )
    plt.savefig(
        f"{img_dir}/lc2st_{experiment_id}.png", dpi=100, bbox_inches="tight"
    )  # uncomment to save the figure
    plt.show()


if __name__ == "__main__":
    main()
