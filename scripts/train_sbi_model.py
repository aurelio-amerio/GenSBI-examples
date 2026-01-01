# this uses EMA for the model weights
# %%

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.runtime_version")

import os

if __name__ != "__main__":
    os.environ["JAX_PLATFORMS"] = "cpu"
else:
    os.environ["JAX_PLATFORMS"] = "cuda"  # change to 'cpu' if no GPU is available

# Set JAX backend (use 'cuda' for GPU, 'cpu' otherwise)
# os.environ["JAX_PLATFORMS"] = "cuda"

import argparse
import yaml
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from gensbi_examples.tasks import get_task
from gensbi_examples.c2st import c2st

from gensbi.models import SimformerParams, Flux1JointParams, Flux1Params
from gensbi.recipes import (
    SimformerFlowPipeline,
    SimformerDiffusionPipeline,
    Flux1JointFlowPipeline,
    Flux1JointDiffusionPipeline,
    Flux1FlowPipeline,
    Flux1DiffusionPipeline,
)

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
        kind="joint"
    elif model_type == "flux1joint" and method == "diffusion":
        PipelineClass = Flux1JointDiffusionPipeline
        kind="joint"
    elif model_type == "flux" and method == "flow":
        PipelineClass = Flux1FlowPipeline
        kind="conditional"
    elif model_type == "flux" and method == "diffusion":
        PipelineClass = Flux1DiffusionPipeline
        kind="conditional"
    else:
        raise ValueError(
            f"Invalid combination of model_type {model_type} and method {method}"
        )
    
    # Task and dataset setup
    task = get_task(task_name, kind=kind)


    # Training parameters
    train_params = config.get("training", {})
    multistep = train_params.get("multistep", 1)
    experiment_id = train_params.get("experiment_id", 1)
    restore_model = train_params.get("restore_model", False)
    train_model = train_params.get("train_model", True)
    batch_size = train_params.get("batch_size", 4096)
    early_stopping = train_params.get("early_stopping", True)
    nsteps = train_params.get("nsteps", 30000) * multistep
    val_every = train_params.get("val_every", 100) * multistep


    # Set checkpoint directory (new structure)
    checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
    checkpoint_dir_ema = os.path.join(checkpoint_dir, "ema")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(checkpoint_dir_ema, exist_ok=True)


    # Optimizer parameters
    opt_params = config.get("optimizer", {})
    PATIENCE = opt_params.get("patience", 10)
    COOLDOWN = opt_params.get("cooldown", 2)
    FACTOR = opt_params.get("factor", 0.5)
    ACCUMULATION_SIZE = opt_params.get("accumulation_size", 100) * multistep
    RTOL = opt_params.get("rtol", 1e-4)
    MAX_LR = opt_params.get("max_lr", 1e-3)
    MIN_LR = opt_params.get("min_lr", 0.0)
    MIN_SCALE = MIN_LR / MAX_LR if MAX_LR > 0 else 0.0

    ema_decay = opt_params.get("ema_decay", 0.99)

    train_dataset = task.get_train_dataset(batch_size)
    val_dataset = task.get_val_dataset(512) # we are using the mean loss, so batch size does not matter
    # dataset_iter = iter(train_dataset)
    # val_dataset_iter = iter(val_dataset)

    dim_obs = task.dim_obs
    dim_cond = task.dim_cond
    dim_joint = task.dim_joint


    # Model parameters from config
    model_params = config.get("model", {})

    if model_type == "simformer":
        params = SimformerParams(
            rngs=nnx.Rngs(0),
            in_channels=model_params.get("in_channels", 1),
            dim_value=model_params.get("dim_value", 40),
            dim_id=model_params.get("dim_id", 40),
            dim_condition=model_params.get("dim_condition", 10),
            dim_joint=dim_joint,
            fourier_features=model_params.get("fourier_features", 128),
            num_heads=model_params.get("num_heads", 6),
            num_layers=model_params.get("num_layers", 8),
            widening_factor=model_params.get("widening_factor", 3),
            qkv_features=model_params.get("qkv_features", 90),
            num_hidden_layers=model_params.get("num_hidden_layers", 1),
        )
    elif model_type == "flux1joint":
        theta = model_params.get("theta", -1)
        if theta == -1:
            theta = 5 * (dim_obs + dim_cond)
        params = Flux1JointParams(
            in_channels=model_params.get("in_channels", 1),
            vec_in_dim=model_params.get("vec_in_dim", None),
            mlp_ratio=model_params.get("mlp_ratio", 4),
            num_heads=model_params.get("num_heads", 4),
            depth_single_blocks=model_params.get("depth_single_blocks", 16),
            axes_dim=model_params.get("axes_dim", [10]),
            condition_dim=model_params.get("condition_dim", [4]),
            qkv_bias=model_params.get("qkv_bias", True),
            dim_joint=dim_joint,
            theta=theta,
            rngs=nnx.Rngs(default=42),
            param_dtype=getattr(jnp, model_params.get("param_dtype", "float32")),
        )
    elif model_type == "flux":
        theta = model_params.get("theta", -1)
        if theta == -1:
            theta = 4 * (dim_obs + dim_cond)

        params = Flux1Params(
            in_channels=model_params.get("in_channels", 1),
            vec_in_dim=model_params.get("vec_in_dim", None),
            context_in_dim=model_params.get("context_in_dim", 1),
            mlp_ratio=model_params.get("mlp_ratio", 4),
            num_heads=model_params.get("num_heads", 4),
            depth=model_params.get("depth", 8),
            depth_single_blocks=model_params.get("depth_single_blocks", 16),
            axes_dim=model_params.get("axes_dim", [10]),
            qkv_bias=model_params.get("qkv_bias", True),
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            theta=theta,
            rngs=nnx.Rngs(default=42),
            param_dtype=getattr(jnp, model_params.get("param_dtype", "float32")),
        )

    


    training_config = PipelineClass._get_default_training_config()
    # overwrite the defaults with the config file values
    training_config["num_steps"] = nsteps
    training_config["ema_decay"] = ema_decay
    training_config["patience"] = PATIENCE
    training_config["cooldown"] = COOLDOWN
    training_config["factor"] = FACTOR
    training_config["accumulation_size"] = ACCUMULATION_SIZE
    training_config["rtol"] = RTOL
    training_config["max_lr"] = MAX_LR
    training_config["min_lr"] = MIN_LR
    training_config["min_scale"] = MIN_SCALE
    training_config["val_every"] = val_every
    training_config["early_stopping"] = early_stopping
    training_config["experiment_id"] = experiment_id
    training_config["multistep"] = multistep
    training_config["checkpoint_dir"] = checkpoint_dir


    pipeline = PipelineClass(
        train_dataset,
        val_dataset,
        dim_obs=dim_obs,
        dim_cond=dim_cond,
        params=params,
        training_config=training_config,
    )

    # current training config


    if restore_model:
        print("Restoring model from checkpoint...")
        pipeline.restore_model()

    if train_model:
        print("Starting training...")
        pipeline.train(nnx.Rngs(0))
        print("Training complete.")

    # --------- C2ST TEST ---------


    def get_samples(idx, nsamples=10_000, use_ema=False, key=None):
        observation, reference_samples = task.get_reference(idx)
        true_param = jnp.array(task.get_true_parameters(idx))

        if key is None:
            key = jax.random.PRNGKey(42)

        samples = pipeline.sample(key, observation, nsamples, use_ema=use_ema)
        return samples, true_param, reference_samples


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
        c2st_accuracy = c2st(reference_samples, samples[...,0])
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
        c2st_accuracy = c2st(reference_samples, samples[...,0])
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
    results_data = {"mean_accuracy": float(np.mean(c2st_accuracies_ema)), "std_dev": float(np.std(c2st_accuracies_ema))}

    markdown = create_markdown_content(config_data, results_data)

    # save the model card

    with open("README.md", "w") as f:
        f.write(markdown)   

    print("Model card generated as README.md")

if __name__ == "__main__":
    main()