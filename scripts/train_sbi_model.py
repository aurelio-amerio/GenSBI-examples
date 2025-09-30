# this uses EMA for the model weights
# %%

import os

# Set JAX backend (use 'cuda' for GPU, 'cpu' otherwise)
os.environ["JAX_PLATFORMS"] = "cuda"

import argparse
import yaml
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from gensbi_examples.tasks import get_task
from gensbi_examples.c2st import c2st

from gensbi.models import SimformerParams, FluxParams
from gensbi.recipes import (
    SimformerFlowPipeline,
    SimformerDiffusionPipeline,
    FluxFlowPipeline,
    FluxDiffusionPipeline,
)

# %%

# Argument parser for config file
parser = argparse.ArgumentParser(description="Simformer Training Script")
parser.add_argument(
    "--config",
    type=str,
    default="config_simformer.yaml",
    help="Path to YAML config file",
)
args, _ = parser.parse_known_args()

# Load config
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

# methodology
strategy = config.get("strategy", {})
method = strategy.get("method", "flow")
model_type = strategy.get("model", "simformer")

# Change working directory to experiment_directory from config
task_name = config.get("task_name", None)
experiment_directory = f"examples/sbi-benchmarks/{task_name}"

if experiment_directory is not None:
    os.chdir(experiment_directory)

# Task and dataset setup
task = get_task(task_name)

assert model_type in [
    "simformer",
    "flux",
], f"Model type must be 'simformer' or 'flux', got {model_type}."
assert method in [
    "flow",
    "diffusion",
], f"Method must be 'flow' or 'diffusion', got {method}."

# Training parameters
train_params = config.get("training", {})
experiment_id = train_params.get("experiment_id", 3)
restore_model = train_params.get("restore_model", False)
train_model = train_params.get("train_model", True)
batch_size = train_params.get("batch_size", 4096)
nsteps = train_params.get("nsteps", 30000)
multistep = train_params.get("multistep", 1)
early_stopping = train_params.get("early_stopping", True)
val_every = train_params.get("val_every", 100)

# Set checkpoint directory
notebook_path = os.getcwd()
checkpoint_dir = f"{notebook_path}/checkpoints/{task_name}_{method}_{model_type}"
checkpoint_dir_ema = (
    f"{notebook_path}/checkpoints/{task_name}_{method}_{model_type}/ema"
)
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(checkpoint_dir_ema, exist_ok=True)


# Optimizer parameters
opt_params = config.get("optimizer", {})
PATIENCE = opt_params.get("patience", 10)
COOLDOWN = opt_params.get("cooldown", 2)
FACTOR = opt_params.get("factor", 0.5)
ACCUMULATION_SIZE = opt_params.get("accumulation_size", 100)
RTOL = opt_params.get("rtol", 1e-4)
MAX_LR = opt_params.get("max_lr", 1e-3)
MIN_LR = opt_params.get("min_lr", 0.0)
MIN_SCALE = MIN_LR / MAX_LR if MAX_LR > 0 else 0.0

ema_decay = opt_params.get("ema_decay", 0.99)

train_dataset = task.get_train_dataset(batch_size)
val_dataset = task.get_val_dataset()
dataset_iter = iter(train_dataset)
val_dataset_iter = iter(val_dataset)


dim_theta = task.dim_theta
dim_data = task.dim_data
dim_joint = task.dim_joint


# Model parameters from config
model_params = config.get("model", {})

if model_type == "simformer":
    params = SimformerParams(
        rngs=nnx.Rngs(0),
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
elif model_type == "flux":
    params = FluxParams(
        in_channels=model_params.get("in_channels", 1),
        vec_in_dim=model_params.get("vec_in_dim", None),
        context_in_dim=model_params.get("context_in_dim", 1),
        mlp_ratio=model_params.get("mlp_ratio", 4),
        qkv_multiplier=model_params.get("qkv_multiplier", 1),
        num_heads=model_params.get("num_heads", 4),
        depth=model_params.get("depth", 8),
        depth_single_blocks=model_params.get("depth_single_blocks", 16),
        axes_dim=model_params.get("axes_dim", [10]),
        qkv_bias=model_params.get("qkv_bias", True),
        obs_dim=dim_theta,
        cond_dim=dim_data,
        theta=model_params.get("theta", (dim_theta+dim_data)*10),
        rngs=nnx.Rngs(default=42),
        param_dtype=getattr(jnp, model_params.get("param_dtype", "float32")),
    )

# define the appropriate pipeline
if model_type == "simformer" and method == "flow":
    PipelineClass = SimformerFlowPipeline
elif model_type == "simformer" and method == "diffusion":
    PipelineClass = SimformerDiffusionPipeline
elif model_type == "flux" and method == "flow":
    PipelineClass = FluxFlowPipeline
elif model_type == "flux" and method == "diffusion":
    PipelineClass = FluxDiffusionPipeline
else:
    raise ValueError(
        f"Invalid combination of model_type {model_type} and method {method}"
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
    dim_theta.item(),
    dim_data.item(),
    params,
    training_config,
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


def get_samples(idx, nsamples=10_000, use_ema=False, rng=None):
    observation, reference_samples = task.get_reference(idx)
    true_param = jnp.array(task.get_true_parameters(idx))

    if rng is None:
        rng = jax.random.PRNGKey(42)

    samples = pipeline.sample(rng, observation, nsamples, use_ema=use_ema)
    return samples, true_param, reference_samples


# Run C2ST
print("Running C2ST tests...")

c2st_accuracies = []
for idx in range(1, 11):
    samples, true_param, reference_samples = get_samples(
        idx, nsamples=10_000, use_ema=False
    )
    c2st_accuracy = c2st(reference_samples, samples)
    c2st_accuracies.append(c2st_accuracy)
    print(f"C2ST accuracy for observation={idx}: {c2st_accuracy:.4f}\n")

print(
    f"Average C2ST accuracy: {np.mean(c2st_accuracies):.4f} +- {np.std(c2st_accuracies):.4f}"
)
# Save C2ST results in a txt file
c2st_results_file = f"{notebook_path}/c2st_results_{experiment_id}_{method}_{model_type}.txt"
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
    c2st_accuracy = c2st(reference_samples, samples)
    c2st_accuracies_ema.append(c2st_accuracy)
    print(f"C2ST accuracy EMA for observation={idx}: {c2st_accuracy:.4f}\n")
print(
    f"Average C2ST accuracy EMA: {np.mean(c2st_accuracies_ema):.4f} +- {np.std(c2st_accuracies_ema):.4f}"
)
# Save C2ST results in a txt file
c2st_results_file_ema = f"{notebook_path}/c2st_results_ema_{experiment_id}_{method}_{model_type}.txt"
with open(c2st_results_file_ema, "w") as f:
    for idx, accuracy in enumerate(c2st_accuracies_ema, start=1):
        f.write(f"C2ST accuracy EMA for observation={idx}: {accuracy:.4f}\n")

    # print mean and std accuracy
    f.write(
        f"Average C2ST accuracy EMA: {np.mean(c2st_accuracies_ema):.4f} +- {np.std(c2st_accuracies_ema):.4f}\n"
    )
print("C2ST tests complete.")
