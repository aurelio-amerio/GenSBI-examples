import os

# Set JAX backend (use 'cuda' for GPU, 'cpu' otherwise)
os.environ["JAX_PLATFORMS"] = "cuda"

import argparse
import yaml
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import optax
from optax.contrib import reduce_on_plateau
from numpyro import distributions as dist
from tqdm.auto import tqdm
from functools import partial
import orbax.checkpoint as ocp
from gensbi.flow_matching.path.scheduler import CondOTScheduler
from gensbi.flow_matching.path import AffineProbPath
from gensbi_examples.tasks import get_task
from gensbi.models import Flux1, Flux1Params, Flux1CFMLoss, Flux1Wrapper
from gensbi_examples.c2st import c2st
from gensbi.flow_matching.solver import ODESolver

# Argument parser for config file
parser = argparse.ArgumentParser(description="Flux1 Training Script")
parser.add_argument(
    "--config",
    type=str,
    default="config_flux.yaml",
    help="Path to YAML config file",
)
args, _ = parser.parse_known_args()

# Load config
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

# Change working directory to experiment_directory from config
task_name = config.get("task_name", None)
experiment_directory = f"examples/sbi-benchmarks/{task_name}"

if experiment_directory is not None:
    os.chdir(experiment_directory)

# Task and dataset setup
task = get_task(task_name)

# Training parameters
train_params = config.get("training", {})
experiment_id = train_params.get("experiment_id", 1)
restore_model = train_params.get("restore_model", False)
train_model = train_params.get("train_model", True)
batch_size = train_params.get("batch_size", 8192)
nsteps = train_params.get("nsteps", 10000)
nepochs = train_params.get("nepochs", 3)
multistep = train_params.get("multistep", 1)
early_stopping = train_params.get("early_stopping", True)
print_every = train_params.get("print_every", 50)
val_every = train_params.get("val_every", 50)
val_error_ratio = train_params.get("val_error_ratio", 1.1)
cmax = train_params.get("cmax", 5)
step_size = train_params.get("step_size", 0.01)

# Set checkpoint directory
notebook_path = os.getcwd()
checkpoint_dir = f"{notebook_path}/checkpoints/{task_name}_flux"
os.makedirs(checkpoint_dir, exist_ok=True)

# JAX mesh setup
devices = jax.devices()
mesh = jax.sharding.Mesh(devices, axis_names=("data",))

# Optimizer parameters
opt_params = config.get("optimizer", {})
PATIENCE = opt_params.get("patience", 5)
COOLDOWN = opt_params.get("cooldown", 2)
FACTOR = opt_params.get("factor", 0.5)
ACCUMULATION_SIZE = opt_params.get("accumulation_size", 50)
RTOL = opt_params.get("rtol", 1e-2)
MAX_LR = opt_params.get("max_lr", 5e-4)
MIN_LR = opt_params.get("min_lr", 1e-6)
MIN_SCALE = MIN_LR / MAX_LR if MAX_LR > 0 else 0.0

train_dataset = task.get_train_dataset(batch_size)
val_dataset = task.get_val_dataset()
dataset_iter = iter(train_dataset)
val_dataset_iter = iter(val_dataset)

def next_batch():
    return next(dataset_iter)

def next_val_batch():
    return next(val_dataset_iter)

# Model definition
path = AffineProbPath(scheduler=CondOTScheduler())
dim_theta = task.dim_theta
dim_data = task.dim_data
dim_joint = task.dim_joint

# Define observation and condition IDs
cond_ids = jnp.arange(dim_data, dim_joint, dtype=jnp.int32).reshape(1, -1, 1)
obs_ids = jnp.arange(dim_data, dtype=jnp.int32).reshape(1, -1, 1)

# Model parameters from config
model_params = config.get("model", {})
params = Flux1Params(
    in_channels=model_params.get("in_channels", 1),
    vec_in_dim=model_params.get("vec_in_dim", None),
    context_in_dim=model_params.get("context_in_dim", 1),
    mlp_ratio=model_params.get("mlp_ratio", 4),
    qkv_multiplier=model_params.get("qkv_multiplier", 1),
    num_heads=model_params.get("num_heads", 6),
    depth=model_params.get("depth", 8),
    depth_single_blocks=model_params.get("depth_single_blocks", 16),
    axes_dim=model_params.get("axes_dim", [6]),
    qkv_bias=model_params.get("qkv_bias", True),
    obs_dim=dim_theta,
    cond_dim=dim_data,
    theta=model_params.get("theta", 20),
    rngs=nnx.Rngs(default=42),
    param_dtype=getattr(jnp, model_params.get("param_dtype", "float32")),
)

loss_fn_cfm = Flux1CFMLoss(path)

p0_dist_model = dist.Independent(
    dist.Normal(loc=jnp.zeros((dim_theta,)), scale=jnp.ones((dim_theta,))),
    reinterpreted_batch_ndims=1,
)

def loss_fn_(vf_model, batch, key: jax.random.PRNGKey):
    obs = batch[:, :dim_theta][..., None]
    cond = batch[:, dim_theta:][..., None]

    key1, key2 = jax.random.split(key, 2)

    x_1 = obs
    x_0 = jax.random.normal(key1, x_1.shape)
    t = jax.random.uniform(key2, x_1.shape[0])

    batch = (x_0, x_1, t)

    loss = loss_fn_cfm(vf_model, batch, cond, obs_ids, cond_ids)
    return loss

@nnx.jit
def train_loss(vf_model, key: jax.random.PRNGKey):
    x_1 = next_batch()
    return loss_fn_(vf_model, x_1, key)

@nnx.jit
def val_loss(vf_model, key):
    x_1 = next_val_batch()
    return loss_fn_(vf_model, x_1, key)

@nnx.jit
def train_step(model, optimizer, rng):
    loss_fn = lambda model: train_loss(model, rng)
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads, value=loss)
    return loss

vf_model = Flux1(params)

if restore_model:
    model_state = nnx.state(vf_model)
    graphdef, abstract_state = nnx.split(vf_model)
    with ocp.CheckpointManager(
        checkpoint_dir, options=ocp.CheckpointManagerOptions(read_only=True)
    ) as read_mgr:
        restored = read_mgr.restore(
            experiment_id,
            args=ocp.args.Composite(state=ocp.args.PyTreeRestore(item=model_state)),
        )
    vf_model = nnx.merge(graphdef, restored["state"])
    print("Restored model from checkpoint")

# Optimizer setup
opt = optax.chain(
    # optax.adaptive_grad_clip(10.0),
    optax.adamw(MAX_LR),
    reduce_on_plateau(
        patience=PATIENCE,
        cooldown=COOLDOWN,
        factor=FACTOR,
        rtol=RTOL,
        accumulation_size=ACCUMULATION_SIZE,
        min_scale=MIN_SCALE,
    ),
)
if multistep > 1:
    opt = optax.MultiSteps(opt, multistep)
optimizer = nnx.Optimizer(vf_model, opt)

rngs = nnx.Rngs(0)
best_state = nnx.state(vf_model)
min_val = val_loss(vf_model, jax.random.PRNGKey(0))
counter = 0
loss_array = []
val_loss_array = []

if train_model:
    vf_model.train()
    for ep in range(nepochs):
        pbar = tqdm(range(nsteps))
        l = 0
        v_l = 0
        for j in pbar:
            if counter > cmax and early_stopping:
                print("Early stopping")
                graphdef, abstract_state = nnx.split(vf_model)
                vf_model = nnx.merge(graphdef, best_state)
                break
                
            loss = train_step(vf_model, optimizer, rngs.train_step())
            l += loss.item()
            v_loss = val_loss(vf_model, rngs.val_step())
            v_l += v_loss.item()
            
            if j > 0 and j % val_every == 0:
                loss_ = l / val_every
                val_ = v_l / val_every
                ratio1 = val_ / loss_
                ratio2 = val_ / min_val

                if ratio1 < val_error_ratio:
                    counter = 0
                else:
                    counter += 1

                if val_ < min_val:
                    min_val = val_
                    best_state = nnx.state(vf_model)

                pbar.set_postfix(
                    loss=f"{loss_:.4f}",
                    ratio=f"{ratio1:.4f}",
                    counter=counter,
                    val_loss=f"{val_:.4f}",
                )
                loss_array.append(loss_)
                val_loss_array.append(val_)
                l = 0
                v_l = 0
    vf_model.eval()
    
    # Save the model
    checkpoint_manager = ocp.CheckpointManager(
        checkpoint_dir,
        options=ocp.CheckpointManagerOptions(
            max_to_keep=None,
            keep_checkpoints_without_metrics=True,
            create=True,
        ),
    )
    model_state = nnx.state(vf_model)
    checkpoint_manager.save(
        experiment_id, args=ocp.args.Composite(state=ocp.args.PyTreeSave(model_state))
    )
    checkpoint_manager.close()
    print("Training complete and model saved.")

# --------- C2ST TEST ---------

# Wrap the trained model for conditional sampling
vf_wrapped = Flux1Wrapper(vf_model)

def get_samples(vf_wrapped, idx, nsamples=10_000):
    observation, reference_samples = task.get_reference(idx)
    true_param = jnp.array(task.get_true_parameters(idx))

    rng = jax.random.PRNGKey(45)
    key1, key2 = jax.random.split(rng, 2)

    x_init = jax.random.normal(key1, (nsamples, dim_theta))
    cond = jnp.broadcast_to(observation[..., None], (1, dim_data, 1))

    solver = ODESolver(velocity_model=vf_wrapped)
    model_extras = {"cond": cond, "obs_ids": obs_ids, "cond_ids": cond_ids}

    sampler_ = solver.get_sampler(
        method="Dopri5",
        step_size=step_size,
        return_intermediates=False,
        model_extras=model_extras,
    )
    samples = sampler_(x_init)
    return samples, true_param, reference_samples

# Run C2ST
c2st_accuracies = []
for idx in range(1, 11):
    samples, true_param, reference_samples = get_samples(
        vf_wrapped, idx, nsamples=10_000
    )
    c2st_accuracy = c2st(reference_samples, samples)
    c2st_accuracies.append(c2st_accuracy)

print(
    f"Average C2ST accuracy: {np.mean(c2st_accuracies):.4f} +- {np.std(c2st_accuracies):.4f}"
)

# Save C2ST results in a txt file
c2st_results_file = f"{notebook_path}/c2st_results_flux.txt"
with open(c2st_results_file, "w") as f:
    for idx, accuracy in enumerate(c2st_accuracies, start=1):
        f.write(f"C2ST accuracy for observation={idx}: {accuracy:.4f}\n")

    # print mean and std accuracy
    f.write(
        f"Average C2ST accuracy: {np.mean(c2st_accuracies):.4f} +- {np.std(c2st_accuracies):.4f}\n"
    )
