
# -------------------
# 1. Imports and Config Loading
# -------------------
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
from gensbi.models import Simformer, SimformerParams, SimformerCFMLoss, SimformerWrapper
from gensbi_examples.c2st import c2st
from gensbi_examples.mask import get_condition_mask_fn


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

# -------------------
# 2. Parameter Extraction from Config
# -------------------
task_name = config.get("task_name", None)
experiment_directory = f"examples/sbi-benchmarks/{task_name}"
train_params = config.get("training", {})
opt_params = config.get("optimizer", {})
model_params = config.get("model", {})

# Training parameters
experiment_id = train_params.get("experiment_id", 3)
restore_model = train_params.get("restore_model", False)
train_model = train_params.get("train_model", True)
batch_size = train_params.get("batch_size", 4096)
early_stopping = train_params.get("early_stopping", True)
print_every = train_params.get("print_every", 100)
total_number_steps_scaling = train_params.get("total_number_steps_scaling", 3)
max_number_steps = train_params.get("max_number_steps", 100000)
min_number_steps = train_params.get("min_number_steps", 5000)

total_number_steps = int(
    max(
        min(
            1e5 * total_number_steps_scaling,
            max_number_steps,
        ),
        min_number_steps,
    )
)

# Optimizer parameters
MAX_LR = opt_params.get("max_lr", 1e-3)
MIN_LR = opt_params.get("min_lr", 0.0)


# -------------------
# 3. Dataset and Task Setup
# -------------------
if experiment_directory is not None:
    os.chdir(experiment_directory)

task = get_task(task_name)
train_dataset = task.get_train_dataset(batch_size)
val_dataset = task.get_val_dataset()
dataset_iter = iter(train_dataset)
val_dataset_iter = iter(val_dataset)

# Set checkpoint directory and notebook path
notebook_path = os.getcwd()
checkpoint_dir = f"{notebook_path}/checkpoints/{task_name}_simformer"
os.makedirs(checkpoint_dir, exist_ok=True)

# JAX mesh setup
devices = jax.devices()
mesh = jax.sharding.Mesh(devices, axis_names=("data",))


def marginalize(rng: jax.random.PRNGKey, edge_mask: jax.Array, marginal_ids=None):
    if marginal_ids is None:
        marginal_ids = jnp.arange(edge_mask.shape[0])

    idx = jax.random.choice(
        rng, marginal_ids, shape=(1,), replace=False
    )
    edge_mask = edge_mask.at[idx, :].set(False)
    edge_mask = edge_mask.at[:, idx].set(False)
    edge_mask = edge_mask.at[idx, idx].set(True)
    return edge_mask


def next_batch():
    return next(dataset_iter)


def next_val_batch():
    return next(val_dataset_iter)


# Model definition
path = AffineProbPath(scheduler=CondOTScheduler())
dim_theta = task.dim_theta
dim_data = task.dim_data
dim_joint = task.dim_joint
node_ids = jnp.arange(dim_joint)
obs_ids = jnp.arange(dim_theta)  # observation ids
cond_ids = jnp.arange(dim_theta, dim_joint)  # conditional ids

# Model parameters from config
model_params = config.get("model", {})
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

loss_fn_cfm = SimformerCFMLoss(path)

undirected_edge_mask = jnp.ones((dim_joint, dim_joint), dtype=jnp.bool_)
posterior_mask = jnp.concatenate(
    [jnp.zeros((dim_theta), dtype=jnp.bool_), jnp.ones((dim_data), dtype=jnp.bool_)],
    axis=-1,
)
posterior_faithfull = task.get_edge_mask_fn("faithfull")(
    node_ids, condition_mask=posterior_mask
)

p0_dist_model = dist.Independent(
    dist.Normal(loc=jnp.zeros((dim_joint,)), scale=jnp.ones((dim_joint,))),
    reinterpreted_batch_ndims=1,
)


condition_mask_random_fn = get_condition_mask_fn(
    name="structured_random", theta_dim=dim_theta.item(), x_dim=dim_data.item()
)

condition_mask_posterior_fn = get_condition_mask_fn(
    name="posterior", theta_dim=dim_theta.item(), x_dim=dim_data.item()
)

def loss_fn_(vf_model, x_1, key: jax.random.PRNGKey, mask="structured_random"):
    batch_size = x_1.shape[0]
    rng_x0, rng_t, rng_condition, rng_edge_mask1, rng_edge_mask2 = jax.random.split(
        key, 5
    )
    x_0 = p0_dist_model.sample(rng_x0, (batch_size,))
    t = jax.random.uniform(rng_t, x_1.shape[0])
    batch = (x_0, x_1, t)

    if mask == "structured_random":
        condition_mask_fn = condition_mask_random_fn
    elif mask == "posterior":
        condition_mask_fn = condition_mask_posterior_fn

    condition_mask = condition_mask_fn(key=rng_condition, num_samples=batch_size)

    undirected_edge_mask_ = jnp.repeat(
        undirected_edge_mask[None, ...], 4*batch_size//5, axis=0
    )
    # faithfull_edge_mask_ = jnp.repeat(
    #     posterior_faithfull[None, ...], 3 * batch_size, axis=0
    # )
    marginal_mask = jax.vmap(marginalize, in_axes=(0, None, None))(
        jax.random.split(rng_edge_mask1, (batch_size//5,)), undirected_edge_mask, obs_ids
    )
    edge_masks = jnp.concatenate(
        [undirected_edge_mask_, marginal_mask], axis=0
    )

    edge_masks = jax.random.choice(rng_edge_mask2, edge_masks, shape=(batch_size,), axis=0) # Randomly choose between dense and marginal mask

    loss = loss_fn_cfm(
        vf_model,
        batch,
        node_ids=node_ids,
        edge_mask=edge_masks,
        condition_mask=condition_mask,
    )
    return loss


@nnx.jit
def train_loss(vf_model, key: jax.random.PRNGKey):
    x_1 = next_batch()
    return loss_fn_(vf_model, x_1, key, "posterior")


@nnx.jit
def val_loss(vf_model, key):
    x_1 = next_val_batch()
    return loss_fn_(vf_model, x_1, key, "posterior")


@nnx.jit
def train_step(model, optimizer, rng):
    loss_fn = lambda model: train_loss(model, rng)
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads, value=loss)
    return loss


vf_model = Simformer(params)

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


# optimizer setup 
schedule = optax.linear_schedule(
    MAX_LR,
    MIN_LR,
    total_number_steps // 2,
    total_number_steps // 2,
)
opt = optax.chain(
    optax.adaptive_grad_clip(10.0), optax.adam(schedule)
)

optimizer = nnx.Optimizer(vf_model, opt)

rngs = nnx.Rngs(0)
best_state = nnx.state(vf_model)
min_val = val_loss(vf_model, jax.random.PRNGKey(0))
val_error_ratio = 1.1
counter = 0
cmax = 10
print_every = 100
loss_array = []
val_loss_array = []
early_stopping = True

if train_model:
    vf_model.train()

    pbar = tqdm(range(total_number_steps))
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
        if j > 0 and j % 100 == 0:
            loss_ = l / 100
            val_ = v_l / 100
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

print("Running C2ST test...")

from gensbi.flow_matching.solver import ODESolver


# Wrap the trained model for conditional sampling
vf_wrapped = SimformerWrapper(vf_model)

step_size = 0.01


def get_samples(vf_wrapped, idx, nsamples=10_000, edge_mask=posterior_faithfull):
    observation, reference_samples = task.get_reference(idx)
    true_param = jnp.array(task.get_true_parameters(idx))

    rng = jax.random.PRNGKey(45)
    key1, key2 = jax.random.split(rng, 2)

    x_init = jax.random.normal(key1, (nsamples, dim_theta))
    cond = jnp.broadcast_to(observation[..., None], (1, dim_data, 1))

    solver = ODESolver(velocity_model=vf_wrapped)
    model_extras = {
        "cond": cond,
        "obs_ids": obs_ids,
        "cond_ids": cond_ids,
        "edge_mask": edge_mask,
    }

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
for idx in tqdm(range(1, 11), desc="C2ST Test"):  # Progress bar for C2ST test
    samples, true_param, reference_samples = get_samples(
        vf_wrapped, idx, nsamples=10_000
    )
    c2st_accuracy = c2st(reference_samples, samples)
    c2st_accuracies.append(c2st_accuracy)

print(
    f"Average C2ST accuracy: {np.mean(c2st_accuracies):.4f} +- {np.std(c2st_accuracies):.4f}"
)
# Save C2ST results in a txt file
c2st_results_file = f"{notebook_path}/c2st_results.txt"
with open(c2st_results_file, "w") as f:
    for idx, accuracy in enumerate(c2st_accuracies, start=1):
        f.write(f"C2ST accuracy for observation={idx}: {accuracy:.4f}\n")

    # print mean and std accuracy
    f.write(
        f"Average C2ST accuracy: {np.mean(c2st_accuracies):.4f} +- {np.std(c2st_accuracies):.4f}\n"
    )
