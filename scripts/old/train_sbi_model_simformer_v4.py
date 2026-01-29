# this model adopts an averaged loss


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
from gensbi.models import Simformer, SimformerParams, JointCFMLoss, JointWrapper
from gensbi.diagnostics.metrics import c2st

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


# Change working directory to experiment_directory from config
task_name = config.get("task_name", None)
experiment_directory = f"examples/sbi-benchmarks/{task_name}"

if experiment_directory is not None:
    os.chdir(experiment_directory)

# Task and dataset setup
task = get_task(task_name)

# Training parameters
train_params = config.get("training", {})
experiment_id = train_params.get("experiment_id", 3)
restore_model = train_params.get("restore_model", False)
train_model = train_params.get("train_model", True)
batch_size = train_params.get("batch_size", 4096)
nsteps = train_params.get("nsteps", 10000)
nepochs = train_params.get("nepochs", 3)
multistep = train_params.get("multistep", 1)
early_stopping = train_params.get("early_stopping", True)
print_every = train_params.get("print_every", 100)

# Set checkpoint directory
notebook_path = os.getcwd()
checkpoint_dir = f"{notebook_path}/checkpoints/{task_name}_simformer"
os.makedirs(checkpoint_dir, exist_ok=True)

# JAX mesh setup
devices = jax.devices()
mesh = jax.sharding.Mesh(devices, axis_names=("data",))

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

train_dataset = task.get_train_dataset(batch_size)
val_dataset = task.get_val_dataset()
dataset_iter = iter(train_dataset)
val_dataset_iter = iter(val_dataset)

# Helper functions (restored from original script)
from functools import partial


from gensbi_examples.mask import get_condition_mask_fn


# @partial(jax.jit, static_argnames=["nsamples"])
# def get_random_condition_mask(key: jax.random.PRNGKey, nsamples):
#     mask_joint = jnp.zeros((5 * nsamples, dim_joint), dtype=jnp.bool_)
#     mask_posterior = jnp.concatenate(
#         [
#             jnp.zeros((nsamples, dim_obs), dtype=jnp.bool_),
#             jnp.ones((nsamples, dim_cond), dtype=jnp.bool_),
#         ],
#         axis=-1,
#     )
#     mask1 = jax.random.bernoulli(key, p=0.3, shape=(nsamples, dim_joint))
#     filter = ~jnp.all(mask1, axis=-1)
#     mask1 = jnp.logical_and(mask1, filter.reshape(-1, 1))
#     masks = jnp.concatenate([mask_joint, mask1, mask_posterior], axis=0)
#     return jax.random.choice(key, masks, shape=(nsamples,), replace=False, axis=0)


def marginalize(key: jax.random.PRNGKey, edge_mask: jax.Array, marginal_ids=None):
    if marginal_ids is None:
        marginal_ids = jnp.arange(edge_mask.shape[0])

    idx = jax.random.choice(key, marginal_ids, shape=(1,), replace=False)
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
dim_obs = task.dim_obs
dim_cond = task.dim_cond
dim_joint = task.dim_joint
node_ids = jnp.arange(dim_joint)
obs_ids = jnp.arange(dim_obs)  # observation ids
cond_ids = jnp.arange(dim_obs, dim_joint)  # conditional ids

# Model parameters from config
model_params = config.get("model", {})
params = SimformerParams(
    rngs=nnx.Rngs(0),
    in_channels=model_params.get("in_channels", 1),
    value_emb_dim=model_params.get("value_emb_dim", 40),
    id_emb_dim=model_params.get("id_emb_dim", 40),
    cond_emb_dim=model_params.get("cond_emb_dim", 10),
    dim_joint=dim_joint,
    fourier_features=model_params.get("fourier_features", 128),
    num_heads=model_params.get("num_heads", 6),
    num_layers=model_params.get("num_layers", 8),
    widening_factor=model_params.get("widening_factor", 3),
    qkv_features=model_params.get("qkv_features", 90),
    num_hidden_layers=model_params.get("num_hidden_layers", 1),
)

loss_fn_cfm = JointCFMLoss(path)

undirected_edge_mask = jnp.ones((dim_joint, dim_joint), dtype=jnp.bool_)

# posterior_mask = jnp.concatenate(
#     [jnp.zeros((dim_obs), dtype=jnp.bool_), jnp.ones((dim_cond), dtype=jnp.bool_)],
#     axis=-1,
# )
# posterior_faithfull = task.get_edge_mask_fn("faithfull")(
#     node_ids, condition_mask=posterior_mask
# )

p0_dist_model = dist.Independent(
    dist.Normal(loc=jnp.zeros((dim_joint,)), scale=jnp.ones((dim_joint,))),
    reinterpreted_batch_ndims=1,
)


# @partial(jax.jit(static_argnames=["num_samples", "theta_dim", "x_dim"]))
# @partial(jax.jit, static_argnames=["num_samples", "theta_dim", "x_dim","p_joint", "p_posterior", "p_likelihood", "p_rnd1", "p_rnd2", "rnd1_prob", "rnd2_prob"])
# def sample_structured_conditional_mask(
#     key,
#     num_samples,
#     theta_dim,
#     x_dim,
#     p_joint=0.2,
#     p_posterior=0.2,
#     p_likelihood=0.2,
#     p_rnd1=0.2,
#     p_rnd2=0.2,
#     rnd1_prob=0.3,
#     rnd2_prob=0.7,
# ):
#     # Joint, posterior, likelihood, random1_mask, random2_mask
#     key1, key2, key3 = jax.random.split(key, 3)
#     condition_mask = jax.random.choice(
#         key1,
#         jnp.array(
#             [[False] * (theta_dim + x_dim)]
#             + [[False] * theta_dim + [True] * x_dim]
#             + [
#                 [True] * theta_dim + [False] * x_dim,
#                 jax.random.bernoulli(
#                     key2, rnd1_prob, shape=(theta_dim + x_dim,)
#                 ).astype(jnp.bool_),
#                 jax.random.bernoulli(
#                     key3, rnd2_prob, shape=(theta_dim + x_dim,)
#                 ).astype(jnp.bool_),
#             ]
#         ),
#         shape=(num_samples,),
#         p=jnp.array([p_joint, p_posterior, p_likelihood, p_rnd1, p_rnd2]),
#         axis=0,
#     )
#     all_ones_mask = jnp.all(condition_mask, axis=-1)
#     # If all are ones, then set to false
#     condition_mask = jnp.where(all_ones_mask[..., None], False, condition_mask)
#     return condition_mask


def sample_structured_conditional_mask(
    key,
    num_samples,
    theta_dim,
    x_dim,
    p_joint=0.2,
    p_posterior=0.2,
    p_likelihood=0.2,
    p_rnd1=0.2,
    p_rnd2=0.2,
    rnd1_prob=0.3,
    rnd2_prob=0.7,
):
    # Joint, posterior, likelihood, random1_mask, random2_mask
    key1, key2, key3 = jax.random.split(key, 3)
    joint_mask = jnp.array([False] * (theta_dim + x_dim), dtype=jnp.bool_)
    posterior_mask = jnp.array([False] * theta_dim + [True] * x_dim, dtype=jnp.bool_)
    likelihood_mask = jnp.array([True] * theta_dim + [False] * x_dim, dtype=jnp.bool_)
    random1_mask = jax.random.bernoulli(
        key2, rnd1_prob, shape=(theta_dim + x_dim,)
    ).astype(jnp.bool_)
    random2_mask = jax.random.bernoulli(
        key3, rnd2_prob, shape=(theta_dim + x_dim,)
    ).astype(jnp.bool_)
    mask_options = jnp.stack(
        [joint_mask, posterior_mask, likelihood_mask, random1_mask, random2_mask],
        axis=0,
    )  # (5, theta_dim + x_dim)
    idx = jax.random.choice(
        key1,
        5,
        shape=(num_samples,),
        p=jnp.array([p_joint, p_posterior, p_likelihood, p_rnd1, p_rnd2]),
    )
    condition_mask = mask_options[idx]
    all_ones_mask = jnp.all(condition_mask, axis=-1)
    # If all are ones, then set to false
    condition_mask = jnp.where(all_ones_mask[..., None], False, condition_mask)
    return condition_mask


# @partial(jax.jit, static_argnames=["nsamples"])
# def get_random_condition_mask(key: jax.random.PRNGKey, nsamples):
#     mask_joint = jnp.zeros((5 * nsamples, dim_joint), dtype=jnp.bool_)
#     mask_posterior = jnp.concatenate(
#         [
#             jnp.zeros((nsamples, dim_obs), dtype=jnp.bool_),
#             jnp.ones((nsamples, dim_cond), dtype=jnp.bool_),
#         ],
#         axis=-1,
#     )

#     mask1 = jax.random.bernoulli(key, p=0.3, shape=(nsamples, dim_joint))
#     filter = ~jnp.all(mask1, axis=-1)
#     mask1 = jnp.logical_and(mask1, filter.reshape(-1, 1))

#     # masks = jnp.concatenate([mask_joint, mask1, mask_posterior, mask_likelihood], axis=0)
#     masks = jnp.concatenate([mask_joint, mask1, mask_posterior], axis=0)
#     return jax.random.choice(key, masks, shape=(nsamples,), replace=False, axis=0)


def loss_fn_(vf_model, x_1, key: jax.random.PRNGKey, mask="structured_random"):
    batch_size = x_1.shape[0]
    rng_x0, rng_t, rng_condition, rng_edge_mask1, rng_edge_mask2 = jax.random.split(
        key, 5
    )
    x_0 = p0_dist_model.sample(rng_x0, (batch_size,))
    t = jax.random.uniform(rng_t, x_1.shape[0])
    batch = (x_0, x_1, t)

    # condition_mask = get_random_condition_mask(rng_condition, batch_size)
    condition_mask = sample_structured_conditional_mask(
        rng_condition,
        batch_size,
        dim_obs.item(),
        dim_cond.item(),
    )

    edge_masks = undirected_edge_mask

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
def train_step(model, optimizer, key):
    loss_fn = lambda model: train_loss(model, key)
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads, value=loss)
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

# Optimizer setup
opt = optax.chain(
    optax.adaptive_grad_clip(10.0),
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
optimizer = nnx.Optimizer(vf_model, opt, wrt=nnx.Param)

rngs = nnx.Rngs(0)
best_state = nnx.state(vf_model)
min_val = val_loss(vf_model, jax.random.PRNGKey(0))
val_error_ratio = 1.1
counter = 0
cmax = 10
loss_array = []
val_loss_array = []

if train_model:
    vf_model.train()
    for ep in range(nepochs):
        pbar = tqdm(range(nsteps))  # todo fixme
        # pbar = tqdm(range(total_number_steps))
        l_train = None
        l_val = None

        for j in pbar:
            if counter > cmax and early_stopping:
                print("Early stopping")
                graphdef, abstract_state = nnx.split(vf_model)
                vf_model = nnx.merge(graphdef, best_state)
                break
            loss = train_step(vf_model, optimizer, rngs.train_step())
            v_loss = val_loss(vf_model, rngs.val_step())

            if j == 0:
                l_train = loss
                l_val = v_loss
            else:
                # l_train = 0.9 * l_train + 0.1 * loss
                l_train += loss
                l_val += v_loss

            if j > 50 and j % print_every == 0:
                l_train /= print_every
                l_val /= print_every
                ratio = l_val / l_train
                if ratio > val_error_ratio:
                    counter += 1
                else:
                    counter = 0

                pbar.set_postfix(
                    loss=f"{l_train:.4f}",
                    ratio=f"{ratio:.4f}",
                    counter=counter,
                    val_loss=f"{l_val:.4f}",
                )
                loss_array.append(l_train)
                val_loss_array.append(l_val)

                if l_val < min_val:
                    min_val = l_val
                    best_state = nnx.state(vf_model)

                l_val = 0
                l_train = 0

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

from gensbi.flow_matching.solver import ODESolver


# Wrap the trained model for conditional sampling
vf_wrapped = JointWrapper(vf_model)

step_size = 0.01


def get_samples(vf_wrapped, idx, nsamples=10_000, edge_mask=None):
    observation, reference_samples = task.get_reference(idx)
    true_param = jnp.array(task.get_true_parameters(idx))

    key = jax.random.PRNGKey(45)
    key1, key2 = jax.random.split(key, 2)

    x_init = jax.random.normal(key1, (nsamples, dim_obs))
    cond = jnp.broadcast_to(observation[..., None], (1, dim_cond, 1))

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
print("Running C2ST tests...")

c2st_accuracies = []
for idx in range(1, 11):
    samples, true_param, reference_samples = get_samples(
        vf_wrapped, idx, nsamples=10_000
    )
    c2st_accuracy = c2st(reference_samples, samples)
    c2st_accuracies.append(c2st_accuracy)
    print(f"C2ST accuracy for observation={idx}: {c2st_accuracy:.4f}\n")

print(
    f"Average C2ST accuracy: {np.mean(c2st_accuracies):.4f} +- {np.std(c2st_accuracies):.4f}"
)
# Save C2ST results in a txt file
c2st_results_file = f"{notebook_path}/c2st_results_{experiment_id}.txt"
with open(c2st_results_file, "w") as f:
    for idx, accuracy in enumerate(c2st_accuracies, start=1):
        f.write(f"C2ST accuracy for observation={idx}: {accuracy:.4f}\n")

    # print mean and std accuracy
    f.write(
        f"Average C2ST accuracy: {np.mean(c2st_accuracies):.4f} +- {np.std(c2st_accuracies):.4f}\n"
    )
