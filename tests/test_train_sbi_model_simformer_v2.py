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
from gensbi_examples.c2st import c2st

from gensbi.models import SimformerParams
from gensbi.recipes import SimformerFlowPipeline


from functools import partial

from time import time

# %%

######## new ema code


class ModelEMA(nnx.Optimizer):

    def __init__(
        self,
        model: nnx.Module,
        tx: optax.GradientTransformation,
    ):
        super().__init__(model, tx, wrt=[nnx.Param, nnx.BatchStat])

    def update(self, model, model_orginal: nnx.Module):
        params = nnx.state(model_orginal, self.wrt)
        ema_params = nnx.state(model, self.wrt)
        self.step.value += 1

        ema_state = optax.EmaState(count=self.step, ema=ema_params)

        _, new_ema_state = self.tx.update(params, ema_state)

        nnx.update(model, new_ema_state.ema)


########
# %%
# Argument parser for config file

root_dir = "/home/aure/Documents/GitHub/GenSBI-examples"

config = (
    f"{root_dir}/examples/sbi-benchmarks/two_moons/config/config_flow_simformer_2.yaml"
)

# Load config
with open(config, "r") as f:
    config = yaml.safe_load(f)


# Change working directory to experiment_directory from config
task_name = config.get("task_name", None)
experiment_directory = f"{root_dir}/examples/sbi-benchmarks/two_moons"

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
nsteps = 1000  # train_params.get("nsteps", 10000)
nepochs = train_params.get("nepochs", 3)
multistep = train_params.get("multistep", 1)
early_stopping = train_params.get("early_stopping", True)
print_every = train_params.get("print_every", 100)

# Set checkpoint directory
notebook_path = os.getcwd()
checkpoint_dir = (
    f"{root_dir}/examples/sbi-benchmarks/two_moons/checkpoints/two_moons_flow_simformer"
)
checkpoint_dir_ema = f"{root_dir}/examples/sbi-benchmarks/two_moons/checkpoints/two_moons_flow_simformer/ema"
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(checkpoint_dir_ema, exist_ok=True)

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


# def next_batch():
#     return next(dataset_iter)


# def next_val_batch():
#     return next(val_dataset_iter)


# %% define all model parameters and training settings


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

p0_dist_model = dist.Independent(
    dist.Normal(loc=jnp.zeros((dim_joint, 1)), scale=jnp.ones((dim_joint, 1))),
    reinterpreted_batch_ndims=1,
)


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
def train_step(model, optimizer, x_1, key):
    loss, grads = nnx.value_and_grad(loss_fn_)(model, x_1, key)
    optimizer.update(model, grads, value=loss)
    return loss


@nnx.jit
def ema_step(ema_model, model, ema_optimizer: nnx.Optimizer):
    ema_optimizer.update(ema_model, model)


# define the ema params from the main model
vf_model = Simformer(params)
ema_model = nnx.clone(vf_model)

model_ema_decay = 0.99
ema_tx = optax.ema(model_ema_decay)
ema_optimizer = ModelEMA(ema_model, ema_tx)


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
best_state_ema = nnx.state(ema_model)

min_val = loss_fn_(vf_model, next(val_dataset_iter), jax.random.PRNGKey(0))
val_error_ratio = 1.1
counter = 0
cmax = 10

loss_array = []
val_loss_array = []


if train_model:
    vf_model.train()

    pbar = tqdm(range(nsteps))
    l = 0
    v_l = 0
    t0 = time()
    for j in pbar:
        if counter > cmax and early_stopping:
            print("Early stopping")
            graphdef, abstract_state = nnx.split(vf_model)
            vf_model = nnx.merge(graphdef, best_state)
            break

        x_1 = next(dataset_iter)

        loss = train_step(vf_model, optimizer, x_1, rngs.train_step())
        l += loss.item()

        x_1 = next(val_dataset_iter)
        v_loss = loss_fn_(vf_model, x_1, rngs.val_step())
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

    print("Manual training complete in {:.2f} seconds".format(time() - t0))

# %%
# define the pipeline
pipeline = SimformerFlowPipeline(train_dataset, val_dataset, 2, 2, params=params)

t0 = time()
pipeline.train(nnx.Rngs(0), nsteps=nsteps, save_model=False)

print("Pipeline training complete in {:.2f} seconds".format(time() - t0))
