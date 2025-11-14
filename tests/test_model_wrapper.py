# this uses EMA for the model weights
# %%

import os

# Set JAX backend (use 'cuda' for GPU, 'cpu' otherwise)
os.environ["JAX_PLATFORMS"] = "cpu"

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

from typing import Optional

from time import time

# %% old model wrapper

from abc import ABC
from flax import nnx
from jax import Array
import jax.numpy as jnp

from typing import Callable

from gensbi.utils.math import divergence

from einops import rearrange


class ModelWrapper_old(nnx.Module):
    """
    This class is used to wrap around another model. We define a call method which returns the model output. 
    Furthermore, we define a vector_field method which computes the vector field of the model,
    and a divergence method which computes the divergence of the model, in a form useful for diffrax.
    This is useful for ODE solvers that require the vector field and divergence of the model.

    """

    def __init__(self, model: nnx.Module):
        self.model = model

    def _call_model(self, x: Array, t: Array, args, **kwargs) -> Array:
        r"""
        This method is a wrapper around the model's call method. It allows us to pass additional arguments
        to the model, such as text conditions or other auxiliary information.

        Args:
            x (Array): input data to the model (batch_size, ...).
            t (Array): time (batch_size).
            args: additional information forwarded to the model, e.g., text condition.
            **kwargs: additional keyword arguments.

        Returns:
            Array: model output.
        """
        return self.model(x, t, args=args, **kwargs) # type: ignore

    def __call__(self, x: Array, t: Array, args=None, **kwargs) -> Array:
        r"""
        This method defines how inputs should be passed through the wrapped model.
        Here, we're assuming that the wrapped model takes both :math:`x` and :math:`t` as input,
        along with any additional keyword arguments.

        Optional things to do here:
            - check that t is in the dimensions that the model is expecting.
            - add a custom forward pass logic.
            - call the wrapped model.

        | given x, t
        | returns the model output for input x at time t, with extra information `extra`.

        Args:
            x (Array): input data to the model (batch_size, ...).
            t (Array): time (batch_size).
            **extras: additional information forwarded to the model, e.g., text condition.

        Returns:
            Array: model output.
        """
        return self._call_model(x, t, args, **kwargs)

    def get_vector_field(self, **kwargs) -> Callable:
        r"""Compute the vector field of the model, properly squeezed for the ODE term.

        Args:
            x (Array): input data to the model (batch_size, ...).
            t (Array): time (batch_size).
            args: additional information forwarded to the model, e.g., text condition.

        Returns:
            Array: vector field of the model.
        """
        def vf(t, x, args):
            vf = self._call_model(x, t, args, **kwargs)
            # squeeze the first dimension of the vector field if it is 1
            if vf.shape[0] == 1:
                vf = jnp.squeeze(vf, axis=0)
            return vf
        return vf
    

    def get_divergence(self, **kwargs) -> Callable:
        r"""Compute the divergence of the model.

        Args:
            t (Array): time (batch_size).
            x (Array): input data to the model (batch_size, ...).
            args: additional information forwarded to the model, e.g., text condition.

        Returns:
            Array: divergence of the model.
        """
        vf = self.get_vector_field(**kwargs)
        def div_(t, x, args):
            div = divergence(vf, t, x, args)
            # squeeze the first dimension of the divergence if it is 1
            if div.shape[0] == 1:
                div = jnp.squeeze(div, axis=0)
            return div

        
        return div_
        

class SimformerConditioner_old(nnx.Module):
    """
    Module to handle conditioning in the Simformer model.

    Args:
        model (Simformer): Simformer model instance.
    """
    def __init__(self, model: Simformer, dim_joint):
        self.model = model
        self.dim_joint = dim_joint

    def conditioned(
        self, 
        obs: Array, 
        obs_ids: Array, 
        cond: Array, 
        cond_ids: Array, 
        t: Array, 
        edge_mask: Optional[Array] = None
    ) -> Array:
        """
        Perform conditioned inference.

        Args:
            obs (Array): Observations.
            obs_ids (Array): Observation identifiers.
            cond (Array): Conditioning values.
            cond_ids (Array): Conditioning identifiers.
            t (Array): Time steps.
            edge_mask (Optional[Array]): Mask for edges.

        Returns:
            Array: Conditioned output.
        """
        obs = jnp.atleast_1d(obs)
        cond = jnp.atleast_1d(cond)
        t = jnp.atleast_1d(t)

        if obs.ndim < 3:
            obs = rearrange(obs, "... -> 1 ... 1" if obs.ndim == 1 else "... -> ... 1")

        if cond.ndim < 3:
            cond = rearrange(
                cond, "... -> 1 ... 1" if cond.ndim == 1 else "... -> ... 1"
            )
        
        # repeat cond on the first dimension to match obs
        cond = jnp.broadcast_to(
            cond, (obs.shape[0], *cond.shape[1:])
        )

        condition_mask_dim = len(obs_ids) + len(cond_ids)

        condition_mask = jnp.zeros((condition_mask_dim,), dtype=jnp.bool_)
        condition_mask = condition_mask.at[cond_ids].set(True)

        x = jnp.concatenate([obs, cond], axis=1)
        node_ids = jnp.concatenate([obs_ids, cond_ids])

        # Sort the nodes and the corresponding values
        # nodes_sort = jnp.argsort(node_ids)
        # x = x[:, nodes_sort]
        # node_ids = node_ids[nodes_sort]

        res = self.model(
            obs=x,
            t=t,
            node_ids=node_ids,
            condition_mask=condition_mask,
            edge_mask=edge_mask,
        )
        # now return only the values on which we are not conditioning
        res = res[:, :len(obs_ids)]
        # res = jnp.take_along_axis(res, obs_ids, axis=1)
        return res

    def unconditioned(
        self, 
        obs: Array, 
        obs_ids: Array, 
        t: Array, 
        edge_mask: Optional[Array] = None
    ) -> Array:
        """
        Perform unconditioned inference.

        Args:
            obs (Array): Observations.
            obs_ids (Array): Observation identifiers.
            t (Array): Time steps.
            edge_mask (Optional[Array]): Mask for edges.

        Returns:
            Array: Unconditioned output.
        """
        obs = jnp.atleast_1d(obs)
        t = jnp.atleast_1d(t)

        if obs.ndim < 3:
            obs = rearrange(obs, "... -> 1 ... 1" if obs.ndim == 1 else "... -> ... 1")

        condition_mask = jnp.zeros((obs.shape[1],), dtype=jnp.bool_)

        node_ids = obs_ids

        res = self.model(
            obs=obs,
            t=t,
            node_ids=node_ids,
            condition_mask=condition_mask,
            edge_mask=edge_mask,
        )

        return res

    def __call__(
        self, 
        obs: Array, 
        obs_ids: Array, 
        cond: Array, 
        cond_ids: Array, 
        t: Array, 
        conditioned: bool = True, 
        edge_mask: Optional[Array] = None
    ) -> Array:
        """
        Perform inference based on conditioning.

        Args:
            obs (Array): Observations.
            obs_ids (Array): Observation identifiers.
            cond (Array): Conditioning values.
            cond_ids (Array): Conditioning identifiers.
            timesteps (Array): Time steps.
            conditioned (bool): Whether to perform conditioned inference.
            edge_mask (Optional[Array]): Mask for edges.

        Returns:
            Array: Model output.
        """
        if conditioned:
            return self.conditioned(
                obs, obs_ids, cond, cond_ids, t, edge_mask=edge_mask
            )
        else:
            return self.unconditioned(obs, obs_ids, t, edge_mask=edge_mask)


class JointWrapper_old(ModelWrapper_old):
    def __init__(self, model, dim_joint):
        model_conditioned = SimformerConditioner_old(model, dim_joint)
        super().__init__(model_conditioned)

    def _call_model(self, x, t, args, **kwargs):
        return self.model(obs=x, t=t, **kwargs)

#%% 

######## new ema cod

# Task and dataset setup
task = get_task("two_moons")




train_dataset = task.get_train_dataset(32)
val_dataset = task.get_val_dataset()



# %% define all model parameters and training settings


# Model definition
dim_theta = task.dim_theta
dim_data = task.dim_data
dim_joint = task.dim_joint

params = SimformerParams(
    rngs=nnx.Rngs(0),
    dim_value=10,
    dim_id=10,
    dim_condition=10,
    dim_joint=dim_joint,
    fourier_features=128,
    num_heads=2,
    num_layers=2,
    widening_factor=3,
    qkv_features=10,
    num_hidden_layers=1,
)

# define the pipeline
pipeline = SimformerFlowPipeline(train_dataset, val_dataset, 2, 2, params=params)
# %%
rngs = nnx.Rngs(0)
pipeline.train(rngs,nsteps=30,save_model=False)
# %%
pipeline._wrap_model()
#%%
vf_model_wrapped = pipeline.model_wrapped
#%% get the old wrapped model

vf_model_wrapped_old = JointWrapper_old(pipeline.model, dim_joint)
# %%
# x = jnp.array([[-1.0, 0.0], [1.0, 0.0]]).reshape(2,2,1)
# t = jnp.array([0.5, 0.5])
# args = {
#     "obs_ids": jnp.array([0, 1]).reshape(1,2,1),
#     "cond": jnp.array([[0.0, 1.0], [0.0, 1.0]]).reshape(2,2,1),
#     "cond_ids": jnp.array([2, 3]).reshape(1,2,1),
#     "conditioned": True,
# }

batch_size = 5
obs = jax.random.normal(jax.random.PRNGKey(0), (batch_size, dim_theta, 1))*2
cond = jax.random.normal(jax.random.PRNGKey(1), (batch_size, dim_data, 1))*2
t = jax.random.uniform(jax.random.PRNGKey(2), (batch_size,), minval=0.0, maxval=1.0)


model_extras = {
            "cond": cond,
            "obs_ids": pipeline.obs_ids,
            "cond_ids": pipeline.cond_ids,
            "edge_mask": None,
        }
# %%

res = vf_model_wrapped(t, obs, **model_extras)
# %%
vf_old = vf_model_wrapped_old.get_vector_field(**model_extras, conditioned=True)
# %%
res_old = vf_old(t, obs, None)
# %%

res
#%%
res_old
# %%
jnp.isclose(res, res_old, atol=1e-1).all()
# %%
