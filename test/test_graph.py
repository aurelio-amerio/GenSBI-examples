import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import pytest
from gensbi_examples.graph import find_ancestors_jax


def test_basic_chain():
    # 0 -> 1 -> 2
    num_nodes = 3
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.int32)
    mask = mask.at[1, 0].set(1)
    mask = mask.at[2, 1].set(1)

    ancestors = find_ancestors_jax(mask, 2)

    assert ancestors[1] == True
    assert ancestors[0] == True
    assert ancestors[2] == False

    ancestors = find_ancestors_jax(mask, 1)
    assert ancestors[0] == True
    assert ancestors[1] == False
    assert ancestors[2] == False


def test_diamond():
    # 0 -> 1 -> 3
    # 0 -> 2 -> 3
    num_nodes = 4
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.int32)
    mask = mask.at[1, 0].set(1)
    mask = mask.at[2, 0].set(1)
    mask = mask.at[3, 1].set(1)
    mask = mask.at[3, 2].set(1)

    ancestors = find_ancestors_jax(mask, 3)

    assert ancestors[1] == True
    assert ancestors[2] == True
    assert ancestors[0] == True
    assert ancestors[3] == False


def test_branching_parents_bug():
    # 3 -> 1 -> 5
    # 4 -> 2 -> 5
    # Node 0 is unused (isolated) to avoid lucky hit.

    num_nodes = 6
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.int32)
    mask = mask.at[1, 3].set(1)
    mask = mask.at[2, 4].set(1)
    mask = mask.at[5, 1].set(1)
    mask = mask.at[5, 2].set(1)

    ancestors = find_ancestors_jax(mask, 5)

    assert ancestors[1] == True
    assert ancestors[2] == True
    assert ancestors[3] == True
    assert ancestors[4] == True
    assert ancestors[5] == False


def test_cycle():
    # 0 -> 1 -> 0
    num_nodes = 2
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.int32)
    mask = mask.at[1, 0].set(1)
    mask = mask.at[0, 1].set(1)

    ancestors = find_ancestors_jax(mask, 0)

    assert ancestors[1] == True
    assert ancestors[0] == True


def test_disconnected():
    # 0   1
    num_nodes = 2
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.int32)

    ancestors = find_ancestors_jax(mask, 0)
    assert not jnp.any(ancestors)
