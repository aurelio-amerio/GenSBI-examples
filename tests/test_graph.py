
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.numpy as jnp
import pytest
from gensbi_examples.graph import find_ancestors_jax

def test_find_ancestors_branching():
    # 0 -> 1, 0 -> 2
    # 1 -> 3
    # Expected ancestors of 0: {1, 2, 3}

    num_nodes = 4
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.int32)
    # mask[child, parent] = 1
    mask = mask.at[0, 1].set(1)
    mask = mask.at[0, 2].set(1)
    mask = mask.at[1, 3].set(1)

    ancestors = find_ancestors_jax(mask, 0)

    assert ancestors[1]
    assert ancestors[2]
    assert ancestors[3], "Failed to find ancestor 3 (grandparent via node 1)"
    assert not ancestors[0] # Self is not ancestor (unless cycle)

def test_find_ancestors_linear():
    # 0 -> 1 -> 2
    # Expected ancestors of 0: {1, 2}

    num_nodes = 3
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.int32)
    mask = mask.at[0, 1].set(1)
    mask = mask.at[1, 2].set(1)

    ancestors = find_ancestors_jax(mask, 0)

    assert ancestors[1]
    assert ancestors[2]
    assert not ancestors[0]

def test_find_ancestors_cycle():
    # 0 -> 1 -> 0
    # Expected ancestors of 0: {1, 0} because 1 is parent, and 0 is parent of 1
    # Actually, is_ancestor is initialized to False.
    # 0 -> process 0. stack=[1], is_ancestor[1]=True.
    # 1 -> process 1. mask[1,0]=1. 0 is neighbor.
    # if 0 not visited:
    #   is_ancestor[0]=True, stack=[1, 0]
    # So yes, in a cycle, self becomes ancestor.

    num_nodes = 2
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.int32)
    mask = mask.at[0, 1].set(1)
    mask = mask.at[1, 0].set(1)

    ancestors = find_ancestors_jax(mask, 0)

    assert ancestors[1]
    assert ancestors[0]

def test_find_ancestors_disconnected():
    # 0 -> 1
    # 2 -> 3
    # Query 0. Expected {1}.

    num_nodes = 4
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.int32)
    mask = mask.at[0, 1].set(1)
    mask = mask.at[2, 3].set(1)

    ancestors = find_ancestors_jax(mask, 0)

    assert ancestors[1]
    assert not ancestors[2]
    assert not ancestors[3]
    assert not ancestors[0]

if __name__ == "__main__":
    test_find_ancestors_branching()
    test_find_ancestors_linear()
    test_find_ancestors_cycle()
    test_find_ancestors_disconnected()
