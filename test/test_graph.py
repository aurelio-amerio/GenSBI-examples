
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import jax.numpy as jnp
import jax
import pytest
from gensbi_examples.graph import find_ancestors_jax

def test_find_ancestors_simple_chain():
    # 0 -> 1 -> 2
    # Mask [child, parent]
    # 0: []
    # 1: [0]
    # 2: [1]
    mask = jnp.zeros((3, 3), dtype=jnp.int32)
    mask = mask.at[1, 0].set(1)
    mask = mask.at[2, 1].set(1)

    # Ancestors of 2 should be {0, 1}
    ancestors = find_ancestors_jax(mask, 2)
    expected = jnp.array([True, True, False]) # 0, 1 are ancestors. 2 is not ancestor of itself.
    assert jnp.all(ancestors == expected), f"Expected {expected}, got {ancestors}"

    # Ancestors of 1 should be {0}
    ancestors = find_ancestors_jax(mask, 1)
    expected = jnp.array([True, False, False])
    assert jnp.all(ancestors == expected), f"Expected {expected}, got {ancestors}"

    # Ancestors of 0 should be {}
    ancestors = find_ancestors_jax(mask, 0)
    expected = jnp.array([False, False, False])
    assert jnp.all(ancestors == expected), f"Expected {expected}, got {ancestors}"

def test_find_ancestors_branching():
    # 0 -> 2
    # 1 -> 2
    # Mask [child, parent]
    # 2 depends on 0 and 1
    mask = jnp.zeros((3, 3), dtype=jnp.int32)
    mask = mask.at[2, 0].set(1)
    mask = mask.at[2, 1].set(1)

    ancestors = find_ancestors_jax(mask, 2)
    expected = jnp.array([True, True, False])
    assert jnp.all(ancestors == expected), f"Expected {expected}, got {ancestors}"

def test_find_ancestors_deep_branching():
    # Diamond / Multi-path
    # 3 -> 1 -> 0
    # 4 -> 2 -> 0
    # Ancestors of 0 should be {1, 2, 3, 4}
    mask = jnp.zeros((5, 5), dtype=jnp.int32)
    mask = mask.at[0, 1].set(1)
    mask = mask.at[0, 2].set(1)
    mask = mask.at[1, 3].set(1)
    mask = mask.at[2, 4].set(1)

    ancestors = find_ancestors_jax(mask, 0)
    # Indices: 0, 1, 2, 3, 4
    expected = jnp.array([False, True, True, True, True])
    assert jnp.all(ancestors == expected), f"Expected {expected}, got {ancestors}"

def test_find_ancestors_cycle():
    # 0 -> 1 -> 0
    mask = jnp.zeros((2, 2), dtype=jnp.int32)
    mask = mask.at[1, 0].set(1) # 0 is parent of 1
    mask = mask.at[0, 1].set(1) # 1 is parent of 0

    # Ancestors of 0 should include 1.
    ancestors = find_ancestors_jax(mask, 0)
    assert ancestors[1] == True

def test_find_ancestors_self_loop():
    # 0 -> 0
    mask = jnp.zeros((1, 1), dtype=jnp.int32)
    mask = mask.at[0, 0].set(1)

    ancestors = find_ancestors_jax(mask, 0)
    # Should be empty because direct self-loops are ignored.
    assert ancestors[0] == False
