
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import pytest
from gensbi_examples.graph import find_ancestors_jax

def test_find_ancestors_linear():
    # 2 -> 1 -> 0
    # Adjacency (rows=children, cols=parents)
    mask = jnp.array([
        [0, 1, 0], # 0 has parent 1
        [0, 0, 1], # 1 has parent 2
        [0, 0, 0]  # 2 has no parents
    ], dtype=jnp.bool_)

    ancestors = find_ancestors_jax(mask, 0)
    expected = jnp.array([False, True, True], dtype=jnp.bool_)
    assert jnp.array_equal(ancestors, expected)

def test_find_ancestors_diamond():
    # 3 -> 1 -> 0
    # 3 -> 2 -> 0
    # Ancestors of 0 should be 1, 2, 3
    mask = jnp.array([
        [0, 1, 1, 0], # 0 has parents 1, 2
        [0, 0, 0, 1], # 1 has parent 3
        [0, 0, 0, 1], # 2 has parent 3
        [0, 0, 0, 0]  # 3 has no parents
    ], dtype=jnp.bool_)

    ancestors = find_ancestors_jax(mask, 0)
    expected = jnp.array([False, True, True, True], dtype=jnp.bool_)
    assert jnp.array_equal(ancestors, expected)

def test_find_ancestors_branching_unique():
    # 3 -> 1 -> 0
    # 4 -> 2 -> 0
    # Ancestors of 0 should be 1, 2, 3, 4
    mask = jnp.array([
        [0, 1, 1, 0, 0], # 0 has parents 1, 2
        [0, 0, 0, 1, 0], # 1 has parent 3
        [0, 0, 0, 0, 1], # 2 has parent 4
        [0, 0, 0, 0, 0], # 3 has no parents
        [0, 0, 0, 0, 0]  # 4 has no parents
    ], dtype=jnp.bool_)

    ancestors = find_ancestors_jax(mask, 0)
    expected = jnp.array([False, True, True, True, True], dtype=jnp.bool_)
    assert jnp.array_equal(ancestors, expected)

def test_find_ancestors_cycle():
    # 0 -> 1 -> 0
    mask = jnp.array([
        [0, 1], # 0 has parent 1
        [1, 0]  # 1 has parent 0
    ], dtype=jnp.bool_)

    # Ancestors of 0: 1. 1 has parent 0. 0 has parent 1.
    # Should include 1. Might include 0 if implementation marks it.
    ancestors = find_ancestors_jax(mask, 0)
    # 1 is ancestor.
    assert ancestors[1] == True

def test_find_ancestors_disconnected():
    # 1 -> 0
    # 2 -> 3
    mask = jnp.array([
        [0, 1, 0, 0], # 0 has parent 1
        [0, 0, 0, 0], # 1 has no parents
        [0, 0, 0, 0], # 2 has no parents
        [0, 0, 1, 0]  # 3 has parent 2
    ], dtype=jnp.bool_)

    ancestors_0 = find_ancestors_jax(mask, 0)
    expected_0 = jnp.array([False, True, False, False], dtype=jnp.bool_)
    assert jnp.array_equal(ancestors_0, expected_0)

    ancestors_3 = find_ancestors_jax(mask, 3)
    expected_3 = jnp.array([False, False, True, False], dtype=jnp.bool_)
    assert jnp.array_equal(ancestors_3, expected_3)
