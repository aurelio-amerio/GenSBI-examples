import os

# select device
os.environ["JAX_PLATFORMS"] = "cpu" # noqa: E402

import jax
import jax.numpy as jnp
import pytest
from gensbi_examples.graph import find_ancestors_jax

def test_find_ancestors_chain():
    # 0 -> 1 -> 2
    # Ancestors of 2 should be {0, 1}
    mask = jnp.zeros((3, 3), dtype=jnp.int32)
    # mask[child, parent] = 1
    mask = mask.at[1, 0].set(1)
    mask = mask.at[2, 1].set(1)

    ancestors = find_ancestors_jax(mask, 2)
    expected = jnp.array([True, True, False], dtype=bool)
    assert jnp.array_equal(ancestors, expected)

    ancestors = find_ancestors_jax(mask, 1)
    expected = jnp.array([True, False, False], dtype=bool)
    assert jnp.array_equal(ancestors, expected)

    ancestors = find_ancestors_jax(mask, 0)
    expected = jnp.array([False, False, False], dtype=bool)
    assert jnp.array_equal(ancestors, expected)

def test_find_ancestors_branching():
    # 0 -> 2
    # 1 -> 2
    # 2 -> 3
    # Ancestors of 3 should be {0, 1, 2}
    mask = jnp.zeros((4, 4), dtype=jnp.int32)
    mask = mask.at[2, 0].set(1)
    mask = mask.at[2, 1].set(1)
    mask = mask.at[3, 2].set(1)

    ancestors = find_ancestors_jax(mask, 3)
    expected = jnp.array([True, True, True, False], dtype=bool)
    assert jnp.array_equal(ancestors, expected)

def test_find_ancestors_cycle():
    # 0 <-> 1
    # 0 -> 1 -> 0
    # Ancestors of 0 should be {1} (and 0 itself via cycle)
    mask = jnp.zeros((2, 2), dtype=jnp.int32)
    mask = mask.at[1, 0].set(1)
    mask = mask.at[0, 1].set(1)

    ancestors = find_ancestors_jax(mask, 0)
    expected = jnp.array([True, True], dtype=bool)
    assert jnp.array_equal(ancestors, expected)

def test_find_ancestors_self_loop():
    # 0 -> 0
    # Ancestors of 0 should be {} (empty) because immediate self-loops are ignored.
    mask = jnp.zeros((1, 1), dtype=jnp.int32)
    mask = mask.at[0, 0].set(1)

    ancestors = find_ancestors_jax(mask, 0)
    expected = jnp.array([False], dtype=bool)
    assert jnp.array_equal(ancestors, expected)
