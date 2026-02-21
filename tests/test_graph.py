# %%
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import pytest

from gensbi_examples.graph import find_ancestors_jax


def test_find_ancestors_jax_simple_chain():
    # 0 <- 1 <- 2
    num_nodes = 3
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.int32)
    # 1 -> 0
    mask = mask.at[0, 1].set(1)
    # 2 -> 1
    mask = mask.at[1, 2].set(1)

    ancestors = find_ancestors_jax(mask, 0)
    expected = jnp.array([False, True, True])
    assert jnp.all(ancestors == expected), f"Expected {expected}, got {ancestors}"


def test_find_ancestors_jax_multiple_parents():
    # 0 <- 1, 0 <- 2, 1 <- 3
    # This test is designed to catch the bug where multiple parents overwrite each other in the stack
    num_nodes = 4
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.int32)

    # 3 -> 1
    mask = mask.at[1, 3].set(1)
    # 1 -> 0
    mask = mask.at[0, 1].set(1)
    # 2 -> 0
    mask = mask.at[0, 2].set(1)

    ancestors = find_ancestors_jax(mask, 0)
    # Ancestors of 0 should be {1, 2, 3}
    expected = jnp.array([False, True, True, True])

    assert jnp.all(ancestors == expected), f"Expected {expected}, got {ancestors}"


def test_find_ancestors_jax_cycle():
    # 0 <- 1 <- 0 (cycle)
    num_nodes = 2
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.int32)
    # 1 -> 0
    mask = mask.at[0, 1].set(1)
    # 0 -> 1
    mask = mask.at[1, 0].set(1)

    ancestors = find_ancestors_jax(mask, 0)
    # Ancestors of 0: 1 (direct parent), which has parent 0.
    # Should contain 1. Should it contain 0?
    # Current implementation explicitly excludes 'current_node' from being added unless via cycle?
    # Actually: `value & (j != current_node) & (~is_ancestor[j])`
    # When processing 0: adds 1.
    # When processing 1: adds 0.
    # So ancestors of 0 should include 0 and 1.

    expected = jnp.array([True, True])
    assert jnp.all(ancestors == expected), f"Expected {expected}, got {ancestors}"


def test_find_ancestors_jax_disconnected():
    # 0, 1 (no edges)
    num_nodes = 2
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.int32)
    ancestors = find_ancestors_jax(mask, 0)
    expected = jnp.array([False, False])
    assert jnp.all(ancestors == expected), f"Expected {expected}, got {ancestors}"
