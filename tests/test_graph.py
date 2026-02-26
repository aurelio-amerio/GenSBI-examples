
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import pytest
from gensbi_examples.graph import find_ancestors_jax

# Disable strict JAX checking for this test file as we might run on CPU without full accelerator support
jax.config.update("jax_platform_name", "cpu")

def test_find_ancestors_simple_chain():
    # 0 -> 1 -> 2
    # Ancestors of 2 should be {1, 0}
    # Ancestors of 1 should be {0}
    # Ancestors of 0 should be {}

    # Mask: rows=children, cols=parents
    # 1 has parent 0 => mask[1, 0] = 1
    # 2 has parent 1 => mask[2, 1] = 1

    num_nodes = 3
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.int32)
    mask = mask.at[1, 0].set(1)
    mask = mask.at[2, 1].set(1)

    ancestors_2 = find_ancestors_jax(mask, 2)
    assert ancestors_2[1]
    assert ancestors_2[0]
    assert not ancestors_2[2]
    assert jnp.sum(ancestors_2) == 2

    ancestors_1 = find_ancestors_jax(mask, 1)
    assert ancestors_1[0]
    assert not ancestors_1[1]
    assert not ancestors_1[2]
    assert jnp.sum(ancestors_1) == 1

def test_find_ancestors_branching_bug_repro():
    # Graph structure that exposed the stack overwrite bug:
    # 5 -> 1 -> 3
    #      2 -> 3
    #           3 -> 4

    # Mask construction:
    # 1 -> 3 => mask[3, 1] = 1
    # 2 -> 3 => mask[3, 2] = 1
    # 5 -> 1 => mask[1, 5] = 1
    # 3 -> 4 => mask[4, 3] = 1

    num_nodes = 6
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.int32)
    mask = mask.at[1, 5].set(1)
    mask = mask.at[3, 1].set(1)
    mask = mask.at[3, 2].set(1)
    mask = mask.at[4, 3].set(1)

    # Find ancestors of 4
    # Expected: {3, 1, 2, 5}
    ancestors = find_ancestors_jax(mask, 4)

    expected_indices = {1, 2, 3, 5}
    found_indices = set(int(i) for i in jnp.where(ancestors)[0])

    assert found_indices == expected_indices, f"Expected {expected_indices}, found {found_indices}"

def test_find_ancestors_multiple_parents_simple():
    # 0 -> 2
    # 1 -> 2
    # Ancestors of 2: {0, 1}

    num_nodes = 3
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.int32)
    mask = mask.at[2, 0].set(1)
    mask = mask.at[2, 1].set(1)

    ancestors = find_ancestors_jax(mask, 2)
    assert ancestors[0]
    assert ancestors[1]
    assert not ancestors[2]
    assert jnp.sum(ancestors) == 2

def test_find_ancestors_cycle():
    # 0 -> 1 -> 0
    # Ancestors of 0 should include 1 and 0 (due to cycle)

    num_nodes = 2
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.int32)
    mask = mask.at[1, 0].set(1)
    mask = mask.at[0, 1].set(1)

    ancestors = find_ancestors_jax(mask, 0)
    assert ancestors[1]
    assert ancestors[0] # Because 0 -> 1 -> 0, so 0 is ancestor of itself via 1

    ancestors_1 = find_ancestors_jax(mask, 1)
    assert ancestors_1[0]
    assert ancestors_1[1]

def test_find_ancestors_self_loop():
    # 0 -> 0
    # Ancestors of 0 should be empty (since direct self-loops are ignored in code)

    num_nodes = 1
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.int32)
    mask = mask.at[0, 0].set(1)

    ancestors = find_ancestors_jax(mask, 0)
    assert not ancestors[0]
    assert jnp.sum(ancestors) == 0

def test_find_ancestors_disconnected():
    # 0   1
    # Ancestors of 0: {}

    num_nodes = 2
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.int32)

    ancestors = find_ancestors_jax(mask, 0)
    assert jnp.sum(ancestors) == 0
