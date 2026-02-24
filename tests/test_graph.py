# %%
import os

# select device
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import pytest
from gensbi_examples.graph import find_ancestors_jax

# Adjacency matrix convention in find_ancestors_jax seems to be:
# mask[child, parent] = 1 (based on reproduction)
# Or mask[i, j] = 1 means j -> i.

def test_find_ancestors_diamond():
    # Nodes:
    # 0: GP1
    # 1: GP2
    # 2: P1 (parents: GP1)
    # 3: P2 (parents: GP2)
    # 4: C (parents: P1, P2)

    num_nodes = 5
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.int32)

    # mask[child, parent] = 1
    mask = mask.at[2, 0].set(1) # GP1 -> P1
    mask = mask.at[3, 1].set(1) # GP2 -> P2
    mask = mask.at[4, 2].set(1) # P1 -> C
    mask = mask.at[4, 3].set(1) # P2 -> C

    # Find ancestors of C (node 4)
    ancestors_mask = find_ancestors_jax(mask, 4)
    ancestors_indices = jnp.where(ancestors_mask)[0]

    expected_ancestors = jnp.array([0, 1, 2, 3])

    # Check if all expected ancestors are found
    missing = jnp.setdiff1d(expected_ancestors, ancestors_indices)
    assert len(missing) == 0, f"Missing ancestors: {missing}"

    # Check no extra ancestors
    extra = jnp.setdiff1d(ancestors_indices, expected_ancestors)
    assert len(extra) == 0, f"Extra ancestors: {extra}"

def test_find_ancestors_linear():
    # 0 -> 1 -> 2 -> 3
    num_nodes = 4
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.int32)
    mask = mask.at[1, 0].set(1)
    mask = mask.at[2, 1].set(1)
    mask = mask.at[3, 2].set(1)

    ancestors_mask = find_ancestors_jax(mask, 3)
    ancestors_indices = jnp.where(ancestors_mask)[0]
    expected_ancestors = jnp.array([0, 1, 2])

    missing = jnp.setdiff1d(expected_ancestors, ancestors_indices)
    assert len(missing) == 0, f"Missing ancestors: {missing}"
    extra = jnp.setdiff1d(ancestors_indices, expected_ancestors)
    assert len(extra) == 0, f"Extra ancestors: {extra}"

def test_find_ancestors_cycle():
    # 0 -> 1 -> 2 -> 0
    num_nodes = 3
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.int32)
    mask = mask.at[1, 0].set(1)
    mask = mask.at[2, 1].set(1)
    mask = mask.at[0, 2].set(1)

    # Ancestors of 0 should be {0, 1, 2} eventually if we traverse full cycle
    # But usually ancestors means strict ancestors?
    # If 0 -> 1 -> 2 -> 0.
    # Parents of 0: {2}.
    # Parents of 2: {1}.
    # Parents of 1: {0}.
    # Parents of 0: {2} (visited).
    # So {1, 2, 0} (or just {1, 2} depending on if self is included).

    ancestors_mask = find_ancestors_jax(mask, 0)
    ancestors_indices = jnp.where(ancestors_mask)[0]

    # Since it's a cycle, everything is an ancestor of everything.
    # The question is whether 0 itself is included.
    # Based on traversal logic, if 0 is reached via 2 -> 0, it should be marked.

    expected_ancestors = jnp.array([0, 1, 2])

    missing = jnp.setdiff1d(expected_ancestors, ancestors_indices)
    assert len(missing) == 0, f"Missing ancestors in cycle: {missing}"

def test_find_ancestors_disconnected():
    # 0 -> 1   2 -> 3
    num_nodes = 4
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.int32)
    mask = mask.at[1, 0].set(1)
    mask = mask.at[3, 2].set(1)

    ancestors_mask = find_ancestors_jax(mask, 1)
    ancestors_indices = jnp.where(ancestors_mask)[0]
    expected_ancestors = jnp.array([0])

    missing = jnp.setdiff1d(expected_ancestors, ancestors_indices)
    assert len(missing) == 0, f"Missing ancestors: {missing}"
    extra = jnp.setdiff1d(ancestors_indices, expected_ancestors)
    assert len(extra) == 0, f"Extra ancestors: {extra}"

    ancestors_mask_3 = find_ancestors_jax(mask, 3)
    ancestors_indices_3 = jnp.where(ancestors_mask_3)[0]
    expected_ancestors_3 = jnp.array([2])

    missing_3 = jnp.setdiff1d(expected_ancestors_3, ancestors_indices_3)
    assert len(missing_3) == 0, f"Missing ancestors: {missing_3}"
    extra_3 = jnp.setdiff1d(ancestors_indices_3, expected_ancestors_3)
    assert len(extra_3) == 0, f"Extra ancestors: {extra_3}"

def test_find_ancestors_self_loop():
    # 0 -> 0. 1 -> 0.
    # Ancestors of 0: {0, 1}.
    num_nodes = 2
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.int32)
    mask = mask.at[0, 0].set(1)
    mask = mask.at[0, 1].set(1)

    ancestors_mask = find_ancestors_jax(mask, 0)
    ancestors_indices = jnp.where(ancestors_mask)[0]

    # With self loop, 0 is parent of 0.
    # 1 is parent of 0.
    # So both should be found.
    expected_ancestors = jnp.array([0, 1])

    missing = jnp.setdiff1d(expected_ancestors, ancestors_indices)
    assert len(missing) == 0, f"Missing ancestors: {missing}"
