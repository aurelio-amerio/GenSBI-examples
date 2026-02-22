# %%
import os

# select device
os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp  # noqa: E402
from gensbi_examples.graph import find_ancestors_jax  # noqa: E402

# We should make sure we disable preallocation
# for deterministic tests on CPU if needed
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def test_find_ancestors_jax_linear():
    """
    Test linear graph: 0 -> 1 -> 2
    Ancestors of 2: {0, 1}
    """
    num_nodes = 3
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.int32)
    # 0 -> 1 => mask[1, 0] = 1 (row=child, col=parent)
    mask = mask.at[1, 0].set(1)
    # 1 -> 2 => mask[2, 1] = 1
    mask = mask.at[2, 1].set(1)

    ancestors = find_ancestors_jax(mask, 2)

    # 0, 1 are ancestors. 2 is not (unless self loop)
    expected = jnp.array([True, True, False], dtype=jnp.bool_)

    assert jnp.all(ancestors == expected), \
        f"Expected {expected}, got {ancestors}"


def test_find_ancestors_jax_branching():
    """
    Test branching (converging) graph:
    0 -> 2
    1 -> 2
    Ancestors of 2: {0, 1}
    """
    num_nodes = 3
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.int32)
    # 0 -> 2 => mask[2, 0] = 1
    mask = mask.at[2, 0].set(1)
    # 1 -> 2 => mask[2, 1] = 1
    mask = mask.at[2, 1].set(1)

    ancestors = find_ancestors_jax(mask, 2)

    expected = jnp.array([True, True, False], dtype=jnp.bool_)

    assert jnp.all(ancestors == expected), \
        f"Expected {expected}, got {ancestors}"


def test_find_ancestors_jax_diamond():
    """
    Test diamond graph (BUG REPRODUCTION):
    4 -> 1 -> 3
    0 -> 2 -> 3
    Ancestors of 3: {0, 1, 2, 4}
    """
    num_nodes = 5
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.int32)

    # 4 -> 1 => mask[1, 4] = 1
    mask = mask.at[1, 4].set(1)
    # 1 -> 3 => mask[3, 1] = 1
    mask = mask.at[3, 1].set(1)
    # 0 -> 2 => mask[2, 0] = 1
    mask = mask.at[2, 0].set(1)
    # 2 -> 3 => mask[3, 2] = 1
    mask = mask.at[3, 2].set(1)

    ancestors = find_ancestors_jax(mask, 3)

    # Expected: 0, 1, 2, 4 are True. 3 is False.
    expected = jnp.array([True, True, True, False, True], dtype=jnp.bool_)

    assert jnp.all(ancestors == expected), \
        f"Expected {expected}, got {ancestors}"


def test_find_ancestors_jax_cycle():
    """
    Test cycle handling (should not loop forever):
    0 -> 1 -> 0
    Ancestors of 0: {1} (and implicitly 0 if cycle includes it?
    Logic usually excludes self unless self-loop)
    But 0 depends on 1, 1 depends on 0.
    """
    num_nodes = 2
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.int32)
    # 0 -> 1 => mask[1, 0] = 1
    mask = mask.at[1, 0].set(1)
    # 1 -> 0 => mask[0, 1] = 1
    mask = mask.at[0, 1].set(1)

    ancestors = find_ancestors_jax(mask, 0)

    # 0 depends on 1. 1 depends on 0.
    # Logic:
    # Stack: [0]
    # Pop 0. Parents: 1. Mark 1. Stack: [1].
    # Pop 1. Parents: 0. 0 is visited? No, self not marked visited initially?
    # Usually `is_ancestor` initialized to False.
    # If we mark 0 as ancestor?
    # If 0 is marked, we stop.
    # If 0 is NOT marked (it is `node`), but we check `~is_ancestor[j]`.
    # And we check `j != node`.
    # So if parent is `node`, we skip it.
    # So we don't mark 0.
    # Result: {1}.

    # Correction: If cycle exists 0->1->0.
    # Processing 0: adds 1.
    # Processing 1: adds 0 (since 0 != current_node(1)).
    # So 0 IS marked as ancestor.

    expected = jnp.array([True, True], dtype=jnp.bool_)
    assert jnp.all(ancestors == expected), \
        f"Expected {expected}, got {ancestors}"
