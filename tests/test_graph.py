import os

# select device
os.environ["JAX_PLATFORMS"] = "cpu"

import jax  # noqa: E402, F401
import jax.numpy as jnp  # noqa: E402

from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_jax_empty():
    mask = jnp.zeros((1, 1), dtype=jnp.bool_)
    ancestors = find_ancestors_jax(mask, 0)
    expected = jnp.array([False])
    assert (ancestors == expected).all()


def test_find_ancestors_jax_chain():
    # Chain: A -> B -> C -> D
    # Nodes: 0 -> 1 -> 2 -> 3
    # Rows are children, columns are parents
    mask = jnp.zeros((4, 4), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)
    mask = mask.at[2, 1].set(True)
    mask = mask.at[3, 2].set(True)

    # Ancestors of 3 (D) should be 0, 1, 2
    ancestors = find_ancestors_jax(mask, 3)
    expected = jnp.array([True, True, True, False])
    assert (ancestors == expected).all()

    # Ancestors of 2 (C) should be 0, 1
    ancestors = find_ancestors_jax(mask, 2)
    expected = jnp.array([True, True, False, False])
    assert (ancestors == expected).all()


def test_find_ancestors_jax_multiple_parents():
    # Graph: 0 -> 2, 1 -> 2
    mask = jnp.zeros((3, 3), dtype=jnp.bool_)
    mask = mask.at[2, 0].set(True)
    mask = mask.at[2, 1].set(True)

    # Ancestors of 2 should be 0, 1
    ancestors = find_ancestors_jax(mask, 2)
    expected = jnp.array([True, True, False])
    assert (ancestors == expected).all()


def test_find_ancestors_jax_disconnected():
    # Graph: 0 -> 1, 2 -> 3
    mask = jnp.zeros((4, 4), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)
    mask = mask.at[3, 2].set(True)

    # Ancestors of 1 should be 0
    ancestors = find_ancestors_jax(mask, 1)
    expected = jnp.array([True, False, False, False])
    assert (ancestors == expected).all()

    # Ancestors of 3 should be 2
    ancestors = find_ancestors_jax(mask, 3)
    expected = jnp.array([False, False, True, False])
    assert (ancestors == expected).all()


def test_find_ancestors_jax_cycle():
    # Cycle: 0 -> 1 -> 2 -> 0
    mask = jnp.zeros((3, 3), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)
    mask = mask.at[2, 1].set(True)
    mask = mask.at[0, 2].set(True)

    # In a cycle, the node will be marked as its own ancestor
    ancestors = find_ancestors_jax(mask, 0)
    expected = jnp.array([True, True, True])
    assert (ancestors == expected).all()


def test_find_ancestors_jax_self_loop():
    # Self loop: 0 -> 0
    mask = jnp.zeros((1, 1), dtype=jnp.bool_)
    mask = mask.at[0, 0].set(True)

    # Immediate self loops are ignored during traversal as cond includes (j != current_node)
    ancestors = find_ancestors_jax(mask, 0)
    expected = jnp.array([False])
    assert (ancestors == expected).all()
