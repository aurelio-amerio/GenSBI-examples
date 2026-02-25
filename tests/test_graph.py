import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402, F401
from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_linear():
    # 0 -> 1 -> 2
    # Rows are children, cols are parents
    # mask[1, 0] = 1
    # mask[2, 1] = 1
    mask = jnp.zeros((3, 3), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)
    mask = mask.at[2, 1].set(True)

    # Ancestors of 2
    ancestors = find_ancestors_jax(mask, 2)
    expected = jnp.array([True, True, False], dtype=jnp.bool_)  # 0 and 1 are ancestors

    assert jnp.array_equal(ancestors, expected)


def test_find_ancestors_diamond():
    # 0 -> 1 -> 3
    # 0 -> 2 -> 3
    # mask[1, 0] = 1
    # mask[2, 0] = 1
    # mask[3, 1] = 1
    # mask[3, 2] = 1
    mask = jnp.zeros((4, 4), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)
    mask = mask.at[2, 0].set(True)
    mask = mask.at[3, 1].set(True)
    mask = mask.at[3, 2].set(True)

    # Ancestors of 3 should be 0, 1, 2
    ancestors = find_ancestors_jax(mask, 3)
    expected = jnp.array([True, True, True, False], dtype=jnp.bool_)

    assert jnp.array_equal(ancestors, expected)


def test_find_ancestors_disconnected():
    # 0   1
    mask = jnp.zeros((2, 2), dtype=jnp.bool_)

    # Ancestors of 0
    ancestors = find_ancestors_jax(mask, 0)
    expected = jnp.array([False, False], dtype=jnp.bool_)

    assert jnp.array_equal(ancestors, expected)


def test_find_ancestors_cycle():
    # 0 -> 1 -> 0
    mask = jnp.zeros((2, 2), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)
    mask = mask.at[0, 1].set(True)

    # Ancestors of 0
    # 1 is parent. 1's parent is 0.
    # So ancestors are {1, 0}
    ancestors = find_ancestors_jax(mask, 0)
    expected = jnp.array([True, True], dtype=jnp.bool_)

    assert jnp.array_equal(ancestors, expected)


def test_find_ancestors_branching_parents():
    # Test specifically for multiple parents
    # 1 -> 0
    # 2 -> 0
    mask = jnp.zeros((3, 3), dtype=jnp.bool_)
    mask = mask.at[0, 1].set(True)
    mask = mask.at[0, 2].set(True)

    ancestors = find_ancestors_jax(mask, 0)
    expected = jnp.array([False, True, True], dtype=jnp.bool_)

    assert jnp.array_equal(ancestors, expected)


def test_find_ancestors_missing_branch():
    # 4 -> 1 -> 3
    # 5 -> 2 -> 3
    # Query 3. Expected ancestors: {1, 2, 4, 5}
    # Nodes: 0..5
    mask = jnp.zeros((6, 6), dtype=jnp.bool_)
    mask = mask.at[1, 4].set(True)
    mask = mask.at[2, 5].set(True)
    mask = mask.at[3, 1].set(True)
    mask = mask.at[3, 2].set(True)

    ancestors = find_ancestors_jax(mask, 3)
    # Expected: 1, 2, 4, 5 are True. 0, 3 are False.
    expected = jnp.array([False, True, True, False, True, True], dtype=jnp.bool_)

    assert jnp.array_equal(ancestors, expected)
