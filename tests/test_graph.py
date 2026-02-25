
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402, F401
from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_linear_ancestors():
    # 0 -> 1 -> 2
    # Rows are children, columns are parents
    # mask[1, 0] = 1
    # mask[2, 1] = 1
    mask = jnp.zeros((3, 3), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)
    mask = mask.at[2, 1].set(True)

    # Query 2
    ancestors = find_ancestors_jax(mask, 2)
    # 0, 1 are ancestors. 2 is not.
    expected = jnp.array([True, True, False], dtype=jnp.bool_)

    assert jnp.array_equal(ancestors, expected), \
        f"Expected {expected}, got {ancestors}"


def test_diamond_ancestors():
    # 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
    # mask[1, 0] = 1
    # mask[2, 0] = 1
    # mask[3, 1] = 1
    # mask[3, 2] = 1
    mask = jnp.zeros((4, 4), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)
    mask = mask.at[2, 0].set(True)
    mask = mask.at[3, 1].set(True)
    mask = mask.at[3, 2].set(True)

    # Query 3
    ancestors = find_ancestors_jax(mask, 3)
    # Expected: 0, 1, 2 are ancestors. 3 is not.
    expected = jnp.array([True, True, True, False], dtype=jnp.bool_)

    assert jnp.array_equal(ancestors, expected), \
        f"Expected {expected}, got {ancestors}"


def test_branching_ancestors():
    # 0 -> 1 -> 3
    # 2 -> 3
    # Query 3. Parents 1, 2.
    # If 1 is overwritten by 2 in stack, 0 will not be found.
    mask = jnp.zeros((4, 4), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)
    mask = mask.at[3, 1].set(True)
    mask = mask.at[3, 2].set(True)

    # Query 3
    ancestors = find_ancestors_jax(mask, 3)
    # Expected: 0, 1, 2 are ancestors.
    expected = jnp.array([True, True, True, False], dtype=jnp.bool_)

    assert jnp.array_equal(ancestors, expected), \
        (f"Expected {expected}, got {ancestors}. "
         "0 likely missed if 1 was overwritten in stack.")


def test_disconnected_ancestors():
    # 0 -> 1, 2
    # mask[1, 0] = 1
    mask = jnp.zeros((3, 3), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)

    # Query 2
    ancestors = find_ancestors_jax(mask, 2)
    expected = jnp.array([False, False, False], dtype=jnp.bool_)

    assert jnp.array_equal(ancestors, expected), \
        f"Expected {expected}, got {ancestors}"


def test_cycle_ancestors():
    # 0 -> 1 -> 0
    # mask[1, 0] = 1
    # mask[0, 1] = 1
    mask = jnp.zeros((2, 2), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)
    mask = mask.at[0, 1].set(True)

    # Query 0
    ancestors = find_ancestors_jax(mask, 0)
    # 0 is ancestor of itself in a cycle
    expected = jnp.array([True, True], dtype=jnp.bool_)

    assert jnp.array_equal(ancestors, expected), \
        f"Expected {expected}, got {ancestors}"


if __name__ == "__main__":
    pytest.main([__file__])
