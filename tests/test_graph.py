import os
import pytest

# Select device
os.environ["JAX_PLATFORMS"] = "cpu"

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_jax_linear():
    # 0 -> 1 -> 2
    # Rows are children, columns are parents
    # mask[i, j] = 1 means j -> i
    mask = jnp.array([
        [0, 0, 0],  # Node 0 has no parents
        [1, 0, 0],  # Node 1 has parent 0
        [0, 1, 0],  # Node 2 has parent 1
    ], dtype=jnp.bool_)

    # Ancestors of 2 should be 0, 1
    is_ancestor = find_ancestors_jax(mask, 2)
    expected = jnp.array([True, True, False], dtype=jnp.bool_)
    assert jnp.all(is_ancestor == expected)


def test_find_ancestors_jax_branching():
    # 0 -> 1 -> 3
    # 0 -> 2 -> 3
    mask = jnp.array([
        [0, 0, 0, 0],  # 0 has no parents
        [1, 0, 0, 0],  # 1 has parent 0
        [1, 0, 0, 0],  # 2 has parent 0
        [0, 1, 1, 0],  # 3 has parents 1, 2
    ], dtype=jnp.bool_)

    # Ancestors of 3 should be 0, 1, 2
    is_ancestor = find_ancestors_jax(mask, 3)
    expected = jnp.array([True, True, True, False], dtype=jnp.bool_)
    assert jnp.all(is_ancestor == expected)


def test_find_ancestors_jax_no_ancestors():
    # 0 -> 1
    mask = jnp.array([
        [0, 0],
        [1, 0],
    ], dtype=jnp.bool_)

    # Ancestors of 0 should be none
    is_ancestor = find_ancestors_jax(mask, 0)
    expected = jnp.array([False, False], dtype=jnp.bool_)
    assert jnp.all(is_ancestor == expected)


def test_find_ancestors_jax_multiple_components():
    # 0 -> 1 -> 2
    # 3 -> 4
    mask = jnp.array([
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
    ], dtype=jnp.bool_)

    # Ancestors of 2
    is_ancestor = find_ancestors_jax(mask, 2)
    expected = jnp.array([True, True, False, False, False], dtype=jnp.bool_)
    assert jnp.all(is_ancestor == expected)

    # Ancestors of 4
    is_ancestor = find_ancestors_jax(mask, 4)
    expected = jnp.array([False, False, False, True, False], dtype=jnp.bool_)
    assert jnp.all(is_ancestor == expected)


def test_find_ancestors_jax_cycle():
    # Cycle: 0 -> 1 -> 2 -> 0
    mask = jnp.array([
        [0, 0, 1],  # 0 has parent 2
        [1, 0, 0],  # 1 has parent 0
        [0, 1, 0],  # 2 has parent 1
    ], dtype=jnp.bool_)

    # Ancestors of 0 should be 0, 1, 2
    # Because 2 is parent, 1 is parent of 2, 0 is parent of 1
    # Thus, 0 is its own ancestor.
    is_ancestor = find_ancestors_jax(mask, 0)
    expected = jnp.array([True, True, True], dtype=jnp.bool_)
    assert jnp.all(is_ancestor == expected)


def test_find_ancestors_jax_complex():
    # 0 -> 2
    # 1 -> 2
    # 2 -> 3
    # 2 -> 4
    # 3 -> 5
    # 4 -> 5
    # 5 -> 6
    # 7 -> 6
    mask = jnp.array([
        [0, 0, 0, 0, 0, 0, 0, 0],  # 0
        [0, 0, 0, 0, 0, 0, 0, 0],  # 1
        [1, 1, 0, 0, 0, 0, 0, 0],  # 2 has parents 0, 1
        [0, 0, 1, 0, 0, 0, 0, 0],  # 3 has parent 2
        [0, 0, 1, 0, 0, 0, 0, 0],  # 4 has parent 2
        [0, 0, 0, 1, 1, 0, 0, 0],  # 5 has parents 3, 4
        [0, 0, 0, 0, 0, 1, 0, 1],  # 6 has parents 5, 7
        [0, 0, 0, 0, 0, 0, 0, 0],  # 7
    ], dtype=jnp.bool_)

    # Ancestors of 6 should be 0, 1, 2, 3, 4, 5, 7
    is_ancestor = find_ancestors_jax(mask, 6)
    expected = jnp.array([True, True, True, True, True, True, False, True], dtype=jnp.bool_)
    assert jnp.all(is_ancestor == expected)
