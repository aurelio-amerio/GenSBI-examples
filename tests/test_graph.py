import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp  # noqa: E402

from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_jax_linear():
    # Adjacency matrix: rows are children, columns are parents.
    # M[i, j] = 1 means edge j -> i.
    # Linear: 0 -> 1 -> 2
    mask = jnp.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=jnp.bool_)

    ancestors = find_ancestors_jax(mask, 2)
    expected = jnp.array([True, True, False], dtype=jnp.bool_)
    assert jnp.all(ancestors == expected)


def test_find_ancestors_jax_branching():
    # Branching: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
    mask = jnp.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 1, 0]
    ], dtype=jnp.bool_)

    ancestors = find_ancestors_jax(mask, 3)
    expected = jnp.array([True, True, True, False], dtype=jnp.bool_)
    assert jnp.all(ancestors == expected)


def test_find_ancestors_jax_no_ancestors():
    # No incoming edges: 0
    mask = jnp.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=jnp.bool_)

    ancestors = find_ancestors_jax(mask, 0)
    expected = jnp.array([False, False, False], dtype=jnp.bool_)
    assert jnp.all(ancestors == expected)


def test_find_ancestors_jax_cycle():
    # Cycle: 0 -> 1 -> 2 -> 0
    mask = jnp.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=jnp.bool_)

    ancestors = find_ancestors_jax(mask, 0)
    expected = jnp.array([True, True, True], dtype=jnp.bool_)
    assert jnp.all(ancestors == expected)


def test_find_ancestors_jax_integer_matrix():
    # Integer adjacency matrix
    mask = jnp.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=jnp.int32)

    ancestors = find_ancestors_jax(mask, 2)
    expected = jnp.array([True, True, False], dtype=jnp.bool_)
    assert jnp.all(ancestors == expected)
