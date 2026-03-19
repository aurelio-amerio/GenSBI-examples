import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp  # noqa: E402

from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_jax_simple():
    """Test find_ancestors_jax with a simple chain graph 0 -> 1 -> 2."""
    # Adjacency matrix: rows are children, columns are parents
    # M[i, j] = 1 means j -> i
    mask = jnp.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=jnp.bool_)

    # Ancestors of node 2 should be 0 and 1
    ans = find_ancestors_jax(mask, 2)
    assert (ans == jnp.array([True, True, False])).all()

    # Ancestors of node 1 should be 0
    ans = find_ancestors_jax(mask, 1)
    assert (ans == jnp.array([True, False, False])).all()

    # Ancestors of node 0 should be empty
    ans = find_ancestors_jax(mask, 0)
    assert (ans == jnp.array([False, False, False])).all()


def test_find_ancestors_jax_multiple_parents():
    """Test find_ancestors_jax with a node having multiple parents."""
    # 0 -> 2
    # 1 -> 2
    # 2 -> 3
    mask = jnp.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 0]
    ], dtype=jnp.bool_)

    ans = find_ancestors_jax(mask, 3)
    assert (ans == jnp.array([True, True, True, False])).all()


def test_find_ancestors_jax_disconnected():
    """Test find_ancestors_jax with a disconnected graph."""
    mask = jnp.zeros((3, 3), dtype=jnp.bool_)

    ans = find_ancestors_jax(mask, 2)
    assert (ans == jnp.array([False, False, False])).all()


def test_find_ancestors_jax_cycle():
    """Test find_ancestors_jax with a cycle."""
    # 0 -> 1 -> 2 -> 0
    mask = jnp.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=jnp.bool_)

    ans = find_ancestors_jax(mask, 2)
    assert (ans == jnp.array([True, True, True])).all()
