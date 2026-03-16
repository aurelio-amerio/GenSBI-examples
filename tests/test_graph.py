import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp  # noqa: E402

from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_linear():
    # 0 -> 1 -> 2
    # Rows are children, cols are parents
    # M_ij = 1 means j -> i
    mask = jnp.array([
        [0, 0, 0],  # 0 has no parents
        [1, 0, 0],  # 1 has parent 0
        [0, 1, 0],  # 2 has parent 1
    ], dtype=jnp.bool_)

    # Ancestors of 2 should be 0 and 1
    expected = jnp.array([True, True, False], dtype=jnp.bool_)
    result = find_ancestors_jax(mask, 2)
    assert (result == expected).all()

    # Ancestors of 1 should be 0
    expected = jnp.array([True, False, False], dtype=jnp.bool_)
    result = find_ancestors_jax(mask, 1)
    assert (result == expected).all()

    # Ancestors of 0 should be empty
    expected = jnp.array([False, False, False], dtype=jnp.bool_)
    result = find_ancestors_jax(mask, 0)
    assert (result == expected).all()


def test_find_ancestors_branching():
    # 0 -> 2
    # 1 -> 2
    # 2 -> 3
    mask = jnp.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 1, 0, 0],  # 2 has parents 0 and 1
        [0, 0, 1, 0],  # 3 has parent 2
    ], dtype=jnp.bool_)

    # Ancestors of 3 should be 0, 1, 2
    expected = jnp.array([True, True, True, False], dtype=jnp.bool_)
    result = find_ancestors_jax(mask, 3)
    assert (result == expected).all()

    # Ancestors of 2 should be 0, 1
    expected = jnp.array([True, True, False, False], dtype=jnp.bool_)
    result = find_ancestors_jax(mask, 2)
    assert (result == expected).all()


def test_find_ancestors_disconnected():
    # 0 -> 1
    # 2 -> 3
    mask = jnp.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0],
    ], dtype=jnp.bool_)

    # Ancestors of 1 should be 0
    expected = jnp.array([True, False, False, False], dtype=jnp.bool_)
    result = find_ancestors_jax(mask, 1)
    assert (result == expected).all()

    # Ancestors of 3 should be 2
    expected = jnp.array([False, False, True, False], dtype=jnp.bool_)
    result = find_ancestors_jax(mask, 3)
    assert (result == expected).all()


def test_find_ancestors_cycle_and_self_loops():
    # 0 -> 1 -> 2 -> 0 (cycle)
    # 3 -> 3 (self loop)
    mask = jnp.array([
        [0, 0, 1, 0],  # 0 has parent 2
        [1, 0, 0, 0],  # 1 has parent 0
        [0, 1, 0, 0],  # 2 has parent 1
        [0, 0, 0, 1],  # 3 has parent 3
    ], dtype=jnp.bool_)

    # Ancestors of 0 should be 0, 1, 2 (it's in a cycle, so it is its own ancestor via larger cycle)
    expected = jnp.array([True, True, True, False], dtype=jnp.bool_)
    result = find_ancestors_jax(mask, 0)
    assert (result == expected).all()

    # Ancestors of 3 should be empty (immediate self-loops are ignored in traversal logic)
    expected = jnp.array([False, False, False, False], dtype=jnp.bool_)
    result = find_ancestors_jax(mask, 3)
    assert (result == expected).all()
