import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp  # noqa: E402

from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_jax_simple_dag():
    # A -> B -> C
    # 0 -> 1 -> 2
    # Rows are children, columns are parents
    mask = jnp.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=jnp.bool_)

    # Ancestors of 2 (C) should be 0 (A) and 1 (B)
    expected = jnp.array([True, True, False], dtype=jnp.bool_)
    result = find_ancestors_jax(mask, 2)
    assert (result == expected).all()


def test_find_ancestors_jax_multi_parent_dag():
    # 0 -> 1, 0 -> 2
    # 1 -> 3, 2 -> 3
    mask = jnp.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 1, 0]
    ], dtype=jnp.bool_)

    # Ancestors of 3 should be 0, 1, 2
    expected = jnp.array([True, True, True, False], dtype=jnp.bool_)
    result = find_ancestors_jax(mask, 3)
    assert (result == expected).all()


def test_find_ancestors_jax_no_ancestors():
    # 0 -> 1 -> 2
    mask = jnp.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=jnp.bool_)

    # Ancestors of 0 should be none
    expected = jnp.array([False, False, False], dtype=jnp.bool_)
    result = find_ancestors_jax(mask, 0)
    assert (result == expected).all()


def test_find_ancestors_jax_disconnected():
    # Disconnected graph (all zeros)
    mask = jnp.zeros((3, 3), dtype=jnp.bool_)

    # Ancestors of 1 should be none
    expected = jnp.array([False, False, False], dtype=jnp.bool_)
    result = find_ancestors_jax(mask, 1)
    assert (result == expected).all()


def test_find_ancestors_jax_cycle():
    # 0 -> 1 -> 0
    mask = jnp.array([
        [0, 1],
        [1, 0]
    ], dtype=jnp.bool_)

    # Ancestors of 0 should be 1 and 0
    expected = jnp.array([True, True], dtype=jnp.bool_)
    result = find_ancestors_jax(mask, 0)
    assert (result == expected).all()


def test_find_ancestors_jax_self_loop():
    # 0 -> 0
    mask = jnp.array([
        [1]
    ], dtype=jnp.bool_)

    # find_ancestors_jax explicitly ignores immediate self-loops
    # cond = value & (j != current_node) & (~is_ancestor[j])
    expected = jnp.array([False], dtype=jnp.bool_)
    result = find_ancestors_jax(mask, 0)
    assert (result == expected).all()
