import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax  # noqa: E402, F401
import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402, F401

from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_jax_no_ancestors():
    # Node 0 has no ancestors
    mask = jnp.zeros((3, 3), dtype=jnp.bool_)
    result = find_ancestors_jax(mask, 0)
    expected = jnp.array([False, False, False])
    assert (result == expected).all()


def test_find_ancestors_jax_single_parent():
    # Node 1 has parent 0
    # Rows are children, columns are parents: M[child, parent]
    mask = jnp.array([
        [False, False, False],
        [True, False, False],
        [False, False, False],
    ], dtype=jnp.bool_)
    result = find_ancestors_jax(mask, 1)
    expected = jnp.array([True, False, False])
    assert (result == expected).all()


def test_find_ancestors_jax_chain():
    # 0 -> 1 -> 2
    # M[1, 0] = True, M[2, 1] = True
    mask = jnp.array([
        [False, False, False],
        [True, False, False],
        [False, True, False],
    ], dtype=jnp.bool_)
    result = find_ancestors_jax(mask, 2)
    expected = jnp.array([True, True, False])
    assert (result == expected).all()


def test_find_ancestors_jax_multiple_parents():
    # 0 -> 2, 1 -> 2
    # M[2, 0] = True, M[2, 1] = True
    mask = jnp.array([
        [False, False, False],
        [False, False, False],
        [True, True, False],
    ], dtype=jnp.bool_)
    result = find_ancestors_jax(mask, 2)
    expected = jnp.array([True, True, False])
    assert (result == expected).all()


def test_find_ancestors_jax_self_loop():
    # Immediate self-loop: 0 -> 0
    # M[0, 0] = True
    mask = jnp.array([
        [True, False],
        [False, False],
    ], dtype=jnp.bool_)
    result = find_ancestors_jax(mask, 0)
    # Should ignore immediate self-loop
    expected = jnp.array([False, False])
    assert (result == expected).all()


def test_find_ancestors_jax_cycle():
    # Cycle: 0 -> 1 -> 0
    # M[1, 0] = True, M[0, 1] = True
    mask = jnp.array([
        [False, True],
        [True, False],
    ], dtype=jnp.bool_)
    result = find_ancestors_jax(mask, 0)
    # 0 is its own ancestor due to larger cycle
    expected = jnp.array([True, True])
    assert (result == expected).all()


def test_find_ancestors_jax_int_matrix():
    # Check that it works with integer matrix
    mask = jnp.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
    ], dtype=jnp.int32)
    result = find_ancestors_jax(mask, 2)
    expected = jnp.array([True, True, False])
    assert (result == expected).all()
