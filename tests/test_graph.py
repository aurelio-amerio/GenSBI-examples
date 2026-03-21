import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax  # noqa: F401, E402
import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: F401, E402

from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_jax_simple_chain():
    # 0 <- 1 <- 2
    mask = jnp.zeros((3, 3), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)
    mask = mask.at[2, 1].set(True)

    expected_0 = jnp.array([False, False, False], dtype=jnp.bool_)
    result_0 = find_ancestors_jax(mask, 0)
    assert (result_0 == expected_0).all()
    assert result_0.dtype == expected_0.dtype

    expected_1 = jnp.array([True, False, False], dtype=jnp.bool_)
    result_1 = find_ancestors_jax(mask, 1)
    assert (result_1 == expected_1).all()

    expected_2 = jnp.array([True, True, False], dtype=jnp.bool_)
    result_2 = find_ancestors_jax(mask, 2)
    assert (result_2 == expected_2).all()


def test_find_ancestors_jax_branching():
    # 0 <- 1, 0 <- 3, 1 <- 2, 4 <- 3
    # M_ij = 1 implies edge j -> i (row is child, col is parent)
    mask = jnp.zeros((5, 5), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)
    mask = mask.at[3, 0].set(True)
    mask = mask.at[2, 1].set(True)
    mask = mask.at[4, 3].set(True)

    expected_2 = jnp.array([True, True, False, False, False], dtype=jnp.bool_)
    result_2 = find_ancestors_jax(mask, 2)
    assert (result_2 == expected_2).all()

    expected_4 = jnp.array([True, False, False, True, False], dtype=jnp.bool_)
    result_4 = find_ancestors_jax(mask, 4)
    assert (result_4 == expected_4).all()


def test_find_ancestors_jax_cycle():
    # 0 <- 1 <- 2 <- 0
    mask = jnp.zeros((3, 3), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)
    mask = mask.at[2, 1].set(True)
    mask = mask.at[0, 2].set(True)

    expected_0 = jnp.array([True, True, True], dtype=jnp.bool_)
    result_0 = find_ancestors_jax(mask, 0)
    assert (result_0 == expected_0).all()


def test_find_ancestors_jax_disconnected():
    mask = jnp.zeros((3, 3), dtype=jnp.bool_)

    expected = jnp.array([False, False, False], dtype=jnp.bool_)
    result = find_ancestors_jax(mask, 0)
    assert (result == expected).all()


def test_find_ancestors_jax_self_loop():
    # 0 <- 0
    mask = jnp.zeros((1, 1), dtype=jnp.bool_)
    mask = mask.at[0, 0].set(True)

    expected = jnp.array([False], dtype=jnp.bool_)
    result = find_ancestors_jax(mask, 0)
    assert (result == expected).all()
