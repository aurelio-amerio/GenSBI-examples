import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: F401, E402

from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_jax_basic_chain():
    # 0 -> 1 -> 2
    # Rows are children, columns are parents
    mask = jnp.array([
        [False, False, False],
        [True, False, False],
        [False, True, False],
    ], dtype=jnp.bool_)

    # Ancestors of 2 should be 0 and 1
    ancestors = find_ancestors_jax(mask, 2)
    expected = jnp.array([True, True, False], dtype=jnp.bool_)
    assert (ancestors == expected).all()
    assert ancestors.dtype == jnp.bool_

    # Ancestors of 1 should be 0
    ancestors = find_ancestors_jax(mask, 1)
    expected = jnp.array([True, False, False], dtype=jnp.bool_)
    assert (ancestors == expected).all()

    # Ancestors of 0 should be empty
    ancestors = find_ancestors_jax(mask, 0)
    expected = jnp.array([False, False, False], dtype=jnp.bool_)
    assert (ancestors == expected).all()


def test_find_ancestors_jax_multiple_parents():
    # 0 -> 2, 1 -> 2, 2 -> 3
    mask = jnp.array([
        [False, False, False, False],
        [False, False, False, False],
        [True, True, False, False],
        [False, False, True, False],
    ], dtype=jnp.bool_)

    ancestors = find_ancestors_jax(mask, 3)
    expected = jnp.array([True, True, True, False], dtype=jnp.bool_)
    assert (ancestors == expected).all()

    ancestors = find_ancestors_jax(mask, 2)
    expected = jnp.array([True, True, False, False], dtype=jnp.bool_)
    assert (ancestors == expected).all()


def test_find_ancestors_jax_disconnected():
    # 0 -> 1, 2 -> 3
    mask = jnp.array([
        [False, False, False, False],
        [True, False, False, False],
        [False, False, False, False],
        [False, False, True, False],
    ], dtype=jnp.bool_)

    ancestors = find_ancestors_jax(mask, 1)
    expected = jnp.array([True, False, False, False], dtype=jnp.bool_)
    assert (ancestors == expected).all()

    ancestors = find_ancestors_jax(mask, 3)
    expected = jnp.array([False, False, True, False], dtype=jnp.bool_)
    assert (ancestors == expected).all()


def test_find_ancestors_jax_self_loop_and_cycle():
    # 0 -> 0 (self loop)
    # 1 -> 2 -> 1 (cycle)
    # 3 -> 4
    mask = jnp.array([
        [True, False, False, False, False],
        [False, False, True, False, False],
        [False, True, False, False, False],
        [False, False, False, False, False],
        [False, False, False, True, False],
    ], dtype=jnp.bool_)

    # Node 0 self loop is ignored
    ancestors = find_ancestors_jax(mask, 0)
    expected = jnp.array([False, False, False, False, False], dtype=jnp.bool_)
    assert (ancestors == expected).all()

    # Node 1 is part of cycle 1-2
    ancestors = find_ancestors_jax(mask, 1)
    expected = jnp.array([False, True, True, False, False], dtype=jnp.bool_)
    assert (ancestors == expected).all()

    # Node 2 is part of cycle 1-2
    ancestors = find_ancestors_jax(mask, 2)
    expected = jnp.array([False, True, True, False, False], dtype=jnp.bool_)
    assert (ancestors == expected).all()
