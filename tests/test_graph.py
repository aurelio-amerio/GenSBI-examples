import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp  # noqa: E402
from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_jax_simple():
    # Construct a simple DAG
    # 0 -> 1 -> 2
    # 0 -> 2
    # Rows are children, cols are parents
    # M[i, j] = 1 means j -> i
    mask = jnp.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0]
    ], dtype=jnp.bool_)

    # Ancestors of 2 should be 0 and 1
    ancestors = find_ancestors_jax(mask, 2)
    assert (ancestors == jnp.array([True, True, False])).all()

    # Ancestors of 1 should be 0
    ancestors = find_ancestors_jax(mask, 1)
    assert (ancestors == jnp.array([True, False, False])).all()

    # Ancestors of 0 should be none
    ancestors = find_ancestors_jax(mask, 0)
    assert (ancestors == jnp.array([False, False, False])).all()


def test_find_ancestors_jax_cycle():
    # 0 -> 1 -> 2 -> 0
    mask = jnp.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=jnp.bool_)

    # Everyone is an ancestor of everyone
    # Except that 0 is not an ancestor of 0 initially
    # but cycles make it an ancestor
    ancestors_0 = find_ancestors_jax(mask, 0)
    assert (ancestors_0 == jnp.array([True, True, True])).all()

    ancestors_1 = find_ancestors_jax(mask, 1)
    assert (ancestors_1 == jnp.array([True, True, True])).all()

    ancestors_2 = find_ancestors_jax(mask, 2)
    assert (ancestors_2 == jnp.array([True, True, True])).all()


def test_find_ancestors_jax_disconnected():
    # 0 -> 1
    # 2 -> 3
    mask = jnp.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0]
    ], dtype=jnp.bool_)

    # Ancestors of 1 should be 0
    ancestors = find_ancestors_jax(mask, 1)
    assert (ancestors == jnp.array([True, False, False, False])).all()

    # Ancestors of 3 should be 2
    ancestors = find_ancestors_jax(mask, 3)
    assert (ancestors == jnp.array([False, False, True, False])).all()


def test_find_ancestors_jax_self_loop():
    # 0 -> 0
    mask = jnp.array([
        [1]
    ], dtype=jnp.bool_)

    ancestors = find_ancestors_jax(mask, 0)
    assert (ancestors == jnp.array([False])).all()
