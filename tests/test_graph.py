import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp  # noqa: E402

from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_jax_simple():
    mask = jnp.array([
        [False, False, False],
        [True, False, False],
        [False, True, False]
    ])
    result = find_ancestors_jax(mask, 2)
    expected = jnp.array([True, True, False])
    assert (result == expected).all()


def test_find_ancestors_jax_branching():
    mask = jnp.array([
        [False, False, False, False],
        [False, False, False, False],
        [True, True, False, False],
        [False, False, True, False]
    ])
    result = find_ancestors_jax(mask, 3)
    expected = jnp.array([True, True, True, False])
    assert (result == expected).all()


def test_find_ancestors_jax_cycle():
    mask = jnp.array([
        [False, False, True],
        [True, False, False],
        [False, True, False]
    ])
    result = find_ancestors_jax(mask, 0)
    expected = jnp.array([True, True, True])
    assert (result == expected).all()


def test_find_ancestors_jax_self_loop():
    mask = jnp.array([
        [False, False],
        [True, True]
    ])
    result = find_ancestors_jax(mask, 1)
    expected = jnp.array([True, False])
    assert (result == expected).all()
