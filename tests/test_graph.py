import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp  # noqa: E402
from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_jax_simple_chain():
    # 0 -> 1 -> 2
    # Rows are children, cols are parents
    mask = jnp.zeros((3, 3), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)
    mask = mask.at[2, 1].set(True)

    ancestors = find_ancestors_jax(mask, 2)
    # Expected: 0 and 1 are ancestors of 2
    assert ancestors[0]
    assert ancestors[1]
    assert not ancestors[2]


def test_find_ancestors_jax_branching():
    # 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
    # Ancestors of 3 are 0, 1, 2
    mask = jnp.zeros((4, 4), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)
    mask = mask.at[2, 0].set(True)
    mask = mask.at[3, 1].set(True)
    mask = mask.at[3, 2].set(True)

    ancestors = find_ancestors_jax(mask, 3)
    assert ancestors[0]
    assert ancestors[1]
    assert ancestors[2]
    assert not ancestors[3]


def test_find_ancestors_jax_disconnected():
    # 0 -> 1, 2 -> 3
    mask = jnp.zeros((4, 4), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)
    mask = mask.at[3, 2].set(True)

    ancestors_1 = find_ancestors_jax(mask, 1)
    assert ancestors_1[0]
    assert not ancestors_1[1]
    assert not ancestors_1[2]
    assert not ancestors_1[3]

    ancestors_3 = find_ancestors_jax(mask, 3)
    assert not ancestors_3[0]
    assert not ancestors_3[1]
    assert ancestors_3[2]
    assert not ancestors_3[3]


def test_find_ancestors_jax_cycle():
    # 0 -> 1, 1 -> 2, 2 -> 0
    mask = jnp.zeros((3, 3), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)
    mask = mask.at[2, 1].set(True)
    mask = mask.at[0, 2].set(True)

    ancestors = find_ancestors_jax(mask, 2)
    assert ancestors[0]
    assert ancestors[1]
    assert ancestors[2]


def test_find_ancestors_jax_self_loop():
    # 0 -> 0, 0 -> 1
    mask = jnp.zeros((2, 2), dtype=jnp.bool_)
    mask = mask.at[0, 0].set(True)
    mask = mask.at[1, 0].set(True)

    ancestors = find_ancestors_jax(mask, 1)
    assert ancestors[0]
    assert not ancestors[1]
