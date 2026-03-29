import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp  # noqa: E402
from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_jax_dag():
    # 0 -> 1 -> 2 -> 3
    # 0 -> 2
    # mask[i, j] = 1 means j -> i (row is child, col is parent)
    mask = jnp.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 0]
    ], dtype=jnp.bool_)

    # Ancestors of 3: 2, 1, 0
    ans_3 = find_ancestors_jax(mask, 3)
    expected_3 = jnp.array([True, True, True, False])
    assert (ans_3 == expected_3).all()

    # Ancestors of 2: 1, 0
    ans_2 = find_ancestors_jax(mask, 2)
    expected_2 = jnp.array([True, True, False, False])
    assert (ans_2 == expected_2).all()

    # Ancestors of 0: none
    ans_0 = find_ancestors_jax(mask, 0)
    expected_0 = jnp.array([False, False, False, False])
    assert (ans_0 == expected_0).all()


def test_find_ancestors_jax_cycle():
    # 0 -> 1 -> 2 -> 0
    mask = jnp.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=jnp.bool_)

    # Ancestors of 0: 0, 1, 2
    ans_0 = find_ancestors_jax(mask, 0)
    expected_0 = jnp.array([True, True, True])
    assert (ans_0 == expected_0).all()


def test_find_ancestors_jax_self_loop():
    # 0 -> 0 (immediate self loop)
    # 1 -> 0
    mask = jnp.array([
        [1, 1],
        [0, 0]
    ], dtype=jnp.bool_)

    ans_0 = find_ancestors_jax(mask, 0)
    # Immediate self loops are ignored, so 0 should not be its own ancestor.
    # 1 is an ancestor of 0.
    expected_0 = jnp.array([False, True])
    assert (ans_0 == expected_0).all()
