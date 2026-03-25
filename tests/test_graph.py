import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp  # noqa: E402

from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_jax_linear():
    # 0 -> 1 -> 2
    # Rows are children, cols are parents
    mask = jnp.array([
        [0, 0, 0],  # 0 has no parents
        [1, 0, 0],  # 0 is parent of 1
        [0, 1, 0],  # 1 is parent of 2
    ], dtype=jnp.bool_)

    ancestors_2 = find_ancestors_jax(mask, 2)
    expected_2 = jnp.array([True, True, False], dtype=jnp.bool_)
    assert (ancestors_2 == expected_2).all()

    ancestors_1 = find_ancestors_jax(mask, 1)
    expected_1 = jnp.array([True, False, False], dtype=jnp.bool_)
    assert (ancestors_1 == expected_1).all()

    ancestors_0 = find_ancestors_jax(mask, 0)
    expected_0 = jnp.array([False, False, False], dtype=jnp.bool_)
    assert (ancestors_0 == expected_0).all()


def test_find_ancestors_jax_disconnected():
    mask = jnp.array([
        [0, 0],
        [0, 0],
    ], dtype=jnp.bool_)

    ancestors = find_ancestors_jax(mask, 0)
    expected = jnp.array([False, False], dtype=jnp.bool_)
    assert (ancestors == expected).all()


def test_find_ancestors_jax_cycle():
    # 0 -> 1 -> 2 -> 0
    mask = jnp.array([
        [0, 0, 1],  # 2 is parent of 0
        [1, 0, 0],  # 0 is parent of 1
        [0, 1, 0],  # 1 is parent of 2
    ], dtype=jnp.bool_)

    ancestors_0 = find_ancestors_jax(mask, 0)
    # Node 0 is its own ancestor via 2->1->0
    expected = jnp.array([True, True, True], dtype=jnp.bool_)
    assert (ancestors_0 == expected).all()


def test_find_ancestors_jax_immediate_self_loop():
    # 0 -> 0
    mask = jnp.array([
        [1],
    ], dtype=jnp.bool_)

    ancestors = find_ancestors_jax(mask, 0)
    expected = jnp.array([False], dtype=jnp.bool_)
    assert (ancestors == expected).all()


def test_find_ancestors_jax_complex():
    # 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3, 3 -> 4
    mask = jnp.array([
        [0, 0, 0, 0, 0],  # 0
        [1, 0, 0, 0, 0],  # 1
        [1, 0, 0, 0, 0],  # 2
        [0, 1, 1, 0, 0],  # 3
        [0, 0, 0, 1, 0],  # 4
    ], dtype=jnp.bool_)

    ancestors_4 = find_ancestors_jax(mask, 4)
    expected_4 = jnp.array([True, True, True, True, False], dtype=jnp.bool_)
    assert (ancestors_4 == expected_4).all()

    ancestors_3 = find_ancestors_jax(mask, 3)
    expected_3 = jnp.array([True, True, True, False, False], dtype=jnp.bool_)
    assert (ancestors_3 == expected_3).all()
