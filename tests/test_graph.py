import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp  # noqa: E402

from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_linear():
    # 0 -> 1 -> 2
    # Rows are children, columns are parents
    mask = jnp.zeros((3, 3), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)
    mask = mask.at[2, 1].set(True)

    # Ancestors of 2 should be 0 and 1
    expected_2 = jnp.array([True, True, False], dtype=jnp.bool_)
    result_2 = find_ancestors_jax(mask, 2)
    assert (result_2 == expected_2).all()

    # Ancestors of 1 should be 0
    expected_1 = jnp.array([True, False, False], dtype=jnp.bool_)
    result_1 = find_ancestors_jax(mask, 1)
    assert (result_1 == expected_1).all()

    # Ancestors of 0 should be none
    expected_0 = jnp.array([False, False, False], dtype=jnp.bool_)
    result_0 = find_ancestors_jax(mask, 0)
    assert (result_0 == expected_0).all()


def test_find_ancestors_multiple_paths():
    # 0 -> 1 -> 3
    # 0 -> 2 -> 3
    mask = jnp.zeros((4, 4), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)
    mask = mask.at[2, 0].set(True)
    mask = mask.at[3, 1].set(True)
    mask = mask.at[3, 2].set(True)

    # Ancestors of 3 should be 0, 1, 2
    expected_3 = jnp.array([True, True, True, False], dtype=jnp.bool_)
    result_3 = find_ancestors_jax(mask, 3)
    assert (result_3 == expected_3).all()


def test_find_ancestors_disconnected():
    # 0 -> 1
    # 2 -> 3
    mask = jnp.zeros((4, 4), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)
    mask = mask.at[3, 2].set(True)

    # Ancestors of 1 should be 0
    expected_1 = jnp.array([True, False, False, False], dtype=jnp.bool_)
    result_1 = find_ancestors_jax(mask, 1)
    assert (result_1 == expected_1).all()

    # Ancestors of 3 should be 2
    expected_3 = jnp.array([False, False, True, False], dtype=jnp.bool_)
    result_3 = find_ancestors_jax(mask, 3)
    assert (result_3 == expected_3).all()


def test_find_ancestors_cycle():
    # 0 -> 1 -> 2 -> 0
    mask = jnp.zeros((3, 3), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)
    mask = mask.at[2, 1].set(True)
    mask = mask.at[0, 2].set(True)

    # Ancestors of 0 should be 0, 1, 2
    expected_0 = jnp.array([True, True, True], dtype=jnp.bool_)
    result_0 = find_ancestors_jax(mask, 0)
    assert (result_0 == expected_0).all()


def test_find_ancestors_self_loop():
    # 0 -> 0
    mask = jnp.zeros((1, 1), dtype=jnp.bool_)
    mask = mask.at[0, 0].set(True)

    # Ancestors of 0 should be none since immediate self-loops are ignored
    expected_0 = jnp.array([False], dtype=jnp.bool_)
    result_0 = find_ancestors_jax(mask, 0)
    assert (result_0 == expected_0).all()
