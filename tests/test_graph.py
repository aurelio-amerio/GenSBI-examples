import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp  # noqa: E402

from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_jax_basic():
    # Adjacency matrix: rows are children, columns are parents.
    mask = jnp.zeros((5, 5), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)  # 0 -> 1
    mask = mask.at[2, 1].set(True)  # 1 -> 2
    mask = mask.at[2, 0].set(True)  # 0 -> 2
    mask = mask.at[4, 3].set(True)  # 3 -> 4

    ancestors_2 = find_ancestors_jax(mask, 2)
    expected_2 = jnp.array([True, True, False, False, False])
    assert (ancestors_2 == expected_2).all()

    ancestors_4 = find_ancestors_jax(mask, 4)
    expected_4 = jnp.array([False, False, False, True, False])
    assert (ancestors_4 == expected_4).all()


def test_find_ancestors_jax_integer_mask():
    mask = jnp.zeros((3, 3), dtype=jnp.int32)
    mask = mask.at[1, 0].set(1)
    mask = mask.at[2, 1].set(1)

    ancestors_2 = find_ancestors_jax(mask, 2)
    expected_2 = jnp.array([True, True, False])
    assert (ancestors_2 == expected_2).all()


def test_find_ancestors_jax_cycle():
    # 0 -> 1 -> 2 -> 0
    mask = jnp.zeros((3, 3), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)
    mask = mask.at[2, 1].set(True)
    mask = mask.at[0, 2].set(True)

    ancestors_0 = find_ancestors_jax(mask, 0)
    # Node 0's ancestors in a larger cycle should include itself
    expected_0 = jnp.array([True, True, True])
    assert (ancestors_0 == expected_0).all()


def test_find_ancestors_jax_self_loop():
    # 0 -> 0 (immediate self loop should be ignored)
    mask = jnp.zeros((2, 2), dtype=jnp.bool_)
    mask = mask.at[0, 0].set(True)

    ancestors_0 = find_ancestors_jax(mask, 0)
    expected_0 = jnp.array([False, False])
    assert (ancestors_0 == expected_0).all()
