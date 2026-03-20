import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp  # noqa: E402

from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_jax():
    # Graph:
    # 0 -> 1 -> 2
    # 0 -> 3 -> 4
    # 2 -> 4

    num_nodes = 5
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)
    mask = mask.at[2, 1].set(True)
    mask = mask.at[3, 0].set(True)
    mask = mask.at[4, 2].set(True)
    mask = mask.at[4, 3].set(True)

    # Ancestors of 4: 0, 1, 2, 3
    expected_4 = jnp.array([True, True, True, True, False])
    result_4 = find_ancestors_jax(mask, 4)
    assert (result_4 == expected_4).all()

    # Ancestors of 2: 0, 1
    expected_2 = jnp.array([True, True, False, False, False])
    result_2 = find_ancestors_jax(mask, 2)
    assert (result_2 == expected_2).all()

    # Ancestors of 0: none
    expected_0 = jnp.array([False, False, False, False, False])
    result_0 = find_ancestors_jax(mask, 0)
    assert (result_0 == expected_0).all()


def test_find_ancestors_jax_with_cycle():
    # Graph with cycle: 0 -> 1 -> 2 -> 0
    # and 2 -> 3
    num_nodes = 4
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)
    mask = mask.at[2, 1].set(True)
    mask = mask.at[0, 2].set(True)
    mask = mask.at[3, 2].set(True)

    # Ancestors of 3: 0, 1, 2
    expected_3 = jnp.array([True, True, True, False])
    result_3 = find_ancestors_jax(mask, 3)
    assert (result_3 == expected_3).all()

    # Ancestors of 1: 0, 2 (and itself because of cycle)
    expected_1 = jnp.array([True, True, True, False])
    result_1 = find_ancestors_jax(mask, 1)
    assert (result_1 == expected_1).all()


def test_find_ancestors_jax_self_loop():
    # Graph with self-loop: 0 -> 0
    num_nodes = 1
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.bool_)
    mask = mask.at[0, 0].set(True)

    # Ancestors of 0: self-loop is explicitly ignored
    expected_0 = jnp.array([False])
    result_0 = find_ancestors_jax(mask, 0)
    assert (result_0 == expected_0).all()
