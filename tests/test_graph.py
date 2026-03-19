import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp  # noqa: E402

from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_jax():
    # Test linear graph: 0 -> 1 -> 2
    # Rows are children, columns are parents
    linear_mask = jnp.array([
        [0, 0, 0],  # 0 has no parents
        [1, 0, 0],  # 1 has parent 0
        [0, 1, 0],  # 2 has parent 1
    ], dtype=jnp.int32)

    expected_2 = jnp.array([True, True, False], dtype=jnp.bool_)
    result_2 = find_ancestors_jax(linear_mask, 2)
    assert result_2.dtype == expected_2.dtype
    assert (result_2 == expected_2).all()

    expected_1 = jnp.array([True, False, False], dtype=jnp.bool_)
    result_1 = find_ancestors_jax(linear_mask, 1)
    assert (result_1 == expected_1).all()

    expected_0 = jnp.array([False, False, False], dtype=jnp.bool_)
    result_0 = find_ancestors_jax(linear_mask, 0)
    assert (result_0 == expected_0).all()

    # Test branching DAG: 0 -> 1 -> 3, 0 -> 2 -> 3
    branching_mask = jnp.array([
        [0, 0, 0, 0],  # 0 has no parents
        [1, 0, 0, 0],  # 1 has parent 0
        [1, 0, 0, 0],  # 2 has parent 0
        [0, 1, 1, 0],  # 3 has parents 1, 2
    ], dtype=jnp.int32)

    expected_3 = jnp.array([True, True, True, False], dtype=jnp.bool_)
    result_3 = find_ancestors_jax(branching_mask, 3)
    assert (result_3 == expected_3).all()

    # Test cycle: 0 -> 1 -> 2 -> 0
    cycle_mask = jnp.array([
        [0, 0, 1],  # 0 has parent 2
        [1, 0, 0],  # 1 has parent 0
        [0, 1, 0],  # 2 has parent 1
    ], dtype=jnp.int32)

    # All nodes are ancestors of all other nodes, including themselves
    expected_cycle = jnp.array([True, True, True], dtype=jnp.bool_)
    result_cycle = find_ancestors_jax(cycle_mask, 0)
    assert (result_cycle == expected_cycle).all()
