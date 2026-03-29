import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax  # noqa: E402, F401
import jax.numpy as jnp  # noqa: E402
from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_jax_simple_dag():
    # Test with a simple DAG
    # 0 -> 1 -> 2
    # Rows are children, columns are parents
    # 0 has no parents
    # 1 has 0 as parent
    # 2 has 1 as parent
    mask = jnp.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=jnp.bool_)

    expected_0 = jnp.array([False, False, False], dtype=jnp.bool_)
    result_0 = find_ancestors_jax(mask, 0)
    assert (result_0 == expected_0).all()
    assert result_0.dtype == expected_0.dtype

    expected_1 = jnp.array([True, False, False], dtype=jnp.bool_)
    result_1 = find_ancestors_jax(mask, 1)
    assert (result_1 == expected_1).all()

    expected_2 = jnp.array([True, True, False], dtype=jnp.bool_)
    result_2 = find_ancestors_jax(mask, 2)
    assert (result_2 == expected_2).all()


def test_find_ancestors_jax_disconnected():
    # Test with a standalone node
    # 0 -> 1 -> 2
    # 3
    mask = jnp.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0]
    ], dtype=jnp.bool_)

    expected_3 = jnp.array([False, False, False, False], dtype=jnp.bool_)
    result_3 = find_ancestors_jax(mask, 3)
    assert (result_3 == expected_3).all()

    expected_2 = jnp.array([True, True, False, False], dtype=jnp.bool_)
    result_2 = find_ancestors_jax(mask, 2)
    assert (result_2 == expected_2).all()


def test_find_ancestors_jax_cycle():
    # Cycle test
    # 0 -> 1 -> 2 -> 0
    mask_cycle = jnp.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=jnp.bool_)

    expected_0 = jnp.array([True, True, True], dtype=jnp.bool_)
    result_0 = find_ancestors_jax(mask_cycle, 0)
    assert (result_0 == expected_0).all()
    assert result_0.dtype == expected_0.dtype


def test_find_ancestors_jax_multiple_parents():
    # 0 -> 2
    # 1 -> 2
    # 2 -> 3
    mask = jnp.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 0]
    ], dtype=jnp.bool_)

    expected_0 = jnp.array([False, False, False, False], dtype=jnp.bool_)
    result_0 = find_ancestors_jax(mask, 0)
    assert (result_0 == expected_0).all()

    expected_2 = jnp.array([True, True, False, False], dtype=jnp.bool_)
    result_2 = find_ancestors_jax(mask, 2)
    assert (result_2 == expected_2).all()

    expected_3 = jnp.array([True, True, True, False], dtype=jnp.bool_)
    result_3 = find_ancestors_jax(mask, 3)
    assert (result_3 == expected_3).all()
