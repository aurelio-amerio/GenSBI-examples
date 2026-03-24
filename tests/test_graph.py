import os

# select device
os.environ["JAX_PLATFORMS"] = "cpu"

import jax  # noqa: E402, F401
import jax.numpy as jnp  # noqa: E402

from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_disconnected():
    # Disconnected graph test
    mask = jnp.zeros((3, 3), dtype=jnp.int32)
    expected = jnp.array([False, False, False])
    result = find_ancestors_jax(mask, 1)
    assert (result == expected).all()


def test_find_ancestors_linear_chain():
    # Linear chain: 0 -> 1 -> 2 -> 3
    # Rows are children, columns are parents.
    # [1, 0] means 1 is child of 0
    mask = jnp.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ], dtype=jnp.int32)

    expected_3 = jnp.array([True, True, True, False])
    result_3 = find_ancestors_jax(mask, 3)
    assert (result_3 == expected_3).all()

    expected_1 = jnp.array([True, False, False, False])
    result_1 = find_ancestors_jax(mask, 1)
    assert (result_1 == expected_1).all()


def test_find_ancestors_cycle():
    # Cycle: 0 -> 1 -> 2 -> 0
    mask = jnp.array([
        [0, 0, 1],  # 0 has parent 2
        [1, 0, 0],  # 1 has parent 0
        [0, 1, 0],  # 2 has parent 1
    ], dtype=jnp.int32)
    expected = jnp.array([True, True, True])
    result = find_ancestors_jax(mask, 0)
    assert (result == expected).all()


def test_find_ancestors_multi_parent():
    # Multi-parent: 0 -> 2, 1 -> 2
    mask = jnp.array([
        [0, 0, 0],
        [0, 0, 0],
        [1, 1, 0]
    ], dtype=jnp.int32)
    expected = jnp.array([True, True, False])
    result = find_ancestors_jax(mask, 2)
    assert (result == expected).all()


def test_find_ancestors_multi_branching():
    # Multi-branching: 0 -> 1 -> 3, 0 -> 2 -> 3
    mask = jnp.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 1, 0]
    ], dtype=jnp.int32)
    expected = jnp.array([True, True, True, False])
    result = find_ancestors_jax(mask, 3)
    assert (result == expected).all()
