import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp  # noqa: E402

from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_jax_simple():
    # 0 has parents 1 and 2
    # 1 has parent 3
    # 2 has parent 3
    # 3 has no parents
    mask = jnp.array(
        [
            [0, 1, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ],
        dtype=jnp.bool_,
    )

    ancestors = find_ancestors_jax(mask, 0)
    expected = jnp.array([False, True, True, True], dtype=jnp.bool_)
    assert (ancestors == expected).all()


def test_find_ancestors_jax_linear():
    # 0 -> 1 -> 2 -> 3
    # In mask notation (rows are children, cols are parents):
    # 1 has parent 0
    # 2 has parent 1
    # 3 has parent 2
    mask = jnp.array(
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ],
        dtype=jnp.bool_,
    )

    ancestors_3 = find_ancestors_jax(mask, 3)
    expected_3 = jnp.array([True, True, True, False], dtype=jnp.bool_)
    assert (ancestors_3 == expected_3).all()

    ancestors_1 = find_ancestors_jax(mask, 1)
    expected_1 = jnp.array([True, False, False, False], dtype=jnp.bool_)
    assert (ancestors_1 == expected_1).all()

    ancestors_0 = find_ancestors_jax(mask, 0)
    expected_0 = jnp.array([False, False, False, False], dtype=jnp.bool_)
    assert (ancestors_0 == expected_0).all()


def test_find_ancestors_jax_disjoint():
    # Two disconnected components:
    # 0 <- 1
    # 2 <- 3
    mask = jnp.array(
        [
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ],
        dtype=jnp.bool_,
    )

    ancestors_0 = find_ancestors_jax(mask, 0)
    expected_0 = jnp.array([False, True, False, False], dtype=jnp.bool_)
    assert (ancestors_0 == expected_0).all()

    ancestors_2 = find_ancestors_jax(mask, 2)
    expected_2 = jnp.array([False, False, False, True], dtype=jnp.bool_)
    assert (ancestors_2 == expected_2).all()


def test_find_ancestors_jax_cycle():
    # 0 <- 1 <- 2 <- 0
    mask = jnp.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ],
        dtype=jnp.bool_,
    )

    # 0 has 1 as parent, 1 has 2 as parent, 2 has 0 as parent.
    # Therefore, 0's ancestors are 1, 2, and 0 itself.
    ancestors_0 = find_ancestors_jax(mask, 0)
    expected_0 = jnp.array([True, True, True], dtype=jnp.bool_)
    assert (ancestors_0 == expected_0).all()


def test_find_ancestors_jax_no_parents():
    mask = jnp.zeros((3, 3), dtype=jnp.bool_)
    ancestors_0 = find_ancestors_jax(mask, 0)
    expected_0 = jnp.array([False, False, False], dtype=jnp.bool_)
    assert (ancestors_0 == expected_0).all()
