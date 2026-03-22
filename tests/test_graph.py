import os

# select device
os.environ["JAX_PLATFORMS"] = "cpu"

import jax  # noqa: E402, F401
import jax.numpy as jnp  # noqa: E402

from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_jax_linear():
    # 0 -> 1 -> 2
    mask = jnp.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=jnp.bool_)

    ancestors_0 = find_ancestors_jax(mask, 0)
    expected_0 = jnp.array([False, False, False], dtype=jnp.bool_)
    assert (ancestors_0 == expected_0).all()

    ancestors_1 = find_ancestors_jax(mask, 1)
    expected_1 = jnp.array([True, False, False], dtype=jnp.bool_)
    assert (ancestors_1 == expected_1).all()

    ancestors_2 = find_ancestors_jax(mask, 2)
    expected_2 = jnp.array([True, True, False], dtype=jnp.bool_)
    assert (ancestors_2 == expected_2).all()


def test_find_ancestors_jax_tree():
    # 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 4
    mask = jnp.array([
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0]
    ], dtype=jnp.bool_)

    ancestors_3 = find_ancestors_jax(mask, 3)
    expected_3 = jnp.array([True, True, False, False, False], dtype=jnp.bool_)
    assert (ancestors_3 == expected_3).all()

    ancestors_4 = find_ancestors_jax(mask, 4)
    expected_4 = jnp.array([True, False, True, False, False], dtype=jnp.bool_)
    assert (ancestors_4 == expected_4).all()


def test_find_ancestors_jax_multiple_parents():
    # 0 -> 2, 1 -> 2
    mask = jnp.array([
        [0, 0, 0],
        [0, 0, 0],
        [1, 1, 0]
    ], dtype=jnp.bool_)

    ancestors_2 = find_ancestors_jax(mask, 2)
    expected_2 = jnp.array([True, True, False], dtype=jnp.bool_)
    assert (ancestors_2 == expected_2).all()


def test_find_ancestors_jax_diamond():
    # 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
    mask = jnp.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 1, 0]
    ], dtype=jnp.bool_)

    ancestors_3 = find_ancestors_jax(mask, 3)
    expected_3 = jnp.array([True, True, True, False], dtype=jnp.bool_)
    assert (ancestors_3 == expected_3).all()


def test_find_ancestors_jax_complex():
    # 0 -> 1 -> 2
    # 0 -> 3 -> 4
    # 1 -> 4
    mask = jnp.array([
        [0, 0, 0, 0, 0],  # 0
        [1, 0, 0, 0, 0],  # 1 (parent is 0)
        [0, 1, 0, 0, 0],  # 2 (parent is 1)
        [1, 0, 0, 0, 0],  # 3 (parent is 0)
        [0, 1, 0, 1, 0],  # 4 (parents are 1 and 3)
    ], dtype=jnp.bool_)

    ancestors_4 = find_ancestors_jax(mask, 4)
    expected_4 = jnp.array([True, True, False, True, False], dtype=jnp.bool_)
    assert (ancestors_4 == expected_4).all()

    ancestors_2 = find_ancestors_jax(mask, 2)
    expected_2 = jnp.array([True, True, False, False, False], dtype=jnp.bool_)
    assert (ancestors_2 == expected_2).all()
