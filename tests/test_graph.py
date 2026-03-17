import os

# select device
os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp  # noqa: E402

from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_jax_basic_linear():
    mask = jnp.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
    ], dtype=jnp.bool_)

    expected_0 = jnp.array([False, False, False, False], dtype=jnp.bool_)
    result_0 = find_ancestors_jax(mask, 0)
    assert (result_0 == expected_0).all()

    expected_2 = jnp.array([True, True, False, False], dtype=jnp.bool_)
    result_2 = find_ancestors_jax(mask, 2)
    assert (result_2 == expected_2).all()


def test_find_ancestors_jax_multi_parents():
    # Multiple parents test:
    # 2 has parents 0 and 1. 0 has parent 3. 1 has parent 3.
    mask_multi = jnp.array([
        [0, 0, 0, 1],  # 0 parent 3
        [0, 0, 0, 1],  # 1 parent 3
        [1, 1, 0, 0],  # 2 parents 0, 1
        [0, 0, 0, 0],  # 3 no parents
    ], dtype=jnp.bool_)

    expected = jnp.array([True, True, False, True], dtype=jnp.bool_)
    result = find_ancestors_jax(mask_multi, 2)
    assert (result == expected).all()


def test_find_ancestors_jax_cycle():
    # Cycle: 0 -> 1 -> 2 -> 0
    mask_cycle = jnp.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
    ], dtype=jnp.bool_)

    expected = jnp.array([True, True, True], dtype=jnp.bool_)
    result = find_ancestors_jax(mask_cycle, 0)
    assert (result == expected).all()


def test_find_ancestors_jax_disconnected():
    # Disconnected
    mask_disc = jnp.array([
        [0, 0],
        [0, 0],
    ], dtype=jnp.bool_)

    expected = jnp.array([False, False], dtype=jnp.bool_)
    result = find_ancestors_jax(mask_disc, 0)
    assert (result == expected).all()
