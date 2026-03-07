import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_jax_simple():
    # 0 -> 1 -> 2
    # 3 -> 1
    # 2 -> 4
    # Rows represent children, columns represent parents
    mask = np.zeros((5, 5), dtype=bool)
    mask[1, 0] = True
    mask[2, 1] = True
    mask[1, 3] = True
    mask[4, 2] = True

    # Ancestors of 4: 0, 1, 2, 3
    ans = find_ancestors_jax(jnp.array(mask), 4)
    assert np.array_equal(ans, np.array([True, True, True, True, False]))

    # Ancestors of 2: 0, 1, 3
    ans = find_ancestors_jax(jnp.array(mask), 2)
    assert np.array_equal(ans, np.array([True, True, False, True, False]))

    # Ancestors of 1: 0, 3
    ans = find_ancestors_jax(jnp.array(mask), 1)
    assert np.array_equal(ans, np.array([True, False, False, True, False]))

    # Ancestors of 0: none
    ans = find_ancestors_jax(jnp.array(mask), 0)
    assert np.array_equal(ans, np.array([False, False, False, False, False]))


def test_find_ancestors_jax_cycle():
    # 0 -> 1 -> 2 -> 0
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 0] = True
    mask[2, 1] = True
    mask[0, 2] = True

    ans = find_ancestors_jax(jnp.array(mask), 0)
    assert np.array_equal(ans, np.array([True, True, True]))


def test_find_ancestors_jax_self_loop():
    # 0 -> 0
    mask = np.zeros((1, 1), dtype=bool)
    mask[0, 0] = True

    ans = find_ancestors_jax(jnp.array(mask), 0)
    assert np.array_equal(ans, np.array([False]))


def test_find_ancestors_jax_disconnected():
    # 0, 1, 2 disconnected
    mask = np.zeros((3, 3), dtype=bool)

    ans = find_ancestors_jax(jnp.array(mask), 1)
    assert np.array_equal(ans, np.array([False, False, False]))
