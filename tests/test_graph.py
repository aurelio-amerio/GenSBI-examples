import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_jax_linear():
    # 0 -> 1 -> 2
    # Rows are children, cols are parents
    # M[1, 0] = 1
    # M[2, 1] = 1
    mask = jnp.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=jnp.bool_)

    ancestors = find_ancestors_jax(mask, 2)
    np.testing.assert_array_equal(ancestors, np.array([True, True, False]))


def test_find_ancestors_jax_branching():
    # 0 -> 2
    # 1 -> 2
    # 3 -> 0
    mask = jnp.array([
        [0, 0, 0, 1],  # 0 has parent 3
        [0, 0, 0, 0],  # 1 has no parents
        [1, 1, 0, 0],  # 2 has parents 0 and 1
        [0, 0, 0, 0]   # 3 has no parents
    ], dtype=jnp.bool_)

    ancestors = find_ancestors_jax(mask, 2)
    np.testing.assert_array_equal(ancestors, np.array([True, True, False, True]))


def test_find_ancestors_jax_cyclic():
    # 0 -> 1 -> 2 -> 0
    mask = jnp.array([
        [0, 0, 1],  # 0 has parent 2
        [1, 0, 0],  # 1 has parent 0
        [0, 1, 0]   # 2 has parent 1
    ], dtype=jnp.bool_)

    ancestors = find_ancestors_jax(mask, 0)
    np.testing.assert_array_equal(ancestors, np.array([True, True, True]))


def test_find_ancestors_jax_disconnected():
    # 0, 1, 2 disconnected
    mask = jnp.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ], dtype=jnp.bool_)

    ancestors = find_ancestors_jax(mask, 1)
    np.testing.assert_array_equal(ancestors, np.array([False, False, False]))
