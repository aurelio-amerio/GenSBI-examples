import os

# Set JAX platforms to CPU before importing jax, as requested by memory
os.environ["JAX_PLATFORMS"] = "cpu"

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


@pytest.mark.parametrize(
    "mask, node, expected_ancestors",
    [
        (
            np.array([
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
            ], dtype=bool),
            3,
            np.array([1, 1, 1, 0], dtype=bool)
        ),
        (
            np.array([
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
            ], dtype=bool),
            0,
            np.array([0, 0, 0, 0], dtype=bool)
        ),
        (
            np.array([
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 0],
            ], dtype=bool),
            3,
            np.array([1, 1, 1, 0], dtype=bool)
        ),
        (
            np.array([
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 1, 1, 0],
            ], dtype=bool),
            3,
            np.array([1, 1, 1, 0], dtype=bool)
        ),
        (
            np.array([
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 0],
            ], dtype=bool),
            1,
            np.array([1, 0, 0, 0], dtype=bool)
        ),
        (
            np.array([
                [0, 1],
                [1, 0],
            ], dtype=bool),
            0,
            np.array([1, 1], dtype=bool)
        ),
        (
            np.array([
                [1, 0],
                [0, 0],
            ], dtype=bool),
            0,
            np.array([0, 0], dtype=bool)
        )
    ]
)
def test_find_ancestors_jax(mask, node, expected_ancestors):
    jax_mask = jnp.array(mask)
    result = find_ancestors_jax(jax_mask, node)

    assert result.dtype == jnp.bool_
    np.testing.assert_array_equal(np.array(result), expected_ancestors)
