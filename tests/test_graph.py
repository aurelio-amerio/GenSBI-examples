import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402

from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


@pytest.mark.parametrize(
    "mask_array, node, expected_array",
    [
        # Linear chain: 0 -> 1 -> 2
        (
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0]
            ],
            2,
            [True, True, False]
        ),
        # Branching: 0 -> 2, 1 -> 2
        (
            [
                [0, 0, 0],
                [0, 0, 0],
                [1, 1, 0]
            ],
            2,
            [True, True, False]
        ),
        # Diamond: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
        (
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 1, 1, 0]
            ],
            3,
            [True, True, True, False]
        ),
        # Disconnected components: 0 -> 1, 2 -> 3
        (
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 0]
            ],
            3,
            [False, False, True, False]
        ),
        # Node with no parents (root)
        (
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0]
            ],
            0,
            [False, False, False]
        ),
        # Self-loop A -> A is explicitly ignored in body by `j != current_node`
        (
            [
                [1, 0],
                [1, 0]
            ],
            0,
            [False, False]
        ),
        # Cycle A -> B -> A
        (
            [
                [0, 1],
                [1, 0]
            ],
            0,
            [True, True]
        ),
    ]
)
def test_find_ancestors_jax(mask_array, node, expected_array):
    mask = jnp.array(mask_array, dtype=jnp.bool_)
    expected = jnp.array(expected_array, dtype=jnp.bool_)

    result = find_ancestors_jax(mask, node)

    assert result.shape == expected.shape
    assert result.dtype == expected.dtype
    assert (result == expected).all()
