import os

# select device
os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402
import numpy as np  # noqa: E402

from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


@pytest.mark.parametrize(
    "mask_list, node, expected_list",
    [
        # Test Case 1: Simple linear graph 0 -> 1 -> 2
        # Recall: mask[i, j] means j is a parent of i. Rows are children, cols are parents.
        # So mask[1, 0] = True (0 is parent of 1), mask[2, 1] = True (1 is parent of 2).
        (
            [
                [False, False, False],
                [True, False, False],
                [False, True, False]
            ],
            2,
            [True, True, False]  # Ancestors of 2 are 0 and 1
        ),

        # Test Case 2: Branching 0 -> 1 -> 3, 0 -> 2 -> 3
        # Parents of 1 is 0. Parents of 2 is 0. Parents of 3 is 1 and 2.
        (
            [
                [False, False, False, False],
                [True, False, False, False],
                [True, False, False, False],
                [False, True, True, False]
            ],
            3,
            [True, True, True, False]  # Ancestors of 3 are 0, 1, 2
        ),

        # Test Case 3: Cycle 0 -> 1 -> 2 -> 0
        # Parents of 1 is 0. Parents of 2 is 1. Parents of 0 is 2.
        (
            [
                [False, False, True],
                [True, False, False],
                [False, True, False]
            ],
            0,
            [True, True, True]  # Ancestors of 0 are 0, 1, 2. (0 becomes its own ancestor via 2)
        ),

        # Test Case 4: Disconnected graph
        (
            [
                [False, False, False],
                [False, False, False],
                [False, False, False]
            ],
            1,
            [False, False, False]  # No ancestors
        ),

        # Test Case 5: Immediate self-loop 0 -> 0
        (
            [
                [True, False],
                [False, False]
            ],
            0,
            [False, False]  # The code explicitly ignores immediate self-loops `j != current_node`
        )
    ]
)
def test_find_ancestors_jax(mask_list, node, expected_list):
    mask = jnp.array(mask_list, dtype=jnp.bool_)
    expected = jnp.array(expected_list, dtype=jnp.bool_)

    result = find_ancestors_jax(mask, node)

    # Assert result matches expected exactly
    np.testing.assert_array_equal(result, expected)
