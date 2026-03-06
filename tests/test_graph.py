import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402

from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


@pytest.mark.parametrize(
    "mask_list, node, expected_ancestors_list",
    [
        # Simple Chain: 0 -> 1 -> 2 -> 3
        # Rows are children, Columns are parents (M[i, j] = 1 means j -> i)
        (
            [
                [0, 0, 0, 0],  # 0 has no parents
                [1, 0, 0, 0],  # 1 has parent 0
                [0, 1, 0, 0],  # 2 has parent 1
                [0, 0, 1, 0]   # 3 has parent 2
            ],
            3,
            [1, 1, 1, 0]   # 0, 1, 2 are ancestors
        ),
        (
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0]
            ],
            2,
            [1, 1, 0, 0]   # 0, 1 are ancestors
        ),
        (
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0]
            ],
            0,
            [0, 0, 0, 0]   # No ancestors
        ),
        # Branching: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
        (
            [
                [0, 0, 0, 0],  # 0 has no parents
                [1, 0, 0, 0],  # 1 has parent 0
                [1, 0, 0, 0],  # 2 has parent 0
                [0, 1, 1, 0]   # 3 has parents 1 and 2
            ],
            3,
            [1, 1, 1, 0]   # 0, 1, 2 are ancestors
        ),
        # Disconnected: 0 -> 1, 2 -> 3
        (
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 0]
            ],
            3,
            [0, 0, 1, 0]   # 2 is the only ancestor
        ),
        # Immediate Self-loop: 0 -> 1 -> 2, 2 -> 2
        (
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 1]      # 2 has parents 1 and 2
            ],
            2,
            # 0 and 1 are ancestors, 2 is ignored
            [1, 1, 0]
        ),
        # Larger Cycle: 0 -> 1 -> 2 -> 0
        (
            [
                [0, 0, 1],     # 0 has parent 2
                [1, 0, 0],     # 1 has parent 0
                [0, 1, 0]      # 2 has parent 1
            ],
            0,
            # 0, 1, 2 are all ancestors (it finds itself via cycle)
            [1, 1, 1]
        ),
    ],
)
def test_find_ancestors_jax(mask_list, node, expected_ancestors_list):
    mask = jnp.array(mask_list, dtype=jnp.bool_)
    expected_ancestors = jnp.array(expected_ancestors_list, dtype=jnp.bool_)

    ancestors = find_ancestors_jax(mask, node)

    assert jnp.array_equal(ancestors, expected_ancestors)
    assert ancestors.dtype == jnp.bool_
