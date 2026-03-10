import os

# select device
os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402

from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


@pytest.mark.parametrize(
    "mask_list, node, expected_list, dtype",
    [
        # Simple DAG: 0 -> 1 -> 2. Ancestors of 2 are 0 and 1.
        (
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
            ],
            2,
            [True, True, False],
            jnp.bool_
        ),
        # Node with multiple parents: 0 -> 2, 1 -> 2. Ancestors of 2 are 0 and 1.
        (
            [
                [0, 0, 0],
                [0, 0, 0],
                [1, 1, 0],
            ],
            2,
            [True, True, False],
            jnp.bool_
        ),
        # Chain: 0 -> 1 -> 2 -> 3. Ancestors of 3 are 0, 1, and 2.
        (
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
            ],
            3,
            [True, True, True, False],
            jnp.bool_
        ),
        # Self-loop: 0 -> 0. Ancestors of 0: self-loop is ignored as per rules
        (
            [
                [1],
            ],
            0,
            [False],
            jnp.bool_
        ),
        # Cycle A -> B -> A. 0 -> 1, 1 -> 0
        (
            [
                [0, 1],
                [1, 0],
            ],
            0,
            [True, True],
            jnp.bool_
        ),
        # Integer adjacency matrix
        (
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
            ],
            2,
            [True, True, False],
            jnp.int32
        ),
        # Node with no ancestors
        (
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
            ],
            0,
            [False, False, False],
            jnp.bool_
        ),
    ],
)
def test_find_ancestors_jax(mask_list, node, expected_list, dtype):
    mask = jnp.array(mask_list, dtype=dtype)
    expected = jnp.array(expected_list, dtype=jnp.bool_)

    result = find_ancestors_jax(mask, node)

    # Explicit boolean assertion for flake8 compliance
    assert (result == expected).all()
    assert result.dtype == jnp.bool_
