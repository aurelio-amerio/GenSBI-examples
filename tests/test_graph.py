import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402

from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


@pytest.mark.parametrize(
    "adj_matrix, node, expected_ancestors, dtype",
    [
        # Simple chain: 0 depends on 1, 1 depends on 2
        # Row 0: parent 1
        # Row 1: parent 2
        # Row 2: no parents
        (
            [[False, True, False],
             [False, False, True],
             [False, False, False]],
            0,
            [False, True, True],
            jnp.bool_
        ),
        # Same chain but int32
        (
            [[0, 1, 0],
             [0, 0, 1],
             [0, 0, 0]],
            0,
            [False, True, True],
            jnp.int32
        ),
        # Node 1 ancestors in simple chain
        (
            [[False, True, False],
             [False, False, True],
             [False, False, False]],
            1,
            [False, False, True],
            jnp.bool_
        ),
        # Node 2 ancestors in simple chain
        (
            [[False, True, False],
             [False, False, True],
             [False, False, False]],
            2,
            [False, False, False],
            jnp.bool_
        ),
        # Disconnected graph
        (
            [[False, False, False],
             [False, False, False],
             [False, False, False]],
            0,
            [False, False, False],
            jnp.bool_
        ),
        # DAG: 0 depends on 1 and 2. 1 depends on 3. 2 depends on 3.
        # Row 0: parents 1, 2
        # Row 1: parent 3
        # Row 2: parent 3
        # Row 3: no parents
        (
            [[False, True, True, False],
             [False, False, False, True],
             [False, False, False, True],
             [False, False, False, False]],
            0,
            [False, True, True, True],
            jnp.bool_
        ),
        # Cycle: 0 depends on 1. 1 depends on 2. 2 depends on 0.
        # Row 0: parent 1
        # Row 1: parent 2
        # Row 2: parent 0
        (
            [[False, True, False],
             [False, False, True],
             [True, False, False]],
            0,
            [True, True, True],
            jnp.bool_
        ),
        # Self-loop (immediate): 0 depends on 0
        # Row 0: parent 0
        (
            [[True, False],
             [False, False]],
            0,
            [False, False],  # immediate self-loops are ignored per memory
            jnp.bool_
        )
    ]
)
def test_find_ancestors_jax(adj_matrix, node, expected_ancestors, dtype):
    mask = jnp.array(adj_matrix, dtype=dtype)
    expected = jnp.array(expected_ancestors, dtype=jnp.bool_)

    result = find_ancestors_jax(mask, node)

    # Assert result is boolean array
    assert result.dtype == jnp.bool_

    # Assert result matches expected (flake8 compliant)
    assert (result == expected).all()
