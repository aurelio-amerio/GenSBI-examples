import os

# Set device
os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402
from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


@pytest.mark.parametrize(
    "node, expected_ancestors",
    [
        (0, [False, False, False, False]),
        (1, [True, False, False, False]),
        (2, [True, True, False, True]),
        (3, [True, False, False, False]),
    ],
)
def test_find_ancestors_jax_basic(node, expected_ancestors):
    # Graph structure:
    # 0 -> 1 -> 2
    # 0 -> 3 -> 2
    # Adjacency matrix: rows are children, columns are parents
    # M[i, j] = True means j -> i
    mask = jnp.zeros((4, 4), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)
    mask = mask.at[3, 0].set(True)
    mask = mask.at[2, 1].set(True)
    mask = mask.at[2, 3].set(True)

    ancestors = find_ancestors_jax(mask, node)
    expected = jnp.array(expected_ancestors, dtype=jnp.bool_)

    assert (ancestors == expected).all()


def test_find_ancestors_jax_isolated():
    # Graph with isolated nodes
    mask = jnp.zeros((3, 3), dtype=jnp.bool_)

    for i in range(3):
        ancestors = find_ancestors_jax(mask, i)
        expected = jnp.array([False, False, False], dtype=jnp.bool_)
        assert (ancestors == expected).all()


def test_find_ancestors_jax_chain():
    # Graph: 0 -> 1 -> 2 -> 3 -> 4
    mask = jnp.zeros((5, 5), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)
    mask = mask.at[2, 1].set(True)
    mask = mask.at[3, 2].set(True)
    mask = mask.at[4, 3].set(True)

    expected_0 = jnp.array([False, False, False, False, False], dtype=jnp.bool_)
    assert (find_ancestors_jax(mask, 0) == expected_0).all()

    expected_4 = jnp.array([True, True, True, True, False], dtype=jnp.bool_)
    assert (find_ancestors_jax(mask, 4) == expected_4).all()


def test_find_ancestors_jax_self_loop_handling():
    # Graph: 0 -> 1 -> 2, with 2 having a self-loop 2 -> 2
    # According to find_ancestors_jax, immediate self-loops aren't added
    # since `j != current_node` check is done, but let's test behavior
    mask = jnp.zeros((3, 3), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)
    mask = mask.at[2, 1].set(True)
    mask = mask.at[2, 2].set(True)

    ancestors = find_ancestors_jax(mask, 2)
    expected = jnp.array([True, True, False], dtype=jnp.bool_)
    assert (ancestors == expected).all()
