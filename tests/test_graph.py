import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp  # noqa: E402

from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_jax_simple():
    # Linear graph: 0 <- 1 <- 2
    # Rows are children, columns are parents
    # M[i, j] = 1 implies edge j -> i
    mask = jnp.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0]
    ], dtype=jnp.bool_)

    # Ancestors of 0 should be 1 and 2
    is_ancestor = find_ancestors_jax(mask, 0)
    expected = jnp.array([False, True, True], dtype=jnp.bool_)
    assert jnp.all(is_ancestor == expected)

    # Ancestors of 1 should be 2
    is_ancestor = find_ancestors_jax(mask, 1)
    expected = jnp.array([False, False, True], dtype=jnp.bool_)
    assert jnp.all(is_ancestor == expected)

    # Ancestors of 2 should be none
    is_ancestor = find_ancestors_jax(mask, 2)
    expected = jnp.array([False, False, False], dtype=jnp.bool_)
    assert jnp.all(is_ancestor == expected)


def test_find_ancestors_jax_branching():
    # Graph:
    # 3 -> 1 -> 0
    # 4 -> 2 -> 0
    mask = jnp.array([
        [0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=jnp.bool_)

    # Ancestors of 0 should be 1, 2, 3, 4
    is_ancestor = find_ancestors_jax(mask, 0)
    expected = jnp.array([False, True, True, True, True], dtype=jnp.bool_)
    assert jnp.all(is_ancestor == expected)

    # Ancestors of 1 should be 3
    is_ancestor = find_ancestors_jax(mask, 1)
    expected = jnp.array([False, False, False, True, False], dtype=jnp.bool_)
    assert jnp.all(is_ancestor == expected)

    # Ancestors of 2 should be 4
    is_ancestor = find_ancestors_jax(mask, 2)
    expected = jnp.array([False, False, False, False, True], dtype=jnp.bool_)
    assert jnp.all(is_ancestor == expected)


def test_find_ancestors_jax_cyclic():
    # Graph: 0 <- 1 <- 2 <- 0
    mask = jnp.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ], dtype=jnp.bool_)

    # Ancestors of 0 should be 1, 2, and itself (0) due to cycle
    # Wait, the function ignores immediate self-loops `(j != current_node)`,
    # but for larger cycles like 0 <- 1 <- 2 <- 0, it should find 0, 1, 2 as ancestors of 0.
    is_ancestor = find_ancestors_jax(mask, 0)
    expected = jnp.array([True, True, True], dtype=jnp.bool_)
    assert jnp.all(is_ancestor == expected)


def test_find_ancestors_jax_self_loop():
    # Graph: 0 <- 0
    mask = jnp.array([
        [1]
    ], dtype=jnp.bool_)

    # Immediate self loops are explicitly ignored: `j != current_node`
    is_ancestor = find_ancestors_jax(mask, 0)
    expected = jnp.array([False], dtype=jnp.bool_)
    assert jnp.all(is_ancestor == expected)
