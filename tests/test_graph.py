import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp  # noqa: E402

from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_jax_simple_dag():
    # 0 -> 1 -> 2
    # 3 -> 1
    # Adjacency matrix: rows are children, columns are parents.
    # M[i, j] = 1 means j is parent of i.
    mask = jnp.zeros((4, 4), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)
    mask = mask.at[2, 1].set(True)
    mask = mask.at[1, 3].set(True)

    # Ancestors of 2 should be 1, 0, 3
    ancestors = find_ancestors_jax(mask, 2)
    assert ancestors[0]
    assert ancestors[1]
    assert ancestors[3]
    assert not ancestors[2]

    # Ancestors of 1 should be 0, 3
    ancestors = find_ancestors_jax(mask, 1)
    assert ancestors[0]
    assert ancestors[3]
    assert not ancestors[1]
    assert not ancestors[2]

    # Ancestors of 0 should be none
    ancestors = find_ancestors_jax(mask, 0)
    assert not jnp.any(ancestors)


def test_find_ancestors_jax_cycle():
    # 0 -> 1 -> 2 -> 0
    mask = jnp.zeros((3, 3), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)
    mask = mask.at[2, 1].set(True)
    mask = mask.at[0, 2].set(True)

    # Ancestors of 0 should be 1, 2, and 0 (itself, due to cycle)
    ancestors = find_ancestors_jax(mask, 0)
    assert ancestors[0]
    assert ancestors[1]
    assert ancestors[2]


def test_find_ancestors_jax_self_loop():
    # 0 -> 0
    mask = jnp.zeros((1, 1), dtype=jnp.bool_)
    mask = mask.at[0, 0].set(True)

    # It explicitly ignores immediate self-loops (e.g., A->A) during traversal
    ancestors = find_ancestors_jax(mask, 0)
    assert not ancestors[0]


def test_find_ancestors_jax_int_mask():
    # 0 -> 1
    mask = jnp.zeros((2, 2), dtype=jnp.int32)
    mask = mask.at[1, 0].set(1)

    ancestors = find_ancestors_jax(mask, 1)
    assert ancestors[0]
    assert not ancestors[1]


def test_find_ancestors_jax_disconnected():
    # 0, 1 disconnected
    mask = jnp.zeros((2, 2), dtype=jnp.bool_)

    ancestors = find_ancestors_jax(mask, 0)
    assert not ancestors[0]
    assert not ancestors[1]

    ancestors = find_ancestors_jax(mask, 1)
    assert not ancestors[0]
    assert not ancestors[1]
