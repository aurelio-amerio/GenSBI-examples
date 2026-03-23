import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp  # noqa: E402
from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_jax_simple_chain():
    # 0 -> 1 -> 2
    # Rows are children, columns are parents.
    # mask[i, j] = 1 means j is a parent of i.
    mask = jnp.array([
        [False, False, False],
        [True, False, False],  # 0 is parent of 1
        [False, True, False],  # 1 is parent of 2
    ])

    # Ancestors of 2 should be 0 and 1
    ancestors = find_ancestors_jax(mask, 2)
    expected = jnp.array([True, True, False])
    assert (ancestors == expected).all()


def test_find_ancestors_jax_multiple_parents():
    # 0 -> 2
    # 1 -> 2
    mask = jnp.array([
        [False, False, False],
        [False, False, False],
        [True, True, False],  # 0 and 1 are parents of 2
    ])

    ancestors = find_ancestors_jax(mask, 2)
    expected = jnp.array([True, True, False])
    assert (ancestors == expected).all()


def test_find_ancestors_jax_branching():
    # 0 -> 1 -> 3
    # 0 -> 2 -> 3
    mask = jnp.array([
        [False, False, False, False],
        [True, False, False, False],  # 0 is parent of 1
        [True, False, False, False],  # 0 is parent of 2
        [False, True, True, False],  # 1 and 2 are parents of 3
    ])

    ancestors = find_ancestors_jax(mask, 3)
    expected = jnp.array([True, True, True, False])
    assert (ancestors == expected).all()


def test_find_ancestors_jax_cycle():
    # 0 -> 1 -> 2 -> 0
    mask = jnp.array([
        [False, False, True],  # 2 is parent of 0
        [True, False, False],  # 0 is parent of 1
        [False, True, False],  # 1 is parent of 2
    ])

    ancestors = find_ancestors_jax(mask, 0)
    # Ancestors of 0 should be 1 and 2, and since 0 is in the cycle, it might
    # mark itself as ancestor. It explicitly ignores immediate self-loops
    # (e.g., A->A) during traversal, but will mark a node as its own ancestor
    # if a larger cycle exists (e.g., A->B->A).
    expected = jnp.array([True, True, True])
    assert (ancestors == expected).all()


def test_find_ancestors_jax_no_ancestors():
    # 0 -> 1
    # Node 0 has no parents
    mask = jnp.array([
        [False, False],
        [True, False],  # 0 is parent of 1
    ])

    ancestors = find_ancestors_jax(mask, 0)
    expected = jnp.array([False, False])
    assert (ancestors == expected).all()


def test_find_ancestors_jax_isolated_node():
    # Node 0 is isolated
    mask = jnp.array([
        [False]
    ])

    ancestors = find_ancestors_jax(mask, 0)
    expected = jnp.array([False])
    assert (ancestors == expected).all()


def test_find_ancestors_jax_self_loop():
    # 0 -> 0 (self-loop)
    mask = jnp.array([
        [True]
    ])

    ancestors = find_ancestors_jax(mask, 0)
    # The current implementation explicitly ignores immediate self-loops
    expected = jnp.array([False])
    assert (ancestors == expected).all()
