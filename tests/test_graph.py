import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp  # noqa: E402

from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_jax_simple_chain():
    # A -> B -> C
    # Adjacency matrix: rows are children, columns are parents
    # M[i, j] = 1 means j -> i
    # Nodes: 0: A, 1: B, 2: C
    # 0 -> 1 (A -> B)
    # 1 -> 2 (B -> C)
    mask = jnp.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
    ], dtype=jnp.bool_)

    # Ancestors of 2 (C) should be 0 (A) and 1 (B)
    ancestors_2 = find_ancestors_jax(mask, 2)
    expected_2 = jnp.array([True, True, False], dtype=jnp.bool_)
    assert (ancestors_2 == expected_2).all()

    # Ancestors of 1 (B) should be 0 (A)
    ancestors_1 = find_ancestors_jax(mask, 1)
    expected_1 = jnp.array([True, False, False], dtype=jnp.bool_)
    assert (ancestors_1 == expected_1).all()

    # Ancestors of 0 (A) should be empty
    ancestors_0 = find_ancestors_jax(mask, 0)
    expected_0 = jnp.array([False, False, False], dtype=jnp.bool_)
    assert (ancestors_0 == expected_0).all()


def test_find_ancestors_jax_multiple_parents():
    # A -> C, B -> C
    # Nodes: 0: A, 1: B, 2: C
    mask = jnp.array([
        [0, 0, 0],
        [0, 0, 0],
        [1, 1, 0],
    ], dtype=jnp.bool_)

    # Ancestors of 2 (C) should be 0 (A) and 1 (B)
    ancestors_2 = find_ancestors_jax(mask, 2)
    expected_2 = jnp.array([True, True, False], dtype=jnp.bool_)
    assert (ancestors_2 == expected_2).all()


def test_find_ancestors_jax_cycle():
    # A -> B -> C -> A
    # Nodes: 0: A, 1: B, 2: C
    mask = jnp.array([
        [0, 0, 1],  # C -> A
        [1, 0, 0],  # A -> B
        [0, 1, 0],  # B -> C
    ], dtype=jnp.bool_)

    # Because of the cycle, every node is an ancestor of every other node
    # find_ancestors_jax doesn't set self as ancestor initially,
    # but A -> B -> C -> A means A is its own ancestor.
    # Ancestors of 0 (A) should be all true including itself
    ancestors_0 = find_ancestors_jax(mask, 0)
    expected = jnp.array([True, True, True], dtype=jnp.bool_)
    assert (ancestors_0 == expected).all()


def test_find_ancestors_jax_self_loop():
    # A -> A, A -> B
    # Nodes: 0: A, 1: B
    mask = jnp.array([
        [1, 0],  # A -> A
        [1, 0],  # A -> B
    ], dtype=jnp.bool_)

    # find_ancestors_jax ignores immediate self-loops
    # so A's ancestors should be empty
    ancestors_0 = find_ancestors_jax(mask, 0)
    expected_0 = jnp.array([False, False], dtype=jnp.bool_)
    assert (ancestors_0 == expected_0).all()

    # Ancestors of 1 (B) should be A
    ancestors_1 = find_ancestors_jax(mask, 1)
    expected_1 = jnp.array([True, False], dtype=jnp.bool_)
    assert (ancestors_1 == expected_1).all()


def test_find_ancestors_jax_disconnected():
    # A, B
    mask = jnp.array([
        [0, 0],
        [0, 0],
    ], dtype=jnp.bool_)

    ancestors_0 = find_ancestors_jax(mask, 0)
    expected_0 = jnp.array([False, False], dtype=jnp.bool_)
    assert (ancestors_0 == expected_0).all()
