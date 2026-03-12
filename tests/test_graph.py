import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp  # noqa: E402
from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_jax_chain():
    # A -> B -> C
    # Rows are children, columns are parents.
    # Node 0 (A), Node 1 (B), Node 2 (C)
    # B has parent A. C has parent B.
    mask = jnp.array([
        [False, False, False],
        [True, False, False],
        [False, True, False]
    ], dtype=jnp.bool_)

    # Ancestors of C (Node 2) are A (0) and B (1)
    ans_c = find_ancestors_jax(mask, 2)
    expected_c = jnp.array([True, True, False], dtype=jnp.bool_)
    assert (ans_c == expected_c).all()

    # Ancestors of B (Node 1) is A (0)
    ans_b = find_ancestors_jax(mask, 1)
    expected_b = jnp.array([True, False, False], dtype=jnp.bool_)
    assert (ans_b == expected_b).all()

    # Ancestors of A (Node 0) is none
    ans_a = find_ancestors_jax(mask, 0)
    expected_a = jnp.array([False, False, False], dtype=jnp.bool_)
    assert (ans_a == expected_a).all()


def test_find_ancestors_jax_branching():
    # A -> B, A -> C
    # Rows are children, columns are parents.
    # Node 0 (A), Node 1 (B), Node 2 (C)
    mask = jnp.array([
        [False, False, False],
        [True, False, False],
        [True, False, False]
    ], dtype=jnp.bool_)

    # Ancestors of C (Node 2) is A (0)
    ans_c = find_ancestors_jax(mask, 2)
    expected_c = jnp.array([True, False, False], dtype=jnp.bool_)
    assert (ans_c == expected_c).all()

    # Ancestors of B (Node 1) is A (0)
    ans_b = find_ancestors_jax(mask, 1)
    expected_b = jnp.array([True, False, False], dtype=jnp.bool_)
    assert (ans_b == expected_b).all()

    # Ancestors of A (Node 0) is none
    ans_a = find_ancestors_jax(mask, 0)
    expected_a = jnp.array([False, False, False], dtype=jnp.bool_)
    assert (ans_a == expected_a).all()


def test_find_ancestors_jax_multiple_parents():
    # B -> A, C -> A
    # Rows are children, columns are parents.
    # Node 0 (A), Node 1 (B), Node 2 (C)
    # A has parents B and C.
    mask = jnp.array([
        [False, True, True],
        [False, False, False],
        [False, False, False]
    ], dtype=jnp.bool_)

    # Ancestors of A (Node 0) are B (1) and C (2)
    ans_a = find_ancestors_jax(mask, 0)
    expected_a = jnp.array([False, True, True], dtype=jnp.bool_)
    assert (ans_a == expected_a).all()

    # Ancestors of B (Node 1) is none
    ans_b = find_ancestors_jax(mask, 1)
    expected_b = jnp.array([False, False, False], dtype=jnp.bool_)
    assert (ans_b == expected_b).all()


def test_find_ancestors_jax_cycle():
    # A -> B -> C -> A
    # Rows are children, columns are parents.
    # Node 0 (A), Node 1 (B), Node 2 (C)
    mask = jnp.array([
        [False, False, True],  # A has parent C
        [True, False, False],  # B has parent A
        [False, True, False]   # C has parent B
    ], dtype=jnp.bool_)

    # For node C (2), ancestors are A (0), B (1), and itself C (2)
    ans_c = find_ancestors_jax(mask, 2)
    expected_c = jnp.array([True, True, True], dtype=jnp.bool_)
    assert (ans_c == expected_c).all()

    # Node A (0), ancestors are B, C, A
    ans_a = find_ancestors_jax(mask, 0)
    expected_a = jnp.array([True, True, True], dtype=jnp.bool_)
    assert (ans_a == expected_a).all()
