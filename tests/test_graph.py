import os

# select device
os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp  # noqa: E402
from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


# Adjacency matrix convention: rows are children, columns are parents
# M[i, j] = 1 means j -> i (j is parent of i)

def test_find_ancestors_linear():
    # 0 -> 1 -> 2
    # Ancestors of 2: 0, 1
    # Matrix:
    # 0: [0, 0, 0]
    # 1: [1, 0, 0] (0->1)
    # 2: [0, 1, 0] (1->2)

    mask = jnp.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=jnp.int32)

    ancestors = find_ancestors_jax(mask, 2)
    expected = jnp.array([True, True, False], dtype=bool)  # 0 and 1 are ancestors

    assert jnp.array_equal(ancestors, expected)


def test_find_ancestors_branching():
    # Diamond graph:
    # 0 -> 1
    # 0 -> 2
    # 1 -> 3
    # 2 -> 3
    # Ancestors of 3: 0, 1, 2

    # Matrix (4 nodes):
    # 0: [0, 0, 0, 0]
    # 1: [1, 0, 0, 0] (0->1)
    # 2: [1, 0, 0, 0] (0->2)
    # 3: [0, 1, 1, 0] (1->3, 2->3)

    mask = jnp.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 1, 0]
    ], dtype=jnp.int32)

    ancestors = find_ancestors_jax(mask, 3)
    expected = jnp.array([True, True, True, False], dtype=bool)

    assert jnp.array_equal(ancestors, expected)


def test_find_ancestors_complex_branching():
    # E -> A -> B
    # F -> C -> B
    # B -> D
    # Ancestors of D (5): B(4), A(2), C(3), E(0), F(1)
    # Nodes: 0:E, 1:F, 2:A, 3:C, 4:B, 5:D

    mask = jnp.zeros((6, 6), dtype=jnp.int32)
    # 0:E -> 2:A
    mask = mask.at[2, 0].set(1)
    # 1:F -> 3:C
    mask = mask.at[3, 1].set(1)
    # 2:A -> 4:B
    mask = mask.at[4, 2].set(1)
    # 3:C -> 4:B
    mask = mask.at[4, 3].set(1)
    # 4:B -> 5:D
    mask = mask.at[5, 4].set(1)

    ancestors = find_ancestors_jax(mask, 5)
    # Expected: 0, 1, 2, 3, 4 are True. 5 is False.
    expected = jnp.array([True, True, True, True, True, False], dtype=bool)

    assert jnp.array_equal(ancestors, expected), f"Ancestors found: {ancestors}, Expected: {expected}"


def test_find_ancestors_cycle():
    # 0 -> 1 -> 0
    mask = jnp.array([
        [0, 1],
        [1, 0]
    ], dtype=jnp.int32)

    ancestors = find_ancestors_jax(mask, 0)
    # Expect both to be marked if it handles cycles gracefully
    expected = jnp.array([True, True], dtype=bool)

    assert jnp.array_equal(ancestors, expected)


def test_find_ancestors_disconnected():
    # 0   1 -> 2
    # Ancestors of 0: None

    mask = jnp.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 1, 0]
    ], dtype=jnp.int32)

    ancestors = find_ancestors_jax(mask, 0)
    expected = jnp.array([False, False, False], dtype=bool)

    assert jnp.array_equal(ancestors, expected)
