import os

# select device
os.environ["JAX_PLATFORMS"] = "cpu"

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: F401
from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_basic_dag():
    """
    Test finding ancestors in a simple linear DAG.
    Graph: 0 -> 1 -> 2
    Matrix (Row=Child, Col=Parent):
       0 1 2
    0 [0 0 0]
    1 [1 0 0]
    2 [0 1 0]
    """
    matrix = jnp.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=jnp.int32)

    # Ancestors of 2: {0, 1}
    ancestors_2 = find_ancestors_jax(matrix, 2)
    assert ancestors_2[0]
    assert ancestors_2[1]
    assert not ancestors_2[2]

    # Ancestors of 1: {0}
    ancestors_1 = find_ancestors_jax(matrix, 1)
    assert ancestors_1[0]
    assert not ancestors_1[1]
    assert not ancestors_1[2]

    # Ancestors of 0: {}
    ancestors_0 = find_ancestors_jax(matrix, 0)
    assert not jnp.any(ancestors_0)


def test_find_ancestors_branching_dag():
    r"""
    Test finding ancestors in a diamond-shaped DAG.
    Graph:
      0
     / \
    1   2
     \ /
      3

    Edges: 0->1, 0->2, 1->3, 2->3
    Matrix (Row=Child, Col=Parent):
       0 1 2 3
    0 [0 0 0 0]
    1 [1 0 0 0]
    2 [1 0 0 0]
    3 [0 1 1 0]
    """
    matrix = jnp.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 1, 0]
    ], dtype=jnp.int32)

    # Ancestors of 3: {0, 1, 2}
    ancestors_3 = find_ancestors_jax(matrix, 3)
    expected = jnp.array([True, True, True, False])
    assert jnp.all(ancestors_3 == expected)

    # Ancestors of 1: {0}
    ancestors_1 = find_ancestors_jax(matrix, 1)
    expected_1 = jnp.array([True, False, False, False])
    assert jnp.all(ancestors_1 == expected_1)

    # Ancestors of 2: {0}
    ancestors_2 = find_ancestors_jax(matrix, 2)
    expected_2 = jnp.array([True, False, False, False])
    assert jnp.all(ancestors_2 == expected_2)

    # Ancestors of 0: {}
    ancestors_0 = find_ancestors_jax(matrix, 0)
    assert not jnp.any(ancestors_0)


def test_find_ancestors_disconnected():
    """
    Test finding ancestors in a graph with disconnected components.
    Graph: 0->1   2->3
    Matrix:
       0 1 2 3
    0 [0 0 0 0]
    1 [1 0 0 0]
    2 [0 0 0 0]
    3 [0 0 1 0]
    """
    matrix = jnp.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0]
    ], dtype=jnp.int32)

    # Ancestors of 1: {0}
    ancestors_1 = find_ancestors_jax(matrix, 1)
    assert ancestors_1[0]
    assert not ancestors_1[2]
    assert not ancestors_1[3]

    # Ancestors of 3: {2}
    ancestors_3 = find_ancestors_jax(matrix, 3)
    assert ancestors_3[2]
    assert not ancestors_3[0]
    assert not ancestors_3[1]


def test_find_ancestors_cycle():
    """
    Test finding ancestors in a cyclic graph (robustness check).
    Graph: 0 <-> 1
    Matrix:
       0 1
    0 [0 1]
    1 [1 0]
    """
    matrix = jnp.array([
        [0, 1],
        [1, 0]
    ], dtype=jnp.int32)

    # Ancestors of 0: {1} and since 1->0->1, 0 is ancestor of 1 is ancestor of 0...
    # The algorithm should visit both nodes and terminate.
    ancestors_0 = find_ancestors_jax(matrix, 0)
    assert ancestors_0[1]
    # In a cycle, the node itself might be marked as ancestor if encountered again
    assert ancestors_0[0]

    ancestors_1 = find_ancestors_jax(matrix, 1)
    assert ancestors_1[0]
    assert ancestors_1[1]
