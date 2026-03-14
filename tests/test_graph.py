import os

# select device
os.environ["JAX_PLATFORMS"] = "cpu"

import jax  # noqa: E402, F401
import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402, F401

from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_jax_simple_dag():
    """Test find_ancestors_jax on a simple DAG."""
    # Nodes: A(0), B(1), C(2), D(3)
    # Edges: A->B, A->C, B->D, C->D
    # Adjacency matrix: rows are children, columns are parents.
    num_nodes = 4
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)  # A -> B
    mask = mask.at[2, 0].set(True)  # A -> C
    mask = mask.at[3, 1].set(True)  # B -> D
    mask = mask.at[3, 2].set(True)  # C -> D

    # Test ancestors of A (node 0)
    ancestors_0 = find_ancestors_jax(mask, 0)
    expected_0 = jnp.array([False, False, False, False], dtype=jnp.bool_)
    assert (ancestors_0 == expected_0).all()

    # Test ancestors of B (node 1)
    ancestors_1 = find_ancestors_jax(mask, 1)
    expected_1 = jnp.array([True, False, False, False], dtype=jnp.bool_)
    assert (ancestors_1 == expected_1).all()

    # Test ancestors of C (node 2)
    ancestors_2 = find_ancestors_jax(mask, 2)
    expected_2 = jnp.array([True, False, False, False], dtype=jnp.bool_)
    assert (ancestors_2 == expected_2).all()

    # Test ancestors of D (node 3)
    ancestors_3 = find_ancestors_jax(mask, 3)
    expected_3 = jnp.array([True, True, True, False], dtype=jnp.bool_)
    assert (ancestors_3 == expected_3).all()


def test_find_ancestors_jax_cycle():
    """Test find_ancestors_jax on a graph with cycles."""
    # Nodes: A(0), B(1), C(2)
    # Edges: A->B, B->C, C->A
    num_nodes = 3
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)  # A -> B
    mask = mask.at[2, 1].set(True)  # B -> C
    mask = mask.at[0, 2].set(True)  # C -> A

    # Test ancestors of A (node 0)
    # Since there's a cycle A->B->C->A, A is its own ancestor (along with B and C).
    ancestors_0 = find_ancestors_jax(mask, 0)
    expected_0 = jnp.array([True, True, True], dtype=jnp.bool_)
    assert (ancestors_0 == expected_0).all()

    # Self loops should be ignored explicitly for a single node A->A.
    # But if A is in a cycle of length > 1, it's marked as an ancestor.

    # Create graph with just a self loop on D(3)
    num_nodes_self = 4
    mask_self = jnp.zeros((num_nodes_self, num_nodes_self), dtype=jnp.bool_)
    mask_self = mask_self.at[3, 3].set(True)  # D -> D

    ancestors_3 = find_ancestors_jax(mask_self, 3)
    expected_3 = jnp.array([False, False, False, False], dtype=jnp.bool_)
    assert (ancestors_3 == expected_3).all()


def test_find_ancestors_jax_disconnected():
    """Test find_ancestors_jax on a disconnected graph."""
    num_nodes = 3
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.bool_)
    # No edges

    ancestors_0 = find_ancestors_jax(mask, 0)
    expected_0 = jnp.array([False, False, False], dtype=jnp.bool_)
    assert (ancestors_0 == expected_0).all()
