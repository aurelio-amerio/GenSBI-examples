import os
import jax
import jax.numpy as jnp
from gensbi_examples.graph import find_ancestors_jax

# Set JAX to CPU for testing
os.environ["JAX_PLATFORMS"] = "cpu"


def test_find_ancestors_jax():
    """Test find_ancestors_jax with a simple DAG."""
    # Example adjacency matrix (DAG)
    # 0 -> 1 -> 3
    # |    ^
    # v    |
    # 2 -> 4
    #
    # Adjacency matrix M[i, j] = 1 if j -> i (columns are parents, rows are children)

    num_nodes = 5
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.int32)

    # 0 -> 1
    mask = mask.at[1, 0].set(1)
    # 0 -> 2
    mask = mask.at[2, 0].set(1)
    # 2 -> 4
    mask = mask.at[4, 2].set(1)
    # 4 -> 1
    mask = mask.at[1, 4].set(1)
    # 1 -> 3
    mask = mask.at[3, 1].set(1)

    # Test ancestors for node 3
    # Ancestors of 3: 1 (direct), 4 (via 1), 2 (via 4), 0 (via 1 and 2)
    # Expected: {0, 1, 2, 4}
    node = 3
    ancestors = find_ancestors_jax(mask, node)
    expected_ancestors = jnp.array([True, True, True, False, True]) # 0, 1, 2, 4 are True
    assert jnp.all(ancestors == expected_ancestors)

    # Test ancestors for node 0 (root)
    # Expected: None
    node = 0
    ancestors = find_ancestors_jax(mask, node)
    expected_ancestors = jnp.zeros(num_nodes, dtype=jnp.bool_)
    assert jnp.all(ancestors == expected_ancestors)

    # Test ancestors for node 2
    # Ancestors of 2: 0
    node = 2
    ancestors = find_ancestors_jax(mask, node)
    expected_ancestors = jnp.array([True, False, False, False, False])
    assert jnp.all(ancestors == expected_ancestors)


def test_find_ancestors_jax_single_node():
    """Test find_ancestors_jax with a single node graph."""
    num_nodes = 1
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.int32)

    node = 0
    ancestors = find_ancestors_jax(mask, node)
    expected_ancestors = jnp.zeros(num_nodes, dtype=jnp.bool_)
    assert jnp.all(ancestors == expected_ancestors)


def test_find_ancestors_jax_disconnected():
    """Test find_ancestors_jax with a disconnected graph."""
    # 0 -> 1   2 -> 3
    num_nodes = 4
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.int32)

    mask = mask.at[1, 0].set(1)
    mask = mask.at[3, 2].set(1)

    # Ancestors of 1 should only be 0
    node = 1
    ancestors = find_ancestors_jax(mask, node)
    expected_ancestors = jnp.array([True, False, False, False])
    assert jnp.all(ancestors == expected_ancestors)

    # Ancestors of 3 should only be 2
    node = 3
    ancestors = find_ancestors_jax(mask, node)
    expected_ancestors = jnp.array([False, False, True, False])
    assert jnp.all(ancestors == expected_ancestors)


def test_find_ancestors_jax_cycle():
    """Test find_ancestors_jax behavior with a cycle (though typically for DAGs)."""
    # 0 -> 1 -> 0 (cycle)
    num_nodes = 2
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.int32)

    mask = mask.at[1, 0].set(1)
    mask = mask.at[0, 1].set(1)

    # Ancestors of 0 should include 1.
    # Due to cycle, 0 is ancestor of 1, and 1 is ancestor of 0.
    # The implementation might or might not include 0 as its own ancestor depending on implementation details.
    # Let's check what it does.

    node = 0
    ancestors = find_ancestors_jax(mask, node)

    # Based on implementation, if we visit children recursively:
    # Stack starts with [0].
    # Pop 0. Parents of 0 is {1}.
    # Add 1 to stack. Mark 1 as ancestor.
    # Pop 1. Parents of 1 is {0}.
    # 0 is NOT marked as ancestor yet (is_ancestor array).
    # Add 0 to stack?
    # The condition is `cond = value & (j != current_node) & (~is_ancestor[j])`
    # When processing 1: parent 0. j=0. current_node=1. is_ancestor[0]=False.
    # So 0 is added to stack and marked as ancestor.

    # Expected: both 0 and 1 are ancestors of each other in this cyclic definition if we include self in cycle.
    expected_ancestors = jnp.array([True, True])
    assert jnp.all(ancestors == expected_ancestors)
