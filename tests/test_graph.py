
import pytest
import jax
import jax.numpy as jnp
import os

os.environ["JAX_PLATFORMS"] = "cpu"

from gensbi_examples.graph import find_ancestors_jax

@pytest.fixture
def run_jax_on_cpu():
    with jax.default_device(jax.devices("cpu")[0]):
        yield

def test_find_ancestors_diamond(run_jax_on_cpu):
    # Nodes:
    # 0: GP1
    # 1: GP2
    # 2: P1 (parents: GP1)
    # 3: P2 (parents: GP2)
    # 4: C (parents: P1, P2)

    num_nodes = 5
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.int32)

    # mask[child, parent] = 1
    # This implies adjacency matrix M[i, j] = 1 => j -> i (j is parent of i)
    # This matches the convention used in find_ancestors_jax based on my analysis.

    mask = mask.at[2, 0].set(1) # GP1 -> P1
    mask = mask.at[3, 1].set(1) # GP2 -> P2
    mask = mask.at[4, 2].set(1) # P1 -> C
    mask = mask.at[4, 3].set(1) # P2 -> C

    # Find ancestors of C (node 4)
    ancestors_mask = find_ancestors_jax(mask, 4)
    ancestors_indices = jnp.where(ancestors_mask)[0]
    ancestors_indices = jnp.sort(ancestors_indices)

    expected_ancestors = jnp.array([0, 1, 2, 3])

    assert jnp.array_equal(ancestors_indices, expected_ancestors), \
        f"Expected {expected_ancestors}, got {ancestors_indices}"

def test_find_ancestors_linear(run_jax_on_cpu):
    # 0 -> 1 -> 2
    num_nodes = 3
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.int32)
    mask = mask.at[1, 0].set(1)
    mask = mask.at[2, 1].set(1)

    ancestors_mask = find_ancestors_jax(mask, 2)
    ancestors_indices = jnp.where(ancestors_mask)[0]
    ancestors_indices = jnp.sort(ancestors_indices)

    expected_ancestors = jnp.array([0, 1])
    assert jnp.array_equal(ancestors_indices, expected_ancestors)

def test_find_ancestors_tree(run_jax_on_cpu):
    # 0 -> 1, 0 -> 2
    # 1 -> 3
    # 2 -> 3
    # Wait, this is diamond again.
    # Proper tree (descendants): Root -> Children.
    # But we look at ancestors (parents).
    # Tree where arrows point to child.
    # 0 -> 1, 0 -> 2.
    # Ancestors of 1 is {0}.
    # Ancestors of 2 is {0}.

    num_nodes = 3
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.int32)
    mask = mask.at[1, 0].set(1)
    mask = mask.at[2, 0].set(1)

    ancestors_mask = find_ancestors_jax(mask, 1)
    ancestors_indices = jnp.where(ancestors_mask)[0]
    assert jnp.array_equal(ancestors_indices, jnp.array([0]))

    ancestors_mask = find_ancestors_jax(mask, 2)
    ancestors_indices = jnp.where(ancestors_mask)[0]
    assert jnp.array_equal(ancestors_indices, jnp.array([0]))

def test_find_ancestors_cycle(run_jax_on_cpu):
    # 0 -> 1 -> 0
    num_nodes = 2
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.int32)
    mask = mask.at[1, 0].set(1)
    mask = mask.at[0, 1].set(1)

    # Ancestors of 0: {1, 0} (since 1 is parent, and 0 is parent of 1)
    # Usually strictly ancestors exclude self unless cycle.
    # Here 0 is ancestor of 0 via cycle.

    ancestors_mask = find_ancestors_jax(mask, 0)
    ancestors_indices = jnp.where(ancestors_mask)[0]
    ancestors_indices = jnp.sort(ancestors_indices)

    expected_ancestors = jnp.array([0, 1])
    assert jnp.array_equal(ancestors_indices, expected_ancestors)

def test_find_ancestors_disconnected(run_jax_on_cpu):
    # 0 -> 1   2 -> 3
    num_nodes = 4
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.int32)
    mask = mask.at[1, 0].set(1)
    mask = mask.at[3, 2].set(1)

    # Ancestors of 1: {0}
    ancestors_mask = find_ancestors_jax(mask, 1)
    ancestors_indices = jnp.where(ancestors_mask)[0]
    assert jnp.array_equal(ancestors_indices, jnp.array([0]))

    # Ancestors of 3: {2}
    ancestors_mask = find_ancestors_jax(mask, 3)
    ancestors_indices = jnp.where(ancestors_mask)[0]
    assert jnp.array_equal(ancestors_indices, jnp.array([2]))

    # Ancestors of 0: {}
    ancestors_mask = find_ancestors_jax(mask, 0)
    assert not jnp.any(ancestors_mask)

def test_find_ancestors_self_loop(run_jax_on_cpu):
    # 0 -> 0
    num_nodes = 1
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.int32)
    mask = mask.at[0, 0].set(1)

    # Ancestors of 0: {} (immediate self loop ignored by implementation)
    # The current implementation has `j != current_node` check.
    # So 0 is not added as ancestor of 0.

    ancestors_mask = find_ancestors_jax(mask, 0)
    assert not jnp.any(ancestors_mask)
