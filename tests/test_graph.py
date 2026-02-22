
import os
import jax.numpy as jnp
import pytest
from gensbi_examples.graph import find_ancestors_jax

# Set JAX to CPU
os.environ["JAX_PLATFORMS"] = "cpu"


@pytest.fixture
def linear_graph():
    # 0 -> 1 -> 2
    num_nodes = 3
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)  # 0 parent of 1
    mask = mask.at[2, 1].set(True)  # 1 parent of 2
    return mask


@pytest.fixture
def branching_graph():
    # 0 -> 2
    # 1 -> 2
    num_nodes = 3
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.bool_)
    mask = mask.at[2, 0].set(True)
    mask = mask.at[2, 1].set(True)
    return mask


@pytest.fixture
def diamond_graph():
    # 4 -> 1 -> 3
    # 0 -> 2 -> 3
    # Ancestors of 3: {0, 1, 2, 4}
    num_nodes = 5
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.bool_)
    mask = mask.at[1, 4].set(True)
    mask = mask.at[3, 1].set(True)
    mask = mask.at[2, 0].set(True)
    mask = mask.at[3, 2].set(True)
    return mask


@pytest.mark.parametrize(
    "graph_fixture, node, expected_ancestors",
    [
        ("linear_graph", 2, {0, 1}),
        ("linear_graph", 1, {0}),
        ("linear_graph", 0, set()),
        ("branching_graph", 2, {0, 1}),
        ("branching_graph", 1, set()),
        ("diamond_graph", 3, {0, 1, 2, 4}),
        ("diamond_graph", 1, {4}),
        ("diamond_graph", 0, set()),
    ]
)
def test_find_ancestors(graph_fixture, node, expected_ancestors, request):
    mask = request.getfixturevalue(graph_fixture)
    ancestors_mask = find_ancestors_jax(mask, node)

    # Convert bool array to set of indices
    ancestor_indices = set(int(x) for x in jnp.where(ancestors_mask)[0])

    msg = (f"For node {node}, expected {expected_ancestors}, "
           f"got {ancestor_indices}")
    assert ancestor_indices == expected_ancestors, msg


def test_disconnected_graph():
    # 0 -> 1   2 -> 3
    num_nodes = 4
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)
    mask = mask.at[3, 2].set(True)

    # Check ancestors for 1
    ancestors_1 = find_ancestors_jax(mask, 1)
    assert jnp.array_equal(ancestors_1, jnp.array([True, False, False, False]))

    # Check ancestors for 3
    ancestors_3 = find_ancestors_jax(mask, 3)
    assert jnp.array_equal(ancestors_3, jnp.array([False, False, True, False]))


def test_cycle_graph():
    # 0 -> 1 -> 0
    num_nodes = 2
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)
    mask = mask.at[0, 1].set(True)

    # Ancestors of 0 should be {1} (and itself if it counts as cycle ancestor?)
    # The function marks visited nodes.

    ancestors_0 = find_ancestors_jax(mask, 0)
    # Expect both to be True (0 is ancestor of itself via 1)
    assert jnp.all(ancestors_0)
