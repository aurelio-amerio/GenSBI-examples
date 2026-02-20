import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp  # noqa: E402
from gensbi_examples.graph import find_ancestors_jax  # noqa: E402

# The adjacency matrix convention used by find_ancestors_jax is:
# mask[child, parent] = 1 (or True) means there is an edge parent -> child.
# Rows represent children, columns represent parents.


def test_find_ancestors_basic_dag():
    """Test find_ancestors_jax with a simple DAG: 0->1->2, 0->2"""
    # Graph:
    # 0 -> 1
    # 1 -> 2
    # 0 -> 2

    # Matrix (Row=Child, Col=Parent):
    # 0: []         -> row 0: [0, 0, 0]
    # 1: [0]        -> row 1: [1, 0, 0]
    # 2: [0, 1]     -> row 2: [1, 1, 0]

    mask = jnp.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0]
    ], dtype=jnp.bool_)

    # Ancestors of 2 should be {0, 1}
    ancestors_2 = find_ancestors_jax(mask, 2)
    expected_2 = jnp.array([True, True, False])
    assert jnp.all(ancestors_2 == expected_2)

    # Ancestors of 1 should be {0}
    ancestors_1 = find_ancestors_jax(mask, 1)
    expected_1 = jnp.array([True, False, False])
    assert jnp.all(ancestors_1 == expected_1)

    # Ancestors of 0 should be {}
    ancestors_0 = find_ancestors_jax(mask, 0)
    expected_0 = jnp.array([False, False, False])
    assert jnp.all(ancestors_0 == expected_0)


def test_find_ancestors_cycle():
    """Test find_ancestors_jax with a cycle: 0 <-> 1"""
    # Graph:
    # 0 -> 1
    # 1 -> 0

    # Matrix (Row=Child, Col=Parent):
    # 0: [1] -> row 0: [0, 1]
    # 1: [0] -> row 1: [1, 0]

    mask = jnp.array([
        [0, 1],
        [1, 0]
    ], dtype=jnp.bool_)

    # Ancestors of 0 should include 1. Since 1 depends on 0,
    # 0 is an ancestor of 1.
    # So ancestors(0) -> {1} -> {1, 0} (since 0 is parent of 1).
    # However, find_ancestors_jax implementation explicitly ignores
    # immediate self-loops, but traversing a cycle should eventually reach
    # the node itself.

    ancestors_0 = find_ancestors_jax(mask, 0)
    expected_0 = jnp.array([True, True])
    assert jnp.all(ancestors_0 == expected_0)

    ancestors_1 = find_ancestors_jax(mask, 1)
    expected_1 = jnp.array([True, True])
    assert jnp.all(ancestors_1 == expected_1)


def test_find_ancestors_disconnected():
    """Test find_ancestors_jax with disconnected components: 0->1, 2->3"""
    # Graph:
    # 0 -> 1
    # 2 -> 3

    # Matrix (Row=Child, Col=Parent):
    # 0: []
    # 1: [0]
    # 2: []
    # 3: [2]

    mask = jnp.zeros((4, 4), dtype=jnp.bool_)
    # 0 -> 1 (1 depends on 0) => mask[1, 0] = True
    mask = mask.at[1, 0].set(True)
    # 2 -> 3 (3 depends on 2) => mask[3, 2] = True
    mask = mask.at[3, 2].set(True)

    # Ancestors of 1 should be {0}
    ancestors_1 = find_ancestors_jax(mask, 1)
    assert ancestors_1[0]
    assert not ancestors_1[1]
    assert not ancestors_1[2]
    assert not ancestors_1[3]

    # Ancestors of 3 should be {2}
    ancestors_3 = find_ancestors_jax(mask, 3)
    assert ancestors_3[2]
    assert not ancestors_3[0]
    assert not ancestors_3[1]
    assert not ancestors_3[3]

    # Ancestors of 0 should be {}
    ancestors_0 = find_ancestors_jax(mask, 0)
    assert not jnp.any(ancestors_0)


def test_find_ancestors_self_loop():
    """Test find_ancestors_jax with a self-loop: 0->0"""
    # Graph:
    # 0 -> 0

    # Matrix:
    # 0: [0] -> row 0: [1]

    mask = jnp.array([[1]], dtype=jnp.bool_)

    # The implementation explicitly ignores immediate self-loops:
    # cond = value & (j != current_node) & ...
    # So ancestors of 0 should be {}

    ancestors_0 = find_ancestors_jax(mask, 0)
    expected_0 = jnp.array([False])
    assert jnp.all(ancestors_0 == expected_0)
