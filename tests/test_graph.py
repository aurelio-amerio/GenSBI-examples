import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp  # noqa: E402

from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_jax_single_node():
    """Test find_ancestors_jax on a graph with no edges."""
    M = jnp.zeros((3, 3), dtype=jnp.bool_)
    ancestors = find_ancestors_jax(M, 0)
    assert not ancestors.any()


def test_find_ancestors_jax_linear_chain():
    """Test a linear chain: 0 -> 1 -> 2."""
    M = jnp.array([
        [0, 0, 0],  # 0 has no parents
        [1, 0, 0],  # 1 has parent 0
        [0, 1, 0],  # 2 has parent 1
    ], dtype=jnp.bool_)

    # Ancestors of 2 should be 0 and 1
    ancestors = find_ancestors_jax(M, 2)
    assert ancestors[0]
    assert ancestors[1]
    assert not ancestors[2]

    # Ancestors of 1 should be 0
    ancestors = find_ancestors_jax(M, 1)
    assert ancestors[0]
    assert not ancestors[1]
    assert not ancestors[2]


def test_find_ancestors_jax_multi_branch_dag():
    """Test a DAG with multiple branches.
    0 -> 1, 0 -> 2
    1 -> 3, 2 -> 3
    3 -> 4
    """
    M = jnp.array([
        [0, 0, 0, 0, 0],  # 0
        [1, 0, 0, 0, 0],  # 1 (0 -> 1)
        [1, 0, 0, 0, 0],  # 2 (0 -> 2)
        [0, 1, 1, 0, 0],  # 3 (1 -> 3, 2 -> 3)
        [0, 0, 0, 1, 0],  # 4 (3 -> 4)
    ], dtype=jnp.bool_)

    # Ancestors of 3: 0, 1, 2
    ancestors_3 = find_ancestors_jax(M, 3)
    assert ancestors_3[0]
    assert ancestors_3[1]
    assert ancestors_3[2]
    assert not ancestors_3[3]
    assert not ancestors_3[4]

    # Ancestors of 4: 0, 1, 2, 3
    ancestors_4 = find_ancestors_jax(M, 4)
    assert ancestors_4[0]
    assert ancestors_4[1]
    assert ancestors_4[2]
    assert ancestors_4[3]
    assert not ancestors_4[4]


def test_find_ancestors_jax_immediate_self_loop():
    """Test that an immediate self-loop does not count as an ancestor."""
    M = jnp.array([
        [1, 0, 0],  # 0 has parent 0
        [1, 0, 0],  # 1 has parent 0
        [0, 1, 0],  # 2 has parent 1
    ], dtype=jnp.bool_)

    ancestors = find_ancestors_jax(M, 0)
    assert not ancestors.any()


def test_find_ancestors_jax_cycle():
    """Test a cycle: 0 -> 1 -> 2 -> 0."""
    M = jnp.array([
        [0, 1, 0],  # 0 has parent 1
        [0, 0, 1],  # 1 has parent 2
        [1, 0, 0],  # 2 has parent 0
    ], dtype=jnp.bool_)

    ancestors = find_ancestors_jax(M, 0)
    # Since 0 -> 1 -> 2 -> 0, all nodes are ancestors of 0
    assert ancestors[0]
    assert ancestors[1]
    assert ancestors[2]


def test_find_ancestors_jax_int_type():
    """Test that integer adjacency matrices work."""
    M = jnp.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
    ], dtype=jnp.int32)

    ancestors = find_ancestors_jax(M, 2)
    assert ancestors[0]
    assert ancestors[1]
    assert not ancestors[2]
