import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp  # noqa: E402
from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_jax():
    # Adjacency matrix: rows are children, columns are parents
    # M_ij = 1 implies edge j -> i
    # Let's create a simple graph: 0 -> 1 -> 2
    # So 1 has parent 0, 2 has parent 1
    # Nodes: 0, 1, 2
    mask = jnp.array([
        [0, 0, 0],  # Node 0 has no parents
        [1, 0, 0],  # Node 1 has parent 0
        [0, 1, 0],  # Node 2 has parent 1
    ], dtype=jnp.bool_)

    # Ancestors of 0: none
    ans_0 = find_ancestors_jax(mask, 0)
    assert (ans_0 == jnp.array([False, False, False])).all()

    # Ancestors of 1: {0}
    ans_1 = find_ancestors_jax(mask, 1)
    assert (ans_1 == jnp.array([True, False, False])).all()

    # Ancestors of 2: {0, 1}
    ans_2 = find_ancestors_jax(mask, 2)
    assert (ans_2 == jnp.array([True, True, False])).all()


def test_find_ancestors_jax_cycle():
    # Graph with cycle: 0 -> 1 -> 2 -> 0
    # M_ij = 1 implies edge j -> i
    mask = jnp.array([
        [0, 0, 1],  # Node 0 has parent 2
        [1, 0, 0],  # Node 1 has parent 0
        [0, 1, 0],  # Node 2 has parent 1
    ], dtype=jnp.bool_)

    # Because of the cycle, every node is an ancestor of every other node,
    # and they should be marked as their own ancestors too, EXCEPT the
    # initial node which ignores immediate self-loop but does find itself via cycle.
    ans_0 = find_ancestors_jax(mask, 0)
    assert (ans_0 == jnp.array([True, True, True])).all()

    ans_1 = find_ancestors_jax(mask, 1)
    assert (ans_1 == jnp.array([True, True, True])).all()

    ans_2 = find_ancestors_jax(mask, 2)
    assert (ans_2 == jnp.array([True, True, True])).all()
