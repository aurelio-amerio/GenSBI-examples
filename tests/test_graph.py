import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp  # noqa: E402

from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_linear():
    """Test find_ancestors_jax on a linear graph 0 -> 1 -> 2."""
    mask = jnp.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
    ], dtype=jnp.bool_)

    # Ancestors of 2 should be 0 and 1
    ancestors = find_ancestors_jax(mask, 2)
    assert ancestors[0]
    assert ancestors[1]
    assert not ancestors[2]

    # Ancestors of 1 should be 0
    ancestors = find_ancestors_jax(mask, 1)
    assert ancestors[0]
    assert not ancestors[1]
    assert not ancestors[2]

    # Ancestors of 0 should be empty
    ancestors = find_ancestors_jax(mask, 0)
    assert not ancestors[0]
    assert not ancestors[1]
    assert not ancestors[2]


def test_find_ancestors_branching():
    """Test find_ancestors_jax on branching graph 0->1->3, 0->2->3."""
    mask = jnp.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 1, 0],
    ], dtype=jnp.bool_)

    # Ancestors of 3 should be 0, 1, and 2
    ancestors = find_ancestors_jax(mask, 3)
    assert ancestors[0]
    assert ancestors[1]
    assert ancestors[2]
    assert not ancestors[3]


def test_find_ancestors_cyclic():
    """Test find_ancestors_jax on a cyclic graph 0 -> 1 -> 2 -> 0."""
    mask = jnp.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
    ], dtype=jnp.bool_)

    # Ancestors of 0 should be 0, 1, 2
    ancestors = find_ancestors_jax(mask, 0)
    assert ancestors[0]
    assert ancestors[1]
    assert ancestors[2]

    # Ancestors of 1 should be 0, 1, 2
    ancestors = find_ancestors_jax(mask, 1)
    assert ancestors[0]
    assert ancestors[1]
    assert ancestors[2]


def test_find_ancestors_self_loop():
    """Test find_ancestors_jax ignoring immediate self loop 0 -> 0."""
    mask = jnp.array([
        [1, 0],
        [1, 0],
    ], dtype=jnp.bool_)

    # Immediate self loops are avoided due to j != current_node
    # in inner_body_fn prevents taking immediate self loop as ancestor.
    ancestors = find_ancestors_jax(mask, 0)
    assert not ancestors[0]
    assert not ancestors[1]

    # Ancestors of 1
    ancestors = find_ancestors_jax(mask, 1)
    assert ancestors[0]
    assert not ancestors[1]
