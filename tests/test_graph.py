import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp  # noqa: E402
from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_jax_linear():
    """Test a simple linear graph 0 -> 1 -> 2."""
    # Rows are children, columns are parents
    mask = jnp.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
    ], dtype=jnp.bool_)

    ancestors = find_ancestors_jax(mask, 2)
    expected = jnp.array([True, True, False])
    assert (ancestors == expected).all()


def test_find_ancestors_jax_branching():
    """Test a branching graph where multiple paths lead to a node."""
    # 0 -> 2, 1 -> 2, 2 -> 3
    mask = jnp.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 0]
    ], dtype=jnp.bool_)

    ancestors = find_ancestors_jax(mask, 3)
    expected = jnp.array([True, True, True, False])
    assert (ancestors == expected).all()


def test_find_ancestors_jax_deep_branching():
    """Test a graph where early padded nodes could be overwritten in old logic."""
    # 0 -> 1 -> 3
    # 2 -> 3
    mask = jnp.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 1, 0]
    ], dtype=jnp.bool_)

    ancestors = find_ancestors_jax(mask, 3)
    expected = jnp.array([True, True, True, False])
    assert (ancestors == expected).all()


def test_find_ancestors_jax_cycle():
    """Test a graph with a cycle."""
    # 0 -> 1 -> 0
    mask = jnp.array([
        [0, 1],
        [1, 0]
    ], dtype=jnp.bool_)

    ancestors = find_ancestors_jax(mask, 0)
    expected = jnp.array([True, True])
    assert (ancestors == expected).all()


def test_find_ancestors_jax_self_loop():
    """Test a node with a self-loop (ignored by find_ancestors_jax)."""
    # 0 -> 0
    mask = jnp.array([
        [1]
    ], dtype=jnp.bool_)

    ancestors = find_ancestors_jax(mask, 0)
    expected = jnp.array([False])
    assert (ancestors == expected).all()


def test_find_ancestors_jax_int32():
    """Test graph with int32 adjacency matrix."""
    # 0 -> 1 -> 2
    mask = jnp.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
    ], dtype=jnp.int32)

    ancestors = find_ancestors_jax(mask, 2)
    expected = jnp.array([True, True, False])
    assert (ancestors == expected).all()
