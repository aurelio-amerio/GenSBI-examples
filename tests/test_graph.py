import os

# Set device to CPU before importing jax to ensure tests run reliably
os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp  # noqa: E402
from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_jax_linear_chain():
    # Linear chain: 0 -> 1 -> 2 -> 3
    mask = jnp.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ], dtype=jnp.bool_)

    # Ancestors of 3 should be 0, 1, 2
    ancestors = find_ancestors_jax(mask, 3)
    expected = jnp.array([True, True, True, False])
    assert (ancestors == expected).all()


def test_find_ancestors_jax_branching():
    # Multiple parents (V-structure): 0 -> 2, 1 -> 2
    mask = jnp.array([
        [0, 0, 0],
        [0, 0, 0],
        [1, 1, 0]
    ], dtype=jnp.bool_)

    ancestors = find_ancestors_jax(mask, 2)
    expected = jnp.array([True, True, False])
    assert (ancestors == expected).all()


def test_find_ancestors_jax_multi_hop_branching():
    # 0 -> 1 -> 3
    # 2 -> 3
    # 4 -> 2
    mask = jnp.array([
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=jnp.bool_)

    # Ancestors of 3 should be 0, 1, 2, 4
    ancestors = find_ancestors_jax(mask, 3)
    expected = jnp.array([True, True, True, False, True])
    assert (ancestors == expected).all()


def test_find_ancestors_jax_cycle():
    # Cycle: 0 -> 1 -> 2 -> 0
    mask = jnp.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=jnp.bool_)

    # Ancestors of 0 should be 1, 2 (and 0 is its own ancestor due to cycle)
    ancestors = find_ancestors_jax(mask, 0)
    expected = jnp.array([True, True, True])
    assert (ancestors == expected).all()


def test_find_ancestors_jax_self_loop():
    # 0 -> 1, 1 -> 1 (self loop)
    mask = jnp.array([
        [0, 0],
        [1, 1]
    ], dtype=jnp.bool_)

    # Self loop is explicitly ignored in code (j != current_node)
    # So ancestors of 1 should be just 0
    ancestors = find_ancestors_jax(mask, 1)
    expected = jnp.array([True, False])
    assert (ancestors == expected).all()


def test_find_ancestors_jax_no_ancestors():
    # Disconnected graph
    mask = jnp.zeros((3, 3), dtype=jnp.bool_)

    ancestors = find_ancestors_jax(mask, 0)
    expected = jnp.array([False, False, False])
    assert (ancestors == expected).all()
