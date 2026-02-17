
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402
from gensbi_examples.graph import find_ancestors_jax  # noqa: E402

def test_find_ancestors_chain():
    # 0 <- 1 <- 2
    mask = jnp.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0]
    ], dtype=jnp.int32)

    # Ancestors of 0: {1, 2}
    ancestors = find_ancestors_jax(mask, 0)
    expected = jnp.array([False, True, True])
    assert jnp.all(ancestors == expected)

def test_find_ancestors_branching():
    # 1 -> 0
    # 2 -> 0
    # 3 -> 1
    # Ancestors of 0 should be {1, 2, 3}

    mask = jnp.array([
        [0, 1, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ], dtype=jnp.int32)

    ancestors = find_ancestors_jax(mask, 0)
    expected = jnp.array([False, True, True, True])

    assert jnp.all(ancestors == expected), f"Expected {expected}, got {ancestors}"

def test_find_ancestors_disconnected():
    # 0 <- 1
    # 2 <- 3
    # Ancestors of 0: {1}

    mask = jnp.array([
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ], dtype=jnp.int32)

    ancestors = find_ancestors_jax(mask, 0)
    expected = jnp.array([False, True, False, False])
    assert jnp.all(ancestors == expected)

def test_find_ancestors_cycle():
    # 0 <- 1 <- 0
    # Ancestors of 0: {1} (and implicitly 0 via 1, but usually find_ancestors excludes self unless explicitly added?)
    # Current implementation excludes self (j != current_node).
    # If 0 <- 1, 1 is parent.
    # If 1 <- 0, 0 is parent of 1.

    # find_ancestors(0):
    # Queue: [0]
    # Parents of 0: {1} -> Add 1. Is_ancestor[1]=True.
    # Queue: [1]
    # Parents of 1: {0} -> 0 is visited? No, is_ancestor[0] is False initially.
    # But j != current_node check prevents immediate self loop. Here j=0, current=1. Distinct.
    # So 0 would be added to queue?
    # If added, is_ancestor[0] becomes True.
    # So ancestors of 0 would be {1, 0}.

    mask = jnp.array([
        [0, 1],
        [1, 0]
    ], dtype=jnp.int32)

    ancestors = find_ancestors_jax(mask, 0)
    # Ideally, if cyclic dependency exists, it should be found.
    # If 0 depends on 1, and 1 depends on 0.
    # Ancestors of 0 include 1. Ancestors of 1 include 0.
    # So ancestors of 0 should include 0.
    expected = jnp.array([True, True])
    assert jnp.all(ancestors == expected)

def test_find_ancestors_self_loop():
    # 0 <- 0
    mask = jnp.array([
        [1]
    ], dtype=jnp.int32)

    # Ancestors of 0: empty set? or {0}?
    # Current code: `j != current_node` prevents adding self.
    # So result should be [False].
    ancestors = find_ancestors_jax(mask, 0)
    expected = jnp.array([False])
    assert jnp.all(ancestors == expected)

def test_find_ancestors_multiple_paths():
    # 0 <- 1
    # 0 <- 2
    # 1 <- 3
    # 2 <- 3
    # Diamond shape.
    # Ancestors of 0: {1, 2, 3}

    mask = jnp.array([
        [0, 1, 1, 0], # 0
        [0, 0, 0, 1], # 1
        [0, 0, 0, 1], # 2
        [0, 0, 0, 0]  # 3
    ], dtype=jnp.int32)

    ancestors = find_ancestors_jax(mask, 0)
    expected = jnp.array([False, True, True, True])
    assert jnp.all(ancestors == expected)
