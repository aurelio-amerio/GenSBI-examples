
import os
os.environ["JAX_PLATFORMS"] = "cpu"  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_chain():
    # Case 1: A -> B -> C
    # Nodes: 0: A, 1: B, 2: C
    # 0 -> 1: mask[1, 0] = 1
    # 1 -> 2: mask[2, 1] = 1
    mask = jnp.zeros((3, 3), dtype=jnp.int32)
    mask = mask.at[1, 0].set(1)
    mask = mask.at[2, 1].set(1)

    # Ancestors of 2 (C): 1 (B) and 0 (A)
    ancestors = find_ancestors_jax(mask, 2)
    expected = jnp.array([True, True, False])  # 0, 1 are ancestors. 2 is self.
    assert jnp.all(ancestors == expected)


def test_find_ancestors_branching():
    # Case 2: A -> C, B -> C
    # 0 -> 2, 1 -> 2
    mask = jnp.zeros((3, 3), dtype=jnp.int32)
    mask = mask.at[2, 0].set(1)
    mask = mask.at[2, 1].set(1)

    ancestors = find_ancestors_jax(mask, 2)
    expected = jnp.array([True, True, False])
    assert jnp.all(ancestors == expected)


def test_find_ancestors_cycle():
    # Case 3: Cycle A -> B -> A
    # 0 -> 1, 1 -> 0
    mask = jnp.zeros((2, 2), dtype=jnp.int32)
    mask = mask.at[1, 0].set(1)
    mask = mask.at[0, 1].set(1)

    ancestors = find_ancestors_jax(mask, 0)
    # Ancestors of A is B. B depends on A, so A is ancestor of B.
    # So ancestors of A: {B, A}.
    expected = jnp.array([True, True])
    assert jnp.all(ancestors == expected)


def test_find_ancestors_diamond_bug():
    # Graph: A -> B -> D, C -> D
    # Nodes: 0:A, 1:B, 2:C, 3:D
    # Parents of D: B, C.
    # Parents of B: A.
    # Parents of C: None.
    # Ancestors of D: {A, B, C} -> {0, 1, 2}

    mask = jnp.zeros((4, 4), dtype=jnp.int32)
    # A -> B
    mask = mask.at[1, 0].set(1)
    # B -> D
    mask = mask.at[3, 1].set(1)
    # C -> D
    mask = mask.at[3, 2].set(1)

    ancestors = find_ancestors_jax(mask, 3)

    expected = jnp.array([True, True, True, False])

    assert jnp.all(ancestors == expected), f"Expected {expected}, got {ancestors}"
