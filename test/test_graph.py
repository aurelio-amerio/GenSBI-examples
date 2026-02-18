
import os
import jax.numpy as jnp
from gensbi_examples.graph import find_ancestors_jax

# Set JAX to CPU
os.environ["JAX_PLATFORMS"] = "cpu"


def test_find_ancestors_chain():
    # 0 -> 1 -> 2
    # Ancestors of 2: {0, 1}
    # Matrix: Rows=Child, Cols=Parent. M[1,0]=1, M[2,1]=1
    num_nodes = 3
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)
    mask = mask.at[2, 1].set(True)

    ancestors = find_ancestors_jax(mask, 2)
    expected = jnp.array([True, True, False], dtype=jnp.bool_)
    assert jnp.array_equal(ancestors, expected), (
        f"Expected {expected}, got {ancestors}"
    )

    ancestors = find_ancestors_jax(mask, 1)
    expected = jnp.array([True, False, False], dtype=jnp.bool_)
    assert jnp.array_equal(ancestors, expected), (
        f"Expected {expected}, got {ancestors}"
    )

    ancestors = find_ancestors_jax(mask, 0)
    expected = jnp.array([False, False, False], dtype=jnp.bool_)
    assert jnp.array_equal(ancestors, expected), (
        f"Expected {expected}, got {ancestors}"
    )


def test_find_ancestors_branching():
    # 0 -> 2
    # 1 -> 2
    # 2 -> 3
    # Ancestors of 3: {0, 1, 2}
    num_nodes = 4
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.bool_)
    mask = mask.at[2, 0].set(True)
    mask = mask.at[2, 1].set(True)
    mask = mask.at[3, 2].set(True)

    ancestors = find_ancestors_jax(mask, 3)
    expected = jnp.array([True, True, True, False], dtype=jnp.bool_)
    assert jnp.array_equal(ancestors, expected), (
        f"Expected {expected}, got {ancestors}"
    )


def test_find_ancestors_v_structure_depth():
    # 3 -> 0 -> 2
    # 1 -> 2
    # Ancestors of 2: {0, 1, 3}
    num_nodes = 4
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.bool_)
    mask = mask.at[0, 3].set(True)
    mask = mask.at[2, 0].set(True)
    mask = mask.at[2, 1].set(True)

    ancestors = find_ancestors_jax(mask, 2)
    expected = jnp.array([True, True, False, True], dtype=jnp.bool_)
    # 0 (True), 1 (True), 2 (False), 3 (True)
    assert jnp.array_equal(ancestors, expected), (
        f"Expected {expected}, got {ancestors}"
    )


def test_find_ancestors_cycle():
    # 0 -> 1 -> 0
    # Ancestors of 0: {1, 0} (cycle makes it its own ancestor via 1)
    num_nodes = 2
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)
    mask = mask.at[0, 1].set(True)

    ancestors = find_ancestors_jax(mask, 0)
    # 0 reaches 1. 1 reaches 0. So 1 is ancestor. And 0 is ancestor of 1.
    expected = jnp.array([True, True], dtype=jnp.bool_)
    assert jnp.array_equal(ancestors, expected), (
        f"Expected {expected}, got {ancestors}"
    )


def test_find_ancestors_self_loop():
    # 0 -> 0
    # Ancestors of 0: {} (immediate self loop ignored)
    num_nodes = 1
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.bool_)
    mask = mask.at[0, 0].set(True)

    ancestors = find_ancestors_jax(mask, 0)
    expected = jnp.array([False], dtype=jnp.bool_)
    assert jnp.array_equal(ancestors, expected), (
        f"Expected {expected}, got {ancestors}"
    )


def test_find_ancestors_disconnected():
    # 0   1
    num_nodes = 2
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.bool_)

    ancestors = find_ancestors_jax(mask, 0)
    expected = jnp.array([False, False], dtype=jnp.bool_)
    assert jnp.array_equal(ancestors, expected), (
        f"Expected {expected}, got {ancestors}"
    )
