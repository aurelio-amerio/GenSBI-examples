import os

# Set JAX to CPU to avoid issues in CI/CD environments without GPU
os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp  # noqa: E402
from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_jax():
    """Test find_ancestors_jax for various graph structures."""

    # Test case 1: Linear chain 0 -> 1 -> 2
    # Matrix (rows are children, columns are parents):
    # Node 0 has no parents.
    # Node 1 has parent 0.
    # Node 2 has parent 1.
    mask = jnp.zeros((3, 3), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)
    mask = mask.at[2, 1].set(True)

    ancestors_2 = find_ancestors_jax(mask, 2)
    assert ancestors_2[0]  # 0 is ancestor of 2
    assert ancestors_2[1]  # 1 is ancestor of 2
    # 2 is not ancestor of itself (no cycle involving 2)
    assert not ancestors_2[2]

    ancestors_1 = find_ancestors_jax(mask, 1)
    assert ancestors_1[0]
    assert not ancestors_1[1]
    assert not ancestors_1[2]

    ancestors_0 = find_ancestors_jax(mask, 0)
    assert not jnp.any(ancestors_0)

    # Test case 2: Branching (Merge) 0 -> 2, 1 -> 2
    # Node 2 has parents 0 and 1.
    mask = jnp.zeros((3, 3), dtype=jnp.bool_)
    mask = mask.at[2, 0].set(True)
    mask = mask.at[2, 1].set(True)

    ancestors_2 = find_ancestors_jax(mask, 2)
    assert ancestors_2[0]
    assert ancestors_2[1]
    assert not ancestors_2[2]

    # Test case 3: Cycle 0 -> 1 -> 0
    # Node 0 has parent 1.
    # Node 1 has parent 0.
    mask = jnp.zeros((2, 2), dtype=jnp.bool_)
    mask = mask.at[0, 1].set(True)
    mask = mask.at[1, 0].set(True)

    ancestors_0 = find_ancestors_jax(mask, 0)
    assert ancestors_0[0]  # Due to cycle, 0 is an ancestor of itself
    assert ancestors_0[1]  # 1 is an ancestor of 0

    ancestors_1 = find_ancestors_jax(mask, 1)
    assert ancestors_1[0]
    assert ancestors_1[1]

    # Test case 4: Disconnected 0, 1
    mask = jnp.zeros((2, 2), dtype=jnp.bool_)
    ancestors_0 = find_ancestors_jax(mask, 0)
    assert not jnp.any(ancestors_0)
    ancestors_1 = find_ancestors_jax(mask, 1)
    assert not jnp.any(ancestors_1)

    # Test case 5: Complex DAG
    # 0 -> 1
    # 1 -> 2
    # 0 -> 2 (direct edge)
    mask = jnp.zeros((3, 3), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)
    mask = mask.at[2, 1].set(True)
    mask = mask.at[2, 0].set(True)

    ancestors_2 = find_ancestors_jax(mask, 2)
    assert ancestors_2[0]
    assert ancestors_2[1]
    assert not ancestors_2[2]
