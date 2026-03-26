import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp  # noqa: E402
from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_jax():
    # Simple chain: 0 -> 1 -> 2
    mask = jnp.array([
        [0, 0, 0],  # 0 has no parents
        [1, 0, 0],  # 1 has parent 0
        [0, 1, 0]   # 2 has parent 1
    ], dtype=jnp.bool_)

    ancestors_0 = find_ancestors_jax(mask, 0)
    assert not ancestors_0[0]
    assert not ancestors_0[1]
    assert not ancestors_0[2]

    ancestors_1 = find_ancestors_jax(mask, 1)
    assert ancestors_1[0]
    assert not ancestors_1[1]
    assert not ancestors_1[2]

    ancestors_2 = find_ancestors_jax(mask, 2)
    assert ancestors_2[0]
    assert ancestors_2[1]
    assert not ancestors_2[2]

    # Simple tree: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
    mask2 = jnp.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 1, 0]
    ], dtype=jnp.bool_)

    ancestors_3 = find_ancestors_jax(mask2, 3)
    assert ancestors_3[0]
    assert ancestors_3[1]
    assert ancestors_3[2]
    assert not ancestors_3[3]

    # Cycle: 0 -> 1 -> 2 -> 0
    mask3 = jnp.array([
        [0, 0, 1],  # 0 has parent 2
        [1, 0, 0],  # 1 has parent 0
        [0, 1, 0]   # 2 has parent 1
    ], dtype=jnp.bool_)

    ancestors_c0 = find_ancestors_jax(mask3, 0)
    # Ancestors of 0: 2, and parent of 2 is 1, parent of 1 is 0.
    # So ancestors of 0 are 1, 2, and 0 (since it loops back).
    assert ancestors_c0[0]
    assert ancestors_c0[1]
    assert ancestors_c0[2]
