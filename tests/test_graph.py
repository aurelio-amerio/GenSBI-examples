import os

# select device
os.environ["JAX_PLATFORMS"] = "cpu"

import jax  # noqa: E402, F401
import jax.numpy as jnp  # noqa: E402

from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_jax():
    # Adjacency matrix: rows are children, columns are parents.
    # M[i, j] = 1 means an edge from j to i (j is a parent of i).

    # Graph 1:
    # 0 -> 1 -> 2 -> 3
    # 0 -> 3
    mask1 = jnp.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 0, 1, 0],
    ], dtype=jnp.bool_)

    # Node 0 has no ancestors.
    ancestors_0 = find_ancestors_jax(mask1, 0)
    assert not ancestors_0[0]
    assert not ancestors_0[1]
    assert not ancestors_0[2]
    assert not ancestors_0[3]

    # Node 1 has ancestor 0.
    ancestors_1 = find_ancestors_jax(mask1, 1)
    assert ancestors_1[0]
    assert not ancestors_1[1]
    assert not ancestors_1[2]
    assert not ancestors_1[3]

    # Node 2 has ancestors 0 and 1.
    ancestors_2 = find_ancestors_jax(mask1, 2)
    assert ancestors_2[0]
    assert ancestors_2[1]
    assert not ancestors_2[2]
    assert not ancestors_2[3]

    # Node 3 has ancestors 0, 1, and 2.
    ancestors_3 = find_ancestors_jax(mask1, 3)
    assert ancestors_3[0]
    assert ancestors_3[1]
    assert ancestors_3[2]
    assert not ancestors_3[3]

    # Graph 2: Circular dependency
    # 0 -> 1 -> 2 -> 0
    mask2 = jnp.array([
        [0, 0, 1],  # 2 is parent of 0
        [1, 0, 0],  # 0 is parent of 1
        [0, 1, 0],  # 1 is parent of 2
    ], dtype=jnp.bool_)

    ancestors_0 = find_ancestors_jax(mask2, 0)
    # Due to cycle, 0 is ancestor of 0? Yes, wait let's run it.
    # Actually based on "ignoring immediate self-loops, but will mark node as its own ancestor if larger cycle exists"
    assert ancestors_0[0]
    assert ancestors_0[1]
    assert ancestors_0[2]

    # Graph 3: Immediate self-loop
    # 0 -> 0, 0 -> 1
    mask3 = jnp.array([
        [1, 0],
        [1, 0],
    ], dtype=jnp.bool_)

    ancestors_0 = find_ancestors_jax(mask3, 0)
    # Immediate self loops are ignored.
    assert not ancestors_0[0]
    assert not ancestors_0[1]

    ancestors_1 = find_ancestors_jax(mask3, 1)
    assert ancestors_1[0]
    assert not ancestors_1[1]
