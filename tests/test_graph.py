import os

# select device
os.environ["JAX_PLATFORMS"] = "cpu"

# import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import numpy as np  # noqa: E402
# import pytest  # noqa: E402

from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_simple_dag():
    """Test find_ancestors_jax on a simple DAG: 0 -> 1 -> 2."""
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 0] = True  # 0 -> 1
    mask[2, 1] = True  # 1 -> 2
    mask_jnp = jnp.array(mask)

    ancestors = find_ancestors_jax(mask_jnp, 2)
    assert ancestors.dtype == jnp.bool_
    assert ancestors[0]
    assert ancestors[1]
    assert not ancestors[2]

    ancestors = find_ancestors_jax(mask_jnp, 1)
    assert ancestors[0]
    assert not ancestors[1]
    assert not ancestors[2]


def test_find_ancestors_branching_dag():
    """Test find_ancestors_jax on a branching DAG with multiple parents.

    Graph:
    0 -> 1
    1 -> 4
    2 -> 4
    3 -> 4
    """
    mask = np.zeros((5, 5), dtype=bool)
    mask[4, 1] = True  # 1 -> 4
    mask[4, 2] = True  # 2 -> 4
    mask[4, 3] = True  # 3 -> 4
    mask[1, 0] = True  # 0 -> 1

    mask_jnp = jnp.array(mask)
    ancestors = find_ancestors_jax(mask_jnp, 4)

    # Expected ancestors of 4: 0, 1, 2, 3
    assert ancestors.dtype == jnp.bool_
    assert ancestors[0]
    assert ancestors[1]
    assert ancestors[2]
    assert ancestors[3]
    assert not ancestors[4]


def test_find_ancestors_cycle():
    """Test find_ancestors_jax on a graph with a cycle: 0 -> 1 -> 2 -> 0."""
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 0] = True  # 0 -> 1
    mask[2, 1] = True  # 1 -> 2
    mask[0, 2] = True  # 2 -> 0

    mask_jnp = jnp.array(mask)
    ancestors = find_ancestors_jax(mask_jnp, 0)

    assert ancestors.dtype == jnp.bool_
    assert ancestors[0]  # 0 is an ancestor of itself because of the cycle
    assert ancestors[1]
    assert ancestors[2]


def test_find_ancestors_self_loop():
    """Test find_ancestors_jax ignores immediate self-loops."""
    mask = np.zeros((1, 1), dtype=bool)
    mask[0, 0] = True

    mask_jnp = jnp.array(mask)
    ancestors = find_ancestors_jax(mask_jnp, 0)

    assert ancestors.dtype == jnp.bool_
    assert not ancestors[0]  # no self self-loops
