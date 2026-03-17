import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp  # noqa: E402
from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_jax_simple():
    # Simple chain: 0 -> 1 -> 2
    # Rows represent children and columns represent parents.
    mask = jnp.array([
        [0, 0, 0],  # node 0 has no parents
        [1, 0, 0],  # node 1 has parent 0
        [0, 1, 0],  # node 2 has parent 1
    ])

    ancestors_0 = find_ancestors_jax(mask, 0)
    assert (ancestors_0 == jnp.array([False, False, False])).all()

    ancestors_1 = find_ancestors_jax(mask, 1)
    assert (ancestors_1 == jnp.array([True, False, False])).all()

    ancestors_2 = find_ancestors_jax(mask, 2)
    assert (ancestors_2 == jnp.array([True, True, False])).all()


def test_find_ancestors_jax_multiple_parents():
    # 0 -> 2
    # 1 -> 2
    # 2 -> 3
    mask = jnp.array([
        [0, 0, 0, 0],  # 0
        [0, 0, 0, 0],  # 1
        [1, 1, 0, 0],  # 2
        [0, 0, 1, 0],  # 3
    ])

    ancestors_3 = find_ancestors_jax(mask, 3)
    assert (ancestors_3 == jnp.array([True, True, True, False])).all()


def test_find_ancestors_jax_cycle():
    # 0 -> 1 -> 2 -> 0
    mask = jnp.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
    ])

    ancestors_0 = find_ancestors_jax(mask, 0)
    # Marks a node as its own ancestor if a larger cycle exists
    assert (ancestors_0 == jnp.array([True, True, True])).all()
