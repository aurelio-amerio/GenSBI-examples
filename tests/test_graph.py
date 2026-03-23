import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp  # noqa: E402

from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_jax_simple():
    # 0 -> 1 -> 2
    mask = jnp.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0]
    ], dtype=jnp.bool_)

    # Node 2 ancestors should be 0 and 1
    # Note: the test script gives [True True False], meaning 0 and 1
    # because row 2 has edges from 0 and 1
    # Let's verify: row 2 is [1, 1, 0], so 2's parents are 0 and 1.
    # 0's parents: [] (row 0 is [0, 0, 0])
    # 1's parents: 0 (row 1 is [1, 0, 0])

    result = find_ancestors_jax(mask, 2)
    expected = jnp.array([True, True, False], dtype=jnp.bool_)
    assert (result == expected).all()


def test_find_ancestors_jax_branching():
    # 0 -> 1 -> 3
    # 0 -> 2 -> 3
    mask_branching = jnp.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 1, 0]
    ], dtype=jnp.bool_)

    # Node 3 ancestors should be 0, 1, 2
    result = find_ancestors_jax(mask_branching, 3)
    expected = jnp.array([True, True, True, False], dtype=jnp.bool_)
    assert (result == expected).all()


def test_find_ancestors_jax_cycle():
    # 0 -> 1 -> 2 -> 0
    # Actually row is child, col is parent
    # 0 has parent 1: row 0 = [0, 1, 0]
    # 1 has parent 2: row 1 = [0, 0, 1]
    # 2 has parent 0: row 2 = [1, 0, 0]
    # So 1 -> 0, 2 -> 1, 0 -> 2
    # Cycle: 0 -> 2 -> 1 -> 0
    mask_cycle = jnp.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ], dtype=jnp.bool_)

    # Node 2 ancestors should be 0, 1, 2 (it is its own ancestor due to cycle)
    result = find_ancestors_jax(mask_cycle, 2)
    expected = jnp.array([True, True, True], dtype=jnp.bool_)
    assert (result == expected).all()
