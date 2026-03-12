import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax  # noqa: E402, F401
import jax.numpy as jnp  # noqa: E402
from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_jax_dag():
    # 0 -> 1 -> 2
    # Adjacency matrix: rows are children, columns are parents.
    # M[i, j] = 1 means j -> i
    M = jnp.array([
        [0, 0, 0],  # 0 has no parents
        [1, 0, 0],  # 1 has parent 0
        [0, 1, 0],  # 2 has parent 1
    ])

    ancestors_2 = find_ancestors_jax(M, 2)
    expected_2 = jnp.array([True, True, False])
    assert (ancestors_2 == expected_2).all()

    ancestors_1 = find_ancestors_jax(M, 1)
    expected_1 = jnp.array([True, False, False])
    assert (ancestors_1 == expected_1).all()

    ancestors_0 = find_ancestors_jax(M, 0)
    expected_0 = jnp.array([False, False, False])
    assert (ancestors_0 == expected_0).all()


def test_find_ancestors_jax_cycle():
    # Cycle: 0 -> 1 -> 0
    M = jnp.array([
        [0, 1],  # 0 has parent 1
        [1, 0],  # 1 has parent 0
    ])

    ancestors_0 = find_ancestors_jax(M, 0)
    expected_0 = jnp.array([True, True])
    assert (ancestors_0 == expected_0).all()


def test_find_ancestors_jax_self_loop():
    # 0 -> 0 self loop, but find_ancestors_jax ignores immediate self-loops
    # in its traversal as per graph logic
    M = jnp.array([
        [1, 0],
        [0, 0],
    ])

    ancestors_0 = find_ancestors_jax(M, 0)
    expected_0 = jnp.array([False, False])
    assert (ancestors_0 == expected_0).all()


def test_find_ancestors_jax_disconnected():
    # Disconnected nodes 0, 1
    M = jnp.array([
        [0, 0],
        [0, 0],
    ])

    ancestors_0 = find_ancestors_jax(M, 0)
    expected_0 = jnp.array([False, False])
    assert (ancestors_0 == expected_0).all()


def test_find_ancestors_jax_complex_dag():
    # 0 -> 1
    # 0 -> 2
    # 1 -> 3
    # 2 -> 3
    # 3 -> 4
    M = jnp.array([
        [0, 0, 0, 0, 0],  # 0
        [1, 0, 0, 0, 0],  # 1
        [1, 0, 0, 0, 0],  # 2
        [0, 1, 1, 0, 0],  # 3
        [0, 0, 0, 1, 0],  # 4
    ])

    ancestors_4 = find_ancestors_jax(M, 4)
    expected_4 = jnp.array([True, True, True, True, False])
    assert (ancestors_4 == expected_4).all()

    ancestors_3 = find_ancestors_jax(M, 3)
    expected_3 = jnp.array([True, True, True, False, False])
    assert (ancestors_3 == expected_3).all()
