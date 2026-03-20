import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp  # noqa: E402

from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_jax_simple():
    # 3 nodes: 0 -> 1 -> 2
    mask = jnp.array([
        [False, False, False],
        [True,  False, False],
        [False, True,  False],
    ], dtype=jnp.bool_)

    ancestors_2 = find_ancestors_jax(mask, 2)
    expected_2 = jnp.array([True, True, False], dtype=jnp.bool_)
    assert (ancestors_2 == expected_2).all()

    ancestors_1 = find_ancestors_jax(mask, 1)
    expected_1 = jnp.array([True, False, False], dtype=jnp.bool_)
    assert (ancestors_1 == expected_1).all()


def test_find_ancestors_jax_branching():
    # 0 -> 2
    # 1 -> 2
    # 2 -> 3
    mask = jnp.array([
        [False, False, False, False],
        [False, False, False, False],
        [True,  True,  False, False],
        [False, False, True,  False],
    ], dtype=jnp.bool_)

    ancestors = find_ancestors_jax(mask, 3)
    expected = jnp.array([True, True, True, False], dtype=jnp.bool_)
    assert (ancestors == expected).all()


def test_find_ancestors_jax_cycles():
    # 0 -> 1 -> 2 -> 0
    # 3 -> 3
    mask = jnp.array([
        [False, False, True,  False],
        [True,  False, False, False],
        [False, True,  False, False],
        [False, False, False, True],
    ], dtype=jnp.bool_)

    ancestors_0 = find_ancestors_jax(mask, 0)
    expected_0 = jnp.array([True, True, True, False], dtype=jnp.bool_)
    assert (ancestors_0 == expected_0).all()

    ancestors_3 = find_ancestors_jax(mask, 3)
    expected_3 = jnp.array([False, False, False, False], dtype=jnp.bool_)
    assert (ancestors_3 == expected_3).all()
