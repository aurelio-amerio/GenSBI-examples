import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_jax_simple():
    # 0 -> 1 -> 2
    # Rows represent children, columns represent parents
    mask = jnp.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0]
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


def test_find_ancestors_jax_branching():
    # 0 -> 1 -> 3
    # 0 -> 2 -> 3
    mask = jnp.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 1, 0]
    ], dtype=jnp.bool_)

    ancestors_3 = find_ancestors_jax(mask, 3)
    assert ancestors_3[0]
    assert ancestors_3[1]
    assert ancestors_3[2]
    assert not ancestors_3[3]


def test_find_ancestors_jax_disconnected():
    mask = jnp.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ], dtype=jnp.bool_)

    for i in range(3):
        ancestors = find_ancestors_jax(mask, i)
        assert not ancestors[0]
        assert not ancestors[1]
        assert not ancestors[2]


def test_find_ancestors_jax_cycle():
    # 0 -> 1 -> 2 -> 0
    mask = jnp.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=jnp.bool_)

    ancestors_0 = find_ancestors_jax(mask, 0)
    assert ancestors_0[0]
    assert ancestors_0[1]
    assert ancestors_0[2]
