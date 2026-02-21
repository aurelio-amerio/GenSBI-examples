
import os
import jax
import jax.numpy as jnp
import pytest

# Ensure JAX runs on CPU
os.environ["JAX_PLATFORMS"] = "cpu"  # noqa: E402

from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_chain():
    # 0 <- 1 <- 2
    # mask[i, j] = 1 means j -> i
    mask = jnp.zeros((3, 3), dtype=jnp.int32)
    mask = mask.at[0, 1].set(1)
    mask = mask.at[1, 2].set(1)

    ancestors = find_ancestors_jax(mask, 0)
    expected = jnp.array([False, True, True])
    assert jnp.all(ancestors == expected)


def test_multiple_parents():
    # 3 -> 1 -> 0
    #      2 -> 0
    mask = jnp.zeros((4, 4), dtype=jnp.int32)
    mask = mask.at[1, 3].set(1)
    mask = mask.at[0, 1].set(1)
    mask = mask.at[0, 2].set(1)

    ancestors = find_ancestors_jax(mask, 0)
    expected = jnp.array([False, True, True, True])
    assert jnp.all(ancestors == expected)


def test_cycle():
    # 0 -> 1 -> 0
    mask = jnp.zeros((2, 2), dtype=jnp.int32)
    mask = mask.at[1, 0].set(1)
    mask = mask.at[0, 1].set(1)

    ancestors0 = find_ancestors_jax(mask, 0)
    # Ancestors of 0: 1 is parent. 1's parent is 0. So {0, 1}.
    expected = jnp.array([True, True])
    assert jnp.all(ancestors0 == expected)


def test_self_loop():
    # 0 -> 0
    mask = jnp.zeros((1, 1), dtype=jnp.int32)
    mask = mask.at[0, 0].set(1)

    ancestors = find_ancestors_jax(mask, 0)
    # Direct self-loop should be ignored unless it's part of a larger cycle
    # Original logic: j != current_node.
    expected = jnp.array([False])
    assert jnp.all(ancestors == expected)
