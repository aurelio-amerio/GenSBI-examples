import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402

from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_jax_simple_chain():
    """Test finding ancestors in a simple chain 0 -> 1 -> 2."""
    mask = jnp.zeros((3, 3), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)
    mask = mask.at[2, 1].set(True)

    expected = jnp.array([True, True, False], dtype=jnp.bool_)
    result = find_ancestors_jax(mask, 2)
    assert jnp.array_equal(result, expected)


def test_find_ancestors_jax_multiple_parents():
    """Test finding ancestors with multiple branching parents."""
    # 0 -> 2, 1 -> 2, 2 -> 3
    mask = jnp.zeros((4, 4), dtype=jnp.bool_)
    mask = mask.at[2, 0].set(True)
    mask = mask.at[2, 1].set(True)
    mask = mask.at[3, 2].set(True)

    expected = jnp.array([True, True, True, False], dtype=jnp.bool_)
    result = find_ancestors_jax(mask, 3)
    assert jnp.array_equal(result, expected)


def test_find_ancestors_jax_cycle():
    """Test finding ancestors in a graph with a cycle 0 -> 1 -> 2 -> 0."""
    mask = jnp.zeros((3, 3), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)
    mask = mask.at[2, 1].set(True)
    mask = mask.at[0, 2].set(True)

    expected = jnp.array([True, True, True], dtype=jnp.bool_)
    result = find_ancestors_jax(mask, 0)
    assert jnp.array_equal(result, expected)


def test_find_ancestors_jax_self_loop():
    """Test that immediate self-loops are ignored."""
    # 0 -> 0
    mask = jnp.zeros((1, 1), dtype=jnp.bool_)
    mask = mask.at[0, 0].set(True)

    expected = jnp.array([False], dtype=jnp.bool_)
    result = find_ancestors_jax(mask, 0)
    assert jnp.array_equal(result, expected)


def test_find_ancestors_jax_int_mask():
    """Test finding ancestors using an integer mask."""
    # 0 -> 1
    mask = jnp.zeros((2, 2), dtype=jnp.int32)
    mask = mask.at[1, 0].set(1)

    expected = jnp.array([True, False], dtype=jnp.bool_)
    result = find_ancestors_jax(mask, 1)
    assert jnp.array_equal(result, expected)
