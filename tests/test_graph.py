import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp  # noqa: E402

from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_simple_chain():
    """Test finding ancestors in a simple chain A -> B -> C."""
    mask = jnp.zeros((3, 3), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)  # A -> B
    mask = mask.at[2, 1].set(True)  # B -> C

    ancestors = find_ancestors_jax(mask, 2)
    expected = jnp.array([True, True, False])
    assert (ancestors == expected).all()


def test_find_ancestors_multiple_parents():
    """Test finding ancestors when a node has multiple parents."""
    mask = jnp.zeros((5, 5), dtype=jnp.bool_)
    mask = mask.at[2, 0].set(True)
    mask = mask.at[3, 1].set(True)
    mask = mask.at[4, 2].set(True)
    mask = mask.at[4, 3].set(True)

    ancestors = find_ancestors_jax(mask, 4)
    expected = jnp.array([True, True, True, True, False])
    assert (ancestors == expected).all()


def test_find_ancestors_self_loop():
    """Test that self-loops are ignored during traversal."""
    mask = jnp.zeros((2, 2), dtype=jnp.bool_)
    mask = mask.at[0, 0].set(True)
    mask = mask.at[1, 0].set(True)

    ancestors = find_ancestors_jax(mask, 1)
    expected = jnp.array([True, False])
    assert (ancestors == expected).all()


def test_find_ancestors_cycle():
    """Test that cycles do not cause infinite loops and mark the node as its own ancestor."""
    mask = jnp.zeros((3, 3), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)
    mask = mask.at[2, 1].set(True)
    mask = mask.at[0, 2].set(True)

    ancestors = find_ancestors_jax(mask, 1)
    expected = jnp.array([True, True, True])
    assert (ancestors == expected).all()


def test_find_ancestors_no_ancestors():
    """Test a node with no ancestors."""
    mask = jnp.zeros((3, 3), dtype=jnp.bool_)
    ancestors = find_ancestors_jax(mask, 0)
    expected = jnp.array([False, False, False])
    assert (ancestors == expected).all()


def test_find_ancestors_int32_mask():
    """Test that int32 mask is handled correctly."""
    mask = jnp.zeros((2, 2), dtype=jnp.int32)
    mask = mask.at[1, 0].set(1)
    ancestors = find_ancestors_jax(mask, 1)
    expected = jnp.array([True, False])
    assert (ancestors == expected).all()
    assert ancestors.dtype == jnp.bool_
