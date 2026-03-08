import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import pytest  # noqa: E402

from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_jax_basic():
    """Test find_ancestors_jax with a basic linear chain DAG."""
    # 0 -> 1 -> 2
    # Rows are children, columns are parents
    mask = jnp.zeros((3, 3), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)
    mask = mask.at[2, 1].set(True)

    # Ancestors of 2 are 0, 1
    ans_2 = find_ancestors_jax(mask, 2)
    expected_2 = jnp.array([True, True, False], dtype=jnp.bool_)
    assert jnp.all(ans_2 == expected_2)

    # Ancestors of 1 is 0
    ans_1 = find_ancestors_jax(mask, 1)
    expected_1 = jnp.array([True, False, False], dtype=jnp.bool_)
    assert jnp.all(ans_1 == expected_1)

    # Ancestors of 0 is none
    ans_0 = find_ancestors_jax(mask, 0)
    expected_0 = jnp.array([False, False, False], dtype=jnp.bool_)
    assert jnp.all(ans_0 == expected_0)


def test_find_ancestors_jax_int32():
    """Test find_ancestors_jax with an int32 matrix."""
    # 0 -> 1 -> 2
    mask = jnp.zeros((3, 3), dtype=jnp.int32)
    mask = mask.at[1, 0].set(1)
    mask = mask.at[2, 1].set(1)

    ans_2 = find_ancestors_jax(mask, 2)
    expected_2 = jnp.array([True, True, False], dtype=jnp.bool_)
    assert jnp.all(ans_2 == expected_2)


def test_find_ancestors_jax_branching():
    """Test find_ancestors_jax with a branching DAG."""
    # 0 -> 2
    # 1 -> 2
    # 2 -> 3
    mask = jnp.zeros((4, 4), dtype=jnp.bool_)
    mask = mask.at[2, 0].set(True)
    mask = mask.at[2, 1].set(True)
    mask = mask.at[3, 2].set(True)

    # Ancestors of 3 are 0, 1, 2
    ans_3 = find_ancestors_jax(mask, 3)
    expected_3 = jnp.array([True, True, True, False], dtype=jnp.bool_)
    assert jnp.all(ans_3 == expected_3)


def test_find_ancestors_jax_cycles():
    """Test find_ancestors_jax with a cycle."""
    # 0 -> 1 -> 2 -> 0
    mask = jnp.zeros((3, 3), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)
    mask = mask.at[2, 1].set(True)
    mask = mask.at[0, 2].set(True)

    # Since it's a cycle, 0's ancestors include 1 and 2, and also itself
    ans_0 = find_ancestors_jax(mask, 0)
    expected_0 = jnp.array([True, True, True], dtype=jnp.bool_)
    assert jnp.all(ans_0 == expected_0)


def test_find_ancestors_jax_self_loop():
    """Test find_ancestors_jax ignores immediate self-loops."""
    # 0 -> 0
    mask = jnp.zeros((1, 1), dtype=jnp.bool_)
    mask = mask.at[0, 0].set(True)

    # Immediate self loops are ignored by logic `j != current_node`
    ans_0 = find_ancestors_jax(mask, 0)
    expected_0 = jnp.array([False], dtype=jnp.bool_)
    assert jnp.all(ans_0 == expected_0)

    # Let's test with a node having a self loop and another parent
    # 0 -> 0, 1 -> 0
    mask2 = jnp.zeros((2, 2), dtype=jnp.bool_)
    mask2 = mask2.at[0, 0].set(True)
    mask2 = mask2.at[0, 1].set(True)

    ans_0_2 = find_ancestors_jax(mask2, 0)
    expected_0_2 = jnp.array([False, True], dtype=jnp.bool_)
    assert jnp.all(ans_0_2 == expected_0_2)
