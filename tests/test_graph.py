import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax  # noqa: E402, F401
import jax.numpy as jnp  # noqa: E402
from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_jax_simple():
    # 0 -> 1 -> 2
    # Rows are children, cols are parents.
    mask = jnp.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=jnp.bool_)

    ans2 = find_ancestors_jax(mask, 2)
    expected = jnp.array([True, True, False])
    assert (ans2 == expected).all()


def test_find_ancestors_jax_branching_deep():
    # Graph:
    # 0 -> 1 -> 4
    # 2 -> 3 -> 4
    # Parents of 4 are 1 and 3.
    # Parents of 1 is 0.
    # Parents of 3 is 2.
    # Rows are children, cols are parents.
    mask = jnp.zeros((5, 5), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)
    mask = mask.at[3, 2].set(True)
    mask = mask.at[4, 1].set(True)
    mask = mask.at[4, 3].set(True)

    ans4 = find_ancestors_jax(mask, 4)
    # Ancestors of 4 should be 0, 1, 2, 3
    expected = jnp.array([True, True, True, True, False])
    assert (ans4 == expected).all()


def test_find_ancestors_jax_cycle():
    # 0 -> 1 -> 2 -> 0
    mask = jnp.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=jnp.bool_)

    ans0 = find_ancestors_jax(mask, 0)
    expected = jnp.array([True, True, True])
    assert (ans0 == expected).all()


def test_find_ancestors_jax_disconnected():
    # 0 -> 1, 2 -> 3
    mask = jnp.zeros((4, 4), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)
    mask = mask.at[3, 2].set(True)

    ans1 = find_ancestors_jax(mask, 1)
    expected1 = jnp.array([True, False, False, False])
    assert (ans1 == expected1).all()

    ans3 = find_ancestors_jax(mask, 3)
    expected3 = jnp.array([False, False, True, False])
    assert (ans3 == expected3).all()


def test_find_ancestors_jax_self_loop():
    # 0 -> 0 (self-loop), 0 -> 1
    mask = jnp.array([
        [1, 0],
        [1, 0]
    ], dtype=jnp.bool_)

    ans1 = find_ancestors_jax(mask, 1)
    expected = jnp.array([True, False])
    assert (ans1 == expected).all()

    ans0 = find_ancestors_jax(mask, 0)
    expected_ans0 = jnp.array([False, False])
    # The current implementation prevents the node itself from being marked as an ancestor if it is an immediate self loop:
    # `j != current_node` avoids adding immediate self loop
    assert (ans0 == expected_ans0).all()


def test_find_ancestors_jax_multiple_parents():
    # 0 -> 3, 1 -> 3, 2 -> 3
    mask = jnp.zeros((4, 4), dtype=jnp.bool_)
    mask = mask.at[3, 0].set(True)
    mask = mask.at[3, 1].set(True)
    mask = mask.at[3, 2].set(True)

    ans3 = find_ancestors_jax(mask, 3)
    expected = jnp.array([True, True, True, False])
    assert (ans3 == expected).all()
