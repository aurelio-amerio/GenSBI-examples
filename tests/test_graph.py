import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax  # noqa: E402, F401
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_jax_chain():
    """Test find_ancestors_jax on a simple chain A -> B -> C."""
    # Rows represent children, columns represent parents
    # A=0, B=1, C=2
    # Edge A->B: parent 0, child 1 -> mask[1, 0] = True
    # Edge B->C: parent 1, child 2 -> mask[2, 1] = True
    mask = jnp.zeros((3, 3), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)
    mask = mask.at[2, 1].set(True)

    ancestors_c = find_ancestors_jax(mask, 2)
    expected_c = np.array([True, True, False])
    np.testing.assert_array_equal(np.array(ancestors_c), expected_c)

    ancestors_b = find_ancestors_jax(mask, 1)
    expected_b = np.array([True, False, False])
    np.testing.assert_array_equal(np.array(ancestors_b), expected_b)

    ancestors_a = find_ancestors_jax(mask, 0)
    expected_a = np.array([False, False, False])
    np.testing.assert_array_equal(np.array(ancestors_a), expected_a)


def test_find_ancestors_jax_diamond():
    """Test find_ancestors_jax on a diamond graph A->B, A->C, B->D, C->D."""
    # A=0, B=1, C=2, D=3
    mask = jnp.zeros((4, 4), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)  # A->B
    mask = mask.at[2, 0].set(True)  # A->C
    mask = mask.at[3, 1].set(True)  # B->D
    mask = mask.at[3, 2].set(True)  # C->D

    ancestors_d = find_ancestors_jax(mask, 3)
    expected_d = np.array([True, True, True, False])
    np.testing.assert_array_equal(np.array(ancestors_d), expected_d)


def test_find_ancestors_jax_cycle():
    """Test find_ancestors_jax on a cycle A -> B -> C -> A."""
    # A=0, B=1, C=2
    mask = jnp.zeros((3, 3), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)  # A->B
    mask = mask.at[2, 1].set(True)  # B->C
    mask = mask.at[0, 2].set(True)  # C->A

    # Because of the cycle A->B->C->A, A is its own ancestor.
    ancestors_a = find_ancestors_jax(mask, 0)
    expected_a = np.array([True, True, True])
    np.testing.assert_array_equal(np.array(ancestors_a), expected_a)


def test_find_ancestors_jax_self_loop():
    """Test find_ancestors_jax ignores immediate self-loops A -> A."""
    # A=0, B=1
    mask = jnp.zeros((2, 2), dtype=jnp.bool_)
    mask = mask.at[0, 0].set(True)  # A->A
    mask = mask.at[1, 0].set(True)  # A->B

    ancestors_a = find_ancestors_jax(mask, 0)
    expected_a = np.array([False, False])
    np.testing.assert_array_equal(np.array(ancestors_a), expected_a)
