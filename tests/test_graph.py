import os

os.environ["JAX_PLATFORMS"] = "cpu"
import jax.numpy as jnp  # noqa: E402

from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_jax_simple_dag():
    """Test find_ancestors_jax with a simple directed acyclic graph."""
    # Graph with 4 nodes: 0, 1, 2, 3
    # Edges: 0 -> 2, 1 -> 2, 2 -> 3
    mask = jnp.zeros((4, 4), dtype=jnp.bool_)
    mask = mask.at[2, 0].set(True)
    mask = mask.at[2, 1].set(True)
    mask = mask.at[3, 2].set(True)

    # Ancestors of 3: 2, 1, 0
    expected = jnp.array([True, True, True, False], dtype=jnp.bool_)
    result = find_ancestors_jax(mask, 3)

    assert (result == expected).all()


def test_find_ancestors_jax_cyclic_graph():
    """Test find_ancestors_jax with a cyclic graph."""
    # Graph with 3 nodes: 0, 1, 2
    # Edges: 0 -> 1, 1 -> 2, 2 -> 0
    mask = jnp.zeros((3, 3), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)
    mask = mask.at[2, 1].set(True)
    mask = mask.at[0, 2].set(True)

    # Ancestors of 0: 2, 1, 0
    expected = jnp.array([True, True, True], dtype=jnp.bool_)
    result = find_ancestors_jax(mask, 0)

    assert (result == expected).all()


def test_find_ancestors_jax_no_ancestors():
    """Test find_ancestors_jax for a node with no ancestors."""
    # Graph with 3 nodes: 0, 1, 2
    # Edges: 0 -> 1, 1 -> 2
    mask = jnp.zeros((3, 3), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)
    mask = mask.at[2, 1].set(True)

    # Ancestors of 0: none
    expected = jnp.array([False, False, False], dtype=jnp.bool_)
    result = find_ancestors_jax(mask, 0)

    assert (result == expected).all()
