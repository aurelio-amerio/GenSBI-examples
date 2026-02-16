
import pytest
import jax
import jax.numpy as jnp
from gensbi_examples.graph import find_ancestors_jax

class TestFindAncestorsJax:
    def test_line_graph(self):
        # 0 -> 1 -> 2
        # Mask is Parents x Children
        mask = jnp.zeros((3, 3), dtype=jnp.bool_)
        mask = mask.at[0, 1].set(True)
        mask = mask.at[1, 2].set(True)

        # Ancestors of 0: {}
        anc_0 = find_ancestors_jax(mask, 0)
        assert not jnp.any(anc_0), f"Expected no ancestors for 0, got {anc_0}"

        # Ancestors of 1: {0}
        anc_1 = find_ancestors_jax(mask, 1)
        expected_1 = jnp.array([True, False, False])
        assert jnp.array_equal(anc_1, expected_1), f"Expected ancestors for 1: {expected_1}, got {anc_1}"

        # Ancestors of 2: {0, 1}
        anc_2 = find_ancestors_jax(mask, 2)
        expected_2 = jnp.array([True, True, False])
        assert jnp.array_equal(anc_2, expected_2), f"Expected ancestors for 2: {expected_2}, got {anc_2}"

    def test_multiple_parents(self):
        # 1 -> 0, 2 -> 0
        mask = jnp.zeros((3, 3), dtype=jnp.bool_)
        mask = mask.at[1, 0].set(True)
        mask = mask.at[2, 0].set(True)

        # Ancestors of 0: {1, 2}
        anc_0 = find_ancestors_jax(mask, 0)
        expected_0 = jnp.array([False, True, True])
        assert jnp.array_equal(anc_0, expected_0), f"Expected ancestors for 0: {expected_0}, got {anc_0}"

    def test_multiple_children(self):
        # 0 -> 1, 0 -> 2
        mask = jnp.zeros((3, 3), dtype=jnp.bool_)
        mask = mask.at[0, 1].set(True)
        mask = mask.at[0, 2].set(True)

        # Ancestors of 1: {0}
        anc_1 = find_ancestors_jax(mask, 1)
        expected_1 = jnp.array([True, False, False])
        assert jnp.array_equal(anc_1, expected_1), f"Expected ancestors for 1: {expected_1}, got {anc_1}"

        # Ancestors of 2: {0}
        anc_2 = find_ancestors_jax(mask, 2)
        expected_2 = jnp.array([True, False, False])
        assert jnp.array_equal(anc_2, expected_2), f"Expected ancestors for 2: {expected_2}, got {anc_2}"

    def test_cycle(self):
        # 0 -> 1 -> 0
        mask = jnp.zeros((2, 2), dtype=jnp.bool_)
        mask = mask.at[0, 1].set(True)
        mask = mask.at[1, 0].set(True)

        # Ancestors of 0: {1} (and implicitly 0 via cycle, but standard definition excludes self unless specified)
        # But if we want *all nodes that can reach 0*, then 0 reaches 0.
        # Let's see what implementation does. Usually "strict ancestors" excludes self.
        # If I fix implementation to matrix power, it might include self.
        # Let's just check 1 is present.
        anc_0 = find_ancestors_jax(mask, 0)
        assert anc_0[1], "Node 1 should be ancestor of 0 in cycle"

        # Ancestors of 1: {0}
        anc_1 = find_ancestors_jax(mask, 1)
        assert anc_1[0], "Node 0 should be ancestor of 1 in cycle"
