import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402

from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


@pytest.mark.parametrize(
    "node,expected",
    [
        (0, [False, True, True]),   # 0 has ancestors 1 and 2
        (1, [False, False, True]),  # 1 has ancestor 2
        (2, [False, False, False])  # 2 has no ancestors
    ]
)
def test_find_ancestors_linear(node, expected):
    """Test find_ancestors_jax on a simple linear graph: 0 <- 1 <- 2"""
    # Rows are children, columns are parents
    mask = jnp.array([
        [0, 1, 0],  # 0 has parent 1
        [0, 0, 1],  # 1 has parent 2
        [0, 0, 0]   # 2 has no parents
    ], dtype=jnp.bool_)

    ancestors = find_ancestors_jax(mask, node)
    assert ancestors[0] == expected[0]
    assert ancestors[1] == expected[1]
    assert ancestors[2] == expected[2]


@pytest.mark.parametrize(
    "node,expected",
    [
        (0, [False, True, True, True]),     # 0 has parents 1, 3. 1 has parent 2.
        (1, [False, False, True, False]),   # 1 has parent 2
        (2, [False, False, False, False]),  # 2 has no parents
        (3, [False, False, False, False])   # 3 has no parents
    ]
)
def test_find_ancestors_branching(node, expected):
    """Test find_ancestors_jax on a branching graph: 0 <- 1 <- 2, 0 <- 3"""
    mask = jnp.array([
        [0, 1, 0, 1],  # 0 has parents 1, 3
        [0, 0, 1, 0],  # 1 has parent 2
        [0, 0, 0, 0],  # 2 has no parents
        [0, 0, 0, 0]   # 3 has no parents
    ], dtype=jnp.bool_)

    ancestors = find_ancestors_jax(mask, node)
    assert ancestors[0] == expected[0]
    assert ancestors[1] == expected[1]
    assert ancestors[2] == expected[2]
    assert ancestors[3] == expected[3]


@pytest.mark.parametrize(
    "node,expected",
    [
        (0, [True, True, True]),  # 0 <- 1 <- 2 <- 0
        (1, [True, True, True]),  # 1 <- 2 <- 0 <- 1
        (2, [True, True, True])   # 2 <- 0 <- 1 <- 2
    ]
)
def test_find_ancestors_cycle(node, expected):
    """Test find_ancestors_jax on a graph with a cycle: 0 <- 1 <- 2 <- 0"""
    mask = jnp.array([
        [0, 1, 0],  # 0 has parent 1
        [0, 0, 1],  # 1 has parent 2
        [1, 0, 0]   # 2 has parent 0
    ], dtype=jnp.bool_)

    ancestors = find_ancestors_jax(mask, node)
    assert ancestors[0] == expected[0]
    assert ancestors[1] == expected[1]
    assert ancestors[2] == expected[2]


@pytest.mark.parametrize(
    "node,expected",
    [
        (0, [False, True]),  # 0 has parent 0 (ignored) and 1
        (1, [False, False])  # 1 has no parents
    ]
)
def test_find_ancestors_self_loop(node, expected):
    """Test find_ancestors_jax ignores immediate self-loops."""
    mask = jnp.array([
        [1, 1],  # 0 has parent 0 (self-loop) and 1
        [0, 0]   # 1 has no parents
    ], dtype=jnp.bool_)

    ancestors = find_ancestors_jax(mask, node)
    assert ancestors[0] == expected[0]
    assert ancestors[1] == expected[1]


@pytest.mark.parametrize(
    "node,expected",
    [
        (0, [False, True, True]),
        (1, [False, False, True]),
        (2, [False, False, False])
    ]
)
def test_find_ancestors_int32_mask(node, expected):
    """Test find_ancestors_jax with int32 mask instead of boolean."""
    mask = jnp.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0]
    ], dtype=jnp.int32)

    ancestors = find_ancestors_jax(mask, node)
    assert ancestors[0] == expected[0]
    assert ancestors[1] == expected[1]
    assert ancestors[2] == expected[2]
