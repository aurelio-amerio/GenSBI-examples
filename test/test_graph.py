import os

# select device
os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp  # noqa: E402
from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_linear_chain():
    """
    Test a linear chain: 0 -> 1 -> 2
    Ancestors of 2 should be {0, 1}
    Ancestors of 1 should be {0}
    Ancestors of 0 should be {}
    """
    num_nodes = 3
    # Row=Child, Col=Parent
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.int32)

    # 0 -> 1 => mask[1, 0] = 1
    mask = mask.at[1, 0].set(1)
    # 1 -> 2 => mask[2, 1] = 1
    mask = mask.at[2, 1].set(1)

    # Check ancestors of 2
    ancestors_2 = find_ancestors_jax(mask, 2)
    assert ancestors_2[0]
    assert ancestors_2[1]
    assert not ancestors_2[2]  # Self is not ancestor unless cycle

    # Check ancestors of 1
    ancestors_1 = find_ancestors_jax(mask, 1)
    assert ancestors_1[0]
    assert not ancestors_1[1]
    assert not ancestors_1[2]

    # Check ancestors of 0
    ancestors_0 = find_ancestors_jax(mask, 0)
    assert not jnp.any(ancestors_0)


def test_find_ancestors_branching():
    """
    Test branching: 0 -> 2, 1 -> 2
    Ancestors of 2 should be {0, 1}
    """
    num_nodes = 3
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.int32)

    # 0 -> 2 => mask[2, 0] = 1
    mask = mask.at[2, 0].set(1)
    # 1 -> 2 => mask[2, 1] = 1
    mask = mask.at[2, 1].set(1)

    ancestors_2 = find_ancestors_jax(mask, 2)
    assert ancestors_2[0]
    assert ancestors_2[1]
    assert not ancestors_2[2]

    ancestors_0 = find_ancestors_jax(mask, 0)
    assert not jnp.any(ancestors_0)


def test_find_ancestors_cycle():
    """
    Test cycle: 0 -> 1 -> 0
    Ancestors of 0 should include 1.
    Since it's a cycle, 1's parent is 0, so 0 is ancestor of 1.
    Also 0 -> 1 -> 0, so 0 is effectively an ancestor of itself via the cycle.

    Let's trace:
    Start 0. Parents {1}.
    Visit 1. Parents {0}.
    Visit 0. 0 is in ancestors? No. Add 0?
    The code:
      cond = value & (j != current_node) & (~is_ancestor[j])
      ...
      is_ancestor = is_ancestor.at[j].set(True)

    Step 1: Node 0. Parents {1}. j=1. is_ancestor[1]=True. Stack=[1].
    Step 2: Node 1. Parents {0}. j=0. is_ancestor[0]=False. j!=current(1). True.
            is_ancestor[0]=True. Stack=[0].
    Step 3: Node 0. Parents {1}. j=1. is_ancestor[1]=True. Cond False.

    So both 0 and 1 should be marked as ancestors for node 0.
    """
    num_nodes = 2
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.int32)

    # 0 -> 1 => mask[1, 0] = 1
    mask = mask.at[1, 0].set(1)
    # 1 -> 0 => mask[0, 1] = 1
    mask = mask.at[0, 1].set(1)

    ancestors_0 = find_ancestors_jax(mask, 0)
    assert ancestors_0[1]
    assert ancestors_0[0]  # Self-loop via cycle

    ancestors_1 = find_ancestors_jax(mask, 1)
    assert ancestors_1[0]
    assert ancestors_1[1]


def test_find_ancestors_disconnected():
    """
    Test disconnected graph: 0   1
    Ancestors of 0 should be {}
    """
    num_nodes = 2
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.int32)

    ancestors_0 = find_ancestors_jax(mask, 0)
    assert not jnp.any(ancestors_0)


def test_find_ancestors_self_loop():
    """
    Test immediate self loop: 0 -> 0
    The code explicitly checks `j != current_node`.
    So 0 should NOT be added as ancestor of 0.
    """
    num_nodes = 1
    mask = jnp.zeros((num_nodes, num_nodes), dtype=jnp.int32)
    # 0 -> 0 => mask[0, 0] = 1
    mask = mask.at[0, 0].set(1)

    ancestors_0 = find_ancestors_jax(mask, 0)
    assert not ancestors_0[0]
