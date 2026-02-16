
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import pytest
import jax.numpy as jnp
from gensbi_examples.graph import moralize

def test_moralize_v_structure():
    """
    Test moralization on a V-structure: P1 -> C <- P2.
    Convention: Rows = Children, Columns = Parents.
    Matrix M:
    P1 (row 1): No parents (col 1=0, col 2=0, col 0=0)
    P2 (row 2): No parents
    C  (row 0): Parents P1 and P2. M[0, 1]=1, M[0, 2]=1.

    Expected result:
    Undirected edges: (1, 0), (0, 1), (2, 0), (0, 2).
    Moral edge: (1, 2) and (2, 1). (Parents of common child connected).
    """
    # Nodes: 0=Child, 1=Parent1, 2=Parent2
    M = jnp.zeros((3, 3), dtype=jnp.int32)
    # Child 0 has parent 1
    M = M.at[0, 1].set(1)
    # Child 0 has parent 2
    M = M.at[0, 2].set(1)

    moral_graph = moralize(M)

    # Check undirected edges existed (symmetry)
    assert moral_graph[0, 1]
    assert moral_graph[1, 0]
    assert moral_graph[0, 2]
    assert moral_graph[2, 0]

    # Check moral edge (marrying parents)
    assert moral_graph[1, 2]
    assert moral_graph[2, 1]

def test_moralize_chain():
    """
    Test moralization on a Chain: P -> M -> C.
    Convention: Rows = Children, Columns = Parents.
    Nodes: 0=P, 1=M, 2=C.
    M (row 1): Parent P (col 0). M[1, 0]=1. (P->M)
    C (row 2): Parent M (col 1). M[2, 1]=1. (M->C)
    P (row 0): No parents.

    Expected result:
    Undirected edges: (0, 1), (1, 0), (1, 2), (2, 1).
    NO moral edge between P and C (0 and 2).
    """
    M = jnp.zeros((3, 3), dtype=jnp.int32)
    # M (1) has parent P (0)
    M = M.at[1, 0].set(1)
    # C (2) has parent M (1)
    M = M.at[2, 1].set(1)

    moral_graph = moralize(M)

    # Check edges
    assert moral_graph[1, 0]
    assert moral_graph[0, 1]
    assert moral_graph[2, 1]
    assert moral_graph[1, 2]

    # Check NO edge between 0 and 2
    assert not moral_graph[0, 2]
    assert not moral_graph[2, 0]

def test_moralize_fork():
    """
    Test moralization on a Fork: C1 <- P -> C2.
    Convention: Rows = Children, Columns = Parents.
    Nodes: 0=P, 1=C1, 2=C2.
    C1 (row 1): Parent P (col 0). M[1, 0]=1.
    C2 (row 2): Parent P (col 0). M[2, 0]=1.
    P (row 0): No parents.

    Expected result:
    Undirected edges: (1, 0), (0, 1), (2, 0), (0, 2).
    NO moral edge between C1 and C2 (1 and 2). (Siblings are NOT connected).
    """
    M = jnp.zeros((3, 3), dtype=jnp.int32)
    # C1 (1) has parent P (0)
    M = M.at[1, 0].set(1)
    # C2 (2) has parent P (0)
    M = M.at[2, 0].set(1)

    moral_graph = moralize(M)

    # Check edges
    assert moral_graph[1, 0]
    assert moral_graph[0, 1]
    assert moral_graph[2, 0]
    assert moral_graph[0, 2]

    # Check NO edge between 1 and 2
    assert not moral_graph[1, 2]
    assert not moral_graph[2, 1]
