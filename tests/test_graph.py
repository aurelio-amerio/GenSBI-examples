import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax  # noqa: E402, F401
import jax.numpy as jnp  # noqa: E402
from gensbi_examples.graph import find_ancestors_jax  # noqa: E402


def test_find_ancestors_jax():
    # Mask where rows are children, cols are parents
    # mask[child, parent] = True
    mask = jnp.zeros((6, 6), dtype=jnp.bool_)
    mask = mask.at[1, 0].set(True)
    mask = mask.at[2, 1].set(True)
    mask = mask.at[3, 0].set(True)
    mask = mask.at[2, 3].set(True)
    mask = mask.at[3, 4].set(True)

    # Ancestors of 2: 1, 3 (parents), 0 (parent of 1, 3), 4 (parent of 3)
    # so expected: 0, 1, 3, 4
    ans_2 = find_ancestors_jax(mask, 2)
    expected_2 = jnp.array([True, True, False, True, True, False], dtype=jnp.bool_)
    print("Ancestors of 2:", ans_2)

    assert (ans_2 == expected_2).all()
    print("Test passed!")


if __name__ == "__main__":
    test_find_ancestors_jax()
