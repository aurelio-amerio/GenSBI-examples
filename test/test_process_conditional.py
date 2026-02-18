import numpy as np
from gensbi_examples.tasks import process_conditional


def test_process_conditional_1d():
    batch = {
        "xs": np.array([1, 2, 3]),
        "thetas": np.array([4, 5, 6])
    }

    obs, cond = process_conditional(batch)

    # Check shapes
    assert obs.shape == (3, 1)
    assert cond.shape == (3, 1)

    # Check values
    assert np.array_equal(obs, np.array([[4], [5], [6]]))
    assert np.array_equal(cond, np.array([[1], [2], [3]]))


def test_process_conditional_2d():
    batch = {
        "xs": np.array([[1, 2], [3, 4]]),
        "thetas": np.array([[5, 6], [7, 8]])
    }

    obs, cond = process_conditional(batch)

    # Check shapes
    assert obs.shape == (2, 2, 1)
    assert cond.shape == (2, 2, 1)

    # Check values
    expected_obs = np.array([[[5], [6]], [[7], [8]]])
    expected_cond = np.array([[[1], [2]], [[3], [4]]])

    assert np.array_equal(obs, expected_obs)
    assert np.array_equal(cond, expected_cond)


def test_process_conditional_mixed_shapes():
    # It should work even if xs and thetas have different leading dimensions
    # as long as they are compatible with the slicing
    # But usually batch dimensions match.
    # Let's stick to matching batch dimensions.

    batch = {
        "xs": np.zeros((10, 5)),
        "thetas": np.zeros((10, 3))
    }

    obs, cond = process_conditional(batch)

    assert obs.shape == (10, 3, 1)
    assert cond.shape == (10, 5, 1)
