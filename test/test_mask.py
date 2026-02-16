
import os
import pytest
from functools import partial

# Set JAX to use CPU
os.environ["JAX_PLATFORMS"] = "cpu"

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from gensbi_examples.mask import get_condition_mask_fn  # noqa: E402


@pytest.mark.parametrize("name", ["structured_random", "random", "joint", "posterior", "likelihood"])
def test_get_condition_mask_fn_valid_names(name):
    """Test get_condition_mask_fn with all valid names."""
    fn = get_condition_mask_fn(name)
    assert isinstance(fn, partial)

    # Dummy arguments for calling the function
    key = jax.random.PRNGKey(0)
    num_samples = 10
    theta_dim = 2
    x_dim = 3

    mask = fn(key=key, num_samples=num_samples, theta_dim=theta_dim, x_dim=x_dim)

    assert mask.shape == (num_samples, theta_dim + x_dim)
    assert mask.dtype == jnp.bool_


def test_get_condition_mask_fn_invalid_name():
    """Test get_condition_mask_fn with an invalid name."""
    with pytest.raises(NotImplementedError):
        get_condition_mask_fn("invalid_name")


def test_get_condition_mask_fn_kwargs():
    """Test passing kwargs to get_condition_mask_fn."""
    # Test with 'random' which accepts alpha and beta
    fn = get_condition_mask_fn("random", alpha=0.5, beta=2.0)
    assert isinstance(fn, partial)
    assert fn.keywords['alpha'] == 0.5
    assert fn.keywords['beta'] == 2.0

    # Verify execution works with kwargs
    key = jax.random.PRNGKey(0)
    num_samples = 10
    theta_dim = 2
    x_dim = 3

    mask = fn(key=key, num_samples=num_samples, theta_dim=theta_dim, x_dim=x_dim)
    assert mask.shape == (num_samples, theta_dim + x_dim)
