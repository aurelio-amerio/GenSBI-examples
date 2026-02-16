import pytest
import jax
import jax.numpy as jnp
from gensbi_examples.mask import sample_random_conditional_mask

def test_sample_random_conditional_mask():
    """Test sample_random_conditional_mask function."""
    key = jax.random.PRNGKey(42)
    num_samples = 100
    theta_dim = 5
    x_dim = 10

    mask = sample_random_conditional_mask(
        key, num_samples, theta_dim, x_dim
    )

    # Check shape
    assert mask.shape == (num_samples, theta_dim + x_dim)

    # Check dtype
    assert mask.dtype == jnp.bool_

    # Check that not all elements are True in any row
    # The function explicitly sets a row to False if all elements are True
    all_true_mask = jnp.all(mask, axis=-1)
    assert not jnp.any(all_true_mask), "Found a row with all True values"

    # Check reproducibility
    mask2 = sample_random_conditional_mask(
        key, num_samples, theta_dim, x_dim
    )
    assert jnp.array_equal(mask, mask2)

    # Check with different parameters
    key2 = jax.random.PRNGKey(123)
    mask3 = sample_random_conditional_mask(
        key2, num_samples, theta_dim, x_dim
    )
    # With different key, mask should likely be different (though theoretically possible to be same)
    assert not jnp.array_equal(mask, mask3)

    # Test with alpha/beta params
    mask_params = sample_random_conditional_mask(
        key, num_samples, theta_dim, x_dim, alpha=2.0, beta=2.0
    )
    assert mask_params.shape == (num_samples, theta_dim + x_dim)


def test_sample_random_conditional_mask_edge_cases():
    """Test sample_random_conditional_mask with edge cases."""
    key = jax.random.PRNGKey(42)
    num_samples = 10

    # Minimal dimensions
    theta_dim = 1
    x_dim = 1
    mask = sample_random_conditional_mask(
        key, num_samples, theta_dim, x_dim
    )
    assert mask.shape == (num_samples, 2)
    # Even with minimal dimensions, should not be all True
    all_true = jnp.all(mask, axis=-1)
    assert not jnp.any(all_true)

    # Large dimensions
    theta_dim = 100
    x_dim = 100
    mask = sample_random_conditional_mask(
        key, num_samples, theta_dim, x_dim
    )
    assert mask.shape == (num_samples, 200)
    all_true = jnp.all(mask, axis=-1)
    assert not jnp.any(all_true)
