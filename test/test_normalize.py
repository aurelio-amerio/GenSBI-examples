import jax.numpy as jnp
from gensbi_examples.tasks import normalize


def test_normalize_basic():
    """Test basic normalization with scalar mean and std."""
    batch = jnp.array([10.0, 20.0, 30.0])
    mean = 20.0
    std = 10.0
    expected = jnp.array([-1.0, 0.0, 1.0])
    result = normalize(batch, mean, std)
    assert jnp.allclose(result, expected)


def test_normalize_broadcasting():
    """Test normalization with array mean and std (broadcasting)."""
    batch = jnp.array([[10.0, 20.0], [30.0, 40.0]])
    mean = jnp.array([10.0, 20.0])
    std = jnp.array([1.0, 2.0])
    # (10-10)/1 = 0, (20-20)/2 = 0
    # (30-10)/1 = 20, (40-20)/2 = 10
    expected = jnp.array([[0.0, 0.0], [20.0, 10.0]])
    result = normalize(batch, mean, std)
    assert jnp.allclose(result, expected)


def test_normalize_dtype():
    """Test that the output dtype matches the input batch dtype."""
    batch = jnp.array([10, 20, 30], dtype=jnp.float32)
    mean = 20.0
    std = 10.0
    result = normalize(batch, mean, std)
    assert result.dtype == jnp.float32


def test_normalize_zero_std():
    """Test normalization with zero std (should result in inf or nan)."""
    batch = jnp.array([10.0])
    mean = 0.0
    std = 0.0
    result = normalize(batch, mean, std)
    assert jnp.isinf(result).all()
