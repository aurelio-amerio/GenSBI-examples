# %%
import jax.numpy as jnp
import numpy as np
from gensbi_examples.tasks import normalize, unnormalize


def test_unnormalize_scalar():
    """Test unnormalize with scalar inputs."""
    batch = jnp.array([1.0, 2.0, 3.0])
    mean = 0.0
    std = 1.0
    result = unnormalize(batch, mean, std)
    assert jnp.allclose(result, batch)

    mean = 1.0
    std = 2.0
    # expected: batch * 2 + 1
    # [1, 2, 3] * 2 + 1 = [3, 5, 7]
    expected = jnp.array([3.0, 5.0, 7.0])
    result = unnormalize(batch, mean, std)
    assert jnp.allclose(result, expected)


def test_unnormalize_array():
    """Test unnormalize with array inputs."""
    batch = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    mean = jnp.array([0.5, 1.5])
    std = jnp.array([2.0, 0.5])

    # expected:
    # [[1.0*2.0 + 0.5, 2.0*0.5 + 1.5],
    #  [3.0*2.0 + 0.5, 4.0*0.5 + 1.5]]
    # [[2.5, 2.5],
    #  [6.5, 3.5]]

    expected = jnp.array([[2.5, 2.5], [6.5, 3.5]])
    result = unnormalize(batch, mean, std)
    assert jnp.allclose(result, expected)


def test_normalize_unnormalize_roundtrip():
    """Test that normalize then unnormalize returns the original data."""
    rng = np.random.default_rng(42)
    batch = jnp.array(rng.normal(size=(10, 5)))
    mean = jnp.array(rng.normal(size=(5,)))
    std = jnp.array(rng.uniform(0.1, 2.0, size=(5,)))

    normalized = normalize(batch, mean, std)
    unnormalized = unnormalize(normalized, mean, std)

    assert jnp.allclose(unnormalized, batch, atol=1e-6)


def test_broadcasting():
    """Test that shapes broadcast correctly."""
    # Batch (N, D), Mean (D,), Std (D,) -> Output (N, D)
    batch = jnp.ones((10, 3))
    mean = jnp.array([1.0, 2.0, 3.0])
    std = jnp.array([0.5, 0.5, 0.5])

    result = unnormalize(batch, mean, std)
    assert result.shape == (10, 3)

    # Test broadcasting across batch dimension
    # Batch (N, D, C), Mean (1, D, C), Std (1, D, C)
    batch = jnp.ones((2, 3, 4))
    mean = jnp.ones((1, 3, 4)) * 2
    std = jnp.ones((1, 3, 4)) * 3

    result = unnormalize(batch, mean, std)
    # 1 * 3 + 2 = 5
    assert jnp.allclose(result, 5.0)
    assert result.shape == (2, 3, 4)


def test_dtype_casting():
    """Test that mean and std are cast to the batch dtype."""
    batch = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    mean = np.array([0.0], dtype=np.float64)
    std = np.array([1.0], dtype=np.float64)

    result = unnormalize(batch, mean, std)
    assert result.dtype == jnp.float32

    # Test consistency with batch dtype
    # We use what jnp.array gives us, which handles whether x64 is enabled
    batch_default = jnp.array([1.0, 2.0, 3.0])
    result_default = unnormalize(batch_default, mean, std)
    assert result_default.dtype == batch_default.dtype
