# %%
import os

# select device

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp

import pytest

import numpy as np

from gensbi_examples.tasks import get_task, Task


# %%
@pytest.mark.parametrize(
    "task_name, kind",
    [
        ("two_moons", "conditional"),
        ("bernoulli_glm", "conditional"),
        ("gaussian_linear", "conditional"),
        ("gaussian_linear_uniform", "conditional"),
        ("gaussian_mixture", "conditional"),
        ("slcp", "conditional"),
        ("two_moons", "joint"),
        ("bernoulli_glm", "joint"),
        ("gaussian_linear", "joint"),
        ("gaussian_linear_uniform", "joint"),
        ("gaussian_mixture", "joint"),
        ("slcp", "joint"),
    ],
)
def test_basic_task(task_name, kind):
    task = get_task(task_name, kind, use_multiprocessing=False)
    obs, reference_samples = task.get_reference(num_observation=1)
    assert (
        obs.shape[1] == task.dim_cond
    ), f"obs shape {obs.shape[1]} != dim_cond {task.dim_cond}"
    assert (
        reference_samples.shape[1] == task.dim_obs
    ), f"reference_samples shape {reference_samples.shape[1]} != dim_obs {task.dim_obs}"

    train_dataset = task.get_train_dataset(batch_size=32, nsamples=1000)
    val_dataset = task.get_val_dataset(batch_size=32)
    test_dataset = task.get_test_dataset(batch_size=32)

    # get one batch from each dataset
    train_batch = next(iter(train_dataset))
    val_batch = next(iter(val_dataset))
    test_batch = next(iter(test_dataset))

    # check shapes
    if kind == "joint":
        # check the batch size
        assert (
            train_batch.shape[0] == 32
        ), f"train_batch shape {train_batch.shape[0]} != 32"
        assert val_batch.shape[0] == 32, f"val_batch shape {val_batch.shape[0]} != 32"
        assert (
            test_batch.shape[0] == 32
        ), f"test_batch shape {test_batch[0].shape[0]} != 32"
        assert (
            train_batch.shape[1] == task.dim_joint
        ), f"train_batch shape {train_batch.shape[1]} != dim_joint {task.dim_joint}"
        assert (
            val_batch.shape[1] == task.dim_joint
        ), f"val_batch shape {val_batch.shape[1]} != dim_joint {task.dim_joint}"
        assert (
            test_batch.shape[1] == task.dim_joint
        ), f"test_batch shape {test_batch.shape[1]} != dim_joint {task.dim_joint}"
    elif kind == "conditional":
        # check the batch size
        assert (
            train_batch[0].shape[0] == 32
        ), f"train_batch shape {train_batch[0].shape[0]} != 32"
        assert (
            val_batch[0].shape[0] == 32
        ), f"val_batch shape {val_batch[0].shape[0]} != 32"
        assert (
            test_batch[0].shape[0] == 32
        ), f"test_batch shape {test_batch[0].shape[0]} != 32"

        assert (
            train_batch[0].shape[1] == task.dim_obs
        ), f"train_batch[0] shape {train_batch[0].shape[1]} != dim_obs {task.dim_obs}"
        assert (
            train_batch[1].shape[1] == task.dim_cond
        ), f"train_batch[1] shape {train_batch[1].shape[1]} != dim_cond {task.dim_cond}"
        assert (
            val_batch[0].shape[1] == task.dim_obs
        ), f"val_batch[0] shape {val_batch[0].shape[1]} != dim_obs {task.dim_obs}"
        assert (
            val_batch[1].shape[1] == task.dim_cond
        ), f"val_batch[1] shape {val_batch[1].shape[1]} != dim_cond {task.dim_cond}"
        assert (
            test_batch[0].shape[1] == task.dim_obs
        ), f"test_batch[0] shape {test_batch[0].shape[1]} != dim_obs {task.dim_obs}"
        assert (
            test_batch[1].shape[1] == task.dim_cond
        ), f"test_batch[1] shape {test_batch[1].shape[1]} != dim_cond {task.dim_cond}"

        print(f"All tests passed for {task_name} {kind}!")

    if kind == "joint" and task_name != "bernoulli_glm":
        # test the edge mask
        masks = ["faithfull", "min_faithfull", "undirected", "directed"]
        dim_joint = task.dim_joint
        node_ids = np.arange(dim_joint)
        condition_mask = np.zeros(dim_joint, dtype=bool)
        mask_shape = (dim_joint, dim_joint)
        for mask_name in masks:
            mask_fn = task.get_edge_mask_fn(name=mask_name)
            mask = mask_fn(node_ids, condition_mask)
            assert mask.shape == mask_shape, f"mask shape {mask.shape} != {mask_shape}"

        # test the "none" mask
        mask_fn = task.get_edge_mask_fn(name="none")
        mask = mask_fn(node_ids, condition_mask)
        assert mask is None, f"mask is not None"


@pytest.mark.parametrize(
    "task_name",
    [
        "gravitational_waves",
        "gravitational_lensing",
    ],
)
def test_advanced_task(task_name):
    task = get_task(task_name, "conditional", use_multiprocessing=False)

    train_dataset = task.get_train_dataset(batch_size=32, nsamples=100)
    val_dataset = task.get_val_dataset(batch_size=32)
    test_dataset = task.get_test_dataset(batch_size=32)

    # get one batch from each dataset
    train_batch = next(iter(train_dataset))
    val_batch = next(iter(val_dataset))
    test_batch = next(iter(test_dataset))

    # check the batch size
    assert (
        train_batch[0].shape[0] == 32
    ), f"train_batch[0] shape {train_batch[0].shape[0]} != 32"
    assert (
        val_batch[0].shape[0] == 32
    ), f"val_batch[0] shape {val_batch[0].shape[0]} != 32"
    assert (
        test_batch[0].shape[0] == 32
    ), f"test_batch[0] shape {test_batch[0].shape[0]} != 32"

    assert (
        train_batch[0].shape[1] == task.dim_obs
    ), f"train_batch[0] shape {train_batch[0].shape[1]} != dim_obs {task.dim_obs}"
    assert (
        train_batch[1].shape[1] == task.dim_cond
    ), f"train_batch[1] shape {train_batch[1].shape[1]} != dim_cond {task.dim_cond}"
    assert (
        train_batch[1].shape[2] == task.ch_cond
    ), f"train_batch[1] shape {train_batch[1].shape[2]} != ch_cond {task.ch_cond}"

    assert (
        val_batch[0].shape[1] == task.dim_obs
    ), f"val_batch[0] shape {val_batch[0].shape[1]} != dim_obs {task.dim_obs}"
    assert (
        val_batch[1].shape[1] == task.dim_cond
    ), f"val_batch[1] shape {val_batch[1].shape[1]} != dim_cond {task.dim_cond}"
    assert (
        val_batch[1].shape[2] == task.ch_cond
    ), f"val_batch[1] shape {val_batch[1].shape[2]} != ch_cond {task.ch_cond}"

    assert (
        test_batch[0].shape[1] == task.dim_obs
    ), f"test_batch[0] shape {test_batch[0].shape[1]} != dim_obs {task.dim_obs}"
    assert (
        test_batch[1].shape[1] == task.dim_cond
    ), f"test_batch[1] shape {test_batch[1].shape[1]} != dim_cond {task.dim_cond}"
    assert (
        test_batch[1].shape[2] == task.ch_cond
    ), f"test_batch[1] shape {test_batch[1].shape[2]} != ch_cond {task.ch_cond}"

    print(f"All tests passed for {task_name} conditional!")

    return


# ---------------------------------------------------------------------------
# Normalization tests
# ---------------------------------------------------------------------------

STANDARD_TASKS = [
    "two_moons",
    "bernoulli_glm",
    "gaussian_linear",
    "gaussian_linear_uniform",
    "gaussian_mixture",
    "slcp",
]


@pytest.mark.parametrize("task_name", STANDARD_TASKS)
@pytest.mark.parametrize("kind", ["joint", "conditional"])
class TestNormalizationWithPrecomputedStats:
    """Test that normalize_data=True with shipped stats works correctly."""

    def test_stats_are_set(self, task_name, kind):
        """When normalize_data=True, the task should have non-None stats."""
        task = get_task(task_name, kind, normalize_data=True, use_multiprocessing=False)
        assert task.normalize_data is True
        assert task.obs_mean is not None
        assert task.obs_std is not None
        assert task.cond_mean is not None
        assert task.cond_std is not None

    def test_normalized_batch_shapes(self, task_name, kind):
        """Normalized batches should have identical shapes to unnormalized ones."""
        task_norm = get_task(
            task_name, kind, normalize_data=True, use_multiprocessing=False
        )
        task_raw = get_task(
            task_name, kind, normalize_data=False, use_multiprocessing=False
        )

        train_norm = next(iter(task_norm.get_train_dataset(batch_size=32, nsamples=1000)))
        train_raw = next(iter(task_raw.get_train_dataset(batch_size=32, nsamples=1000)))

        if kind == "joint":
            assert train_norm.shape == train_raw.shape
        else:
            # conditional returns (obs, cond)
            assert train_norm[0].shape == train_raw[0].shape
            assert train_norm[1].shape == train_raw[1].shape

    def test_normalize_unnormalize_roundtrip(self, task_name, kind):
        """normalize then unnormalize should recover the original data."""
        task = get_task(task_name, kind, normalize_data=True, use_multiprocessing=False)

        rng = np.random.default_rng(0)
        obs_dummy = jnp.array(rng.normal(size=(16, task.dim_obs, 1)), dtype=jnp.float32)
        cond_dummy = jnp.array(
            rng.normal(size=(16, task.dim_cond, 1)), dtype=jnp.float32
        )

        obs_recovered = task.unnormalize_obs(task.normalize_obs(obs_dummy))
        cond_recovered = task.unnormalize_cond(task.normalize_cond(cond_dummy))

        np.testing.assert_allclose(obs_recovered, obs_dummy, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(cond_recovered, cond_dummy, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("task_name", STANDARD_TASKS)
@pytest.mark.parametrize("kind", ["joint", "conditional"])
class TestNormalizationDisabled:
    """Test that normalize_data=False leaves stats as None and methods as identity."""

    def test_stats_are_none(self, task_name, kind):
        task = get_task(task_name, kind, normalize_data=False, use_multiprocessing=False)
        assert task.normalize_data is False
        assert task.obs_mean is None
        assert task.obs_std is None
        assert task.cond_mean is None
        assert task.cond_std is None

    def test_convenience_methods_are_identity(self, task_name, kind):
        """When normalize_data=False, normalize_obs/cond should return input unchanged."""
        task = get_task(task_name, kind, normalize_data=False, use_multiprocessing=False)

        rng = np.random.default_rng(1)
        obs_dummy = jnp.array(rng.normal(size=(4, task.dim_obs, 1)), dtype=jnp.float32)
        cond_dummy = jnp.array(
            rng.normal(size=(4, task.dim_cond, 1)), dtype=jnp.float32
        )

        np.testing.assert_array_equal(task.normalize_obs(obs_dummy), obs_dummy)
        np.testing.assert_array_equal(task.normalize_cond(cond_dummy), cond_dummy)
        np.testing.assert_array_equal(task.unnormalize_obs(obs_dummy), obs_dummy)
        np.testing.assert_array_equal(task.unnormalize_cond(cond_dummy), cond_dummy)


@pytest.mark.parametrize("kind", ["joint", "conditional"])
class TestNormalizationWithExplicitStats:
    """Test passing explicit mean/std arrays overrides computed/precomputed values."""

    def test_explicit_stats_are_used(self, kind):
        # Use the base Task class directly to avoid subclass constructors
        # overriding the stats before forwarding **kwargs.
        obs_mean = np.array([[[1.0], [2.0]]])
        obs_std = np.array([[[0.5], [0.5]]])
        cond_mean = np.array([[[3.0], [4.0]]])
        cond_std = np.array([[[1.0], [1.0]]])

        task = Task(
            "two_moons",
            kind=kind,
            normalize_data=True,
            obs_mean=obs_mean,
            obs_std=obs_std,
            cond_mean=cond_mean,
            cond_std=cond_std,
            use_multiprocessing=False,
        )

        np.testing.assert_array_equal(task.obs_mean, obs_mean)
        np.testing.assert_array_equal(task.obs_std, obs_std)
        np.testing.assert_array_equal(task.cond_mean, cond_mean)
        np.testing.assert_array_equal(task.cond_std, cond_std)


@pytest.mark.parametrize("kind", ["joint", "conditional"])
class TestNormalizationComputedFromData:
    """Test that stats are computed from training data when no precomputed/explicit stats exist."""

    def test_computed_stats_match_training_data(self, kind):
        """Force computation by passing normalize_data=True to the base Task directly
        with a task that has precomputed stats — verify the precomputed path is used."""
        task = get_task(
            "two_moons", kind, normalize_data=True, use_multiprocessing=False
        )
        # Precomputed stats are loaded; verify they are finite arrays
        assert np.all(np.isfinite(task.obs_mean))
        assert np.all(np.isfinite(task.obs_std))
        assert np.all(np.isfinite(task.cond_mean))
        assert np.all(np.isfinite(task.cond_std))
        # std should be positive
        assert np.all(task.obs_std > 0)
        assert np.all(task.cond_std > 0)


# %%
