# %%
import os

# select device

os.environ["JAX_PLATFORMS"] = "cpu"

import pytest  # noqa: E402

import numpy as np  # noqa: E402

from gensbi_examples.tasks import get_task  # noqa: E402


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
        assert mask is None, "mask is not None"


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


# %%
def test_get_reference_error():
    task = get_task("two_moons", "joint", use_multiprocessing=False)

    with pytest.raises(ValueError, match="num_observation must be between 1 and"):
        task.get_reference(num_observation=0)

    with pytest.raises(ValueError, match="num_observation must be between 1 and"):
        task.get_reference(num_observation=task.num_observations + 1)
