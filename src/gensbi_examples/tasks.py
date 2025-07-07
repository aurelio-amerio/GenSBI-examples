import jax
from jax import numpy as jnp
import grain
import numpy as np

from .utils import download_artifacts


class Task:
    def __init__(self, task_name, data_dir=None):
        self.task_name = task_name
        self.data_dir = data_dir or "./"
        download_artifacts(task=task_name, dir=self.data_dir)
        self.data = jnp.load(f"{self.data_dir}/task_data/data_{task_name}.npz")

        self.max_samples = int(1e6)

        self.xs = self.data["xs"][: self.max_samples]
        self.thetas = self.data["thetas"][: self.max_samples]

        self.xs_val = self.data["xs"][self.max_samples :]
        self.thetas_val = self.data["thetas"][self.max_samples :]

        self.observations = self.data["observations"]
        self.reference_samples = self.data["reference_samples"]
        self.true_parameters = self.data["true_parameters"]
        self.dim_data = self.data["dim_data"]
        self.dim_theta = self.data["dim_theta"]
        self.num_observations = self.data["num_observations"]

    def get_train_dataset(self, batch_size, nsamples=1e5):
        assert (
            nsamples < self.max_samples
        ), f"nsamples must be less than {self.max_samples}"
        xs = self.xs[: int(nsamples)]
        thetas = self.thetas[: int(nsamples)]

        train_data = jnp.concatenate((thetas, xs), axis=-1)

        dataset_grain = (
            grain.MapDataset.source(np.array(train_data))
            .shuffle(42)
            .repeat()
            .to_iter_dataset(
                grain.ReadOptions(num_threads=16, prefetch_buffer_size=batch_size * 5)
            )
            .batch(batch_size=batch_size)  # Batches consecutive elements.
        )
        return dataset_grain

    def get_val_dataset(self):
        xs_val = self.xs_val
        thetas_val = self.thetas_val

        val_data = jnp.concatenate((thetas_val, xs_val), axis=-1)

        val_dataset_grain = (
            grain.MapDataset.source(np.array(val_data))
            .shuffle(42)
            .repeat()
            .to_iter_dataset()
            .batch(batch_size=512)  # Batches consecutive elements.
        )
        return val_dataset_grain

    def get_reference(self, num_observation=1):
        """
        Returns the reference posterior samples for a given number of observations.
        """
        if num_observation < 1 or num_observation > self.num_observations:
            raise ValueError(
                f"num_observation must be between 1 and {self.num_observations}"
            )
        obs = self.observations[num_observation - 1]
        samples = self.reference_samples[num_observation - 1]
        return obs, samples
    
    def get_true_parameters(self, num_observation=1):
        """
        Returns the true parameters for a given number of observations.
        """
        if num_observation < 1 or num_observation > self.num_observations:
            raise ValueError(
                f"num_observation must be between 1 and {self.num_observations}"
            )
        return self.true_parameters[num_observation - 1]