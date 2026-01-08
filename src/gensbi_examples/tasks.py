import jax
from jax import numpy as jnp
import grain
import numpy as np

from datasets import load_dataset
from huggingface_hub import hf_hub_download
import json

# from .utils import download_artifacts
from .graph import faithfull_mask, min_faithfull_mask, moralize


def process_joint(batch):
    cond = batch["xs"][..., None]
    obs = batch["thetas"][..., None]
    data = np.concatenate((obs, cond), axis=1)
    return data


def process_conditional(batch):
    cond = batch["xs"][..., None]
    obs = batch["thetas"][..., None]
    return obs, cond


class Task:
    def __init__(self, task_name, kind="joint"):

        self.repo_name = "aurelio-amerio/SBI-benchmarks"

        self.task_name = task_name

        fname = hf_hub_download(
            repo_id=self.repo_name, filename="metadata.json", repo_type="dataset"
        )
        with open(fname, "r") as f:
            metadata = json.load(f)

        self.dataset = load_dataset(self.repo_name, task_name).with_format("numpy")
        self.dataset_posterior = load_dataset(
            self.repo_name, f"{task_name}_posterior"
        ).with_format("numpy")

        self.max_samples = self.dataset["train"].num_rows

        self.observations = self.dataset_posterior["reference_posterior"][
            "observations"
        ]
        self.reference_samples = self.dataset_posterior["reference_posterior"][
            "reference_samples"
        ]

        self.true_parameters = self.dataset_posterior["reference_posterior"][
            "true_parameters"
        ]

        self.dim_cond = metadata[task_name]["dim_cond"]
        self.dim_obs = metadata[task_name]["dim_obs"]

        self.dim_joint = self.dim_cond + self.dim_obs

        self.num_observations = len(self.observations)
        self.kind = kind

        if kind == "joint":
            self.process_fn = process_joint
        elif kind == "conditional":
            self.process_fn = process_conditional
        else:
            raise ValueError(f"Unknown kind: {kind}")

    def get_train_dataset(self, batch_size, nsamples=1e5):
        assert (
            nsamples < self.max_samples
        ), f"nsamples must be less than {self.max_samples}"

        df = self.dataset["train"].select(range(int(nsamples)))  # [:]

        dataset_grain = (
            grain.MapDataset.source(df).shuffle(42).repeat().to_iter_dataset()
        )

        performance_config = grain.experimental.pick_performance_config(
            ds=dataset_grain,
            ram_budget_mb=1024 * 4,
            max_workers=None,
            max_buffer_size=None,
        )

        dataset_batched = (
            dataset_grain.batch(batch_size)
            .map(self.process_fn)
            .mp_prefetch(performance_config.multiprocessing_options)
        )

        return dataset_batched

    def get_val_dataset(self, batch_size):
        df = self.dataset["validation"]  # [:]

        val_dataset_grain = (
            grain.MapDataset.source(df).shuffle(42).repeat().to_iter_dataset()
        )
        performance_config = grain.experimental.pick_performance_config(
            ds=val_dataset_grain,
            ram_budget_mb=1024 * 4,
            max_workers=None,
            max_buffer_size=None,
        )
        val_dataset_grain = (
            val_dataset_grain.batch(batch_size)
            .map(self.process_fn)
            .mp_prefetch(performance_config.multiprocessing_options)
        )

        return val_dataset_grain

    def get_test_dataset(self, batch_size):
        df = self.dataset["test"]  # [:]

        val_dataset_grain = (
            grain.MapDataset.source(df)
            .shuffle(42)
            .repeat()
            .to_iter_dataset()
            .batch(batch_size)
            .map(self.process_fn)
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

    def get_base_mask_fn(self):
        raise NotImplementedError()

    def get_edge_mask_fn(self, name="undirected"):
        if name.lower() == "faithfull":
            base_mask_fn = self.get_base_mask_fn()

            def faithfull_edge_mask(node_id, condition_mask, meta_data=None):
                base_mask = base_mask_fn(node_id, meta_data)
                return faithfull_mask(base_mask, condition_mask)

            return faithfull_edge_mask
        elif name.lower() == "min_faithfull":
            base_mask_fn = self.get_base_mask_fn()

            def min_faithfull_edge_mask(node_id, condition_mask, meta_data=None):
                base_mask = base_mask_fn(node_id, meta_data)

                return min_faithfull_mask(base_mask, condition_mask)

            return min_faithfull_edge_mask
        elif name.lower() == "undirected":
            base_mask_fn = self.get_base_mask_fn()

            def undirected_edge_mask(node_id, condition_mask, meta_data=None):
                base_mask = base_mask_fn(node_id, meta_data)
                return moralize(base_mask)

            return undirected_edge_mask

        elif name.lower() == "directed":
            base_mask_fn = self.get_base_mask_fn()

            def directed_edge_mask(node_id, condition_mask, meta_data=None):
                base_mask = base_mask_fn(node_id, meta_data)
                return base_mask

            return directed_edge_mask
        elif name.lower() == "none":
            return lambda node_id, condition_mask, *args, **kwargs: None
        else:
            raise NotImplementedError()


class TwoMoons(Task):
    def __init__(self, kind="joint"):
        task_name = "two_moons"
        super().__init__(task_name, kind=kind)

    def get_base_mask_fn(self):
        theta_dim = self.dim_obs
        x_dim = self.dim_cond
        thetas_mask = jnp.eye(theta_dim, dtype=jnp.bool_)
        x_mask = jnp.tril(jnp.ones((theta_dim, x_dim), dtype=jnp.bool_))
        base_mask = jnp.block(
            [
                [thetas_mask, jnp.zeros((theta_dim, x_dim))],
                [jnp.ones((x_dim, theta_dim)), x_mask],
            ]
        )
        base_mask = base_mask.astype(jnp.bool_)

        def base_mask_fn(node_ids, node_meta_data):
            return base_mask[node_ids, :][:, node_ids]

        return base_mask_fn


class BernoulliGLM(Task):
    def __init__(self, kind="joint"):
        task_name = "bernoulli_glm"
        super().__init__(task_name, kind=kind)

    def get_base_mask_fn(self):
        raise NotImplementedError()


class GaussianLinear(Task):
    def __init__(self, kind="joint"):
        task_name = "gaussian_linear"
        super().__init__(task_name, kind=kind)

    def get_base_mask_fn(self):
        theta_dim = self.dim_obs
        x_dim = self.dim_cond
        thetas_mask = jnp.eye(theta_dim, dtype=jnp.bool_)
        x_i_mask = jnp.eye(x_dim, dtype=jnp.bool_)
        base_mask = jnp.block(
            [[thetas_mask, jnp.zeros((theta_dim, x_dim))], [jnp.eye((x_dim)), x_i_mask]]
        )
        base_mask = base_mask.astype(jnp.bool_)

        def base_mask_fn(node_ids, node_meta_data):
            return base_mask[node_ids, :][:, node_ids]

        return base_mask_fn


class GaussianLinearUniform(Task):
    def __init__(self, kind="joint"):
        task_name = "gaussian_linear_uniform"
        super().__init__(task_name, kind=kind)

    def get_base_mask_fn(self):
        theta_dim = self.dim_obs
        x_dim = self.dim_cond
        thetas_mask = jnp.eye(theta_dim, dtype=jnp.bool_)
        x_i_mask = jnp.eye(x_dim, dtype=jnp.bool_)
        base_mask = jnp.block(
            [[thetas_mask, jnp.zeros((theta_dim, x_dim))], [jnp.eye((x_dim)), x_i_mask]]
        )
        base_mask = base_mask.astype(jnp.bool_)

        def base_mask_fn(node_ids, node_meta_data):
            return base_mask[node_ids, :][:, node_ids]

        return base_mask_fn


class GaussianMixture(Task):
    def __init__(self, kind="joint"):
        task_name = "gaussian_mixture"
        super().__init__(task_name, kind=kind)

    def get_base_mask_fn(self):
        theta_dim = self.dim_obs
        x_dim = self.dim_cond
        thetas_mask = jnp.eye(theta_dim, dtype=jnp.bool_)
        x_mask = jnp.tril(jnp.ones((theta_dim, x_dim), dtype=jnp.bool_))
        base_mask = jnp.block(
            [
                [thetas_mask, jnp.zeros((theta_dim, x_dim))],
                [jnp.ones((x_dim, theta_dim)), x_mask],
            ]
        )
        base_mask = base_mask.astype(jnp.bool_)

        def base_mask_fn(node_ids, node_meta_data):
            return base_mask[node_ids, :][:, node_ids]

        return base_mask_fn


class SLCP(Task):
    def __init__(self, kind="joint"):
        task_name = "slcp"
        super().__init__(task_name, kind=kind)

    def get_base_mask_fn(self):
        theta_dim = self.dim_obs
        x_dim = self.dim_cond
        thetas_mask = jnp.eye(theta_dim, dtype=jnp.bool_)
        x_i_dim = x_dim // 4
        x_i_mask = jax.scipy.linalg.block_diag(
            *tuple([jnp.tril(jnp.ones((x_i_dim, x_i_dim), dtype=jnp.bool_))] * 4)
        )
        base_mask = jnp.block(
            [
                [thetas_mask, jnp.zeros((theta_dim, x_dim))],
                [jnp.ones((x_dim, theta_dim)), x_i_mask],
            ]
        )
        base_mask = base_mask.astype(jnp.bool_)

        def base_mask_fn(node_ids, node_meta_data):
            return base_mask[node_ids, :][:, node_ids]

        return base_mask_fn


def get_task(task_name, kind="joint"):
    """
    Returns a Task object based on the task name.
    """
    task_name = task_name.lower()
    if task_name == "two_moons":
        return TwoMoons(kind=kind)
    elif task_name == "bernoulli_glm":
        return BernoulliGLM(kind=kind)
    elif task_name == "gaussian_linear":
        return GaussianLinear(kind=kind)
    elif task_name == "gaussian_linear_uniform":
        return GaussianLinearUniform(kind=kind)
    elif task_name == "gaussian_mixture":
        return GaussianMixture(kind=kind)
    elif task_name == "slcp":
        return SLCP(kind=kind)
    else:
        raise ValueError(f"Unknown task: {task_name}")
