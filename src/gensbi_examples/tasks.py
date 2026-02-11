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


def normalize(batch, mean, std):
    mean = jnp.asarray(mean, dtype=batch.dtype)
    std = jnp.asarray(std, dtype=batch.dtype)
    return (batch - mean) / std


def unnormalize(batch, mean, std):
    mean = jnp.asarray(mean, dtype=batch.dtype)
    std = jnp.asarray(std, dtype=batch.dtype)
    return batch * std + mean


def has_posterior_samples(task_name):
    if task_name in [
        "two_moons",
        "bernoulli_glm",
        "gaussian_linear",
        "gaussian_linear_uniform",
        "gaussian_mixture",
        "slcp",
    ]:
        return True
    else:
        return False


class Task:
    def __init__(self, task_name, kind="joint", seed=42, use_multiprocessing=True):

        self.repo_name = "aurelio-amerio/SBI-benchmarks"

        self.task_name = task_name
        self.seed = seed
        self.use_multiprocessing = use_multiprocessing

        fname = hf_hub_download(
            repo_id=self.repo_name, filename="metadata.json", repo_type="dataset"
        )
        with open(fname, "r") as f:
            metadata = json.load(f)

        self.dataset = load_dataset(self.repo_name, task_name).with_format("numpy")

        self.df_train = self.dataset["train"]
        self.df_val = self.dataset["validation"]
        self.df_test = self.dataset["test"]

        self.max_samples = self.df_train.num_rows

        if has_posterior_samples(task_name):
            self.dataset_posterior = load_dataset(
                self.repo_name, f"{task_name}_posterior"
            ).with_format("numpy")
            self.observations = self.dataset_posterior["reference_posterior"][
                "observations"
            ]
            self.reference_samples = self.dataset_posterior["reference_posterior"][
                "reference_samples"
            ]

            self.true_parameters = self.dataset_posterior["reference_posterior"][
                "true_parameters"
            ]
            self.num_observations = len(self.observations)
        else:
            self.dataset_posterior = None
            self.observations = None
            self.reference_samples = None
            self.true_parameters = None

        self.dim_cond = metadata[task_name]["dim_cond"]
        self.dim_obs = metadata[task_name]["dim_obs"]

        if kind == "joint":
            self.dim_joint = self.dim_cond + self.dim_obs
        elif kind == "conditional":
            self.dim_joint = None
        else:
            raise ValueError(f"Unknown kind: {kind}")

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

        df = self.df_train.select(range(int(nsamples)))  # [:]

        dataset_grain = (
            grain.MapDataset.source(df).shuffle(self.seed).repeat().to_iter_dataset()
        )

        performance_config = grain.experimental.pick_performance_config(
            ds=dataset_grain,
            ram_budget_mb=1024 * 4,
            max_workers=None,
            max_buffer_size=None,
        )

        dataset_batched = dataset_grain.batch(batch_size).map(self.process_fn)
        if self.use_multiprocessing:
            dataset_batched = dataset_batched.mp_prefetch(
                performance_config.multiprocessing_options
            )

        return dataset_batched

    def get_val_dataset(self, batch_size):
        df = self.df_val

        val_dataset_grain = (
            grain.MapDataset.source(df).shuffle(self.seed).repeat().to_iter_dataset()
        )
        performance_config = grain.experimental.pick_performance_config(
            ds=val_dataset_grain,
            ram_budget_mb=1024 * 4,
            max_workers=None,
            max_buffer_size=None,
        )
        val_dataset_grain = val_dataset_grain.batch(batch_size).map(self.process_fn)
        if self.use_multiprocessing:
            val_dataset_grain = val_dataset_grain.mp_prefetch(
                performance_config.multiprocessing_options
            )

        return val_dataset_grain

    def get_test_dataset(self, batch_size):
        df = self.df_test

        test_dataset_grain = (
            grain.MapDataset.source(df)
            .shuffle(self.seed)
            .repeat()
            .to_iter_dataset()
            .batch(batch_size)
            .map(self.process_fn)
        )

        return test_dataset_grain

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
    def __init__(self, kind="joint", **kwargs):
        task_name = "two_moons"
        super().__init__(task_name, kind=kind, **kwargs)

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
    def __init__(self, kind="joint", **kwargs):
        task_name = "bernoulli_glm"
        super().__init__(task_name, kind=kind, **kwargs)

    def get_base_mask_fn(self):
        raise NotImplementedError()


class GaussianLinear(Task):
    def __init__(self, kind="joint", **kwargs):
        task_name = "gaussian_linear"
        super().__init__(task_name, kind=kind, **kwargs)

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
    def __init__(self, kind="joint", **kwargs):
        task_name = "gaussian_linear_uniform"
        super().__init__(task_name, kind=kind, **kwargs)

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
    def __init__(self, kind="joint", **kwargs):
        task_name = "gaussian_mixture"
        super().__init__(task_name, kind=kind, **kwargs)

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
    def __init__(self, kind="joint", **kwargs):
        task_name = "slcp"
        super().__init__(task_name, kind=kind, **kwargs)

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


class GravitationalWaves(Task):
    def __init__(self, **kwargs):
        task_name = "gravitational_waves"
        super().__init__(task_name, kind="conditional", **kwargs)

        self.dim_obs = 2
        self.ch_obs = 1
        dim_cond_tot = self.dim_cond  # from super
        self.dim_cond = 8192
        self.ch_cond = 2
        assert (
            self.dim_cond == dim_cond_tot[0] and self.ch_cond == dim_cond_tot[1]
        ), f"Dimension mismatch, expected ({dim_cond_tot[0]}, {dim_cond_tot[1]}), got ({self.dim_cond}, {self.ch_cond})"

        self.xs_mean = jnp.array([[[0.00051776, -0.00040733]]], dtype=jnp.bfloat16)
        self.thetas_mean = jnp.array([[44.826576, 45.070328]], dtype=jnp.bfloat16)

        self.xs_std = jnp.array([[[60.80799, 59.33193]]], dtype=jnp.bfloat16)
        self.thetas_std = jnp.array([[20.189356, 20.16127]], dtype=jnp.bfloat16)

        self.process_fn = self.split_data

        return

    def split_data(self, batch):
        obs = jnp.array(batch["thetas"], dtype=jnp.bfloat16)
        obs = normalize(obs, self.thetas_mean, self.thetas_std)
        obs = obs.reshape(obs.shape[0], self.dim_obs, self.ch_obs)
        cond = jnp.array(batch["xs"], dtype=jnp.bfloat16)
        cond = normalize(cond, self.xs_mean, self.xs_std)
        cond = cond[..., None]
        return obs, cond

    def get_reference(self, num_observation=1):
        raise NotImplementedError(
            "Reference posterior samples not available for this task."
        )

    def get_true_parameters(self, num_observation=1):
        raise NotImplementedError("True parameters not available for this task.")


class GravitationalLensing(Task):
    def __init__(self, **kwargs):
        task_name = "lensing"
        super().__init__(task_name, kind="conditional", **kwargs)

        self.dim_obs = 2
        self.ch_obs = 1
        dim_cond_tot = self.dim_cond  # from super
        self.dim_cond = 32
        self.ch_cond = 32
        assert (
            self.dim_cond == dim_cond_tot[0] and self.ch_cond == dim_cond_tot[1]
        ), f"Dimension mismatch, expected ({dim_cond_tot[0]}, {dim_cond_tot[1]}), got ({self.dim_cond}, {self.ch_cond})"

        self.xs_mean = jnp.array([-1.1874731e-05], dtype=jnp.bfloat16).reshape(1, 1, 1)
        self.thetas_mean = jnp.array(
            [0.5996428, 0.15998043], dtype=jnp.bfloat16
        ).reshape(1, 2)

        self.xs_std = jnp.array([1.0440514], dtype=jnp.bfloat16).reshape(1, 1, 1)
        self.thetas_std = jnp.array(
            [0.2886958, 0.08657552], dtype=jnp.bfloat16
        ).reshape(1, 2)

        self.process_fn = self.split_data
        return

    def split_data(self, batch):
        obs = jnp.array(batch["thetas"], dtype=jnp.bfloat16)
        obs = normalize(obs, self.thetas_mean, self.thetas_std)
        obs = obs.reshape(obs.shape[0], self.dim_obs, self.ch_obs)
        cond = jnp.array(batch["xs"], dtype=jnp.bfloat16)
        cond = normalize(cond, self.xs_mean, self.xs_std)
        cond = cond[..., None]
        return obs, cond

    def get_reference(self, num_observation=1):
        raise NotImplementedError(
            "Reference posterior samples not available for this task."
        )

    def get_true_parameters(self, num_observation=1):
        raise NotImplementedError("True parameters not available for this task.")


def get_task(task_name, kind="conditional", **kwargs):
    """
    Returns a Task object based on the task name.
    """
    task_name = task_name.lower()
    if task_name == "two_moons":
        return TwoMoons(kind=kind, **kwargs)
    elif task_name == "bernoulli_glm":
        return BernoulliGLM(kind=kind, **kwargs)
    elif task_name == "gaussian_linear":
        return GaussianLinear(kind=kind, **kwargs)
    elif task_name == "gaussian_linear_uniform":
        return GaussianLinearUniform(kind=kind, **kwargs)
    elif task_name == "gaussian_mixture":
        return GaussianMixture(kind=kind, **kwargs)
    elif task_name == "slcp":
        return SLCP(kind=kind, **kwargs)
    elif task_name == "gravitational_waves":
        assert (
            kind == "conditional"
        ), "Gravitational waves task is only available in conditional mode."
        return GravitationalWaves(**kwargs)
    elif task_name == "gravitational_lensing":
        assert (
            kind == "conditional"
        ), "Gravitational lensing task is only available in conditional mode."
        return GravitationalLensing(**kwargs)
    else:
        raise ValueError(f"Unknown task: {task_name}")
