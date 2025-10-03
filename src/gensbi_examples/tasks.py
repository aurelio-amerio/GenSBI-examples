import jax
from jax import numpy as jnp
import grain
import numpy as np

from .utils import download_artifacts
from .graph import faithfull_mask, min_faithfull_mask, moralize


class Task:
    def __init__(self, task_name, data_dir=None, dtype=jnp.float32):
        self.task_name = task_name
        self.data_dir = data_dir or "./"
        download_artifacts(task=task_name, dir=self.data_dir)
        self.data = jnp.load(f"{self.data_dir}/task_data/data_{task_name}.npz")

        self.max_samples = int(1e6)

        self.xs = self.data["xs"][: self.max_samples]
        self.xs = self.xs.astype(dtype)
        self.thetas = self.data["thetas"][: self.max_samples]
        self.thetas = self.thetas.astype(dtype)

        self.xs_val = self.data["xs"][self.max_samples :]
        self.xs_val = self.xs_val.astype(dtype)
        self.thetas_val = self.data["thetas"][self.max_samples :]
        self.thetas_val = self.thetas_val.astype(dtype)

        self.observations = self.data["observations"]
        self.observations = self.observations.astype(dtype)

        self.reference_samples = self.data["reference_samples"]
        self.reference_samples = self.reference_samples.astype(dtype)

        self.true_parameters = self.data["true_parameters"]
        self.true_parameters = self.true_parameters.astype(dtype)
        
        self.dim_data = self.data["dim_data"]
        self.dim_theta = self.data["dim_theta"]
        self.dim_joint = self.dim_data + self.dim_theta
        self.num_observations = self.data["num_observations"]

    def get_train_dataset(self, batch_size, nsamples=1e5):
        assert (
            nsamples < self.max_samples
        ), f"nsamples must be less than {self.max_samples}"
        xs = self.xs[: int(nsamples)][...,None]
        thetas = self.thetas[: int(nsamples)][...,None]

        train_data = jnp.concatenate((thetas, xs), axis=1)

        dataset_grain = (
            grain.MapDataset.source(np.array(train_data))
            .shuffle(42)
            .repeat()
            .to_iter_dataset(
                grain.ReadOptions(num_threads=0, prefetch_buffer_size=batch_size * 5) # we set threads to 0, since the dataset fits in memory
            )
            .batch(batch_size=batch_size)  # Batches consecutive elements.
        )
        return dataset_grain

    def get_val_dataset(self):
        xs_val = self.xs_val[...,None]
        thetas_val = self.thetas_val[...,None]

        val_data = jnp.concatenate((thetas_val, xs_val), axis=1)

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
            def min_faithfull_edge_mask(node_id, condition_mask,meta_data=None):
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
    def __init__(self, data_dir=None, dtype=jnp.float32):
        task_name = "two_moons"
        super().__init__(task_name, data_dir, dtype=dtype)
    def get_base_mask_fn(self):
        theta_dim = self.dim_theta
        x_dim = self.dim_data
        thetas_mask = jnp.eye(theta_dim, dtype=jnp.bool_)
        x_mask = jnp.tril(jnp.ones((theta_dim, x_dim), dtype=jnp.bool_))
        base_mask = jnp.block([[thetas_mask, jnp.zeros((theta_dim, x_dim))], [jnp.ones((x_dim, theta_dim)), x_mask]])
        base_mask = base_mask.astype(jnp.bool_)
        def base_mask_fn(node_ids, node_meta_data):
            return base_mask[node_ids, :][:, node_ids]
        return base_mask_fn

class BernoulliGLM(Task):
    def __init__(self, data_dir=None, dtype=jnp.float32):
        task_name = "bernoulli_glm"
        super().__init__(task_name, data_dir, dtype=dtype)
    def get_base_mask_fn(self):
        raise NotImplementedError()

class GaussianLinear(Task):
    def __init__(self, data_dir=None, dtype=jnp.float32):
        task_name = "gaussian_linear"
        super().__init__(task_name, data_dir, dtype=dtype)
    def get_base_mask_fn(self):
        theta_dim = self.dim_theta
        x_dim = self.dim_data
        thetas_mask = jnp.eye(theta_dim, dtype=jnp.bool_)
        x_i_mask = jnp.eye(x_dim, dtype=jnp.bool_)
        base_mask = jnp.block([[thetas_mask, jnp.zeros((theta_dim, x_dim))], [jnp.eye((x_dim)), x_i_mask]])
        base_mask = base_mask.astype(jnp.bool_)
        def base_mask_fn(node_ids, node_meta_data):
            return base_mask[node_ids, :][:, node_ids]
        return base_mask_fn

class GaussianLinearUniform(Task):
    def __init__(self, data_dir=None, dtype=jnp.float32):
        task_name = "gaussian_linear_uniform"
        super().__init__(task_name, data_dir, dtype=dtype)
    def get_base_mask_fn(self):
        theta_dim = self.dim_theta
        x_dim = self.dim_data
        thetas_mask = jnp.eye(theta_dim, dtype=jnp.bool_)
        x_i_mask = jnp.eye(x_dim, dtype=jnp.bool_)
        base_mask = jnp.block([[thetas_mask, jnp.zeros((theta_dim, x_dim))], [jnp.eye((x_dim)), x_i_mask]])
        base_mask = base_mask.astype(jnp.bool_)
        def base_mask_fn(node_ids, node_meta_data):
            return base_mask[node_ids, :][:, node_ids]
        return base_mask_fn

class GaussianMixture(Task):
    def __init__(self, data_dir=None, dtype=jnp.float32):
        task_name = "gaussian_mixture"
        super().__init__(task_name, data_dir, dtype=dtype)
    def get_base_mask_fn(self):
        theta_dim = self.dim_theta
        x_dim = self.dim_data
        thetas_mask = jnp.eye(theta_dim, dtype=jnp.bool_)
        x_mask = jnp.tril(jnp.ones((theta_dim, x_dim), dtype=jnp.bool_))
        base_mask = jnp.block([[thetas_mask, jnp.zeros((theta_dim, x_dim))], [jnp.ones((x_dim, theta_dim)), x_mask]])
        base_mask = base_mask.astype(jnp.bool_)
        def base_mask_fn(node_ids, node_meta_data):
            return base_mask[node_ids, :][:, node_ids]
        return base_mask_fn

class SLCP(Task):
    def __init__(self, data_dir=None, dtype=jnp.float32):
        task_name = "slcp"
        super().__init__(task_name, data_dir, dtype=dtype)
    def get_base_mask_fn(self):
        theta_dim = self.dim_theta
        x_dim = self.dim_data
        thetas_mask = jnp.eye(theta_dim, dtype=jnp.bool_)
        x_i_dim = x_dim // 4
        x_i_mask = jax.scipy.linalg.block_diag(*tuple([jnp.tril(jnp.ones((x_i_dim,x_i_dim), dtype=jnp.bool_))]*4))
        base_mask = jnp.block([[thetas_mask, jnp.zeros((theta_dim,x_dim))], [jnp.ones((x_dim, theta_dim)), x_i_mask]])
        base_mask = base_mask.astype(jnp.bool_)
        def base_mask_fn(node_ids, node_meta_data):
            return base_mask[node_ids, :][:, node_ids]
        return base_mask_fn
    
def get_task(task_name, dtype=jnp.float32, data_dir=None):
    """
    Returns a Task object based on the task name.
    """
    task_name = task_name.lower()
    if task_name == "two_moons":
        return TwoMoons(data_dir, dtype=dtype)
    elif task_name == "bernoulli_glm":
        return BernoulliGLM(data_dir, dtype=dtype)
    elif task_name == "gaussian_linear":
        return GaussianLinear(data_dir, dtype=dtype)
    elif task_name == "gaussian_linear_uniform":
        return GaussianLinearUniform(data_dir, dtype=dtype)
    elif task_name == "gaussian_mixture":
        return GaussianMixture(data_dir, dtype=dtype)
    elif task_name == "slcp":
        return SLCP(data_dir, dtype=dtype)
    else:
        raise ValueError(f"Unknown task: {task_name}")