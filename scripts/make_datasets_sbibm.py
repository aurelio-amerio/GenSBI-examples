# %%
print("hello")

# %%
import sbibm
import numpy as np
import os

# %%
sbibm.get_available_tasks()

os.makedirs("task_data", exist_ok=True)

# %%
def export_data_from_sbibm(task_name, nsamples=int(1e6)):
    fname = f"task_data/data_{task_name}.npz"
    if os.path.exists(fname):
        print(f"Data for task '{task_name}' already exists. Skipping export.")
        return
    
    task = sbibm.get_task(task_name)
    prior = task.get_prior()
    simulator = task.get_simulator()

    thetas = prior(nsamples)
    xs = simulator(thetas)
    dim_data = xs.shape[1]
    dim_theta = thetas.shape[1]
    num_observations = task.num_observations

    observation = np.array([np.array(task.get_observation(i)) for i in range(1, num_observations + 1)])
    reference_samples = np.array([np.array(task.get_reference_posterior_samples(num_observation=i)) for i in range(1, num_observations + 1)])
    true_parameters = np.array([np.array(task.get_true_parameters(i)) for i in range(1, num_observations + 1)])


    data_dict = {
        "xs": np.array(xs),
        "thetas": np.array(thetas),
        "observations": observation,
        "reference_samples": reference_samples,
        "true_parameters": true_parameters,
        "dim_data": dim_data,
        "dim_theta": dim_theta,
        "num_observations": num_observations,
    }

    np.savez_compressed(f"task_data/data_{task_name}.npz", **data_dict)
    print(f"Data for task '{task_name}' exported successfully.")
    return

# %% [markdown]
# # gaussian_mixture
nsamples = int(1e6) + 512

# %%
task_name = "gaussian_mixture"

# %%
export_data_from_sbibm(task_name, nsamples=nsamples)

# %% [markdown]
# # gaussian_linear_uniform

# %%
task_name = "gaussian_linear_uniform"

# %%
export_data_from_sbibm(task_name, nsamples=nsamples)

# %% [markdown]
# # two_moons 

# %%
task_name = "two_moons"

# %%
export_data_from_sbibm(task_name, nsamples=nsamples)

# %% [markdown]
# # bernoulli_glm 

# %%
task_name = "bernoulli_glm"

# %%
export_data_from_sbibm(task_name, nsamples=nsamples)

# %% [markdown]
# # lotka_volterra

# %%
task_name = "lotka_volterra"

# %%
# export_data_from_sbibm(task_name, nsamples=nsamples)

# %% [markdown]
# # sir

# %%
task_name = "sir"

# %%
# export_data_from_sbibm(task_name, nsamples=nsamples)

# %% [markdown]
# # slcp

# %%
task_name = "slcp"

# %%
export_data_from_sbibm(task_name, nsamples=nsamples)

# %% [markdown]
# # gaussian_linear

# %%
task_name = "gaussian_linear"

# %%
export_data_from_sbibm(task_name, nsamples=nsamples)

# %% [markdown]
# # slcp_distractors

# %%
task_name = "slcp_distractors"

# %%
export_data_from_sbibm(task_name, nsamples=nsamples)

# %% [markdown]
# # bernoulli_glm_raw

# %%
task_name = "bernoulli_glm_raw"

# %%
export_data_from_sbibm(task_name, nsamples=nsamples)


