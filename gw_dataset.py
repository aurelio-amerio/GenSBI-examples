# %%
import os

os.environ["JAX_PLATFORMS"] = "cpu" 


from datasets import Dataset, Features, Array2D, Value, List

from gensbi_examples.tasks import get_task

from jax import numpy as jnp

import json
from huggingface_hub import upload_file

import numpy as np

# Your dictionary with metadata


# %%
import torch

# %%
repo_name = "aurelio-amerio/SBI-benchmarks"

# %%
# metadata = {}

# for task_name in tasks:
#     task = get_task(task_name)
#     dim_data = task.data["dim_data"].item()
#     dim_theta = task.data["dim_theta"].item()

#     metadata[task_name] = {"dim_data": dim_data, "dim_theta": dim_theta}

# file_path = "metadata.json"
# with open(file_path, 'w') as f:
#     json.dump(metadata, f, indent=4)

# %%
# upload_file(
#     path_or_fileobj=file_path,
#     path_in_repo="metadata.json",  # The name of the file in the repo
#     repo_id=repo_name,
#     repo_type="dataset",
# )

# %%
# upload dataset function
# %%
dir_GW = "/lhome/ific/a/aamerio/data/GW"

# %%
thetas = torch.load(f"{dir_GW}/thetas_0.pt")
theta1 = torch.load(f"{dir_GW}/thetas_1.pt")
theta2 = torch.load(f"{dir_GW}/thetas_2.pt")
theta3 = torch.load(f"{dir_GW}/thetas_3.pt")
theta4 = torch.load(f"{dir_GW}/thetas_4.pt")
theta5 = torch.load(f"{dir_GW}/thetas_5.pt")
theta6 = torch.load(f"{dir_GW}/thetas_6.pt")
theta7 = torch.load(f"{dir_GW}/thetas_7.pt")
theta8 = torch.load(f"{dir_GW}/thetas_8.pt")
theta9 = torch.load(f"{dir_GW}/thetas_9.pt")

xs_raw = torch.load(f"{dir_GW}/xs_0.pt")
xs_raw1 = torch.load(f"{dir_GW}/xs_1.pt")
xs_raw2 = torch.load(f"{dir_GW}/xs_2.pt")
xs_raw3 = torch.load(f"{dir_GW}/xs_3.pt")
xs_raw4 = torch.load(f"{dir_GW}/xs_4.pt")
xs_raw5 = torch.load(f"{dir_GW}/xs_5.pt")
xs_raw6 = torch.load(f"{dir_GW}/xs_6.pt")
xs_raw7 = torch.load(f"{dir_GW}/xs_7.pt")
xs_raw8 = torch.load(f"{dir_GW}/xs_8.pt")
xs_raw9 = torch.load(f"{dir_GW}/xs_9.pt")

# %%
thetas = torch.cat([thetas, theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8], dim=0)
xs_raw = torch.cat([xs_raw, xs_raw1, xs_raw2, xs_raw3, xs_raw4, xs_raw5, xs_raw6, xs_raw7, xs_raw8], dim=0)

# %%
thetas = thetas.numpy()
xs_raw = xs_raw.numpy()


thetas_test = theta9.numpy()
xs_test = xs_raw9.numpy()
#%%
thetas.shape, xs_raw.shape
# %%
xs_train = xs_raw[:-512]
xs_val = xs_raw[-512:]
thetas_train = thetas[:-512]
thetas_val = thetas[-512:]



# %%
# dataset_train = Dataset.from_dict({"xs": xs_train, "thetas": thetas_train})
# dataset_val = Dataset.from_dict({"xs": xs_val, "thetas": thetas_val})

#create a generator that yields the data in chunks to avoid memory issues
def data_generator(xs, thetas):
    for i in range(xs.shape[0]):
        yield {"xs": xs[i], "thetas": thetas[i]}    

features = Features({
    "xs": Array2D(shape=(2, 8192), dtype='float32'),
    "thetas": List(Value('float32')),
})

dataset_train = Dataset.from_generator(lambda: data_generator(xs_train, thetas_train), features=features)
dataset_val = Dataset.from_generator(lambda: data_generator(xs_val, thetas_val), features=features)
dataset_test = Dataset.from_generator(lambda: data_generator(xs_test, thetas_test), features=features)
#%%
dataset_test
# %%
dataset_train.push_to_hub(repo_name, config_name="gravitational_waves", split="train", private=False)
dataset_val.push_to_hub(repo_name, config_name="gravitational_waves", split="validation", private=False)
dataset_test.push_to_hub(repo_name, config_name="gravitational_waves", split="test", private=False)



# %%
