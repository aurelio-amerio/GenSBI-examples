#%%
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import jax 
from jax import numpy as jnp

from sbibm_jax.data import OnlineTaskDataset

import matplotlib.pyplot as plt
#%%
task = OnlineTaskDataset("gaussian_random_field", normalize=True, dtype=jnp.bfloat16)
# %%
train_dataset = task.get_online_train_loader(16)
# %%
for i in range(100):
    print(i)
    data = next(iter(train_dataset))
    x = np.asarray(data[1],dtype=np.float32)
# %%
data[0].shape, data[1].shape
# %%
plt.clf()
plt.imshow(x[1,:,:,0], vmin=-1, vmax=1, cmap="coolwarm")
plt.savefig("grf.png", dpi=300)
plt.show()

# %%
