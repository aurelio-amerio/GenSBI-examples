#%%
import os

experiment=3

os.environ["JAX_PLATFORMS"] = "cpu"

from gensbi.utils.misc import scale_lr
#%%
scale_lr(1024,1e-4,256)