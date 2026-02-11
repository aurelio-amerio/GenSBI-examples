#%%
import os

os.environ["JAX_PLATFORMS"] = "cpu"  # change to 'cpu' if no GPU is available

#%%
import numpy as np
from gensbi.diagnostics.metrics import c2st as c2st_gensbi
from gensbi_examples.c2st import c2st as c2st_examples
#%%
X = np.random.randn(10_000, 2)
Y = np.random.randn(10_000, 2) + 0.1*np.random.randn(10_000, 2)
# %%
X.shape
#%%
c2st_gensbi(X, Y)
# %%
c2st_examples(X, Y)
# %%
