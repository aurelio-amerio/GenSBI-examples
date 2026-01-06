#%%
import numpy as np
import math
# %%
def compute_lr(batch_size, base_lr=1e-4, reference_batch_size=256):
    """Compute learning rate based on batch size.

    Args:
        batch_size (int): The current batch size.
        base_lr (float): The base learning rate for the reference batch size.
        reference_batch_size (int, optional): The reference batch size. Defaults to 256.

    Returns:
        float: The adjusted learning rate.
    """
    return base_lr * math.sqrt(batch_size / reference_batch_size)
# %%
compute_lr(4096)
# %%
