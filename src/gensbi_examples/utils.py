import os 
import os
from IPython import get_ipython

def download_artifacts(task=None, dir=None):
    """
    Downloads the artifacts from the GenSBI repository.
    """
    root = "https://github.com/aurelio-amerio/GenSBI-examples/releases/download"
    tag = "data-v0.1"
    if task is not None:
        fnames = [f"data_{task}.npz"]
    else:
        fnames =[
            "data_two_moons.npz",
            "data_bernoulli_glm.npz",
            "data_gaussian_linear.npz",
            "data_gaussian_linear_uniform.npz",
            "data_gaussian_mixture.npz",
            "data_slcp.npz"]
        
    fnames = [os.path.join(root, tag, fname) for fname in fnames]

    if dir is None:
        dir = os.path.join(os.getcwd(), "task_data")
    else:
        dir = os.path.join(dir, "task_data")
    os.makedirs(dir, exist_ok=True)
    for fname in fnames:
        local_fname = os.path.join(dir, os.path.basename(fname))
        if not os.path.exists(local_fname):
            print(f"Downloading {fname} to {local_fname}")
            os.system(f"wget -O {local_fname} {fname}")
        else:
            print(f"{local_fname} already exists, skipping download.")





    