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
            "data_twp_moons.npz",
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




# def get_notebook_path():
#     ipython = get_ipython()
#     if ipython is None:
#         return None
#     try:
#         # For Jupyter Lab/Notebook
#         import ipykernel
#         from notebook import notebookapp
#         import urllib
#         import json
#         import requests
#         connection_file = ipykernel.get_connection_file()
#         kernel_id = connection_file.split('-')[-1].split('.')[0]
#         for srv in notebookapp.list_running_servers():
#             response = requests.get(urllib.parse.urljoin(srv['url'], 'api/sessions'), params={'token': srv.get('token', '')})
#             for sess in json.loads(response.text):
#                 if sess['kernel']['id'] == kernel_id:
#                     return os.path.abspath(os.path.join(srv['notebook_dir'], sess['notebook']['path']))
#     except Exception as e:
#         print(f"Could not get notebook path: {e}")
#         return None
    
# def set_base_path():
#     notebook_path = get_notebook_path()
#     if notebook_path is not None:
#         notebook_dir = os.path.dirname(notebook_path)
#     else:
#         notebook_dir = os.getcwd()  # fallback

#     print("Notebook directory:", notebook_dir)
#     return




    