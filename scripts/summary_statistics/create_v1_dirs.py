"""
create_v1_dirs.py
Copies root-level config yamls (no budget number in filename, e.g. config_diffusion_flux.yaml)
into a v1/ subdirectory, renaming them per-budget for consistency with v2+ configs.
"""
import os
import glob
import shutil
import re

base_dir = "/lhome/ific/a/aamerio/data/github/GenSBI-examples/examples/sbi-benchmarks"

tasks = ["two_moons", "bernoulli_glm", "gaussian_linear", "gaussian_mixture", "slcp"]
methods = [
    "flow_flux", "diffusion_flux", "score_matching_flux",
    "flow_flux1joint", "diffusion_flux1joint", "score_matching_flux1joint"
]
budgets = [10_000, 30_000, 100_000]

# Pattern for budget-named files (e.g. config_diffusion_flux_sbibm_10000.yaml or _30000.yaml)
budget_pattern = re.compile(r"_\d+\.yaml$")

for task in tasks:
    for method in methods:
        config_dir = f"{base_dir}/{task}/{method}/config"
        if not os.path.isdir(config_dir):
            continue

        # Find root-level yamls WITHOUT a budget number in the filename
        root_yamls = [
            y for y in glob.glob(f"{config_dir}/*.yaml")
            if os.path.isfile(y) and not budget_pattern.search(os.path.basename(y))
        ]
        if not root_yamls:
            print(f"No budget-free root yaml found for {task}/{method}, skipping")
            continue

        root_yaml = root_yamls[0]
        v1_dir = os.path.join(config_dir, "v1")
        os.makedirs(v1_dir, exist_ok=True)

        for budget in budgets:
            dest_name = f"config_{method}_sbibm_{budget}.yaml"
            dest_path = os.path.join(v1_dir, dest_name)
            shutil.copy(root_yaml, dest_path)
            print(f"Copied: {root_yaml} -> {dest_path}")

print("Done.")
