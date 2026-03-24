"""
generate_best_config_summary.py
Generates a markdown summary of the best model configuration version per task, model, and budget.
Only shows parameters that vary across versions for a given task/method.
"""

import os
import yaml
import glob
from collections import defaultdict

stats_dir = "/lhome/ific/a/aamerio/data/github/GenSBI-examples/examples/sbi-benchmarks/stats"
output_md = "/lhome/ific/a/aamerio/data/github/GenSBI-examples/examples/sbi-benchmarks/stats/best_configurations_summary.md"
base_dir = "/lhome/ific/a/aamerio/data/github/GenSBI-examples/examples/sbi-benchmarks"

tasks = ["two_moons", "bernoulli_glm", "gaussian_linear", "gaussian_mixture", "slcp"]
models = {
    "Flux1": ["flow_flux", "diffusion_flux", "score_matching_flux"],
    "Flux1Joint": ["flow_flux1joint", "diffusion_flux1joint", "score_matching_flux1joint"]
}
budgets = [10_000, 30_000, 100_000]


def flatten(d, parent_key="", sep="."):
    items = []
    if not isinstance(d, dict):
        return {parent_key: d}
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_best_version(task, method, budget):
    best_v = None
    min_c2st = float("inf")
    for exp in range(1, 15):
        csv_path = f"{stats_dir}/{task}_experiment_{exp}.csv"
        if not os.path.exists(csv_path):
            continue
        try:
            with open(csv_path, "r") as f:
                headers = f.readline().strip().split(",")
                if method not in headers:
                    continue
                m_idx = headers.index(method)
                b_idx = next(
                    (headers.index(x) for x in ["num_simulations", "simulations", "budget"] if x in headers), -1
                )
                if b_idx == -1:
                    continue
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) <= max(m_idx, b_idx):
                        continue
                    if parts[b_idx] == str(budget):
                        try:
                            c2st_f = float(parts[m_idx])
                            if c2st_f < min_c2st:
                                min_c2st = c2st_f
                                best_v = exp
                        except:
                            pass
        except:
            pass
    return best_v


def get_config_for_version(task, method, budget, v):
    """
    Get the config file path for a specific version v.
    v=1: fall back to root-level config (same for all budgets).
    v>1: look in config/v{v}/ for a file with the budget in the name, then any yaml.
    """
    config_dir = f"{base_dir}/{task}/{method}/config"

    if v == 1:
        # root level files are the v1 configs (budget-agnostic)
        root_yamls = [y for y in glob.glob(f"{config_dir}/*.yaml") if not os.path.isdir(y)]
        if root_yamls:
            return root_yamls[0]
        return None

    v_dir = f"{config_dir}/v{v}"
    if not os.path.isdir(v_dir):
        return None

    yamls = glob.glob(f"{v_dir}/*.yaml")
    if not yamls:
        return None

    # Prefer budget-specific file
    b_yamls = [y for y in yamls if str(budget) in os.path.basename(y)]
    if b_yamls:
        return b_yamls[0]

    return yamls[0]


def get_all_version_configs(task, method, budget):
    """
    Collect one config file per version (v1 = root file, v2..vN = versioned dirs).
    Returns dict: {v_str: filepath}
    """
    config_dir = f"{base_dir}/{task}/{method}/config"
    version_files = {}

    # v1 = root-level files
    root_yamls = [y for y in glob.glob(f"{config_dir}/*.yaml") if not os.path.isdir(y)]
    if root_yamls:
        version_files["1"] = root_yamls[0]

    # v2+ = versioned subdirectories
    for v_dir in sorted(glob.glob(f"{config_dir}/v*")):
        v_num = os.path.basename(v_dir).lstrip("v")
        yamls = glob.glob(f"{v_dir}/*.yaml")
        if not yamls:
            continue
        b_yamls = [y for y in yamls if str(budget) in os.path.basename(y)]
        version_files[v_num] = b_yamls[0] if b_yamls else yamls[0]

    return version_files


with open(output_md, "w") as out:
    out.write("# Best Model Configurations\n\n")
    out.write(
        "Only showing parameters that vary across configuration versions for a given task/method. "
        "Rows show the best configuration version for each simulation budget.\n\n"
    )

    for task in tasks:
        out.write(f"## Task: {task}\n\n")

        for model_name, methods in models.items():
            out.write(f"### Model: {model_name}\n\n")

            for budget in budgets:
                out.write(f"#### Budget: {budget // 1000}k\n\n")

                table_data = {}  # method -> {k: v}
                all_varying_keys = set()

                for method in methods:
                    best_v = get_best_version(task, method, budget)
                    if best_v is None:
                        continue

                    version_files = get_all_version_configs(task, method, budget)
                    if not version_files:
                        continue

                    # Find which params vary across versions
                    all_params = defaultdict(dict)
                    for v_str, cf in version_files.items():
                        try:
                            with open(cf, "r") as config_f:
                                flat_d = flatten(yaml.safe_load(config_f))
                                for k, v in flat_d.items():
                                    all_params[k][v_str] = str(v)
                        except:
                            pass

                    varying_params = [k for k, vals in all_params.items() if len(set(vals.values())) > 1]

                    # Load the best version's config
                    best_cf = get_config_for_version(task, method, budget, best_v)
                    if best_cf is None:
                        print(f"WARNING: [{task} - {method} - {budget}] config for best_v={best_v} not found")
                        continue

                    try:
                        with open(best_cf, "r") as f:
                            best_flat = flatten(yaml.safe_load(f))
                            row_data = {"__best_v__": best_v}
                            for k in varying_params:
                                row_data[k] = best_flat.get(k, "N/A")
                                all_varying_keys.add(k)
                            table_data[method] = row_data
                    except:
                        pass

                if not table_data:
                    out.write("*No data available.*\n\n")
                    continue

                sorted_keys = sorted(list(all_varying_keys))

                headers = ["Method", "Best Exp Version"] + sorted_keys
                out.write("| " + " | ".join(headers) + " |\n")
                out.write("|" + "|".join(["---"] * len(headers)) + "|\n")

                for method in methods:
                    if method not in table_data:
                        continue
                    row = table_data[method]
                    cols = [method, f"v{row['__best_v__']}"]
                    for k in sorted_keys:
                        cols.append(str(row.get(k, "-")))
                    out.write("| " + " | ".join(cols) + " |\n")

                out.write("\n")

print(f"Done! Markdown written to: {output_md}")
