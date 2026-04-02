"""
generate_best_config_summary.py
Generates a markdown summary of the best model configuration version per task, model, and budget.
Only shows parameters that vary across versions for a given task/method.

Also writes a compact CSV (best_configurations.csv) with one row per task/model/method/budget,
containing the best version, its C2ST score, and the path to the config file.

All configs are expected to live in config/v{N}/ directories with budget-specific filenames.
"""

import os
import yaml
import glob
import pandas as pd
from collections import defaultdict

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
stats_dir = os.path.join(root_dir, "examples", "sbi-benchmarks", "stats")
output_md = os.path.join(stats_dir, "best_configurations_summary.md")
output_csv = os.path.join(stats_dir, "best_configurations.csv")
base_dir = os.path.join(root_dir, "examples", "sbi-benchmarks")

tasks = ["two_moons", "bernoulli_glm", "gaussian_linear", "gaussian_mixture", "slcp"]
models = {
    "Flux1": ["flow_flux", "diffusion_flux", "score_matching_flux"],
    "Flux1Joint": ["flow_flux1joint", "diffusion_flux1joint", "score_matching_flux1joint"],
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


def get_best_versions(task, method, budget, top_n=2):
    """Find the top N experiment versions with the lowest C2ST for a given task/method/budget.
    Returns list of (version, c2st) tuples sorted by c2st ascending."""
    results = []  # list of (exp, c2st)
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
                    (headers.index(x) for x in ["num_simulations", "simulations", "budget"] if x in headers),
                    -1,
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
                            import math
                            if not math.isnan(c2st_f):
                                results.append((exp, c2st_f))
                        except:
                            pass
        except:
            pass
    results.sort(key=lambda x: x[1])
    return results[:top_n]


def get_config_for_version(task, method, budget, v):
    """Get config yaml path from config/v{v}/ for a given budget."""
    v_dir = f"{base_dir}/{task}/{method}/config/v{v}"
    if not os.path.isdir(v_dir):
        return None
    yamls = glob.glob(f"{v_dir}/*.yaml")
    if not yamls:
        return None
    # Prefer budget-specific
    b_yamls = [y for y in yamls if str(budget) in os.path.basename(y)]
    return b_yamls[0] if b_yamls else yamls[0]


def get_all_version_configs(task, method, budget):
    """Collect one config file per version directory. Returns dict {v_num_str: filepath}."""
    config_dir = f"{base_dir}/{task}/{method}/config"
    version_files = {}
    for v_dir in sorted(glob.glob(f"{config_dir}/v*")):
        v_num = os.path.basename(v_dir).lstrip("v")
        yamls = glob.glob(f"{v_dir}/*.yaml")
        if not yamls:
            continue
        b_yamls = [y for y in yamls if str(budget) in os.path.basename(y)]
        version_files[v_num] = b_yamls[0] if b_yamls else yamls[0]
    return version_files


# --- Main ---
CSV_HEADERS = ["task", "model", "method", "budget", "best_version", "best_c2st", "config_path"]
csv_rows = []

with open(output_md, "w") as out:
    out.write("# Best Model Configurations\n\n")
    out.write(
        "Only showing parameters that vary across configuration versions for a given task/method. "
        "Rows show the best configuration version for each training method.\n\n"
    )

    for task in tasks:
        out.write(f"## Task: {task}\n\n")

        for model_name, methods in models.items():
            out.write(f"### Model: {model_name}\n\n")

            for budget in budgets:
                out.write(f"#### Budget: {budget // 1000}k\n\n")

                table_data = {}
                all_varying_keys = set()

                for method in methods:
                    top_versions = get_best_versions(task, method, budget)
                    if not top_versions:
                        continue
                    best_v, best_c2st = top_versions[0]
                    second_v, second_c2st = top_versions[1] if len(top_versions) > 1 else (None, None)

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

                    varying_params = [
                        k for k, vals in all_params.items()
                        if len(set(vals.values())) > 1 and k != "training.experiment_id"
                    ]

                    # Load the best version's config
                    best_cf = get_config_for_version(task, method, budget, best_v)
                    if best_cf is None:
                        print(f"WARNING: [{task} - {method} - {budget}] config for v{best_v} not found")
                        continue

                    try:
                        with open(best_cf, "r") as f:
                            best_flat = flatten(yaml.safe_load(f))
                            row_data = {
                                "__best_v__": best_v, "__c2st__": best_c2st,
                                "__2nd_v__": second_v, "__2nd_c2st__": second_c2st,
                                "__best_cf__": best_cf,
                            }
                            # Always include id_merge_mode; default to "sum" when absent
                            row_data["model.id_merge_mode"] = best_flat.get("model.id_merge_mode", "sum")
                            for k in varying_params:
                                row_data[k] = best_flat.get(k, "N/A")
                                all_varying_keys.add(k)
                            table_data[method] = row_data
                    except:
                        pass

                if not table_data:
                    out.write("*No data available.*\n\n")
                    continue

                # Remove model.id_merge_mode from varying keys if present (we handle it as fixed column)
                all_varying_keys.discard("model.id_merge_mode")
                sorted_keys = sorted(list(all_varying_keys))

                headers = ["Method", "Best", "2nd Best", "model.id_merge_mode"] + sorted_keys
                out.write("| " + " | ".join(headers) + " |\n")
                out.write("|" + "|".join(["---"] * len(headers)) + "|\n")

                for method in methods:
                    if method not in table_data:
                        continue
                    row = table_data[method]
                    best_str = f"v{row['__best_v__']} ({row['__c2st__']:.3f})"
                    if row["__2nd_v__"] is not None:
                        second_str = f"v{row['__2nd_v__']} ({row['__2nd_c2st__']:.3f})"
                    else:
                        second_str = "-"
                    cols = [method, best_str, second_str, str(row.get("model.id_merge_mode", "sum"))]
                    for k in sorted_keys:
                        cols.append(str(row.get(k, "-")))
                    out.write("| " + " | ".join(cols) + " |\n")

                    # Collect row for CSV
                    csv_rows.append({
                        "task": task,
                        "model": model_name,
                        "method": method,
                        "budget": budget,
                        "best_version": f"v{row['__best_v__']}",
                        "best_c2st": f"{row['__c2st__']:.4f}",
                        "config_path": row.get("__best_cf__", ""),
                    })

                out.write("\n")

# Write compact CSV
pd.DataFrame(csv_rows, columns=CSV_HEADERS).to_csv(output_csv, index=False)

print(f"Done! Markdown written to: {output_md}")
print(f"Done! CSV written to: {output_csv}")
