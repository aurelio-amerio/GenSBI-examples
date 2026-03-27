#!/usr/bin/env python3
"""
Generate LaTeX tables showing training hyperparameters for the best-performing
C2ST model configurations.

Produces 30 tables:  2 models × 5 tasks × 3 methods.
Each table has 3 budget columns (10k, 30k, 100k) with parameter rows.

Usage:
    python scripts/generate_config_tables.py
"""

import os
import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Configuration (mirrors plot_model_c2st_best.py)
# ---------------------------------------------------------------------------

BASE_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "examples",
    "sbi-benchmarks",
)
BASE_DIR = os.path.normpath(BASE_DIR)

STATS_DIR = os.path.join(BASE_DIR, "stats")

TASKS = [
    "two_moons",
    "bernoulli_glm",
    "gaussian_linear",
    "gaussian_mixture",
    "slcp",
]

TASK_LABELS = {
    "two_moons": "Two Moons",
    "bernoulli_glm": "Bernoulli GLM",
    "gaussian_linear": "Gaussian Linear",
    "gaussian_mixture": "Gaussian Mixture",
    "slcp": "SLCP",
}

BUDGETS = [10_000, 30_000, 100_000]

EXPERIMENT_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Methods grouped by model
FLUX_METHODS = ["flow_flux", "diffusion_flux", "score_matching_flux"]
FLUX1JOINT_METHODS = [
    "flow_flux1joint",
    "diffusion_flux1joint",
    "score_matching_flux1joint",
]

MODEL_GROUPS = {
    "Flux1": FLUX_METHODS,
    "Flux1Joint": FLUX1JOINT_METHODS,
}

METHOD_LABELS = {
    "flow_flux": "Flow Matching",
    "flow_flux1joint": "Flow Matching",
    "diffusion_flux": "Diffusion (EDM)",
    "diffusion_flux1joint": "Diffusion (EDM)",
    "score_matching_flux": "Score Matching",
    "score_matching_flux1joint": "Score Matching",
}


# ---------------------------------------------------------------------------
# Data loading (same as plot_model_c2st_best.py)
# ---------------------------------------------------------------------------


def load_all_data():
    """Load all CSV files into a dict keyed by (task, experiment_id)."""
    data = {}
    for task in TASKS:
        for exp_id in EXPERIMENT_IDS:
            csv_path = os.path.join(
                STATS_DIR, f"{task}_experiment_{exp_id}.csv"
            )
            try:
                df = pd.read_csv(csv_path)
                data[(task, exp_id)] = df
            except FileNotFoundError:
                print(f"WARNING: {csv_path} not found, skipping.")
    return data


def find_best_experiment(data, task, method, budget_idx):
    """
    Find the experiment ID with the lowest (best) C2ST for a given
    (task, method, budget_index).  Returns (best_c2st, best_exp_id).
    """
    min_val = np.inf
    min_exp = None
    for exp_id in EXPERIMENT_IDS:
        key = (task, exp_id)
        if key not in data:
            continue
        val = max(float(data[key][method].values[budget_idx]), 0.5)
        if val < min_val:
            min_val = val
            min_exp = exp_id
    return min_val, min_exp


# ---------------------------------------------------------------------------
# YAML config reading
# ---------------------------------------------------------------------------


def config_path(task, method, exp_id, budget):
    """Build path to the versioned YAML config."""
    return os.path.join(
        BASE_DIR,
        task,
        method,
        "config",
        f"v{exp_id}",
        f"config_{method}_sbibm_{budget}.yaml",
    )


def read_config(task, method, exp_id, budget):
    """Read a YAML config and return the parsed dict, or None."""
    path = config_path(task, method, exp_id, budget)
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"WARNING: config not found: {path}")
        return None


# ---------------------------------------------------------------------------
# Parameter extraction
# ---------------------------------------------------------------------------

# Parameters to display, in order: (display_name, extractor_function)
# Each extractor takes the full YAML dict and returns a string.


def _fmt_lr(val):
    """Format a learning rate like 4e-4 → $4 \\times 10^{-4}$."""
    if val is None:
        return "---"
    s = f"{val:.0e}"  # e.g. "4e-04"
    mantissa, exp = s.split("e")
    exp = int(exp)
    if mantissa == "1":
        return f"$10^{{{exp}}}$"
    return f"${mantissa} \\times 10^{{{exp}}}$"


def _fmt_int(val):
    """Format an integer, inserting thin-space thousands separators."""
    if val is None:
        return "---"
    val = int(val)
    if val >= 1000:
        # e.g. 50000 → "50\\,000", 4096 → "4096"
        s = f"{val:,}".replace(",", "\\,")
        return s
    return str(val)


PARAMETERS = [
    ("Batch size", lambda c: _fmt_int(c.get("training", {}).get("batch_size"))),
    ("Training steps", lambda c: _fmt_int(c.get("training", {}).get("nsteps"))),
    ("Peak learning rate", lambda c: _fmt_lr(c.get("optimizer", {}).get("max_lr"))),
    ("Minimum learning rate", lambda c: _fmt_lr(c.get("optimizer", {}).get("min_lr"))),
    ("Warmup steps", lambda c: _fmt_int(c.get("optimizer", {}).get("warmup_steps"))),
    ("EMA decay", lambda c: str(c.get("training", {}).get("ema_decay", "---"))),
    (
        "Single-stream blocks",
        lambda c: _fmt_int(c.get("model", {}).get("depth_single_blocks")),
    ),
    (
        "Double-stream blocks",
        lambda c: _fmt_int(c.get("model", {}).get("depth")) if c.get("model", {}).get("depth") is not None else "---",
    ),
    ("Attention heads", lambda c: _fmt_int(c.get("model", {}).get("num_heads"))),
]


# ---------------------------------------------------------------------------
# LaTeX generation
# ---------------------------------------------------------------------------


def generate_table(task, method, model_name, configs, c2st_values):
    """
    Generate a single LaTeX table string.

    Parameters
    ----------
    task : str
    method : str          e.g. "flow_flux"
    model_name : str      e.g. "Flux1"
    configs : dict        {budget: yaml_dict}  (one per budget)
    c2st_values : dict    {budget: float}
    """
    method_label = METHOD_LABELS[method]
    task_label = TASK_LABELS[task]

    budget_headers = " & ".join(
        [f"\\textbf{{{b // 1000}k}}" for b in BUDGETS]
    )

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        f"\\caption{{Training configuration for {model_name} — "
        f"{task_label} — {method_label}.}}"
    )
    lines.append(
        f"\\label{{tab:config_{model_name.lower()}_{task}_{method_label.lower().replace(' ', '_').replace('(', '').replace(')', '')}}}"
    )
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{@{}l" + "c" * len(BUDGETS) + r"@{}}")
    lines.append(r"\toprule")
    lines.append(
        f"\\textbf{{Parameter}} & {budget_headers} \\\\"
    )
    lines.append(r"\midrule")

    # Parameter rows
    for param_name, extractor in PARAMETERS:
        vals = []
        for budget in BUDGETS:
            cfg = configs.get(budget)
            if cfg is None:
                vals.append("---")
            else:
                vals.append(extractor(cfg))
        row = f"{param_name} & " + " & ".join(vals) + r" \\"
        lines.append(row)

    # C2ST row
    lines.append(r"\midrule")
    c2st_strs = []
    for budget in BUDGETS:
        v = c2st_values.get(budget)
        if v is not None:
            c2st_strs.append(f"{v:.4f}")
        else:
            c2st_strs.append("---")
    lines.append(f"Best C2ST & " + " & ".join(c2st_strs) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    data = load_all_data()

    for model_name, methods in MODEL_GROUPS.items():
        all_tables = []

        for task in TASKS:
            for method in methods:
                configs = {}
                c2st_values = {}

                for i_budget, budget in enumerate(BUDGETS):
                    best_c2st, best_exp = find_best_experiment(
                        data, task, method, i_budget
                    )
                    c2st_values[budget] = best_c2st

                    if best_exp is not None:
                        cfg = read_config(task, method, best_exp, budget)
                        configs[budget] = cfg
                    else:
                        print(
                            f"WARNING: no data for {task}/{method} "
                            f"at budget {budget}"
                        )

                table_tex = generate_table(
                    task, method, model_name, configs, c2st_values
                )
                all_tables.append(table_tex)

        # Write all tables for this model to one .tex file
        out_path = os.path.join(
            STATS_DIR, f"config_tables_{model_name.lower()}.tex"
        )
        with open(out_path, "w") as f:
            f.write(
                f"% Auto-generated configuration tables for {model_name}\n"
                f"% One table per (task × method), 3 budget columns\n\n"
            )
            f.write("\n\n".join(all_tables))
            f.write("\n")

        print(f"✓ Saved {out_path}  ({len(all_tables)} tables)")

    print("Done!")


if __name__ == "__main__":
    main()
