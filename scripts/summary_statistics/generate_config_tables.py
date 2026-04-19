#!/usr/bin/env python3
"""
Generate LaTeX tables showing training hyperparameters for the best-performing
C2ST model configurations.

Produces 10 wide tables: 2 models × 5 tasks.
Each table has 10 columns: 1 parameter name + 3 methods × 3 budgets.
Methods are separated by vertical lines, with multicolumn headers.

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

# STATS_DIR = os.path.join(BASE_DIR, "stats")
STATS_DIR = "/lhome/ific/a/aamerio/data/github/GenSBI-examples/examples/sbi-benchmarks/stats"

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

EXPERIMENT_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9,12,13]

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
    Experiments whose C2ST is NaN for this method/budget are skipped.
    """
    min_val = np.inf
    min_exp = None
    for exp_id in EXPERIMENT_IDS:
        key = (task, exp_id)
        if key not in data:
            continue
        raw = data[key][method].values[budget_idx]
        try:
            val = float(raw)
        except (ValueError, TypeError):
            continue
        if np.isnan(val):
            continue
        val = max(val, 0.5)
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
# Parameter extraction helpers
# ---------------------------------------------------------------------------


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
        s = f"{val:,}".replace(",", "\\,")
        return s
    return str(val)


def _get_merge_mode(cfg):
    """Return the id_merge_mode; defaults to 'sum' if absent."""
    if cfg is None:
        return "---"
    model = cfg.get("model", {})
    return model.get("id_merge_mode", "sum")


def _get_val_emb_dim(cfg):
    """Return val_emb_dim. For 'sum' mode, use axes_dim[0] instead."""
    if cfg is None:
        return "---"
    model = cfg.get("model", {})
    mode = _get_merge_mode(cfg)
    if mode == "sum":
        axes = model.get("axes_dim")
        if axes is not None:
            return str(axes[0]) if isinstance(axes, list) else str(axes)
        return "---"
    val = model.get("val_emb_dim")
    return str(val) if val is not None else "---"


def _get_id_emb_dim(cfg):
    """Return id_emb_dim. For 'sum' mode, return '---' (not applicable)."""
    if cfg is None:
        return "---"
    mode = _get_merge_mode(cfg)
    if mode == "sum":
        return "---"
    val = cfg.get("model", {}).get("id_emb_dim")
    return str(val) if val is not None else "---"


# Base parameters (common to both models)
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
        lambda c: _fmt_int(c.get("model", {}).get("depth"))
        if c.get("model", {}).get("depth") is not None
        else "---",
    ),
    ("Attention heads", lambda c: _fmt_int(c.get("model", {}).get("num_heads"))),
]

# Extra parameters for Flux1 only
FLUX1_EXTRA_PARAMETERS = [
    ("ID merge mode", lambda c: _get_merge_mode(c)),
    ("Val.~emb.~dim", lambda c: _get_val_emb_dim(c)),
    ("ID emb.~dim", lambda c: _get_id_emb_dim(c)),
]


# ---------------------------------------------------------------------------
# LaTeX generation — wide combined table
# ---------------------------------------------------------------------------


def generate_wide_table(task, model_name, methods, all_configs, all_c2st):
    """
    Generate a single wide LaTeX table with 10 columns:
    Parameter | 10k 30k 100k | 10k 30k 100k | 10k 30k 100k
              | -- method1 -- | -- method2 -- | -- method3 --

    Methods separated by vertical lines, multicolumn headers on top.
    """
    task_label = TASK_LABELS[task]
    method_labels = [METHOD_LABELS[m] for m in methods]
    n_budgets = len(BUDGETS)

    # Determine which parameters to show
    params = list(PARAMETERS)
    is_flux1 = model_name == "Flux1"
    if is_flux1:
        params.extend(FLUX1_EXTRA_PARAMETERS)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        f"\\caption{{Training configuration for {model_name} — "
        f"{task_label}.}}"
    )
    lines.append(f"\\label{{tab:config_{model_name.lower()}_{task}}}")
    lines.append(r"\scriptsize")
    lines.append(r"\resizebox{\textwidth}{!}{%")

    # Column spec: l | ccc | ccc | ccc
    col_spec = "@{}l|" + "|".join(["ccc"] * len(methods)) + "@{}"
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Row 1: multicolumn method headers
    header_parts = [""]
    for i, label in enumerate(method_labels):
        # Add right vertical line for all but the last method group
        col_fmt = "c|" if i < len(method_labels) - 1 else "c"
        header_parts.append(
            f"\\multicolumn{{{n_budgets}}}{{{col_fmt}}}{{{label}}}"
        )
    lines.append(" & ".join(header_parts) + r" \\")

    # Row 2: budget sub-headers
    budget_strs = [f"\\textbf{{{b // 1000}k}}" for b in BUDGETS]
    sub_header_parts = ["\\textbf{Parameter}"]
    for _ in methods:
        sub_header_parts.extend(budget_strs)
    lines.append(" & ".join(sub_header_parts) + r" \\")
    lines.append(r"\midrule")

    # Parameter rows
    for param_name, extractor in params:
        vals = []
        for method in methods:
            for budget in BUDGETS:
                cfg = all_configs[method].get(budget)
                if cfg is None:
                    vals.append("---")
                else:
                    vals.append(extractor(cfg))
        row = f"{param_name} & " + " & ".join(vals) + r" \\"
        lines.append(row)

    # C2ST row
    lines.append(r"\midrule")
    c2st_vals = []
    for method in methods:
        for budget in BUDGETS:
            v = all_c2st[method].get(budget)
            if v is not None:
                c2st_vals.append(f"{v:.3f}")
            else:
                c2st_vals.append("---")
    lines.append("Best C2ST & " + " & ".join(c2st_vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
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
            all_configs = {}  # {method: {budget: yaml_dict}}
            all_c2st = {}     # {method: {budget: float}}

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

                all_configs[method] = configs
                all_c2st[method] = c2st_values

            table_tex = generate_wide_table(
                task, model_name, methods, all_configs, all_c2st
            )
            all_tables.append(table_tex)

        # Write all tables for this model to one .tex file
        out_path = os.path.join(
            STATS_DIR, f"config_tables_{model_name.lower()}.tex"
        )
        with open(out_path, "w+") as f:
            f.write(
                f"% Auto-generated configuration tables for {model_name}\n"
                f"% One wide table per task, 3 methods side-by-side\n\n"
            )
            f.write("\n\n".join(all_tables))
            f.write("\n")

        print(f"✓ Saved {out_path}  ({len(all_tables)} tables)")

    print("Done!")


if __name__ == "__main__":
    main()
