#!/usr/bin/env python3
"""
Compile SBIBM C2ST EMA results into CSV tables.

Produces one CSV per task (e.g. two_moons.csv, bernoulli_glm.csv, …)
with columns for each method/model variant and rows for each simulation budget.

Usage:
    python scripts/compile_sbibm_results.py [--base-dir <path>] [--output-dir <path>]
"""

import os
import re
import csv
import argparse

# ---------- configuration ----------

TASKS = [
    "two_moons",
    "bernoulli_glm",
    "gaussian_linear",
    "gaussian_mixture",
    "slcp",
]

METHODS = [
    "flow_flux",
    "flow_flux1joint",
    "diffusion_flux",
    "diffusion_flux1joint",
    "score_matching_flux",
    "score_matching_flux1joint",
]

BUDGETS = [10_000, 30_000, 100_000]

EXPERIMENT_ID = 1  # matches the experiment_id used during training

# regex to extract "Average C2ST accuracy EMA: 0.5771 +- 0.0108"
PATTERN = re.compile(
    r"Average C2ST accuracy EMA:\s+([\d.]+)\s*\+-\s*([\d.]+)"
)


# ---------- helpers ----------

def read_c2st_ema(filepath: str) -> str:
    """Return 'mean +- std' string from a c2st results file, or 'NaN'."""
    try:
        with open(filepath, "r") as f:
            for line in f:
                m = PATTERN.search(line)
                if m:
                    mean, std = m.group(1), m.group(2)
                    return f"{mean} +- {std}"
    except FileNotFoundError:
        return "NaN"
    return "NaN"


def build_result_path(base_dir: str, task: str, method: str, budget: int) -> str:
    """
    Construct the expected path to the EMA C2ST results file.

    Pattern:
        <base_dir>/<task>/<method>/sbibm/<budget>/c2st_results/c2st_results_ema_<id>_<method>.txt
    """
    return os.path.join(
        base_dir,
        task,
        method,
        "sbibm",
        str(budget),
        "c2st_results",
        f"c2st_results_ema_{EXPERIMENT_ID}_{method}.txt",
    )


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(
        description="Compile SBIBM C2ST EMA results into CSV files."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="examples/sbi-benchmarks",
        help="Root directory containing <task>/<method>/sbibm/... results "
             "(default: examples/sbi-benchmarks)",
    )
    args = parser.parse_args()

    base_dir = args.base_dir

    missing = []

    for task in TASKS:
        rows = []
        for budget in BUDGETS:
            row = {"simulations": budget}
            for method in METHODS:
                fpath = build_result_path(base_dir, task, method, budget)
                value = read_c2st_ema(fpath)
                if value == "NaN":
                    missing.append(f"  MISSING: {task} / {method} / {budget}")
                row[method] = value
            rows.append(row)

        # Write CSV inside the task directory
        task_dir = os.path.join(base_dir, task)
        csv_path = os.path.join(task_dir, f"{task}.csv")
        fieldnames = ["simulations"] + METHODS
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"✓ Saved {csv_path}")

    # Report missing combinations
    if missing:
        print(f"\n⚠ {len(missing)} missing result(s):")
        for m in missing:
            print(m)
    else:
        print("\nAll result files found — no missing combinations.")


if __name__ == "__main__":
    main()
