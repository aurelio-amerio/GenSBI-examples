"""
analyze_id_merge_mode.py
Compare concat vs sum id_merge_mode across Flux1 tasks, versions, and budgets.
Flux1Joint always uses concat (verified below).
"""

import os, yaml, glob
from collections import defaultdict

stats_dir = "/lhome/ific/a/aamerio/data/github/GenSBI-examples/examples/sbi-benchmarks/stats"
base_dir = "/lhome/ific/a/aamerio/data/github/GenSBI-examples/examples/sbi-benchmarks"

tasks = ["two_moons", "bernoulli_glm", "gaussian_linear", "gaussian_mixture", "slcp"]
flux1_methods = ["flow_flux", "diffusion_flux", "score_matching_flux"]
flux1joint_methods = ["flow_flux1joint", "diffusion_flux1joint", "score_matching_flux1joint"]
budgets = [10_000, 30_000, 100_000]


def get_c2st(task, method, budget, exp):
    csv_path = f"{stats_dir}/{task}_experiment_{exp}.csv"
    if not os.path.exists(csv_path):
        return None
    try:
        with open(csv_path) as f:
            headers = f.readline().strip().split(",")
            if method not in headers:
                return None
            m_idx = headers.index(method)
            b_idx = next(
                (headers.index(x) for x in ["num_simulations", "simulations", "budget"] if x in headers), -1
            )
            if b_idx == -1:
                return None
            for line in f:
                parts = line.strip().split(",")
                if len(parts) <= max(m_idx, b_idx):
                    continue
                if parts[b_idx] == str(budget):
                    try:
                        import math
                        v = float(parts[m_idx])
                        return v if not math.isnan(v) else None
                    except:
                        return None
    except:
        return None
    return None


def get_merge_mode(task, method, budget, v):
    config_dir = f"{base_dir}/{task}/{method}/config"
    v_dir = f"{config_dir}/v{v}"
    if not os.path.isdir(v_dir):
        return None
    yamls = glob.glob(f"{v_dir}/*.yaml")
    if not yamls:
        return None
    b_yamls = [y for y in yamls if str(budget) in os.path.basename(y)]
    cf = b_yamls[0] if b_yamls else yamls[0]
    try:
        with open(cf) as f:
            d = yaml.safe_load(f)
            model = d.get("model", {})
            return model.get("id_merge_mode", None)  # None means default
    except:
        return None


# ============ Flux1 Analysis ============
print("=" * 80)
print("FLUX1: id_merge_mode analysis (default = 'sum')")
print("=" * 80)

for task in tasks:
    print(f"\n## Task: {task}")
    for method in flux1_methods:
        print(f"\n  ### {method}")
        for budget in budgets:
            sum_results = []
            concat_results = []
            for v in range(1, 15):
                c2st = get_c2st(task, method, budget, v)
                if c2st is None:
                    continue
                mode = get_merge_mode(task, method, budget, v)
                effective_mode = mode if mode else "sum"  # default is sum for flux1
                if effective_mode == "sum":
                    sum_results.append((v, c2st))
                else:
                    concat_results.append((v, c2st))

            if sum_results:
                best_sum = min(sum_results, key=lambda x: x[1])
                print(f"    Budget {budget//1000}k  SUM:    best v{best_sum[0]} ({best_sum[1]:.3f})  [from {len(sum_results)} versions]")
            else:
                print(f"    Budget {budget//1000}k  SUM:    no data")

            if concat_results:
                best_concat = min(concat_results, key=lambda x: x[1])
                print(f"    Budget {budget//1000}k  CONCAT: best v{best_concat[0]} ({best_concat[1]:.3f})  [from {len(concat_results)} versions]")
            else:
                print(f"    Budget {budget//1000}k  CONCAT: no data")

            if sum_results and concat_results:
                best_s = min(sum_results, key=lambda x: x[1])
                best_c = min(concat_results, key=lambda x: x[1])
                winner = "SUM" if best_s[1] <= best_c[1] else "CONCAT"
                diff = abs(best_s[1] - best_c[1])
                print(f"    >>> Winner: {winner} (by {diff:.3f})")

# ============ Flux1Joint Verification ============
print("\n" + "=" * 80)
print("FLUX1JOINT: id_merge_mode verification")
print("=" * 80)

all_concat = True
for task in tasks:
    for method in flux1joint_methods:
        for v in range(1, 15):
            for budget in budgets:
                mode = get_merge_mode(task, method, budget, v)
                if mode is not None and mode != "concat":
                    print(f"  FOUND NON-CONCAT: {task}/{method}/v{v}/{budget} -> {mode}")
                    all_concat = False

if all_concat:
    print("\nConfirmed: All Flux1Joint configs use 'concat' (explicitly or by default).")
else:
    print("\nWARNING: Some Flux1Joint configs use a different merge mode!")

# ============ Summary Table ============
print("\n" + "=" * 80)
print("SUMMARY: Best merge mode per task/method/budget (Flux1 only)")
print("=" * 80)
print(f"\n{'Task':<20} {'Method':<25} {'Budget':<8} {'Best Mode':<10} {'Best C2ST':<10} {'Alt C2ST':<10} {'Diff':<8}")
print("-" * 90)

for task in tasks:
    for method in flux1_methods:
        for budget in budgets:
            sum_results = []
            concat_results = []
            for v in range(1, 15):
                c2st = get_c2st(task, method, budget, v)
                if c2st is None:
                    continue
                mode = get_merge_mode(task, method, budget, v)
                effective_mode = mode if mode else "sum"
                if effective_mode == "sum":
                    sum_results.append((v, c2st))
                else:
                    concat_results.append((v, c2st))

            if sum_results and concat_results:
                best_s = min(sum_results, key=lambda x: x[1])[1]
                best_c = min(concat_results, key=lambda x: x[1])[1]
                winner = "sum" if best_s <= best_c else "concat"
                loser_c2st = best_c if winner == "sum" else best_s
                winner_c2st = best_s if winner == "sum" else best_c
                diff = loser_c2st - winner_c2st
                print(f"{task:<20} {method:<25} {budget//1000}k{'':<5} {winner:<10} {winner_c2st:<10.3f} {loser_c2st:<10.3f} {diff:<8.3f}")
            elif sum_results:
                best_s = min(sum_results, key=lambda x: x[1])[1]
                print(f"{task:<20} {method:<25} {budget//1000}k{'':<5} {'sum*':<10} {best_s:<10.3f} {'N/A':<10} {'N/A':<8}")
            elif concat_results:
                best_c = min(concat_results, key=lambda x: x[1])[1]
                print(f"{task:<20} {method:<25} {budget//1000}k{'':<5} {'concat*':<10} {best_c:<10.3f} {'N/A':<10} {'N/A':<8}")

print("\n* = only one mode was tested, no comparison available")
