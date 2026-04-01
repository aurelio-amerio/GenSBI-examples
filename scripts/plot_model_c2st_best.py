# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gensbi.utils.plotting import set_default_style

set_default_style()

TASKS = [
    "two_moons",
    "bernoulli_glm",
    "gaussian_linear",
    "gaussian_mixture",
    "slcp",
]

# Pretty labels for each task (used as panel titles)
TASK_LABELS = {
    "two_moons": "Two Moons",
    "bernoulli_glm": "Bernoulli GLM",
    "gaussian_linear": "Gaussian Linear",
    "gaussian_mixture": "Gaussian Mixture",
    "slcp": "SLCP",
}

METHODS = [
    "flow_flux",
    "flow_flux1joint",
    "diffusion_flux",
    "diffusion_flux1joint",
    "score_matching_flux",
    "score_matching_flux1joint",
]

BUDGETS = [10_000, 30_000, 100_000]

EXPERIMENT_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# EXPERIMENT_IDS = [1, 2, 4, 5, 6, 8]

STATS_DIR = (
    "/lhome/ific/a/aamerio/data/github/GenSBI-examples/examples/sbi-benchmarks/stats"
)

# ---------- group methods by model ----------

FLUX_METHODS = ["flow_flux", "diffusion_flux", "score_matching_flux"]
FLUX1JOINT_METHODS = [
    "flow_flux1joint",
    "diffusion_flux1joint",
    "score_matching_flux1joint",
]

# Pretty labels for legend
METHOD_LABELS = {
    "flow_flux": "Flow Matching",
    "flow_flux1joint": "Flow Matching",
    "diffusion_flux": "Diffusion (EDM)",
    "diffusion_flux1joint": "Diffusion (EDM)",
    "score_matching_flux": "Score Matching",
    "score_matching_flux1joint": "Score Matching",
}

METHOD_CSV_LABELS = {
    "flow_flux": "flow matching",
    "flow_flux1joint": "flow matching",
    "diffusion_flux": "edm",
    "diffusion_flux1joint": "edm",
    "score_matching_flux": "score matching",
    "score_matching_flux1joint": "score matching",
}

# Colors for each training method
METHOD_COLORS = {
    "flow_flux": "#1f77b4",
    "flow_flux1joint": "#1f77b4",
    "diffusion_flux": "#ff7f0e",
    "diffusion_flux1joint": "#ff7f0e",
    "score_matching_flux": "#2ca02c",
    "score_matching_flux1joint": "#2ca02c",
}

# Marker for each experiment id
EXPERIMENT_MARKERS = {1: "x", 2: "o", 3: "*", 4: "s", 5: "d", 6: "p", 7: "h", 8: "H", 9: "v"}

# ---------- load data ----------
# %%


def load_all_data():
    """Load all CSV files into a dict keyed by (task, experiment_id)."""
    data = {}
    for task in TASKS:
        for exp_id in EXPERIMENT_IDS:
            csv_path = f"{STATS_DIR}/{task}_experiment_{exp_id}.csv"
            try:
                df = pd.read_csv(csv_path)
                data[(task, exp_id)] = df
            except FileNotFoundError:
                print(f"WARNING: {csv_path} not found, skipping.")
    return data


data = load_all_data()

# ---------- plotting ----------
# %%


def plot_c2st_vs_budget_best(model_methods, model_name, data, with_markers=False):
    """
    Create a 1×5 figure with one panel per task.
    Each panel plots the best (minimum) C2ST vs budget for the given methods.

    At each budget, the marker indicates which experiment gave the best result:
      - 'x' for experiment 1
      - 'o' for experiment 2
    """
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)

    # Track which experiment IDs actually appear as best
    used_exp_ids = set()
    csv_data = []

    for ax, task in zip(axes, TASKS):
        for method in model_methods:
            label = METHOD_LABELS[method]
            color = METHOD_COLORS[method]

            best_vals = []
            best_exp_ids = []

            for i_budget in range(len(BUDGETS)):
                # Find the experiment with the smallest C2ST at this budget
                min_val = np.inf
                min_err = np.nan
                min_exp = None
                for exp_id in EXPERIMENT_IDS:
                    key = (task, exp_id)
                    if key not in data:
                        continue
                    val = max(float(data[key][method].values[i_budget]), 0.5)
                    if val < min_val:
                        min_val = val
                        min_exp = exp_id
                        if f"{method}_err" in data[key].columns:
                            min_err = float(data[key][f"{method}_err"].values[i_budget])
                        else:
                            min_err = np.nan

                best_vals.append(min_val)
                best_exp_ids.append(min_exp)
                if min_exp is not None:
                    used_exp_ids.add(min_exp)
                csv_data.append({
                    "task": task,
                    "model": model_name,
                    "method": METHOD_CSV_LABELS[method],
                    "budget": BUDGETS[i_budget],
                    "best_c2st": min_val,
                    "best_c2st_err": min_err,
                    "best_exp_id": min_exp
                })

            # Draw the connecting line (no markers)
            ax.plot(
                BUDGETS,
                best_vals,
                color=color,
                linewidth=1.5,
                label=label,
            )

            if with_markers:
                # Draw individual markers based on which experiment was best
                for budget, val, exp_id in zip(BUDGETS, best_vals, best_exp_ids):
                    ax.plot(
                        budget,
                        val,
                        marker=EXPERIMENT_MARKERS[exp_id],
                        color=color,
                        markersize=10,
                        markeredgewidth=2,
                    )

        ax.set_title(TASK_LABELS[task], fontsize=18)
        ax.set_xlabel("Simulation Budget")
        ax.set_xscale("log")
        ax.set_xticks(BUDGETS)
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_xticklabels([f"{b // 1000}k" for b in BUDGETS])
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_ylim(0.45, 1.0)
        ax.set_xlim(10_000, 100_000)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.grid(lw=1.0)

    axes[0].set_ylabel("C2ST")

    # Print experiments that are never the best
    unused_exp_ids = set(EXPERIMENT_IDS) - used_exp_ids
    if unused_exp_ids:
        print(f"[{model_name}] Experiments never best in any task/budget: {sorted(unused_exp_ids)}")

    # Single legend from first panel (all panels have the same lines)
    handles, labels = axes[0].get_legend_handles_labels()

    if with_markers:
        # Add marker legend entries only for experiments that were actually best
        for exp_id, marker in EXPERIMENT_MARKERS.items():
            if exp_id not in used_exp_ids:
                continue
            h = plt.Line2D(
                [],
                [],
                color="gray",
                marker=marker,
                linestyle="None",
                markersize=7,
                markeredgewidth=2,
                label=f"Experiment {exp_id}",
            )
            handles.append(h)
            labels.append(f"Experiment {exp_id}")

        fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.15), framealpha=0)
        fig.suptitle(f"C2ST vs Budget — {model_name}", y=1.25, fontsize=20)
    else:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=3,
            bbox_to_anchor=(0.5, 1.05),
            framealpha=0,
            fontsize=18,
        )
        fig.suptitle(f"C2ST vs Budget — {model_name}", y=1.15, fontsize=20)
        
    fig.tight_layout()
    
    df_long = pd.DataFrame(csv_data)
    df_best = df_long.pivot(index=["task", "model", "method"], columns="budget", values=["best_c2st", "best_c2st_err"])
    df_best.columns = [str(val) if col == "best_c2st" else f"{val}_err" for col, val in df_best.columns]
    df_best = df_best.reset_index()
    
    return fig, df_best


fig_flux_paper, df_flux_best = plot_c2st_vs_budget_best(FLUX_METHODS, "Flux1", data, with_markers=False)
fig_flux_paper.savefig(
    f"{STATS_DIR}/c2st_vs_budget_flux1_best.png", dpi=150, bbox_inches="tight"
)
fig_flux_paper.savefig(
    f"{STATS_DIR}/c2st_vs_budget_flux1_best.pdf", dpi=150, bbox_inches="tight"
)
df_flux_best.to_csv(f"{STATS_DIR}/best_c2st_flux1.csv", index=False)

fig_flux_comp, _ = plot_c2st_vs_budget_best(FLUX_METHODS, "Flux1", data, with_markers=True)
fig_flux_comp.savefig(
    f"{STATS_DIR}/c2st_vs_budget_flux1_best_comparison.png", dpi=150, bbox_inches="tight"
)
fig_flux_comp.savefig(
    f"{STATS_DIR}/c2st_vs_budget_flux1_best_comparison.pdf", dpi=150, bbox_inches="tight"
)

fig_flux1joint_paper, df_flux1joint_best = plot_c2st_vs_budget_best(FLUX1JOINT_METHODS, "Flux1Joint", data, with_markers=False)
fig_flux1joint_paper.savefig(
    f"{STATS_DIR}/c2st_vs_budget_flux1joint_best.png", dpi=150, bbox_inches="tight"
)
fig_flux1joint_paper.savefig(
    f"{STATS_DIR}/c2st_vs_budget_flux1joint_best.pdf", dpi=150, bbox_inches="tight"
)
df_flux1joint_best.to_csv(f"{STATS_DIR}/best_c2st_flux1joint.csv", index=False)

fig_flux1joint_comp, _ = plot_c2st_vs_budget_best(FLUX1JOINT_METHODS, "Flux1Joint", data, with_markers=True)
fig_flux1joint_comp.savefig(
    f"{STATS_DIR}/c2st_vs_budget_flux1joint_best_comparison.png", dpi=150, bbox_inches="tight"
)
fig_flux1joint_comp.savefig(
    f"{STATS_DIR}/c2st_vs_budget_flux1joint_best_comparison.pdf", dpi=150, bbox_inches="tight"
)

plt.show()
print("Done! Plots saved to:", STATS_DIR)
