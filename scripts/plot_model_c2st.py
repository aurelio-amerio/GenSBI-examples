# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

EXPERIMENT_ID = 6

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

# Colors for each training method
METHOD_COLORS = {
    "flow_flux": "#1f77b4",
    "flow_flux1joint": "#1f77b4",
    "diffusion_flux": "#ff7f0e",
    "diffusion_flux1joint": "#ff7f0e",
    "score_matching_flux": "#2ca02c",
    "score_matching_flux1joint": "#2ca02c",
}

# ---------- load data ----------
# %%


def load_all_data(experiment_id):
    """Load all CSV files for a given experiment_id into a dict keyed by task."""
    data = {}
    for task in TASKS:
        csv_path = f"{STATS_DIR}/{task}_experiment_{experiment_id}.csv"
        try:
            df = pd.read_csv(csv_path)
            data[task] = df
        except FileNotFoundError:
            print(f"WARNING: {csv_path} not found, skipping.")
    return data


data = load_all_data(EXPERIMENT_ID)

# ---------- plotting ----------
# %%


def plot_c2st_vs_budget(model_methods, model_name, data):
    """
    Create a 1×5 figure with one panel per task.
    Each panel plots C2ST vs budget for the given methods.
    """
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)

    for ax, task in zip(axes, TASKS):
        if task not in data:
            continue
        df = data[task]
        for method in model_methods:
            label = METHOD_LABELS[method]
            color = METHOD_COLORS[method]
            vals = np.maximum(df[method].values.astype(float), 0.5)

            ax.plot(
                BUDGETS,
                vals,
                marker="o",
                label=label,
                color=color,
                linewidth=1.5,
                markersize=5,
            )

        ax.set_title(TASK_LABELS[task], fontsize=12)
        ax.set_xlabel("Simulation Budget")
        ax.set_xscale("log")
        ax.set_xticks(BUDGETS)
        ax.set_xticklabels([f"{b // 1000}k" for b in BUDGETS])
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

    axes[0].set_ylabel("C2ST")
    # Single legend from first panel (all panels have the same lines)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle(f"C2ST vs Budget — {model_name}", fontsize=14, y=1.12)
    fig.tight_layout()
    return fig


fig_flux = plot_c2st_vs_budget(FLUX_METHODS, "Flux1", data)
fig_flux.savefig(
    f"{STATS_DIR}/c2st_vs_budget_flux1_{EXPERIMENT_ID}.png",
    dpi=150,
    bbox_inches="tight",
)
# fig_flux.savefig(f"{STATS_DIR}/c2st_vs_budget_flux1_{EXPERIMENT_ID}.pdf", bbox_inches="tight")

fig_flux1joint = plot_c2st_vs_budget(FLUX1JOINT_METHODS, "Flux1Joint", data)
fig_flux1joint.savefig(
    f"{STATS_DIR}/c2st_vs_budget_flux1joint_{EXPERIMENT_ID}.png",
    dpi=150,
    bbox_inches="tight",
)
# fig_flux1joint.savefig(
#     f"{STATS_DIR}/c2st_vs_budget_flux1joint_{EXPERIMENT_ID}.pdf", bbox_inches="tight"
# )

plt.show()
print("Done! Plots saved to:", STATS_DIR)
