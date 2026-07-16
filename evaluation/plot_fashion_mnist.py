"""
plot_fashion_mnist.py — Generate publication-quality plots for Fashion-MNIST testbed.

Generates 4 figures:
  1. Accuracy Comparison Bar Chart (mean ± std across seeds)
  2. Convergence Curves (val accuracy vs. epoch per method)
  3. HP Trajectory Plot (Method C hyperparameter evolution)
  4. Wall-Time vs. Accuracy Scatter (Pareto front)

Usage:
    python evaluation/plot_fashion_mnist.py
"""

import json
import os
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "results", "fashion_mnist")
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
})

COLORS = {
    "default":  "#78909C",  # Blue Grey
    "random":   "#FF7043",  # Deep Orange
    "optuna":   "#42A5F5",  # Blue
    "methodC":  "#66BB6A",  # Green
}
LABELS = {
    "default":  "Default Adam",
    "random":   "Random Search",
    "optuna":   "Optuna TPE",
    "methodC":  "Method C (HHD-ABBO)",
}


def load_results():
    """Load testbed results from JSON."""
    summary_path = os.path.join(RESULTS_DIR, "testbed_summary.json")
    allruns_path = os.path.join(RESULTS_DIR, "all_runs.json")

    if not os.path.exists(summary_path):
        print(f"  [ERROR] {summary_path} not found. Run fashion_mnist_testbed.py first.")
        return None, None

    with open(summary_path) as f:
        summary = json.load(f)
    with open(allruns_path) as f:
        all_runs = json.load(f)
    return summary, all_runs


# ═══════════════════════════════════════════════════════════════════════════
#  PLOT 1: Accuracy Comparison Bar Chart
# ═══════════════════════════════════════════════════════════════════════════

def plot_accuracy_comparison(summary):
    """Bar chart: mean ± std best validation accuracy per method."""
    print("  Plotting 1/4: Accuracy Comparison Bar Chart...")

    methods = [k for k in ["default", "random", "optuna", "methodC"] if k in summary]
    means = [summary[m]["mean_best_acc"] * 100 for m in methods]
    stds  = [summary[m]["std_best_acc"] * 100 for m in methods]
    colors = [COLORS[m] for m in methods]
    labels = [LABELS[m] for m in methods]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, means, yerr=stds, color=colors, alpha=0.88,
                  edgecolor="white", lw=1.5, capsize=6,
                  error_kw={"linewidth": 1.5, "capthick": 1.5})

    # Annotate bars
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.3,
                f"{mean:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)

    ax.set_ylabel("Best Validation Accuracy (%)", fontsize=12)
    ax.set_title("Fashion-MNIST: Method Comparison (5 seeds)", fontsize=14)
    ax.set_ylim(0, max(means) + max(stds) + 5)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = os.path.join(PLOTS_DIR, "fmnist_accuracy_comparison.png")
    plt.savefig(path)
    plt.close()
    print(f"    Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
#  PLOT 2: Convergence Curves
# ═══════════════════════════════════════════════════════════════════════════

def plot_convergence_curves(all_runs):
    """Validation accuracy vs. epoch for each method, with ±1σ bands."""
    print("  Plotting 2/4: Convergence Curves...")

    fig, ax = plt.subplots(figsize=(10, 6))

    for method_key in ["default", "random", "optuna", "methodC"]:
        if method_key not in all_runs or not all_runs[method_key]:
            continue
        runs = all_runs[method_key]
        # Collect all accuracy histories
        histories = [r["acc_history"] for r in runs if r.get("acc_history")]
        if not histories:
            continue

        # Pad to same length
        max_len = max(len(h) for h in histories)
        padded = []
        for h in histories:
            if len(h) < max_len:
                h = h + [h[-1]] * (max_len - len(h))
            padded.append(h[:max_len])

        arr = np.array(padded) * 100  # Convert to percentage
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)

        epochs = np.arange(len(mean))
        ax.plot(epochs, mean, color=COLORS[method_key], label=LABELS[method_key],
                linewidth=2)
        ax.fill_between(epochs, mean - std, mean + std,
                        color=COLORS[method_key], alpha=0.15)

    ax.set_xlabel("Epoch / Trial", fontsize=12)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=12)
    ax.set_title("Fashion-MNIST: Convergence Curves (mean ± 1σ)", fontsize=14)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = os.path.join(PLOTS_DIR, "fmnist_convergence_curves.png")
    plt.savefig(path)
    plt.close()
    print(f"    Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
#  PLOT 3: HP Trajectory (Method C only)
# ═══════════════════════════════════════════════════════════════════════════

def plot_hp_trajectory(all_runs):
    """Method C hyperparameter values over training epochs."""
    print("  Plotting 3/4: HP Trajectory (Method C)...")

    if "methodC" not in all_runs or not all_runs["methodC"]:
        print("    [SKIP] No Method C data found.")
        return

    # Use the first seed's trajectory as representative
    run = None
    for r in all_runs["methodC"]:
        if "hp_trajectory" in r and r["hp_trajectory"]:
            run = r
            break
    if run is None:
        print("    [SKIP] No HP trajectory data in Method C runs.")
        return

    traj = run["hp_trajectory"]
    acc_rate = run.get("acceptance_rate", [])

    hp_keys = ["log_lr", "dropout", "log_wd"]
    hp_labels = ["log₁₀(LR)", "Dropout", "log₁₀(WD)"]
    hp_colors = ["#E53935", "#43A047", "#1E88E5"]

    n_plots = len(hp_keys) + (1 if acc_rate else 0)
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    for i, (key, label, color) in enumerate(zip(hp_keys, hp_labels, hp_colors)):
        if key not in traj:
            continue
        vals = traj[key]
        epochs = np.arange(len(vals))
        axes[i].plot(epochs, vals, color=color, linewidth=1.8, label=label)
        axes[i].set_ylabel(label, fontsize=11)
        axes[i].grid(alpha=0.3, linestyle="--")
        axes[i].spines["top"].set_visible(False)
        axes[i].spines["right"].set_visible(False)

        # Mark phase boundaries
        n_warmup = 5  # from config
        if len(vals) > n_warmup:
            axes[i].axvline(n_warmup, color="gray", linestyle=":", alpha=0.6)
            axes[i].text(n_warmup + 0.3, axes[i].get_ylim()[1] * 0.95,
                         "← HMC starts", fontsize=8, color="gray", va="top")

    if acc_rate:
        ax_acc = axes[-1]
        epochs = np.arange(len(acc_rate))
        ax_acc.plot(epochs, [a * 100 for a in acc_rate],
                    color="#FF8F00", linewidth=1.8)
        ax_acc.set_ylabel("HMC Accept (%)", fontsize=11)
        ax_acc.set_ylim(-5, 105)
        ax_acc.grid(alpha=0.3, linestyle="--")
        ax_acc.spines["top"].set_visible(False)
        ax_acc.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Epoch", fontsize=12)
    fig.suptitle("Method C: Hyperparameter Trajectory on Fashion-MNIST",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "fmnist_hp_trajectory.png")
    plt.savefig(path)
    plt.close()
    print(f"    Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
#  PLOT 4: Wall-Time vs. Accuracy Scatter
# ═══════════════════════════════════════════════════════════════════════════

def plot_time_vs_accuracy(summary):
    """Scatter plot: wall time vs. accuracy for each method."""
    print("  Plotting 4/4: Time vs. Accuracy Scatter...")

    fig, ax = plt.subplots(figsize=(9, 6))

    for method_key in ["default", "random", "optuna", "methodC"]:
        if method_key not in summary:
            continue
        s = summary[method_key]
        ax.scatter(s["mean_time"], s["mean_best_acc"] * 100,
                   s=200, color=COLORS[method_key], edgecolor="white",
                   linewidth=1.5, zorder=5, label=LABELS[method_key])
        ax.annotate(LABELS[method_key],
                    (s["mean_time"], s["mean_best_acc"] * 100),
                    textcoords="offset points", xytext=(10, 5),
                    fontsize=9, color=COLORS[method_key], fontweight="bold")

    ax.set_xlabel("Mean Wall-Clock Time (s)", fontsize=12)
    ax.set_ylabel("Mean Best Accuracy (%)", fontsize=12)
    ax.set_title("Fashion-MNIST: Time vs. Accuracy Trade-off", fontsize=14)
    ax.grid(alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = os.path.join(PLOTS_DIR, "fmnist_time_vs_accuracy.png")
    plt.savefig(path)
    plt.close()
    print(f"    Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Fashion-MNIST Testbed — Plot Generation")
    print("=" * 60)

    summary, all_runs = load_results()
    if summary is None:
        return

    plot_accuracy_comparison(summary)
    plot_convergence_curves(all_runs)
    plot_hp_trajectory(all_runs)
    plot_time_vs_accuracy(summary)

    print("\n  All plots saved to plots/")
    print("=" * 60)


if __name__ == "__main__":
    main()
