"""
plot_all_results.py — Comprehensive plot generation from consolidated results.

Generates additional publication-quality plots beyond the existing ones:
  1. Ablation study bar chart
  2. Sensitivity analysis (epsilon)
  3. Physics benchmarks comparison (3 systems)
  4. Harmonic multiseed box plots
  5. CNN CIFAR-10 multi-seed comparison
  6. Wall-time vs quality scatter
  7. Optimizer benchmark comparison
  8. Statistical tests: CD diagram
  9. Testbed 3-way comparison

Usage:
    python evaluation/plot_all_results.py
"""

import os
import sys
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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
    "A": "#2196F3",
    "B": "#FF5722",
    "C": "#4CAF50",
}
METHOD_NAMES = {
    "A": "Method A (HHD)",
    "B": "Method B (ABBO)",
    "C": "Method C (Unified)",
}
ABLATION_COLORS = {
    "C-full": "#4CAF50",
    "C-noAdam": "#FF9800",
    "C-noHMC": "#9C27B0",
    "C-noLBFGS": "#F44336",
    "C-noPlateauDet": "#2196F3",
    "C-fixedStep": "#795548",
}

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_json(path):
    full_path = os.path.join(RESULTS_DIR, path) if not os.path.isabs(path) else path
    with open(full_path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════
# PLOT 1: Ablation Study — Mean Best Val Loss per Variant
# ═══════════════════════════════════════════════════════════════
def plot_ablation_study():
    print("Plotting 1/9: Ablation Study...")
    try:
        data = load_json("ablation/ablation_summary.json")
    except FileNotFoundError:
        print("  [SKIP] ablation_summary.json not found")
        return

    summary = data["summary"]
    variants = list(summary.keys())
    ho_means = [summary[v]["harmonic_oscillator"]["mean"] for v in variants]
    ho_stds = [summary[v]["harmonic_oscillator"]["std"] for v in variants]

    # Also try to get HPOBench ablation data from individual results
    colors = [ABLATION_COLORS.get(v, "#888") for v in variants]
    nice_names = [v.replace("C-", "C − ") for v in variants]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        nice_names, ho_means, yerr=ho_stds,
        color=colors, alpha=0.85, edgecolor="white", lw=1.5,
        capsize=5, error_kw={"linewidth": 1.5}
    )

    # Annotate
    for bar, mean, std in zip(bars, ho_means, ho_stds):
        y = bar.get_height() + std + max(ho_means) * 0.02
        ax.text(
            bar.get_x() + bar.get_width() / 2, y,
            f"{mean:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    ax.set_ylabel("Best Validation Loss (MSE)", fontweight="bold")
    ax.set_title("Ablation Study: Component Contribution to Method C\n(Harmonic Oscillator, 5 seeds)", pad=15)
    ax.set_yscale("log")
    ax.set_ylim(min(ho_means) * 0.3, max(ho_means) * 3.0)
    ax.grid(True, alpha=0.2, axis="y")

    # Highlight the worst (most important component)
    worst_idx = np.argmax(ho_means)
    bars[worst_idx].set_edgecolor("black")
    bars[worst_idx].set_linewidth(2.5)
    ax.annotate(
        "Most critical\ncomponent",
        xy=(worst_idx, ho_means[worst_idx]),
        xytext=(worst_idx + 0.5, ho_means[worst_idx] * 1.5),
        fontsize=9, fontweight="bold", color="#F44336",
        arrowprops=dict(arrowstyle="->", color="#F44336", lw=1.5),
    )

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "ablation_study.png"))
    plt.close()
    print("  Saved plots/ablation_study.png")


# ═══════════════════════════════════════════════════════════════
# PLOT 2: Sensitivity Analysis — Epsilon
# ═══════════════════════════════════════════════════════════════
def plot_sensitivity():
    print("Plotting 2/9: Sensitivity Analysis...")
    try:
        data = load_json("ablation/sensitivity_results.json")
    except FileNotFoundError:
        print("  [SKIP] sensitivity_results.json not found")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for param_name, param_data in data.items():
        values = []
        means = []
        stds = []
        for key, entry in sorted(param_data.items(), key=lambda x: float(x[0])):
            values.append(entry["value"])
            means.append(entry["mean"])
            stds.append(entry.get("std", 0.0))

        means = np.array(means)
        stds = np.array(stds)
        values = np.array(values)

        ax.plot(values, means, "o-", lw=2.5, ms=8, color=COLORS["C"],
                label=f"ε (step size)")
        ax.fill_between(values, means - stds, means + stds,
                         alpha=0.15, color=COLORS["C"])

        # Annotate best
        best_idx = np.argmin(means)
        ax.annotate(
            f"Best: ε={values[best_idx]:.3f}\nloss={means[best_idx]:.5f}",
            xy=(values[best_idx], means[best_idx]),
            xytext=(values[best_idx] * 1.5, means[best_idx] * 1.1),
            fontsize=9, fontweight="bold",
            arrowprops=dict(arrowstyle="->", lw=1.2),
        )

    ax.set_xlabel("Step Size (ε)", fontweight="bold")
    ax.set_ylabel("Best Validation Loss (MSE)", fontweight="bold")
    ax.set_title("Sensitivity Analysis: HMC Step Size\n(Harmonic Oscillator)", pad=15)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "sensitivity_epsilon.png"))
    plt.close()
    print("  Saved plots/sensitivity_epsilon.png")


# ═══════════════════════════════════════════════════════════════
# PLOT 3: Physics Benchmarks Comparison (3 systems × 3 methods)
# ═══════════════════════════════════════════════════════════════
def plot_physics_benchmarks():
    print("Plotting 3/9: Physics Benchmarks...")
    try:
        data = load_json("physics_benchmarks/physics_summary.json")
    except FileNotFoundError:
        print("  [SKIP] physics_summary.json not found")
        return

    summary = data["summary"]
    systems = list(summary.keys())
    nice_names = {
        "harmonic": "Harmonic\nOscillator",
        "henon_heiles": "Hénon-Heiles",
        "double_well": "Double-Well",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

    for idx, system in enumerate(systems):
        ax = axes[idx]
        methods = sorted(summary[system].keys())
        means = [summary[system][m]["mean"] for m in methods]
        colors = [COLORS[m] for m in methods]
        labels = [METHOD_NAMES[m] for m in methods]

        bars = ax.bar(labels, means, color=colors, alpha=0.85,
                      edgecolor="white", lw=1.5, width=0.5)

        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2, mean + max(means) * 0.03,
                f"{mean:.3f}" if mean >= 0.01 else f"{mean:.5f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold"
            )

        ax.set_title(nice_names.get(system, system), fontsize=13, fontweight="bold")
        ax.set_ylabel("Final Validation Loss" if idx == 0 else "")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.2, axis="y")
        ax.tick_params(axis="x", rotation=15, labelsize=9)

    fig.suptitle("Physics Benchmark Comparison: Methods A, B, C",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "physics_benchmarks_comparison.png"))
    plt.close()
    print("  Saved plots/physics_benchmarks_comparison.png")


# ═══════════════════════════════════════════════════════════════
# PLOT 4: Harmonic Multiseed Box Plots
# ═══════════════════════════════════════════════════════════════
def plot_harmonic_multiseed():
    print("Plotting 4/9: Harmonic Multiseed Box Plots...")
    try:
        data = load_json("harmonic_multiseed/physics_multiseed_summary.json")
    except FileNotFoundError:
        print("  [SKIP] physics_multiseed_summary.json not found")
        return

    metrics = ["best_val_loss", "mae", "rmse", "r2"]
    metric_labels = ["Best Val Loss", "MAE", "RMSE", "R²"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        box_data = []
        labels = []
        colors_list = []

        for method in ["A", "B", "C"]:
            if method not in data:
                continue
            raw = data[method]["raw_results"]
            vals = [r[metric] for r in raw]
            box_data.append(vals)
            labels.append(METHOD_NAMES[method])
            colors_list.append(COLORS[method])

        bp = ax.boxplot(
            box_data, tick_labels=labels, patch_artist=True,
            medianprops=dict(color="black", lw=2),
            whiskerprops=dict(lw=1.5),
            capprops=dict(lw=1.5),
        )

        for patch, color in zip(bp["boxes"], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        # Scatter individual points
        for i, (vals, color) in enumerate(zip(box_data, colors_list)):
            x = np.random.normal(i + 1, 0.04, size=len(vals))
            ax.scatter(x, vals, color=color, s=40, zorder=5, edgecolors="white", lw=0.5)

        ax.set_title(label, fontweight="bold")
        ax.grid(True, alpha=0.2, axis="y")

        if metric in ["best_val_loss", "mae", "rmse"]:
            ax.set_yscale("log")

    fig.suptitle("Harmonic Oscillator: Multi-Seed Performance (5 seeds)",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "harmonic_multiseed_boxplot.png"))
    plt.close()
    print("  Saved plots/harmonic_multiseed_boxplot.png")


# ═══════════════════════════════════════════════════════════════
# PLOT 5: CNN CIFAR-10 Multi-seed Comparison
# ═══════════════════════════════════════════════════════════════
def plot_cnn_multiseed():
    print("Plotting 5/9: CNN CIFAR-10 Multi-seed...")
    try:
        data = load_json("cnn/cifar10_summary.json")
    except FileNotFoundError:
        print("  [SKIP] cifar10_summary.json not found")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Box plot of best_val_acc
    box_data = []
    labels = []
    colors_list = []
    for method in ["A", "B", "C"]:
        if method not in data:
            continue
        runs = data[method]["all_runs"]
        accs = [r["best_val_acc"] * 100 for r in runs]
        box_data.append(accs)
        labels.append(METHOD_NAMES[method])
        colors_list.append(COLORS[method])

    bp = ax1.boxplot(
        box_data, tick_labels=labels, patch_artist=True,
        medianprops=dict(color="black", lw=2),
        whiskerprops=dict(lw=1.5),
    )
    for patch, color in zip(bp["boxes"], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    for i, (vals, color) in enumerate(zip(box_data, colors_list)):
        x = np.random.normal(i + 1, 0.04, size=len(vals))
        ax1.scatter(x, vals, color=color, s=50, zorder=5, edgecolors="white", lw=0.5)

    ax1.set_ylabel("Best Validation Accuracy (%)", fontweight="bold")
    ax1.set_title("Best Accuracy Distribution (5 seeds)")
    ax1.grid(True, alpha=0.2, axis="y")

    # Right: Accuracy progression (mean ± std across seeds)
    for method in ["A", "B", "C"]:
        if method not in data:
            continue
        runs = data[method]["all_runs"]
        histories = [r["acc_history"] for r in runs]
        min_len = min(len(h) for h in histories)
        aligned = np.array([h[:min_len] for h in histories]) * 100

        mean = np.mean(aligned, axis=0)
        std = np.std(aligned, axis=0)
        x = np.arange(1, min_len + 1)

        ax2.plot(x, mean, color=COLORS[method], lw=2.5,
                 label=METHOD_NAMES[method], marker="o", ms=5)
        ax2.fill_between(x, mean - std, mean + std,
                          color=COLORS[method], alpha=0.15)

    ax2.set_xlabel("Epoch / Trial", fontweight="bold")
    ax2.set_ylabel("Validation Accuracy (%)", fontweight="bold")
    ax2.set_title("Mean Accuracy Progression (±1σ)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)

    fig.suptitle("CNN CIFAR-10 Benchmark: Multi-Seed Results",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cnn_cifar10_multiseed.png"))
    plt.close()
    print("  Saved plots/cnn_cifar10_multiseed.png")


# ═══════════════════════════════════════════════════════════════
# PLOT 6: Wall-Time vs Quality Scatter (Pareto)
# ═══════════════════════════════════════════════════════════════
def plot_time_vs_quality():
    print("Plotting 6/9: Wall-Time vs Quality...")

    points = []  # (time, quality, method, label)

    # Testbed results
    try:
        testbed = load_json("testbed/testbed_results.json")
        for m in ["A", "B", "C"]:
            if m in testbed:
                points.append((testbed[m]["time"], testbed[m]["val_loss"],
                                m, f"Testbed-{m}"))
    except FileNotFoundError:
        pass

    # Physics benchmarks
    try:
        physics = load_json("physics_benchmarks/physics_summary.json")
        for entry in physics["all_results"]:
            points.append((entry["wall_time_s"], entry["best_val_loss"],
                            entry["method"], f"Phys-{entry['system']}-{entry['method']}"))
    except FileNotFoundError:
        pass

    # Multiseed harmonic
    try:
        multi = load_json("harmonic_multiseed/physics_multiseed_summary.json")
        for m in ["A", "B", "C"]:
            if m in multi:
                points.append((multi[m]["mean_time"], multi[m]["mean_best_val"],
                                m, f"HO-multiseed-{m}"))
    except FileNotFoundError:
        pass

    if not points:
        print("  [SKIP] No data for time vs quality plot")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    for time_val, quality, method, label in points:
        ax.scatter(time_val, quality, color=COLORS.get(method, "#888"),
                   s=100, edgecolors="white", lw=1, zorder=5, alpha=0.8)
        ax.annotate(label, (time_val, quality), fontsize=7,
                    xytext=(5, 5), textcoords="offset points", alpha=0.7)

    # Create legend
    for m, name in METHOD_NAMES.items():
        ax.scatter([], [], color=COLORS[m], s=80, label=name)

    ax.set_xlabel("Wall-Clock Time (seconds)", fontweight="bold")
    ax.set_ylabel("Best Validation Loss", fontweight="bold")
    ax.set_title("Wall-Time vs. Quality: All Experiments", pad=15)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "time_vs_quality.png"))
    plt.close()
    print("  Saved plots/time_vs_quality.png")


# ═══════════════════════════════════════════════════════════════
# PLOT 7: Optimizer Benchmark (vs SGD / Adam / AdamW)
# ═══════════════════════════════════════════════════════════════
def plot_optimizer_benchmark():
    print("Plotting 7/9: Optimizer Benchmark...")
    try:
        data = load_json("validation/validation_results.json")
    except FileNotFoundError:
        print("  [SKIP] validation_results.json not found")
        return

    bench = data.get("optimizer_benchmark", {})
    if not bench:
        print("  [SKIP] No optimizer_benchmark data")
        return

    names = list(bench.keys())
    best_vals = [bench[n]["best_val"] for n in names]

    # Color coding: standard optimizers in gray, HHD-ABBO highlighted
    bar_colors = []
    for n in names:
        if "HHD" in n or "ABBO" in n or "Method" in n:
            bar_colors.append(COLORS["C"])
        elif "Adam" in n:
            bar_colors.append("#FF9800")
        else:
            bar_colors.append("#9E9E9E")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(names))
    bars = ax.bar(x, best_vals, color=bar_colors, alpha=0.85,
                  edgecolor="white", lw=1.5)

    for bar, val in zip(bars, best_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + max(best_vals) * 0.02,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Best Validation Loss", fontweight="bold")
    ax.set_title("Optimizer Comparison on Harmonic Oscillator\n(Fixed Architecture, Varying Optimizer)", pad=15)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "optimizer_benchmark.png"))
    plt.close()
    print("  Saved plots/optimizer_benchmark.png")


# ═══════════════════════════════════════════════════════════════
# PLOT 8: Statistical Tests — Average Ranks + Pairwise Significance
# ═══════════════════════════════════════════════════════════════
def plot_statistical_tests():
    print("Plotting 8/9: Statistical Tests...")
    try:
        data = load_json("hpobench/statistical_tests.json")
    except FileNotFoundError:
        print("  [SKIP] statistical_tests.json not found")
        return

    avg_ranks = data.get("average_ranks", {})
    cd = data.get("nemenyi_test", {}).get("critical_difference", None)
    friedman_p = data.get("friedman_test", {}).get("p_value", None)

    if not avg_ranks:
        print("  [SKIP] No average_ranks data")
        return

    optimizers = list(avg_ranks.keys())
    ranks = [avg_ranks[o] for o in optimizers]

    # Sort by rank
    sorted_pairs = sorted(zip(optimizers, ranks), key=lambda x: x[1])
    optimizers = [p[0] for p in sorted_pairs]
    ranks = [p[1] for p in sorted_pairs]

    opt_colors = {
        "Random Search": "#9E9E9E",
        "Optuna TPE": "#FF9800",
        "Method A (HHD)": COLORS["A"],
        "Method B (ABBO)": COLORS["B"],
        "Method C (Unified)": COLORS["C"],
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6),
                                     gridspec_kw={"width_ratios": [1.2, 1]})

    # Left: Average Ranks (horizontal bar)
    colors = [opt_colors.get(o, "#888") for o in optimizers]
    bars = ax1.barh(optimizers, ranks, color=colors, alpha=0.85,
                    edgecolor="white", lw=1.5, height=0.5)

    for bar, rank in zip(bars, ranks):
        ax1.text(rank + 0.05, bar.get_y() + bar.get_height() / 2,
                 f"{rank:.2f}", va="center", fontsize=10, fontweight="bold")

    if cd is not None:
        ax1.axvline(x=cd, color="red", ls="--", alpha=0.5, lw=1.5,
                    label=f"CD = {cd:.2f}")
        ax1.legend(fontsize=10)

    ax1.set_xlabel("Average Rank (lower = better)", fontweight="bold")
    ax1.set_title(f"Friedman-Nemenyi Test\n(p = {friedman_p:.5f})" if friedman_p else
                  "Average Ranks Across Benchmarks")
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis="x")

    # Right: Pairwise significance matrix
    pairwise = data.get("pairwise_significance", {})
    n_opt = len(optimizers)
    sig_matrix = np.zeros((n_opt, n_opt))

    for i, o1 in enumerate(optimizers):
        for j, o2 in enumerate(optimizers):
            if i == j:
                continue
            key1 = f"{o1} vs {o2}"
            key2 = f"{o2} vs {o1}"
            entry = pairwise.get(key1, pairwise.get(key2, {}))
            if entry.get("significant", False):
                sig_matrix[i, j] = 1
            else:
                sig_matrix[i, j] = 0.5

    im = ax2.imshow(sig_matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="equal")

    ax2.set_xticks(range(n_opt))
    ax2.set_yticks(range(n_opt))
    short_names = [o.split("(")[0].strip() if "(" in o else o[:12] for o in optimizers]
    ax2.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax2.set_yticklabels(short_names, fontsize=8)
    ax2.set_title("Pairwise Significance\n(green=significant, yellow=not)")

    for i in range(n_opt):
        for j in range(n_opt):
            if i != j:
                text = "✓" if sig_matrix[i, j] == 1 else "—"
                color = "white" if sig_matrix[i, j] == 1 else "black"
                ax2.text(j, i, text, ha="center", va="center",
                         fontsize=12, fontweight="bold", color=color)

    fig.suptitle("Statistical Analysis: Cross-Benchmark Optimizer Comparison\n(11 datasets, Friedman + Nemenyi post-hoc)",
                 fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cd_diagram_statistical.png"))
    plt.close()
    print("  Saved plots/cd_diagram_statistical.png")


# ═══════════════════════════════════════════════════════════════
# PLOT 9: Testbed 3-Way Comparison
# ═══════════════════════════════════════════════════════════════
def plot_testbed_comparison():
    print("Plotting 9/9: Testbed Comparison...")
    try:
        data = load_json("testbed/testbed_results.json")
    except FileNotFoundError:
        print("  [SKIP] testbed_results.json not found")
        return

    metrics = ["val_loss", "mae", "rmse", "r2", "time"]
    metric_labels = ["Val Loss", "MAE", "RMSE", "R²", "Time (s)"]
    methods = ["A", "B", "C"]

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        vals = [data[m].get(metric, 0) for m in methods]
        colors_list = [COLORS[m] for m in methods]
        labels = [METHOD_NAMES[m] for m in methods]

        bars = ax.bar(["A", "B", "C"], vals, color=colors_list, alpha=0.85,
                      edgecolor="white", lw=1.5)

        for bar, val in zip(bars, vals):
            fmt = f"{val:.4f}" if val < 1 else f"{val:.1f}"
            ax.text(bar.get_x() + bar.get_width() / 2, val + max(vals) * 0.02,
                    fmt, ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_title(label, fontweight="bold")
        ax.grid(True, alpha=0.2, axis="y")

        if metric in ["val_loss", "mae", "rmse"]:
            # Highlight the best (lowest)
            best_idx = np.argmin(vals)
            bars[best_idx].set_edgecolor("gold")
            bars[best_idx].set_linewidth(2.5)
        elif metric == "r2":
            best_idx = np.argmax(vals)
            bars[best_idx].set_edgecolor("gold")
            bars[best_idx].set_linewidth(2.5)

    fig.suptitle("Performance Testbed: Extended Harmonic Oscillator Comparison",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "testbed_comparison.png"))
    plt.close()
    print("  Saved plots/testbed_comparison.png")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  Generating Comprehensive Plots from Consolidated Results")
    print("=" * 60)
    print(f"  Results dir: {RESULTS_DIR}")
    print(f"  Plots dir:   {PLOTS_DIR}\n")

    plot_ablation_study()
    plot_sensitivity()
    plot_physics_benchmarks()
    plot_harmonic_multiseed()
    plot_cnn_multiseed()
    plot_time_vs_quality()
    plot_optimizer_benchmark()
    plot_statistical_tests()
    plot_testbed_comparison()

    print(f"\n{'=' * 60}")
    print(f"  All plots saved to {PLOTS_DIR}/")
    print(f"  Total plots: {len(os.listdir(PLOTS_DIR))}")
    print(f"{'=' * 60}")
    for f in sorted(os.listdir(PLOTS_DIR)):
        size_kb = os.path.getsize(os.path.join(PLOTS_DIR, f)) / 1024
        print(f"  {f:<45s} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
