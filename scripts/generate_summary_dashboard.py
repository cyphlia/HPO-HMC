"""
A single "at a glance" dashboard figure summarizing every experiment in the
project. Useful as a closing figure or graphical-abstract-style summary for
the paper or a presentation. All numbers are taken directly from the
project's own verified result files / paper tables (see comments per panel);
nothing here is a new computation.

Usage:
    python scripts/generate_summary_dashboard.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def main():
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

    # Panel 1: Harmonic oscillator MSE (log scale) -- from HO_main.tex Table (results)
    ax1 = fig.add_subplot(gs[0, 0])
    methods_ho = ["A: HHD", "B: ABBO", "C: Unified"]
    mse_ho = [0.2439, 0.0952, 0.0033]
    mse_ho_err = [0.1627, 0.0051, 0.0001]
    colors_ho = ["#4C72B0", "#DD8452", "#55A868"]
    ax1.bar(methods_ho, mse_ho, yerr=mse_ho_err, color=colors_ho, capsize=4)
    ax1.set_yscale("log")
    ax1.set_ylabel("Best Val. MSE (log scale)")
    ax1.set_title("1. Harmonic Oscillator\n(synthetic, ground truth known)")

    # Panel 2: Tabular/NAS average rank -- from HO_main.tex Table 14
    ax2 = fig.add_subplot(gs[0, 1])
    methods_tab = ["Rand", "TPE", "A", "B", "C"]
    ranks = [3.64, 1.36, 4.09, 2.82, 3.09]
    colors_tab = ["#8C8C8C", "#F0B429", "#4C72B0", "#DD8452", "#55A868"]
    bars = ax2.bar(methods_tab, ranks, color=colors_tab)
    ax2.axhline(1.36 + 1.84, color="red", linestyle="--", linewidth=1, label="Nemenyi CD boundary\nfrom TPE (1.84)")
    ax2.set_ylabel("Average rank (lower = better)")
    ax2.set_title("2. 11 Tabular/NAS Benchmarks\n(Friedman p=7.9e-4)")
    ax2.legend(fontsize=7)
    ax2.invert_yaxis()

    # Panel 3: Fashion-MNIST + CIFAR-10 -- from HO_main.tex Tables 16-17
    ax3 = fig.add_subplot(gs[0, 2])
    x = np.arange(2)
    width = 0.35
    baseline_acc = [84.43, 30.90]
    methodc_acc = [85.01, 30.60]
    baseline_err = [0.41, 1.59]
    methodc_err = [0.12, 2.65]
    ax3.bar(x - width/2, baseline_acc, width, yerr=baseline_err, label="Baseline", color="#8C8C8C", capsize=4)
    ax3.bar(x + width/2, methodc_acc, width, yerr=methodc_err, label="Method C", color="#55A868", capsize=4)
    ax3.set_xticks(x); ax3.set_xticklabels(["Fashion-MNIST\n(vs fixed Adam)", "CIFAR-10\n(vs Method A)"])
    ax3.set_ylabel("Accuracy (%)")
    ax3.set_title("3. Image Classification")
    ax3.legend(fontsize=8)

    # Panel 4: Real-world AUROC + speed -- from results/{breast_cancer,diabetes}/summary.json
    ax4 = fig.add_subplot(gs[1, 0])
    datasets = ["Breast\nCancer", "Diabetes"]
    optuna_auroc = [0.9963, 0.8212]
    methodc_auroc = [0.9949, 0.8224]
    x = np.arange(2)
    ax4.bar(x - width/2, optuna_auroc, width, label="Optuna TPE (20 trials)", color="#F0B429")
    ax4.bar(x + width/2, methodc_auroc, width, label="Method C", color="#55A868")
    ax4.set_xticks(x); ax4.set_xticklabels(datasets)
    ax4.set_ylabel("Test AUROC")
    ax4.set_ylim(0.75, 1.02)
    ax4.set_title("4. Real Clinical Data\n(quality: statistically tied, p>0.1)")
    ax4.legend(fontsize=7)

    ax4b = ax4.twinx()
    speedup = [22.2/1.6, 29.0/2.1]
    ax4b.plot(x, speedup, color="#C44E52", marker="D", linewidth=0, markersize=8)
    for xi, s in zip(x, speedup):
        ax4b.annotate(f"{s:.0f}x faster", (xi, s), textcoords="offset points",
                     xytext=(15, 0), fontsize=8, color="#C44E52")
    ax4b.set_ylabel("Method C speedup vs. Optuna", color="#C44E52")
    ax4b.set_ylim(0, 20)
    ax4b.tick_params(axis="y", labelcolor="#C44E52")

    # Panel 5: Validation-overfitting gap -- from Notebook 03 / earlier analysis
    ax5 = fig.add_subplot(gs[1, 1])
    methods_gap = ["Default\nAdam", "Method C", "Optuna\nTPE", "Random\nSearch"]
    gaps = [-0.003, 0.008, 0.025, 0.032]
    colors_gap = ["#8C8C8C", "#55A868", "#F0B429", "#C44E52"]
    ax5.bar(methods_gap, gaps, color=colors_gap)
    ax5.axhline(0, color="black", linewidth=0.8)
    ax5.set_ylabel("Validation - Test AUROC gap\n(Diabetes; higher = more overfit to val)")
    ax5.set_title("5. The Optimizer's Curse\n(more search -> more val-overfitting)")

    # Panel 6: NUTS vs Leapfrog -- from results/nuts_comparison/summary.json
    ax6 = fig.add_subplot(gs[1, 2])
    methods_nuts = ["Fixed-step\nLeapfrog", "NUTS"]
    mse_nuts = [0.206, 0.304]
    mse_nuts_err = [0.146, 0.107]
    time_nuts = [19.8, 109.1]
    colors_nuts = ["#4C72B0", "#DD8452"]
    ax6.bar(methods_nuts, mse_nuts, yerr=mse_nuts_err, color=colors_nuts, capsize=4)
    ax6.set_ylabel("Best Val. MSE")
    ax6.set_title("6. NUTS vs. Leapfrog\n(leapfrog: 5.5x cheaper, p=0.125)")
    for i, t in enumerate(time_nuts):
        ax6.annotate(f"{t:.0f}s", (i, mse_nuts[i] + mse_nuts_err[i] + 0.01),
                    ha="center", fontsize=8, color="#333333")

    fig.suptitle("Hamiltonian Hyperparameter Dynamics — Project Results at a Glance",
                fontsize=15, fontweight="bold", y=1.0)
    plt.savefig(os.path.join(PLOTS_DIR, "project_summary_dashboard.png"), dpi=160, bbox_inches="tight")
    plt.close()
    print("Saved plots/project_summary_dashboard.png")


if __name__ == "__main__":
    main()
