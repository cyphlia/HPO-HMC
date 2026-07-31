"""
Generates a conceptual overview diagram of Method C's three-phase curriculum
and the augmented phase-space idea -- a schematic (not data-driven) figure,
useful as a paper's Figure 1 / graphical abstract. This was missing from the
repo; every existing figure is either a raw Hamiltonian testbed plot or a
results plot, not a pipeline/concept diagram.

Usage:
    python scripts/generate_pipeline_diagram.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.path import Path
import matplotlib.patches as mpatches

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def box(ax, xy, w, h, text, facecolor, fontsize=10, textcolor="black"):
    b = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                       linewidth=1.5, edgecolor="#333333", facecolor=facecolor, zorder=2)
    ax.add_patch(b)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
           fontsize=fontsize, color=textcolor, zorder=3, wrap=True)
    return b


def arrow(ax, start, end, text=None, color="#333333"):
    a = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18,
                        linewidth=1.8, color=color, zorder=2)
    ax.add_patch(a)
    if text:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mx, my + 0.025, text, ha="center", va="bottom", fontsize=8, color=color)


def main():
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    # ---- Top row: the three phases of Method C ----
    phase_y, phase_h = 4.6, 1.3
    box(ax, (0.4, phase_y), 3.4, phase_h,
       "Phase 1: Adam Warmup\n\nFast first-order descent,\nno curvature, no momentum\nfor hyperparameters",
       facecolor="#AED4E8", fontsize=9.5)
    arrow(ax, (3.8, phase_y + phase_h / 2), (4.6, phase_y + phase_h / 2))

    box(ax, (4.6, phase_y), 3.8, phase_h,
       "Phase 2: HMC Co-evolution\n\nWeights θ AND hyperparameters λ\nco-evolve via leapfrog integration\n+ Metropolis-Hastings correction",
       facecolor="#F5C687", fontsize=9.5)
    arrow(ax, (8.4, phase_y + phase_h / 2), (9.2, phase_y + phase_h / 2))

    box(ax, (9.2, phase_y), 3.4, phase_h,
       "Phase 3: L-BFGS Polish\n\nPlateau-triggered curvature-aware\nrefinement using the Hessian's\nsecond-order information",
       facecolor="#B8DFB8", fontsize=9.5)

    ax.text(6.5, phase_y + phase_h + 0.25,
           "Method C — Unified HHD-ABBO: one training run, no outer retraining loop",
           ha="center", fontsize=13, fontweight="bold")

    # ---- Loop-back arrow showing Phase 2 <-> Phase 3 alternation ----
    loop = FancyArrowPatch((8.4, phase_y - 0.15), (6.5, phase_y - 0.15),
                           connectionstyle="arc3,rad=-0.5", arrowstyle="-|>",
                           mutation_scale=15, linewidth=1.3, color="#888888", linestyle="--")
    ax.add_patch(loop)
    ax.text(7.4, phase_y - 0.75, "repeats until convergence\n(plateau-triggered)",
           ha="center", fontsize=7.5, color="#666666")

    # ---- Bottom-left: the augmented phase space concept ----
    ax.text(2.6, 3.15, "The Augmented Phase Space (what makes Phase 2 possible)",
           ha="center", fontsize=10.5, fontweight="bold")
    box(ax, (0.4, 1.5), 2.0, 1.3, "Weights θ\n+ momentum $p_\\theta$", facecolor="#D9D9D9", fontsize=9)
    ax.text(2.6, 2.15, "+", ha="center", fontsize=16, fontweight="bold")
    box(ax, (2.9, 1.5), 2.0, 1.3, "Hyperparams λ\n+ momentum $p_\\lambda$", facecolor="#D9D9D9", fontsize=9)
    arrow(ax, (2.6, 1.45), (2.6, 0.75))
    box(ax, (0.9, 0.05), 3.4, 0.65,
       r"$H(\theta,\lambda,p_\theta,p_\lambda) = KE(p_\theta) + KE(p_\lambda) + \mathcal{L}(\theta,\lambda)$",
       facecolor="#FFF3B0", fontsize=10)

    # ---- Bottom-right: leapfrog conserves this jointly ----
    ax.text(9.6, 3.15, "Leapfrog Integration Conserves H — Validated Empirically",
           ha="center", fontsize=10.5, fontweight="bold")
    import json
    trace_path = os.path.join(os.path.dirname(__file__), "..", "results", "energy_conservation_trace.json")
    ax_inset = fig.add_axes([0.63, 0.08, 0.32, 0.4])
    if os.path.exists(trace_path):
        with open(trace_path) as f:
            trace = json.load(f)
        y = trace["pct_unclipped"]
        x = list(range(len(y)))
        ax_inset.plot(x, y, color="#4C72B0", linewidth=1.8,
                     label=f"Real measured drift: {y[-1]:+.2f}% over\n{len(y)-1} leapfrog sub-steps (no grad clip)")
    else:
        ax_inset.text(0.5, 0.5, "Run generate_conservation_and_trajectory_figures.py\nfirst to populate this with real data",
                      ha="center", va="center", fontsize=7, transform=ax_inset.transAxes)
    ax_inset.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax_inset.set_xlabel("Leapfrog sub-step", fontsize=8)
    ax_inset.set_ylabel("Δ H (%)", fontsize=8)
    ax_inset.tick_params(labelsize=7)
    ax_inset.legend(fontsize=6.5, loc="upper left")
    ax_inset.set_title("See plots/energy_conservation_live.png\nfor the full validation figure", fontsize=7, color="#555555")

    plt.savefig(os.path.join(PLOTS_DIR, "method_c_pipeline_overview.png"), dpi=160, bbox_inches="tight")
    plt.close()
    print("Saved plots/method_c_pipeline_overview.png")


if __name__ == "__main__":
    main()
