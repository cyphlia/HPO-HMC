#!/usr/bin/env python
"""
Hyperparameter Sensitivity Analysis — Phase 10 (SIAM Revision).
================================================================

Sweeps each of Method C's meta-hyperparameters on the harmonic
oscillator to show robustness:

  1. epsilon (leapfrog step size): {0.001, 0.003, 0.005, 0.01, 0.02}
  2. L (leapfrog steps):          {2, 3, 6, 10, 20}
  3. P (plateau patience):        {2, 4, 8, 16}
  4. mass ratio m_lambda/m_theta: {0.1, 0.25, 0.5, 1.0, 2.0}

For each combination:
  - Runs 3 seeds (fast) on the harmonic oscillator
  - Records best validation loss (mean ± std)

Outputs:
  - plots/sensitivity_2x2.pdf  (2×2 grid, one panel per HP)
  - results_ablation/sensitivity_results.json

Usage:
  python sensitivity_analysis.py
  python sensitivity_analysis.py --seeds 0,1,2,3,4 --n-samples 1000
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch

import config
from hybrid_hhd_abbo_improved import ImprovedUnifiedTrainer

warnings.filterwarnings("ignore")

RESULTS_DIR = Path("results_ablation")
PLOTS_DIR   = Path("plots")

# Default sweep values for each meta-HP
SWEEP = {
    "epsilon": [0.001, 0.003, 0.005, 0.01, 0.02],
    "L":       [2, 3, 6, 10, 20],
    "P":       [2, 4, 8, 16],
    "mass_ratio": [0.1, 0.25, 0.5, 1.0, 2.0],
}

# Default/best setting for each (used when other params are fixed)
DEFAULTS = {
    "epsilon":    0.005,
    "L":          6,
    "P":          8,
    "mass_ratio": 0.1,
}

# Labels for plots
LABELS = {
    "epsilon":    r"Leapfrog Step Size $\varepsilon$",
    "L":          r"Leapfrog Steps $L$",
    "P":          r"Plateau Patience $P$",
    "mass_ratio": r"Mass Ratio $m_\lambda / m_\theta$",
}


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_single(
    epsilon: float = 0.005,
    n_leapfrog: int = 6,
    plateau_patience: int = 8,
    mass_ratio: float = 0.1,
    seed: int = 0,
    n_epochs: int = 60,
    n_warmup: int = 20,
    n_samples: int = 800,
    device: str = "cpu",
) -> float:
    """Run Method C with given meta-HPs and return best validation loss."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    mass_lambda = mass_ratio * 1.0  # m_theta = 1.0 fixed

    trainer = ImprovedUnifiedTrainer(
        hyperparam_space=config.HYPERPARAM_SPACE,
        init_hyperparams=config.INIT_HYPERPARAMS,
        initial_step=epsilon,
        n_leapfrog=n_leapfrog,
        temperature=config.TEMPERATURE,
        mass_lambda=mass_lambda,
        lbfgs_patience=plateau_patience,
        lbfgs_tol=1e-3,
        device=device,
    )
    history = trainer.train(
        n_samples=n_samples,
        n_warmup=n_warmup,
        n_hamilton=n_epochs,
    )
    val_losses = history.get("val_loss", [])
    return float(min(val_losses)) if val_losses else float("inf")


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def sweep_hp(
    hp_name: str,
    values: List,
    seeds: List[int],
    n_epochs: int,
    n_warmup: int,
    n_samples: int,
    device: str,
) -> Dict[str, Dict]:
    """Sweep one HP while holding others at their defaults."""
    results = {}
    defaults = dict(DEFAULTS)

    for val in values:
        # Build kwargs with this HP varied and all others at default
        kwargs = {
            "epsilon":         defaults["epsilon"],
            "n_leapfrog":      int(defaults["L"]),
            "plateau_patience": int(defaults["P"]),
            "mass_ratio":      defaults["mass_ratio"],
        }
        # Override with the swept value
        if hp_name == "epsilon":
            kwargs["epsilon"] = val
        elif hp_name == "L":
            kwargs["n_leapfrog"] = int(val)
        elif hp_name == "P":
            kwargs["plateau_patience"] = int(val)
        elif hp_name == "mass_ratio":
            kwargs["mass_ratio"] = val

        best_vals = []
        for seed in seeds:
            bv = run_single(**kwargs, seed=seed, n_epochs=n_epochs,
                            n_warmup=n_warmup, n_samples=n_samples,
                            device=device)
            best_vals.append(bv)
            print(f"    [{hp_name}={val}] seed={seed} -> best_val={bv:.5f}")

        results[str(val)] = {
            "value":  val,
            "mean":   float(np.mean(best_vals)),
            "std":    float(np.std(best_vals)),
            "values": best_vals,
        }
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_panel(ax, hp_name: str, sweep_data: Dict, default_val, label: str,
                logscale: bool = False):
    """Plot a single sensitivity panel."""
    vals  = [float(k) for k in sweep_data]
    means = [sweep_data[k]["mean"] for k in sweep_data]
    stds  = [sweep_data[k]["std"]  for k in sweep_data]

    # Line + shaded uncertainty
    ax.plot(vals, means, "o-", color="#4B91F1", lw=2, ms=7, zorder=3,
            label="mean ± std")
    ax.fill_between(vals,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.18, color="#4B91F1")

    # Mark best value with a star
    best_idx = int(np.argmin(means))
    ax.plot(vals[best_idx], means[best_idx], "*",
            color="#F1A94E", ms=18, zorder=4, label="Best setting")

    # Highlight default setting
    if default_val in [float(k) for k in sweep_data]:
        ax.axvline(default_val, ls="--", color="gray", lw=1.0, alpha=0.6,
                   label=f"Default ({default_val})")

    ax.set_xlabel(label, fontsize=11)
    ax.set_ylabel("Best Val. Loss (MSE)", fontsize=10)
    if logscale:
        ax.set_xscale("log")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25)
    ax.set_facecolor("#FAFAFA")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_sensitivity_plot(all_sweep_data: Dict, save_path: str):
    """Generate 2×2 sensitivity grid plot."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        "Method C Hyperparameter Sensitivity — Harmonic Oscillator\n"
        "(mean ± std over seeds; ★ = best observed)",
        fontsize=13, fontweight="bold", y=0.98,
    )
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 11,
    })

    panels = [
        ("epsilon",    axes[0, 0], True),
        ("L",          axes[0, 1], False),
        ("P",          axes[1, 0], False),
        ("mass_ratio", axes[1, 1], True),
    ]

    for hp_name, ax, logscale in panels:
        data = all_sweep_data.get(hp_name, {})
        if not data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        ax.set_title(LABELS[hp_name], fontsize=11, pad=8)
        _plot_panel(ax, hp_name, data, DEFAULTS[hp_name],
                    LABELS[hp_name], logscale=logscale)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Saved sensitivity plot -> {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_sensitivity_analysis(
    seeds: Optional[List[int]] = None,
    hps: Optional[List[str]] = None,
    n_epochs: int = 60,
    n_warmup: int = 20,
    n_samples: int = 800,
    device: str = "cpu",
) -> Dict:
    """Run the full sensitivity sweep."""
    if seeds is None:
        seeds = [0, 1, 2]
    if hps is None:
        hps = list(SWEEP.keys())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  HYPERPARAMETER SENSITIVITY ANALYSIS — Phase 10 (SIAM)")
    print(f"  HPs     : {hps}")
    print(f"  Seeds   : {seeds}")
    print(f"  Epochs  : {n_epochs} + {n_warmup} warmup")
    print("=" * 70)

    all_sweep_data: Dict = {}

    for hp_name in hps:
        print(f"\n--- Sweeping: {hp_name} over {SWEEP[hp_name]} ---")
        all_sweep_data[hp_name] = sweep_hp(
            hp_name=hp_name,
            values=SWEEP[hp_name],
            seeds=seeds,
            n_epochs=n_epochs,
            n_warmup=n_warmup,
            n_samples=n_samples,
            device=device,
        )

    # Save JSON results
    results_path = RESULTS_DIR / "sensitivity_results.json"
    with open(results_path, "w") as f:
        json.dump(all_sweep_data, f, indent=2)
    print(f"\n  [OK] Saved sensitivity results -> {results_path}")

    # Generate 2×2 plot
    plot_path = str(PLOTS_DIR / "sensitivity_2x2.pdf")
    make_sensitivity_plot(all_sweep_data, plot_path)

    # Also save as PNG for quick preview
    png_path = str(PLOTS_DIR / "sensitivity_2x2.png")
    make_sensitivity_plot(all_sweep_data, png_path)

    print("\n" + "=" * 70)
    print("  SENSITIVITY ANALYSIS COMPLETE")
    print("=" * 70)

    return all_sweep_data


def main():
    parser = argparse.ArgumentParser(
        description="Phase 10: Hyperparameter sensitivity analysis for Method C"
    )
    parser.add_argument("--seeds",     type=str, default="0,1,2")
    parser.add_argument("--hps",       type=str, default="all",
                        help="Comma-separated HPs or 'all': epsilon,L,P,mass_ratio")
    parser.add_argument("--epochs",    type=int, default=60)
    parser.add_argument("--warmup",    type=int, default=20)
    parser.add_argument("--n-samples", type=int, default=800)
    parser.add_argument("--device",    type=str, default="cpu",
                        choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    hps   = None if args.hps.lower() == "all" else [
        h.strip() for h in args.hps.split(",")
    ]

    run_sensitivity_analysis(
        seeds=seeds,
        hps=hps,
        n_epochs=args.epochs,
        n_warmup=args.warmup,
        n_samples=args.n_samples,
        device=args.device,
    )


if __name__ == "__main__":
    main()
