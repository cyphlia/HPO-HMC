#!/usr/bin/env python
"""
Statistical Significance Testing for HPOBench Benchmark Results.
================================================================

Implements the standard Demšar (2006) procedure for comparing multiple
classifiers/optimizers across multiple datasets:

  1. Friedman test — global test for significant differences in ranks
  2. Nemenyi post-hoc test — pairwise critical differences
  3. Critical Difference (CD) diagram — visual summary

References:
  Demšar, J. (2006). Statistical Comparisons of Classifiers over
  Multiple Data Sets. JMLR 7:1–30.

Usage:
  python statistical_tests.py
  python statistical_tests.py --alpha 0.05 --output-dir plots
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional dependency checks
# ---------------------------------------------------------------------------
try:
    import scipy.stats as ss
except ImportError:
    print("ERROR: scipy is required. Install with: pip install scipy")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
except ImportError:
    print("ERROR: matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)

# Try scikit-posthocs for Nemenyi test
_POSTHOCS_AVAILABLE = True
try:
    import scikit_posthocs as sp
except ImportError:
    _POSTHOCS_AVAILABLE = False
    print("WARNING: scikit-posthocs not installed. Using manual Nemenyi implementation.")
    print("         Install with: pip install scikit-posthocs")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / ".." / "results" / "hpobench"
PLOTS_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "plots"

OPTIMIZERS = [
    "RandomSearch",
    "OptunaTPE",
    "MethodA_HHD",
    "MethodB_ABBO",
    "MethodC_Unified",
]

OPTIMIZER_LABELS = {
    "RandomSearch":    "Random Search",
    "OptunaTPE":       "Optuna TPE",
    "MethodA_HHD":     "Method A (HHD)",
    "MethodB_ABBO":    "Method B (ABBO)",
    "MethodC_Unified": "Method C (Unified)",
}

OPTIMIZER_COLORS = {
    "RandomSearch":    "#9E9E9E",
    "OptunaTPE":       "#FF9800",
    "MethodA_HHD":     "#2196F3",
    "MethodB_ABBO":    "#FF5722",
    "MethodC_Unified": "#4CAF50",
}

SEEDS = list(range(5))

ALL_DATASETS = {
    "hpobench": ["australian", "blood_transfusion", "vehicle", "segment"],
    "hpolib": ["naval_propulsion", "parkinsons_telemonitoring",
               "protein_structure", "slice_localization"],
    "nasbench201": ["cifar10", "cifar100", "imagenet"],
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_final_best_costs() -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """
    Load per-seed final best cost for every (suite, dataset, optimizer).

    Returns:
        nested dict: suite → dataset → optimizer → [cost_seed0, ..., cost_seed4]
    """
    results: Dict[str, Dict[str, Dict[str, List[float]]]] = {}

    for suite, datasets in ALL_DATASETS.items():
        results[suite] = {}
        for dataset in datasets:
            results[suite][dataset] = {}
            for opt in OPTIMIZERS:
                seed_costs = []
                for seed in SEEDS:
                    fname = f"{opt}_seed{seed}.json"
                    path = RESULTS_DIR / suite / dataset / fname
                    if not path.exists():
                        continue
                    try:
                        with open(path, "r") as f:
                            data = json.load(f)
                        traj = data.get("trajectory", [])
                        if traj:
                            seed_costs.append(traj[-1]["best_cost"])
                    except (json.JSONDecodeError, IOError, KeyError):
                        continue
                if seed_costs:
                    results[suite][dataset][opt] = seed_costs
    return results


def build_rank_matrix(
    results: Dict[str, Dict[str, Dict[str, List[float]]]]
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Build a (n_datasets × n_optimizers) rank matrix from per-seed results.

    For each dataset, the optimizer with the lowest mean final cost gets rank 1.
    Ties are handled via average ranking.

    Returns:
        rank_matrix: shape (n_datasets, n_optimizers)
        dataset_names: list of dataset display names
        optimizer_names: list of optimizer keys
    """
    dataset_names = []
    mean_costs_per_dataset = []  # list of dicts: opt → mean_cost

    for suite, suite_data in results.items():
        for dataset, opt_data in suite_data.items():
            # Only include if all optimizers have data
            if len(opt_data) < len(OPTIMIZERS):
                # Still include but with NaN for missing
                pass
            dataset_names.append(f"{suite}/{dataset}")
            means = {}
            for opt in OPTIMIZERS:
                if opt in opt_data and opt_data[opt]:
                    means[opt] = np.mean(opt_data[opt])
                else:
                    means[opt] = np.nan
            mean_costs_per_dataset.append(means)

    n_datasets = len(dataset_names)
    n_opt = len(OPTIMIZERS)
    rank_matrix = np.full((n_datasets, n_opt), np.nan)

    for i, means in enumerate(mean_costs_per_dataset):
        costs = np.array([means[opt] for opt in OPTIMIZERS])
        valid_mask = ~np.isnan(costs)

        if valid_mask.sum() >= 2:
            # Rank using scipy (handles ties via average method)
            valid_costs = costs[valid_mask]
            ranks = ss.rankdata(valid_costs, method="average")
            rank_matrix[i, valid_mask] = ranks

    return rank_matrix, dataset_names, OPTIMIZERS


def build_cost_table_with_std(
    results: Dict[str, Dict[str, Dict[str, List[float]]]]
) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Build a table with 'mean ± std' strings for all results.

    Returns:
        nested dict: suite → dataset → optimizer → "mean ± std" string
    """
    table = {}
    for suite, suite_data in results.items():
        table[suite] = {}
        for dataset, opt_data in suite_data.items():
            table[suite][dataset] = {}
            for opt in OPTIMIZERS:
                if opt in opt_data and opt_data[opt]:
                    costs = opt_data[opt]
                    mean = np.mean(costs)
                    std = np.std(costs)
                    table[suite][dataset][opt] = f"{mean:.6f} ± {std:.6f}"
                else:
                    table[suite][dataset][opt] = "N/A"
    return table


# ---------------------------------------------------------------------------
# Friedman test
# ---------------------------------------------------------------------------

def friedman_test(rank_matrix: np.ndarray) -> Tuple[float, float]:
    """
    Friedman chi-squared test for comparing k related samples.

    Parameters:
        rank_matrix: shape (n_datasets, k_optimizers)

    Returns:
        (chi2_statistic, p_value)
    """
    n, k = rank_matrix.shape

    # Remove rows with any NaN
    valid_rows = ~np.isnan(rank_matrix).any(axis=1)
    R = rank_matrix[valid_rows]
    n_valid = R.shape[0]

    if n_valid < 3:
        print("WARNING: Too few complete datasets for Friedman test")
        return 0.0, 1.0

    # Use scipy's implementation
    # scipy.stats.friedmanchisquare expects each group as a separate array
    groups = [R[:, j] for j in range(k)]
    try:
        stat, pval = ss.friedmanchisquare(*groups)
    except ValueError as e:
        print(f"WARNING: Friedman test failed: {e}")
        return 0.0, 1.0

    return float(stat), float(pval)


# ---------------------------------------------------------------------------
# Nemenyi post-hoc test
# ---------------------------------------------------------------------------

def nemenyi_critical_difference(n_datasets: int, k: int, alpha: float = 0.05) -> float:
    """
    Compute the Nemenyi critical difference.

    CD = q_alpha * sqrt(k(k+1) / (6*N))

    where q_alpha is the critical value from the Studentized Range Distribution.
    """
    # Studentized range critical values (q_alpha) for alpha=0.05
    # Table values from Demšar (2006), Table 5
    # Format: q_alpha_005[k] for k groups
    q_alpha_005 = {
        2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728,
        6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
    }
    q_alpha_010 = {
        2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459,
        6: 2.589, 7: 2.693, 8: 2.780, 9: 2.855, 10: 2.920,
    }

    if alpha <= 0.05:
        q = q_alpha_005.get(k)
    else:
        q = q_alpha_010.get(k)

    if q is None:
        # Fallback: approximate from normal distribution
        from scipy.stats import norm
        q = norm.ppf(1 - alpha / (k * (k - 1)))  # Bonferroni-like

    cd = q * np.sqrt(k * (k + 1) / (6 * n_datasets))
    return cd


def nemenyi_pairwise(
    rank_matrix: np.ndarray, alpha: float = 0.05
) -> Tuple[np.ndarray, float]:
    """
    Compute Nemenyi pairwise p-values (or significance matrix).

    Returns:
        (p_value_matrix, critical_difference)
    """
    valid_rows = ~np.isnan(rank_matrix).any(axis=1)
    R = rank_matrix[valid_rows]
    n, k = R.shape

    cd = nemenyi_critical_difference(n, k, alpha)

    if _POSTHOCS_AVAILABLE:
        try:
            # scikit-posthocs expects raw data, not ranks
            # Use the rank matrix directly with posthoc_nemenyi_friedman
            p_matrix = sp.posthoc_nemenyi_friedman(R)
            return p_matrix.values, cd
        except Exception as e:
            print(f"  scikit-posthocs Nemenyi failed ({e}), using manual implementation")

    # Manual: compute pairwise mean rank differences
    mean_ranks = np.mean(R, axis=0)
    p_matrix = np.ones((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            diff = abs(mean_ranks[i] - mean_ranks[j])
            # Significant if diff > CD
            # Approximate p-value from the difference
            if diff > cd:
                p_matrix[i, j] = 0.01  # significant
                p_matrix[j, i] = 0.01
            else:
                p_matrix[i, j] = 0.5  # not significant (approximate)
                p_matrix[j, i] = 0.5

    return p_matrix, cd


# ---------------------------------------------------------------------------
# Critical Difference Diagram
# ---------------------------------------------------------------------------

def plot_cd_diagram(
    avg_ranks: np.ndarray,
    names: List[str],
    cd: float,
    n_datasets: int,
    alpha: float = 0.05,
    title: str = "Critical Difference Diagram",
    save_path: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None,
):
    """
    Draw a publication-quality Critical Difference (CD) diagram.

    Groups of methods connected by a horizontal bar are NOT significantly
    different (their average rank difference < CD).

    Based on Demšar (2006) Figure 3 style.
    """
    k = len(names)
    sorted_indices = np.argsort(avg_ranks)
    sorted_ranks = avg_ranks[sorted_indices]
    sorted_names = [names[i] for i in sorted_indices]

    # Determine which pairs are NOT significantly different
    # (i.e., their rank difference < CD)
    cliques = []
    for i in range(k):
        for j in range(i + 1, k):
            if abs(sorted_ranks[i] - sorted_ranks[j]) < cd:
                # Check if this pair extends an existing clique
                merged = False
                for clique in cliques:
                    if i in clique or j in clique:
                        clique.add(i)
                        clique.add(j)
                        merged = True
                        break
                if not merged:
                    cliques.append({i, j})

    # Merge overlapping cliques
    merged_cliques = []
    for clique in cliques:
        merged = False
        for mc in merged_cliques:
            if mc & clique:
                mc |= clique
                merged = True
                break
        if not merged:
            merged_cliques.append(clique)

    # --- Figure setup ---
    fig_width = max(8, k * 1.5)
    fig_height = max(3.5, 1.5 + len(merged_cliques) * 0.35 + k * 0.15)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))

    # Scale: rank axis
    rank_min = 1
    rank_max = k
    margin = 0.5

    ax.set_xlim(rank_min - margin, rank_max + margin)
    ax.set_ylim(-0.5 - len(merged_cliques) * 0.4, k * 0.5 + 1.5)

    # Draw the rank axis at the top
    axis_y = k * 0.5 + 0.8
    ax.hlines(axis_y, rank_min, rank_max, colors="black", linewidth=1.5)
    for r in range(rank_min, rank_max + 1):
        ax.vlines(r, axis_y - 0.08, axis_y + 0.08, colors="black", linewidth=1.2)
        ax.text(r, axis_y + 0.2, str(r), ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    # CD bar
    cd_bar_y = axis_y + 0.65
    cd_start = rank_min
    cd_end = rank_min + cd
    ax.annotate(
        "", xy=(cd_end, cd_bar_y), xytext=(cd_start, cd_bar_y),
        arrowprops=dict(arrowstyle="<->", color="red", lw=2.0),
    )
    ax.text(
        (cd_start + cd_end) / 2, cd_bar_y + 0.15,
        f"CD = {cd:.3f}",
        ha="center", va="bottom", fontsize=10, fontweight="bold", color="red",
    )

    # Place methods alternating left and right
    # Split sorted list: best half on left, worst half on right
    half = k // 2
    left_indices = list(range(half))
    right_indices = list(range(half, k))

    name_positions = {}  # index → (x, y, ha)

    # Left side (best ranks) — names go to the left
    for pos, idx in enumerate(left_indices):
        x = sorted_ranks[idx]
        y = axis_y - 0.5 - pos * 0.45
        name_x = rank_min - margin + 0.1
        # Draw line from axis to name
        ax.plot([x, x], [axis_y - 0.08, y], color="gray", linewidth=0.8)
        ax.plot([x, name_x + 0.05], [y, y], color="gray", linewidth=0.8)
        # Draw rank marker
        ax.plot(x, axis_y - 0.08, "o", color="black", markersize=5, zorder=5)
        # Draw name
        label = sorted_names[idx]
        color = colors.get(label, "black") if colors else "black"
        ax.text(name_x, y, f"{label} ({sorted_ranks[idx]:.2f})",
                ha="left", va="center", fontsize=10, fontweight="bold",
                color=color)
        name_positions[idx] = (x, y)

    # Right side (worst ranks) — names go to the right
    for pos, idx in enumerate(right_indices):
        x = sorted_ranks[idx]
        y = axis_y - 0.5 - pos * 0.45
        name_x = rank_max + margin - 0.1
        # Draw line from axis to name
        ax.plot([x, x], [axis_y - 0.08, y], color="gray", linewidth=0.8)
        ax.plot([x, name_x - 0.05], [y, y], color="gray", linewidth=0.8)
        # Draw rank marker
        ax.plot(x, axis_y - 0.08, "o", color="black", markersize=5, zorder=5)
        # Draw name
        label = sorted_names[idx]
        color = colors.get(label, "black") if colors else "black"
        ax.text(name_x, y, f"({sorted_ranks[idx]:.2f}) {label}",
                ha="right", va="center", fontsize=10, fontweight="bold",
                color=color)
        name_positions[idx] = (x, y)

    # Draw clique bars (horizontal bars connecting non-significantly-different methods)
    bar_y_start = -0.3
    for ci, clique in enumerate(merged_cliques):
        clique_sorted = sorted(clique)
        x_left = sorted_ranks[clique_sorted[0]]
        x_right = sorted_ranks[clique_sorted[-1]]
        bar_y = bar_y_start - ci * 0.35
        ax.hlines(bar_y, x_left, x_right, colors="black", linewidth=3.5)
        # Small vertical ticks at endpoints
        ax.vlines(x_left, bar_y - 0.06, bar_y + 0.06, colors="black", linewidth=2)
        ax.vlines(x_right, bar_y - 0.06, bar_y + 0.06, colors="black", linewidth=2)

    # Title and annotations
    ax.set_title(title, fontsize=14, fontweight="bold", pad=30)
    ax.text(
        0.5, -0.02,
        f"Friedman test: {n_datasets} datasets, {k} methods, alpha = {alpha}",
        transform=ax.transAxes, ha="center", va="top", fontsize=9,
        fontstyle="italic", color="gray",
    )

    ax.set_axis_off()
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  [OK] Saved CD diagram -> {save_path}")

    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Summary statistics with mean ± std
# ---------------------------------------------------------------------------

def print_results_table(
    results: Dict[str, Dict[str, Dict[str, List[float]]]],
    rank_matrix: np.ndarray,
    dataset_names: List[str],
):
    """Print a formatted table with mean ± std and ranks."""
    print("\n" + "=" * 120)
    print("  RESULTS TABLE: mean +/- std (rank)")
    print("=" * 120)

    # Header
    header = f"{'Dataset':<30}"
    for opt in OPTIMIZERS:
        header += f"  {OPTIMIZER_LABELS[opt]:>22}"
    print(header)
    print("-" * 120)

    row_idx = 0
    for suite, suite_data in results.items():
        for dataset, opt_data in suite_data.items():
            row = f"{suite}/{dataset:<25}"
            for j, opt in enumerate(OPTIMIZERS):
                if opt in opt_data and opt_data[opt]:
                    mean = np.mean(opt_data[opt])
                    std = np.std(opt_data[opt])
                    rank = rank_matrix[row_idx, j] if row_idx < rank_matrix.shape[0] else np.nan
                    if not np.isnan(rank):
                        cell = f"{mean:.4f}±{std:.4f} ({int(rank)})"
                    else:
                        cell = f"{mean:.4f}±{std:.4f}"
                else:
                    cell = "N/A"
                row += f"  {cell:>22}"
            print(row)
            row_idx += 1

    # Average ranks
    print("-" * 120)
    avg_row = f"{'Average Rank':<30}"
    for j in range(len(OPTIMIZERS)):
        col = rank_matrix[:, j]
        valid = col[~np.isnan(col)]
        avg = np.mean(valid) if len(valid) > 0 else np.nan
        avg_row += f"  {avg:>22.2f}"
    print(avg_row)
    print("=" * 120)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_statistical_tests(alpha: float = 0.05, output_dir: str = None):
    """
    Run the full statistical testing pipeline.

    1. Load all per-seed results
    2. Build rank matrix
    3. Friedman test
    4. Nemenyi post-hoc test
    5. Generate CD diagram
    6. Save JSON summary
    """
    if output_dir is None:
        output_dir = str(PLOTS_DIR)

    print("\n" + "=" * 70)
    print("  STATISTICAL SIGNIFICANCE TESTING")
    print("  Demsar (2006) Friedman + Nemenyi procedure")
    print("=" * 70)

    # Step 1: Load data
    print("\n[Step 1] Loading per-seed results from results/hpobench/ ...")
    results = load_final_best_costs()

    total_datasets = sum(len(d) for d in results.values())
    print(f"  Loaded {total_datasets} datasets across {len(results)} suites")

    # Step 2: Build rank matrix
    print("\n[Step 2] Building rank matrix ...")
    rank_matrix, dataset_names, opt_names = build_rank_matrix(results)
    print(f"  Rank matrix shape: {rank_matrix.shape} "
          f"({rank_matrix.shape[0]} datasets × {rank_matrix.shape[1]} optimizers)")

    # Print full results table
    print_results_table(results, rank_matrix, dataset_names)

    # Step 3: Friedman test
    print("\n[Step 3] Friedman test ...")
    chi2, p_value = friedman_test(rank_matrix)
    print(f"  Friedman chi2 = {chi2:.4f}, p-value = {p_value:.6f}")

    if p_value < alpha:
        print(f"  >> SIGNIFICANT at alpha = {alpha}: reject H0 (methods differ)")
    else:
        print(f"  >> NOT significant at alpha = {alpha}: cannot reject H0")
        print("  (Nemenyi post-hoc test still computed for reference)")

    # Step 4: Nemenyi post-hoc test
    print("\n[Step 4] Nemenyi post-hoc test ...")
    p_matrix, cd = nemenyi_pairwise(rank_matrix, alpha)
    print(f"  Critical Difference (CD) = {cd:.4f}")

    # Average ranks
    valid_rows = ~np.isnan(rank_matrix).any(axis=1)
    R_valid = rank_matrix[valid_rows]
    avg_ranks = np.mean(R_valid, axis=0)

    print("\n  Average ranks:")
    for j, opt in enumerate(OPTIMIZERS):
        label = OPTIMIZER_LABELS[opt]
        marker = " *BEST*" if avg_ranks[j] == np.min(avg_ranks) else ""
        print(f"    {label:<22}: {avg_ranks[j]:.4f}{marker}")

    # Pairwise significance
    print(f"\n  Pairwise comparisons (CD = {cd:.4f}):")
    for i in range(len(OPTIMIZERS)):
        for j in range(i + 1, len(OPTIMIZERS)):
            diff = abs(avg_ranks[i] - avg_ranks[j])
            sig = "SIG" if diff > cd else "n.s."
            print(f"    {OPTIMIZER_LABELS[OPTIMIZERS[i]]:>22} vs "
                  f"{OPTIMIZER_LABELS[OPTIMIZERS[j]]:<22}: "
                  f"dRank = {diff:.4f}  [{sig}]")

    # Step 5: CD Diagram
    print(f"\n[Step 5] Generating Critical Difference diagram ...")
    label_names = [OPTIMIZER_LABELS[opt] for opt in OPTIMIZERS]
    label_colors = {OPTIMIZER_LABELS[opt]: OPTIMIZER_COLORS[opt] for opt in OPTIMIZERS}

    n_valid = valid_rows.sum()

    # Save as both PDF (vector) and PNG
    for ext in ["pdf", "png"]:
        save_path = os.path.join(output_dir, f"cd_diagram.{ext}")
        plot_cd_diagram(
            avg_ranks=avg_ranks,
            names=label_names,
            cd=cd,
            n_datasets=n_valid,
            alpha=alpha,
            title=f"Critical Difference Diagram -- {n_valid} Benchmarks, alpha={alpha}",
            save_path=save_path,
            colors=label_colors,
        )

    # Step 6: Save JSON summary
    test_results = {
        "alpha": alpha,
        "n_datasets": int(n_valid),
        "n_optimizers": len(OPTIMIZERS),
        "friedman_test": {
            "chi2_statistic": round(chi2, 6),
            "p_value": round(p_value, 8),
            "significant": bool(p_value < alpha),
        },
        "nemenyi_test": {
            "critical_difference": round(cd, 6),
        },
        "average_ranks": {
            OPTIMIZER_LABELS[opt]: round(float(avg_ranks[j]), 4)
            for j, opt in enumerate(OPTIMIZERS)
        },
        "rank_matrix": {
            dataset_names[i]: {
                OPTIMIZER_LABELS[OPTIMIZERS[j]]: float(rank_matrix[i, j])
                if not np.isnan(rank_matrix[i, j]) else None
                for j in range(len(OPTIMIZERS))
            }
            for i in range(len(dataset_names))
        },
        "pairwise_significance": {
            f"{OPTIMIZER_LABELS[OPTIMIZERS[i]]} vs {OPTIMIZER_LABELS[OPTIMIZERS[j]]}": {
                "rank_difference": round(abs(avg_ranks[i] - avg_ranks[j]), 4),
                "significant": bool(abs(avg_ranks[i] - avg_ranks[j]) > cd),
            }
            for i in range(len(OPTIMIZERS))
            for j in range(i + 1, len(OPTIMIZERS))
        },
        "results_mean_std": {
            suite: {
                dataset: {
                    OPTIMIZER_LABELS.get(opt, opt): {
                        "mean": round(float(np.mean(costs)), 6),
                        "std": round(float(np.std(costs)), 6),
                        "n_seeds": len(costs),
                    }
                    for opt, costs in opt_data.items()
                }
                for dataset, opt_data in suite_data.items()
            }
            for suite, suite_data in results.items()
        },
        "reference": "Demšar, J. (2006). Statistical Comparisons of Classifiers "
                     "over Multiple Data Sets. JMLR 7:1–30.",
    }

    json_path = RESULTS_DIR / "statistical_tests.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    print(f"\n  [OK] Saved test results -> {json_path}")

    print("\n" + "=" * 70)
    print("  STATISTICAL TESTING COMPLETE")
    print("=" * 70 + "\n")

    return test_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Statistical significance testing for HPOBench results"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Significance level (default: 0.05)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for plots (default: plots/)"
    )
    args = parser.parse_args()

    run_statistical_tests(alpha=args.alpha, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
