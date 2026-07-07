"""
plot_hpobench.py — Publication-quality plots for HPOBench benchmark results.

Generates:
  plots/fig6_hpobench_regret.png   — 2×2 HPOBench regret curves
  plots/fig6b_hpolib_regret.png    — 2×2 HPOLib regret curves (if data exists)
  plots/fig7_nasbench201_regret.png — 1×3 NAS-Bench-201 regret curves
  plots/fig8_hpobench_summary.png  — Cross-benchmark summary (bar + heatmap)
"""

import matplotlib
matplotlib.use('Agg')

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.dpi': 200,
})

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_hpobench')
PLOTS_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')

OPTIMIZERS = [
    'RandomSearch',
    'OptunaTPE',
    'MethodA_HHD',
    'MethodB_ABBO',
    'MethodC_Unified',
]

OPTIMIZER_LABELS = {
    'RandomSearch':     'Random Search',
    'OptunaTPE':        'Optuna TPE',
    'MethodA_HHD':      'Method A (HHD)',
    'MethodB_ABBO':     'Method B (ABBO)',
    'MethodC_Unified':  'Method C (Unified)',
}

OPTIMIZER_COLORS = {
    'RandomSearch':     '#9E9E9E',
    'OptunaTPE':        '#FF9800',
    'MethodA_HHD':      '#2196F3',
    'MethodB_ABBO':     '#FF5722',
    'MethodC_Unified':  '#4CAF50',
}

OPTIMIZER_LINESTYLES = {
    'RandomSearch':     'dotted',
    'OptunaTPE':        'dashed',
    'MethodA_HHD':      'solid',
    'MethodB_ABBO':     'dashdot',
    'MethodC_Unified':  'solid',
}

OPTIMIZER_LW = {
    'RandomSearch':     1.5,
    'OptunaTPE':        1.5,
    'MethodA_HHD':      1.5,
    'MethodB_ABBO':     1.5,
    'MethodC_Unified':  2.5,
}

SEEDS = list(range(5))

HPOBENCH_DATASETS  = ['australian', 'blood_transfusion', 'vehicle', 'segment']
HPOLIB_DATASETS    = ['naval_propulsion', 'parkinsons_telemonitoring',
                       'protein_structure', 'slice_localization']
NASBENCH_DATASETS  = ['cifar10', 'cifar100', 'imagenet']

SUITE_DIR_MAP = {
    'hpobench':     'hpobench',
    'hpolib':       'hpolib',
    'nasbench201':  'nasbench201',
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nice_dataset(name: str) -> str:
    """Capitalise and replace underscores with spaces."""
    return name.replace('_', ' ').title()


def _load_trajectory(suite: str, dataset: str, optimizer: str, seed: int):
    """Load a single result JSON; return trajectory array or None."""
    fname = f"{optimizer}_seed{seed}.json"
    path  = os.path.join(RESULTS_DIR, SUITE_DIR_MAP[suite], dataset, fname)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return data.get('trajectory', None)
    except (json.JSONDecodeError, IOError) as exc:
        print(f"  [WARN] Could not read {path}: {exc}")
        return None


def _collect_best_curves(suite: str, dataset: str, optimizer: str):
    """
    Return (trials, mean_best, std_best) across seeds.
    All curves are aligned to the shortest common length.
    Returns (None, None, None) if no data found.
    """
    curves = []
    for seed in SEEDS:
        traj = _load_trajectory(suite, dataset, optimizer, seed)
        if traj is None:
            continue
        best_costs = [entry['best_cost'] for entry in traj]
        curves.append(best_costs)

    if not curves:
        return None, None, None

    # Align to shortest length
    min_len = min(len(c) for c in curves)
    aligned = np.array([c[:min_len] for c in curves])
    trials  = np.arange(1, min_len + 1)
    mean    = np.mean(aligned, axis=0)
    std     = np.std(aligned, axis=0)
    return trials, mean, std


def _load_summary():
    """Load summary.json and return dict (or empty dict on failure)."""
    path = os.path.join(RESULTS_DIR, 'summary.json')
    if not os.path.isfile(path):
        print(f"  [WARN] summary.json not found at {path}")
        return {}
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as exc:
        print(f"  [WARN] Could not read summary.json: {exc}")
        return {}


# ---------------------------------------------------------------------------
# Plotting — regret curves
# ---------------------------------------------------------------------------

def _plot_regret_grid(suite, datasets, fig_title, figsize, save_path,
                      nrows=None, ncols=None):
    """
    Generic regret-curve figure for a list of datasets within a suite.
    Returns True if the figure was created, False if no data was available.
    """
    n = len(datasets)
    if nrows is None or ncols is None:
        if n <= 3:
            nrows, ncols = 1, n
        else:
            nrows, ncols = 2, 2

    has_any_data = False
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             constrained_layout=False)
    if n == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()

    for idx, dataset in enumerate(datasets):
        ax = axes_flat[idx]
        for opt in OPTIMIZERS:
            trials, mean, std = _collect_best_curves(suite, dataset, opt)
            if trials is None:
                continue
            has_any_data = True
            color = OPTIMIZER_COLORS[opt]
            ls    = OPTIMIZER_LINESTYLES[opt]
            lw    = OPTIMIZER_LW[opt]
            label = OPTIMIZER_LABELS[opt]
            ax.plot(trials, mean, color=color, linestyle=ls, linewidth=lw,
                    label=label)
            ax.fill_between(trials, mean - std, mean + std,
                            color=color, alpha=0.15)
        ax.set_title(_nice_dataset(dataset))
        ax.set_xlabel('Number of Trials')
        ax.set_ylabel('Best Objective Value')
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.tick_params(labelsize=9)

    # Hide unused axes
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    if not has_any_data:
        plt.close(fig)
        return False

    # Shared legend — outside bottom
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center',
                   ncol=len(handles), frameon=True, fontsize=10,
                   bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(fig_title, fontsize=15, fontweight='bold', y=1.01)
    fig.subplots_adjust(hspace=0.35, wspace=0.30, bottom=0.10)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] Saved {save_path}")
    return True


def plot_fig6():
    """Figure 6: HPOBench 2×2 regret curves."""
    return _plot_regret_grid(
        suite='hpobench',
        datasets=HPOBENCH_DATASETS,
        fig_title='HPOBench Tabular Benchmark — Optimization Trajectories',
        figsize=(14, 10),
        save_path=os.path.join(PLOTS_DIR, 'fig6_hpobench_regret.png'),
        nrows=2, ncols=2,
    )


def plot_fig6b():
    """Figure 6b: HPOLib 2×2 regret curves (only if data exists)."""
    return _plot_regret_grid(
        suite='hpolib',
        datasets=HPOLIB_DATASETS,
        fig_title='HPOLib Benchmark — Optimization Trajectories',
        figsize=(14, 10),
        save_path=os.path.join(PLOTS_DIR, 'fig6b_hpolib_regret.png'),
        nrows=2, ncols=2,
    )


def plot_fig7():
    """Figure 7: NAS-Bench-201 1×3 regret curves."""
    return _plot_regret_grid(
        suite='nasbench201',
        datasets=NASBENCH_DATASETS,
        fig_title='NAS-Bench-201 — Optimization Trajectories',
        figsize=(18, 5.5),
        save_path=os.path.join(PLOTS_DIR, 'fig7_nasbench201_regret.png'),
        nrows=1, ncols=3,
    )


# ---------------------------------------------------------------------------
# Plotting — summary (Figure 8)
# ---------------------------------------------------------------------------

def _compute_summary_from_trajectories():
    """
    Build summary statistics directly from individual result JSONs.
    Returns dict[suite][dataset][optimizer] = {mean_best, std_best}.
    """
    all_suites = {
        'hpobench':    HPOBENCH_DATASETS,
        'hpolib':      HPOLIB_DATASETS,
        'nasbench201': NASBENCH_DATASETS,
    }
    summary = {}
    for suite, datasets in all_suites.items():
        summary[suite] = {}
        for dataset in datasets:
            summary[suite][dataset] = {}
            for opt in OPTIMIZERS:
                final_bests = []
                for seed in SEEDS:
                    traj = _load_trajectory(suite, dataset, opt, seed)
                    if traj is None:
                        continue
                    final_bests.append(traj[-1]['best_cost'])
                if final_bests:
                    summary[suite][dataset][opt] = {
                        'mean_best': float(np.mean(final_bests)),
                        'std_best':  float(np.std(final_bests)),
                    }
    return summary


def plot_fig8():
    """Figure 8: Cross-benchmark performance summary (bar + heatmap)."""

    # Try summary.json first, fall back to computing from trajectories
    summary_raw = _load_summary()
    if not summary_raw:
        print("  → Computing summary from individual trajectory files …")
        summary_raw = _compute_summary_from_trajectories()

    # Flatten into a list of (suite/dataset, {optimizer: mean_best})
    benchmarks = []      # list of nice-name strings
    bench_keys = []      # list of (suite, dataset)
    opt_means  = {o: [] for o in OPTIMIZERS}
    opt_stds   = {o: [] for o in OPTIMIZERS}

    for suite in ['hpobench', 'hpolib', 'nasbench201']:
        suite_data = summary_raw.get(suite, {})
        for dataset in sorted(suite_data.keys()):
            ds_data = suite_data[dataset]
            if not ds_data:
                continue
            # Only include if at least one optimizer has data
            has_data = any(opt in ds_data for opt in OPTIMIZERS)
            if not has_data:
                continue
            benchmarks.append(_nice_dataset(dataset))
            bench_keys.append((suite, dataset))
            for opt in OPTIMIZERS:
                info = ds_data.get(opt, {})
                opt_means[opt].append(info.get('mean_best', np.nan))
                opt_stds[opt].append(info.get('std_best', 0.0))

    if not benchmarks:
        print("  [WARN] No summary data available — skipping Figure 8.")
        return False

    n_bench = len(benchmarks)
    n_opt   = len(OPTIMIZERS)

    # --- Build rank matrix (benchmarks × optimizers) ---
    mean_matrix = np.full((n_bench, n_opt), np.nan)
    for j, opt in enumerate(OPTIMIZERS):
        for i in range(n_bench):
            mean_matrix[i, j] = opt_means[opt][i]

    rank_matrix = np.full_like(mean_matrix, np.nan)
    for i in range(n_bench):
        row = mean_matrix[i]
        valid = ~np.isnan(row)
        if valid.any():
            order = np.argsort(row[valid])
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(1, valid.sum() + 1)
            rank_matrix[i, valid] = ranks

    # --- Figure ---
    fig = plt.figure(figsize=(16, 10), constrained_layout=False)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.2], hspace=0.40)

    # ---- Top: grouped bar chart (mean final best, averaged across benchmarks) ----
    ax_bar = fig.add_subplot(gs[0])

    avg_across_bench = []
    std_across_bench = []
    colors_bar = []
    labels_bar = []
    for opt in OPTIMIZERS:
        vals = [v for v in opt_means[opt] if not np.isnan(v)]
        avg_across_bench.append(np.mean(vals) if vals else 0)
        std_across_bench.append(np.std(vals) if vals else 0)
        colors_bar.append(OPTIMIZER_COLORS[opt])
        labels_bar.append(OPTIMIZER_LABELS[opt])

    x_bar = np.arange(n_opt)
    bars = ax_bar.bar(x_bar, avg_across_bench, yerr=std_across_bench,
                      color=colors_bar, edgecolor='white', linewidth=0.8,
                      capsize=4, error_kw={'linewidth': 1.2})
    ax_bar.set_xticks(x_bar)
    ax_bar.set_xticklabels(labels_bar, fontsize=10)
    ax_bar.set_ylabel('Mean Final Best Cost\n(averaged across benchmarks)')
    ax_bar.set_title('Mean Final Best Cost per Optimizer', fontsize=13,
                     fontweight='bold')
    ax_bar.grid(axis='y', alpha=0.3, linewidth=0.5)
    ax_bar.tick_params(labelsize=10)

    # Annotate bars
    for bar, val in zip(bars, avg_across_bench):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.4f}', ha='center', va='bottom', fontsize=8,
                    fontweight='bold')

    # ---- Bottom: rank heatmap ----
    ax_hm = fig.add_subplot(gs[1])

    # rank_matrix shape: (n_bench, n_opt)  — rows=benchmarks, cols=optimizers
    # We want X=benchmarks, Y=optimizers  →  transpose
    rank_display = rank_matrix.T  # (n_opt, n_bench)

    im = ax_hm.imshow(rank_display, cmap='coolwarm', aspect='auto',
                      vmin=1, vmax=n_opt)
    ax_hm.set_xticks(np.arange(n_bench))
    ax_hm.set_xticklabels(benchmarks, rotation=40, ha='right', fontsize=9)
    ax_hm.set_yticks(np.arange(n_opt))
    ax_hm.set_yticklabels([OPTIMIZER_LABELS[o] for o in OPTIMIZERS],
                          fontsize=10)
    ax_hm.set_title('Optimizer Ranking per Benchmark (1 = best)', fontsize=13,
                     fontweight='bold')

    # Annotate cells
    for i in range(n_opt):
        for j in range(n_bench):
            val = rank_display[i, j]
            if not np.isnan(val):
                text_color = 'white' if val >= (n_opt / 2 + 1) else 'black'
                ax_hm.text(j, i, f'{int(val)}', ha='center', va='center',
                           fontsize=11, fontweight='bold', color=text_color)

    cbar = fig.colorbar(im, ax=ax_hm, shrink=0.6, pad=0.02)
    cbar.set_label('Rank', fontsize=10)

    fig.suptitle('Cross-Benchmark Performance Summary',
                 fontsize=15, fontweight='bold', y=1.01)

    os.makedirs(PLOTS_DIR, exist_ok=True)
    save_path = os.path.join(PLOTS_DIR, 'fig8_hpobench_summary.png')
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] Saved {save_path}")

    # ---- Console ranking table ----
    _print_ranking_table(benchmarks, rank_matrix)
    return True


def _print_ranking_table(benchmarks, rank_matrix):
    """Print a nicely formatted ranking table to stdout."""
    n_bench = len(benchmarks)
    n_opt   = len(OPTIMIZERS)

    col_w = max(len(b) for b in benchmarks) + 2
    opt_w = max(len(OPTIMIZER_LABELS[o]) for o in OPTIMIZERS) + 2

    header = f"{'Benchmark':<{col_w}}" + "".join(
        f"{OPTIMIZER_LABELS[o]:>{opt_w}}" for o in OPTIMIZERS
    )
    print("\n" + "=" * len(header))
    print("  OPTIMIZER RANKINGS  (1 = best)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for i, bname in enumerate(benchmarks):
        row = f"{bname:<{col_w}}"
        for j in range(n_opt):
            val = rank_matrix[i, j]
            cell = f"{int(val)}" if not np.isnan(val) else "—"
            row += f"{cell:>{opt_w}}"
        print(row)

    # Average rank
    avg_rank_row = f"{'Avg Rank':<{col_w}}"
    for j in range(n_opt):
        col = rank_matrix[:, j]
        valid = col[~np.isnan(col)]
        avg = np.mean(valid) if len(valid) > 0 else np.nan
        cell = f"{avg:.2f}" if not np.isnan(avg) else "—"
        avg_rank_row += f"{cell:>{opt_w}}"
    print("-" * len(header))
    print(avg_rank_row)
    print("=" * len(header) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print(f"Results directory : {RESULTS_DIR}")
    print(f"Output directory  : {PLOTS_DIR}\n")

    print("[Figure 6] HPOBench regret curves …")
    ok = plot_fig6()
    if not ok:
        print("  ⚠  No HPOBench data found — figure skipped.\n")

    print("[Figure 6b] HPOLib regret curves …")
    ok = plot_fig6b()
    if not ok:
        print("  ⚠  No HPOLib data found — figure skipped.\n")

    print("[Figure 7] NAS-Bench-201 regret curves …")
    ok = plot_fig7()
    if not ok:
        print("  ⚠  No NAS-Bench-201 data found — figure skipped.\n")

    print("[Figure 8] Cross-benchmark summary …")
    ok = plot_fig8()
    if not ok:
        print("  ⚠  No summary data — figure skipped.\n")

    print("Done.")


if __name__ == '__main__':
    main()
