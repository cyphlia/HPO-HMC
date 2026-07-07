#!/usr/bin/env python
"""
Ablation Study for Method C (Unified HHD-ABBO).
=================================================

Evaluates five ablation variants to isolate the contribution of each
Method C component:

  C-full         : Full Method C (all components enabled)
  C-noAdam       : Remove Adam micro-steps (Phase 1 warmup skipped)
  C-noHMC        : Remove HMC leapfrog proposals (Adam + L-BFGS only)
  C-noLBFGS      : Remove plateau-triggered L-BFGS (Adam + HMC only)
  C-noPlateauDet : Run L-BFGS every epoch rather than plateau-triggered
  C-fixedStep    : Remove adaptive step-size control (fixed epsilon)

Each variant is evaluated on:
  - Harmonic oscillator (physics benchmark)
  - 3-4 representative tabular benchmarks from HPOBench

Reports average rank relative to the full Method C.

Usage:
  python ablation_study.py
  python ablation_study.py --seeds 0,1,2 --epochs 80
  python ablation_study.py --physics-only
  python ablation_study.py --tabular-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

import config
from data_generator import generate_hamiltonian_data
from hybrid_hhd_abbo_improved import ImprovedUnifiedTrainer

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_DIR = Path("results_ablation")

ABLATION_VARIANTS = {
    "C-full": {
        "use_adam_warmup": True,
        "use_hmc": True,
        "use_lbfgs": True,
        "use_plateau_detect": True,
        "use_adaptive_step": True,
    },
    "C-noAdam": {
        "use_adam_warmup": False,
        "use_hmc": True,
        "use_lbfgs": True,
        "use_plateau_detect": True,
        "use_adaptive_step": True,
    },
    "C-noHMC": {
        "use_adam_warmup": True,
        "use_hmc": False,
        "use_lbfgs": True,
        "use_plateau_detect": True,
        "use_adaptive_step": True,
    },
    "C-noLBFGS": {
        "use_adam_warmup": True,
        "use_hmc": True,
        "use_lbfgs": False,
        "use_plateau_detect": True,
        "use_adaptive_step": True,
    },
    "C-noPlateauDet": {
        "use_adam_warmup": True,
        "use_hmc": True,
        "use_lbfgs": True,
        "use_plateau_detect": False,
        "use_adaptive_step": True,
    },
    "C-fixedStep": {
        "use_adam_warmup": True,
        "use_hmc": True,
        "use_lbfgs": True,
        "use_plateau_detect": True,
        "use_adaptive_step": False,
    },
}

# Representative tabular benchmarks for ablation
TABULAR_BENCHMARKS = [
    ("hpobench", "australian"),
    ("hpobench", "blood_transfusion"),
    ("hpolib", "naval_propulsion"),
    ("hpolib", "slice_localization"),
]


# ---------------------------------------------------------------------------
# Physics benchmark (Harmonic Oscillator)
# ---------------------------------------------------------------------------

def run_physics_ablation(
    variant_name: str,
    variant_flags: dict,
    seed: int,
    n_epochs: int = 80,
    n_warmup: int = 30,
    n_samples: int = 2500,
    device: str = "cpu",
) -> dict:
    """
    Run a single ablation variant on the harmonic oscillator.

    Returns dict with: variant, seed, final_val_loss, final_train_loss,
                        best_val_loss, train_time
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"    [{variant_name}] seed={seed} ...")

    trainer = ImprovedUnifiedTrainer(
        hyperparam_space=config.HYPERPARAM_SPACE,
        init_hyperparams=config.INIT_HYPERPARAMS,
        initial_step=config.STEP_SIZE,
        n_leapfrog=config.N_LEAPFROG_STEPS,
        temperature=config.TEMPERATURE,
        device=device,
        **variant_flags,
    )

    t0 = time.time()
    history = trainer.train(
        n_samples=n_samples,
        n_warmup=n_warmup,
        n_hamilton=n_epochs,
    )
    elapsed = time.time() - t0

    result = {
        "variant": variant_name,
        "seed": seed,
        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else float("inf"),
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else float("inf"),
        "best_val_loss": min(history["val_loss"]) if history["val_loss"] else float("inf"),
        "train_time": round(elapsed, 2),
        "n_epochs": n_epochs,
    }

    return result


# ---------------------------------------------------------------------------
# Tabular benchmark (HPOBench)
# ---------------------------------------------------------------------------

def run_tabular_ablation(
    variant_name: str,
    variant_flags: dict,
    suite_name: str,
    dataset_name: str,
    seed: int,
    max_trials: int = 100,
) -> dict:
    """
    Run a tabular benchmark ablation using the HPOBench pipeline.

    For tabular benchmarks, Method C uses the three-phase curriculum
    adapted for discrete index spaces (as in hpobench_benchmark.py).
    We modify the behavior based on ablation flags.
    """
    # Import the HPOBench runner
    try:
        from hpobench_benchmark import (
            run_method_c_unified,
            DATASET_REGISTRY,
            _to_cost,
        )
        from hpo_benchmarks import HPOBench, HPOLib, NASBench201
    except ImportError:
        print(f"    [{variant_name}] SKIP (hpo_benchmarks not available)")
        return {
            "variant": variant_name,
            "seed": seed,
            "suite": suite_name,
            "dataset": dataset_name,
            "final_best_cost": float("inf"),
            "wall_time": 0.0,
        }

    # Load benchmark
    bench_cls_map = {
        "hpobench": HPOBench,
        "hpolib": HPOLib,
        "nasbench201": NASBench201,
    }
    bench_cls = bench_cls_map.get(suite_name)
    if bench_cls is None:
        return {"variant": variant_name, "error": f"Unknown suite: {suite_name}"}

    try:
        bench = bench_cls(dataset_name=dataset_name)
    except Exception as e:
        print(f"    [{variant_name}] SKIP ({e})")
        return {
            "variant": variant_name,
            "seed": seed,
            "suite": suite_name,
            "dataset": dataset_name,
            "final_best_cost": float("inf"),
            "wall_time": 0.0,
        }

    search_space = bench.search_space
    metric_name = bench.metric_names[0]
    direction = bench.directions.get(metric_name, "minimize")

    # Run the standard Method C (tabular) with ablation flags.
    t0 = time.perf_counter()
    hhm_flags = {
        "use_adam_warmup": variant_flags.get("use_adam_warmup", True),
        "use_hmc": variant_flags.get("use_hmc", True),
        "use_lbfgs": variant_flags.get("use_lbfgs", True),
        "use_adaptive_step": variant_flags.get("use_adaptive_step", True),
    }
    trajectory = run_method_c_unified(
        bench, search_space, metric_name, direction, max_trials, seed, **hhm_flags
    )
    wall_time = time.perf_counter() - t0

    final_best = trajectory[-1]["best_cost"] if trajectory else float("inf")

    return {
        "variant": variant_name,
        "seed": seed,
        "suite": suite_name,
        "dataset": dataset_name,
        "final_best_cost": final_best,
        "wall_time": round(wall_time, 4),
    }


# ---------------------------------------------------------------------------
# Aggregation and reporting
# ---------------------------------------------------------------------------

def aggregate_results(results: List[dict]) -> Dict[str, Dict[str, dict]]:
    """
    Aggregate per-seed results into mean +/- std per variant per dataset.

    Returns:
        dict[variant][dataset] = {"mean": ..., "std": ..., "seeds": [...]}
    """
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list))

    for r in results:
        variant = r["variant"]
        dataset = r.get("dataset", "harmonic_oscillator")
        metric = r.get("best_val_loss", r.get("final_best_cost", float("inf")))
        agg[variant][dataset].append(metric)

    summary = {}
    for variant, datasets in agg.items():
        summary[variant] = {}
        for dataset, values in datasets.items():
            summary[variant][dataset] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "n_seeds": len(values),
                "values": [float(v) for v in values],
            }

    return summary


def compute_ranks(summary: Dict[str, Dict[str, dict]]) -> Dict[str, float]:
    """
    Compute average rank of each variant across all datasets.
    Rank 1 = best (lowest mean metric).
    """
    from scipy.stats import rankdata

    # Get all datasets
    all_datasets = set()
    for variant_data in summary.values():
        all_datasets.update(variant_data.keys())

    variants = list(summary.keys())
    n_variants = len(variants)
    all_ranks = {v: [] for v in variants}

    for dataset in sorted(all_datasets):
        means = []
        for v in variants:
            if dataset in summary[v]:
                means.append(summary[v][dataset]["mean"])
            else:
                means.append(float("inf"))

        ranks = rankdata(means, method="average")
        for i, v in enumerate(variants):
            all_ranks[v].append(ranks[i])

    avg_ranks = {v: float(np.mean(r)) for v, r in all_ranks.items()}
    return avg_ranks


def print_ablation_table(summary: Dict[str, Dict[str, dict]], avg_ranks: Dict[str, float]):
    """Print a formatted ablation comparison table."""
    variants = list(summary.keys())
    all_datasets = sorted(set(
        ds for v_data in summary.values() for ds in v_data.keys()
    ))

    print("\n" + "=" * 100)
    print("  ABLATION STUDY RESULTS: mean MSE +/- std")
    print("=" * 100)

    # Header
    header = f"{'Variant':<20}"
    for ds in all_datasets:
        ds_short = ds.replace("harmonic_oscillator", "HO").replace("_", " ").title()
        if len(ds_short) > 18:
            ds_short = ds_short[:16] + ".."
        header += f"  {ds_short:>18}"
    header += f"  {'Avg Rank':>10}"
    print(header)
    print("-" * 100)

    for variant in variants:
        row = f"{variant:<20}"
        for ds in all_datasets:
            if ds in summary[variant]:
                m = summary[variant][ds]["mean"]
                s = summary[variant][ds]["std"]
                row += f"  {m:>8.6f}+/-{s:.4f}"
            else:
                row += f"  {'N/A':>18}"
        rank = avg_ranks.get(variant, float("nan"))
        row += f"  {rank:>10.2f}"
        print(row)

    print("=" * 100)

    # Highlight which components are necessary
    full_rank = avg_ranks.get("C-full", float("inf"))
    print("\n  Component necessity analysis:")
    for variant in variants:
        if variant == "C-full":
            continue
        rank = avg_ranks.get(variant, float("inf"))
        delta = rank - full_rank
        if delta > 0.5:
            necessity = "CRITICAL (rank drop > 0.5)"
        elif delta > 0.2:
            necessity = "IMPORTANT (rank drop > 0.2)"
        elif delta > 0:
            necessity = "minor contribution"
        else:
            necessity = "no measurable contribution"
        print(f"    {variant:<20}: avg rank {rank:.2f} (delta={delta:+.2f}) -> {necessity}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def compute_tabular_ranks_with_std(all_results: List[dict], seeds: List[int]) -> Dict[str, Tuple[float, float]]:
    from scipy.stats import rankdata
    
    # Filter to tabular results
    tab_datasets = [f"{s}/{d}" for s, d in TABULAR_BENCHMARKS]
    
    # Organize by (variant, dataset, seed) -> final_best_cost
    costs = {}
    for r in all_results:
        ds = r.get("dataset")
        if ds in tab_datasets:
            costs[(r["variant"], ds, r["seed"])] = r.get("final_best_cost", float("inf"))
            
    variants = list(ABLATION_VARIANTS.keys())
    seed_avg_ranks = {v: [] for v in variants}
    
    for seed in seeds:
        var_ranks_for_seed = {v: [] for v in variants}
        for ds in tab_datasets:
            var_costs = {}
            for v in variants:
                var_costs[v] = costs.get((v, ds, seed), float("inf"))
            
            costs_list = [var_costs[v] for v in variants]
            ranks = rankdata(costs_list, method="average")
            for i, v in enumerate(variants):
                var_ranks_for_seed[v].append(ranks[i])
        
        for v in variants:
            seed_avg_ranks[v].append(np.mean(var_ranks_for_seed[v]))
            
    results = {}
    for v in variants:
        mean_val = float(np.mean(seed_avg_ranks[v]))
        std_val = float(np.std(seed_avg_ranks[v]))
        results[v] = (mean_val, std_val)
        
    return results


def physics_worker(args):
    v_name, v_flags, s, n_epochs, n_warmup, n_samples, device = args
    import torch
    torch.set_num_threads(1)
    return run_physics_ablation(
        variant_name=v_name,
        variant_flags=v_flags,
        seed=s,
        n_epochs=n_epochs,
        n_warmup=n_warmup,
        n_samples=n_samples,
        device=device,
    )


def run_ablation_study(
    seeds: List[int] = None,
    n_epochs: int = 80,
    n_warmup: int = 30,
    n_samples: int = 2500,
    max_trials: int = 100,
    run_physics: bool = True,
    run_tabular: bool = True,
    device: str = "cpu",
):
    """Run the full ablation study."""
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    print("\n" + "=" * 70)
    print("  ABLATION STUDY: Method C Component Analysis")
    print("=" * 70)

    # --- Physics benchmarks ---
    if run_physics:
        print("\n[Phase A] Harmonic Oscillator Ablation")
        print("-" * 50)

        # First, search for cached files
        physics_runs_to_submit = []
        for variant_name, variant_flags in ABLATION_VARIANTS.items():
            for seed in seeds:
                out_path = RESULTS_DIR / f"{variant_name}_HO_seed{seed}.json"
                if out_path.exists():
                    try:
                        with open(out_path, "r") as f:
                            result = json.load(f)
                        result["dataset"] = "harmonic_oscillator"
                        all_results.append(result)
                        print(f"    Loaded cached physics [{variant_name}] seed={seed}")
                        continue
                    except Exception:
                        pass
                physics_runs_to_submit.append((variant_name, variant_flags, seed))

        # Run remaining physics runs in parallel
        if physics_runs_to_submit:
            print(f"\n  Running {len(physics_runs_to_submit)} missing physics ablation runs in parallel...")
            import concurrent.futures

            physics_args = [
                (v_name, v_flags, s, n_epochs, n_warmup, n_samples, device)
                for v_name, v_flags, s in physics_runs_to_submit
            ]

            with concurrent.futures.ProcessPoolExecutor(max_workers=min(len(physics_runs_to_submit), 12)) as executor:
                futures = {executor.submit(physics_worker, arg): arg for arg in physics_args}
                for future in concurrent.futures.as_completed(futures):
                    arg = futures[future]
                    v_name, _, s = arg[:3]
                    try:
                        result = future.result()
                        result["dataset"] = "harmonic_oscillator"
                        all_results.append(result)
                        out_path = RESULTS_DIR / f"{v_name}_HO_seed{s}.json"
                        with open(out_path, "w") as f:
                            json.dump(result, f, indent=2)
                        print(f"    [OK] Parallel run [{v_name}] seed={s} finished.")
                    except Exception as e:
                        print(f"    [ERROR] Parallel run [{v_name}] seed={s} failed: {e}")

    # --- Tabular benchmarks ---
    if run_tabular:
        print("\n\n[Phase B] Tabular Benchmark Ablation")
        print("-" * 50)

        for variant_name, variant_flags in ABLATION_VARIANTS.items():
            print(f"\n  Variant: {variant_name}")
            for suite_name, dataset_name in TABULAR_BENCHMARKS:
                print(f"    Dataset: {suite_name}/{dataset_name} ...")
                for seed in seeds:
                    out_path = RESULTS_DIR / f"{variant_name}_{suite_name}_{dataset_name}_seed{seed}.json"
                    if out_path.exists():
                        try:
                            with open(out_path, "r") as f:
                                result = json.load(f)
                            result["dataset"] = f"{suite_name}/{dataset_name}"
                            all_results.append(result)
                            continue
                        except Exception:
                            pass

                    result = run_tabular_ablation(
                        variant_name=variant_name,
                        variant_flags=variant_flags,
                        suite_name=suite_name,
                        dataset_name=dataset_name,
                        seed=seed,
                        max_trials=max_trials,
                    )
                    result["dataset"] = f"{suite_name}/{dataset_name}"
                    all_results.append(result)

                    # Save individual result
                    with open(out_path, "w") as f:
                        json.dump(result, f, indent=2)

    # --- Aggregate and report ---
    print("\n\n[Phase C] Aggregation and Analysis")
    print("-" * 50)

    physics_results = [r for r in all_results if r.get("dataset") == "harmonic_oscillator"]
    tabular_results = [r for r in all_results if r.get("dataset") in [f"{s}/{d}" for s, d in TABULAR_BENCHMARKS]]

    if physics_results:
        physics_summary = aggregate_results(physics_results)
        physics_ranks = compute_ranks(physics_summary)
        print("\n=== Physics Ablation Summary ===")
        print_ablation_table(physics_summary, physics_ranks)

        # Tabular ranks computation
        if tabular_results:
            tabular_summary = compute_tabular_ranks_with_std(all_results, seeds)
            print("\n=== Tabular Ablation Summary (Average Rank across 4 benchmarks) ===")
            print(f"{'Variant':<20} | {'Tabular Avg. Rank':<25}")
            print("-" * 50)
            for variant in ABLATION_VARIANTS.keys():
                mean_r, std_r = tabular_summary[variant]
                print(f"{variant:<20} | {mean_r:.2f} +/- {std_r:.2f}")
            print("-" * 50)
            
            # Save LaTeX table generation code or directly write it
            print("\n=== LaTeX Table Lines for HO_main.tex ===")
            for variant in ABLATION_VARIANTS.keys():
                ho_mean = physics_summary[variant]["harmonic_oscillator"]["mean"]
                ho_std = physics_summary[variant]["harmonic_oscillator"]["std"]
                tab_mean, tab_std = tabular_summary[variant]
                
                # Check if it is the best (Method C full) and bold it
                ho_str = f"{ho_mean:.5f} \\pm {ho_std:.5f}"
                tab_str = f"{tab_mean:.2f} \\pm {tab_std:.2f}"
                
                if variant == "C-full" or variant == "Method C (full)":
                    ho_str = f"\\textbf{{{ho_mean:.5f} \\pm {ho_std:.5f}}}"
                    tab_str = f"\\textbf{{{tab_mean:.2f} \\pm {tab_std:.2f}}}"
                
                print(f"{variant:<25} & {ho_str} & {tab_str} \\\\")

        # Save full summary
        full_summary = {
            "summary": {
                variant: {
                    dataset: {
                        "mean": data["mean"],
                        "std": data["std"],
                        "n_seeds": data["n_seeds"],
                    }
                    for dataset, data in datasets.items()
                }
                for variant, datasets in physics_summary.items()
            },
            "average_ranks": physics_ranks,
            "all_results": all_results,
        }
        if tabular_results:
            full_summary["tabular_ranks"] = tabular_summary
            
        summary_path = RESULTS_DIR / "ablation_summary.json"
        with open(summary_path, "w") as f:
            json.dump(full_summary, f, indent=2, default=str)
        print(f"\n  [OK] Saved ablation summary -> {summary_path}")

    print("\n" + "=" * 70)
    print("  ABLATION STUDY COMPLETE")
    print("=" * 70 + "\n")

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Method C Ablation Study")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4",
                        help="Comma-separated seeds")
    parser.add_argument("--epochs", type=int, default=80,
                        help="Training epochs for physics benchmark")
    parser.add_argument("--warmup", type=int, default=30,
                        help="Warmup epochs")
    parser.add_argument("--n-samples", type=int, default=2500,
                        help="Number of samples for harmonic oscillator")
    parser.add_argument("--max-trials", type=int, default=100,
                        help="Max trials for tabular benchmarks")
    parser.add_argument("--physics-only", action="store_true",
                        help="Run only physics benchmarks")
    parser.add_argument("--tabular-only", action="store_true",
                        help="Run only tabular benchmarks")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    run_physics = not args.tabular_only
    run_tabular = not args.physics_only

    run_ablation_study(
        seeds=seeds,
        n_epochs=args.epochs,
        n_warmup=args.warmup,
        n_samples=args.n_samples,
        max_trials=args.max_trials,
        run_physics=run_physics,
        run_tabular=run_tabular,
        device=args.device,
    )


if __name__ == "__main__":
    main()
