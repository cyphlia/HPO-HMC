#!/usr/bin/env python
"""
Physics Benchmark Runner — Phase 4 (SIAM Revision).
======================================================

Runs all three methods (A, B, C) on all three Hamiltonian systems:
  1. Simple Harmonic Oscillator (2D)  — baseline
  2. Hénon-Heiles (4D, non-integrable, chaotic) — harder
  3. Double-Well (2D, bimodal potential) — harder

For each (method, system, seed) triplet the script:
  - Generates dataset via the appropriate data generator
  - Trains the model with the chosen method
  - Records final validation loss, best validation loss, and wall time
  - Saves individual JSON results and an aggregated summary table

Usage:
  python run_physics_benchmarks.py
  python run_physics_benchmarks.py --seeds 0,1,2 --epochs 80
  python run_physics_benchmarks.py --systems harmonic,henon_heiles
  python run_physics_benchmarks.py --methods A,C
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import config
from data_generator import (
    generate_hamiltonian_data,
    generate_henon_heiles_data,
    generate_double_well_data,
)
from train_hamiltonian import HamiltonianTrainer
from hybrid_adam_bfgs import HybridAdamBFGSTrainer
from hybrid_hhd_abbo_improved import ImprovedUnifiedTrainer

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYSTEMS = {
    "harmonic": {
        "gen_fn": generate_hamiltonian_data,
        "input_dim": 2,
        "label": "Harmonic Oscillator (2D)",
        "results_dir": "results_hamiltonian",
    },
    "henon_heiles": {
        "gen_fn": generate_henon_heiles_data,
        "input_dim": 4,
        "label": "Hénon-Heiles (4D)",
        "results_dir": "results_henon_heiles",
    },
    "double_well": {
        "gen_fn": generate_double_well_data,
        "input_dim": 2,
        "label": "Double-Well (2D)",
        "results_dir": "results_double_well",
    },
}

METHODS = {
    "A": "Method A (HHD)",
    "B": "Method B (Hybrid BO)",
    "C": "Method C (Unified)",
}

RESULTS_ROOT = Path("results_physics_benchmarks")


# ---------------------------------------------------------------------------
# Single-run helper
# ---------------------------------------------------------------------------

def run_single(
    system_name: str,
    method_key: str,
    seed: int,
    n_epochs: int = 80,
    n_warmup: int = 30,
    n_samples: int = 2000,
    device: str = "cpu",
) -> dict:
    """Run one (system, method, seed) triplet and return a result dict."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    sys_cfg = SYSTEMS[system_name]
    input_dim = sys_cfg["input_dim"]
    gen_fn = sys_cfg["gen_fn"]

    print(f"    [{system_name}][Method {method_key}] seed={seed} …", flush=True)

    # Generate data
    train_loader, val_loader, _ = gen_fn(n_samples=n_samples, seed=seed)

    t0 = time.time()

    if method_key == "A":
        trainer = HamiltonianTrainer(
            hyperparam_space=config.HYPERPARAM_SPACE,
            init_hyperparams=config.INIT_HYPERPARAMS,
            step_size=config.STEP_SIZE,
            n_leapfrog=config.N_LEAPFROG_STEPS,
            temperature=config.TEMPERATURE,
            device=device,
            input_dim=input_dim,
        )
        history = trainer.train(
            n_samples=n_samples, n_warmup=n_warmup, n_hamilton=n_epochs,
            train_loader=train_loader, val_loader=val_loader,
        )

    elif method_key == "B":
        trainer = HybridAdamBFGSTrainer(n_bo_trials=15, input_dim=input_dim)
        history = trainer.train(
            n_samples=n_samples,
            train_loader=train_loader, val_loader=val_loader,
        )

    else:  # method_key == "C"
        trainer = ImprovedUnifiedTrainer(
            hyperparam_space=config.HYPERPARAM_SPACE,
            init_hyperparams=config.INIT_HYPERPARAMS,
            initial_step=config.STEP_SIZE,
            n_leapfrog=config.N_LEAPFROG_STEPS,
            temperature=config.TEMPERATURE,
            device=device,
            input_dim=input_dim,
        )
        history = trainer.train(
            n_samples=n_samples, n_warmup=n_warmup, n_hamilton=n_epochs,
            train_loader=train_loader, val_loader=val_loader,
        )


    elapsed = time.time() - t0

    val_losses = history.get("val_loss", [])
    final_val = float(val_losses[-1]) if val_losses else float("inf")
    best_val  = float(min(val_losses)) if val_losses else float("inf")

    return {
        "system": system_name,
        "method": method_key,
        "seed": seed,
        "final_val_loss": final_val,
        "best_val_loss": best_val,
        "wall_time_s": round(elapsed, 2),
        "n_epochs": n_epochs,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(results: List[dict]) -> Dict[str, Dict[str, dict]]:
    """mean ± std per (system, method)."""
    from collections import defaultdict
    agg: Dict[Tuple, List[float]] = defaultdict(list)

    for r in results:
        key = (r["system"], r["method"])
        agg[key].append(r["best_val_loss"])

    summary = {}
    for (sys_name, method_key), vals in agg.items():
        summary.setdefault(sys_name, {})[method_key] = {
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals)),
            "n_seeds": len(vals),
            "values": vals,
        }
    return summary


def print_summary_table(summary: Dict[str, Dict[str, dict]]):
    """Print a compact mean ± std table."""
    method_keys = sorted({m for sys_data in summary.values() for m in sys_data})
    header = f"{'System':<25}" + "".join(f"  {METHODS[m]:>30}" for m in method_keys)

    print("\n" + "=" * (25 + 34 * len(method_keys)))
    print("  PHYSICS BENCHMARK RESULTS: best val loss (mean ± std)")
    print("=" * (25 + 34 * len(method_keys)))
    print(header)
    print("-" * (25 + 34 * len(method_keys)))

    for sys_name, sys_cfg in SYSTEMS.items():
        if sys_name not in summary:
            continue
        row = f"{sys_cfg['label']:<25}"
        for m in method_keys:
            if m in summary[sys_name]:
                d = summary[sys_name][m]
                row += f"  {d['mean']:.6f} ± {d['std']:.6f}              "
            else:
                row += f"  {'N/A':>30}"
        print(row)
    print("=" * (25 + 34 * len(method_keys)))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_all_physics_benchmarks(
    systems: Optional[List[str]] = None,
    methods: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    n_epochs: int = 80,
    n_warmup: int = 30,
    n_samples: int = 2000,
    device: str = "cpu",
) -> List[dict]:
    """Run all requested (system, method, seed) combinations."""
    if systems is None:
        systems = list(SYSTEMS.keys())
    if methods is None:
        methods = list(METHODS.keys())
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    all_results: List[dict] = []

    print("\n" + "=" * 70)
    print("  PHYSICS BENCHMARKS — Phase 4 (SIAM Revision)")
    print(f"  Systems : {', '.join(systems)}")
    print(f"  Methods : {', '.join(METHODS[m] for m in methods)}")
    print(f"  Seeds   : {seeds}")
    print("=" * 70)

    for sys_name in systems:
        if sys_name not in SYSTEMS:
            print(f"  WARNING: Unknown system '{sys_name}', skipping.")
            continue

        out_dir = RESULTS_ROOT / sys_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- System: {SYSTEMS[sys_name]['label']} ---")

        for method_key in methods:
            print(f"\n  Method {method_key}: {METHODS[method_key]}")
            for seed in seeds:
                result = run_single(
                    system_name=sys_name,
                    method_key=method_key,
                    seed=seed,
                    n_epochs=n_epochs,
                    n_warmup=n_warmup,
                    n_samples=n_samples,
                    device=device,
                )
                all_results.append(result)

                # Save individual result
                fname = out_dir / f"Method{method_key}_seed{seed}.json"
                with open(fname, "w") as f:
                    json.dump(result, f, indent=2)

    # Aggregate and report
    print("\n\n[Aggregation]")
    summary = aggregate(all_results)
    print_summary_table(summary)

    # Save summary JSON
    summary_path = RESULTS_ROOT / "physics_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "summary": summary,
            "all_results": all_results,
        }, f, indent=2, default=str)
    print(f"\n  [OK] Saved physics benchmark summary -> {summary_path}")

    print("\n" + "=" * 70)
    print("  PHYSICS BENCHMARKS COMPLETE")
    print("=" * 70)

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 4: Run harder physics benchmarks"
    )
    parser.add_argument("--systems", type=str, default="all",
                        help="Comma-separated systems: harmonic,henon_heiles,double_well or 'all'")
    parser.add_argument("--methods", type=str, default="A,B,C",
                        help="Comma-separated methods: A,B,C")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4",
                        help="Comma-separated seeds")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()

    systems = None if args.systems.lower() == "all" else [
        s.strip() for s in args.systems.split(",")
    ]
    methods = [m.strip() for m in args.methods.split(",")]
    seeds   = [int(s.strip()) for s in args.seeds.split(",")]

    run_all_physics_benchmarks(
        systems=systems,
        methods=methods,
        seeds=seeds,
        n_epochs=args.epochs,
        n_warmup=args.warmup,
        n_samples=args.n_samples,
        device=args.device,
    )


if __name__ == "__main__":
    main()
