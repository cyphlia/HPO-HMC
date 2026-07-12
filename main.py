"""
Hamiltonian Hyperparameter Dynamics - Main Entry Point.

Self-Tuning Neural Networks via Symplectic Integration.

Usage:
  # Run all harmonic oscillator methods and evaluate
  python main.py --task harmonic --compare

  # Run single method
  python main.py --task harmonic --method pure

  # Run CNN benchmark
  python main.py --task cnn

  # Run performance testbed (Method C optimized)
  python main.py --task testbed

  # Run harder physics benchmarks (Phase 4)
  python main.py --task henon_heiles --method improved
  python main.py --task double_well  --method improved
  python main.py --task physics_all           # all 3 methods on all 3 systems

  # Evaluate saved results only
  python main.py --evaluate-only

  # Full pipeline (both benchmarks + evaluation)
  python main.py --full
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import numpy as np
import torch
# Inject subdirectory paths to sys.path so we can import modules seamlessly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'scripts')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'evaluation')))

import config
from train_hamiltonian import HamiltonianTrainer
from hybrid_adam_bfgs import HybridAdamBFGSTrainer
from hybrid_hhd_abbo_improved import ImprovedUnifiedTrainer


BANNER = """
======================================================================
  Hamiltonian Hyperparameter Dynamics (HHD)
  Self-Tuning Neural Networks via Symplectic Integration

  H(theta, p_theta, lambda, p_lambda) = T(p) + V(theta, lambda)

  Method A: Pure HHD       (HMC co-evolution)
  Method B: Hybrid BO      (GP + Adam + L-BFGS)
  Method C: Unified HHD-ABBO (three-phase curriculum)
======================================================================
"""


def parse_args():
    p = argparse.ArgumentParser(description="Hamiltonian Hyperparameter Dynamics")

    p.add_argument("--task", type=str, default="harmonic",
                   choices=["harmonic", "cnn", "testbed", "hpobench", "both",
                            "henon_heiles", "double_well", "physics_all"],
                   help="Benchmark task")
    p.add_argument("--method", type=str, default="improved",
                   choices=["pure", "hybrid", "improved"],
                   help="Method: pure (A), hybrid (B), improved (C)")
    p.add_argument("--compare", action="store_true",
                   help="Run all 3 methods and compare")
    p.add_argument("--full", action="store_true",
                   help="Run full pipeline: both tasks + evaluation")
    p.add_argument("--evaluate-only", action="store_true",
                   help="Only evaluate saved results")

    # Training params
    p.add_argument("--epochs", type=int, default=config.N_EPOCHS)
    p.add_argument("--warmup", type=int, default=config.N_WARMUP_EPOCHS)
    p.add_argument("--n-samples", type=int, default=config.N_SAMPLES)
    p.add_argument("--step-size", type=float, default=config.STEP_SIZE)
    p.add_argument("--n-leapfrog", type=int, default=config.N_LEAPFROG_STEPS)
    p.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--device", type=str, default="cpu",
                   choices=["cpu", "cuda", "mps"])

    return p.parse_args()


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        print("  CUDA unavailable, falling back to CPU")
        return "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        print("  MPS unavailable, falling back to CPU")
        return "cpu"
    return requested


def run_harmonic_comparison(args, device):
    """Run all 3 methods on harmonic oscillator and save results."""
    print("\n" + "=" * 70)
    print("  HARMONIC OSCILLATOR: 3-WAY COMPARISON")
    print("=" * 70)

    t_total = time.time()

    # Method A: Pure HHD
    print("\n--- Method A: Pure HHD ---")
    set_seeds(args.seed)
    trainerA = HamiltonianTrainer(
        hyperparam_space=config.HYPERPARAM_SPACE,
        init_hyperparams=config.INIT_HYPERPARAMS,
        step_size=args.step_size,
        n_leapfrog=args.n_leapfrog,
        temperature=args.temperature,
        device=device,
    )
    trainerA.train(n_samples=args.n_samples, n_warmup=args.warmup, n_hamilton=args.epochs)
    trainerA.save("results_hamiltonian")

    # Method B: Hybrid BO
    print("\n--- Method B: Hybrid BO ---")
    set_seeds(args.seed)
    trainerB = HybridAdamBFGSTrainer(n_bo_trials=15)
    trainerB.train(n_samples=args.n_samples)
    trainerB.save("results_hybrid")

    # Method C: Unified
    print("\n--- Method C: Unified HHD-ABBO ---")
    set_seeds(args.seed)
    trainerC = ImprovedUnifiedTrainer(
        hyperparam_space=config.HYPERPARAM_SPACE,
        init_hyperparams=config.INIT_HYPERPARAMS,
        initial_step=args.step_size,
        n_leapfrog=args.n_leapfrog,
        temperature=args.temperature,
        device=device,
    )
    trainerC.train(n_samples=args.n_samples, n_warmup=args.warmup, n_hamilton=args.epochs)
    trainerC.save("results_unified_improved")

    elapsed = time.time() - t_total
    print(f"\n  Total comparison time: {elapsed:.1f}s")


def run_single_harmonic(args, device):
    """Run a single method on harmonic oscillator. Returns (method_key, save_dir)
    so the caller can avoid mixing this run's results with stale results from
    other methods left over in results_hybrid/ or results_unified_improved/
    (see BUGFIX note in main() below)."""
    method_map = {
        "pure":     ("A", "Method A: Pure HHD", HamiltonianTrainer, "results_hamiltonian"),
        "hybrid":   ("B", "Method B: Hybrid BO", HybridAdamBFGSTrainer, "results_hybrid"),
        "improved": ("C", "Method C: Unified HHD-ABBO", ImprovedUnifiedTrainer, "results_unified_improved"),
    }

    key, name, cls, save_dir = method_map[args.method]
    print(f"\n--- {name} ---")

    if args.method == "hybrid":
        trainer = cls(n_bo_trials=15)
        trainer.train(n_samples=args.n_samples)
    elif args.method == "pure":
        trainer = cls(
            hyperparam_space=config.HYPERPARAM_SPACE,
            init_hyperparams=config.INIT_HYPERPARAMS,
            step_size=args.step_size,
            n_leapfrog=args.n_leapfrog,
            temperature=args.temperature,
            device=device,
        )
        trainer.train(n_samples=args.n_samples, n_warmup=args.warmup, n_hamilton=args.epochs)
    else:
        trainer = cls(
            hyperparam_space=config.HYPERPARAM_SPACE,
            init_hyperparams=config.INIT_HYPERPARAMS,
            initial_step=args.step_size,
            n_leapfrog=args.n_leapfrog,
            temperature=args.temperature,
            device=device,
        )
        trainer.train(n_samples=args.n_samples, n_warmup=args.warmup, n_hamilton=args.epochs)

    trainer.save(save_dir)
    return key, save_dir


def run_physics_task(args, device, task: str):
    """Run a single physics benchmark system (harmonic, henon_heiles, double_well)."""
    from data_generator import (
        generate_hamiltonian_data,
        generate_henon_heiles_data,
        generate_double_well_data,
    )
    from hamiltonian import HamiltonianNN

    TASK_CFG = {
        "harmonic":     {"gen_fn": generate_hamiltonian_data,    "input_dim": 2,
                         "name": "Harmonic Oscillator",          "save_dir": "results_hamiltonian"},
        "henon_heiles": {"gen_fn": generate_henon_heiles_data,   "input_dim": 4,
                         "name": "H\u00e9non-Heiles",             "save_dir": "results_henon_heiles"},
        "double_well":  {"gen_fn": generate_double_well_data,    "input_dim": 2,
                         "name": "Double-Well",                  "save_dir": "results_double_well"},
    }
    cfg = TASK_CFG[task]
    print(f"\n{'=' * 70}")
    print(f"  PHYSICS BENCHMARK: {cfg['name']}  [Method: {args.method}]")
    print(f"{'=' * 70}")

    set_seeds(args.seed)
    train_loader, val_loader, _ = cfg["gen_fn"](n_samples=args.n_samples, seed=args.seed)

    input_dim = cfg["input_dim"]

    if args.method == "pure":
        trainer = HamiltonianTrainer(
            hyperparam_space=config.HYPERPARAM_SPACE,
            init_hyperparams=config.INIT_HYPERPARAMS,
            step_size=args.step_size,
            n_leapfrog=args.n_leapfrog,
            temperature=args.temperature,
            device=device,
            input_dim=input_dim,
        )
        trainer.train(n_samples=args.n_samples, n_warmup=args.warmup, n_hamilton=args.epochs)
    elif args.method == "hybrid":
        trainer = HybridAdamBFGSTrainer(n_bo_trials=15, input_dim=input_dim)
        trainer.train(n_samples=args.n_samples)
    else:
        trainer = ImprovedUnifiedTrainer(
            hyperparam_space=config.HYPERPARAM_SPACE,
            init_hyperparams=config.INIT_HYPERPARAMS,
            initial_step=args.step_size,
            n_leapfrog=args.n_leapfrog,
            temperature=args.temperature,
            device=device,
            input_dim=input_dim,
        )
        trainer.train(n_samples=args.n_samples, n_warmup=args.warmup, n_hamilton=args.epochs)

    import os
    os.makedirs(cfg["save_dir"], exist_ok=True)
    trainer.save(cfg["save_dir"])
    print(f"  Results saved to {cfg['save_dir']}/")


def main():
    print(BANNER)
    args   = parse_args()
    device = pick_device(args.device)
    set_seeds(args.seed)

    # Full pipeline
    if args.full:
        run_harmonic_comparison(args, device)

        print("\n" + "=" * 70)
        print("  CNN BENCHMARK")
        print("=" * 70)
        from cnn_benchmark import run_cnn_benchmark
        run_cnn_benchmark()

        print("\n" + "=" * 70)
        print("  PERFORMANCE TESTBED")
        print("=" * 70)
        from performance_testbed import run_testbed
        run_testbed()

        from evaluate import run_full_evaluation
        run_full_evaluation()
        return

    # Evaluate only
    if args.evaluate_only:
        from evaluate import run_full_evaluation
        run_full_evaluation()
        return

    # Task-specific execution
    if args.task == "testbed":
        from performance_testbed import run_testbed
        run_testbed()
        return

    if args.task == "hpobench":
        from hpobench_benchmark import run_full_hpobench_pipeline
        run_full_hpobench_pipeline()
        return

    if args.task in ("henon_heiles", "double_well"):
        run_physics_task(args, device, args.task)
        return

    if args.task == "physics_all":
        from run_physics_benchmarks import run_all_physics_benchmarks
        run_all_physics_benchmarks(
            seeds=[args.seed],
            n_epochs=args.epochs,
            n_warmup=args.warmup,
            n_samples=args.n_samples,
            device=device,
        )
        return

    if args.task in ("harmonic", "both"):
        if args.compare:
            run_harmonic_comparison(args, device)
            from evaluate import evaluate_harmonic
            evaluate_harmonic()
        else:
            # BUGFIX: previously this branch called evaluate_harmonic() with
            # no arguments, which unconditionally reads results_hamiltonian/,
            # results_hybrid/, AND results_unified_improved/ from disk and
            # prints a 3-way comparison table -- even though only ONE method
            # was actually just trained. Any leftover results from a
            # different previous run silently appeared in the table, with
            # nothing to indicate they weren't from this run. Now we only
            # evaluate the method that was actually just trained, and say so.
            method_key, save_dir = run_single_harmonic(args, device)
            print(f"\n  [NOTE] Only Method {method_key} was trained this run; "
                  f"evaluating {save_dir} in isolation (not a 3-way comparison, "
                  f"even if other results_* directories exist on disk).")
            from evaluate import evaluate_harmonic
            evaluate_harmonic(results_dirs={method_key: save_dir})

    if args.task in ("cnn", "both"):
        from cnn_benchmark import run_cnn_benchmark
        run_cnn_benchmark()

        from evaluate import evaluate_cnn
        evaluate_cnn()

    print(f"\n{'=' * 70}")
    print("  Done! Check results/ for output files.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
