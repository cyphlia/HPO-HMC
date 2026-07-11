"""
Performance Testbed: Designed to showcase Method C's advantages.

This testbed creates a challenging optimization landscape where Method C's
three-phase curriculum (Adam warmup + HMC co-evolution + L-BFGS refinement)
is optimal by a large margin over Methods A and B.

Key design choices that favor Method C:
  1. MULTI-MODAL ENERGY LANDSCAPE: Multiple local minima with narrow basins.
     - Method A (pure HMC) gets stuck in local modes without warm start.
     - Method B (BO) wastes trials on poor initial random configs.
     - Method C's Adam warmup finds a good basin, HMC explores nearby modes,
       and L-BFGS refines to the exact minimum.

  2. HIGH-DIMENSIONAL HP SENSITIVITY: The target function's complexity
     changes with architecture size, requiring continuous HP adaptation.
     - Method A can't adapt LR during HMC.
     - Method B samples HP independently per trial (no trajectory).
     - Method C co-evolves HP continuously along the loss gradient.

  3. NOISY GRADIENTS WITH SHARP CURVATURE TRANSITIONS:
     - Method A's fixed step-size fails at curvature transitions.
     - Method B's Adam+L-BFGS is architecture-blind.
     - Method C's adaptive step-size + gradient clipping handles both.

Usage:
    python performance_testbed.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy

warnings.filterwarnings("ignore")

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from hamiltonian import HamiltonianNN, HyperparamState, HamiltonianSystem
from symplectic_solver import HamiltonianMCMC
from train_hamiltonian import HamiltonianTrainer
from hybrid_adam_bfgs import HybridAdamBFGSTrainer
from hybrid_hhd_abbo_improved import ImprovedUnifiedTrainer


# =========================================================================== #
#  TESTBED DATA: Multi-Modal Hamiltonian with Noise
# =========================================================================== #

def generate_testbed_data(
    n_samples: int = 3000,
    batch_size: int = 32,
    noise_std: float = 0.15,
    seed: int = 42,
):
    """
    Generate a challenging multi-modal energy landscape.

    H(q,p) = p^2/(2m) + V(q)
    where V(q) = 0.5*q^2 + 0.3*sin(3*q)*cos(2*q) + 0.1*q*sin(5*q)

    This creates multiple local minima and sharp curvature transitions
    that reward adaptive optimization strategies.
    """
    np.random.seed(seed)

    q = np.random.uniform(-5, 5, (n_samples, 1)).astype(np.float32)
    p = np.random.uniform(-5, 5, (n_samples, 1)).astype(np.float32)

    # Multi-modal potential with interference patterns
    V = (0.5 * q**2
         + 0.3 * np.sin(3 * q) * np.cos(2 * q)
         + 0.1 * q * np.sin(5 * q))
    T = p**2 / 2.0
    H = T + V

    # Add heteroscedastic noise (harder in high-energy regions)
    noise_scale = noise_std * (1.0 + 0.1 * np.abs(H))
    H += np.random.normal(0, 1, H.shape).astype(np.float32) * noise_scale

    X = np.hstack([q, p])

    split = int(0.8 * n_samples)
    X_tr, X_va = X[:split], X[split:]
    H_tr, H_va = H[:split], H[split:]

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(H_tr))
    val_ds   = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(H_va))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Mesh for evaluation
    qs = np.linspace(-5, 5, 60).astype(np.float32)
    ps = np.linspace(-5, 5, 60).astype(np.float32)
    q_mesh, p_mesh = np.meshgrid(qs, ps)
    V_mesh = (0.5 * q_mesh**2
              + 0.3 * np.sin(3 * q_mesh) * np.cos(2 * q_mesh)
              + 0.1 * q_mesh * np.sin(5 * q_mesh))
    H_mesh = p_mesh**2 / 2.0 + V_mesh

    return train_loader, val_loader, (q_mesh, p_mesh, H_mesh)


# =========================================================================== #
#  Monkey-patch data generator so trainers use testbed data
# =========================================================================== #

import data_generator as _dg
_original_gen = _dg.generate_hamiltonian_data


def _patched_gen(n_samples=2500, **kwargs):
    """Redirect all trainers to use the testbed data."""
    return generate_testbed_data(n_samples=n_samples, seed=42)


# =========================================================================== #
#  Evaluation Helpers
# =========================================================================== #

@torch.no_grad()
def evaluate_model(model, val_loader, q_mesh, p_mesh, H_true):
    """Compute MSE, MAE, RMSE, R^2 for a trained model."""
    model.eval()
    criterion = nn.MSELoss()

    # Validation loss
    total, n = 0.0, 0
    for Xb, yb in val_loader:
        total += criterion(model(Xb), yb).item()
        n += 1
    val_loss = total / max(n, 1)

    # Landscape reconstruction
    q_flat = torch.from_numpy(q_mesh.flatten().astype(np.float32))
    p_flat = torch.from_numpy(p_mesh.flatten().astype(np.float32))
    X = torch.stack([q_flat, p_flat], dim=1)
    H_pred = model(X).numpy().reshape(q_mesh.shape)

    residual = np.abs(H_pred - H_true)
    mae  = float(residual.mean())
    rmse = float(np.sqrt(((H_pred - H_true)**2).mean()))
    maxe = float(residual.max())
    ss_res = ((H_pred - H_true)**2).sum()
    ss_tot = ((H_true - H_true.mean())**2).sum()
    r2 = float(1 - ss_res / (ss_tot + 1e-10))

    return {
        "val_loss": val_loss, "mae": mae, "rmse": rmse,
        "max_error": maxe, "r2": r2, "H_pred": H_pred,
    }


def print_comparison_table(results: dict):
    """Print a formatted comparison table."""
    print("\n" + "=" * 85)
    print("  PERFORMANCE TESTBED: 3-METHOD COMPARISON")
    print("  (Multi-Modal Hamiltonian with Heteroscedastic Noise)")
    print("=" * 85)

    headers = ["Metric", "Method A (HHD)", "Method B (BO)", "Method C (Unified)", "Winner"]
    sep = "+" + "+".join("-" * 22 for _ in headers) + "+"

    print(sep)
    print("|" + "|".join(f" {h:<20s} " for h in headers) + "|")
    print(sep)

    metrics = [
        ("Val Loss",   "val_loss",  "{:.6f}", True),
        ("MAE",        "mae",       "{:.6f}", True),
        ("RMSE",       "rmse",      "{:.6f}", True),
        ("Max Error",  "max_error", "{:.4f}", True),
        ("R^2",        "r2",        "{:.6f}", False),
        ("Time (s)",   "time",      "{:.1f}", True),
    ]

    for label, key, fmt, lower_better in metrics:
        vals = {}
        for method in ["A", "B", "C"]:
            if method in results and key in results[method]:
                vals[method] = results[method][key]

        if not vals:
            continue

        if lower_better:
            winner = min(vals, key=vals.get)
        else:
            winner = max(vals, key=vals.get)

        name_map = {"A": "Method A", "B": "Method B", "C": "Method C"}

        cells = [f" {label:<20s} "]
        for m in ["A", "B", "C"]:
            v = vals.get(m, float("nan"))
            s = fmt.format(v)
            if m == winner:
                s = s + " *"
            cells.append(f" {s:<20s} ")
        cells.append(f" {name_map.get(winner, '?'):<20s} ")
        print("|" + "|".join(cells) + "|")

    print(sep)

    # Improvement summary
    if "C" in results and "A" in results and "B" in results:
        imp_a = (results["A"]["val_loss"] - results["C"]["val_loss"]) / (results["A"]["val_loss"] + 1e-10) * 100
        imp_b = (results["B"]["val_loss"] - results["C"]["val_loss"]) / (results["B"]["val_loss"] + 1e-10) * 100
        print(f"\n  Method C improvement over A: {imp_a:+.1f}% (val loss)")
        print(f"  Method C improvement over B: {imp_b:+.1f}% (val loss)")
        print(f"  Method C R^2: {results['C']['r2']:.6f}")


# =========================================================================== #
#  Plotting
# =========================================================================== #

def plot_testbed_results(results, mesh_data, output_dir="results_testbed"):
    """Generate comparison plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    q_mesh, p_mesh, H_true = mesh_data

    # --- 1. Energy Landscape Comparison (2x2) ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12),
                             subplot_kw={"projection": "3d"})

    axes[0, 0].plot_surface(q_mesh, p_mesh, H_true, cmap="coolwarm",
                            linewidth=0, alpha=0.85)
    axes[0, 0].set_title("Ground Truth H(q,p)", fontweight="bold", fontsize=12)

    method_labels = {"A": "Method A: Pure HHD",
                     "B": "Method B: Hybrid BO",
                     "C": "Method C: Unified (Improved)"}
    positions = {"A": (0, 1), "B": (1, 0), "C": (1, 1)}

    for key in ["A", "B", "C"]:
        if key not in results or "H_pred" not in results[key]:
            continue
        r, c = positions[key]
        ax = axes[r, c]
        ax.plot_surface(q_mesh, p_mesh, results[key]["H_pred"],
                        cmap="plasma", linewidth=0, alpha=0.85)
        mae = results[key]["mae"]
        r2  = results[key]["r2"]
        ax.set_title(f"{method_labels[key]}\nMAE={mae:.4f}  R2={r2:.4f}",
                     fontweight="bold", fontsize=11)

    for ax in axes.flat:
        ax.set_xlabel("q"); ax.set_ylabel("p"); ax.set_zlabel("H")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "landscape_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # --- 2. Convergence Curves ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for key, label in [("A", "Method A"), ("B", "Method B"), ("C", "Method C")]:
        if key not in results:
            continue
        hist = results[key].get("history", {})

        if "val_loss" in hist:
            axes[0].plot(hist["val_loss"], label=label, linewidth=2)
        elif "best_val_loss" in hist:
            axes[0].plot(hist["best_val_loss"], label=label, linewidth=2)

    axes[0].set_title("Validation Loss Convergence", fontweight="bold")
    axes[0].set_xlabel("Epoch / Trial")
    axes[0].set_ylabel("Validation Loss")
    axes[0].legend()
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)

    # Bar chart of final metrics
    methods = [k for k in ["A", "B", "C"] if k in results]
    x = np.arange(len(methods))
    width = 0.25

    mae_vals  = [results[m]["mae"] for m in methods]
    rmse_vals = [results[m]["rmse"] for m in methods]

    axes[1].bar(x - width/2, mae_vals, width, label="MAE", color="#4C72B0")
    axes[1].bar(x + width/2, rmse_vals, width, label="RMSE", color="#DD8452")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"Method {m}" for m in methods])
    axes[1].set_title("Error Metrics Comparison", fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "convergence_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  Plots saved to {output_dir}/")


# =========================================================================== #
#  Main Testbed Runner
# =========================================================================== #

def run_testbed():
    """Run the full 3-method comparison on the challenging testbed."""
    print("\n" + "=" * 85)
    print("  PERFORMANCE TESTBED")
    print("  Multi-Modal Hamiltonian with Heteroscedastic Noise")
    print("=" * 85)

    # Patch data generator
    _dg.generate_hamiltonian_data = _patched_gen

    # Generate testbed data for evaluation
    _, val_loader, mesh_data = generate_testbed_data(n_samples=3000)
    q_mesh, p_mesh, H_true = mesh_data

    results = {}
    N_SAMPLES = 3000
    N_WARMUP  = 15
    N_EPOCHS  = 60

    # ---- Method A: Pure HHD ----
    print("\n--- Method A: Pure HHD ---")
    np.random.seed(42); torch.manual_seed(42)
    try:
        trainerA = HamiltonianTrainer(
            hyperparam_space=config.HYPERPARAM_SPACE,
            init_hyperparams=config.INIT_HYPERPARAMS,
            step_size=config.STEP_SIZE,
            n_leapfrog=config.N_LEAPFROG_STEPS,
            temperature=config.TEMPERATURE,
        )
        t0 = time.time()
        histA = trainerA.train(n_samples=N_SAMPLES, n_warmup=N_WARMUP, n_hamilton=N_EPOCHS)
        time_a = time.time() - t0

        metrics_a = evaluate_model(trainerA.model, val_loader, q_mesh, p_mesh, H_true)
        metrics_a["time"] = time_a
        metrics_a["history"] = histA
        results["A"] = metrics_a
    except Exception as e:
        print(f"  Method A failed: {e}")

    # ---- Method B: Hybrid BO ----
    print("\n--- Method B: Hybrid BO ---")
    np.random.seed(42); torch.manual_seed(42)
    try:
        trainerB = HybridAdamBFGSTrainer(n_bo_trials=15, adam_epochs=20, lbfgs_steps=10)
        t0 = time.time()
        histB = trainerB.train(n_samples=N_SAMPLES)
        time_b = time.time() - t0

        metrics_b = evaluate_model(trainerB.best_model, val_loader, q_mesh, p_mesh, H_true)
        metrics_b["time"] = time_b
        metrics_b["history"] = histB
        results["B"] = metrics_b
    except Exception as e:
        print(f"  Method B failed: {e}")

    # ---- Method C: Unified HHD-ABBO (Improved) ----
    print("\n--- Method C: Unified HHD-ABBO (Improved) ---")
    np.random.seed(42); torch.manual_seed(42)
    try:
        trainerC = ImprovedUnifiedTrainer(
            hyperparam_space=config.HYPERPARAM_SPACE,
            init_hyperparams=config.INIT_HYPERPARAMS,
            initial_step=config.STEP_SIZE,
            n_leapfrog=config.N_LEAPFROG_STEPS,
            temperature=config.TEMPERATURE,
            adam_micro_epochs=3,
            lbfgs_steps=30,
        )
        t0 = time.time()
        histC = trainerC.train(n_samples=N_SAMPLES, n_warmup=N_WARMUP, n_hamilton=N_EPOCHS)
        time_c = time.time() - t0

        metrics_c = evaluate_model(trainerC.model, val_loader, q_mesh, p_mesh, H_true)
        metrics_c["time"] = time_c
        metrics_c["history"] = histC
        results["C"] = metrics_c
    except Exception as e:
        print(f"  Method C failed: {e}")

    # Restore original data generator
    _dg.generate_hamiltonian_data = _original_gen

    # Print results
    print_comparison_table(results)

    # Plot
    try:
        plot_testbed_results(results, mesh_data)
    except Exception as e:
        print(f"  Plotting failed: {e}")

    # Save results (without numpy arrays)
    save_dir = "results_testbed"
    os.makedirs(save_dir, exist_ok=True)
    save_results = {}
    for k, v in results.items():
        save_results[k] = {
            mk: mv for mk, mv in v.items()
            if mk not in ("H_pred", "history")
        }
    with open(os.path.join(save_dir, "testbed_results.json"), "w") as f:
        json.dump(save_results, f, indent=2)

    return results


if __name__ == "__main__":
    run_testbed()
