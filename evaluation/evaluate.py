"""
Evaluation & Results Reporting.

Generates structured tables and selective plots for:
  1. Harmonic Oscillator benchmark (3-method comparison)
  2. CNN Classification benchmark (3-method comparison)
  3. Energy landscape reconstruction quality
  4. Hyperparameter evolution tracking
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from hamiltonian import HamiltonianNN


# --------------------------------------------------------------------------- #
#  I/O Helpers
# --------------------------------------------------------------------------- #

def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_model(hp, model_path):
    """Load model, inferring architecture from state dict if HP mismatch."""
    state = torch.load(model_path, map_location="cpu", weights_only=True)

    # Infer architecture from state dict
    n_neurons = state["input_layer.weight"].shape[0]
    n_layers = sum(1 for k in state if k.startswith("hidden.") and k.endswith(".0.weight"))

    model = HamiltonianNN(n_layers=n_layers, n_neurons=n_neurons, dropout=0.0)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict_landscape(model, q_mesh, p_mesh):
    q_flat = q_mesh.flatten().astype(np.float32)
    p_flat = p_mesh.flatten().astype(np.float32)
    X = torch.from_numpy(np.stack([q_flat, p_flat], axis=1))
    return model(X).numpy().reshape(q_mesh.shape)


# --------------------------------------------------------------------------- #
#  Table Printing
# --------------------------------------------------------------------------- #

def print_table(headers, rows, title=None, col_widths=None):
    """Print a formatted ASCII table."""
    if col_widths is None:
        col_widths = [max(len(str(h)), max(len(str(r[i])) for r in rows))
                      for i, h in enumerate(headers)]

    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    if title:
        total_w = sum(col_widths) + 3 * len(col_widths) + 1
        print("\n" + "=" * total_w)
        print(f"  {title}")
        print("=" * total_w)

    print(sep)
    print("|" + "|".join(f" {h:<{w}s} " for h, w in zip(headers, col_widths)) + "|")
    print(sep)
    for row in rows:
        cells = []
        for val, w in zip(row, col_widths):
            cells.append(f" {str(val):<{w}s} ")
        print("|" + "|".join(cells) + "|")
    print(sep)


def save_table_to_file(headers, rows, title, filepath):
    """Save table to a text file."""
    old_stdout = sys.stdout
    with open(filepath, "w") as f:
        sys.stdout = f
        print_table(headers, rows, title)
    sys.stdout = old_stdout


# --------------------------------------------------------------------------- #
#  Harmonic Oscillator Evaluation
# --------------------------------------------------------------------------- #

def evaluate_harmonic(results_dirs=None, mesh_data=None, output_dir="results"):
    """
    Evaluate and compare all three methods on the harmonic oscillator task.
    Produces summary tables and a single comparison plot.
    """
    if results_dirs is None:
        results_dirs = {
            "A": "results/harmonic_oscillator/method_a",
            "B": "results/harmonic_oscillator/method_b",
            "C": "results/harmonic_oscillator/method_c",
        }

    os.makedirs(output_dir, exist_ok=True)

    # Generate mesh if not provided
    if mesh_data is None:
        from data_generator import generate_hamiltonian_data
        _, _, mesh_data = generate_hamiltonian_data(n_samples=500)

    q_mesh, p_mesh, H_true = mesh_data

    method_names = {
        "A": "Pure HHD",
        "B": "Hybrid BO",
        "C": "Unified HHD-ABBO",
    }

    # Collect results
    results = {}
    for key, rdir in results_dirs.items():
        if not os.path.exists(rdir):
            print(f"  [WARN] {rdir} not found, skipping Method {key}")
            continue

        hist_file = os.path.join(rdir, "history.json")
        hp_file   = os.path.join(rdir, "hyperparameters.json")
        model_file = os.path.join(rdir, "model.pt")

        if not all(os.path.exists(f) for f in [hist_file, hp_file, model_file]):
            print(f"  [WARN] Incomplete results in {rdir}, skipping")
            continue

        hist = load_json(hist_file)
        hp   = load_json(hp_file)
        model = load_model(hp, model_file)
        H_pred = predict_landscape(model, q_mesh, p_mesh)

        residual = np.abs(H_pred - H_true)
        mae  = residual.mean()
        rmse = np.sqrt(((H_pred - H_true) ** 2).mean())
        maxe = residual.max()
        r2   = 1 - ((H_pred - H_true) ** 2).sum() / ((H_true - H_true.mean()) ** 2).sum()

        # Get val_loss - handle different history formats
        if "val_loss" in hist:
            val_losses = hist["val_loss"]
        elif "best_val_loss" in hist:
            val_losses = hist["best_val_loss"]
        else:
            val_losses = [0.0]

        results[key] = {
            "name":      method_names[key],
            "best_val":  min(val_losses),
            "final_val": val_losses[-1],
            "mae":       mae,
            "rmse":      rmse,
            "max_error": maxe,
            "r2":        r2,
            "hp":        hp,
            "hist":      hist,
            "H_pred":    H_pred,
        }

    if not results:
        print("  No results found. Run training first.")
        return

    # --- Table 1: Performance Comparison ---
    headers = ["Method", "Best Val Loss", "Final Val Loss", "MAE", "RMSE", "R^2"]
    rows = []
    for key in sorted(results.keys()):
        r = results[key]
        rows.append([
            r["name"],
            f"{r['best_val']:.6f}",
            f"{r['final_val']:.6f}",
            f"{r['mae']:.6f}",
            f"{r['rmse']:.6f}",
            f"{r['r2']:.6f}",
        ])
    print_table(headers, rows, "HARMONIC OSCILLATOR: PERFORMANCE COMPARISON",
                col_widths=[22, 14, 14, 10, 10, 10])
    save_table_to_file(headers, rows, "HARMONIC OSCILLATOR: PERFORMANCE COMPARISON",
                       os.path.join(output_dir, "harmonic_results.txt"))

    # --- Table 2: Optimized Hyperparameters ---
    hp_headers = ["Method", "lr", "n_layers", "n_neurons", "dropout", "batch_size"]
    hp_rows = []
    for key in sorted(results.keys()):
        hp = results[key]["hp"]
        lr_val = hp.get("lr", "N/A")
        lr_str = f"{lr_val:.6f}" if isinstance(lr_val, (int, float)) else str(lr_val)
        do_val = hp.get("dropout", "N/A")
        do_str = f"{do_val:.4f}" if isinstance(do_val, (int, float)) else str(do_val)
        hp_rows.append([
            results[key]["name"],
            lr_str,
            f"{hp.get('n_layers', 'N/A')}",
            f"{hp.get('n_neurons', 'N/A')}",
            do_str,
            f"{hp.get('batch_size', 'N/A')}",
        ])
    print_table(hp_headers, hp_rows, "OPTIMIZED HYPERPARAMETERS",
                col_widths=[22, 12, 10, 10, 10, 12])

    # --- Plot: Energy Landscape Comparison (only plot worth having) ---
    n_methods = len(results)
    fig, axes = plt.subplots(1, n_methods + 1, figsize=(6 * (n_methods + 1), 5),
                              subplot_kw={"projection": "3d"})
    if n_methods + 1 == 1:
        axes = [axes]

    # True landscape
    axes[0].plot_surface(q_mesh, p_mesh, H_true, cmap="coolwarm",
                         linewidth=0, alpha=0.85)
    axes[0].set_title("True H(q,p)", fontweight="bold")
    axes[0].set_xlabel("q"); axes[0].set_ylabel("p"); axes[0].set_zlabel("H")

    for i, key in enumerate(sorted(results.keys())):
        r = results[key]
        axes[i + 1].plot_surface(q_mesh, p_mesh, r["H_pred"], cmap="plasma",
                                  linewidth=0, alpha=0.85)
        axes[i + 1].set_title(f"{r['name']}\nMAE={r['mae']:.4f}", fontweight="bold")
        axes[i + 1].set_xlabel("q"); axes[i + 1].set_ylabel("p")
        axes[i + 1].set_zlabel("H")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "energy_landscape_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved plot: {plot_path}")

    return results


# --------------------------------------------------------------------------- #
#  CNN Evaluation
# --------------------------------------------------------------------------- #

def evaluate_cnn(results_file="results/cnn/benchmark_results.json", output_dir="results"):
    """Evaluate CNN benchmark results with tables."""
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(results_file):
        print(f"  CNN results not found at {results_file}. Run cnn_benchmark.py first.")
        return

    data = load_json(results_file)

    # --- Table: CNN Classification Results ---
    headers = ["Method", "Best Acc", "Final Acc", "Time (s)", "LR", "Dropout"]
    rows = []
    for method_name, res in data.items():
        hp = res.get("final_hps", {})
        do_val = hp.get('dropout', 'N/A')
        do_str = f"{do_val:.4f}" if isinstance(do_val, (int, float)) else str(do_val)
        rows.append([
            method_name,
            f"{res['best_val_acc']:.2%}",
            f"{res.get('final_val_acc', res['best_val_acc']):.2%}",
            f"{res['time']:.1f}",
            f"{hp.get('lr', 'N/A'):.6f}" if isinstance(hp.get('lr'), (int, float)) else "N/A",
            do_str,
        ])

    print_table(headers, rows, "CNN MNIST CLASSIFICATION: BENCHMARK RESULTS",
                col_widths=[30, 10, 10, 10, 12, 10])
    save_table_to_file(headers, rows, "CNN MNIST CLASSIFICATION: BENCHMARK RESULTS",
                       os.path.join(output_dir, "cnn_results.txt"))

    return data


# --------------------------------------------------------------------------- #
#  Full Evaluation Pipeline
# --------------------------------------------------------------------------- #

def run_full_evaluation():
    """Run complete evaluation for both benchmarks."""
    print("\n" + "=" * 70)
    print("  FULL EVALUATION PIPELINE")
    print("=" * 70)

    print("\n--- 1. Harmonic Oscillator Benchmark ---")
    harmonic_results = evaluate_harmonic()

    print("\n--- 2. CNN MNIST Benchmark ---")
    cnn_results = evaluate_cnn()

    print("\n" + "=" * 70)
    print("  EVALUATION COMPLETE")
    print(f"  Results saved to: {config.RESULTS_DIR}/")
    print("=" * 70)

    return harmonic_results, cnn_results


if __name__ == "__main__":
    run_full_evaluation()
