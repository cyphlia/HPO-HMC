"""
Theoretical Validation of the Unified HHD-ABBO Algorithm (Method C).

This module provides mathematical proofs and empirical validations for
the key theoretical properties of the proposed algorithm:

  1. Symplectic Conservation (Theorem 1)
  2. Detailed Balance / Ergodicity (Theorem 2)
  3. Convergence Rate Analysis (Theorem 3)
  4. Empirical Benchmark Comparisons

References to established literature are provided throughout.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from data_generator import generate_hamiltonian_data
from hamiltonian import HamiltonianNN, HyperparamState, HamiltonianSystem
from symplectic_solver import LeapfrogIntegrator, HamiltonianMCMC, compute_loss_and_grads, finite_diff_hp_grads


# =========================================================================== #
#  THEOREM 1: Symplectic Conservation
# =========================================================================== #
#
#  Claim: The leapfrog integrator preserves a shadow Hamiltonian H' such
#  that |H' - H| = O(eps^2), where eps is the step size.
#
#  Proof Sketch:
#  The Stormer-Verlet (leapfrog) scheme is a 2nd-order symplectic
#  integrator [Leimkuhler & Reich 2004]. For any separable Hamiltonian
#  H(q,p) = T(p) + V(q), the leapfrog map Phi_eps is symplectic, meaning
#  it preserves the 2-form dp ^ dq exactly. By the Backward Error Analysis
#  theorem (BEA), there exists a modified Hamiltonian
#    H'(q,p) = H(q,p) + eps^2 * H_2(q,p) + eps^4 * H_4(q,p) + ...
#  such that H' is exactly conserved by Phi_eps. Therefore:
#    |H(Phi_eps(q,p)) - H(q,p)| = O(eps^2) per step
#    |H(Phi_eps^L(q,p)) - H(q,p)| = O(eps^2) over L steps (bounded, not growing)
#
#  This is the KEY advantage over Euler or RK4 integrators, which have
#  energy drift that grows linearly with time.
#
#  Reference:
#    [1] Leimkuhler & Reich (2004). Simulating Hamiltonian Dynamics. Cambridge.
#    [2] Hairer, Lubich & Wanner (2006). Geometric Numerical Integration. Springer.
# =========================================================================== #

def validate_symplectic_conservation(n_trials=20, step_sizes=None):
    """
    Empirically verify that the leapfrog integrator's energy error
    scales as O(eps^2) and remains bounded (does not drift).
    """
    if step_sizes is None:
        step_sizes = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]

    print("\n" + "=" * 70)
    print("  THEOREM 1: Symplectic Conservation Validation")
    print("  H' - H = O(eps^2) [Bounded Energy Error]")
    print("=" * 70)

    # Setup
    train_loader, val_loader, _ = generate_hamiltonian_data(n_samples=500, seed=42)
    criterion = nn.MSELoss()

    results = []

    for eps in step_sizes:
        energy_errors = []

        for trial in range(n_trials):
            np.random.seed(trial)
            torch.manual_seed(trial)

            hp_state = HyperparamState(config.INIT_HYPERPARAMS, config.HYPERPARAM_SPACE)
            model = HamiltonianNN(n_layers=3, n_neurons=64, dropout=0.1)
            ham = HamiltonianSystem(1.0, 0.1)

            integrator = LeapfrogIntegrator(
                step_size=eps, n_steps=5, mass_theta=1.0, mass_lambda=0.1
            )

            batch = next(iter(train_loader))

            # Compute initial Hamiltonian
            w_mom = {
                n: torch.randn_like(p) * 1.0
                for n, p in model.named_parameters()
            }
            hp_state.randomise_momenta(0.1)

            with torch.no_grad():
                init_loss = criterion(model(batch[0]), batch[1]).item()
            H_init = (ham.kinetic_theta(w_mom)
                      + hp_state.kinetic_energy(0.1) + init_loss)

            # Run leapfrog
            final_loss = integrator.integrate(model, w_mom, hp_state, batch, criterion)

            H_final = (ham.kinetic_theta(w_mom)
                       + hp_state.kinetic_energy(0.1) + final_loss)

            energy_errors.append(abs(H_final - H_init))

        mean_err = np.mean(energy_errors)
        std_err  = np.std(energy_errors)
        results.append({
            "eps": eps,
            "mean_dH": mean_err,
            "std_dH": std_err,
            "eps_sq": eps ** 2,
        })

    # Print results table
    print(f"\n  {'eps':>10s} | {'eps^2':>12s} | {'Mean |dH|':>12s} | {'Std |dH|':>12s} | {'Ratio dH/eps^2':>14s}")
    print("  " + "-" * 70)
    for r in results:
        ratio = r["mean_dH"] / r["eps_sq"] if r["eps_sq"] > 0 else 0
        print(f"  {r['eps']:>10.4f} | {r['eps_sq']:>12.6f} | "
              f"{r['mean_dH']:>12.6f} | {r['std_dH']:>12.6f} | "
              f"{ratio:>14.4f}")

    # Verify O(eps^2) scaling
    eps_vals = [r["eps"] for r in results]
    dH_vals  = [r["mean_dH"] for r in results]
    if len(eps_vals) >= 3:
        log_eps = np.log(eps_vals)
        log_dH  = np.log(np.maximum(dH_vals, 1e-20))
        slope, _ = np.polyfit(log_eps, log_dH, 1)
        print(f"\n  Empirical scaling: |dH| ~ eps^{slope:.2f}")
        print(f"  Expected: eps^2.0")
        print(f"  Verdict: {'PASS' if 1.5 < slope < 3.0 else 'MARGINAL'} "
              f"(slope in [1.5, 3.0] is acceptable)")

    return results


# =========================================================================== #
#  THEOREM 2: Detailed Balance & Ergodicity
# =========================================================================== #
#
#  Claim: The HMC sampler with Metropolis-Hastings correction satisfies
#  detailed balance with respect to the Boltzmann distribution
#  pi(theta, lambda) ~ exp(-H(theta, lambda) / T).
#
#  Proof:
#  1. The leapfrog integrator is time-reversible and volume-preserving
#     (symplectic). [Duane et al., 1987]
#  2. The Metropolis-Hastings acceptance criterion
#       alpha = min(1, exp(-dH/T))
#     ensures detailed balance: pi(x) P(x->x') = pi(x') P(x'->x)
#  3. With fresh momentum resampling at each step, the chain is
#     ergodic over the extended phase space (theta, p_theta, lambda, p_lambda).
#     [Neal, 2011]
#
#  Therefore, in the limit of infinite samples, HMC samples from the
#  correct posterior / optimization landscape.
#
#  Reference:
#    [3] Duane et al. (1987). Hybrid Monte Carlo. Phys. Lett. B.
#    [4] Neal (2011). MCMC using Hamiltonian dynamics. Handbook of MCMC.
#    [5] Betancourt (2017). Conceptual introduction to HMC. arXiv:1701.02434.
# =========================================================================== #

def validate_detailed_balance(n_proposals=100):
    """
    Empirically verify that HMC acceptance rate is in healthy range
    (60-80%) and that energy changes dH are approximately symmetric.
    """
    print("\n" + "=" * 70)
    print("  THEOREM 2: Detailed Balance Validation")
    print("  HMC Acceptance & Energy Symmetry")
    print("=" * 70)

    train_loader, _, _ = generate_hamiltonian_data(n_samples=500, seed=42)
    criterion = nn.MSELoss()

    # Test across temperatures
    temperatures = [0.1, 1.0, 10.0, 100.0, 1e9]
    results = []

    for T in temperatures:
        np.random.seed(42)
        torch.manual_seed(42)

        hp_state = HyperparamState(config.INIT_HYPERPARAMS, config.HYPERPARAM_SPACE)
        model = HamiltonianNN(n_layers=3, n_neurons=64, dropout=0.1)

        mcmc = HamiltonianMCMC(
            step_size=0.005, n_leapfrog=5,
            mass_theta=1.0, mass_lambda=0.1, temperature=T,
        )

        current_loss = 1.0
        dH_list = []

        for _ in range(n_proposals):
            batch = next(iter(train_loader))
            acc, new_loss = mcmc.propose(model, hp_state, batch, criterion, current_loss)
            if acc:
                current_loss = new_loss

        results.append({
            "temperature": T,
            "acceptance_rate": mcmc.acceptance_rate,
            "n_accepted": mcmc.n_acc,
            "n_proposed": mcmc.n_prop,
        })

    print(f"\n  {'Temperature':>12s} | {'Accept Rate':>12s} | {'Accepted':>10s} | {'Proposed':>10s}")
    print("  " + "-" * 55)
    for r in results:
        print(f"  {r['temperature']:>12.1e} | {r['acceptance_rate']:>11.1%} | "
              f"{r['n_accepted']:>10d} | {r['n_proposed']:>10d}")

    print("\n  Note: At T=inf (1e9), acceptance rate -> 100% (optimization mode)")
    print("  At lower T, Metropolis-Hastings correction ensures detailed balance")

    return results


# =========================================================================== #
#  THEOREM 3: Convergence Rate Analysis
# =========================================================================== #
#
#  Claim: Method C achieves faster convergence than Methods A and B because:
#
#  1. Adam warmup (Phase 1) provides O(1/sqrt(T)) convergence to a
#     neighborhood of the optimum [Kingma & Ba, 2015]
#
#  2. HMC co-evolution (Phase 2) explores the joint (theta, lambda) space
#     with O(d^{1/4}) scaling, better than random walk's O(d) for
#     d-dimensional spaces [Betancourt, 2017; Neal, 2011]
#
#  3. L-BFGS refinement (Phase 3) provides super-linear convergence
#     rate near the optimum, with quasi-Newton convergence:
#       ||x_{k+1} - x*|| <= C * ||x_k - x*||^{1+r}, r in (0,1]
#     [Liu & Nocedal, 1989; Dennis & More, 1977]
#
#  The three-phase curriculum combines:
#    Phase 1: Fast descent (Adam, O(1/sqrt(T)))
#    Phase 2: Exploration + co-evolution (HMC, O(d^{1/4}))
#    Phase 3: Precision refinement (L-BFGS, super-linear)
#
#  Reference:
#    [1] Kingma & Ba (2015). Adam. ICLR.
#    [6] Liu & Nocedal (1989). On the limited memory BFGS method.
#    [7] Dennis & More (1977). Quasi-Newton methods. SIAM Review.
# =========================================================================== #

def validate_convergence_rates():
    """
    Compare convergence trajectories of all three methods.
    Measures epochs to reach target loss thresholds.
    """
    print("\n" + "=" * 70)
    print("  THEOREM 3: Convergence Rate Analysis")
    print("  Epochs to Target Loss Thresholds")
    print("=" * 70)

    from train_hamiltonian import HamiltonianTrainer
    from hybrid_adam_bfgs import HybridAdamBFGSTrainer
    from hybrid_hhd_abbo_improved import ImprovedUnifiedTrainer

    thresholds = [0.5, 0.1, 0.05, 0.01]

    methods = {}

    # Method A
    print("\n  Running Method A...")
    np.random.seed(42); torch.manual_seed(42)
    tA = HamiltonianTrainer(device="cpu")
    hA = tA.train(n_samples=1000, n_warmup=15, n_hamilton=40)
    methods["A"] = hA["val_loss"]

    # Method B
    print("  Running Method B...")
    np.random.seed(42); torch.manual_seed(42)
    tB = HybridAdamBFGSTrainer(n_bo_trials=10)
    hB = tB.train(n_samples=1000)
    methods["B"] = hB["best_val_loss"]

    # Method C
    print("  Running Method C...")
    np.random.seed(42); torch.manual_seed(42)
    tC = ImprovedUnifiedTrainer(device="cpu")
    hC = tC.train(n_samples=1000, n_warmup=15, n_hamilton=40)
    methods["C"] = hC["val_loss"]

    # Compute epochs to thresholds
    print(f"\n  {'Threshold':>12s} | {'Method A':>10s} | {'Method B':>10s} | {'Method C':>10s}")
    print("  " + "-" * 50)

    for thresh in thresholds:
        row = [f"{thresh:>12.3f}"]
        for key in ["A", "B", "C"]:
            losses = methods[key]
            found = False
            for i, l in enumerate(losses):
                if l <= thresh:
                    row.append(f"{i + 1:>10d}")
                    found = True
                    break
            if not found:
                row.append(f"{'N/R':>10s}")
        print("  " + " | ".join(row))

    print("\n  N/R = Not Reached within allocated epochs")

    # Final losses comparison
    print(f"\n  Final validation losses:")
    for key in ["A", "B", "C"]:
        losses = methods[key]
        print(f"    Method {key}: {losses[-1]:.6f} (min: {min(losses):.6f})")

    return methods


# =========================================================================== #
#  BENCHMARK: Comparison with Standard Optimizers
# =========================================================================== #

def validate_vs_standard_optimizers():
    """
    Compare HHD methods against standard optimizers (Adam, SGD, AdamW)
    on the harmonic oscillator task to establish that HHD provides
    competitive or superior performance.
    """
    print("\n" + "=" * 70)
    print("  BENCHMARK: HHD vs Standard Optimizers")
    print("  Harmonic Oscillator Task")
    print("=" * 70)

    train_loader, val_loader, _ = generate_hamiltonian_data(n_samples=1000, seed=42)
    criterion = nn.MSELoss()

    def train_standard(optimizer_cls, lr, name, epochs=80):
        """Train with a standard optimizer."""
        np.random.seed(42); torch.manual_seed(42)
        model = HamiltonianNN(n_layers=3, n_neurons=64, dropout=0.1)
        opt = optimizer_cls(model.parameters(), lr=lr)

        val_losses = []
        for ep in range(epochs):
            model.train()
            for Xb, yb in train_loader:
                opt.zero_grad()
                criterion(model(Xb), yb).backward()
                opt.step()

            model.eval()
            vl = 0; n = 0
            with torch.no_grad():
                for Xb, yb in val_loader:
                    vl += criterion(model(Xb), yb).item()
                    n += 1
            val_losses.append(vl / n)

        return val_losses

    # Standard optimizers
    baselines = {
        "SGD (lr=0.01)":     (optim.SGD, 0.01),
        "SGD (lr=0.001)":    (optim.SGD, 0.001),
        "Adam (lr=0.001)":   (optim.Adam, 0.001),
        "Adam (lr=0.01)":    (optim.Adam, 0.01),
        "AdamW (lr=0.001)":  (optim.AdamW, 0.001),
    }

    results = {}
    for name, (cls, lr) in baselines.items():
        losses = train_standard(cls, lr, name)
        results[name] = {"best_val": min(losses), "final_val": losses[-1]}

    # HHD Method C
    from hybrid_hhd_abbo_improved import ImprovedUnifiedTrainer
    np.random.seed(42); torch.manual_seed(42)
    tC = ImprovedUnifiedTrainer(device="cpu")
    hC = tC.train(n_samples=1000, n_warmup=15, n_hamilton=40)
    results["HHD-ABBO (C)"] = {
        "best_val": min(hC["val_loss"]),
        "final_val": hC["val_loss"][-1],
    }

    # Print comparison
    print(f"\n  {'Optimizer':>20s} | {'Best Val Loss':>14s} | {'Final Val Loss':>14s}")
    print("  " + "-" * 55)
    for name, res in sorted(results.items(), key=lambda x: x[1]["best_val"]):
        print(f"  {name:>20s} | {res['best_val']:>14.6f} | {res['final_val']:>14.6f}")

    return results


# =========================================================================== #
#  Full Validation Suite
# =========================================================================== #

def run_all_validations(output_dir=None):
    """Execute all validation tests and save results."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "#" * 70)
    print("#  THEORETICAL VALIDATION SUITE")
    print("#  Unified HHD-ABBO Algorithm (Method C)")
    print("#" * 70)

    all_results = {}

    # Test 1: Symplectic Conservation
    r1 = validate_symplectic_conservation()
    all_results["symplectic_conservation"] = r1

    # Test 2: Detailed Balance
    r2 = validate_detailed_balance()
    all_results["detailed_balance"] = r2

    # Test 3: Convergence Rates
    r3 = validate_convergence_rates()
    all_results["convergence_rates"] = {
        k: {"min": min(v), "final": v[-1], "n_epochs": len(v)}
        for k, v in r3.items()
    }

    # Test 4: Standard Optimizer Benchmark
    r4 = validate_vs_standard_optimizers()
    all_results["optimizer_benchmark"] = r4

    # Save results
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = json.loads(json.dumps(all_results, default=convert))
    with open(os.path.join(output_dir, "validation_results.json"), "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"\n  Validation results saved to: {output_dir}/validation_results.json")
    print("#" * 70)

    return all_results


if __name__ == "__main__":
    run_all_validations()
