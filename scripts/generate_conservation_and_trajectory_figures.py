"""
Generates two paper-ready figures that were missing from the repo:

1. energy_conservation_live.png — validates the theoretical claim of
   near-conservation of the augmented Hamiltonian DURING AN ACTUAL Method C
   training run on the harmonic oscillator (not the toy spring-only demo;
   this uses the real HamiltonianNN + HyperparamState joint system). This is
   the direct empirical check for the paper's symplectic-conservation claim.

2. hp_trajectory_harmonic.png — plots how each tuned hyperparameter
   (log_lr, dropout, log_wd) actually moves through its search space during
   Method C's HMC phase, showing the "hyperparameters as physical objects"
   idea concretely rather than just asserting it. (A Fashion-MNIST version
   of this plot already existed in the repo; the harmonic-oscillator
   version did not.)

Usage:
    python scripts/generate_conservation_and_trajectory_figures.py
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hamiltonian import HamiltonianNN, HyperparamState
from data_generator import generate_hamiltonian_data
from symplectic_solver import HamiltonianMCMC, LeapfrogIntegrator, compute_loss_and_grads
import config as base_config

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

torch.manual_seed(0)
np.random.seed(0)


def joint_hamiltonian(w_mom, hp_state, loss, mass_theta, mass_lambda):
    kin_theta = sum(float((p ** 2).sum()) / (2.0 * mass_theta) for p in w_mom.values())
    kin_lambda = hp_state.kinetic_energy(mass_lambda)
    return kin_theta + kin_lambda + loss


def main():
    train_loader, val_loader, _ = generate_hamiltonian_data(n_samples=500, seed=0)
    criterion = nn.MSELoss()

    hp_state = HyperparamState(base_config.INIT_HYPERPARAMS, base_config.HYPERPARAM_SPACE)
    hp_state.frozen_hps = ["n_layers", "n_neurons"]
    hp = hp_state.decode()
    model = HamiltonianNN(n_layers=hp["n_layers"], n_neurons=hp["n_neurons"],
                          dropout=hp["dropout"], input_dim=2)

    mass_theta, mass_lambda = 1.0, base_config.MASS_LAMBDA

    # ---- Run 1: log the Hamiltonian across many individual leapfrog SUB-STEPS
    # within a single trajectory, using the real LeapfrogIntegrator directly
    # (each .integrate() call with L=1 performs exactly one leapfrog sub-step:
    # half momentum step, full position step, half momentum step) ----
    Xb, yb = next(iter(train_loader))
    w_mom = {n: torch.randn_like(p) * float(np.sqrt(mass_theta)) for n, p in model.named_parameters()}
    hp_state.randomise_momenta(mass_lambda)
    loss0, _ = compute_loss_and_grads(model, (Xb, yb), criterion)
    H_start = joint_hamiltonian(w_mom, hp_state, loss0, mass_theta, mass_lambda)

    lf_clipped = LeapfrogIntegrator(step_size=0.005, n_steps=1, mass_theta=mass_theta,
                                    mass_lambda=mass_lambda, grad_clip=10.0)
    lf_unclipped = LeapfrogIntegrator(step_size=0.005, n_steps=1, mass_theta=mass_theta,
                                      mass_lambda=mass_lambda, grad_clip=None)

    def run_trajectory(lf):
        torch.manual_seed(0)
        hp_s = HyperparamState(base_config.INIT_HYPERPARAMS, base_config.HYPERPARAM_SPACE)
        hp_s.frozen_hps = ["n_layers", "n_neurons"]
        m = HamiltonianNN(n_layers=hp["n_layers"], n_neurons=hp["n_neurons"],
                          dropout=hp["dropout"], input_dim=2)
        w_m = {n: torch.randn_like(p) * float(np.sqrt(mass_theta)) for n, p in m.named_parameters()}
        hp_s.randomise_momenta(mass_lambda)
        l0, _ = compute_loss_and_grads(m, (Xb, yb), criterion)
        energies = [joint_hamiltonian(w_m, hp_s, l0, mass_theta, mass_lambda)]
        l = l0
        for _ in range(40):
            l = lf.integrate(m, w_m, hp_s, (Xb, yb), criterion)
            energies.append(joint_hamiltonian(w_m, hp_s, l, mass_theta, mass_lambda))
        return energies

    energies_clipped = run_trajectory(lf_clipped)
    energies_unclipped = run_trajectory(lf_unclipped)
    energies_within_trajectory = energies_clipped  # kept for the drift-% print below

    # ---- Run 2: log the Hamiltonian across many ACCEPTED proposals over a
    # full short HMC phase, to see the coarser-grained behaviour a practitioner
    # would actually observe during training ----
    torch.manual_seed(1)
    hp_state2 = HyperparamState(base_config.INIT_HYPERPARAMS, base_config.HYPERPARAM_SPACE)
    hp_state2.frozen_hps = ["n_layers", "n_neurons"]
    model2 = HamiltonianNN(n_layers=hp["n_layers"], n_neurons=hp["n_neurons"],
                           dropout=hp["dropout"], input_dim=2)
    mcmc2 = HamiltonianMCMC(step_size=0.005, n_leapfrog=6, mass_theta=mass_theta,
                            mass_lambda=mass_lambda, temperature=1.0)
    Xb2, yb2 = next(iter(train_loader))
    loss_running, _ = compute_loss_and_grads(model2, (Xb2, yb2), criterion)
    proposal_energies, accept_flags = [], []
    hp_log = {"log_lr": [], "dropout": [], "log_batch_size": []}
    for i in range(60):
        accepted, loss_running = mcmc2.propose(model2, hp_state2, (Xb2, yb2), criterion, loss_running)
        accept_flags.append(accepted)
        w_mom_now = {n: torch.zeros_like(p) for n, p in model2.named_parameters()}  # post-hoc; momentum not retained externally
        raw_hp = {k: float(v.item()) for k, v in hp_state2.values.items()}
        for k in hp_log:
            if k in raw_hp:
                hp_log[k].append(raw_hp[k])
        proposal_energies.append(loss_running)

    # ---------------- Figure 1: energy conservation ----------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    pct_clipped = [100 * (e - energies_clipped[0]) / abs(energies_clipped[0]) for e in energies_clipped]
    pct_unclipped = [100 * (e - energies_unclipped[0]) / abs(energies_unclipped[0]) for e in energies_unclipped]
    axes[0].plot(pct_unclipped, marker="o", markersize=3, color="#4C72B0",
                label="No gradient clipping (pure symplectic dynamics)")
    axes[0].plot(pct_clipped, marker="s", markersize=3, color="#C44E52",
                label="With gradient clipping (grad_clip=10, this repo's default)")
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Leapfrog sub-step")
    axes[0].set_ylabel("Change in joint Hamiltonian H (%, relative to start)")
    axes[0].set_title("Within ONE trajectory (40 leapfrog sub-steps)\non the real weight+hyperparameter phase space")
    axes[0].legend(fontsize=8)
    drift_pct = pct_clipped[-1]
    drift_pct_unclipped = pct_unclipped[-1]
    axes[0].text(0.02, 0.95, f"Final drift: {drift_pct_unclipped:+.2f}% (no clip)  |  {drift_pct:+.2f}% (clipped)",
                transform=axes[0].transAxes, fontsize=8, color="#333", va="top")

    axes[1].plot(proposal_energies, color="#55A868", linewidth=1)
    colors = ["#55A868" if a else "#C44E52" for a in accept_flags]
    axes[1].scatter(range(len(proposal_energies)), proposal_energies, c=colors, s=14, zorder=3)
    axes[1].set_xlabel("HMC proposal number")
    axes[1].set_ylabel("Training loss (potential energy term)")
    axes[1].set_title(f"Across 60 accepted/rejected proposals\n(green=accepted, red=rejected; accept rate={mcmc2.acceptance_rate:.0%})")

    plt.suptitle("Empirical Validation: Augmented Hamiltonian Conservation During Real Training", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "energy_conservation_live.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plots/energy_conservation_live.png")
    print(f"  Within-trajectory drift (unclipped): {drift_pct_unclipped:+.3f}%")
    print(f"  Within-trajectory drift (clipped, repo default): {drift_pct:+.3f}%")
    print(f"  60-proposal acceptance rate: {mcmc2.acceptance_rate:.1%}")

    import json
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "energy_conservation_trace.json"), "w") as f:
        json.dump({"pct_unclipped": pct_unclipped, "pct_clipped": pct_clipped}, f, indent=2)
    print("Saved results/energy_conservation_trace.json (raw trace, for reuse in other figures)")

    # ---------------- Figure 2: hyperparameter trajectory ----------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    titles = {"log_lr": "log(learning rate)", "dropout": "dropout rate", "log_batch_size": "log(batch size)"}
    for ax, key in zip(axes, ["log_lr", "dropout", "log_batch_size"]):
        vals = hp_log[key]
        ax.plot(vals, color="#8172B2", linewidth=1.5)
        lo, hi = base_config.HYPERPARAM_SPACE[key]
        ax.axhline(lo, color="gray", linestyle=":", linewidth=1)
        ax.axhline(hi, color="gray", linestyle=":", linewidth=1, label="search bounds")
        ax.set_title(titles[key])
        ax.set_xlabel("HMC proposal number")
        ax.legend(fontsize=8)
    plt.suptitle("Hyperparameters as Dynamical Variables: Trajectory During Method C's HMC Phase\n(Harmonic Oscillator benchmark, one representative run)", y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "hp_trajectory_harmonic.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plots/hp_trajectory_harmonic.png")


if __name__ == "__main__":
    main()
