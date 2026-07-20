"""
Method A: Pure Hamiltonian Hyperparameter Dynamics (HHD).

Uses Adam for a warmup phase, then Hamiltonian Monte Carlo (HMC) to
co-evolve both network weights (theta) and hyperparameters (lambda)
via symplectic leapfrog integration.
"""

from __future__ import annotations

import json
import os
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional

import config
from data_generator import generate_hamiltonian_data
from hamiltonian import HamiltonianNN, HyperparamState, HamiltonianSystem
from symplectic_solver import HamiltonianMCMC


class HamiltonianTrainer:
    """
    Method A: Pure HHD trainer.

    Phase 1: Adam warmup (first-order stochastic descent)
    Phase 2: HMC co-evolution (symplectic joint theta+lambda updates)
    """

    def __init__(
        self,
        hyperparam_space=None,
        init_hyperparams=None,
        mass_theta: float = 1.0,
        mass_lambda: float = None,  # defaults to config.MASS_LAMBDA (matches Method C)
        step_size: float = 0.01,
        n_leapfrog: int = 5,
        temperature: float = 1.0,
        device: str = "cpu",
        input_dim: int = 2,
        momentum_refresh: float = 1.0,
    ):
        """
        momentum_refresh: see HamiltonianMCMC docstring in symplectic_solver.py.
        Kept at 1.0 (original full-resample behaviour) by default; set below
        1.0 to enable persistent-momentum (Generalized HMC). Testing showed
        persistent momentum is only a clear win when combined with a real
        (non-1e9) temperature -- see MOMENTUM_AND_TEMPERATURE_NOTES.md.

        mass_lambda: previously hardcoded to 0.1, silently ignoring
        config.MASS_LAMBDA (=5.0) even when the caller didn't override it.
        Method C's trainer already used a None-fallback pattern for this;
        Method A now matches it for consistency.
        """
        if mass_lambda is None:
            mass_lambda = config.MASS_LAMBDA
        self.device = device
        self.input_dim = input_dim
        hp_space = hyperparam_space or config.HYPERPARAM_SPACE
        hp_init  = init_hyperparams or config.INIT_HYPERPARAMS

        self.hp_state = HyperparamState(hp_init, hp_space)
        self.ham_sys  = HamiltonianSystem(mass_theta, mass_lambda)
        self.mcmc     = HamiltonianMCMC(step_size, n_leapfrog, mass_theta,
                                        mass_lambda, temperature,
                                        momentum_refresh=momentum_refresh)
        self.criterion = nn.MSELoss()
        self.model     = self._build_model()

        self.history = {
            "train_loss": [], "val_loss": [], "best_val_loss": [],
            "acceptance_rate": [],
            "hyperparams": {k: [] for k in hp_init},
        }
        self._best_val   = float("inf")
        self._best_state = None
        self._best_hp    = None

    def _build_model(self):
        hp = self.hp_state.decode()
        return HamiltonianNN(
            n_layers=hp.get("n_layers", 3),
            n_neurons=hp.get("n_neurons", 64),
            dropout=hp.get("dropout", 0.1),
            input_dim=self.input_dim,
        ).to(self.device)

    def _evaluate(self, loader):
        self.model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for Xb, yb in loader:
                total += self.criterion(
                    self.model(Xb.to(self.device)), yb.to(self.device)
                ).item()
                n += 1
        return total / max(n, 1)

    def train(self, n_samples=1000, n_warmup=10, n_hamilton=50,
              train_loader=None, val_loader=None):
        t0 = time.time()
        if train_loader is None or val_loader is None:
            train_loader, val_loader, self.mesh_data = generate_hamiltonian_data(
                n_samples=n_samples
            )
        else:
            self.mesh_data = None

        # Phase 1: Adam warmup
        opt = optim.Adam(self.model.parameters(),
                         lr=self.hp_state.decode().get("lr", 1e-3))
        print(f"  [Method A] Adam warmup: {n_warmup} epochs")
        for epoch in range(n_warmup):
            self.model.train()
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                self.criterion(self.model(Xb), yb).backward()
                opt.step()

        # Phase 2: HMC co-evolution
        print(f"  [Method A] HMC co-evolution: {n_hamilton} epochs")
        current_loss = self._evaluate(val_loader)
        val_iter = iter(val_loader)
        for epoch in range(n_hamilton):
            Xb, yb = next(iter(train_loader))
            Xb, yb = Xb.to(self.device), yb.to(self.device)
            try:
                X_val_b, y_val_b = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                X_val_b, y_val_b = next(val_iter)
            X_val_b, y_val_b = X_val_b.to(self.device), y_val_b.to(self.device)

            acc, current_loss = self.mcmc.propose(
                self.model, self.hp_state, (Xb, yb),
                self.criterion, current_loss, val_batch=(X_val_b, y_val_b)
            )

            tl = self._evaluate(train_loader)
            vl = self._evaluate(val_loader)

            # ----- Best-Model Checkpointing (was missing; Methods B & C
            # already do this, which is why their reported "Best Val Loss"
            # always equals the metric of the model actually saved/scored,
            # while Method A's did not -- see evaluate.py / README notes). -----
            if vl < self._best_val:
                self._best_val   = vl
                self._best_state = deepcopy(self.model.state_dict())
                self._best_hp    = self.hp_state.snapshot()

            self.history["train_loss"].append(tl)
            self.history["val_loss"].append(vl)
            self.history["best_val_loss"].append(self._best_val)
            self.history["acceptance_rate"].append(self.mcmc.acceptance_rate)
            for k in self.hp_state.values:
                self.history["hyperparams"][k].append(
                    float(self.hp_state.values[k].item()))

            if epoch % 10 == 0 or epoch == n_hamilton - 1:
                tag = "ACC" if acc else "REJ"
                print(f"    Ep {epoch:3d}/{n_hamilton} [{tag}] | "
                      f"Train: {tl:.5f} | Val: {vl:.5f} | "
                      f"Best: {self._best_val:.5f} | "
                      f"Acc: {self.mcmc.acceptance_rate:.1%}")

        # ----- Restore best checkpoint before returning/saving -----
        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)
            if self._best_hp is not None:
                self.hp_state.restore(self._best_hp)
            final_val = self._evaluate(val_loader)
            final_tr  = self._evaluate(train_loader)
            print(f"  [Method A] Restored best checkpoint "
                  f"(val={self._best_val:.6f}, re-eval={final_val:.6f})")
            self.history["val_loss"].append(final_val)
            self.history["best_val_loss"].append(self._best_val)
            self.history["train_loss"].append(final_tr)
            self.history["acceptance_rate"].append(self.history["acceptance_rate"][-1])
            for k in self.hp_state.values:
                self.history["hyperparams"][k].append(self.history["hyperparams"][k][-1])

        self.train_time = time.time() - t0
        return self.history

    def save(self, save_dir="results_hamiltonian"):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_dir, "model.pt"))
        with open(os.path.join(save_dir, "history.json"), "w") as f:
            json.dump(self.history, f, indent=2)
        with open(os.path.join(save_dir, "hyperparameters.json"), "w") as f:
            json.dump(self.hp_state.decode(), f, indent=2)
