"""
Method B: Hybrid Adam + L-BFGS with Bayesian Optimization (ABBO).

Traditional baseline approach:
  - Outer loop: Gaussian Process surrogate with Expected Improvement
    to select hyperparameters
  - Inner loop: Adam training followed by L-BFGS refinement
"""

from __future__ import annotations

import json
import os
import time
import warnings
from copy import deepcopy
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize
from scipy.stats import norm

import config
from data_generator import generate_hamiltonian_data
from hamiltonian import HamiltonianNN

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
#  Gaussian Process Surrogate
# --------------------------------------------------------------------------- #

class GaussianProcessSurrogate:
    """Simple GP surrogate with RBF kernel for Bayesian optimization."""

    def __init__(self, length_scale=1.0, signal_var=1.0, noise_var=1e-4):
        self.ell = length_scale
        self.sf  = signal_var
        self.sn  = noise_var
        self.X_obs = None
        self.y_obs = None
        self.K_inv = None

    def _rbf(self, A, B):
        sq = (A[:, None, :] - B[None, :, :]) ** 2
        dist = sq.sum(axis=2)
        return self.sf ** 2 * np.exp(-dist / (2 * self.ell ** 2))

    def update(self, X, y):
        self.X_obs, self.y_obs = X.copy(), y.copy()
        K = self._rbf(X, X) + self.sn * np.eye(len(X))
        self.K_inv = np.linalg.inv(K + 1e-8 * np.eye(len(K)))

    def predict(self, X_star):
        if self.X_obs is None:
            return np.zeros(len(X_star)), np.ones(len(X_star)) * self.sf
        Ks  = self._rbf(X_star, self.X_obs)
        Kss = self._rbf(X_star, X_star)
        mu  = Ks @ self.K_inv @ self.y_obs
        cov = Kss - Ks @ self.K_inv @ Ks.T
        return mu, np.sqrt(np.maximum(np.diag(cov), 1e-10))


def expected_improvement(X, gp, y_best, xi=0.01):
    mu, std = gp.predict(X)
    imp = y_best - mu - xi
    Z = imp / (std + 1e-10)
    return imp * norm.cdf(Z) + std * norm.pdf(Z)


def suggest_next(gp, bounds, y_best):
    """Multi-start L-BFGS-B optimization of the acquisition function."""
    best_x, best_v = None, -np.inf
    for _ in range(5):
        x0 = np.random.uniform(bounds[:, 0], bounds[:, 1])
        res = minimize(
            lambda x: -expected_improvement(x[None], gp, y_best)[0],
            x0, method="L-BFGS-B", bounds=bounds,
        )
        if -res.fun > best_v:
            best_v, best_x = -res.fun, res.x
    return best_x if best_x is not None else np.random.uniform(
        bounds[:, 0], bounds[:, 1])


# --------------------------------------------------------------------------- #
#  Hyperparameter Bounds & Decoder
# --------------------------------------------------------------------------- #

BOUNDS = np.array([
    [-4.0, -1.0],    # log_lr
    [0.0,  0.3],     # dropout
    [1.0,  8.0],     # n_layers
    [16.0, 256.0],   # n_neurons
    [4.0,  6.0],     # log_batch_size
])


def decode_lambda(lam):
    return {
        "lr":         10 ** np.clip(lam[0], -4, -1),
        "dropout":    np.clip(lam[1], 0, 0.3),
        "n_layers":   int(round(np.clip(lam[2], 1, 8))),
        "n_neurons":  int(round(np.clip(lam[3], 16, 256) / 16)) * 16,
        "batch_size": int(2 ** round(np.clip(lam[4], 4, 6))),
    }


# --------------------------------------------------------------------------- #
#  Trainer
# --------------------------------------------------------------------------- #

class HybridAdamBFGSTrainer:
    """
    Method B: Hybrid BO trainer.

    Outer loop: GP-based Bayesian optimization over hyperparameters
    Inner loop: Adam training + L-BFGS refinement
    """

    def __init__(self, n_bo_trials=15, adam_epochs=20, lbfgs_steps=10,
                 input_dim: int = 2):
        self.n_bo    = n_bo_trials
        self.n_adam  = adam_epochs
        self.n_lbfgs = lbfgs_steps
        self.input_dim = input_dim
        self.gp         = GaussianProcessSurrogate()
        self.best_val   = float("inf")
        self.best_model = None
        self.best_hp    = None
        self.history = {
            "val_loss_per_trial": [],
            "best_val_loss": [],
        }

    def train(self, n_samples=1000, train_loader=None, val_loader=None):
        t0 = time.time()
        if train_loader is None or val_loader is None:
            train_loader, val_loader, _ = generate_hamiltonian_data(n_samples=n_samples)

        # Collect full-batch tensors for L-BFGS
        Xs, ys = [], []
        for Xb, yb in train_loader:
            Xs.append(Xb)
            ys.append(yb)
        Xf, yf = torch.cat(Xs), torch.cat(ys)

        X_obs, y_obs = [], []
        print(f"  [Method B] Bayesian Optimization: {self.n_bo} trials")

        for t in range(self.n_bo):
            # Select hyperparameters
            if t < 5:
                lam = np.random.uniform(BOUNDS[:, 0], BOUNDS[:, 1])
            else:
                lam = suggest_next(self.gp, BOUNDS, min(y_obs))

            hp    = decode_lambda(lam)
            model = HamiltonianNN(hp["n_layers"], hp["n_neurons"], hp["dropout"],
                                  input_dim=self.input_dim)

            # Adam training
            opt = optim.Adam(model.parameters(), lr=hp["lr"])
            for _ in range(self.n_adam):
                for Xb, yb in train_loader:
                    opt.zero_grad()
                    nn.MSELoss()(model(Xb), yb).backward()
                    opt.step()

            # L-BFGS refinement
            lbfgs = optim.LBFGS(model.parameters(), max_iter=self.n_lbfgs)
            def closure():
                lbfgs.zero_grad()
                l = nn.MSELoss()(model(Xf), yf)
                l.backward()
                return l
            lbfgs.step(closure)

            # Evaluate
            vl, n = 0, 0
            with torch.no_grad():
                for Xb, yb in val_loader:
                    vl += nn.MSELoss()(model(Xb), yb).item()
                    n += 1
            vl /= n

            X_obs.append(lam)
            y_obs.append(vl)
            self.gp.update(np.stack(X_obs), np.array(y_obs))

            if vl < self.best_val:
                self.best_val   = vl
                self.best_model = deepcopy(model)
                self.best_hp    = hp

            self.history["val_loss_per_trial"].append(vl)
            self.history["best_val_loss"].append(self.best_val)
            print(f"    Trial {t:2d}/{self.n_bo} | "
                  f"Val: {vl:.6f} | Best: {self.best_val:.6f}")

        self.train_time = time.time() - t0
        return self.history

    def save(self, save_dir="results_hybrid"):
        os.makedirs(save_dir, exist_ok=True)
        if self.best_model:
            torch.save(self.best_model.state_dict(),
                       os.path.join(save_dir, "model.pt"))
        with open(os.path.join(save_dir, "history.json"), "w") as f:
            json.dump(self.history, f, indent=2)
        if self.best_hp:
            with open(os.path.join(save_dir, "hyperparameters.json"), "w") as f:
                json.dump(self.best_hp, f, indent=2)
