"""
Fashion-MNIST Deep MLP Testbed — Method C Showcase.

Benchmarks the Unified HHD-ABBO (Method C) hyperparameter optimizer against
three standard baselines on a configurable deep MLP trained on Fashion-MNIST:

  1. Default Adam      — Fixed HPs, no tuning
  2. Random Search     — 20 random HP configs, pick best
  3. Optuna TPE        — 20 trials via Tree-Parzen Estimator
  4. Method C (HHD-ABBO) — Three-phase curriculum: Adam → HMC → L-BFGS

Hyperparameter space (5D):
  log_lr   ∈ [-5, -1]    (learning rate in log₁₀ scale)
  dropout  ∈ [0.0, 0.7]
  log_wd   ∈ [-6, -1]    (weight decay in log₁₀ scale)
  n_hidden ∈ [64, 512]   (hidden layer width, rounded)
  n_layers ∈ [1, 5]      (number of hidden layers, rounded)

Usage:
  python scripts/fashion_mnist_testbed.py
  python scripts/fashion_mnist_testbed.py --seeds 0,1,2 --methods default,methodC
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import warnings
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from hamiltonian import HyperparamState
from symplectic_solver import HamiltonianMCMC
from hybrid_hhd_abbo_improved import (
    AdaptiveStepSizeController,
    PlateauDetector,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_DIR = os.path.join("results", "fashion_mnist")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Model: Configurable Deep MLP
# ═══════════════════════════════════════════════════════════════════════════

class FashionMLP(nn.Module):
    """
    Configurable deep MLP for Fashion-MNIST classification.

    Architecture:
        Input(784) → [Linear(n_hidden) → BatchNorm → ReLU → Dropout] × n_layers → Linear(10)
    """

    def __init__(self, n_hidden: int = 256, n_layers: int = 3,
                 dropout: float = 0.2, input_dim: int = 784, num_classes: int = 10):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim, n_hidden))
            layers.append(nn.BatchNorm1d(n_hidden))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_dim = n_hidden
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(n_hidden, num_classes)
        self._dropout_rate = dropout
        self._n_hidden = n_hidden
        self._n_layers = n_layers

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten 28×28 → 784
        return self.classifier(self.features(x))

    @property
    def dropout_rate(self) -> float:
        return self._dropout_rate

    @dropout_rate.setter
    def dropout_rate(self, p: float):
        self._dropout_rate = p
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.p = p

    def set_dropout(self, p: float):
        self.dropout_rate = p


# ═══════════════════════════════════════════════════════════════════════════
#  HP Decoding
# ═══════════════════════════════════════════════════════════════════════════

def decode_hp(hp_dict: dict) -> Tuple[float, float, float, int, int]:
    """Decode a raw HP dict → (lr, dropout, wd, n_hidden, n_layers)."""
    if "log_lr" in hp_dict:
        lr = 10 ** float(np.clip(hp_dict["log_lr"], -5, -1))
    else:
        lr = float(np.clip(hp_dict.get("lr", 1e-3), 1e-5, 1e-1))

    dropout = float(np.clip(hp_dict.get("dropout", 0.2), 0.0, 0.7))

    if "log_wd" in hp_dict:
        wd = 10 ** float(np.clip(hp_dict["log_wd"], -6, -1))
    else:
        wd = float(hp_dict.get("weight_decay", 1e-4))

    n_hidden = int(np.clip(round(hp_dict.get("n_hidden", 256)), 64, 512))
    n_layers = int(np.clip(round(hp_dict.get("n_layers", 3)), 1, 5))

    return lr, dropout, wd, n_hidden, n_layers


def _raw_hp(hp_state) -> dict:
    """Extract raw values from HyperparamState."""
    return {k: float(v.item()) for k, v in hp_state.values.items()}


# ═══════════════════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════════════════

def get_fmnist_loaders(batch_size: int = 128, train_size: int = None,
                       test_size: int = None, seed: int = 0):
    """Load Fashion-MNIST with normalisation and optional subsampling."""
    torch.manual_seed(seed)
    tr = train_size or config.FMNIST_TRAIN_SUBSET
    te = test_size or config.FMNIST_TEST_SUBSET

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),   # Fashion-MNIST stats
    ])

    train_ds = datasets.FashionMNIST("./data", train=True,  download=True, transform=tf)
    test_ds  = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    rng = np.random.RandomState(seed)
    tr_idx = rng.choice(len(train_ds), min(tr, len(train_ds)), replace=False)
    te_idx = rng.choice(len(test_ds),  min(te, len(test_ds)),  replace=False)

    train_loader = DataLoader(Subset(train_ds, tr_idx), batch_size=batch_size,
                              shuffle=True,  num_workers=0, pin_memory=False)
    test_loader  = DataLoader(Subset(test_ds,  te_idx), batch_size=256,
                              shuffle=False, num_workers=0, pin_memory=False)
    return train_loader, test_loader


def eval_accuracy(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            _, pred = torch.max(model(X), 1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    return correct / total if total > 0 else 0.0


def eval_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            total += criterion(model(X), y).item()
            n += 1
    return total / max(n, 1)


# ═══════════════════════════════════════════════════════════════════════════
#  Training Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _train_one_epoch(model, loader, criterion, optimizer, grad_clip=1.0):
    """Train model for one epoch, return average loss."""
    model.train()
    total_loss, n_batches = 0.0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def _train_n_epochs(model, loader, test_loader, criterion, lr, wd,
                    n_epochs=15, record_history=True):
    """Train for n_epochs with Adam, return (best_acc, history_dict)."""
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_acc = 0.0
    best_state = None
    history = {"train_loss": [], "val_acc": []}

    for ep in range(n_epochs):
        tl = _train_one_epoch(model, loader, criterion, opt)
        acc = eval_accuracy(model, test_loader)
        if record_history:
            history["train_loss"].append(tl)
            history["val_acc"].append(acc)
        if acc > best_acc:
            best_acc = acc
            best_state = deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_acc, history


# ═══════════════════════════════════════════════════════════════════════════
#  BASELINE 1: Default Adam (no tuning)
# ═══════════════════════════════════════════════════════════════════════════

def run_default_adam(seed: int = 0) -> dict:
    """Fixed HPs: lr=0.001, dropout=0.2, wd=1e-4, 256 hidden, 3 layers."""
    print(f"\n  [Default Adam] seed={seed}")
    np.random.seed(seed); torch.manual_seed(seed)
    t0 = time.time()

    train_loader, test_loader = get_fmnist_loaders(
        batch_size=config.FMNIST_BATCH_SIZE, seed=seed)
    model = FashionMLP(n_hidden=256, n_layers=3, dropout=0.2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    best_acc, history = _train_n_epochs(
        model, train_loader, test_loader, criterion,
        lr=0.001, wd=1e-4, n_epochs=config.FMNIST_TRAIN_EPOCHS)

    final_acc = eval_accuracy(model, test_loader)
    elapsed = time.time() - t0

    print(f"    Best: {best_acc:.2%} | Final: {final_acc:.2%} | Time: {elapsed:.1f}s")
    return {
        "method": "Default Adam",
        "seed": seed,
        "best_val_acc": best_acc,
        "final_val_acc": final_acc,
        "time": elapsed,
        "final_hps": {"lr": 0.001, "dropout": 0.2, "weight_decay": 1e-4,
                      "n_hidden": 256, "n_layers": 3},
        "acc_history": history["val_acc"],
        "loss_history": history["train_loss"],
    }


# ═══════════════════════════════════════════════════════════════════════════
#  BASELINE 2: Random Search
# ═══════════════════════════════════════════════════════════════════════════

def run_random_search(seed: int = 0, n_trials: int = None) -> dict:
    """Random HP sampling: sample n_trials configs, train each, pick best."""
    n_trials = n_trials or config.FMNIST_BO_TRIALS
    print(f"\n  [Random Search] seed={seed}, trials={n_trials}")
    np.random.seed(seed); torch.manual_seed(seed)
    t0 = time.time()

    criterion = nn.CrossEntropyLoss()
    rng = np.random.RandomState(seed)

    best_overall_acc = 0.0
    best_hp = {}
    best_acc_history = []
    cumulative_best = []

    for trial in range(n_trials):
        # Sample random HPs
        hp_dict = {
            "log_lr":   rng.uniform(-5.0, -1.0),
            "dropout":  rng.uniform(0.0, 0.7),
            "log_wd":   rng.uniform(-6.0, -1.0),
            "n_hidden": rng.uniform(64.0, 512.0),
            "n_layers": rng.uniform(1.0, 5.0),
        }
        lr, dropout, wd, n_hidden, n_layers = decode_hp(hp_dict)

        train_loader, test_loader = get_fmnist_loaders(
            batch_size=config.FMNIST_BATCH_SIZE, seed=seed)
        model = FashionMLP(n_hidden=n_hidden, n_layers=n_layers,
                           dropout=dropout).to(DEVICE)

        trial_acc, _ = _train_n_epochs(
            model, train_loader, test_loader, criterion,
            lr=lr, wd=wd, n_epochs=config.FMNIST_TRAIN_EPOCHS,
            record_history=False)

        if trial_acc > best_overall_acc:
            best_overall_acc = trial_acc
            best_hp = {"lr": lr, "dropout": dropout, "weight_decay": wd,
                       "n_hidden": n_hidden, "n_layers": n_layers}

        cumulative_best.append(best_overall_acc)

        if trial % 5 == 0 or trial == n_trials - 1:
            print(f"    Trial {trial:2d}/{n_trials} | "
                  f"This: {trial_acc:.2%} | Best: {best_overall_acc:.2%}")

    elapsed = time.time() - t0
    print(f"    DONE | Best: {best_overall_acc:.2%} | Time: {elapsed:.1f}s")
    return {
        "method": "Random Search",
        "seed": seed,
        "best_val_acc": best_overall_acc,
        "final_val_acc": best_overall_acc,
        "time": elapsed,
        "final_hps": best_hp,
        "acc_history": cumulative_best,
        "loss_history": [],
    }


# ═══════════════════════════════════════════════════════════════════════════
#  BASELINE 3: Optuna TPE
# ═══════════════════════════════════════════════════════════════════════════

def run_optuna_tpe(seed: int = 0, n_trials: int = None) -> dict:
    """Optuna Tree-Parzen Estimator for HP optimisation."""
    n_trials = n_trials or config.FMNIST_BO_TRIALS
    print(f"\n  [Optuna TPE] seed={seed}, trials={n_trials}")
    t0 = time.time()

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("    [SKIP] Optuna not installed. Returning empty result.")
        return {
            "method": "Optuna TPE",
            "seed": seed,
            "best_val_acc": 0.0,
            "final_val_acc": 0.0,
            "time": 0.0,
            "final_hps": {},
            "acc_history": [],
            "loss_history": [],
        }

    criterion = nn.CrossEntropyLoss()
    cumulative_best = []
    running_best = 0.0

    def objective(trial):
        nonlocal running_best
        np.random.seed(seed + trial.number)
        torch.manual_seed(seed + trial.number)

        log_lr   = trial.suggest_float("log_lr",   -5.0, -1.0)
        dropout  = trial.suggest_float("dropout",  0.0,  0.7)
        log_wd   = trial.suggest_float("log_wd",   -6.0, -1.0)
        n_hidden = trial.suggest_int("n_hidden",   64,   512, step=32)
        n_layers = trial.suggest_int("n_layers",   1,    5)

        lr = 10 ** log_lr
        wd = 10 ** log_wd

        train_loader, test_loader = get_fmnist_loaders(
            batch_size=config.FMNIST_BATCH_SIZE, seed=seed)
        model = FashionMLP(n_hidden=n_hidden, n_layers=n_layers,
                           dropout=dropout).to(DEVICE)

        trial_acc, _ = _train_n_epochs(
            model, train_loader, test_loader, criterion,
            lr=lr, wd=wd, n_epochs=config.FMNIST_TRAIN_EPOCHS,
            record_history=False)

        running_best = max(running_best, trial_acc)
        cumulative_best.append(running_best)
        return trial_acc

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_trial = study.best_trial
    best_hp = {
        "lr": 10 ** best_trial.params["log_lr"],
        "dropout": best_trial.params["dropout"],
        "weight_decay": 10 ** best_trial.params["log_wd"],
        "n_hidden": best_trial.params["n_hidden"],
        "n_layers": best_trial.params["n_layers"],
    }

    elapsed = time.time() - t0
    print(f"    DONE | Best: {study.best_value:.2%} | Time: {elapsed:.1f}s")
    return {
        "method": "Optuna TPE",
        "seed": seed,
        "best_val_acc": study.best_value,
        "final_val_acc": study.best_value,
        "time": elapsed,
        "final_hps": best_hp,
        "acc_history": cumulative_best,
        "loss_history": [],
    }


# ═══════════════════════════════════════════════════════════════════════════
#  METHOD C: Unified HHD-ABBO
# ═══════════════════════════════════════════════════════════════════════════

def run_method_c(seed: int = 0) -> dict:
    """
    Method C (Unified HHD-ABBO) adapted for Fashion-MNIST classification.

    Three-phase curriculum:
      Phase 1: Adam warmup with cosine LR annealing
      Phase 2: HMC co-evolution of weights + HPs with adaptive step control
      Phase 3: L-BFGS refinement on plateau detection + final polish
    """
    print(f"\n  [Method C] seed={seed}")
    np.random.seed(seed); torch.manual_seed(seed)
    t0 = time.time()

    n_warmup = config.FMNIST_WARMUP_EPOCHS
    n_hmc    = config.FMNIST_HMC_EPOCHS
    criterion = nn.CrossEntropyLoss()

    # Initialise HP state
    hp_state = HyperparamState(config.FMNIST_INIT_HP, config.FMNIST_HP_SPACE)
    hp_state.frozen_hps = ["n_layers", "n_hidden"]  # Freeze structural HPs

    lr, dropout, wd, n_hidden, n_layers = decode_hp(_raw_hp(hp_state))
    train_loader, test_loader = get_fmnist_loaders(
        batch_size=config.FMNIST_BATCH_SIZE, seed=seed)

    model = FashionMLP(n_hidden=n_hidden, n_layers=n_layers,
                       dropout=dropout).to(DEVICE)

    mcmc = HamiltonianMCMC(
        step_size=0.005, n_leapfrog=4,
        mass_theta=1.0, mass_lambda=config.MASS_LAMBDA,
        temperature=1e9)
    step_ctrl = AdaptiveStepSizeController(initial_step=0.005, target_accept=0.65)
    plateau   = PlateauDetector(patience=4, tol=5e-4)

    # Recording
    history = {
        "train_loss": [], "val_acc": [], "best_val_acc": [],
        "acceptance_rate": [], "step_size": [],
        "hp_trajectory": {k: [] for k in config.FMNIST_INIT_HP},
    }
    best_acc = 0.0
    best_state = None
    best_hp_dict = None

    # ── Phase 1: Adam Warmup ──
    print(f"    Phase 1: Adam warmup ({n_warmup} epochs)")
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for ep in range(n_warmup):
        # Cosine anneal
        import math
        cos_lr = 1e-6 + 0.5 * (lr - 1e-6) * (1 + math.cos(math.pi * ep / n_warmup))
        for pg in opt.param_groups:
            pg["lr"] = cos_lr
        tl = _train_one_epoch(model, train_loader, criterion, opt)
        acc = eval_accuracy(model, test_loader)
        history["train_loss"].append(tl)
        history["val_acc"].append(acc)
        history["best_val_acc"].append(max(acc, best_acc))
        history["acceptance_rate"].append(0.0)
        history["step_size"].append(mcmc.leapfrog.eps)
        for k in config.FMNIST_INIT_HP:
            history["hp_trajectory"][k].append(float(hp_state.values[k].item()))
        if acc > best_acc:
            best_acc = acc
            best_state = deepcopy(model.state_dict())
            best_hp_dict = _raw_hp(hp_state)

    print(f"    Post-warmup accuracy: {best_acc:.2%}")

    # ── Phase 2+3: HMC co-evolution + L-BFGS ──
    print(f"    Phase 2+3: HMC co-evolution + L-BFGS ({n_hmc} epochs)")
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lbfgs_count = 0

    val_iter = iter(val_loader)
    for ep in range(n_hmc):
        # HMC proposal
        Xb, yb = next(iter(train_loader))
        Xb, yb = Xb.view(Xb.size(0), -1).to(DEVICE), yb.to(DEVICE)

        try:
            X_val_b, y_val_b = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            X_val_b, y_val_b = next(val_iter)
        X_val_b = X_val_b.view(X_val_b.size(0), -1).to(DEVICE)
        y_val_b = y_val_b.to(DEVICE)

        model.eval()
        with torch.no_grad():
            curr_loss = criterion(model(X_val_b), y_val_b).item()
        model.train()

        acc_flag, curr_loss = mcmc.propose(
            model, hp_state, (Xb, yb), criterion, curr_loss, val_batch=(X_val_b, y_val_b))

        # Adaptive step size
        mcmc.leapfrog.eps = step_ctrl.update(mcmc.acceptance_rate)

        # Update model HPs
        lr, dropout, wd, n_hidden, n_layers = decode_hp(_raw_hp(hp_state))
        model.set_dropout(dropout)

        # Refresh Adam with updated LR
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        # Multi-step Adam micro-epochs (3 per HMC step)
        for _micro in range(3):
            _train_one_epoch(model, train_loader, criterion, opt)

        # Evaluate
        train_loss = eval_loss(model, train_loader, criterion)
        acc = eval_accuracy(model, test_loader)

        # L-BFGS on plateau
        if plateau.update(train_loss):
            lbfgs_count += 1
            # Collect a few batches for L-BFGS
            Xs, ys_list = [], []
            for X, y in train_loader:
                Xs.append(X.view(X.size(0), -1)); ys_list.append(y)
                if len(Xs) >= 4:
                    break
            Xf = torch.cat(Xs).to(DEVICE)
            yf = torch.cat(ys_list).to(DEVICE)
            lbfgs_opt = optim.LBFGS(model.parameters(), max_iter=15, lr=0.01,
                                    line_search_fn="strong_wolfe")
            def closure():
                lbfgs_opt.zero_grad()
                l = criterion(model(Xf), yf)
                l.backward()
                return l
            try:
                lbfgs_opt.step(closure)
            except Exception:
                pass
            plateau.reset()
            # Refresh Adam
            opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            acc = eval_accuracy(model, test_loader)

        # Record
        history["train_loss"].append(train_loss)
        history["val_acc"].append(acc)
        history["best_val_acc"].append(max(acc, best_acc))
        history["acceptance_rate"].append(mcmc.acceptance_rate)
        history["step_size"].append(mcmc.leapfrog.eps)
        for k in config.FMNIST_INIT_HP:
            history["hp_trajectory"][k].append(float(hp_state.values[k].item()))

        if acc > best_acc:
            best_acc = acc
            best_state = deepcopy(model.state_dict())
            best_hp_dict = _raw_hp(hp_state)

        if ep % 3 == 0 or ep == n_hmc - 1:
            tag = "ACC" if acc_flag else "REJ"
            print(f"      Ep {ep:2d}/{n_hmc} [{tag}] | Acc: {acc:.2%} | "
                  f"Best: {best_acc:.2%} | HMC: {mcmc.acceptance_rate:.1%}")

    # ── Final L-BFGS Polish ──
    print("    Phase 3: Final L-BFGS polish")
    if best_state is not None:
        model.load_state_dict(best_state)

    Xs, ys_list = [], []
    for X, y in train_loader:
        Xs.append(X.view(X.size(0), -1)); ys_list.append(y)
    Xf = torch.cat(Xs).to(DEVICE)
    yf = torch.cat(ys_list).to(DEVICE)

    lbfgs_final = optim.LBFGS(model.parameters(), max_iter=50, lr=1.0,
                               line_search_fn="strong_wolfe", history_size=20)
    def closure_final():
        lbfgs_final.zero_grad()
        l = criterion(model(Xf), yf)
        l.backward()
        return l
    try:
        lbfgs_final.step(closure_final)
    except Exception as e:
        print(f"      L-BFGS warning: {e}")

    final_acc = eval_accuracy(model, test_loader)
    if final_acc > best_acc:
        best_acc = final_acc

    history["val_acc"].append(final_acc)
    history["best_val_acc"].append(best_acc)
    history["train_loss"].append(eval_loss(model, train_loader, criterion))

    elapsed = time.time() - t0
    print(f"    DONE | Best: {best_acc:.2%} | Final: {final_acc:.2%} | "
          f"Time: {elapsed:.1f}s | L-BFGS triggered {lbfgs_count}x")

    return {
        "method": "Method C (HHD-ABBO)",
        "seed": seed,
        "best_val_acc": best_acc,
        "final_val_acc": final_acc,
        "time": elapsed,
        "final_hps": best_hp_dict or _raw_hp(hp_state),
        "acc_history": history["val_acc"],
        "loss_history": history["train_loss"],
        "hp_trajectory": history["hp_trajectory"],
        "acceptance_rate": history["acceptance_rate"],
        "step_size_history": history["step_size"],
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Orchestrator
# ═══════════════════════════════════════════════════════════════════════════

METHOD_FNS = {
    "default":  run_default_adam,
    "random":   run_random_search,
    "optuna":   run_optuna_tpe,
    "methodC":  run_method_c,
}

METHOD_LABELS = {
    "default":  "Default Adam",
    "random":   "Random Search",
    "optuna":   "Optuna TPE",
    "methodC":  "Method C (HHD-ABBO)",
}


def run_full_testbed(
    seeds: Optional[List[int]] = None,
    methods: Optional[List[str]] = None,
) -> dict:
    """Run all methods across seeds and save comprehensive results."""
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]
    if methods is None:
        methods = ["default", "random", "optuna", "methodC"]

    print("=" * 70)
    print("  FASHION-MNIST DEEP MLP TESTBED")
    print(f"  Methods: {methods}  |  Seeds: {seeds}")
    print("=" * 70)

    all_results: Dict[str, List[dict]] = {m: [] for m in methods}

    for method_key in methods:
        fn = METHOD_FNS[method_key]
        label = METHOD_LABELS[method_key]
        print(f"\n{'-' * 70}")
        print(f"  Running: {label}")
        print(f"{'-' * 70}")

        for seed in seeds:
            try:
                result = fn(seed=seed)
                all_results[method_key].append(result)
            except Exception as e:
                print(f"    [ERROR] {label} seed={seed}: {e}")
                traceback.print_exc()

    # ── Summary Table ──
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY: Fashion-MNIST Deep MLP")
    print("=" * 70)
    print(f"{'Method':<30} | {'Best Acc (mean±std)':>20} | {'Time (mean)':>12}")
    print("-" * 70)

    summary = {}
    for method_key in methods:
        runs = all_results[method_key]
        if not runs:
            continue
        accs  = [r["best_val_acc"] for r in runs]
        times = [r["time"]         for r in runs]
        mean_acc = np.mean(accs)
        std_acc  = np.std(accs)
        mean_time = np.mean(times)
        label = METHOD_LABELS[method_key]
        print(f"{label:<30} | {mean_acc:>8.2%} ± {std_acc:<8.4f} | {mean_time:>9.1f}s")

        summary[method_key] = {
            "label": label,
            "mean_best_acc": float(mean_acc),
            "std_best_acc":  float(std_acc),
            "mean_time":     float(mean_time),
            "per_seed": runs,
        }

    print("=" * 70)

    # ── Save Results ──
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(os.path.join(RESULTS_DIR, "testbed_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    with open(os.path.join(RESULTS_DIR, "all_runs.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n  Results saved to {RESULTS_DIR}/")
    return summary


# ═══════════════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fashion-MNIST Deep MLP Testbed")
    p.add_argument("--seeds",   type=str, default="0,1,2,3,4",
                   help="Comma-separated seeds")
    p.add_argument("--methods", type=str, default="default,random,optuna,methodC",
                   help="Comma-separated methods: default,random,optuna,methodC")
    args = p.parse_args()

    seeds   = [int(s.strip()) for s in args.seeds.split(",")]
    methods = [m.strip() for m in args.methods.split(",")]
    run_full_testbed(seeds=seeds, methods=methods)
"""
Description: Self-contained benchmark script for Fashion-MNIST Deep MLP testbed.
"""
