"""
Generates ROC curve figures for the real-world clinical showcase, using an
illustrative single seed (seed=0) per method per dataset. This is a standard
figure for any paper reporting a diagnostic classifier and was missing from
the repo -- the existing results only reported point-estimate AUROC/recall,
not the full ROC curve shape.

Reuses the exact model/data/training functions from
scripts/real_world_showcase.py so the numbers are consistent with the
headline 5-seed results reported elsewhere (this is seed=0 specifically,
illustrative of a single representative run, not a new experiment).

Usage:
    python scripts/generate_roc_curves.py
"""
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from real_world_showcase import (
    get_data, ClinicalMLP, decode_hp, INIT_HP, DEVICE, train_n_epochs,
)
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


@torch.no_grad()
def get_probs_and_labels(model, loader):
    model.eval()
    probs, labels = [], []
    for X, y in loader:
        X = X.to(DEVICE)
        probs.append(torch.sigmoid(model(X)).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def run_default_adam_get_probs(seed, dataset, input_dim):
    torch.manual_seed(seed); np.random.seed(seed)
    train_loader, val_loader, test_loader = get_data(seed, dataset)[:3]
    model = ClinicalMLP(input_dim, n_hidden=32, n_layers=2, dropout=0.2).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    train_n_epochs(model, train_loader, val_loader, criterion, lr=1e-3, wd=1e-4)
    return get_probs_and_labels(model, test_loader)


def run_method_c_get_probs(seed, dataset, input_dim, n_hmc_epochs=15, n_warmup=8):
    # Mirrors run_method_c in real_world_showcase.py but returns probabilities
    # on the test set for the best checkpoint, instead of just metrics.
    import math
    from hamiltonian import HyperparamState
    from symplectic_solver import HamiltonianMCMC
    from hybrid_hhd_abbo_improved import AdaptiveStepSizeController, PlateauDetector
    import config as base_config

    torch.manual_seed(seed); np.random.seed(seed)
    criterion = nn.BCEWithLogitsLoss()
    train_loader, val_loader, test_loader, _ = get_data(seed, dataset)

    hp_state = HyperparamState(INIT_HP, {
        "log_lr": (-5.0, -1.0), "dropout": (0.0, 0.6), "log_wd": (-6.0, -2.0),
        "n_hidden": (8.0, 128.0), "n_layers": (1.0, 4.0)})
    hp_state.frozen_hps = ["n_layers", "n_hidden"]
    raw_hp = {k: float(v.item()) for k, v in hp_state.values.items()}
    lr, dropout, wd, n_hidden, n_layers = decode_hp(raw_hp)
    model = ClinicalMLP(input_dim, n_hidden, n_layers, dropout).to(DEVICE)

    mcmc = HamiltonianMCMC(step_size=0.01, n_leapfrog=4, mass_theta=1.0,
                           mass_lambda=base_config.MASS_LAMBDA, temperature=1e9)
    plateau = PlateauDetector(patience=4, tol=5e-4)
    best_auroc, best_state = -1.0, None

    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for ep in range(n_warmup):
        cos_lr = 1e-6 + 0.5 * (lr - 1e-6) * (1 + math.cos(math.pi * ep / n_warmup))
        for pg in opt.param_groups:
            pg["lr"] = cos_lr
        model.train()
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); criterion(model(X), y).backward(); opt.step()
        probs, labels = get_probs_and_labels(model, val_loader)
        from sklearn.metrics import roc_auc_score
        score = roc_auc_score(labels, probs)
        if score > best_auroc:
            best_auroc, best_state = score, deepcopy(model.state_dict())

    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for ep in range(n_hmc_epochs):
        Xb, yb = next(iter(train_loader))
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        curr_loss = criterion(model(Xb), yb).item()
        mcmc.propose(model, hp_state, (Xb, yb), criterion, curr_loss)
        lr2, dropout2, wd2, _, _ = decode_hp({k: float(v.item()) for k, v in hp_state.values.items()})
        model.set_dropout(dropout2)
        opt = optim.Adam(model.parameters(), lr=lr2, weight_decay=wd2)
        for _ in range(3):
            model.train()
            for X, y in train_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                opt.zero_grad(); criterion(model(X), y).backward(); opt.step()
        probs, labels = get_probs_and_labels(model, val_loader)
        from sklearn.metrics import roc_auc_score
        score = roc_auc_score(labels, probs)
        if score > best_auroc:
            best_auroc, best_state = score, deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return get_probs_and_labels(model, test_loader)


def make_roc_figure(dataset_name, input_dim, ax):
    seed = 0
    curves = {}

    probs, labels = run_default_adam_get_probs(seed, dataset_name, input_dim)
    curves["Default Adam"] = (labels, probs)

    probs_c, labels_c = run_method_c_get_probs(seed, dataset_name, input_dim)
    curves["Method C (HHD-ABBO)"] = (labels_c, probs_c)

    for name, (y_true, y_score) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(dataset_name.replace("_", " ").title())
    ax.legend(fontsize=8, loc="lower right")


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    print("Breast cancer: training Default Adam + Method C (seed 0)...")
    make_roc_figure("breast_cancer", 30, axes[0])
    print("Diabetes: training Default Adam + Method C (seed 0)...")
    make_roc_figure("diabetes", 8, axes[1])
    plt.suptitle("ROC Curves: Default Adam vs. Method C (illustrative single-seed run)", y=1.02)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "roc_curves_real_world.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
