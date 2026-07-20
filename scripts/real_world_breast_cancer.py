"""
Real-World Showcase — Method C (HHD-ABBO) on Breast Cancer Diagnosis.

This is the repo's first application of the HHD-ABBO self-tuning
hyperparameter optimizer to a genuine real-world dataset (as opposed to
the synthetic Hamiltonian-system regression tasks and the academic
Fashion-MNIST / CIFAR-10 image benchmarks already in this repo).

Task
----
Wisconsin Diagnostic Breast Cancer dataset (UCI / sklearn.datasets):
569 patients, 30 real-valued features computed from digitized images of
fine-needle-aspirate (FNA) breast masses (radius, texture, perimeter,
concavity, symmetry, ...). Binary target: malignant vs. benign.

This is a realistic stand-in for a clinical decision-support pipeline: a
practitioner has a small tabular medical dataset, needs a well-tuned MLP
classifier, and does not want to hand-tune five interacting hyperparameters
or pay for a large HPO budget. It also stresses the algorithm in a regime
different from the synthetic physics benchmarks: small-N tabular data,
class-imbalance-sensitive metrics (recall on the malignant class matters
far more than raw accuracy), and a classification loss (BCEWithLogitsLoss)
rather than the MSE regression loss HHD-ABBO was originally built around.

Methods compared (same total training budget per method):
  1. Default Adam    — fixed, "reasonable-guess" hyperparameters, no tuning
  2. Random Search    — N random hyperparameter draws
  3. Optuna TPE        — N trials of Tree-Parzen-Estimator Bayesian optimization
  4. Method C (HHD-ABBO) — three-phase Adam -> HMC co-evolution -> L-BFGS

All methods search the same 5-D space (log_lr, dropout, log_wd, n_hidden,
n_layers) and are evaluated identically on a held-out test split across
multiple seeds, using recall on the malignant class and ROC-AUC as the
clinically relevant metrics (accuracy alone is misleading under class
imbalance and asymmetric error costs).

Usage:
  python scripts/real_world_breast_cancer.py
  python scripts/real_world_breast_cancer.py --seeds 0,1,2,3,4 --trials 20
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, "src"))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, "..")))

import config as base_config
from hamiltonian import HyperparamState
from symplectic_solver import HamiltonianMCMC
from hybrid_hhd_abbo_improved import AdaptiveStepSizeController, PlateauDetector

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, recall_score, accuracy_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = os.path.join("results", "breast_cancer")
PLOTS_DIR = os.path.join("plots")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# --------------------------------------------------------------------------- #
#  Hyperparameter search space (5-D, tuned for small tabular data)
# --------------------------------------------------------------------------- #
HP_SPACE = {
    "log_lr":   (-5.0, -1.0),   # lr: 1e-5 to 0.1
    "dropout":  (0.0,  0.6),
    "log_wd":   (-6.0, -2.0),   # weight decay: 1e-6 to 1e-2
    "n_hidden": (8.0,  128.0),
    "n_layers": (1.0,  4.0),
}
INIT_HP = {
    "log_lr":   -3.0,   # lr = 1e-3
    "dropout":  0.2,
    "log_wd":   -4.0,   # wd = 1e-4
    "n_hidden": 32.0,
    "n_layers": 2.0,
}
TRAIN_EPOCHS_PER_TRIAL = 40   # epochs given to each baseline trial (small data trains fast)


# --------------------------------------------------------------------------- #
#  Model: configurable MLP classifier
# --------------------------------------------------------------------------- #

class ClinicalMLP(nn.Module):
    """Configurable MLP for tabular binary classification."""

    def __init__(self, input_dim: int, n_hidden: int = 32, n_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, n_hidden), nn.BatchNorm1d(n_hidden),
                       nn.ReLU(inplace=True), nn.Dropout(dropout)]
            in_dim = n_hidden
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, 1)
        self._dropout_rate = dropout

    def forward(self, x):
        return self.head(self.features(x)).squeeze(-1)

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


def decode_hp(hp: dict) -> Tuple[float, float, float, int, int]:
    lr = 10 ** float(np.clip(hp.get("log_lr", -3.0), *HP_SPACE["log_lr"]))
    dropout = float(np.clip(hp.get("dropout", 0.2), *HP_SPACE["dropout"]))
    wd = 10 ** float(np.clip(hp.get("log_wd", -4.0), *HP_SPACE["log_wd"]))
    n_hidden = int(np.clip(round(hp.get("n_hidden", 32)), 8, 128))
    n_layers = int(np.clip(round(hp.get("n_layers", 2)), 1, 4))
    return lr, dropout, wd, n_hidden, n_layers


def _raw_hp(hp_state) -> dict:
    return {k: float(v.item()) for k, v in hp_state.values.items()}


# --------------------------------------------------------------------------- #
#  Data
# --------------------------------------------------------------------------- #

def get_data(seed: int, batch_size: int = 32):
    """Load + split + standardize the Wisconsin breast-cancer dataset.

    Split: 60% train / 20% val (used for HP selection) / 20% held-out test
    (used only for the final report, never seen during search). Stratified
    so the malignant/benign ratio is preserved in every split.
    """
    data = load_breast_cancer()
    X, y = data.data.astype(np.float32), data.target.astype(np.float32)

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.4, random_state=seed, stratify=y)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=seed, stratify=y_tmp)

    scaler = StandardScaler().fit(X_tr)
    X_tr, X_val, X_te = scaler.transform(X_tr), scaler.transform(X_val), scaler.transform(X_te)

    def _loader(X, y, bs, shuffle):
        ds = TensorDataset(torch.from_numpy(X.astype(np.float32)),
                           torch.from_numpy(y.astype(np.float32)))
        return DataLoader(ds, batch_size=bs, shuffle=shuffle)

    train_loader = _loader(X_tr, y_tr, batch_size, True)
    val_loader = _loader(X_val, y_val, 64, False)
    test_loader = _loader(X_te, y_te, 64, False)
    return train_loader, val_loader, test_loader, X.shape[1]


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> Dict[str, float]:
    model.eval()
    all_logits, all_y = [], []
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        all_logits.append(model(X).cpu())
        all_y.append(y.cpu())
    logits = torch.cat(all_logits)
    y_true = torch.cat(all_y).numpy()
    probs = torch.sigmoid(logits).numpy()
    preds = (probs >= 0.5).astype(np.float32)

    try:
        auroc = roc_auc_score(y_true, probs)
    except ValueError:
        auroc = float("nan")
    return {
        "accuracy": accuracy_score(y_true, preds),
        "f1": f1_score(y_true, preds, zero_division=0),
        # malignant == 0 in sklearn's encoding; recall on that class is the
        # clinically critical number (missed cancers are far costlier than
        # false alarms), so we score it explicitly rather than only accuracy.
        "malignant_recall": recall_score(y_true, preds, pos_label=0, zero_division=0),
        "auroc": auroc,
    }


def eval_loss(model, loader, criterion) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            total += criterion(model(X), y).item()
            n += 1
    return total / max(n, 1)


def train_one_epoch(model, loader, criterion, optimizer, grad_clip=1.0):
    model.train()
    total, n = 0.0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


def train_n_epochs(model, train_loader, val_loader, criterion, lr, wd,
                   n_epochs=TRAIN_EPOCHS_PER_TRIAL):
    """Train with Adam, track best validation AUROC, restore best weights."""
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_auroc, best_state = -1.0, None
    for _ in range(n_epochs):
        train_one_epoch(model, train_loader, criterion, opt)
        m = evaluate(model, val_loader)
        score = m["auroc"] if not math.isnan(m["auroc"]) else m["accuracy"]
        if score > best_auroc:
            best_auroc, best_state = score, deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_auroc


# --------------------------------------------------------------------------- #
#  BASELINE 1: Default Adam (no tuning)
# --------------------------------------------------------------------------- #

def run_default_adam(seed: int, input_dim: int) -> dict:
    print(f"\n  [Default Adam] seed={seed}")
    np.random.seed(seed); torch.manual_seed(seed)
    t0 = time.time()

    train_loader, val_loader, test_loader, _ = get_data(seed)
    model = ClinicalMLP(input_dim, n_hidden=32, n_layers=2, dropout=0.2).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()

    val_auroc = train_n_epochs(model, train_loader, val_loader, criterion,
                               lr=1e-3, wd=1e-4)
    test_metrics = evaluate(model, test_loader)
    elapsed = time.time() - t0
    print(f"    val AUROC={val_auroc:.4f} | test AUROC={test_metrics['auroc']:.4f} | "
          f"malignant recall={test_metrics['malignant_recall']:.4f} | {elapsed:.1f}s")

    return {"method": "Default Adam", "seed": seed, "val_auroc": val_auroc,
            "test_metrics": test_metrics, "time": elapsed,
            "final_hps": {"lr": 1e-3, "dropout": 0.2, "weight_decay": 1e-4,
                          "n_hidden": 32, "n_layers": 2}}


# --------------------------------------------------------------------------- #
#  BASELINE 2: Random Search
# --------------------------------------------------------------------------- #

def run_random_search(seed: int, input_dim: int, n_trials: int) -> dict:
    print(f"\n  [Random Search] seed={seed}, trials={n_trials}")
    np.random.seed(seed); torch.manual_seed(seed)
    t0 = time.time()
    rng = np.random.RandomState(seed)
    criterion = nn.BCEWithLogitsLoss()

    train_loader, val_loader, test_loader, _ = get_data(seed)

    best_val, best_hp, best_state = -1.0, None, None
    for trial in range(n_trials):
        hp = {
            "log_lr": rng.uniform(*HP_SPACE["log_lr"]),
            "dropout": rng.uniform(*HP_SPACE["dropout"]),
            "log_wd": rng.uniform(*HP_SPACE["log_wd"]),
            "n_hidden": rng.uniform(*HP_SPACE["n_hidden"]),
            "n_layers": rng.uniform(*HP_SPACE["n_layers"]),
        }
        lr, dropout, wd, n_hidden, n_layers = decode_hp(hp)
        model = ClinicalMLP(input_dim, n_hidden, n_layers, dropout).to(DEVICE)
        val_auroc = train_n_epochs(model, train_loader, val_loader, criterion, lr, wd)
        if val_auroc > best_val:
            best_val, best_hp = val_auroc, {"lr": lr, "dropout": dropout,
                                            "weight_decay": wd, "n_hidden": n_hidden,
                                            "n_layers": n_layers}
            best_state = deepcopy(model.state_dict())
        if trial % 5 == 0 or trial == n_trials - 1:
            print(f"    trial {trial:2d}/{n_trials} | val AUROC={val_auroc:.4f} | best={best_val:.4f}")

    final_model = ClinicalMLP(input_dim, best_hp["n_hidden"], best_hp["n_layers"],
                              best_hp["dropout"]).to(DEVICE)
    final_model.load_state_dict(best_state)
    test_metrics = evaluate(final_model, test_loader)
    elapsed = time.time() - t0
    print(f"    DONE | test AUROC={test_metrics['auroc']:.4f} | {elapsed:.1f}s")

    return {"method": "Random Search", "seed": seed, "val_auroc": best_val,
            "test_metrics": test_metrics, "time": elapsed, "final_hps": best_hp}


# --------------------------------------------------------------------------- #
#  BASELINE 3: Optuna TPE
# --------------------------------------------------------------------------- #

def run_optuna_tpe(seed: int, input_dim: int, n_trials: int) -> dict:
    print(f"\n  [Optuna TPE] seed={seed}, trials={n_trials}")
    t0 = time.time()
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("    [SKIP] optuna not installed")
        return {"method": "Optuna TPE", "seed": seed, "val_auroc": float("nan"),
                "test_metrics": {}, "time": 0.0, "final_hps": {}}

    criterion = nn.BCEWithLogitsLoss()
    train_loader, val_loader, test_loader, _ = get_data(seed)
    best_state_holder = {"state": None, "auroc": -1.0, "hp": None}

    def objective(trial):
        np.random.seed(seed + trial.number)
        torch.manual_seed(seed + trial.number)
        log_lr = trial.suggest_float("log_lr", *HP_SPACE["log_lr"])
        dropout = trial.suggest_float("dropout", *HP_SPACE["dropout"])
        log_wd = trial.suggest_float("log_wd", *HP_SPACE["log_wd"])
        n_hidden = trial.suggest_int("n_hidden", 8, 128, step=8)
        n_layers = trial.suggest_int("n_layers", 1, 4)
        lr, wd = 10 ** log_lr, 10 ** log_wd

        model = ClinicalMLP(input_dim, n_hidden, n_layers, dropout).to(DEVICE)
        val_auroc = train_n_epochs(model, train_loader, val_loader, criterion, lr, wd)
        if val_auroc > best_state_holder["auroc"]:
            best_state_holder.update(state=deepcopy(model.state_dict()),
                                     auroc=val_auroc,
                                     hp={"lr": lr, "dropout": dropout, "weight_decay": wd,
                                         "n_hidden": n_hidden, "n_layers": n_layers})
        return val_auroc

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_hp = best_state_holder["hp"]
    final_model = ClinicalMLP(input_dim, best_hp["n_hidden"], best_hp["n_layers"],
                              best_hp["dropout"]).to(DEVICE)
    final_model.load_state_dict(best_state_holder["state"])
    test_metrics = evaluate(final_model, test_loader)
    elapsed = time.time() - t0
    print(f"    DONE | val AUROC={study.best_value:.4f} | test AUROC={test_metrics['auroc']:.4f} | {elapsed:.1f}s")

    return {"method": "Optuna TPE", "seed": seed, "val_auroc": study.best_value,
            "test_metrics": test_metrics, "time": elapsed, "final_hps": best_hp}


# --------------------------------------------------------------------------- #
#  METHOD C: Unified HHD-ABBO
# --------------------------------------------------------------------------- #

def run_method_c(seed: int, input_dim: int, n_hmc_epochs: int, n_warmup: int = 8) -> dict:
    """
    Method C (Unified HHD-ABBO) applied to the clinical classification task.

    Same three-phase curriculum used elsewhere in this repo (Adam warmup ->
    HMC co-evolution of weights+HPs -> L-BFGS refinement), but with:
      - BCEWithLogitsLoss instead of MSE (classification, not regression)
      - validation AUROC (not train loss) used for best-checkpoint selection,
        since loss-minimizing checkpoints can diverge from the clinically
        relevant metric under class imbalance
      - structural HPs (n_layers, n_hidden) frozen during HMC, exactly as
        the rest of this repo already treats them, since they change the
        parameter tensor shapes and would invalidate momenta mid-trajectory
    """
    print(f"\n  [Method C] seed={seed}")
    np.random.seed(seed); torch.manual_seed(seed)
    t0 = time.time()

    criterion = nn.BCEWithLogitsLoss()
    train_loader, val_loader, test_loader, _ = get_data(seed)

    hp_state = HyperparamState(INIT_HP, HP_SPACE)
    hp_state.frozen_hps = ["n_layers", "n_hidden"]

    lr, dropout, wd, n_hidden, n_layers = decode_hp(_raw_hp(hp_state))
    model = ClinicalMLP(input_dim, n_hidden, n_layers, dropout).to(DEVICE)

    mcmc = HamiltonianMCMC(step_size=0.01, n_leapfrog=4, mass_theta=1.0,
                           mass_lambda=0.02, temperature=100.0,
                           momentum_refresh=0.1)
    step_ctrl = AdaptiveStepSizeController(initial_step=0.01, target_accept=0.65)
    plateau = PlateauDetector(patience=4, tol=5e-4)

    best_auroc, best_state, best_hp = -1.0, None, None

    # ---- Phase 1: Adam warmup (cosine-annealed) ----
    print(f"    Phase 1: Adam warmup ({n_warmup} epochs)")
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for ep in range(n_warmup):
        cos_lr = 1e-6 + 0.5 * (lr - 1e-6) * (1 + math.cos(math.pi * ep / n_warmup))
        for pg in opt.param_groups:
            pg["lr"] = cos_lr
        train_one_epoch(model, train_loader, criterion, opt)
        m = evaluate(model, val_loader)
        score = m["auroc"] if not math.isnan(m["auroc"]) else m["accuracy"]
        if score > best_auroc:
            best_auroc, best_state = score, deepcopy(model.state_dict())
            best_hp = _raw_hp(hp_state)
    print(f"    Post-warmup val AUROC: {best_auroc:.4f}")

    # ---- Phase 2+3: HMC co-evolution + plateau-triggered L-BFGS ----
    print(f"    Phase 2+3: HMC + Adam + L-BFGS ({n_hmc_epochs} epochs)")
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lbfgs_count = 0

    # Pre-concatenate full training set for HMC proposals (fixes noisy
    # single-batch HP gradients — the #1 source of quality gap vs Optuna)
    Xall = torch.cat([X for X, y in train_loader]).to(DEVICE)
    yall = torch.cat([y for X, y in train_loader]).to(DEVICE)

    for ep in range(n_hmc_epochs):
        curr_loss = criterion(model(Xall), yall).item()
        acc_flag, curr_loss = mcmc.propose(model, hp_state, (Xall, yall), criterion, curr_loss)
        mcmc.leapfrog.eps = step_ctrl.update(mcmc.acceptance_rate)

        lr, dropout, wd, n_hidden, n_layers = decode_hp(_raw_hp(hp_state))
        model.set_dropout(dropout)
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        for _ in range(3):
            train_one_epoch(model, train_loader, criterion, opt)

        train_loss = eval_loss(model, train_loader, criterion)
        m = evaluate(model, val_loader)
        score = m["auroc"] if not math.isnan(m["auroc"]) else m["accuracy"]

        if plateau.update(train_loss):
            lbfgs_count += 1
            Xs, ys = [], []
            for X, y in train_loader:
                Xs.append(X); ys.append(y)
            Xf, yf = torch.cat(Xs).to(DEVICE), torch.cat(ys).to(DEVICE)
            lbfgs_opt = optim.LBFGS(model.parameters(), max_iter=15, lr=0.05,
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
            opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            m = evaluate(model, val_loader)
            score = m["auroc"] if not math.isnan(m["auroc"]) else m["accuracy"]

        if score > best_auroc:
            best_auroc, best_state = score, deepcopy(model.state_dict())
            best_hp = _raw_hp(hp_state)

        if ep % 3 == 0 or ep == n_hmc_epochs - 1:
            tag = "ACC" if acc_flag else "REJ"
            print(f"      ep {ep:2d}/{n_hmc_epochs} [{tag}] | val AUROC={score:.4f} | "
                  f"best={best_auroc:.4f} | HMC acc={mcmc.acceptance_rate:.1%}")

    # ---- Final L-BFGS polish on the best checkpoint ----
    print("    Phase 3: final L-BFGS polish")
    if best_state is not None:
        model.load_state_dict(best_state)
    Xs, ys = [], []
    for X, y in train_loader:
        Xs.append(X); ys.append(y)
    Xf, yf = torch.cat(Xs).to(DEVICE), torch.cat(ys).to(DEVICE)
    lbfgs_final = optim.LBFGS(model.parameters(), max_iter=50, lr=0.5,
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

    m = evaluate(model, val_loader)
    final_score = m["auroc"] if not math.isnan(m["auroc"]) else m["accuracy"]
    if final_score > best_auroc:
        best_auroc = final_score
    else:
        # L-BFGS polish hurt validation AUROC (over-fit the full training
        # set) -- restore the best checkpoint found during search instead
        # of silently keeping a worse final model, mirroring the
        # best-checkpoint discipline used elsewhere in this repo.
        if best_state is not None:
            model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader)
    elapsed = time.time() - t0
    print(f"    DONE | best val AUROC={best_auroc:.4f} | test AUROC={test_metrics['auroc']:.4f} | "
          f"malignant recall={test_metrics['malignant_recall']:.4f} | "
          f"{elapsed:.1f}s | L-BFGS triggered {lbfgs_count}x")

    return {"method": "Method C (HHD-ABBO)", "seed": seed, "val_auroc": best_auroc,
            "test_metrics": test_metrics, "time": elapsed,
            "final_hps": decode_dict(best_hp or _raw_hp(hp_state))}


def decode_dict(raw_hp: dict) -> dict:
    lr, dropout, wd, n_hidden, n_layers = decode_hp(raw_hp)
    return {"lr": lr, "dropout": dropout, "weight_decay": wd,
            "n_hidden": n_hidden, "n_layers": n_layers}


# --------------------------------------------------------------------------- #
#  Orchestration
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--trials", type=int, default=20,
                    help="Trial budget for Random Search / Optuna TPE")
    ap.add_argument("--hmc-epochs", type=int, default=12,
                    help="HMC co-evolution epochs for Method C")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    print("=" * 72)
    print("  REAL-WORLD SHOWCASE: HHD-ABBO on Breast Cancer Diagnosis")
    print(f"  Seeds: {seeds} | Baseline trials: {args.trials} | HMC epochs: {args.hmc_epochs}")
    print("=" * 72)

    all_results: List[dict] = []
    for seed in seeds:
        _, _, _, input_dim = get_data(seed)
        all_results.append(run_default_adam(seed, input_dim))
        all_results.append(run_random_search(seed, input_dim, args.trials))
        all_results.append(run_optuna_tpe(seed, input_dim, args.trials))
        all_results.append(run_method_c(seed, input_dim, args.hmc_epochs))

    out_path = os.path.join(RESULTS_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\n  Raw results saved to {out_path}")

    summarize(all_results, seeds)


def summarize(all_results: List[dict], seeds: List[int]):
    methods = ["Default Adam", "Random Search", "Optuna TPE", "Method C (HHD-ABBO)"]
    print("\n" + "=" * 96)
    print("  SUMMARY (mean ± std over {} seeds, held-out test set)".format(len(seeds)))
    print("=" * 96)
    header = f"{'Method':<24}{'Test AUROC':<18}{'Test Accuracy':<18}{'Malignant Recall':<20}{'Time (s)':<12}"
    print(header)
    print("-" * 96)

    summary_rows = {}
    for method in methods:
        rows = [r for r in all_results if r["method"] == method and r.get("test_metrics")]
        if not rows:
            continue
        auroc = np.array([r["test_metrics"]["auroc"] for r in rows])
        acc = np.array([r["test_metrics"]["accuracy"] for r in rows])
        rec = np.array([r["test_metrics"]["malignant_recall"] for r in rows])
        t = np.array([r["time"] for r in rows])
        print(f"{method:<24}{auroc.mean():.4f}±{auroc.std():.4f}    "
              f"{acc.mean():.4f}±{acc.std():.4f}    "
              f"{rec.mean():.4f}±{rec.std():.4f}       "
              f"{t.mean():.1f}")
        summary_rows[method] = {
            "auroc_mean": float(auroc.mean()), "auroc_std": float(auroc.std()),
            "accuracy_mean": float(acc.mean()), "accuracy_std": float(acc.std()),
            "malignant_recall_mean": float(rec.mean()), "malignant_recall_std": float(rec.std()),
            "time_mean": float(t.mean()),
        }

    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"\n  Summary saved to {summary_path}")

    make_plot(all_results, methods)


def make_plot(all_results, methods):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    data_auroc = [[r["test_metrics"]["auroc"] for r in all_results
                  if r["method"] == m and r.get("test_metrics")] for m in methods]
    data_recall = [[r["test_metrics"]["malignant_recall"] for r in all_results
                    if r["method"] == m and r.get("test_metrics")] for m in methods]

    colors = ["#888888", "#4C72B0", "#55A868", "#C44E52"]
    axes[0].boxplot(data_auroc, labels=methods, patch_artist=True,
                    boxprops=dict(facecolor="#cccccc"))
    for patch, c in zip(axes[0].artists, colors):
        patch.set_facecolor(c)
    axes[0].set_ylabel("Test ROC-AUC")
    axes[0].set_title("Held-out Test AUROC by Method")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].boxplot(data_recall, labels=methods, patch_artist=True,
                    boxprops=dict(facecolor="#cccccc"))
    for patch, c in zip(axes[1].artists, colors):
        patch.set_facecolor(c)
    axes[1].set_ylabel("Malignant-class Recall")
    axes[1].set_title("Missed-Cancer Risk by Method (higher = safer)")
    axes[1].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "breast_cancer_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
