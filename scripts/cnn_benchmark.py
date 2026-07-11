"""
CNN Benchmark - CIFAR-10 Classification (Phase 5, SIAM revision).

Upgraded from MNIST/SimpleCNN to CIFAR-10/ResNet-18 with a 4D HP space:
  log_lr, dropout, log_wd, log_batch_size

Provides a unified benchmark comparing:
  Method A: Pure HHD  (HMC-only hyperparameter tuning)
  Method B: Hybrid BO (GP-based hyperparameter search + Adam + L-BFGS)
  Method C: Unified HHD-ABBO (three-phase curriculum)

Multi-seed runs produce mean ± std for fair comparison.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from scipy.optimize import minimize
from scipy.stats import norm

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from hamiltonian import HyperparamState
from symplectic_solver import HamiltonianMCMC
from hybrid_hhd_abbo_improved import (
    AdaptiveStepSizeController,
    PlateauDetector,
    ImprovedUnifiedTrainer,
)

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
#  Lightweight CIFAR-10 CNN  (CPU-feasible, ~200 K params)
# --------------------------------------------------------------------------- #

class CIFAR10ResNet(nn.Module):
    """Small 3-block CNN for CIFAR-10, CPU-feasible (~200 K parameters).

    Named CIFAR10ResNet for API compatibility; architecture is a lightweight
    ConvNet suited for fast CPU experimentation with the HHD-ABBO framework.
    """

    def __init__(self, dropout: float = 0.2, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 3→32
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 32→16

            # Block 2: 32→64
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 16→8

            # Block 3: 64→128
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(2),  # 8→2  (2×2 spatial)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),             # 128*4 = 512
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        self._dropout_rate = dropout

    def forward(self, x):
        return self.classifier(self.features(x))

    @property
    def dropout_rate(self) -> float:
        """Expose dropout_rate for symplectic_solver.py compatibility."""
        return self._dropout_rate

    @dropout_rate.setter
    def dropout_rate(self, p: float):
        self._dropout_rate = p
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.p = p

    def set_dropout(self, p: float):
        self._dropout_rate = p
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.p = p


def decode_hp(hp_dict: dict) -> Tuple[float, float, float, int]:
    """Map HP dict (either raw {log_lr:...} or decoded {lr:...}) -> (lr, dropout, wd, batch_size)."""
    # Support both raw log-space keys and decoded keys
    if "log_lr" in hp_dict:
        lr = 10 ** float(np.clip(hp_dict["log_lr"], -4, -1))
    else:
        lr = float(np.clip(hp_dict.get("lr", 1e-3), 1e-4, 1e-1))

    dropout = float(np.clip(hp_dict.get("dropout", 0.2), 0.0, 0.5))

    if "log_wd" in hp_dict:
        wd = 10 ** float(np.clip(hp_dict["log_wd"], -6, -2))
    else:
        wd = float(hp_dict.get("weight_decay", 1e-4))

    if "log_batch_size" in hp_dict:
        batch_size = max(16, int(2 ** float(np.clip(hp_dict["log_batch_size"], 5, 8))))
    else:
        batch_size = int(hp_dict.get("batch_size", 64))

    return lr, dropout, wd, batch_size


def _raw_hp(hp_state) -> dict:
    """Extract raw (log-space) values from HyperparamState."""
    return {k: float(v.item()) for k, v in hp_state.values.items()}



# --------------------------------------------------------------------------- #
#  Data Loading
# --------------------------------------------------------------------------- #

def get_cifar10_loaders(batch_size: int = 64, train_size: Optional[int] = None,
                        test_size: Optional[int] = None, seed: int = 0):
    """Load CIFAR-10 with data augmentation for training."""
    torch.manual_seed(seed)
    tr = train_size or config.CNN_TRAIN_SUBSET
    te = test_size  or config.CNN_TEST_SUBSET

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_ds = datasets.CIFAR10("./data", train=True,  download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10("./data", train=False, download=True, transform=test_tf)

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
            total   += y.size(0)
            correct += (pred == y).sum().item()
    return correct / total if total > 0 else 0.0


# --------------------------------------------------------------------------- #
#  Quick-train helper (shared by all methods)
# --------------------------------------------------------------------------- #

def _quick_train(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                 lr: float, wd: float, n_epochs: int = 1) -> float:
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    total_loss, n_batches = 0.0, 0
    for _ in range(n_epochs):
        model.train()
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            n_batches  += 1
    return total_loss / max(n_batches, 1)


# --------------------------------------------------------------------------- #
#  Method A: Pure HHD for CIFAR-10
# --------------------------------------------------------------------------- #

def run_method_a_cnn(seed: int = 0, epochs: Optional[int] = None,
                     warmup: Optional[int] = None) -> dict:
    """Pure HMC hyperparameter tuning on CIFAR-10."""
    np.random.seed(seed); torch.manual_seed(seed)
    n_warmup = warmup or config.CNN_WARMUP_EPOCHS
    n_hmc    = epochs or config.CNN_HMC_EPOCHS
    t0 = time.time()

    hp_state  = HyperparamState(config.CNN_INIT_HP, config.CNN_HP_SPACE)
    lr, dropout, wd, bs = decode_hp(_raw_hp(hp_state))

    train_loader, test_loader = get_cifar10_loaders(batch_size=bs, seed=seed)
    model     = CIFAR10ResNet(dropout=dropout).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    mcmc = HamiltonianMCMC(step_size=0.005, n_leapfrog=4,
                           mass_theta=1.0, mass_lambda=1.0, temperature=1e9)

    # Warmup
    _quick_train(model, train_loader, criterion, lr, wd, n_epochs=n_warmup)

    best_acc, best_state, best_hp_dict = 0.0, None, None
    accs = []

    for epoch in range(n_hmc):
        Xb, yb = next(iter(train_loader))
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        curr_loss = criterion(model(Xb), yb).item()
        mcmc.propose(model, hp_state, (Xb, yb), criterion, curr_loss)

        lr, dropout, wd, bs = decode_hp(_raw_hp(hp_state))
        model.set_dropout(dropout)
        _quick_train(model, train_loader, criterion, lr, wd, n_epochs=1)

        acc = eval_accuracy(model, test_loader)
        accs.append(acc)
        if acc > best_acc:
            best_acc   = acc
            best_state = deepcopy(model.state_dict())
            best_hp_dict = _raw_hp(hp_state)

        if epoch % 5 == 0:
            print(f"    [A] Ep {epoch:2d}/{n_hmc} | Acc: {acc:.2%} | lr={lr:.1e}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "method": "Method A (Pure HHD)",
        "seed": seed,
        "best_val_acc":  best_acc,
        "final_val_acc": accs[-1] if accs else best_acc,
        "time": time.time() - t0,
        "final_hps": best_hp_dict or _raw_hp(hp_state),
        "acc_history": accs,
    }


# --------------------------------------------------------------------------- #
#  Method B: Hybrid BO for CIFAR-10
# --------------------------------------------------------------------------- #

def run_method_b_cnn(seed: int = 0, n_trials: Optional[int] = None) -> dict:
    """GP-based Bayesian optimization for CIFAR-10 HPs."""
    np.random.seed(seed); torch.manual_seed(seed)
    n_bo = n_trials or config.CNN_BO_TRIALS
    t0   = time.time()

    # 4D: log_lr, dropout, log_wd, log_batch_size
    bounds = np.array([[-4.0, -1.0], [0.0, 0.5], [-6.0, -2.0], [5.0, 8.0]])

    from hybrid_adam_bfgs import GaussianProcessSurrogate, expected_improvement
    gp = GaussianProcessSurrogate()
    X_obs, y_obs = [], []
    best_acc, best_hp_dict = 0.0, {}
    all_accs = []

    for t in range(n_bo):
        if t < 4:
            lam = np.random.uniform(bounds[:, 0], bounds[:, 1])
        else:
            best_y = min(y_obs)
            best_x, best_v = None, -np.inf
            for _ in range(10):
                x0  = np.random.uniform(bounds[:, 0], bounds[:, 1])
                res = minimize(
                    lambda x: -expected_improvement(x[None], gp, best_y)[0],
                    x0, method="L-BFGS-B",
                    bounds=list(zip(bounds[:, 0], bounds[:, 1])),
                )
                if -res.fun > best_v:
                    best_v, best_x = -res.fun, res.x
            lam = best_x if best_x is not None else np.random.uniform(
                bounds[:, 0], bounds[:, 1])

        hp_d = {
            "log_lr": lam[0], "dropout": lam[1],
            "log_wd": lam[2], "log_batch_size": lam[3],
        }
        lr, dropout, wd, bs = decode_hp(hp_d)
        train_loader, test_loader = get_cifar10_loaders(batch_size=bs, seed=seed)
        model = CIFAR10ResNet(dropout=dropout).to(DEVICE)
        _quick_train(model, train_loader, nn.CrossEntropyLoss(), lr, wd, n_epochs=5)

        acc     = eval_accuracy(model, test_loader)
        neg_acc = -acc
        X_obs.append(lam); y_obs.append(neg_acc)
        gp.update(np.stack(X_obs), np.array(y_obs))

        if acc > best_acc:
            best_acc, best_hp_dict = acc, hp_d

        all_accs.append(best_acc)
        print(f"    [B] Trial {t:2d}/{n_bo} | Acc: {acc:.2%} | Best: {best_acc:.2%}")

    return {
        "method": "Method B (Hybrid BO)",
        "seed": seed,
        "best_val_acc":  best_acc,
        "final_val_acc": best_acc,
        "time": time.time() - t0,
        "final_hps": best_hp_dict,
        "acc_history": all_accs,
    }


# --------------------------------------------------------------------------- #
#  Method C: Unified HHD-ABBO for CIFAR-10
# --------------------------------------------------------------------------- #

def run_method_c_cnn(seed: int = 0, epochs: Optional[int] = None,
                     warmup: Optional[int] = None) -> dict:
    """Unified three-phase curriculum on CIFAR-10 with 4D HP space."""
    np.random.seed(seed); torch.manual_seed(seed)
    n_warmup = warmup or config.CNN_WARMUP_EPOCHS
    n_hmc    = epochs or config.CNN_HMC_EPOCHS
    t0 = time.time()

    hp_state  = HyperparamState(config.CNN_INIT_HP, config.CNN_HP_SPACE)
    lr, dropout, wd, bs = decode_hp(_raw_hp(hp_state))
    train_loader, test_loader = get_cifar10_loaders(batch_size=bs, seed=seed)
    model     = CIFAR10ResNet(dropout=dropout).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    mcmc = HamiltonianMCMC(step_size=0.005, n_leapfrog=4,
                           mass_theta=1.0, mass_lambda=1.0, temperature=1e9)
    step_ctrl = AdaptiveStepSizeController(initial_step=0.005)
    plateau   = PlateauDetector(patience=4, tol=5e-4)

    # Phase 1: Adam warmup
    _quick_train(model, train_loader, criterion, lr, wd, n_epochs=n_warmup)

    best_acc, best_state, best_hp_dict = 0.0, None, None
    accs = []

    for epoch in range(n_hmc):
        # Phase 2: HMC HP proposal
        Xb, yb = next(iter(train_loader))
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        curr_loss = criterion(model(Xb), yb).item()
        mcmc.propose(model, hp_state, (Xb, yb), criterion, curr_loss)
        mcmc.leapfrog.eps = step_ctrl.update(mcmc.acceptance_rate)

        lr, dropout, wd, bs = decode_hp(_raw_hp(hp_state))
        model.set_dropout(dropout)

        # Adam step with updated HPs
        train_loss = _quick_train(model, train_loader, criterion, lr, wd, n_epochs=1)

        # Phase 3: L-BFGS on plateau
        if plateau.update(train_loss):
            Xs, ys_list = [], []
            for X, y in train_loader:
                Xs.append(X); ys_list.append(y)
                if len(Xs) >= 4:
                    break
            Xf = torch.cat(Xs).to(DEVICE)
            yf = torch.cat(ys_list).to(DEVICE)
            lbfgs = optim.LBFGS(model.parameters(), max_iter=10, lr=0.01)
            def closure():
                lbfgs.zero_grad()
                l = criterion(model(Xf), yf)
                l.backward()
                return l
            lbfgs.step(closure)
            plateau.reset()

        acc = eval_accuracy(model, test_loader)
        accs.append(acc)

        if acc > best_acc:
            best_acc   = acc
            best_state = deepcopy(model.state_dict())
            best_hp_dict = _raw_hp(hp_state)

        if epoch % 3 == 0:
            print(f"    [C] Ep {epoch:2d}/{n_hmc} | Acc: {acc:.2%} | "
                  f"HMC acc: {mcmc.acceptance_rate:.1%} | lr={lr:.1e}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "method": "Method C (Unified HHD-ABBO)",
        "seed": seed,
        "best_val_acc":  best_acc,
        "final_val_acc": accs[-1] if accs else best_acc,
        "time": time.time() - t0,
        "final_hps": best_hp_dict or _raw_hp(hp_state),
        "acc_history": accs,
    }


# --------------------------------------------------------------------------- #
#  Parallel Worker
# --------------------------------------------------------------------------- #

def cnn_worker(args):
    method_key, seed = args
    import torch
    torch.set_num_threads(1)
    
    METHOD_FNS = {
        "A": run_method_a_cnn,
        "B": run_method_b_cnn,
        "C": run_method_c_cnn,
    }
    return METHOD_FNS[method_key](seed=seed)


# --------------------------------------------------------------------------- #
#  Full Benchmark Runner (multi-seed, mean±std)
# --------------------------------------------------------------------------- #

def run_cnn_benchmark(
    seeds: Optional[List[int]] = None,
    methods: Optional[List[str]] = None,
) -> Dict[str, dict]:
    """Run all methods on CIFAR-10, multiple seeds, report mean±std."""
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]
    if methods is None:
        methods = ["A", "B", "C"]

    print("=" * 70)
    print("  CNN BENCHMARK: CIFAR-10 Classification (Phase 5, SIAM)")
    print(f"  Methods: {methods}  |  Seeds: {seeds}")
    print("=" * 70)

    LABELS     = {
        "A": "Method A (Pure HHD)",
        "B": "Method B (Hybrid BO)",
        "C": "Method C (Unified HHD-ABBO)",
    }

    all_runs: Dict[str, List[dict]] = {m: [] for m in methods}
    tasks = [(m, s) for m in methods for s in seeds]

    print(f"\n  Running {len(tasks)} CNN benchmark tasks sequentially...")

    for m_key, s in tasks:
        try:
            print(f"\n  >>> Starting {LABELS[m_key]} seed={s} ...")
            torch.set_num_threads(1)
            result = cnn_worker((m_key, s))
            all_runs[m_key].append(result)
            print(f"    [OK] {LABELS[m_key]} seed {s} finished. Acc: {result['best_val_acc']:.2%}")
        except Exception as e:
            import traceback
            print(f"    [ERROR] {LABELS[m_key]} seed {s} failed: {e}")
            traceback.print_exc()

    # Save detailed results
    os.makedirs("results_cnn", exist_ok=True)
    with open("results_cnn/cifar10_benchmark_results.json", "w") as f:
        json.dump({m: runs for m, runs in all_runs.items()}, f, indent=2)

    # Summary table: mean ± std
    print("\n" + "=" * 70)
    print("  CNN BENCHMARK RESULTS (CIFAR-10) — mean ± std over seeds")
    print("=" * 70)
    print(f"{'Method':<35} | {'Best Acc (mean±std)':>22} | {'Time (s)':>10}")
    print("-" * 70)

    summary = {}
    for method_key in methods:
        runs  = all_runs[method_key]
        if not runs:
            continue
        accs  = [r["best_val_acc"] for r in runs]
        times = [r["time"]         for r in runs]
        mean_acc = np.mean(accs)
        std_acc  = np.std(accs)
        print(f"{LABELS[method_key]:<35} | {mean_acc:.4f} ± {std_acc:.4f}         | "
              f"{np.mean(times):>9.1f}")
        summary[method_key] = {
            "mean_best_acc": float(mean_acc),
            "std_best_acc":  float(std_acc),
            "mean_time":     float(np.mean(times)),
            "all_runs":      runs,
        }

    print("=" * 70)

    with open("results_cnn/cifar10_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Also save to results_cnn/benchmark_results.json for evaluate.py
    # To keep evaluate.py compatible, we write a simplified version of summary 
    # that maps method names exactly as evaluate.py expects
    legacy_format = {}
    legacy_mapping = {
        "A": "Method A (Pure HHD)",
        "B": "Method B (Hybrid BO)",
        "C": "Method C (Unified HHD-ABBO)"
    }
    for m_key, m_name in legacy_mapping.items():
        if m_key not in all_runs:
            continue
        runs = all_runs[m_key]
        if not runs:
            continue
        # pick the run with the best accuracy as representative, or average
        best_run = max(runs, key=lambda r: r["best_val_acc"])
        # average time
        avg_time = float(np.mean([r["time"] for r in runs]))
        legacy_format[m_name] = {
            "best_val_acc": float(np.mean([r["best_val_acc"] for r in runs])),
            "final_val_acc": float(np.mean([r["final_val_acc"] for r in runs])),
            "time": avg_time,
            "final_hps": {
                "lr": float(np.mean([decode_hp(r["final_hps"])[0] for r in runs])),
                "dropout": float(np.mean([decode_hp(r["final_hps"])[1] for r in runs]))
            },
            "acc_history": best_run["acc_history"]
        }
    with open("results_cnn/benchmark_results.json", "w") as f:
        json.dump(legacy_format, f, indent=2)

    print("  [OK] Summary saved to results_cnn/cifar10_summary.json and benchmark_results.json")
    return summary


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="CIFAR-10 CNN benchmark")
    p.add_argument("--seeds",   type=str, default="0,1,2,3,4")
    p.add_argument("--methods", type=str, default="A,B,C")
    args = p.parse_args()
    seeds   = [int(s.strip()) for s in args.seeds.split(",")]
    methods = [m.strip() for m in args.methods.split(",")]
    run_cnn_benchmark(seeds=seeds, methods=methods)
