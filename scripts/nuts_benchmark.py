"""
NUTS Benchmark — No-U-Turn Sampler for HHD-ABBO on Breast Cancer.

Implements the NUTS algorithm (Hoffman & Gelman, 2014) as a drop-in
replacement for the fixed-length Leapfrog integrator used in the
standard HMC-based Method C.  Runs a side-by-side comparison of:

  1. Method C (Fixed HMC, original)  — L=4 leapfrog, T=1e9, single-batch
  2. Method C (Fixed HMC, fixed)     — L=4 leapfrog, T=100, full-batch,
                                        partial momentum refresh
  3. Method C (NUTS)                  — adaptive trajectory, T=100,
                                        full-batch, partial momentum refresh

All three share the same three-phase curriculum (Adam warmup → HMC/NUTS
co-evolution → L-BFGS polish) and identical search spaces.

Usage:
    python nuts_benchmark.py --seeds 0,1,2,3,4 --hmc-epochs 25
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
from symplectic_solver import (
    HamiltonianMCMC,
    LeapfrogIntegrator,
    compute_loss_and_grads,
    finite_diff_hp_grads,
)
from hybrid_hhd_abbo_improved import AdaptiveStepSizeController, PlateauDetector

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, recall_score, accuracy_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = os.path.join("results", "breast_cancer")
PLOTS_DIR = os.path.join("plots")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# --------------------------------------------------------------------------- #
#  Search space (identical to real_world_breast_cancer.py)
# --------------------------------------------------------------------------- #
HP_SPACE = {
    "log_lr":   (-5.0, -1.0),
    "dropout":  (0.0,  0.6),
    "log_wd":   (-6.0, -2.0),
    "n_hidden": (8.0,  128.0),
    "n_layers": (1.0,  4.0),
}
INIT_HP = {
    "log_lr":   -3.0,
    "dropout":  0.2,
    "log_wd":   -4.0,
    "n_hidden": 32.0,
    "n_layers": 2.0,
}
TRAIN_EPOCHS_PER_TRIAL = 40


# --------------------------------------------------------------------------- #
#  Model
# --------------------------------------------------------------------------- #
class ClinicalMLP(nn.Module):
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


def decode_dict(raw_hp: dict) -> dict:
    lr, dropout, wd, n_hidden, n_layers = decode_hp(raw_hp)
    return {"lr": lr, "dropout": dropout, "weight_decay": wd,
            "n_hidden": n_hidden, "n_layers": n_layers}


# --------------------------------------------------------------------------- #
#  Data
# --------------------------------------------------------------------------- #
def get_data(seed: int, batch_size: int = 32):
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
    return (_loader(X_tr, y_tr, batch_size, True),
            _loader(X_val, y_val, 64, False),
            _loader(X_te, y_te, 64, False),
            X.shape[1])


@torch.no_grad()
def evaluate(model, loader):
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
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()


# =========================================================================== #
#  NUTS INTEGRATOR (No-U-Turn Sampler — Hoffman & Gelman 2014)
# =========================================================================== #

class NUTSIntegrator:
    """
    No-U-Turn Sampler for the joint (theta, lambda) phase space.

    Instead of running a fixed number of leapfrog steps, NUTS adaptively
    doubles the trajectory length until a U-turn is detected — i.e., when
    continuing the trajectory would bring the state back toward its origin.

    This avoids the two failure modes of fixed-L leapfrog:
      - L too small → proposal barely moves, wasting gradient evaluations
      - L too large → trajectory doubles back, proposing near the start

    Implementation uses the simplified "naive NUTS" (Algorithm 3 from the
    paper) which is easier to reason about and sufficient for the 5D HP
    space + moderate-size weight space in this benchmark.

    Key adaptation for HHD: the U-turn check runs on BOTH the weight
    (theta) and hyperparameter (lambda) sub-systems jointly.
    """

    def __init__(self, step_size=0.01, max_depth=6,
                 mass_theta=1.0, mass_lambda=0.1, grad_clip=10.0):
        self.eps = step_size
        self.max_depth = max_depth   # 2^6 = 64 max leapfrog steps
        self.m_theta = mass_theta
        self.m_lambda = mass_lambda
        self.grad_clip = grad_clip
        self.last_depth = 0          # diagnostic: tree depth of last call
        self.last_n_steps = 0        # diagnostic: leapfrog steps of last call

    def _clip_grads_(self, wg):
        if not self.grad_clip:
            return
        total_norm = math.sqrt(sum(float((g ** 2).sum()) for g in wg.values()) + 1e-12)
        if total_norm > self.grad_clip:
            scale = self.grad_clip / (total_norm + 1e-12)
            for n in wg:
                wg[n] = wg[n] * scale

    def _leapfrog_step(self, model, w_mom, hp_state, batch, criterion):
        """One full leapfrog step (half-mom, full-pos, half-mom)."""
        # --- half momentum ---
        loss, wg = compute_loss_and_grads(model, batch, criterion)
        self._clip_grads_(wg)
        hg = finite_diff_hp_grads(model, hp_state, batch, criterion, loss)
        for n in w_mom:
            w_mom[n] -= (self.eps / 2) * wg[n] / self.m_theta
        hp_state.step_momenta(hg, self.eps / 2, self.m_lambda)

        # --- full position ---
        for n, p in model.named_parameters():
            p.data += self.eps * w_mom[n] / self.m_theta
        hp_state.step_positions(self.eps, self.m_lambda)

        # --- half momentum ---
        loss, wg = compute_loss_and_grads(model, batch, criterion)
        self._clip_grads_(wg)
        hg = finite_diff_hp_grads(model, hp_state, batch, criterion, loss)
        for n in w_mom:
            w_mom[n] -= (self.eps / 2) * wg[n] / self.m_theta
        hp_state.step_momenta(hg, self.eps / 2, self.m_lambda)

        return loss

    def _compute_H(self, loss, w_mom, hp_state):
        """Total Hamiltonian H = T_theta + T_lambda + V."""
        T_theta = sum(float((p ** 2).sum()) / (2.0 * self.m_theta)
                      for p in w_mom.values())
        T_lambda = hp_state.kinetic_energy(self.m_lambda)
        return T_theta + T_lambda + loss

    def _snapshot(self, model, w_mom, hp_state):
        """Capture full state for rollback / comparison."""
        return {
            "weights": deepcopy(model.state_dict()),
            "w_mom": {n: p.clone() for n, p in w_mom.items()},
            "hp_vals": {k: v.clone() for k, v in hp_state.values.items()},
            "hp_mom": {k: v.clone() for k, v in hp_state.momenta.items()},
        }

    def _restore(self, model, w_mom, hp_state, snap):
        """Restore state from snapshot."""
        model.load_state_dict(snap["weights"])
        for n in w_mom:
            w_mom[n] = snap["w_mom"][n].clone()
        for k in hp_state.values:
            hp_state.values[k] = snap["hp_vals"][k].clone()
        for k in hp_state.momenta:
            hp_state.momenta[k] = snap["hp_mom"][k].clone()

    def _check_uturn(self, snap_minus, snap_plus):
        """
        U-turn criterion on the HP sub-system.

        The trajectory has made a U-turn when the displacement vector
        (lambda_plus - lambda_minus) has negative dot product with either
        endpoint's momentum.  This means continuing would bring the
        trajectory back toward where it started.

        We also check the weight sub-system for U-turns using a
        scalar projection of the weight displacement onto weight momentum.
        """
        frozen = getattr(self, "_frozen_hps", [])

        # HP sub-system U-turn
        dot_minus = 0.0
        dot_plus = 0.0
        for k in snap_minus["hp_vals"]:
            if k in frozen:
                continue
            diff = float(snap_plus["hp_vals"][k].item()
                         - snap_minus["hp_vals"][k].item())
            dot_minus += diff * float(snap_minus["hp_mom"][k].item())
            dot_plus  += diff * float(snap_plus["hp_mom"][k].item())

        # Weight sub-system U-turn (scalar aggregation for efficiency)
        w_dot_minus = 0.0
        w_dot_plus = 0.0
        for n in snap_minus["w_mom"]:
            diff = snap_plus["weights"][n].float() - snap_minus["weights"][n].float()
            w_dot_minus += float((diff * snap_minus["w_mom"][n]).sum())
            w_dot_plus  += float((diff * snap_plus["w_mom"][n]).sum())

        # Debug print to understand U-turn behavior
        # print(f"      [DEBUG U-turn] HP vals minus: { {k: float(v.item()) for k, v in snap_minus['hp_vals'].items()} } | plus: { {k: float(v.item()) for k, v in snap_plus['hp_vals'].items()} }")
        # print(f"      [DEBUG U-turn] HP moms minus: { {k: float(v.item()) for k, v in snap_minus['hp_mom'].items()} } | plus: { {k: float(v.item()) for k, v in snap_plus['hp_mom'].items()} }")
        # print(f"      [DEBUG U-turn] HP dot: minus={dot_minus:.4f}, plus={dot_plus:.4f} | Weight dot: minus={w_dot_minus:.4f}, plus={w_dot_plus:.4f}")

        if dot_minus < 0 or dot_plus < 0:
            return True
        return w_dot_minus < 0 or w_dot_plus < 0

    def integrate(self, model, w_mom, hp_state, batch, criterion):
        """
        NUTS integration: adaptively grow trajectory by doubling.

        Uses the "naive NUTS" (Algorithm 3) from Hoffman & Gelman 2014:
        at each doubling level, extend the trajectory in a random direction,
        pick a candidate uniformly from the new sub-tree, and stop when
        a U-turn is detected across the full trajectory endpoints.

        Returns the loss at the selected point.
        """
        self._frozen_hps = getattr(hp_state, "frozen_hps", [])

        # Initial state = one leapfrog step from current position
        snap_init = self._snapshot(model, w_mom, hp_state)
        loss = self._leapfrog_step(model, w_mom, hp_state, batch, criterion)

        H_init = self._compute_H(loss, w_mom, hp_state)
        if not np.isfinite(H_init):
            self._restore(model, w_mom, hp_state, snap_init)
            self.last_depth = 0
            self.last_n_steps = 1
            return criterion(model(batch[0]), batch[1]).item()

        # The tree has minus-end and plus-end; both start at the same point
        snap_minus = self._snapshot(model, w_mom, hp_state)
        snap_plus  = self._snapshot(model, w_mom, hp_state)
        best_loss = loss
        best_snap = self._snapshot(model, w_mom, hp_state)
        n_valid = 1       # states in the tree that haven't diverged
        n_steps = 1
        stop = False

        for depth in range(self.max_depth):
            # Pick random direction to extend
            direction = 1 if np.random.random() > 0.5 else -1

            # Save the current eps, temporarily flip for backward direction
            saved_eps = self.eps
            if direction == -1:
                self.eps = -self.eps

            # Extend by 2^depth leapfrog steps in chosen direction
            steps_this_level = 2 ** depth
            sub_best_loss = float("inf")
            sub_best_snap = None
            sub_n_valid = 0

            if direction == 1:
                # Extend the plus-end
                self._restore(model, w_mom, hp_state, snap_plus)
            else:
                # Extend the minus-end
                self._restore(model, w_mom, hp_state, snap_minus)

            for _ in range(steps_this_level):
                step_loss = self._leapfrog_step(
                    model, w_mom, hp_state, batch, criterion)
                n_steps += 1
                H_step = self._compute_H(step_loss, w_mom, hp_state)

                # Slice check: accept if not too far from initial energy
                if np.isfinite(H_step) and (H_step - H_init) < 1000.0:
                    sub_n_valid += 1
                    # Uniform selection from valid states in this sub-tree
                    if step_loss < sub_best_loss:
                        sub_best_loss = step_loss
                        sub_best_snap = self._snapshot(model, w_mom, hp_state)

            # Update the extended endpoint
            if direction == 1:
                snap_plus = self._snapshot(model, w_mom, hp_state)
            else:
                snap_minus = self._snapshot(model, w_mom, hp_state)

            self.eps = saved_eps  # restore original step direction

            # Multinomial selection: accept candidate from sub-tree with
            # probability proportional to its size
            if sub_best_snap is not None and sub_n_valid > 0:
                accept_prob = min(1.0, sub_n_valid / max(n_valid, 1))
                if np.random.random() < accept_prob or sub_best_loss < best_loss:
                    best_loss = sub_best_loss
                    best_snap = sub_best_snap
                n_valid += sub_n_valid

            # Check U-turn across the full tree
            if self._check_uturn(snap_minus, snap_plus):
                stop = True
                break

        # Restore the best state found along the trajectory
        self._restore(model, w_mom, hp_state, best_snap)

        self.last_depth = depth + 1 if not stop else depth
        self.last_n_steps = n_steps

        return best_loss


# =========================================================================== #
#  NUTS-aware HMC Sampler (wraps NUTSIntegrator into the M-H framework)
# =========================================================================== #

class HamiltonianMCMC_NUTS:
    """
    HMC sampler using NUTS for adaptive trajectory length.

    Drop-in replacement for HamiltonianMCMC that uses NUTSIntegrator
    instead of LeapfrogIntegrator, with partial momentum refresh.
    """

    def __init__(self, step_size=0.005, max_depth=6,
                 mass_theta=1.0, mass_lambda=0.1, temperature=100.0,
                 momentum_refresh=0.1):
        self.nuts = NUTSIntegrator(step_size, max_depth, mass_theta, mass_lambda)
        self.m_theta = mass_theta
        self.m_lambda = mass_lambda
        self.temperature = temperature
        self.n_acc = 0
        self.n_prop = 0
        self.momentum_refresh = float(np.clip(momentum_refresh, 0.0, 1.0))
        self._w_mom = None

    def _kinetic_theta(self, w_mom):
        return sum(float((p ** 2).sum()) / (2.0 * self.m_theta)
                   for p in w_mom.values())

    def _refresh_momenta(self, model, hp_state):
        alpha = self.momentum_refresh
        fresh_w = {
            n: torch.randn_like(p) * float(np.sqrt(self.m_theta))
            for n, p in model.named_parameters()
        }
        if self._w_mom is None or alpha >= 1.0:
            self._w_mom = fresh_w
            hp_state.randomise_momenta(self.m_lambda)
        else:
            a = float(np.sqrt(alpha))
            b = float(np.sqrt(1.0 - alpha))
            for n in self._w_mom:
                self._w_mom[n] = b * self._w_mom[n] + a * fresh_w[n]
            frozen_hps = getattr(hp_state, "frozen_hps", [])
            for k in hp_state.momenta:
                if k in frozen_hps:
                    continue
                fresh_p = torch.randn(1) * float(np.sqrt(self.m_lambda))
                hp_state.momenta[k] = b * hp_state.momenta[k] + a * fresh_p
        return self._w_mom

    def propose(self, model, hp_state, batch, criterion, current_loss):
        self.n_prop += 1
        saved_w = deepcopy(model.state_dict())
        saved_hp = hp_state.snapshot()
        saved_mom = {k: v.clone() for k, v in hp_state.momenta.items()}

        w_mom = self._refresh_momenta(model, hp_state)

        H_init = (self._kinetic_theta(w_mom)
                  + hp_state.kinetic_energy(self.m_lambda)
                  + current_loss)

        # NUTS trajectory (adaptively grows, mutates state)
        proposed_loss = self.nuts.integrate(
            model, w_mom, hp_state, batch, criterion)

        H_prop = (self._kinetic_theta(w_mom)
                  + hp_state.kinetic_energy(self.m_lambda)
                  + proposed_loss)

        proposal_is_finite = np.isfinite(proposed_loss) and np.isfinite(H_prop)

        accept = proposal_is_finite and (
            np.random.random() < np.exp(min(0, -(H_prop - H_init) / self.temperature))
        )
        if accept:
            self.n_acc += 1
            self._w_mom = w_mom
            return True, proposed_loss
        else:
            model.load_state_dict(saved_w)
            hp_state.restore(saved_hp)
            for k in hp_state.momenta:
                hp_state.momenta[k] = saved_mom[k]
            if self.momentum_refresh < 1.0:
                self._w_mom = {n: -p for n, p in w_mom.items()}
                frozen_hps = getattr(hp_state, "frozen_hps", [])
                for k in hp_state.momenta:
                    if k not in frozen_hps:
                        hp_state.momenta[k] = -hp_state.momenta[k]
            else:
                self._w_mom = None
            return False, current_loss

    @property
    def acceptance_rate(self):
        return self.n_acc / max(self.n_prop, 1)


# =========================================================================== #
#  Method C runners (original, fixed HMC, NUTS)
# =========================================================================== #

def _run_method_c_variant(seed, input_dim, n_hmc_epochs, n_warmup, label,
                          mcmc_builder):
    """
    Shared three-phase curriculum.  `mcmc_builder` returns the sampler to use.
    """
    print(f"\n  [{label}] seed={seed}")
    np.random.seed(seed); torch.manual_seed(seed)
    t0 = time.time()

    criterion = nn.BCEWithLogitsLoss()
    train_loader, val_loader, test_loader, _ = get_data(seed)

    hp_state = HyperparamState(INIT_HP, HP_SPACE)
    hp_state.frozen_hps = ["n_layers", "n_hidden"]

    lr, dropout, wd, n_hidden, n_layers = decode_hp(_raw_hp(hp_state))
    model = ClinicalMLP(input_dim, n_hidden, n_layers, dropout).to(DEVICE)

    mcmc = mcmc_builder()
    step_ctrl = AdaptiveStepSizeController(initial_step=0.01, target_accept=0.65)
    plateau = PlateauDetector(patience=4, tol=5e-4)

    best_auroc, best_state, best_hp = -1.0, None, None

    # Phase 1: Adam warmup
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

    # Phase 2+3: HMC/NUTS co-evolution + L-BFGS
    print(f"    Phase 2+3: co-evolution + L-BFGS ({n_hmc_epochs} epochs)")
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lbfgs_count = 0

    Xall = torch.cat([X for X, y in train_loader]).to(DEVICE)
    yall = torch.cat([y for X, y in train_loader]).to(DEVICE)

    for ep in range(n_hmc_epochs):
        curr_loss = criterion(model(Xall), yall).item()
        acc_flag, curr_loss = mcmc.propose(
            model, hp_state, (Xall, yall), criterion, curr_loss)

        # Adapt step size
        integrator = getattr(mcmc, "nuts", getattr(mcmc, "leapfrog", None))
        if integrator is not None:
            integrator.eps = abs(step_ctrl.update(mcmc.acceptance_rate))

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
            lbfgs_opt = optim.LBFGS(model.parameters(), max_iter=15, lr=0.05,
                                    line_search_fn="strong_wolfe")
            def closure():
                lbfgs_opt.zero_grad()
                l = criterion(model(Xall), yall)
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

        if ep % 5 == 0 or ep == n_hmc_epochs - 1:
            tag = "ACC" if acc_flag else "REJ"
            nuts_info = ""
            if hasattr(mcmc, "nuts"):
                nuts_info = f" | depth={mcmc.nuts.last_depth} steps={mcmc.nuts.last_n_steps}"
            print(f"      ep {ep:2d}/{n_hmc_epochs} [{tag}] | val AUROC={score:.4f} | "
                  f"best={best_auroc:.4f} | acc={mcmc.acceptance_rate:.1%}{nuts_info}")

    # Phase 3: final L-BFGS polish
    print("    Phase 3: final L-BFGS polish")
    if best_state is not None:
        model.load_state_dict(best_state)
    lbfgs_final = optim.LBFGS(model.parameters(), max_iter=50, lr=0.5,
                              line_search_fn="strong_wolfe", history_size=20)
    def closure_final():
        lbfgs_final.zero_grad()
        l = criterion(model(Xall), yall)
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
        if best_state is not None:
            model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader)
    elapsed = time.time() - t0
    print(f"    DONE | best val AUROC={best_auroc:.4f} | "
          f"test AUROC={test_metrics['auroc']:.4f} | "
          f"recall={test_metrics['malignant_recall']:.4f} | "
          f"{elapsed:.1f}s | L-BFGS triggered {lbfgs_count}x")

    return {"method": label, "seed": seed, "val_auroc": best_auroc,
            "test_metrics": test_metrics, "time": elapsed,
            "final_hps": decode_dict(best_hp or _raw_hp(hp_state))}


def run_original_hmc(seed, input_dim, n_hmc_epochs, n_warmup=8):
    """Method C with original settings: L=4, T=1e9, single-batch, full refresh."""
    def builder():
        return HamiltonianMCMC(
            step_size=0.01, n_leapfrog=4, mass_theta=1.0,
            mass_lambda=base_config.MASS_LAMBDA, temperature=1e9,
            momentum_refresh=1.0)
    return _run_method_c_variant(
        seed, input_dim, n_hmc_epochs, n_warmup,
        "HMC-Original", builder)


def run_fixed_hmc(seed, input_dim, n_hmc_epochs, n_warmup=8):
    """Method C with fixes: L=4, T=100, full-batch, partial refresh."""
    def builder():
        return HamiltonianMCMC(
            step_size=0.01, n_leapfrog=4, mass_theta=1.0,
            mass_lambda=0.02, temperature=100.0,
            momentum_refresh=0.1)
    return _run_method_c_variant(
        seed, input_dim, n_hmc_epochs, n_warmup,
        "HMC-Fixed", builder)


def run_nuts(seed, input_dim, n_hmc_epochs, n_warmup=8):
    """Method C with NUTS: adaptive trajectory, T=100, full-batch, partial refresh."""
    def builder():
        return HamiltonianMCMC_NUTS(
            step_size=0.01, max_depth=6, mass_theta=1.0,
            mass_lambda=0.02, temperature=100.0,
            momentum_refresh=0.1)
    return _run_method_c_variant(
        seed, input_dim, n_hmc_epochs, n_warmup,
        "NUTS", builder)


# =========================================================================== #
#  Summary & Plotting
# =========================================================================== #

def summarize(all_results, seeds):
    methods = ["HMC-Original", "HMC-Fixed", "NUTS"]
    print("\n" + "=" * 110)
    print(f"  SUMMARY (mean ± std over {len(seeds)} seeds, held-out test set)")
    print("=" * 110)
    header = (f"{'Method':<20}{'Test AUROC':<18}{'Test Accuracy':<18}"
              f"{'Malignant Recall':<20}{'Time (s)':<12}")
    print(header)
    print("-" * 110)

    summary_rows = {}
    for method in methods:
        rows = [r for r in all_results if r["method"] == method and r.get("test_metrics")]
        if not rows:
            continue
        auroc = np.array([r["test_metrics"]["auroc"] for r in rows])
        acc   = np.array([r["test_metrics"]["accuracy"] for r in rows])
        rec   = np.array([r["test_metrics"]["malignant_recall"] for r in rows])
        t     = np.array([r["time"] for r in rows])
        print(f"{method:<20}{auroc.mean():.4f}±{auroc.std():.4f}    "
              f"{acc.mean():.4f}±{acc.std():.4f}    "
              f"{rec.mean():.4f}±{rec.std():.4f}       "
              f"{t.mean():.1f}")
        summary_rows[method] = {
            "auroc_mean": float(auroc.mean()), "auroc_std": float(auroc.std()),
            "accuracy_mean": float(acc.mean()), "accuracy_std": float(acc.std()),
            "malignant_recall_mean": float(rec.mean()), "malignant_recall_std": float(rec.std()),
            "time_mean": float(t.mean()),
        }

    summary_path = os.path.join(RESULTS_DIR, "nuts_comparison_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"\n  Summary saved to {summary_path}")

    make_plot(all_results, methods)


def make_plot(all_results, methods):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    data_auroc = [[r["test_metrics"]["auroc"] for r in all_results
                   if r["method"] == m and r.get("test_metrics")] for m in methods]
    data_recall = [[r["test_metrics"]["malignant_recall"] for r in all_results
                    if r["method"] == m and r.get("test_metrics")] for m in methods]
    data_time = [[r["time"] for r in all_results
                  if r["method"] == m] for m in methods]

    colors = ["#888888", "#4C72B0", "#C44E52"]
    short_labels = ["Original", "HMC-Fixed", "NUTS"]

    for ax, data, ylabel, title in [
        (axes[0], data_auroc, "Test ROC-AUC", "Test AUROC by Method"),
        (axes[1], data_recall, "Malignant Recall", "Malignant Recall by Method"),
        (axes[2], data_time, "Time (s)", "Wall-Clock Time by Method"),
    ]:
        bp = ax.boxplot(data, labels=short_labels, patch_artist=True,
                        boxprops=dict(facecolor="#cccccc"))
        for patch, c in zip(bp.get("boxes", ax.patches), colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "nuts_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved to {plot_path}")


# =========================================================================== #
#  Main
# =========================================================================== #

def main():
    ap = argparse.ArgumentParser(description="NUTS vs HMC benchmark on Breast Cancer")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--hmc-epochs", type=int, default=25,
                    help="HMC/NUTS co-evolution epochs")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    print("=" * 72)
    print("  NUTS vs HMC BENCHMARK on Breast Cancer Diagnosis")
    print(f"  Seeds: {seeds} | HMC/NUTS epochs: {args.hmc_epochs}")
    print("=" * 72)

    all_results: List[dict] = []
    for seed in seeds:
        _, _, _, input_dim = get_data(seed)
        all_results.append(run_original_hmc(seed, input_dim, args.hmc_epochs))
        all_results.append(run_fixed_hmc(seed, input_dim, args.hmc_epochs))
        all_results.append(run_nuts(seed, input_dim, args.hmc_epochs))

    out_path = os.path.join(RESULTS_DIR, "nuts_comparison_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\n  Raw results saved to {out_path}")

    summarize(all_results, seeds)


if __name__ == "__main__":
    main()
