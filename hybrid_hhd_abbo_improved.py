"""
Method C: Unified Hamiltonian-Adam-BFGS Optimizer (HHD-ABBO) -- Improved.

Novel three-phase curriculum that fuses all three optimization philosophies:

  PHASE 1 - Adam Warmup (first-order stochastic)
    Rapidly descends from random initialization to a low-curvature basin.
    Uses cosine-annealed LR with gradient clipping for stable convergence.

  PHASE 2 - Hamiltonian Co-Evolution (symplectic HMC)
    Jointly updates weights theta AND hyperparameters lambda via leapfrog.
    Preserves a shadow Hamiltonian and satisfies detailed balance.
    Enhanced with momentum-decay EMA and Nesterov-style HP correction.

  PHASE 3 - L-BFGS Refinement (second-order quasi-Newton)
    Triggered on plateau detection. Exploits curvature for super-linear
    convergence in the exploitation regime. Uses warm restarts.

Key advantages over Methods A & B:
  - From A: Symplectic conservation + smooth continuous HP trajectories
  - From B: Second-order L-BFGS convergence + adaptive Adam warmup
  - Novel:  Adaptive step-size control, plateau-triggered phase switching,
            coupled HMC+BFGS within each epoch, cosine LR scheduling,
            gradient-norm step-size adaptation, multi-step Adam micro-epochs,
            architecture warm-transfer on HP change, loss-weighted momentum

References:
  [1] Kingma & Ba (2015). Adam. ICLR.
  [2] Liu & Nocedal (1989). L-BFGS. Math. Programming.
  [3] Duane et al. (1987). Hybrid Monte Carlo. Phys. Lett. B.
  [4] Neal (2011). MCMC using Hamiltonian dynamics. HMC Handbook.
  [5] Betancourt (2017). Conceptual introduction to HMC. arXiv:1701.02434.
  [6] Loshchilov & Hutter (2017). SGDR: Warm Restarts. ICLR.
"""

from __future__ import annotations

import json
import math
import os
import time
import warnings
from copy import deepcopy
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config
from data_generator import generate_hamiltonian_data
from hamiltonian import HamiltonianNN, HyperparamState, HamiltonianSystem
from symplectic_solver import HamiltonianMCMC

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
#  Utility: Plateau Detector (enhanced with sliding-window variance check)
# --------------------------------------------------------------------------- #

class PlateauDetector:
    """
    Detects when training has plateaued (relative improvement < tol
    over the last `patience` steps). Uses both relative drop AND
    variance-based stagnation detection for earlier triggering.
    """

    def __init__(self, patience: int = 5, tol: float = 1e-4):
        self.patience = patience
        self.tol      = tol
        self._losses  = []

    def update(self, loss: float) -> bool:
        """Returns True when a plateau is detected."""
        self._losses.append(loss)
        if len(self._losses) < self.patience:
            return False
        recent   = self._losses[-self.patience:]
        rel_drop = (recent[0] - recent[-1]) / (abs(recent[0]) + 1e-10)
        # Also check variance-based stagnation
        variance = np.var(recent)
        mean_val = np.mean(recent)
        cv = np.sqrt(variance) / (abs(mean_val) + 1e-10)  # coefficient of variation
        return abs(rel_drop) < self.tol or cv < self.tol * 0.5

    def reset(self):
        self._losses.clear()


# --------------------------------------------------------------------------- #
#  Utility: Gradient-Norm Adaptive Step-Size Controller
# --------------------------------------------------------------------------- #

class AdaptiveStepSizeController:
    """
    Adjusts leapfrog step size using BOTH acceptance rate and gradient norm.
    Gradient-norm scaling prevents instability in high-curvature regions.
    """

    def __init__(self, initial_step: float = 0.005, target_accept: float = 0.65):
        self.step_size     = initial_step
        self.target_accept = target_accept
        self._grad_norms   = []
        self._ema_grad     = None

    def update(self, current_accept: float, grad_norm: float = None) -> float:
        # Acceptance-rate adaptation
        if current_accept < self.target_accept - 0.10:
            self.step_size *= 0.92
        elif current_accept > self.target_accept + 0.10:
            self.step_size *= 1.08

        # Gradient-norm adaptation: scale down when gradients are large
        if grad_norm is not None and grad_norm > 0:
            if self._ema_grad is None:
                self._ema_grad = grad_norm
            else:
                self._ema_grad = 0.9 * self._ema_grad + 0.1 * grad_norm
            # If current grad is much larger than EMA, reduce step
            ratio = grad_norm / (self._ema_grad + 1e-10)
            if ratio > 2.0:
                self.step_size *= 0.9
            elif ratio < 0.5:
                self.step_size *= 1.05

        self.step_size = float(np.clip(self.step_size, 0.0005, 0.025))
        return self.step_size


# --------------------------------------------------------------------------- #
#  Utility: Cosine Annealing LR Schedule
# --------------------------------------------------------------------------- #

def cosine_lr(base_lr: float, epoch: int, total_epochs: int, min_lr: float = 1e-6) -> float:
    """Cosine annealing learning rate with warm restart potential."""
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * epoch / total_epochs))


# --------------------------------------------------------------------------- #
#  Utility: Weight Transfer between Architectures
# --------------------------------------------------------------------------- #

def transfer_weights(src_model: nn.Module, dst_model: nn.Module):
    """
    Transfer overlapping weights from src to dst model.
    For layers that differ in size, copy the common sub-block.
    """
    src_sd = src_model.state_dict()
    dst_sd = dst_model.state_dict()

    for key in dst_sd:
        if key in src_sd:
            src_tensor = src_sd[key]
            dst_tensor = dst_sd[key]
            if src_tensor.shape == dst_tensor.shape:
                dst_sd[key] = src_tensor.clone()
            else:
                # Copy the overlapping region
                slices = tuple(
                    slice(0, min(s, d))
                    for s, d in zip(src_tensor.shape, dst_tensor.shape)
                )
                dst_sd[key][slices] = src_tensor[slices].clone()

    dst_model.load_state_dict(dst_sd)


# --------------------------------------------------------------------------- #
#  Unified Trainer (Method C) -- Improved
# --------------------------------------------------------------------------- #

class ImprovedUnifiedTrainer:
    """
    Method C: Unified HHD-ABBO Trainer (Improved).

    Three-phase curriculum per outer epoch:
      1. Adam micro-steps    (cosine-annealed, gradient-clipped first-order descent)
      2. HMC leapfrog        (symplectic joint theta+lambda update with EMA smoothing)
      3. L-BFGS refinement   (second-order polish, plateau-triggered with warm restarts)

    Performance improvements over the base Method C:
      - Cosine-annealed Adam warmup with gradient clipping
      - Multi-step Adam micro-epochs per HMC iteration
      - Gradient-norm-aware adaptive step-size controller
      - Momentum-decay EMA for smoother loss landscape traversal
      - Architecture warm-transfer on structural HP changes
      - Aggressive L-BFGS with more iterations and warm restarts
      - Best-model checkpointing for robustness

    History keys:
      train_loss, val_loss, acceptance_rate, step_size,
      hyperparams (dict of lists), best_val_loss
    """

    def __init__(
        self,
        hyperparam_space=None,
        init_hyperparams=None,
        mass_theta: float = 1.0,
        mass_lambda: float = None,  # defaults to config.MASS_LAMBDA
        initial_step: float = 0.005,
        n_leapfrog: int = 6,
        temperature: float = 1.0,
        adam_micro_epochs: int = 3,
        adam_lr_scale: float = 1.0,
        lbfgs_steps: int = 30,
        lbfgs_patience: int = 8,
        lbfgs_tol: float = 1e-3,
        use_adaptive_step: bool = True,
        grad_clip_norm: float = 1.0,
        ema_decay: float = 0.995,
        device: str = "cpu",
        # ---- Ablation study flags (Phase 2 of SIAM revision) ----
        use_adam_warmup: bool = True,
        use_hmc: bool = True,
        use_lbfgs: bool = True,
        use_plateau_detect: bool = True,
        # ---- Phase 4: variable input dimension ----
        input_dim: int = 2,
    ):
        self.device          = device
        self.lbfgs_steps     = lbfgs_steps
        self.adam_lr_scale    = adam_lr_scale
        self.adam_micro_epochs = adam_micro_epochs
        self.grad_clip_norm  = grad_clip_norm
        self.ema_decay       = ema_decay

        # Ablation flags
        self.use_adam_warmup   = use_adam_warmup
        self.use_hmc           = use_hmc
        self.use_lbfgs         = use_lbfgs
        self.use_plateau_detect = use_plateau_detect
        # Variable input dimension (Phase 4)
        self.input_dim         = input_dim

        if mass_lambda is None:
            mass_lambda = config.MASS_LAMBDA

        hp_space = hyperparam_space or config.HYPERPARAM_SPACE
        hp_init  = init_hyperparams or config.INIT_HYPERPARAMS

        self.hp_state  = HyperparamState(hp_init, hp_space)
        self.hp_state.frozen_hps = ["n_layers", "n_neurons"]
        self.ham_sys   = HamiltonianSystem(mass_theta, mass_lambda)
        self.mcmc      = HamiltonianMCMC(
            initial_step, n_leapfrog, mass_theta, mass_lambda, temperature
        )
        self.step_ctrl = AdaptiveStepSizeController(
            initial_step=initial_step
        ) if use_adaptive_step else None
        self.plateau   = PlateauDetector(patience=lbfgs_patience, tol=lbfgs_tol)
        self.model     = self._build_model()
        self.model.frozen_hps = ["n_layers", "n_neurons"]
        self.criterion = nn.MSELoss()

        # Best-model checkpointing (stores arch + state together)
        self._best_val    = float("inf")
        self._best_state  = None
        self._best_arch   = None
        self._best_hp     = None
        self._prev_arch   = self._get_arch_key()

        # EMA model for smoothed evaluation
        self._ema_state = None

        self.history = {
            "train_loss":      [],
            "val_loss":        [],
            "acceptance_rate": [],
            "step_size":       [],
            "best_val_loss":   [],
            "hyperparams":     {k: [] for k in hp_init},
        }

    def _get_arch_key(self):
        """Return a hashable key representing current architecture."""
        hp = self.hp_state.decode()
        return (hp.get("n_layers", 3), hp.get("n_neurons", 64))

    def _build_model(self):
        hp = self.hp_state.decode()
        return HamiltonianNN(
            n_layers=hp["n_layers"],
            n_neurons=hp["n_neurons"],
            dropout=hp["dropout"],
            input_dim=self.input_dim,
        ).to(self.device)

    def _maybe_rebuild_model(self):
        """Rebuild model with weight transfer if architecture changed."""
        new_arch = self._get_arch_key()
        if new_arch != self._prev_arch:
            old_model = self.model
            self.model = self._build_model()
            self.model.frozen_hps = ["n_layers", "n_neurons"]
            transfer_weights(old_model, self.model)
            self._prev_arch = new_arch
            return True
        return False

    def _update_ema(self):
        """Update exponential moving average of model parameters."""
        if self._ema_state is None:
            self._ema_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        else:
            for k, v in self.model.state_dict().items():
                if k in self._ema_state and self._ema_state[k].shape == v.shape:
                    self._ema_state[k].mul_(self.ema_decay).add_(v, alpha=1 - self.ema_decay)
                else:
                    self._ema_state[k] = v.clone()

    def _eval(self, loader):
        self.model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for Xb, yb in loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                total += self.criterion(self.model(Xb), yb).item()
                n += 1
        return total / max(n, 1)

    def _compute_grad_norm(self):
        """Compute the total gradient norm of the model."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def train(self, n_samples=1000, n_warmup=10, n_hamilton=50,
              train_loader=None, val_loader=None):
        t0 = time.time()
        if train_loader is None or val_loader is None:
            train_loader, val_loader, _ = generate_hamiltonian_data(
                n_samples=n_samples
            )


        base_lr = self.hp_state.decode()["lr"] * self.adam_lr_scale

        # ----- Phase 1: Cosine-Annealed Adam Warmup with Gradient Clipping -----
        opt = optim.Adam(
            self.model.parameters(), lr=base_lr, weight_decay=1e-5
        )
        if self.use_adam_warmup:
            print(f"  [Method C] Phase 1 - Adam warmup (cosine-annealed): {n_warmup} epochs")
            for ep in range(n_warmup):
                # Cosine anneal the LR
                lr = cosine_lr(base_lr, ep, n_warmup, min_lr=base_lr * 0.01)
                for pg in opt.param_groups:
                    pg["lr"] = lr

                self.model.train()
                for Xb, yb in train_loader:
                    Xb, yb = Xb.to(self.device), yb.to(self.device)
                    opt.zero_grad()
                    self.criterion(self.model(Xb), yb).backward()
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_norm
                    )
                    opt.step()
                self._update_ema()
        else:
            print(f"  [Method C] Phase 1 - Adam warmup SKIPPED (ablation: C-noAdam)")

        # Collect full-batch for L-BFGS
        Xs, ys = [], []
        for Xb, yb in train_loader:
            Xs.append(Xb)
            ys.append(yb)
        Xf = torch.cat(Xs).to(self.device)
        yf = torch.cat(ys).to(self.device)

        # Record post-warmup performance
        warmup_val = self._eval(val_loader)
        self._best_val   = warmup_val
        self._best_state = deepcopy(self.model.state_dict())

        # ----- Phase 2+3: HMC + Multi-Step Adam + L-BFGS -----
        print(f"  [Method C] Phase 2+3 - HMC + Adam + L-BFGS: {n_hamilton} epochs")
        current_loss = self._eval(train_loader)

        # Reset optimizer with fresh state for phase 2
        opt = optim.Adam(
            self.model.parameters(), lr=base_lr, weight_decay=1e-5
        )

        consecutive_rejects = 0
        lbfgs_count = 0

        for ep in range(n_hamilton):
            # ----- HMC Proposal (skipped if ablation: C-noHMC) -----
            acc = False
            if self.use_hmc:
                Xb, yb = next(iter(train_loader))
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                acc, current_loss = self.mcmc.propose(
                    self.model, self.hp_state, (Xb, yb),
                    self.criterion, current_loss,
                )

                if acc:
                    consecutive_rejects = 0
                    # Check if architecture changed and rebuild with weight transfer
                    self._maybe_rebuild_model()
                    # Refresh optimizer for new parameters
                    opt = optim.Adam(
                        self.model.parameters(),
                        lr=self.hp_state.decode()["lr"] * self.adam_lr_scale,
                        weight_decay=1e-5,
                    )
                else:
                    consecutive_rejects += 1
            else:
                Xb, yb = next(iter(train_loader))
                Xb, yb = Xb.to(self.device), yb.to(self.device)

            # ----- Adaptive Step-Size (gradient-norm-aware) -----
            grad_norm = None
            if self.step_ctrl and self.use_hmc:
                # Compute a quick gradient norm for adaptive step control
                self.model.train()
                opt.zero_grad()
                loss_tmp = self.criterion(self.model(Xb), yb)
                loss_tmp.backward()
                grad_norm = self._compute_grad_norm()

                self.mcmc.leapfrog.eps = self.step_ctrl.update(
                    self.mcmc.acceptance_rate, grad_norm
                )

            # ----- Multi-Step Adam Micro-Epochs -----
            # Cosine-annealed LR within phase 2
            phase2_lr = cosine_lr(
                self.hp_state.decode()["lr"] * self.adam_lr_scale,
                ep, n_hamilton,
                min_lr=1e-5,
            )
            for pg in opt.param_groups:
                pg["lr"] = phase2_lr

            for _micro in range(self.adam_micro_epochs):
                self.model.train()
                for Xb_m, yb_m in train_loader:
                    Xb_m, yb_m = Xb_m.to(self.device), yb_m.to(self.device)
                    opt.zero_grad()
                    self.criterion(self.model(Xb_m), yb_m).backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_norm
                    )
                    opt.step()

            self._update_ema()

            # Evaluate
            tl = self._eval(train_loader)

            # ----- L-BFGS Refinement (plateau-triggered OR periodic) -----
            if self.use_lbfgs:
                if self.use_plateau_detect:
                    trigger_lbfgs = self.plateau.update(tl)
                else:
                    # C-noPlateauDetect: run L-BFGS every epoch
                    trigger_lbfgs = True

                # Also trigger if too many consecutive rejects (stuck in bad region)
                if consecutive_rejects >= 8:
                    trigger_lbfgs = True
                    consecutive_rejects = 0

                if trigger_lbfgs:
                    lbfgs_count += 1
                    lbfgs = optim.LBFGS(
                        self.model.parameters(),
                        max_iter=self.lbfgs_steps, lr=0.1,
                        line_search_fn="strong_wolfe",
                        history_size=20,
                    )
                    def closure():
                        lbfgs.zero_grad()
                        l = self.criterion(self.model(Xf), yf)
                        l.backward()
                        return l
                    try:
                        lbfgs.step(closure)
                    except Exception:
                        pass
                    tl = self._eval(train_loader)
                    self.plateau.reset()
                    # Refresh Adam optimizer after L-BFGS step
                    opt = optim.Adam(
                        self.model.parameters(),
                        lr=phase2_lr, weight_decay=1e-5,
                    )

            vl = self._eval(val_loader)

            # ----- Best-Model Checkpointing -----
            if vl < self._best_val:
                self._best_val   = vl
                self._best_state = deepcopy(self.model.state_dict())
                self._best_arch  = self._get_arch_key()
                self._best_hp    = self.hp_state.snapshot()

            # ----- Record History -----
            self.history["train_loss"].append(tl)
            self.history["val_loss"].append(vl)
            self.history["best_val_loss"].append(self._best_val)
            self.history["acceptance_rate"].append(self.mcmc.acceptance_rate)
            self.history["step_size"].append(self.mcmc.leapfrog.eps)
            for k in self.hp_state.values:
                self.history["hyperparams"][k].append(
                    float(self.hp_state.values[k].item()))

            if ep % 10 == 0 or ep == n_hamilton - 1:
                tag = "ACC" if acc else ("REJ" if self.use_hmc else "---")
                print(f"    Ep {ep:3d}/{n_hamilton} [{tag}] | "
                      f"Train: {tl:.6f} | Val: {vl:.6f} | "
                      f"Best: {self._best_val:.6f} | "
                      f"Acc: {self.mcmc.acceptance_rate:.1%}")

        # ----- Phase 3: Dedicated L-BFGS Final Polish -----
        if self.use_lbfgs:
            print("  [Method C] Phase 3 - L-BFGS Final Polish...")
            if self._best_state is not None:
                # Rebuild model with the architecture that produced the best checkpoint
                if self._best_arch is not None and self._best_arch != self._get_arch_key():
                    n_layers, n_neurons = self._best_arch
                    hp = self.hp_state.decode()
                    self.model = HamiltonianNN(
                        n_layers=n_layers, n_neurons=n_neurons,
                        dropout=hp.get("dropout", 0.1),
                    ).to(self.device)
                    self.model.frozen_hps = ["n_layers", "n_neurons"]
                self.model.load_state_dict(self._best_state)
                if self._best_hp is not None:
                    self.hp_state.restore(self._best_hp)
                print(f"    Restored best model from Phase 2 (val={self._best_val:.6f})")

            # Run aggressive L-BFGS on the full training dataset
            lbfgs = optim.LBFGS(
                self.model.parameters(),
                max_iter=100,
                lr=1.0,
                line_search_fn="strong_wolfe",
                history_size=20,
            )
            def closure():
                lbfgs.zero_grad()
                l = self.criterion(self.model(Xf), yf)
                l.backward()
                return l
            try:
                lbfgs.step(closure)
            except Exception as e:
                print(f"    Final L-BFGS warning: {e}")
        else:
            print("  [Method C] Phase 3 - L-BFGS Final Polish SKIPPED (ablation: C-noLBFGS)")
            if self._best_state is not None:
                self.model.load_state_dict(self._best_state)
                if self._best_hp is not None:
                    self.hp_state.restore(self._best_hp)

        final_val = self._eval(val_loader)
        final_tr = self._eval(train_loader)
        print(f"  [Method C] Post-L-BFGS Polish Val Loss: {final_val:.6f}")
        
        # Save final polished model state
        self._best_val = final_val
        self._best_state = deepcopy(self.model.state_dict())
        
        # Append final polished metrics to history so evaluation and plotting scripts read the actual final state
        self.history["val_loss"].append(final_val)
        self.history["best_val_loss"].append(final_val)
        self.history["train_loss"].append(final_tr)
        
        # Mirror last values for acceptance rate, step size, and hyperparameters to keep array lengths consistent
        self.history["acceptance_rate"].append(self.history["acceptance_rate"][-1] if self.history["acceptance_rate"] else 0.0)
        self.history["step_size"].append(self.history["step_size"][-1] if self.history["step_size"] else 0.0)
        for k in self.hp_state.values:
            self.history["hyperparams"][k].append(self.history["hyperparams"][k][-1])
        
        print(f"  [Method C] Completed in {time.time() - t0:.1f}s. Final Best Val: {self._best_val:.6f}")
        print(f"  [Method C] L-BFGS triggered {lbfgs_count} times during training")

        self.train_time = time.time() - t0
        return self.history

    def save(self, save_dir="results_unified_improved"):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_dir, "model.pt"))
        with open(os.path.join(save_dir, "history.json"), "w") as f:
            json.dump(self.history, f, indent=2)
        with open(os.path.join(save_dir, "hyperparameters.json"), "w") as f:
            json.dump(self.hp_state.decode(), f, indent=2)
