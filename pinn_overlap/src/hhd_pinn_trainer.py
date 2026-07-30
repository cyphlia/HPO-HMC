"""
HHD-Driven PINN Trainer.

This module adapts Method A (Hamiltonian Hyperparameter Dynamics) from the
main HHD project to the specific needs of Physics-Informed Neural Networks.

The key adaptation:
    In the original HHD, the hyperparameters are: lr, dropout, n_layers, etc.
    In the PINN setting, the critical hyperparameters are the LOSS WEIGHTS:
        w_r  (PDE residual weight)
        w_b  (boundary condition weight)
        w_i  (initial condition weight)
        lr   (learning rate)

    These are promoted to dynamical variables in an augmented phase space
    and co-evolved with the network weights θ under Hamilton's equations
    via symplectic leapfrog integration.

Architecture:
    Phase 1 — Adam warmup with fixed weights (get θ to a reasonable basin)
    Phase 2 — HMC co-evolution of θ and (w_r, w_b, w_i, lr)
"""

from __future__ import annotations

import sys
import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from typing import Dict, Optional, Tuple, List

from pinn_model import PINN
from pinn_loss import PINNLoss, relative_l2_error


# --------------------------------------------------------------------------- #
#  PINN Hyperparameter State (adapted from src/hamiltonian.py HyperparamState)
# --------------------------------------------------------------------------- #

class PINNHyperparamState:
    """
    Continuous phase-space representation of PINN hyperparameters.

    The PINN-specific HPs are stored in LOG SPACE so that:
        - Actual weight = 10^(log_w)  →  log_w ∈ [-2, 2] maps to w ∈ [0.01, 100]
        - Actual lr     = 10^(log_lr) →  log_lr ∈ [-5, -2] maps to lr ∈ [1e-5, 0.01]

    Log-space ensures positivity and makes the dynamics scale-invariant.
    """

    def __init__(self, init_values: Dict[str, float],
                 bounds: Dict[str, Tuple[float, float]]):
        self.bounds = bounds
        self.values = {k: torch.tensor([v], dtype=torch.float32)
                       for k, v in init_values.items()}
        self.momenta = {k: torch.zeros(1) for k in init_values}

    def decode(self) -> Dict[str, float]:
        """Convert log-space values to actual hyperparameter values."""
        out = {}
        for k, v in self.values.items():
            lo, hi = self.bounds[k]
            val = float(np.clip(v.item(), lo, hi))
            if k.startswith("log_"):
                # log-scale: actual = 10^val
                actual_name = k[4:]  # "log_w_r" → "w_r"
                out[actual_name] = 10 ** val
            else:
                out[k] = val
        return out

    def step_positions(self, eps: float, mass: float):
        """Update positions: λ += ε · p_λ · range / m_λ (with reflection BCs)."""
        with torch.no_grad():
            for k in self.values:
                lo, hi = self.bounds[k]
                hp_range = hi - lo
                new_val = self.values[k] + eps * self.momenta[k] * hp_range / mass
                # Reflect off boundaries
                for _ in range(3):
                    if new_val > hi:
                        new_val = 2.0 * hi - new_val
                        self.momenta[k] = -self.momenta[k]
                    elif new_val < lo:
                        new_val = 2.0 * lo - new_val
                        self.momenta[k] = -self.momenta[k]
                    else:
                        break
                new_val.clamp_(lo, hi)
                self.values[k].copy_(new_val)

    def step_momenta(self, grads: Dict[str, torch.Tensor], eps: float, mass: float):
        """Update momenta: p_λ -= ε · ∂L/∂λ · range / m_λ."""
        with torch.no_grad():
            for k, g in grads.items():
                if g is not None:
                    lo, hi = self.bounds[k]
                    hp_range = hi - lo
                    self.momenta[k] -= eps * g * hp_range / mass

    def randomise_momenta(self, mass: float):
        """Sample p_λ ~ N(0, m_λ) for fresh HMC trajectory."""
        for k in self.momenta:
            self.momenta[k] = torch.randn(1) * float(np.sqrt(mass))

    def kinetic_energy(self, mass: float) -> float:
        """T_λ = Σ p_k² / (2·m_λ)."""
        return sum(float((p**2).sum()) / (2.0 * mass) for p in self.momenta.values())

    def snapshot(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.values.items()}

    def restore(self, snap: Dict[str, torch.Tensor]):
        for k, v in snap.items():
            self.values[k] = v.clone()


# --------------------------------------------------------------------------- #
#  Finite-Difference HP Gradients for PINN Loss Weights
# --------------------------------------------------------------------------- #

def finite_diff_pinn_hp_grads(
    model: PINN,
    hp_state: PINNHyperparamState,
    loss_fn: PINNLoss,
    residuals: torch.Tensor,
    bc_pred: torch.Tensor,
    bc_true: torch.Tensor,
    ic_pred: torch.Tensor,
    ic_true: torch.Tensor,
    base_loss: float,
) -> Dict[str, torch.Tensor]:
    """
    Finite-difference approximation of ∂L_total/∂(log_w) for each HP.

    For each loss weight w_k:
        1. Perturb log_w_k by ±5% of its range
        2. Recompute total loss with the perturbed weight
        3. Gradient ≈ (L(w+) - L(w-)) / (2·ε)

    This is the same approach used in the main HHD project (see
    src/symplectic_solver.py::finite_diff_hp_grads), adapted for PINN weights.
    """
    grads = {}

    for k, v in hp_state.values.items():
        old_val = float(v.item())
        lo, hi = hp_state.bounds[k]
        eps = 0.05 * (hi - lo)
        vp = min(hi, old_val + eps)
        vm = max(lo, old_val - eps)
        denom = vp - vm

        if denom < 1e-10:
            grads[k] = torch.tensor([0.0])
            continue

        # Evaluate loss at perturbed weight values
        def _eval_at(val):
            decoded = {}
            for kk, vv in hp_state.values.items():
                lo_k, hi_k = hp_state.bounds[kk]
                v_use = val if kk == k else float(np.clip(vv.item(), lo_k, hi_k))
                if kk.startswith("log_"):
                    decoded[kk[4:]] = 10 ** v_use
                else:
                    decoded[kk] = v_use
            tmp_loss = PINNLoss(
                w_residual=decoded.get("w_r", loss_fn.w_r),
                w_boundary=decoded.get("w_b", loss_fn.w_b),
                w_initial=decoded.get("w_i", loss_fn.w_i),
            )
            total, _ = tmp_loss.compute(residuals, bc_pred, bc_true, ic_pred, ic_true)
            return float(total.item())

        loss_plus = _eval_at(vp)
        loss_minus = _eval_at(vm)
        grads[k] = torch.tensor([(loss_plus - loss_minus) / denom])

    return grads


# --------------------------------------------------------------------------- #
#  HHD-PINN Trainer
# --------------------------------------------------------------------------- #

class HHDPINNTrainer:
    """
    Hamiltonian Hyperparameter Dynamics trainer for PINNs.

    This adapts Method A (HHD-HMC) from the main project to PINN training.
    The hyperparameters co-evolved are the PINN loss weights and learning rate.

    Parameters
    ----------
    model : PINN
        The PINN model to train
    problem : object
        PDE problem instance (HeatEquation1D, BurgersEquation1D, etc.)
    n_warmup : int
        Number of Adam warmup epochs (Phase 1)
    n_hmc : int
        Number of HMC co-evolution epochs (Phase 2)
    step_size : float
        Leapfrog step size ε
    n_leapfrog : int
        Number of leapfrog sub-steps L per HMC proposal
    mass_theta : float
        Inertia for network weights
    mass_lambda : float
        Inertia for loss weights (higher = more conservative weight changes)
    temperature : float
        Boltzmann temperature (1e9 = always-accept for optimization mode)
    lr_adam : float
        Adam learning rate for warmup phase
    device : str
        'cpu' or 'cuda'
    """

    def __init__(
        self,
        model: PINN,
        problem,
        n_warmup: int = 100,
        n_hmc: int = 200,
        step_size: float = 0.003,
        n_leapfrog: int = 3,
        mass_theta: float = 1.0,
        mass_lambda: float = 5.0,
        temperature: float = 1e9,
        lr_adam: float = 1e-3,
        device: str = "cpu",
        n_collocation: int = 2000,
        n_boundary: int = 200,
        n_initial: int = 200,
        grad_clip: float = 1.0,
    ):
        self.model = model.to(device)
        self.problem = problem
        self.device = device
        self.n_warmup = n_warmup
        self.n_hmc = n_hmc
        self.step_size = step_size
        self.n_leapfrog = n_leapfrog
        self.mass_theta = mass_theta
        self.mass_lambda = mass_lambda
        self.temperature = temperature
        self.lr_adam = lr_adam
        self.n_collocation = n_collocation
        self.n_boundary = n_boundary
        self.n_initial = n_initial
        self.grad_clip = grad_clip

        # Determine if problem is time-dependent or steady-state
        self.is_time_dependent = hasattr(problem, 't_range')

        # PINN loss with initial weights = 1.0 each
        self.loss_fn = PINNLoss(w_residual=1.0, w_boundary=1.0, w_initial=1.0)

        # HP state: loss weights in log space
        hp_init = {
            "log_w_r": 0.0,   # w_r = 10^0 = 1.0
            "log_w_b": 0.0,   # w_b = 1.0
            "log_lr":  -3.0,  # lr = 0.001
        }
        hp_bounds = {
            "log_w_r": (-2.0, 2.0),   # w_r ∈ [0.01, 100]
            "log_w_b": (-2.0, 2.0),   # w_b ∈ [0.01, 100]
            "log_lr":  (-5.0, -1.0),  # lr ∈ [1e-5, 0.1]
        }
        if self.is_time_dependent:
            hp_init["log_w_i"] = 0.0       # w_i = 1.0
            hp_bounds["log_w_i"] = (-2.0, 2.0)

        self.hp_state = PINNHyperparamState(hp_init, hp_bounds)

        # Training history
        self.history = {
            "total_loss": [],
            "L_residual": [],
            "L_boundary": [],
            "L_initial": [],
            "w_r": [],
            "w_b": [],
            "w_i": [],
            "lr": [],
            "acceptance_rate": [],
            "rel_l2_error": [],
            "phase": [],  # 'warmup' or 'hmc'
        }
        self._best_loss = float("inf")
        self._best_state = None
        self._best_hp = None

    def _sample_points(self) -> Dict[str, torch.Tensor]:
        """Sample fresh collocation points (re-sampled each epoch for diversity)."""
        if self.is_time_dependent:
            return self.problem.sample_collocation(
                n_interior=self.n_collocation,
                n_boundary=self.n_boundary,
                n_initial=self.n_initial,
                device=self.device,
            )
        else:
            return self.problem.sample_collocation(
                n_interior=self.n_collocation,
                n_boundary=self.n_boundary,
                device=self.device,
            )

    def _compute_pinn_loss(self, points: Dict[str, torch.Tensor]):
        """Compute PINN loss components from sampled collocation points."""
        if self.is_time_dependent:
            # Time-dependent PDE (Heat, Burgers)
            residuals = self.problem.residual(self.model, points["x_int"], points["t_int"])
            xt_bc = torch.cat([points["x_bc"], points["t_bc"]], dim=1)
            bc_pred = self.model(xt_bc)
            bc_true = points["bc_true"]
            xt_ic = torch.cat([points["x_ic"], points["t_ic"]], dim=1)
            ic_pred = self.model(xt_ic)
            ic_true = points["ic_true"]
        else:
            # Steady-state PDE (Poisson)
            residuals = self.problem.residual(self.model, points["x_int"], points["y_int"])
            xy_bc = torch.cat([points["x_bc"], points["y_bc"]], dim=1)
            bc_pred = self.model(xy_bc)
            bc_true = points["bc_true"]
            ic_pred = None
            ic_true = None

        total_loss, components = self.loss_fn.compute(
            residuals, bc_pred, bc_true, ic_pred, ic_true
        )
        return total_loss, components, residuals, bc_pred, bc_true, ic_pred, ic_true

    def _compute_rel_l2(self) -> float:
        """Compute relative L2 error against exact solution on a test grid."""
        self.model.eval()
        with torch.no_grad():
            if self.is_time_dependent:
                x_test = torch.linspace(
                    self.problem.x_range[0], self.problem.x_range[1], 50,
                    device=self.device
                ).unsqueeze(1)
                t_test = torch.ones_like(x_test) * 0.5  # evaluate at t=0.5
                xt = torch.cat([x_test, t_test], dim=1)
                u_pred = self.model(xt)
                u_exact = self.problem.exact_solution(x_test, t_test)
            else:
                x_test = torch.linspace(0, 1, 30, device=self.device)
                y_test = torch.linspace(0, 1, 30, device=self.device)
                xx, yy = torch.meshgrid(x_test, y_test, indexing='ij')
                xy = torch.stack([xx.flatten(), yy.flatten()], dim=1)
                u_pred = self.model(xy)
                u_exact = self.problem.exact_solution(
                    xy[:, 0:1], xy[:, 1:2]
                )
            error = relative_l2_error(u_pred, u_exact)
        self.model.train()
        return error

    def _update_loss_weights(self):
        """Sync loss_fn weights from hp_state decoded values."""
        decoded = self.hp_state.decode()
        self.loss_fn.set_weights(
            w_r=decoded.get("w_r", 1.0),
            w_b=decoded.get("w_b", 1.0),
            w_i=decoded.get("w_i", 1.0),
        )

    def _log_epoch(self, components: Dict, phase: str, acc_rate: float = 0.0):
        """Record training history for this epoch."""
        decoded = self.hp_state.decode()
        self.history["total_loss"].append(components["L_total"])
        self.history["L_residual"].append(components["L_residual"])
        self.history["L_boundary"].append(components["L_boundary"])
        self.history["L_initial"].append(components["L_initial"])
        self.history["w_r"].append(decoded.get("w_r", 1.0))
        self.history["w_b"].append(decoded.get("w_b", 1.0))
        self.history["w_i"].append(decoded.get("w_i", 1.0))
        self.history["lr"].append(decoded.get("lr", self.lr_adam))
        self.history["acceptance_rate"].append(acc_rate)
        self.history["phase"].append(phase)

        # Relative L2 error (every 10 epochs to save compute)
        epoch = len(self.history["total_loss"])
        if epoch % 10 == 0 or epoch <= 5:
            self.history["rel_l2_error"].append(self._compute_rel_l2())
        elif self.history["rel_l2_error"]:
            self.history["rel_l2_error"].append(self.history["rel_l2_error"][-1])
        else:
            self.history["rel_l2_error"].append(1.0)

        # Best checkpoint
        if components["L_total"] < self._best_loss:
            self._best_loss = components["L_total"]
            self._best_state = deepcopy(self.model.state_dict())
            self._best_hp = deepcopy(decoded)

    # ======================================================================= #
    #  Phase 1: Adam Warmup                                                   #
    # ======================================================================= #

    def _warmup_phase(self):
        """
        Phase 1: Standard Adam training with fixed loss weights.

        Purpose: get θ into a reasonable basin before HMC takes over.
        The loss weights stay at (1, 1, 1) — no HP evolution yet.
        """
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr_adam)
        self.model.train()

        for epoch in range(self.n_warmup):
            optimizer.zero_grad()
            points = self._sample_points()
            total_loss, components, *_ = self._compute_pinn_loss(points)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            optimizer.step()
            self._log_epoch(components, phase="warmup")

            if (epoch + 1) % 25 == 0:
                print(f"  [Warmup {epoch+1}/{self.n_warmup}] "
                      f"Loss={components['L_total']:.6f} "
                      f"(Lr={components['L_residual']:.4f}, "
                      f"Lb={components['L_boundary']:.4f}, "
                      f"Li={components['L_initial']:.4f})")

    # ======================================================================= #
    #  Phase 2: HMC Co-Evolution                                              #
    # ======================================================================= #

    def _hmc_phase(self):
        """
        Phase 2: Hamiltonian Monte Carlo co-evolution.

        Each HMC step:
            1. Randomize momenta p_θ, p_w ~ N(0, m)
            2. Run L leapfrog sub-steps jointly updating θ and w
            3. Accept/reject via Metropolis criterion on ΔH
        """
        accepts = 0
        self.model.train()

        for epoch in range(self.n_hmc):
            # --- Save current state ---
            theta_snap = deepcopy(self.model.state_dict())
            hp_snap = self.hp_state.snapshot()
            mom_snap = {k: v.clone() for k, v in self.hp_state.momenta.items()}

            # --- Compute current Hamiltonian ---
            points = self._sample_points()
            loss_cur, comp_cur, res, bc_p, bc_t, ic_p, ic_t = self._compute_pinn_loss(points)
            H_cur = float(loss_cur.item()) + self.hp_state.kinetic_energy(self.mass_lambda)
            # Add weight kinetic energy
            w_mom = {n: torch.randn_like(p) * math.sqrt(self.mass_theta)
                     for n, p in self.model.named_parameters()}
            H_cur += sum(float((p**2).sum()) / (2*self.mass_theta) for p in w_mom.values())

            # --- Randomize momenta ---
            self.hp_state.randomise_momenta(self.mass_lambda)

            # --- Leapfrog integration ---
            for step in range(self.n_leapfrog):
                # Half-step momenta (θ gradients via backprop)
                self.model.zero_grad()
                points_lf = self._sample_points()
                loss_lf, _, res_lf, bc_p_lf, bc_t_lf, ic_p_lf, ic_t_lf = self._compute_pinn_loss(points_lf)
                loss_lf.backward()

                # Clip weight gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                # Half-step weight momenta
                with torch.no_grad():
                    for n, p in self.model.named_parameters():
                        if p.grad is not None:
                            g = p.grad.clone()
                            w_mom[n] -= (self.step_size / 2) * g

                # HP gradients via finite difference
                hp_grads = finite_diff_pinn_hp_grads(
                    self.model, self.hp_state, self.loss_fn,
                    res_lf.detach(), bc_p_lf.detach(), bc_t_lf,
                    ic_p_lf.detach() if ic_p_lf is not None else None,
                    ic_t_lf,
                    float(loss_lf.item()),
                )
                self.hp_state.step_momenta(hp_grads, self.step_size / 2, self.mass_lambda)

                # Full-step positions
                with torch.no_grad():
                    for n, p in self.model.named_parameters():
                        p.add_(self.step_size * w_mom[n] / self.mass_theta)

                self.hp_state.step_positions(self.step_size, self.mass_lambda)
                self._update_loss_weights()

                # Half-step momenta again
                self.model.zero_grad()
                points_lf2 = self._sample_points()
                loss_lf2, _, res_lf2, bc_p_lf2, bc_t_lf2, ic_p_lf2, ic_t_lf2 = self._compute_pinn_loss(points_lf2)
                if torch.isfinite(loss_lf2):
                    loss_lf2.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    with torch.no_grad():
                        for n, p in self.model.named_parameters():
                            if p.grad is not None:
                                w_mom[n] -= (self.step_size / 2) * p.grad.clone()

                    hp_grads2 = finite_diff_pinn_hp_grads(
                        self.model, self.hp_state, self.loss_fn,
                        res_lf2.detach(), bc_p_lf2.detach(), bc_t_lf2,
                        ic_p_lf2.detach() if ic_p_lf2 is not None else None,
                        ic_t_lf2,
                        float(loss_lf2.item()),
                    )
                    self.hp_state.step_momenta(hp_grads2, self.step_size / 2, self.mass_lambda)

            # --- Compute proposed Hamiltonian ---
            with torch.no_grad():
                points_prop = self._sample_points()
                loss_prop, comp_prop, *_ = self._compute_pinn_loss(points_prop)

            H_prop = float(loss_prop.item()) + self.hp_state.kinetic_energy(self.mass_lambda)
            H_prop += sum(float((p**2).sum()) / (2*self.mass_theta) for p in w_mom.values())

            # --- Metropolis accept/reject ---
            dH = H_prop - H_cur
            accept = False
            if math.isfinite(dH):
                if dH < 0:
                    accept = True
                else:
                    accept = (np.random.rand() < math.exp(-dH / self.temperature))

            if not accept or not math.isfinite(float(loss_prop.item())):
                # Reject: revert to snapshot
                self.model.load_state_dict(theta_snap)
                self.hp_state.restore(hp_snap)
                for k, v in mom_snap.items():
                    self.hp_state.momenta[k] = v
                self._update_loss_weights()
                comp_log = comp_cur
            else:
                accepts += 1
                comp_log = comp_prop

            acc_rate = accepts / (epoch + 1)
            self._log_epoch(comp_log, phase="hmc", acc_rate=acc_rate)

            if (epoch + 1) % 25 == 0:
                decoded = self.hp_state.decode()
                print(f"  [HMC {epoch+1}/{self.n_hmc}] "
                      f"Loss={comp_log['L_total']:.6f} "
                      f"w_r={decoded.get('w_r', 1.0):.3f} "
                      f"w_b={decoded.get('w_b', 1.0):.3f} "
                      f"w_i={decoded.get('w_i', 1.0):.3f} "
                      f"lr={decoded.get('lr', 0.001):.5f} "
                      f"acc={acc_rate:.2f}")

    # ======================================================================= #
    #  Main Training Loop                                                     #
    # ======================================================================= #

    def train(self) -> Dict:
        """
        Run the full HHD-PINN training pipeline.

        Returns
        -------
        history : dict
            Training history with all loss components, weights, and metrics.
        """
        print(f"\n{'='*60}")
        print(f"HHD-PINN Training: {self.problem.name}")
        print(f"  Phase 1: Adam warmup ({self.n_warmup} epochs)")
        print(f"  Phase 2: HMC co-evolution ({self.n_hmc} epochs)")
        print(f"{'='*60}")

        t0 = time.time()

        print("\n--- Phase 1: Adam Warmup (fixed weights) ---")
        self._warmup_phase()

        print("\n--- Phase 2: HMC Co-Evolution (loss weights evolve) ---")
        self._hmc_phase()

        elapsed = time.time() - t0
        print(f"\nTraining complete in {elapsed:.1f}s")

        # Restore best checkpoint
        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)
            print(f"Restored best checkpoint (loss={self._best_loss:.6f})")
            if self._best_hp:
                print(f"Best HP: {self._best_hp}")

        final_error = self._compute_rel_l2()
        print(f"Final relative L2 error: {final_error:.6f}")

        return self.history


# --------------------------------------------------------------------------- #
#  Baseline PINN Trainer (Fixed Weights — for Comparison)
# --------------------------------------------------------------------------- #

class BaselinePINNTrainer:
    """
    Standard PINN trainer with FIXED loss weights.

    This serves as the baseline comparison — same architecture and training
    budget, but loss weights (w_r, w_b, w_i) are fixed at 1.0 throughout.
    Uses Adam optimizer only (no HMC, no weight evolution).
    """

    def __init__(
        self,
        model: PINN,
        problem,
        n_epochs: int = 300,
        lr: float = 1e-3,
        w_residual: float = 1.0,
        w_boundary: float = 1.0,
        w_initial: float = 1.0,
        device: str = "cpu",
        n_collocation: int = 2000,
        n_boundary_pts: int = 200,
        n_initial_pts: int = 200,
        grad_clip: float = 1.0,
    ):
        self.model = model.to(device)
        self.problem = problem
        self.device = device
        self.n_epochs = n_epochs
        self.lr = lr
        self.loss_fn = PINNLoss(w_residual, w_boundary, w_initial)
        self.n_collocation = n_collocation
        self.n_boundary = n_boundary_pts
        self.n_initial = n_initial_pts
        self.grad_clip = grad_clip
        self.is_time_dependent = hasattr(problem, 't_range')

        self.history = {
            "total_loss": [], "L_residual": [], "L_boundary": [],
            "L_initial": [], "rel_l2_error": [],
        }
        self._best_loss = float("inf")
        self._best_state = None

    def _sample_points(self):
        if self.is_time_dependent:
            return self.problem.sample_collocation(
                self.n_collocation, self.n_boundary, self.n_initial, self.device)
        else:
            return self.problem.sample_collocation(
                self.n_collocation, self.n_boundary, device=self.device)

    def _compute_pinn_loss(self, points):
        if self.is_time_dependent:
            residuals = self.problem.residual(self.model, points["x_int"], points["t_int"])
            xt_bc = torch.cat([points["x_bc"], points["t_bc"]], dim=1)
            bc_pred = self.model(xt_bc)
            xt_ic = torch.cat([points["x_ic"], points["t_ic"]], dim=1)
            ic_pred = self.model(xt_ic)
            return self.loss_fn.compute(residuals, bc_pred, points["bc_true"],
                                        ic_pred, points["ic_true"])
        else:
            residuals = self.problem.residual(self.model, points["x_int"], points["y_int"])
            xy_bc = torch.cat([points["x_bc"], points["y_bc"]], dim=1)
            bc_pred = self.model(xy_bc)
            return self.loss_fn.compute(residuals, bc_pred, points["bc_true"])

    def _compute_rel_l2(self):
        self.model.eval()
        with torch.no_grad():
            if self.is_time_dependent:
                x_t = torch.linspace(self.problem.x_range[0], self.problem.x_range[1],
                                     50, device=self.device).unsqueeze(1)
                t_t = torch.ones_like(x_t) * 0.5
                u_pred = self.model(torch.cat([x_t, t_t], dim=1))
                u_exact = self.problem.exact_solution(x_t, t_t)
            else:
                x = torch.linspace(0, 1, 30, device=self.device)
                y = torch.linspace(0, 1, 30, device=self.device)
                xx, yy = torch.meshgrid(x, y, indexing='ij')
                xy = torch.stack([xx.flatten(), yy.flatten()], dim=1)
                u_pred = self.model(xy)
                u_exact = self.problem.exact_solution(xy[:, 0:1], xy[:, 1:2])
        self.model.train()
        return relative_l2_error(u_pred, u_exact)

    def train(self) -> Dict:
        print(f"\n{'='*60}")
        print(f"Baseline PINN Training: {self.problem.name}")
        print(f"  Fixed weights: w_r={self.loss_fn.w_r}, w_b={self.loss_fn.w_b}, w_i={self.loss_fn.w_i}")
        print(f"  Epochs: {self.n_epochs}, lr: {self.lr}")
        print(f"{'='*60}")

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        t0 = time.time()

        for epoch in range(self.n_epochs):
            optimizer.zero_grad()
            points = self._sample_points()
            total_loss, components = self._compute_pinn_loss(points)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            optimizer.step()

            self.history["total_loss"].append(components["L_total"])
            self.history["L_residual"].append(components["L_residual"])
            self.history["L_boundary"].append(components["L_boundary"])
            self.history["L_initial"].append(components["L_initial"])

            if epoch % 10 == 0 or epoch < 5:
                self.history["rel_l2_error"].append(self._compute_rel_l2())
            elif self.history["rel_l2_error"]:
                self.history["rel_l2_error"].append(self.history["rel_l2_error"][-1])
            else:
                self.history["rel_l2_error"].append(1.0)

            if components["L_total"] < self._best_loss:
                self._best_loss = components["L_total"]
                self._best_state = deepcopy(self.model.state_dict())

            if (epoch + 1) % 50 == 0:
                print(f"  [Epoch {epoch+1}/{self.n_epochs}] "
                      f"Loss={components['L_total']:.6f} "
                      f"(Lr={components['L_residual']:.4f}, "
                      f"Lb={components['L_boundary']:.4f}, "
                      f"Li={components['L_initial']:.4f})")

        elapsed = time.time() - t0
        if self._best_state:
            self.model.load_state_dict(self._best_state)
        final_err = self._compute_rel_l2()
        print(f"\nDone in {elapsed:.1f}s. Final relative L2 error: {final_err:.6f}")
        return self.history
