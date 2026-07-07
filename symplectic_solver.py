"""
Symplectic Leapfrog (Stormer-Verlet) Integrator + HMC Sampler.

The leapfrog scheme exactly preserves a modified Hamiltonian H' ~ H,
guaranteeing bounded energy error over arbitrarily long trajectories.

One full leapfrog trajectory (L sub-steps):
  1. Half-momentum:  p <- p - (eps/2) * grad_q H
  2. For i = 1..L-1:
        Full-position: q <- q + eps * p/m
        Full-momentum: p <- p - eps * grad_q H
  3. Final position:  q <- q + eps * p/m
  4. Half-momentum:   p <- p - (eps/2) * grad_q H
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from typing import Dict, Tuple

from hamiltonian import HamiltonianNN, HyperparamState


# --------------------------------------------------------------------------- #
#  Gradient Computation
# --------------------------------------------------------------------------- #

def compute_loss_and_grads(model, batch, criterion):
    """Forward + backward pass; returns loss scalar and weight gradients."""
    X, y = batch
    model.train()
    model.zero_grad()
    loss = criterion(model(X), y)
    loss.backward()
    grads = {
        n: (p.grad.clone() if p.grad is not None else torch.zeros_like(p))
        for n, p in model.named_parameters()
    }
    return float(loss.item()), grads


def finite_diff_hp_grads(model, hp_state, batch, criterion, base_loss):
    """
    Finite-difference approximation of dL/d(lambda) for ALL hyperparameters.

    Each HP is perturbed +/- eps along its range, and the gradient is
    estimated as (L(lambda+eps) - L(lambda-eps)) / (2*eps).

    For structural HPs (n_layers, n_neurons): a fresh model is built
    since the architecture changes. The gradient here is "how much does
    loss improve if I change the architecture?" — a proxy for capacity.

    For log_lr: we perturb the optimizer LR and run a micro-step to
    estimate the sensitivity of loss to learning rate changes.
    """
    X, y = batch
    grads = {}

    frozen_hps = getattr(model, "frozen_hps", []) or getattr(hp_state, "frozen_hps", [])
    for k, v in hp_state.values.items():
        if k in frozen_hps:
            grads[k] = torch.tensor([0.0])
            continue
        old_val = float(v.item())
        lo, hi = hp_state.bounds[k]
        eps = 0.05 * (hi - lo)  # 5% perturbation
        vp = min(hi, old_val + eps)
        vm = max(lo, old_val - eps)
        denom = vp - vm

        if denom < 1e-10:
            grads[k] = torch.tensor([0.0])
            continue

        if k == "dropout":
            # Perturb dropout rate and evaluate (use eval mode for clean gradients)
            def _eval_dropout(dval):
                old_d = model.dropout_rate
                model.dropout_rate = dval
                for m in model.modules():
                    if isinstance(m, nn.Dropout):
                        m.p = dval
                model.eval()
                with torch.no_grad():
                    loss_val = float(criterion(model(X), y).item())
                model.dropout_rate = old_d
                for m in model.modules():
                    if isinstance(m, nn.Dropout):
                        m.p = old_d
                model.train()
                return loss_val

            grads[k] = torch.tensor([(_eval_dropout(vp) - _eval_dropout(vm)) / denom])

        elif k in ("n_layers", "n_neurons"):
            # Structural HP: build temporary model with perturbed architecture
            def _eval_struct(val):
                saved = float(hp_state.values[k].item())
                hp_state.values[k].data.fill_(val)
                dec = hp_state.decode()
                hp_state.values[k].data.fill_(saved)
                device = next(model.parameters()).device
                tmp = HamiltonianNN(
                    n_layers=dec["n_layers"],
                    n_neurons=dec["n_neurons"],
                    dropout=dec["dropout"],
                ).to(device)
                tmp.eval()
                with torch.no_grad():
                    return float(criterion(tmp(X), y).item())

            grads[k] = torch.tensor([(_eval_struct(vp) - _eval_struct(vm)) / denom])

        elif k == "log_lr":
            # Perturb learning rate: run 1 micro-step at each LR and compare
            import torch.optim as _optim
            device = next(model.parameters()).device
            Xd, yd = X.to(device), y.to(device)

            def _eval_lr(log_lr_val):
                lr = 10 ** np.clip(log_lr_val, -4, -1)
                saved_state = {n: p.data.clone() for n, p in model.named_parameters()}
                opt = _optim.SGD(model.parameters(), lr=lr)
                model.train()
                opt.zero_grad()
                criterion(model(Xd), yd).backward()
                opt.step()
                model.eval()
                with torch.no_grad():
                    loss_after = float(criterion(model(Xd), yd).item())
                # Restore weights
                for n, p in model.named_parameters():
                    p.data.copy_(saved_state[n])
                model.train()
                return loss_after

            grads[k] = torch.tensor([(_eval_lr(vp) - _eval_lr(vm)) / denom])

        elif k == "log_batch_size":
            # Batch size gradient: larger batches = lower gradient noise
            # Approximate as sensitivity to noise level
            def _eval_bs(log_bs_val):
                bs = max(8, int(2 ** round(np.clip(log_bs_val, 4, 6))))
                n = min(bs, len(X))
                idx = torch.randperm(len(X))[:n]
                model.eval()
                with torch.no_grad():
                    return float(criterion(model(X[idx]), y[idx]).item())

            # Average over a few samples for stability
            lp = np.mean([_eval_bs(vp) for _ in range(3)])
            lm = np.mean([_eval_bs(vm) for _ in range(3)])
            grads[k] = torch.tensor([(lp - lm) / denom])

        else:
            grads[k] = torch.tensor([0.0])

    return grads


# --------------------------------------------------------------------------- #
#  Leapfrog Integrator
# --------------------------------------------------------------------------- #

class LeapfrogIntegrator:
    """
    Symplectic leapfrog for joint (theta, lambda) dynamics.

    Parameters
    ----------
    step_size    : eps - leapfrog step size
    n_steps      : L - number of leapfrog sub-steps
    mass_theta   : inertia of the weight sub-system
    mass_lambda  : inertia of the hyperparameter sub-system
    """

    def __init__(self, step_size=0.01, n_steps=5, mass_theta=1.0, mass_lambda=0.1):
        self.eps       = step_size
        self.L         = n_steps
        self.m_theta   = mass_theta
        self.m_lambda  = mass_lambda

    def _mom_step(self, model, w_mom, hp_state, batch, criterion, coeff):
        """Momentum update for both theta and lambda."""
        loss, wg = compute_loss_and_grads(model, batch, criterion)
        hg = finite_diff_hp_grads(model, hp_state, batch, criterion, loss)
        for n in w_mom:
            w_mom[n] -= coeff * wg[n] / self.m_theta
        hp_state.step_momenta(hg, coeff, self.m_lambda)
        return loss

    def _pos_step(self, model, w_mom, hp_state):
        """Position update for both theta and lambda."""
        for n, p in model.named_parameters():
            p.data += self.eps * w_mom[n] / self.m_theta
        hp_state.step_positions(self.eps, self.m_lambda)

    def integrate(self, model, w_mom, hp_state, batch, criterion):
        """Execute one full leapfrog trajectory (L sub-steps)."""
        self._mom_step(model, w_mom, hp_state, batch, criterion, self.eps / 2)
        for _ in range(self.L - 1):
            self._pos_step(model, w_mom, hp_state)
            self._mom_step(model, w_mom, hp_state, batch, criterion, self.eps)
        self._pos_step(model, w_mom, hp_state)
        final_loss = self._mom_step(model, w_mom, hp_state, batch, criterion, self.eps / 2)
        return final_loss


# --------------------------------------------------------------------------- #
#  Hamiltonian Monte Carlo Sampler
# --------------------------------------------------------------------------- #

class HamiltonianMCMC:
    """
    HMC sampler with Metropolis-Hastings accept/reject.

    Uses the leapfrog trajectory as the proposal. Accepts with
    probability min(1, exp(-dH)), which preserves the target
    distribution. Recommended acceptance rate: 60-80%.
    """

    def __init__(self, step_size=0.005, n_leapfrog=5,
                 mass_theta=1.0, mass_lambda=0.1, temperature=1.0):
        self.leapfrog    = LeapfrogIntegrator(step_size, n_leapfrog, mass_theta, mass_lambda)
        self.m_theta     = mass_theta
        self.m_lambda    = mass_lambda
        self.temperature = temperature
        self.n_acc       = 0
        self.n_prop      = 0

    def _kinetic_theta(self, w_mom):
        return sum(float((p ** 2).sum()) / (2.0 * self.m_theta)
                   for p in w_mom.values())

    def propose(self, model, hp_state, batch, criterion, current_loss):
        """
        Propose a new (theta, lambda) state via leapfrog and accept/reject.

        Returns (accepted: bool, loss: float)
        """
        self.n_prop += 1
        saved_w  = deepcopy(model.state_dict())
        saved_hp = hp_state.snapshot()

        # Fresh momenta
        w_mom = {
            n: torch.randn_like(p) * float(np.sqrt(self.m_theta))
            for n, p in model.named_parameters()
        }
        hp_state.randomise_momenta(self.m_lambda)

        H_init = (self._kinetic_theta(w_mom)
                  + hp_state.kinetic_energy(self.m_lambda)
                  + current_loss / self.temperature)

        # Leapfrog trajectory
        proposed_loss = self.leapfrog.integrate(model, w_mom, hp_state, batch, criterion)

        H_prop = (self._kinetic_theta(w_mom)
                  + hp_state.kinetic_energy(self.m_lambda)
                  + proposed_loss / self.temperature)

        # Metropolis-Hastings acceptance
        if np.random.random() < np.exp(min(0, -(H_prop - H_init))):
            self.n_acc += 1
            return True, proposed_loss
        else:
            model.load_state_dict(saved_w)
            hp_state.restore(saved_hp)
            return False, current_loss

    @property
    def acceptance_rate(self):
        return self.n_acc / max(self.n_prop, 1)
