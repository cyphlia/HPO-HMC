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

import math
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
            # Perturb dropout rate and evaluate.
            # Fix: Keep in train mode and set seed so the dropout mask is
            # deterministic, otherwise dropout is completely disabled in eval mode
            # and returns a zero gradient.
            def _eval_dropout(dval):
                old_d = model.dropout_rate
                model.dropout_rate = dval
                for m in model.modules():
                    if isinstance(m, nn.Dropout):
                        m.p = dval
                torch.manual_seed(42)
                model.train()
                loss_val = float(criterion(model(X), y).item())
                model.dropout_rate = old_d
                for m in model.modules():
                    if isinstance(m, nn.Dropout):
                        m.p = old_d
                return loss_val

            grads[k] = torch.tensor([(_eval_dropout(vp) - _eval_dropout(vm)) / denom])

        elif k == "log_wd":
            # Analytical gradient for weight decay log_wd (wd = 10^log_wd)
            # L_reg = L_data + 0.5 * wd * ||w||^2
            # dL/d(log_wd) = dL/d(wd) * d(wd)/d(log_wd) = 0.5 * ||w||^2 * 10^log_wd * ln(10)
            wd = 10 ** old_val
            w_norm_sq = sum(float((p ** 2).sum()) for p in model.parameters())
            grads[k] = torch.tensor([0.5 * w_norm_sq * wd * np.log(10.0)])

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
                    input_dim=model.input_dim if hasattr(model, "input_dim") else 2,
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

        # Add Logarithmic Barrier Potential gradient to prevent boundary sticking/chatter
        # V_barrier = -barrier_coef * [ln(val - lo) + ln(hi - val)]
        # dV/dlambda = -barrier_coef * [1/(val - lo) - 1/(hi - val)]
        barrier_coef = 1e-4
        dist_lo = max(1e-6, old_val - lo)
        dist_hi = max(1e-6, hi - old_val)
        g_barrier = -barrier_coef * (1.0 / dist_lo - 1.0 / dist_hi)
        grads[k] = grads[k] + g_barrier

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
    grad_clip    : max L2 norm for the weight-gradient vector used in each
                   momentum half/full-step (None disables clipping).

                   BUGFIX CONTEXT: before the temperature-scaling fix in
                   HamiltonianMCMC, "always-accept" mode (T=1e9) didn't
                   actually behave that way (see that class's docstring),
                   which accidentally limited how far theta/lambda could
                   wander. Now that acceptance genuinely approaches 100%,
                   long runs (e.g. 80 epochs) can occasionally accumulate
                   large enough gradients that theta or log_lr diverges to
                   inf, and inf - inf in the finite-difference HP-gradient
                   computation produces nan, which then corrupts hp_state
                   permanently and crashes optimizer construction downstream.
                   Clipping the gradient norm used inside the leapfrog step
                   (mirroring the grad-clipping already used in the Adam
                   phases elsewhere in this codebase) keeps individual
                   leapfrog steps bounded and prevents this divergence at
                   the source, rather than only detecting it after the fact.
    """

    def __init__(self, step_size=0.01, n_steps=5, mass_theta=1.0, mass_lambda=0.1,
                 grad_clip: float = 10.0):
        self.eps       = step_size
        self.L         = n_steps
        self.m_theta   = mass_theta
        self.m_lambda  = mass_lambda
        self.grad_clip = grad_clip

    def _clip_grads_(self, wg: Dict[str, torch.Tensor]):
        if not self.grad_clip:
            return
        total_norm = math.sqrt(sum(float((g ** 2).sum()) for g in wg.values()) + 1e-12)
        if total_norm > self.grad_clip:
            scale = self.grad_clip / (total_norm + 1e-12)
            for n in wg:
                wg[n] = wg[n] * scale

    def _mom_step(self, model, w_mom, hp_state, batch, criterion, coeff, val_batch=None):
        """Momentum update for both theta and lambda."""
        loss, wg = compute_loss_and_grads(model, batch, criterion)
        self._clip_grads_(wg)
        
        hp_batch = val_batch if val_batch is not None else batch
        loss_hp = loss
        if val_batch is not None:
            model.eval()
            with torch.no_grad():
                loss_hp = float(criterion(model(val_batch[0]), val_batch[1]).item())
            model.train()

        hg = finite_diff_hp_grads(model, hp_state, hp_batch, criterion, loss_hp)
        for n in w_mom:
            w_mom[n] -= coeff * wg[n] / self.m_theta
        hp_state.step_momenta(hg, coeff, self.m_lambda)
        return loss_hp

    def _pos_step(self, model, w_mom, hp_state):
        """Position update for both theta and lambda."""
        for n, p in model.named_parameters():
            p.data += self.eps * w_mom[n] / self.m_theta
        hp_state.step_positions(self.eps, self.m_lambda)

    def integrate(self, model, w_mom, hp_state, batch, criterion, val_batch=None):
        """Execute one full leapfrog trajectory (L sub-steps)."""
        self._mom_step(model, w_mom, hp_state, batch, criterion, self.eps / 2, val_batch=val_batch)
        for _ in range(self.L - 1):
            self._pos_step(model, w_mom, hp_state)
            self._mom_step(model, w_mom, hp_state, batch, criterion, self.eps, val_batch=val_batch)
        self._pos_step(model, w_mom, hp_state)
        final_loss = self._mom_step(model, w_mom, hp_state, batch, criterion, self.eps / 2, val_batch=val_batch)
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

    momentum_refresh
    -----------------
    Controls how much of the momentum survives between proposals.

      momentum_refresh = 1.0 (default, ORIGINAL BEHAVIOUR)
        Momentum is fully resampled ~N(0, m) before every proposal. This
        is correct HMC, but for T=1e9 "optimisation mode" (always-accept)
        it means every proposal starts from a fresh random direction with
        no memory of previous progress -- the trajectory has no way to
        accumulate a directed descent, so it behaves like a noisy random
        walk around the loss landscape rather than an optimizer. This is
        why Method A's *final* epoch is often much worse than its *best*
        epoch (see train_hamiltonian.py's lack of checkpointing, fixed
        separately below): the dynamics wander away from good regions
        with nothing pulling them back.

      momentum_refresh < 1.0 (e.g. 0.05-0.2)
        Partial momentum refresh (Generalized HMC / Horowitz 1991):
            p <- sqrt(1 - alpha) * p_prev + sqrt(alpha) * fresh_noise
        so momentum persists across proposals and can accumulate real
        directed velocity, similar in spirit to momentum/Nesterov SGD,
        while still being derived correctly from the Hamiltonian
        formalism. On REJECTION, momentum is negated (p <- -p) rather
        than discarded -- this is required for detailed balance to still
        hold under partial refresh (Horowitz 1991); without it, rejected
        proposals would bias the chain.
    """

    def __init__(self, step_size=0.005, n_leapfrog=5,
                 mass_theta=1.0, mass_lambda=0.1, temperature=1.0,
                 momentum_refresh: float = 1.0):
        self.leapfrog    = LeapfrogIntegrator(step_size, n_leapfrog, mass_theta, mass_lambda)
        self.m_theta     = mass_theta
        self.m_lambda    = mass_lambda
        self.temperature = temperature
        self.n_acc       = 0
        self.n_prop      = 0
        self.momentum_refresh = float(np.clip(momentum_refresh, 0.0, 1.0))
        self._w_mom      = None   # persistent weight momenta (None until first proposal)

    def _kinetic_theta(self, w_mom):
        return sum(float((p ** 2).sum()) / (2.0 * self.m_theta)
                   for p in w_mom.values())

    def _refresh_momenta(self, model, hp_state):
        """Full resample (alpha=1) or partial refresh (alpha<1) of momenta."""
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

    def propose(self, model, hp_state, batch, criterion, current_loss, val_batch=None):
        """
        Propose a new (theta, lambda) state via leapfrog and accept/reject.

        Returns (accepted: bool, loss: float)
        """
        self.n_prop += 1
        saved_w  = deepcopy(model.state_dict())
        saved_hp = hp_state.snapshot()

        w_mom = self._refresh_momenta(model, hp_state)

        loss_init = current_loss
        if val_batch is not None:
            model.eval()
            with torch.no_grad():
                loss_init = float(criterion(model(val_batch[0]), val_batch[1]).item())
            model.train()

        H_init = (self._kinetic_theta(w_mom)
                  + hp_state.kinetic_energy(self.m_lambda)
                  + loss_init)

        # Leapfrog trajectory (mutates w_mom / hp_state.momenta in place)
        proposed_loss = self.leapfrog.integrate(model, w_mom, hp_state, batch, criterion, val_batch=val_batch)

        H_prop = (self._kinetic_theta(w_mom)
                  + hp_state.kinetic_energy(self.m_lambda)
                  + proposed_loss)

        # SAFETY GUARD (new): a proposal that produced nan/inf -- e.g. from
        # weights or a hyperparameter (log_lr) diverging during the leapfrog
        # trajectory, which then makes the finite-difference HP-gradient
        # compute inf - inf = nan -- must never be accepted, no matter what
        # temperature says. Accepting a corrupted state doesn't just make
        # this proposal bad, it permanently poisons hp_state/theta for every
        # future proposal (this is what previously crashed Method C with a
        # "nan learning rate" error partway through an 80-epoch run: the
        # temperature-scaling fix above made acceptance genuinely near 100%,
        # which occasionally let a nan-loss proposal through). Gradient
        # clipping in LeapfrogIntegrator (above) reduces how often this state
        # is reached at all; this guard makes sure it's never accepted if it
        # is reached.
        proposal_is_finite = np.isfinite(proposed_loss) and np.isfinite(H_prop)

        # Metropolis-Hastings acceptance: min(1, exp(-dH/T)) (Eq. eq:mh).
        # BUGFIX: temperature must scale the *entire* delta-H at the
        # acceptance step, not just the loss term before H is composed.
        # The previous version divided only `current_loss`/`proposed_loss`
        # by temperature before summing with the (unscaled) kinetic terms,
        # which are what the leapfrog dynamics actually evolve under (the
        # gradients driving position/momentum updates are never scaled by
        # temperature). At T=1e9 that made the loss term ~0 inside H, but
        # left the acceptance criterion fully exposed to whatever kinetic
        # energy drift the leapfrog trajectory picked up -- so instead of
        # "always accept" (the documented "optimisation mode"), it produced
        # acceptance rates of 25-45%, rejecting perfectly good loss-improving
        # proposals purely because of a kinetic-energy mismatch that a true
        # high-temperature limit is supposed to make irrelevant.
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
            if self.momentum_refresh < 1.0:
                # Negate momentum on rejection (Generalized HMC / Horowitz 1991)
                # -- required for detailed balance under partial refresh.
                self._w_mom = {n: -p for n, p in w_mom.items()}
                frozen_hps = getattr(hp_state, "frozen_hps", [])
                for k in hp_state.momenta:
                    if k not in frozen_hps:
                        hp_state.momenta[k] = -hp_state.momenta[k]
            else:
                self._w_mom = None  # next call resamples fresh, as before
            return False, current_loss

    @property
    def acceptance_rate(self):
        return self.n_acc / max(self.n_prop, 1)
