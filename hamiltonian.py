"""
Hamiltonian Formulation for Self-Tuning Neural Networks.

Defines:
  - HamiltonianNN    : Feed-forward network that learns the energy surface H(q,p)
  - HyperparamState  : Continuous phase-space representation of hyperparameters
  - HamiltonianSystem: Computes the full Hamiltonian H = T_theta + T_lambda + L

Full Hamiltonian:
    H(theta, p_theta, lambda, p_lambda) = T(p) + V(theta, lambda)

Hamilton's Equations:
    d(theta)/dt  =  dH/dp_theta  = p_theta / m_theta
    dp_theta/dt  = -dH/d(theta)  = -dL/d(theta)
    d(lambda)/dt =  dH/dp_lambda = p_lambda / m_lambda
    dp_lambda/dt = -dH/d(lambda) = -dL/d(lambda)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple


# --------------------------------------------------------------------------- #
#  Neural-Network Architecture
# --------------------------------------------------------------------------- #

class HamiltonianNN(nn.Module):
    """
    Feed-forward network that learns the Hamiltonian energy surface.

    Input  : phase-space vector [q, p] (dim=2) or [q1, q2, p1, p2] (dim=4)
    Output : scalar energy H
    Depth/width/regularization controlled by hyperparameters lambda.
    """

    def __init__(self, n_layers: int, n_neurons: int, dropout: float,
                 input_dim: int = 2):
        super().__init__()
        self.n_layers     = n_layers
        self.n_neurons    = n_neurons
        self.dropout_rate = dropout
        self.input_dim    = input_dim

        self.input_layer = nn.Linear(input_dim, n_neurons)
        self.activation  = nn.ReLU()

        self.hidden = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_neurons, n_neurons),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for _ in range(n_layers)
        ])

        self.output_layer = nn.Linear(n_neurons, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.input_layer(x))
        for layer in self.hidden:
            x = layer(x)
        return self.output_layer(x)


# --------------------------------------------------------------------------- #
#  Continuous Hyperparameter State (lambda and p_lambda)
# --------------------------------------------------------------------------- #

class HyperparamState:
    """
    Maps hyperparameters to continuous phase-space variables with momenta.

    Continuous relaxations:
      log_lr         -> lr = 10^log_lr
      log_batch_size -> bs = 2^round(log_batch_size)
      n_layers       -> round(n_layers)
      n_neurons      -> round(n_neurons / 16) * 16
      dropout        -> clipped to [0, 0.3]
    """

    def __init__(
        self,
        init_values: Dict[str, float],
        bounds: Dict[str, Tuple[float, float]],
    ):
        self.bounds  = bounds
        self.values  = {k: torch.tensor([v], dtype=torch.float32)
                        for k, v in init_values.items()}
        self.momenta = {k: torch.zeros(1) for k in init_values}

    def decode(self) -> Dict:
        """Convert continuous values to actual hyperparameter values."""
        out: Dict = {}
        for k, v in self.values.items():
            lo, hi = self.bounds[k]
            val = float(np.clip(v.item(), lo, hi))

            if k == "log_lr":
                out["lr"] = 10 ** val
            elif k == "log_batch_size":
                out["batch_size"] = max(8, int(2 ** round(val)))
            elif k == "n_layers":
                out["n_layers"] = max(1, int(round(val)))
            elif k == "n_neurons":
                out["n_neurons"] = max(16, int(round(val / 16)) * 16)
            elif k == "dropout":
                out["dropout"] = float(val)
            else:
                out[k] = val
        return out

    def step_positions(self, eps: float, mass: float):
        """Update positions: lambda += eps * p_lambda * range / m_lambda"""
        frozen_hps = getattr(self, "frozen_hps", [])
        with torch.no_grad():
            for k in self.values:
                if k in frozen_hps:
                    continue
                lo, hi = self.bounds[k]
                hp_range = hi - lo
                self.values[k] += eps * self.momenta[k] * hp_range / mass
                self.values[k].clamp_(lo, hi)

    def step_momenta(self, grads: Dict[str, torch.Tensor], eps: float, mass: float):
        """Update momenta: p_lambda -= eps * dL/dlambda * range / m_lambda"""
        frozen_hps = getattr(self, "frozen_hps", [])
        with torch.no_grad():
            for k, g in grads.items():
                if k in frozen_hps:
                    continue
                if g is not None:
                    lo, hi = self.bounds[k]
                    hp_range = hi - lo
                    self.momenta[k] -= eps * g * hp_range / mass

    def randomise_momenta(self, mass: float):
        """Sample p_lambda ~ N(0, m_lambda) for fresh HMC trajectory."""
        frozen_hps = getattr(self, "frozen_hps", [])
        for k in self.momenta:
            if k in frozen_hps:
                self.momenta[k] = torch.zeros(1)
            else:
                self.momenta[k] = torch.randn(1) * float(np.sqrt(mass))

    def kinetic_energy(self, mass: float) -> float:
        """T_lambda = sum p_k^2 / (2 m_lambda)"""
        frozen_hps = getattr(self, "frozen_hps", [])
        return sum(
            float((p ** 2).sum()) / (2.0 * mass)
            for k, p in self.momenta.items() if k not in frozen_hps
        )

    def snapshot(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.values.items()}

    def restore(self, snap: Dict[str, torch.Tensor]):
        for k, v in snap.items():
            self.values[k] = v.clone()


# --------------------------------------------------------------------------- #
#  Hamiltonian System
# --------------------------------------------------------------------------- #

class HamiltonianSystem:
    """Computes the full Hamiltonian H = T_theta + T_lambda + L."""

    def __init__(self, mass_theta: float = 1.0, mass_lambda: float = 0.1):
        self.m_theta  = mass_theta
        self.m_lambda = mass_lambda

    def kinetic_theta(self, momenta: Dict[str, torch.Tensor]) -> float:
        return sum(
            float((p ** 2).sum()) / (2.0 * self.m_theta)
            for p in momenta.values()
        )

    def hamiltonian(
        self,
        loss: float,
        weight_momenta: Dict[str, torch.Tensor],
        hp_state: HyperparamState,
    ) -> float:
        T_theta  = self.kinetic_theta(weight_momenta)
        T_lambda = hp_state.kinetic_energy(self.m_lambda)
        return T_theta + T_lambda + loss
