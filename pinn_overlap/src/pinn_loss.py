"""
Multi-Component PINN Loss.

Why this matters:
    A PINN's loss function isn't like a normal ML loss. It's a *weighted sum*
    of multiple physics-based terms that often conflict with each other:

        L_total = w_r · L_residual  +  w_b · L_boundary  +  w_i · L_initial

    If the weights are wrong, training fails:
    - w_r too large  → network satisfies PDE interior but ignores boundary conditions
    - w_b too large  → network nails boundaries but PDE residual is garbage
    - w_i too large  → network fits initial condition but can't propagate forward in time

    Finding the right weights is THE central open problem in PINN training.
    This module computes each component separately so HHD can tune the weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Tuple


class PINNLoss:
    """
    Computes the weighted multi-component PINN loss.

    The three standard components are:
        L_r = (1/N_r) Σ |residual(x_i, t_i)|²   — PDE residual at collocation points
        L_b = (1/N_b) Σ |u(x_bc) - g(x_bc)|²    — boundary condition error
        L_i = (1/N_i) Σ |u(x_ic, 0) - u_0(x_ic)|² — initial condition error

    Parameters
    ----------
    w_residual : float
        Weight for PDE residual loss (default 1.0)
    w_boundary : float
        Weight for boundary condition loss (default 1.0)
    w_initial : float
        Weight for initial condition loss (default 1.0)
    """

    def __init__(
        self,
        w_residual: float = 1.0,
        w_boundary: float = 1.0,
        w_initial: float = 1.0,
    ):
        self.w_r = w_residual
        self.w_b = w_boundary
        self.w_i = w_initial

    def compute(
        self,
        residuals: torch.Tensor,
        bc_pred: torch.Tensor,
        bc_true: torch.Tensor,
        ic_pred: torch.Tensor = None,
        ic_true: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute weighted total loss and individual components.

        Parameters
        ----------
        residuals : Tensor (N_r, 1)
            PDE residual values at collocation points (should be → 0)
        bc_pred : Tensor (N_b, 1)
            Model predictions at boundary points
        bc_true : Tensor (N_b, 1)
            True boundary values
        ic_pred : Tensor (N_i, 1), optional
            Model predictions at initial condition points
        ic_true : Tensor (N_i, 1), optional
            True initial condition values

        Returns
        -------
        total_loss : Tensor (scalar)
            Weighted sum of all components
        components : dict
            Individual (unweighted) loss values for monitoring
        """
        # PDE residual loss: mean squared residual
        L_r = torch.mean(residuals ** 2)

        # Boundary condition loss
        L_b = torch.mean((bc_pred - bc_true) ** 2)

        # Initial condition loss (if applicable — not used for steady-state PDEs)
        if ic_pred is not None and ic_true is not None:
            L_i = torch.mean((ic_pred - ic_true) ** 2)
        else:
            L_i = torch.tensor(0.0, device=L_r.device)

        # Weighted total
        total = self.w_r * L_r + self.w_b * L_b + self.w_i * L_i

        components = {
            "L_residual": float(L_r.item()),
            "L_boundary": float(L_b.item()),
            "L_initial": float(L_i.item()),
            "L_total": float(total.item()),
            "w_r": self.w_r,
            "w_b": self.w_b,
            "w_i": self.w_i,
        }

        return total, components

    def set_weights(self, w_r: float, w_b: float, w_i: float):
        """Update loss weights (called by HHD trainer during co-evolution)."""
        self.w_r = w_r
        self.w_b = w_b
        self.w_i = w_i


def relative_l2_error(u_pred: torch.Tensor, u_exact: torch.Tensor) -> float:
    """
    Relative L2 error: ||u_pred - u_exact||₂ / ||u_exact||₂

    This is the standard accuracy metric for PINNs. A value of 0.01 means
    the prediction is within 1% of the exact solution.
    """
    with torch.no_grad():
        error = torch.norm(u_pred - u_exact) / (torch.norm(u_exact) + 1e-12)
    return float(error.item())
