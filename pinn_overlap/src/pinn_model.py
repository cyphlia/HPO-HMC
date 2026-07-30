"""
PINN Neural Network Model.

A Physics-Informed Neural Network is just a standard feed-forward network,
but with one crucial difference: we use torch.autograd.grad to compute the
*exact* partial derivatives of the network output with respect to its inputs.

This lets us evaluate PDE residuals:
    If the PDE says  u_t = ν·u_xx,  then we compute:
        u_hat    = model(x, t)
        u_t      = d(u_hat)/dt       (via autograd)
        u_xx     = d²(u_hat)/dx²     (via autograd, twice)
        residual = u_t - ν·u_xx      (should be ≈ 0)

Architecture choices that matter for PINNs:
    - Activation: tanh is the standard (smooth, infinitely differentiable).
      ReLU is bad because its second derivative is zero everywhere.
    - Initialization: Xavier/Glorot uniform works well.
    - Input: (x, t) coordinates — NOT discretized grid data.
    - Output: scalar u(x, t) — the PDE solution value at that point.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Optional


class PINN(nn.Module):
    """
    Physics-Informed Neural Network.

    Takes spatial-temporal coordinates as input, outputs the field value.
    Supports autograd-based derivative computation for PDE residuals.

    Parameters
    ----------
    input_dim : int
        Number of input coordinates (2 for 1D+time, 3 for 2D+time, 2 for 2D steady)
    output_dim : int
        Number of output fields (1 for scalar PDEs like heat/Burgers)
    hidden_layers : list of int
        Width of each hidden layer, e.g. [64, 64, 64] for 3 layers of 64 neurons
    activation : str
        'tanh' (default, best for PINNs), 'sin' (Fourier/SIREN-style), or 'swish'
    dropout : float
        Dropout probability (usually 0 for PINNs — we want deterministic output)
    """

    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 1,
        hidden_layers: Optional[List[int]] = None,
        activation: str = "tanh",
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [64, 64, 64]

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_sizes = hidden_layers
        self.activation_name = activation

        # --- Build activation ---
        if activation == "tanh":
            act_fn = nn.Tanh
        elif activation == "sin":
            # SIREN-style: sin activation helps with oscillatory solutions
            act_fn = lambda: _SinActivation()
        elif activation == "swish":
            act_fn = nn.SiLU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # --- Build layers ---
        layers = []
        prev_dim = input_dim
        for width in hidden_layers:
            layers.append(nn.Linear(prev_dim, width))
            layers.append(act_fn() if callable(act_fn) else act_fn)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = width
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization — standard for PINNs."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor of shape (N, input_dim)
            Spatial-temporal coordinates. For 1D+time PDEs: columns are [x, t].
            IMPORTANT: x must have requires_grad=True for autograd derivatives.

        Returns
        -------
        u : Tensor of shape (N, output_dim)
            Predicted field values.
        """
        return self.net(x)


class _SinActivation(nn.Module):
    """Sin activation for SIREN-style networks."""
    def forward(self, x):
        return torch.sin(x)


# --------------------------------------------------------------------------- #
#  Autograd Derivative Utilities
# --------------------------------------------------------------------------- #

def grad(outputs: torch.Tensor, inputs: torch.Tensor,
         create_graph: bool = True) -> torch.Tensor:
    """
    Compute d(outputs)/d(inputs) using torch.autograd.grad.

    This is the workhorse of PINNs: it lets us compute exact partial
    derivatives of the network output w.r.t. its input coordinates.

    Parameters
    ----------
    outputs : Tensor of shape (N, 1)
        Network output (e.g., u_hat)
    inputs : Tensor of shape (N, D)
        Input coordinates (e.g., [x, t]). Must have requires_grad=True.
    create_graph : bool
        If True, allows computing higher-order derivatives (e.g., u_xx).

    Returns
    -------
    Tensor of shape (N, D)
        Gradient d(outputs)/d(inputs). Column i is d(outputs)/d(inputs[:, i]).
    """
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=create_graph,
        retain_graph=True,
    )[0]


def compute_derivatives_1d(model: PINN, x: torch.Tensor, t: torch.Tensor):
    """
    Compute u, u_t, u_x, u_xx for a 1D+time PDE.

    Usage:
        x = torch.rand(N, 1, requires_grad=True)
        t = torch.rand(N, 1, requires_grad=True)
        u, u_t, u_x, u_xx = compute_derivatives_1d(model, x, t)

    Parameters
    ----------
    model : PINN
    x : Tensor (N, 1) with requires_grad=True
    t : Tensor (N, 1) with requires_grad=True

    Returns
    -------
    u, u_t, u_x, u_xx : each Tensor (N, 1)
    """
    xt = torch.cat([x, t], dim=1)  # (N, 2)
    u = model(xt)                   # (N, 1)

    # First derivatives
    du = grad(u, xt)                # (N, 2) = [du/dx, du/dt]
    u_x = du[:, 0:1]               # du/dx
    u_t = du[:, 1:2]               # du/dt

    # Second derivative: u_xx = d(u_x)/dx
    du_x = grad(u_x, xt)           # (N, 2)
    u_xx = du_x[:, 0:1]            # d²u/dx²

    return u, u_t, u_x, u_xx


def compute_derivatives_2d(model: PINN, x: torch.Tensor, y: torch.Tensor):
    """
    Compute u, u_x, u_y, u_xx, u_yy for a 2D steady-state PDE (e.g. Poisson).

    Parameters
    ----------
    model : PINN
    x : Tensor (N, 1) with requires_grad=True
    y : Tensor (N, 1) with requires_grad=True

    Returns
    -------
    u, u_x, u_y, u_xx, u_yy : each Tensor (N, 1)
    """
    xy = torch.cat([x, y], dim=1)  # (N, 2)
    u = model(xy)                   # (N, 1)

    # First derivatives
    du = grad(u, xy)                # (N, 2) = [du/dx, du/dy]
    u_x = du[:, 0:1]
    u_y = du[:, 1:2]

    # Second derivatives
    du_x = grad(u_x, xy)
    u_xx = du_x[:, 0:1]

    du_y = grad(u_y, xy)
    u_yy = du_y[:, 1:2]

    return u, u_x, u_y, u_xx, u_yy
