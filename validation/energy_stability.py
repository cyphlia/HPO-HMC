"""
Theorem 1 & 4: Energy Conservation and Stability Analysis.
Implements Modified Hamiltonian monitoring and Leapfrog stability checks.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class ModifiedHamiltonianMonitor:
    """Theorem 1: Backward Error Analysis. Tracks |DH| <= C*eps^2."""

    def __init__(self):
        self.steps = []
        self.H_values = []
        self.H_modified = []
        self.H_initial = None
        self.grad_norms_theta = []
        self.grad_norms_lambda = []

    def compute_modified_hamiltonian(self, loss, w_mom, hp_state,
                                     mass_theta, mass_lambda, epsilon,
                                     grad_theta_norm, grad_lambda_norm):
        """H_tilde = H + (eps^2/12) * correction terms."""
        T_theta = sum(float((p**2).sum()) / (2.0 * mass_theta)
                      for p in w_mom.values())
        T_lambda = hp_state.kinetic_energy(mass_lambda)
        H = T_theta + T_lambda + loss

        # Second-order correction: H2 = (1/12)*(eps^2/m)*||grad||^2
        H2 = (epsilon**2 / 12.0) * (
            grad_theta_norm**2 / mass_theta +
            grad_lambda_norm**2 / mass_lambda
        )
        return H, H + H2

    def record_step(self, step_idx, H_current, H_modified, epsilon,
                    grad_theta_norm=0.0, grad_lambda_norm=0.0):
        """Store energy values at each leapfrog step."""
        if self.H_initial is None:
            self.H_initial = H_current
        self.steps.append(step_idx)
        self.H_values.append(H_current)
        self.H_modified.append(H_modified)
        self.grad_norms_theta.append(grad_theta_norm)
        self.grad_norms_lambda.append(grad_lambda_norm)

    def compute_drift_bound(self, epsilon, d_theta, d_lambda):
        """C*eps^2 where C ~ (L_smooth/2)*(d+k)."""
        if not self.grad_norms_theta:
            return epsilon**2
        avg_grad = np.mean(self.grad_norms_theta[-20:])
        L_smooth = avg_grad * 2.0  # conservative estimate
        C = (L_smooth / 2.0) * (d_theta + d_lambda)
        return C * epsilon**2

    def stability_report(self, epsilon, d_theta=100, d_lambda=5):
        """Returns dict with max_drift, bound, is_stable."""
        if not self.H_values:
            return {"max_drift": 0, "theoretical_bound": 0,
                    "safety_factor": 0, "is_stable": True}
        drifts = [abs(h - self.H_initial) for h in self.H_values]
        max_drift = max(drifts)
        bound = self.compute_drift_bound(epsilon, d_theta, d_lambda)
        bound = max(bound, 1e-10)
        return {
            "max_drift": max_drift,
            "theoretical_bound": bound,
            "safety_factor": max_drift / bound,
            "is_stable": max_drift < 10 * bound
        }

    def plot_energy_conservation(self, save_path, epsilon=0.005):
        """Three-panel: H(t), |DH|, H_tilde(t)."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        steps = np.array(self.steps)
        H = np.array(self.H_values)
        Hm = np.array(self.H_modified)

        axes[0].plot(steps, H, "b-", lw=1.5)
        axes[0].set_title("H(t) vs Step"); axes[0].set_ylabel("H")
        axes[0].grid(True, alpha=0.3)

        dH = np.abs(H - H[0])
        axes[1].plot(steps, dH, "r-", lw=1.5, label="|DH|")
        axes[1].axhline(epsilon**2, color="green", ls="--",
                        label=f"eps^2={epsilon**2:.2e}")
        axes[1].set_title("|DH(t)| vs Bound"); axes[1].legend(fontsize=8)
        axes[1].set_yscale("log"); axes[1].grid(True, alpha=0.3)

        axes[2].plot(steps, Hm, "purple", lw=1.5)
        axes[2].set_title("Modified H_tilde(t)"); axes[2].set_ylabel("H~")
        axes[2].grid(True, alpha=0.3)

        for ax in axes:
            ax.set_xlabel("Step")
        plt.suptitle("Theorem 1: Backward Error Analysis", fontweight="bold")
        plt.tight_layout(); plt.savefig(save_path, dpi=200); plt.close()


class StabilityAnalyzer:
    """Theorem 4: eps_critical = 2/sqrt(lambda_max(Hessian))."""

    def __init__(self, mass_theta=1.0, mass_lambda=0.5):
        self.m_theta = mass_theta
        self.m_lambda = mass_lambda
        self.history = []

    def estimate_hessian_max_eigenvalue(self, model, X, y, criterion,
                                        n_iter=20):
        """Power iteration for lambda_max(nabla^2 L)."""
        model.eval()
        params = [p for p in model.parameters() if p.requires_grad]
        v = [torch.randn_like(p) for p in params]
        v_norm = np.sqrt(sum(float((vi**2).sum()) for vi in v))
        v = [vi / v_norm for vi in v]

        lam = 1.0
        for _ in range(n_iter):
            model.zero_grad()
            loss = criterion(model(X), y)
            grads = torch.autograd.grad(loss, params, create_graph=True)
            hvp = torch.autograd.grad(
                grads, params, grad_outputs=v, retain_graph=False
            )
            lam = np.sqrt(sum(float((h**2).sum()) for h in hvp))
            if lam < 1e-12:
                break
            v = [h / lam for h in hvp]
        model.train()
        return max(lam, 1e-6)

    def compute_critical_step_size(self, lambda_max_theta,
                                    lambda_max_lambda=1.0):
        """eps_crit = 2 / max(sqrt(lam_theta/m_theta), sqrt(lam_lambda/m_lambda))."""
        omega_theta = np.sqrt(lambda_max_theta / self.m_theta)
        omega_lambda = np.sqrt(lambda_max_lambda / self.m_lambda)
        omega_max = max(omega_theta, omega_lambda, 1e-6)
        return 2.0 / omega_max

    def check_stability(self, current_eps, lambda_max_theta,
                        lambda_max_lambda=1.0):
        """Returns stability diagnostics dict."""
        eps_crit = self.compute_critical_step_size(
            lambda_max_theta, lambda_max_lambda)
        margin = (eps_crit - current_eps) / eps_crit
        kappa = lambda_max_theta / max(1e-6, lambda_max_lambda)
        result = {
            "epsilon_current": current_eps,
            "epsilon_critical": eps_crit,
            "stability_margin": margin,
            "is_stable": current_eps < eps_crit,
            "recommended_epsilon": 0.9 * eps_crit,
            "condition_number_kappa": kappa,
        }
        self.history.append(result)
        return result

    def plot_stability_analysis(self, save_path):
        """Phase diagram of stability over training."""
        if not self.history:
            return
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        eps_curr = [h["epsilon_current"] for h in self.history]
        eps_crit = [h["epsilon_critical"] for h in self.history]
        margins = [h["stability_margin"] for h in self.history]
        kappas = [h["condition_number_kappa"] for h in self.history]
        x = range(len(self.history))

        axes[0].plot(x, eps_curr, "r-", lw=2, label="Current eps")
        axes[0].plot(x, eps_crit, "g--", lw=2, label="Critical eps")
        axes[0].fill_between(x, eps_curr, eps_crit, alpha=0.15, color="green")
        axes[0].set_title("Step Size vs Critical"); axes[0].legend(fontsize=8)
        axes[0].set_ylabel("epsilon"); axes[0].grid(True, alpha=0.3)

        axes[1].plot(x, margins, "b-", lw=2)
        axes[1].axhline(0, color="red", ls="--")
        axes[1].set_title("Stability Margin"); axes[1].set_ylabel("Margin")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(x, kappas, "orange", lw=2)
        axes[2].set_title("Condition Number kappa")
        axes[2].set_yscale("log"); axes[2].grid(True, alpha=0.3)

        for ax in axes:
            ax.set_xlabel("Check Index")
        plt.suptitle("Theorem 4: Leapfrog Stability", fontweight="bold")
        plt.tight_layout(); plt.savefig(save_path, dpi=200); plt.close()
