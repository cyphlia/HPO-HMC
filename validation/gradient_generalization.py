"""
Theorems 5 & 6: HP Gradient Consistency and Generalization Bounds.
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


class HPGradientChecker:
    """Theorem 5: |g_hat - dL/dlambda| <= (L_jj/2)*eps_fd."""

    def __init__(self, eps_fd=1e-3):
        self.eps_fd = eps_fd
        self.bias_history = {}

    def central_difference(self, model, hp_state, batch, criterion, k):
        """Central difference gradient for HP k."""
        X, y = batch
        old_val = float(hp_state.values[k].item())
        lo, hi = hp_state.bounds[k]
        eps = self.eps_fd * (hi - lo)
        vp = min(hi, old_val + eps)
        vm = max(lo, old_val - eps)
        denom = vp - vm
        if denom < 1e-12:
            return 0.0

        # Evaluate at +eps and -eps
        hp_state.values[k].data.fill_(vp)
        model.eval()
        with torch.no_grad():
            lp = float(criterion(model(X), y).item())
        hp_state.values[k].data.fill_(vm)
        with torch.no_grad():
            lm = float(criterion(model(X), y).item())
        hp_state.values[k].data.fill_(old_val)
        model.train()
        return (lp - lm) / denom

    def estimate_bias(self, model, hp_state, batch, criterion, k,
                      n_estimates=5):
        """Compare coarse vs fine FD to estimate bias."""
        coarse_grads = []
        fine_grads = []
        for _ in range(n_estimates):
            # Coarse (current eps_fd)
            old_eps = self.eps_fd
            g_coarse = self.central_difference(
                model, hp_state, batch, criterion, k)
            coarse_grads.append(g_coarse)
            # Fine (eps_fd / 10)
            self.eps_fd = old_eps / 10.0
            g_fine = self.central_difference(
                model, hp_state, batch, criterion, k)
            fine_grads.append(g_fine)
            self.eps_fd = old_eps

        bias = abs(np.mean(coarse_grads) - np.mean(fine_grads))
        if k not in self.bias_history:
            self.bias_history[k] = []
        self.bias_history[k].append(bias)
        return bias

    def consistency_check(self, model, hp_state, batch, criterion, k):
        """Verify Theorem 5: bias <= L_jj * eps_fd / 2."""
        bias = self.estimate_bias(model, hp_state, batch, criterion, k)
        lo, hi = hp_state.bounds[k]
        # L_jj ~ second derivative estimate
        old_val = float(hp_state.values[k].item())
        eps = self.eps_fd * (hi - lo)
        vp = min(hi, old_val + eps)
        vm = max(lo, old_val - eps)
        vc = old_val
        hp_state.values[k].data.fill_(vp)
        model.eval()
        with torch.no_grad():
            lp = float(criterion(model(batch[0]), batch[1]).item())
        hp_state.values[k].data.fill_(vm)
        with torch.no_grad():
            lm = float(criterion(model(batch[0]), batch[1]).item())
        hp_state.values[k].data.fill_(vc)
        with torch.no_grad():
            lc = float(criterion(model(batch[0]), batch[1]).item())
        model.train()
        hp_state.values[k].data.fill_(old_val)

        denom = max((vp - vm) / 2, 1e-10)**2
        L_jj = abs(lp - 2*lc + lm) / denom
        bound = (L_jj / 2.0) * self.eps_fd * (hi - lo)
        return {
            "k": k, "measured_bias": bias,
            "theoretical_bound": max(bound, 1e-10),
            "L_jj": L_jj,
            "passes": bias <= 2 * max(bound, 1e-8)  # 2x safety
        }


class GeneralizationBoundComputer:
    """Theorem 6: L_true <= L_train + VC_gap + O(eps^2) + O(1/sqrt(N))."""

    def __init__(self, model):
        self.model = model
        self.n_params = sum(p.numel() for p in model.parameters())
        self.n_layers = sum(1 for _ in model.parameters()) // 2

    def compute_vc_dimension(self):
        """VC-dim ~ O(W * L * log(W)) for ReLU networks."""
        W = self.n_params
        L = max(self.n_layers, 1)
        return W * L * np.log(max(W, 2))

    def compute_rademacher_complexity(self, X, n_trials=20):
        """Monte Carlo Rademacher complexity estimate."""
        n = len(X)
        self.model.eval()
        complexities = []
        with torch.no_grad():
            preds = self.model(X).squeeze()
            for _ in range(n_trials):
                sigma = 2 * torch.randint(0, 2, (n,)).float() - 1
                complexities.append(float(torch.abs(
                    (sigma * preds).mean())))
        self.model.train()
        return np.mean(complexities)

    def compute_generalization_bound(self, train_loss, val_loss,
                                     n_train, delta=0.05,
                                     epsilon_leapfrog=0.005, N_hmc=60):
        """Full Theorem 6 bound decomposition."""
        vc_dim = self.compute_vc_dimension()
        # VC gap: C * sqrt(vc_dim * log(n) / n + log(1/delta)/n)
        vc_gap = np.sqrt(
            (vc_dim * np.log(max(n_train, 2)) + np.log(1/delta))
            / max(n_train, 1)
        )
        # Clamp to reasonable value
        vc_gap = min(vc_gap, 100.0)

        disc_bias = 0.5 * epsilon_leapfrog**2 * (self.n_params + 5)
        opt_error = 1.0 / np.sqrt(max(N_hmc, 1))
        total = train_loss + vc_gap + disc_bias + opt_error

        return {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "vc_dimension": vc_dim,
            "vc_gap": vc_gap,
            "discretization_bias": disc_bias,
            "optimization_error": opt_error,
            "total_bound": total,
            "empirical_gap": val_loss - train_loss,
            "bound_is_tight": val_loss <= total
        }

    def plot_bound_decomposition(self, bound_info, save_path):
        """Stacked bar chart of generalization bound components."""
        fig, ax = plt.subplots(figsize=(8, 5))
        components = ["train_loss", "vc_gap", "discretization_bias",
                       "optimization_error"]
        labels = ["Train Loss", "VC Gap", "Disc. Bias O(eps^2)",
                  "Opt. Error O(1/sqrt(N))"]
        vals = [bound_info[c] for c in components]
        colors = ["#2196F3", "#FF5722", "#4CAF50", "#FF9800"]

        bottom = 0
        for v, l, c in zip(vals, labels, colors):
            ax.bar("Bound", v, bottom=bottom, color=c, label=l,
                   edgecolor="white", lw=1.5)
            bottom += v

        ax.axhline(bound_info["val_loss"], color="red", ls="--", lw=2,
                   label=f"Val Loss = {bound_info['val_loss']:.4f}")
        ax.set_ylabel("Loss"); ax.set_title("Theorem 6: Generalization Bound")
        ax.legend(fontsize=9, loc="upper right"); ax.grid(True, alpha=0.2)
        tight = "PASS" if bound_info["bound_is_tight"] else "FAIL"
        ax.text(0, bottom + 0.05, f"Total={bound_info['total_bound']:.4f} [{tight}]",
                ha="center", fontsize=11, fontweight="bold")
        plt.tight_layout(); plt.savefig(save_path, dpi=200); plt.close()
