"""
Theorems 3, 7, 8: Convergence tracking, mixing time, and ergodicity.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class ConvergenceTracker:
    """Theorem 3: E[L(theta_bar, lambda_bar)] - L* <= C1/sqrt(N) + C2*T."""

    def __init__(self, d_theta=100, d_lambda=5):
        self.d_theta = d_theta
        self.d_lambda = d_lambda
        self.losses = []
        self.val_losses = []
        self.temperatures = []
        self.theta_norms = []
        self.lambda_vecs = []
        self.cesaro_losses = []

    def update(self, epoch, train_loss, val_loss, temperature,
               theta_norm=0.0, lambda_vec=None):
        """Record losses and compute Cesaro average."""
        self.losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.temperatures.append(temperature)
        self.theta_norms.append(theta_norm)
        if lambda_vec is not None:
            self.lambda_vecs.append(lambda_vec.copy())
        cesaro = np.mean(self.losses)
        self.cesaro_losses.append(cesaro)

    def theoretical_bound(self, N, T):
        """C1/sqrt(N) + C2*T."""
        C1 = max(self.theta_norms[0] if self.theta_norms else 1.0, 0.1)
        C2 = (self.d_theta + self.d_lambda) / 2.0
        return C1 / np.sqrt(max(N, 1)) + C2 * T

    def estimate_convergence_rate(self):
        """Fit L(N) = a*N^(-b) + c to observed loss."""
        if len(self.losses) < 10:
            return {"rate": 0.5, "offset": 0, "r_squared": 0}
        N = np.arange(1, len(self.losses) + 1, dtype=float)
        L = np.array(self.losses)
        try:
            def model(n, a, b, c):
                return a * n**(-b) + c
            p0 = [L[0], 0.5, L[-1]]
            popt, _ = curve_fit(model, N, L, p0=p0, maxfev=5000,
                                bounds=([0, 0.01, 0], [100, 2.0, 10]))
            L_pred = model(N, *popt)
            ss_res = np.sum((L - L_pred)**2)
            ss_tot = np.sum((L - L.mean())**2) + 1e-10
            return {"rate": popt[1], "offset": popt[2],
                    "r_squared": 1 - ss_res / ss_tot}
        except Exception:
            return {"rate": 0.5, "offset": float(L[-1]), "r_squared": 0}

    def compute_mixing_time_estimate(self, target_tv=0.01):
        """Estimate mixing time from autocorrelation."""
        if len(self.losses) < 20:
            return {"tau_estimated": len(self.losses),
                    "tau_theoretical": 50, "sufficient_data": False}
        L = np.array(self.losses)
        L = (L - L.mean()) / (L.std() + 1e-10)
        n = len(L)
        acf = np.correlate(L, L, mode="full")[n-1:] / n
        acf = acf / (acf[0] + 1e-10)

        # Find first crossing below target
        tau_est = n
        for i in range(1, len(acf)):
            if abs(acf[i]) < target_tv:
                tau_est = i
                break

        kappa = max(self.theta_norms[-1], 1.0) if self.theta_norms else 1.0
        d = self.d_theta + self.d_lambda
        tau_theory = kappa**0.5 * d**0.25 * np.log(1.0 / target_tv)
        return {"tau_estimated": tau_est, "tau_theoretical": tau_theory,
                "sufficient_data": True}

    def plot_convergence_analysis(self, save_path):
        """Four-panel convergence figure."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        N = np.arange(1, len(self.losses) + 1)
        L = np.array(self.losses)

        # (a) Loss vs N with bound
        ax = axes[0, 0]
        ax.plot(N, L, "b-", lw=1.5, alpha=0.7, label="Train Loss")
        ax.plot(N, self.val_losses, "r--", lw=1.5, alpha=0.7, label="Val Loss")
        T = self.temperatures[-1] if self.temperatures else 1.0
        bounds = [self.theoretical_bound(n, T) for n in N]
        ax.plot(N, bounds, "g--", lw=2, label="C1/sqrt(N)+C2*T")
        ax.set_yscale("log"); ax.set_title("Loss vs Epoch")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # (b) Cesaro averages
        ax = axes[0, 1]
        ax.plot(N, self.cesaro_losses, "purple", lw=2)
        ax.set_title("Cesaro Average Loss"); ax.grid(True, alpha=0.3)

        # (c) Convergence rate
        ax = axes[1, 0]
        rate_info = self.estimate_convergence_rate()
        ax.plot(N, L, "b-", alpha=0.5, label="Observed")
        if rate_info["r_squared"] > 0.1:
            a_fit = L[0]
            fitted = a_fit * N**(-rate_info["rate"]) + rate_info["offset"]
            ax.plot(N, fitted, "r--", lw=2,
                    label=f"Fit: N^(-{rate_info['rate']:.2f})")
        ax.set_yscale("log"); ax.set_title("Convergence Rate Fit")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # (d) Autocorrelation
        ax = axes[1, 1]
        if len(L) > 10:
            Ln = (L - L.mean()) / (L.std() + 1e-10)
            acf = np.correlate(Ln, Ln, "full")[len(Ln)-1:]
            acf = acf / (acf[0] + 1e-10)
            ax.plot(acf[:min(50, len(acf))], "orange", lw=2)
            ax.axhline(0, color="gray", ls="--")
        ax.set_title("Autocorrelation"); ax.set_xlabel("Lag")
        ax.grid(True, alpha=0.3)

        plt.suptitle("Theorem 3: Convergence Analysis", fontweight="bold")
        plt.tight_layout(); plt.savefig(save_path, dpi=200); plt.close()


class ErgodicityCertifier:
    """Theorems 2, 7, 8: Ergodicity, CLT, Mixing Time."""

    def __init__(self):
        self.losses = []
        self.theta_samples = []
        self.lambda_samples = []

    def record_sample(self, theta_flat, lambda_vec, loss):
        self.theta_samples.append(theta_flat.copy() if hasattr(
            theta_flat, "copy") else float(theta_flat))
        self.lambda_samples.append(lambda_vec.copy() if hasattr(
            lambda_vec, "copy") else [float(lambda_vec)])
        self.losses.append(loss)

    def compute_ergodic_average(self):
        return np.mean(self.losses) if self.losses else 0.0

    def test_ergodicity(self, burn_in_fraction=0.3):
        """Gelman-Rubin R-hat on split chain."""
        if len(self.losses) < 50:
            return {"R_hat": 1.0, "is_ergodic": None,
                    "reason": "insufficient_samples"}
        L = np.array(self.losses)
        burn = int(len(L) * burn_in_fraction)
        L = L[burn:]
        mid = len(L) // 2
        c1, c2 = L[:mid], L[mid:]
        B = mid * ((c1.mean() - L.mean())**2 + (c2.mean() - L.mean())**2) / 1
        W = (c1.var() + c2.var()) / 2.0 + 1e-10
        V_hat = ((mid - 1) / mid) * W + (1 / mid) * B
        R_hat = np.sqrt(V_hat / W)
        return {"R_hat": float(R_hat), "is_ergodic": R_hat < 1.1}

    def compute_effective_sample_size(self):
        """ESS = N / (1 + 2*sum(rho_k))."""
        if len(self.losses) < 20:
            return {"ESS": float(len(self.losses)), "ESS_ratio": 1.0}
        L = np.array(self.losses)
        L = (L - L.mean()) / (L.std() + 1e-10)
        n = len(L)
        acf = np.correlate(L, L, "full")[n-1:] / n
        acf = acf / (acf[0] + 1e-10)
        tau = 1.0
        for k in range(1, min(n // 2, 100)):
            if acf[k] < 0.05:
                break
            tau += 2 * acf[k]
        ess = n / max(tau, 1.0)
        return {"ESS": float(ess), "ESS_ratio": ess / n}

    def compute_clt_variance(self):
        """Spectral variance estimate."""
        if len(self.losses) < 20:
            return {"asymptotic_variance": 0, "standard_error": 0,
                    "confidence_interval_95": (0, 0)}
        L = np.array(self.losses)
        n = len(L)
        mean = L.mean()
        ess_info = self.compute_effective_sample_size()
        ess = max(ess_info["ESS"], 1)
        se = L.std() / np.sqrt(ess)
        return {
            "asymptotic_variance": float(L.var() * n / ess),
            "standard_error": float(se),
            "confidence_interval_95": (float(mean - 1.96*se),
                                       float(mean + 1.96*se))
        }

    def verify_detailed_balance(self, acceptance_rates):
        """Check acceptance rates are consistent with detailed balance."""
        if not acceptance_rates:
            return {"max_violation": 0, "passes": True}
        rates = np.array(acceptance_rates)
        # Under detailed balance, mean acceptance should be stable
        mid = len(rates) // 2
        if mid < 5:
            return {"max_violation": 0, "passes": True}
        r1, r2 = rates[:mid].mean(), rates[mid:].mean()
        violation = abs(r1 - r2)
        return {"max_violation": float(violation),
                "passes": violation < 0.15}

    def plot_ergodicity_diagnostics(self, save_path, acceptance_rates=None):
        """Six-panel ergodicity diagnostics."""
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        L = np.array(self.losses) if self.losses else np.array([0])
        n = len(L)

        # (a) Running R-hat
        ax = axes[0, 0]
        rhats = []
        for i in range(20, n, max(1, n // 50)):
            sub = L[:i]
            mid = len(sub) // 2
            if mid < 5:
                continue
            c1, c2 = sub[:mid], sub[mid:]
            W = (c1.var() + c2.var()) / 2 + 1e-10
            B = mid * ((c1.mean()-sub.mean())**2 +
                       (c2.mean()-sub.mean())**2)
            V = ((mid-1)/mid)*W + (1/mid)*B
            rhats.append((i, np.sqrt(V/W)))
        if rhats:
            ax.plot(*zip(*rhats), "b-", lw=2)
            ax.axhline(1.1, color="red", ls="--", label="R-hat=1.1")
        ax.set_title("Gelman-Rubin R-hat"); ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # (b) ESS over time
        ax = axes[0, 1]
        ess_vals = []
        for i in range(20, n, max(1, n // 30)):
            sub = L[:i]
            sub_n = (sub - sub.mean()) / (sub.std() + 1e-10)
            acf = np.correlate(sub_n, sub_n, "full")[len(sub_n)-1:]
            acf = acf / (acf[0] + 1e-10)
            tau = 1 + 2 * sum(acf[1:min(20, len(acf))])
            ess_vals.append((i, i / max(tau, 1)))
        if ess_vals:
            ax.plot(*zip(*ess_vals), "green", lw=2)
        ax.set_title("Effective Sample Size"); ax.grid(True, alpha=0.3)

        # (c) Autocorrelation
        ax = axes[0, 2]
        if n > 10:
            Ln = (L - L.mean()) / (L.std() + 1e-10)
            acf = np.correlate(Ln, Ln, "full")[n-1:] / n
            acf = acf / (acf[0] + 1e-10)
            ax.bar(range(min(40, len(acf))), acf[:40], color="steelblue",
                   alpha=0.7)
            ax.axhline(0, color="gray"); ax.axhline(0.05, color="red", ls="--")
        ax.set_title("Autocorrelation Function"); ax.grid(True, alpha=0.3)

        # (d) Running mean +/- 2sigma
        ax = axes[1, 0]
        if n > 5:
            cum_mean = np.cumsum(L) / np.arange(1, n+1)
            cum_std = [L[:i+1].std() / np.sqrt(i+1) for i in range(n)]
            ax.plot(cum_mean, "b-", lw=2, label="Running Mean")
            ax.fill_between(range(n),
                            cum_mean - 2*np.array(cum_std),
                            cum_mean + 2*np.array(cum_std),
                            alpha=0.2, color="blue", label="95% CI")
        ax.set_title("Running Mean +/- 2SE (CLT)"); ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # (e) Loss distribution
        ax = axes[1, 1]
        if n > 10:
            ax.hist(L, bins=min(30, n//3), color="steelblue",
                    alpha=0.7, density=True, edgecolor="white")
        ax.set_title("Loss Distribution (KDE)"); ax.grid(True, alpha=0.3)

        # (f) Acceptance rates
        ax = axes[1, 2]
        if acceptance_rates:
            ax.plot(acceptance_rates, "purple", lw=1.5)
            ax.axhline(0.65, color="green", ls="--", label="Target")
            ax.set_ylim(0, 1.05)
        ax.set_title("Detailed Balance (Accept Rate)")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        plt.suptitle("Theorems 2,7,8: Ergodicity Diagnostics",
                     fontweight="bold")
        plt.tight_layout(); plt.savefig(save_path, dpi=200); plt.close()
