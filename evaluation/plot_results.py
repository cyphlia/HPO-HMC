"""
Generate all publication-quality plots for HHD project.
  Figure 1: Testbed (Harmonic Oscillator + MNIST samples)
  Figure 2: Method A results
  Figure 3: Method B results
  Figure 4: Method C results
  Figure 5: Comparative overlay
"""
import json, os, sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from data_generator import generate_hamiltonian_data
from hamiltonian import HamiltonianNN

# ── Style ──
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11, "figure.dpi": 150,
    "savefig.bbox": "tight", "savefig.dpi": 200,
})
COLORS = {"A": "#2196F3", "B": "#FF5722", "C": "#4CAF50"}
hp_colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6"]
hp_labels = {"log_lr": "log(lr)", "dropout": "dropout", "n_layers": "layers",
             "n_neurons": "neurons", "log_batch_size": "log(bs)"}
HP_BOUNDS = config.HYPERPARAM_SPACE  # for normalization
OUT = "plots"
os.makedirs(OUT, exist_ok=True)

def plot_hp_normalized(ax, hp_dict, color_method=None):
    """Plot normalized HP trajectories (each scaled to [0,1] per its bounds)."""
    for i, (k, v) in enumerate(hp_dict.items()):
        if not v: continue
        arr = np.array(v)
        lo, hi = HP_BOUNDS[k]
        normed = (arr - lo) / (hi - lo + 1e-10)
        ax.plot(normed, color=hp_colors[i], lw=2, label=hp_labels.get(k, k), alpha=0.9)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Normalized Value [0, 1]")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Hyperparameter Evolution\n(normalized to search bounds)")
    ax.legend(fontsize=8, loc="upper right", ncol=2); ax.grid(True, alpha=0.2)

# ── Helpers ──
def load_json(p):
    with open(p) as f: return json.load(f)

def load_model_from_dir(d):
    state = torch.load(f"{d}/model.pt", map_location="cpu", weights_only=True)
    nn_ = state["input_layer.weight"].shape[0]
    nl_ = sum(1 for k in state if k.startswith("hidden.") and k.endswith(".0.weight"))
    m = HamiltonianNN(nl_, nn_, 0.0); m.load_state_dict(state); m.eval()
    return m

@torch.no_grad()
def predict_mesh(model, qm, pm):
    X = torch.from_numpy(np.stack([qm.flatten().astype(np.float32),
                                    pm.flatten().astype(np.float32)], 1))
    return model(X).numpy().reshape(qm.shape)

# ── Load everything ──
_, _, mesh = generate_hamiltonian_data(n_samples=500)
qm, pm, H_true = mesh

hA = load_json("results/harmonic_oscillator/method_a/history.json")
hB = load_json("results/harmonic_oscillator/method_b/history.json")
hC = load_json("results/harmonic_oscillator/method_c/history.json")
cnn = load_json("results/cnn/benchmark_results.json")

mA = load_model_from_dir("results/harmonic_oscillator/method_a")
mB = load_model_from_dir("results/harmonic_oscillator/method_b")
mC = load_model_from_dir("results/harmonic_oscillator/method_c")

H_A = predict_mesh(mA, qm, pm)
H_B = predict_mesh(mB, qm, pm)
H_C = predict_mesh(mC, qm, pm)

# ═══════════════════════════════════════════════════════════════
# FIGURE 0 — DATASET VISUALISATION
# ═══════════════════════════════════════════════════════════════
print("Plotting Figure 0: Dataset Visualisation...")
fig0 = plt.figure(figsize=(16, 5))
gs0 = GridSpec(1, 3, figure=fig0, width_ratios=[1.2, 1, 1])

# 0a: 3D energy surface
ax0_1 = fig0.add_subplot(gs0[0], projection="3d")
surf0 = ax0_1.plot_surface(qm, pm, H_true, cmap="coolwarm", linewidth=0, alpha=0.9, antialiased=True)
ax0_1.set_xlabel("q (position)"); ax0_1.set_ylabel("p (momentum)"); ax0_1.set_zlabel("H (energy)")
ax0_1.set_title("True Hamiltonian\n$H(q,p) = p^2/2m + \\frac{1}{2}kq^2$")
ax0_1.view_init(elev=25, azim=-50)
fig0.colorbar(surf0, ax=ax0_1, shrink=0.5, pad=0.08, label="Energy")

# 0b: Contour (2D view)
ax0_2 = fig0.add_subplot(gs0[1])
cf0 = ax0_2.contourf(qm, pm, H_true, levels=25, cmap="coolwarm")
ax0_2.set_xlabel("q (position)"); ax0_2.set_ylabel("p (momentum)")
ax0_2.set_title("Energy Contours")
ax0_2.set_aspect("equal")
fig0.colorbar(cf0, ax=ax0_2, label="H")

# 0c: 640 sampled training points coloured by H
ax0_3 = fig0.add_subplot(gs0[2])
train_loader_800, _, _ = generate_hamiltonian_data(n_samples=800, seed=101)
X_tr, H_tr = train_loader_800.dataset.tensors
sc = ax0_3.scatter(X_tr[:, 0].numpy(), X_tr[:, 1].numpy(), c=H_tr[:, 0].numpy(), cmap="coolwarm", edgecolor='none', alpha=0.8)
ax0_3.set_xlabel("q (position)"); ax0_3.set_ylabel("p (momentum)")
ax0_3.set_title("640 Training Points")
ax0_3.set_aspect("equal")
fig0.colorbar(sc, ax=ax0_3, label="H")

fig0.suptitle("Dataset Visualisation", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT}/0_dataset_visualisation.png")
plt.close()


# ═══════════════════════════════════════════════════════════════
# FIGURE 1 — TESTBED
# ═══════════════════════════════════════════════════════════════
print("Plotting Figure 1: Testbed...")
fig = plt.figure(figsize=(16, 6))
gs = GridSpec(1, 3, figure=fig, width_ratios=[1.2, 1, 1])

# 1a: 3D energy surface
ax1 = fig.add_subplot(gs[0], projection="3d")
surf = ax1.plot_surface(qm, pm, H_true, cmap="coolwarm", linewidth=0, alpha=0.9, antialiased=True)
ax1.set_xlabel("q (position)"); ax1.set_ylabel("p (momentum)"); ax1.set_zlabel("H (energy)")
ax1.set_title("True Hamiltonian\n$H(q,p) = p^2/2m + \\frac{1}{2}kq^2$")
ax1.view_init(elev=25, azim=-50)
fig.colorbar(surf, ax=ax1, shrink=0.5, pad=0.08, label="Energy")

# 1b: Contour (2D view)
ax2 = fig.add_subplot(gs[1])
cf = ax2.contourf(qm, pm, H_true, levels=25, cmap="coolwarm")
ax2.set_xlabel("q (position)"); ax2.set_ylabel("p (momentum)")
ax2.set_title("Energy Contours")
ax2.set_aspect("equal")
fig.colorbar(cf, ax=ax2, label="H")

# 1c: Cross-sections
ax3 = fig.add_subplot(gs[2])
mid = qm.shape[0] // 2
ax3.plot(qm[mid, :], H_true[mid, :], "b-", lw=2, label="H at p=0 (kinetic=0)")
ax3.plot(pm[:, mid], H_true[:, mid], "r--", lw=2, label="H at q=0 (potential=0)")
ax3.set_xlabel("Phase-space coordinate"); ax3.set_ylabel("Energy H")
ax3.set_title("Cross-Sections")
ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)

fig.suptitle("TESTBED: Simple Harmonic Oscillator", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT}/fig1_testbed.png")
plt.close()

# ═══════════════════════════════════════════════════════════════
# FIGURE 2 — METHOD A: Pure HHD
# ═══════════════════════════════════════════════════════════════
print("Plotting Figure 2: Method A...")
fig = plt.figure(figsize=(18, 10))
gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

# 2a: Loss curves
ax = fig.add_subplot(gs[0, 0])
ax.plot(hA["train_loss"], color=COLORS["A"], lw=1.5, alpha=0.7, label="Train")
ax.plot(hA["val_loss"], color=COLORS["A"], lw=2, ls="--", label="Val")
ax.set_yscale("log"); ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
ax.set_title("Loss Curves"); ax.legend(); ax.grid(True, alpha=0.2)

# 2b: HMC Acceptance Rate
ax = fig.add_subplot(gs[0, 1])
ax.plot(hA["acceptance_rate"], color="purple", lw=2)
ax.axhline(0.65, color="green", ls="--", alpha=0.5, label="Target 65%")
ax.fill_between(range(len(hA["acceptance_rate"])), hA["acceptance_rate"], alpha=0.15, color="purple")
ax.set_ylim(0, 1.05); ax.set_xlabel("Epoch"); ax.set_ylabel("Rate")
ax.set_title("HMC Acceptance Rate"); ax.legend(fontsize=9); ax.grid(True, alpha=0.2)

# 2c: Energy Landscape
ax = fig.add_subplot(gs[0, 2], projection="3d")
ax.plot_surface(qm, pm, H_A, cmap="plasma", linewidth=0, alpha=0.85)
res_A = np.abs(H_A - H_true)
ax.set_title(f"Predicted H(q,p)\nMAE={res_A.mean():.4f}  R²={1-((H_A-H_true)**2).sum()/((H_true-H_true.mean())**2).sum():.4f}")
ax.set_xlabel("q"); ax.set_ylabel("p"); ax.set_zlabel("H")
ax.view_init(elev=25, azim=-50)

# 2d: HP Trajectories (normalized)
ax = fig.add_subplot(gs[1, 0])
plot_hp_normalized(ax, hA["hyperparams"])

# 2e: Residual heatmap
ax = fig.add_subplot(gs[1, 1])
im = ax.imshow(res_A, extent=[-4,4,-4,4], origin="lower", cmap="Reds", aspect="equal")
ax.set_xlabel("q"); ax.set_ylabel("p"); ax.set_title(f"Absolute Error\nMax={res_A.max():.4f}")
fig.colorbar(im, ax=ax, label="|Error|")

# 2f: Phase-space trajectory (log_lr vs dropout)
ax = fig.add_subplot(gs[1, 2])
xs = np.array(hA["hyperparams"]["log_lr"])
ys = np.array(hA["hyperparams"]["dropout"])
n = min(len(xs), len(ys))
cols = plt.cm.viridis(np.linspace(0, 1, n))
for j in range(n-1): ax.plot(xs[j:j+2], ys[j:j+2], color=cols[j], lw=1.5)
ax.scatter(xs[0], ys[0], s=120, c="green", zorder=5, marker="o", label="Start")
ax.scatter(xs[-1], ys[-1], s=120, c="red", zorder=5, marker="*", label="End")
ax.set_xlabel("log_lr"); ax.set_ylabel("dropout")
ax.set_title("HP Phase-Space Trajectory"); ax.legend(fontsize=9); ax.grid(True, alpha=0.2)

fig.suptitle("METHOD A: Pure Hamiltonian Hyperparameter Dynamics", fontsize=15, fontweight="bold", color=COLORS["A"])
plt.savefig(f"{OUT}/fig2_method_a.png")
plt.close()

# ═══════════════════════════════════════════════════════════════
# FIGURE 3 — METHOD B: Hybrid BO
# ═══════════════════════════════════════════════════════════════
print("Plotting Figure 3: Method B...")
fig = plt.figure(figsize=(16, 5.5))
gs = GridSpec(1, 3, figure=fig, wspace=0.35)

# 3a: BO Trial results
ax = fig.add_subplot(gs[0])
trials = hB["val_loss_per_trial"]
best = hB["best_val_loss"]
ax.bar(range(len(trials)), trials, color=COLORS["B"], alpha=0.5, label="Per-trial loss")
ax.plot(best, color=COLORS["B"], lw=2.5, marker="o", ms=6, label="Best so far")
ax.set_xlabel("BO Trial"); ax.set_ylabel("Validation Loss")
ax.set_title("Bayesian Optimization Trials"); ax.legend(); ax.grid(True, alpha=0.2)

# 3b: Energy landscape
ax = fig.add_subplot(gs[1], projection="3d")
ax.plot_surface(qm, pm, H_B, cmap="plasma", linewidth=0, alpha=0.85)
res_B = np.abs(H_B - H_true)
ax.set_title(f"Predicted H(q,p)\nMAE={res_B.mean():.4f}  R²={1-((H_B-H_true)**2).sum()/((H_true-H_true.mean())**2).sum():.4f}")
ax.set_xlabel("q"); ax.set_ylabel("p"); ax.set_zlabel("H")
ax.view_init(elev=25, azim=-50)

# 3c: Residual heatmap
ax = fig.add_subplot(gs[2])
im = ax.imshow(res_B, extent=[-4,4,-4,4], origin="lower", cmap="Reds", aspect="equal")
ax.set_xlabel("q"); ax.set_ylabel("p"); ax.set_title(f"Absolute Error\nMax={res_B.max():.4f}")
fig.colorbar(im, ax=ax, label="|Error|")

fig.suptitle("METHOD B: Hybrid Adam + L-BFGS with Bayesian Optimization", fontsize=15, fontweight="bold", color=COLORS["B"])
plt.savefig(f"{OUT}/fig3_method_b.png")
plt.close()

# ═══════════════════════════════════════════════════════════════
# FIGURE 4 — METHOD C: Unified HHD-ABBO
# ═══════════════════════════════════════════════════════════════
print("Plotting Figure 4: Method C...")
fig = plt.figure(figsize=(18, 10))
gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

# 4a: Loss curves
ax = fig.add_subplot(gs[0, 0])
ax.plot(hC["train_loss"], color=COLORS["C"], lw=1.5, alpha=0.7, label="Train")
ax.plot(hC["val_loss"], color=COLORS["C"], lw=2, ls="--", label="Val")
ax.set_yscale("log"); ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
ax.set_title("Loss Curves"); ax.legend(); ax.grid(True, alpha=0.2)

# 4b: Acceptance + Adaptive Step
ax = fig.add_subplot(gs[0, 1])
ax2_ = ax.twinx()
ax.plot(hC["acceptance_rate"], color="purple", lw=2, label="Accept Rate")
ax.axhline(0.65, color="green", ls="--", alpha=0.4, label="Target 65%")
ax2_.plot(hC["step_size"], color="orange", lw=1.5, alpha=0.7, label="Step Size")
ax.set_ylim(0, 1.05); ax.set_xlabel("Epoch"); ax.set_ylabel("Acceptance Rate")
ax2_.set_ylabel("Step Size (eps)", color="orange")
ax.set_title("HMC Acceptance & Adaptive Step"); ax.legend(loc="upper left", fontsize=8); ax2_.legend(loc="upper right", fontsize=8)
ax.grid(True, alpha=0.2)

# 4c: Energy Landscape
ax = fig.add_subplot(gs[0, 2], projection="3d")
ax.plot_surface(qm, pm, H_C, cmap="plasma", linewidth=0, alpha=0.85)
res_C = np.abs(H_C - H_true)
ax.set_title(f"Predicted H(q,p)\nMAE={res_C.mean():.4f}  R²={1-((H_C-H_true)**2).sum()/((H_true-H_true.mean())**2).sum():.4f}")
ax.set_xlabel("q"); ax.set_ylabel("p"); ax.set_zlabel("H")
ax.view_init(elev=25, azim=-50)

# 4d: HP Trajectories (normalized)
ax = fig.add_subplot(gs[1, 0])
plot_hp_normalized(ax, hC["hyperparams"])

# 4e: Residual heatmap
ax = fig.add_subplot(gs[1, 1])
im = ax.imshow(res_C, extent=[-4,4,-4,4], origin="lower", cmap="Reds", aspect="equal")
ax.set_xlabel("q"); ax.set_ylabel("p"); ax.set_title(f"Absolute Error\nMax={res_C.max():.4f}")
fig.colorbar(im, ax=ax, label="|Error|")

# 4f: Phase-space trajectory
ax = fig.add_subplot(gs[1, 2])
xs = np.array(hC["hyperparams"]["log_lr"])
ys = np.array(hC["hyperparams"]["dropout"])
n = min(len(xs), len(ys))
cols = plt.cm.viridis(np.linspace(0, 1, n))
for j in range(n-1): ax.plot(xs[j:j+2], ys[j:j+2], color=cols[j], lw=1.5)
ax.scatter(xs[0], ys[0], s=120, c="green", zorder=5, marker="o", label="Start")
ax.scatter(xs[-1], ys[-1], s=120, c="red", zorder=5, marker="*", label="End")
ax.set_xlabel("log_lr"); ax.set_ylabel("dropout")
ax.set_title("HP Phase-Space Trajectory"); ax.legend(fontsize=9); ax.grid(True, alpha=0.2)

fig.suptitle("METHOD C: Unified HHD-ABBO (Novel Three-Phase Curriculum)", fontsize=15, fontweight="bold", color=COLORS["C"])
plt.savefig(f"{OUT}/fig4_method_c.png")
plt.close()

# ═══════════════════════════════════════════════════════════════
# FIGURE 5 — COMPARATIVE OVERVIEW
# ═══════════════════════════════════════════════════════════════
print("Plotting Figure 5: Comparative...")
fig = plt.figure(figsize=(18, 10))
gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# 5a: Val loss comparison (A vs C, different x-axis for B)
ax = fig.add_subplot(gs[0, 0])
ax.plot(hA["val_loss"], color=COLORS["A"], lw=2, label="A: Pure HHD", alpha=0.85)
ax.plot(hC["val_loss"], color=COLORS["C"], lw=2, label="C: Unified", alpha=0.85)
# B has different x scale - plot as scatter
bx = np.linspace(0, max(len(hA["val_loss"]), len(hC["val_loss"]))-1, len(hB["best_val_loss"]))
ax.plot(bx, hB["best_val_loss"], color=COLORS["B"], lw=2, ls="--", marker="s", ms=5, label="B: Hybrid BO (best)")
ax.set_yscale("log"); ax.set_xlabel("Epoch / Trial (scaled)"); ax.set_ylabel("Val Loss")
ax.set_title("Convergence Comparison"); ax.legend(fontsize=9); ax.grid(True, alpha=0.2)

# 5b: Energy landscapes side-by-side (2D contours)
for i, (tag, Hp, c) in enumerate([("A: Pure HHD", H_A, COLORS["A"]),
                                     ("B: Hybrid BO", H_B, COLORS["B"]),
                                     ("C: Unified", H_C, COLORS["C"])]):
    ax = fig.add_subplot(gs[0, 1]) if i == 0 else fig.add_subplot(gs[0, 1])
    break  # handled below

# 5b: Reconstruction quality bar chart
ax = fig.add_subplot(gs[0, 1])
metrics = {
    "A: Pure HHD": {"MAE": res_A.mean(), "RMSE": np.sqrt(((H_A-H_true)**2).mean())},
    "B: Hybrid BO": {"MAE": res_B.mean(), "RMSE": np.sqrt(((H_B-H_true)**2).mean())},
    "C: Unified": {"MAE": res_C.mean(), "RMSE": np.sqrt(((H_C-H_true)**2).mean())},
}
x = np.arange(3); w = 0.35
mae_vals = [metrics[k]["MAE"] for k in metrics]
rmse_vals = [metrics[k]["RMSE"] for k in metrics]
ax.bar(x - w/2, mae_vals, w, color=[COLORS["A"], COLORS["B"], COLORS["C"]], alpha=0.7, label="MAE")
ax.bar(x + w/2, rmse_vals, w, color=[COLORS["A"], COLORS["B"], COLORS["C"]], alpha=0.4, label="RMSE", hatch="//")
ax.set_xticks(x); ax.set_xticklabels(["A", "B", "C"]); ax.set_ylabel("Error")
ax.set_title("Reconstruction Quality"); ax.legend(); ax.grid(True, alpha=0.2, axis="y")
for xi, (ma, rm) in enumerate(zip(mae_vals, rmse_vals)):
    ax.text(xi-w/2, ma+0.02, f"{ma:.3f}", ha="center", fontsize=8, fontweight="bold")

# 5c: CNN Accuracy comparison
ax = fig.add_subplot(gs[0, 2])
cnn_names = list(cnn.keys())
cnn_short = ["A", "B", "C"]
accs = [cnn[k]["best_val_acc"]*100 for k in cnn_names]
times = [cnn[k]["time"] for k in cnn_names]
bars = ax.bar(cnn_short, accs, color=[COLORS["A"], COLORS["B"], COLORS["C"]], alpha=0.8, edgecolor="white", lw=1.5)
ax.set_ylabel("Best Accuracy (%)"); ax.set_title("CNN MNIST Benchmark")
ax.set_ylim(96.5, 98.5)
for b, a, t in zip(bars, accs, times):
    ax.text(b.get_x()+b.get_width()/2, a+0.05, f"{a:.1f}%\n({t:.0f}s)", ha="center", fontsize=9, fontweight="bold")
ax.grid(True, alpha=0.2, axis="y")

# 5d: CNN accuracy over epochs
ax = fig.add_subplot(gs[1, 0])
for key, c, label in [("Method A (Pure HHD)", COLORS["A"], "A: Pure HHD"),
                        ("Method C (Unified HHD-ABBO)", COLORS["C"], "C: Unified")]:
    ah = cnn[key]["acc_history"]
    ax.plot([x*100 for x in ah], color=c, lw=2, marker="o", ms=4, label=label)
ah_b = cnn["Method B (Hybrid BO)"]["acc_history"]
ax.plot([x*100 for x in ah_b], color=COLORS["B"], lw=2, marker="s", ms=4, ls="--", label="B: Hybrid BO")
ax.set_xlabel("Epoch / Trial"); ax.set_ylabel("Accuracy (%)"); ax.set_title("CNN Accuracy Progression")
ax.legend(fontsize=9); ax.grid(True, alpha=0.2)

# 5e: Acceptance rate comparison A vs C
ax = fig.add_subplot(gs[1, 1])
ax.plot(hA["acceptance_rate"], color=COLORS["A"], lw=2, label="A: Pure HHD")
ax.plot(hC["acceptance_rate"], color=COLORS["C"], lw=2, label="C: Unified")
ax.axhline(0.65, color="gray", ls="--", alpha=0.4, label="Target 65%")
ax.fill_between(range(len(hA["acceptance_rate"])), hA["acceptance_rate"], alpha=0.1, color=COLORS["A"])
ax.fill_between(range(len(hC["acceptance_rate"])), hC["acceptance_rate"], alpha=0.1, color=COLORS["C"])
ax.set_ylim(0, 1.05); ax.set_xlabel("Epoch"); ax.set_ylabel("Rate")
ax.set_title("HMC Acceptance Rate Comparison"); ax.legend(fontsize=9); ax.grid(True, alpha=0.2)

# 5f: R² comparison
ax = fig.add_subplot(gs[1, 2])
r2_vals = [1-((H_A-H_true)**2).sum()/((H_true-H_true.mean())**2).sum(),
           1-((H_B-H_true)**2).sum()/((H_true-H_true.mean())**2).sum(),
           1-((H_C-H_true)**2).sum()/((H_true-H_true.mean())**2).sum()]
bars = ax.barh(["A: Pure HHD", "B: Hybrid BO", "C: Unified"], r2_vals,
               color=[COLORS["A"], COLORS["B"], COLORS["C"]], alpha=0.8, edgecolor="white", lw=1.5)
ax.set_xlabel("R² Score"); ax.set_title("Landscape Reconstruction R²")
ax.set_xlim(0, 1.05)
for b, v in zip(bars, r2_vals):
    ax.text(v + 0.01, b.get_y()+b.get_height()/2, f"{v:.4f}", va="center", fontsize=10, fontweight="bold")
ax.grid(True, alpha=0.2, axis="x")

fig.suptitle("COMPARATIVE ANALYSIS: Method A vs B vs C", fontsize=16, fontweight="bold")
plt.savefig(f"{OUT}/fig5_comparative.png")
plt.close()

print(f"\nAll plots saved to {OUT}/")
for f in sorted(os.listdir(OUT)):
    print(f"  {f}")
