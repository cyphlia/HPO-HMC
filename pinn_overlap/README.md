# Hamiltonian Hyperparameter Dynamics for Physics-Informed Neural Networks

> **What is this folder?**  
> This folder explores how the Hamiltonian Hyperparameter Dynamics (HHD) algorithm — originally developed for general neural network HPO — can be applied to **Physics-Informed Neural Networks (PINNs)**, a class of models that solve partial differential equations (PDEs) using neural networks.  
>  
> If you're unfamiliar with PINNs, read on — this document teaches you PINNs from scratch, then shows exactly *why* and *how* HHD slots in.

---

## Part 1: What Are PINNs?

### 1.1 The Problem: Solving PDEs with Neural Networks

A **partial differential equation (PDE)** describes how a quantity (temperature, velocity, pressure, etc.) changes over space and time. For example, the 1D **heat equation**:

$$\frac{\partial u}{\partial t} = \nu \frac{\partial^2 u}{\partial x^2}$$

says "the rate at which temperature $u(x,t)$ changes at a point is proportional to how curved the temperature profile is at that point." Traditional numerical methods (finite differences, finite elements) discretize the domain into a mesh and solve the equation at mesh nodes. This works well but becomes expensive in high dimensions, irregular geometries, or when you want to solve *inverse* problems (estimating unknown parameters from data).

**Physics-Informed Neural Networks (PINNs)**, introduced by Raissi, Perdikaris & Karniadakis (2019), take a radically different approach: train a neural network $u_\theta(x, t)$ to approximate the solution, and use the PDE itself as a *loss function*.

### 1.2 How PINNs Work

A PINN is a standard feed-forward neural network that takes spatial-temporal coordinates $(x, t)$ as input and outputs the field value $u(x, t)$. The key insight is that because neural networks are differentiable, you can use **automatic differentiation** to compute $\partial u/\partial t$, $\partial^2 u/\partial x^2$, etc. — the exact derivatives that appear in the PDE.

The training loss has **multiple components**:

```
Total Loss = w_r · L_residual  +  w_b · L_boundary  +  w_i · L_initial  +  w_d · L_data
```

Where:

| Loss Component | What It Measures | How It's Computed |
|:---|:---|:---|
| **$\mathcal{L}_{\text{residual}}$ (PDE residual)** | How well the network satisfies the PDE at interior collocation points | Plug $u_\theta$ into the PDE via autograd; the result should be ≈ 0 |
| **$\mathcal{L}_{\text{boundary}}$ (boundary conditions)** | How well the network matches known values at domain boundaries | $\|u_\theta(x_{\text{bc}}) - u_{\text{exact}}(x_{\text{bc}})\|^2$ |
| **$\mathcal{L}_{\text{initial}}$ (initial conditions)** | How well the network matches the known state at $t=0$ | $\|u_\theta(x, 0) - u_0(x)\|^2$ |
| **$\mathcal{L}_{\text{data}}$** | (Optional) fit to sparse observed data | $\|u_\theta(x_d) - u_d\|^2$ |

**Collocation points** are randomly sampled coordinates in the interior of the domain where you evaluate the PDE residual. Unlike finite elements, there's no mesh — just scattered points.

### 1.3 A Concrete Example: The 1D Heat Equation

```
PDE:     u_t = ν · u_xx        on x ∈ [0, 1],  t ∈ [0, 1]
BC:      u(0, t) = u(1, t) = 0
IC:      u(x, 0) = sin(πx)
Exact:   u(x, t) = exp(-ν·π²·t) · sin(πx)
```

To solve this with a PINN:
1. Build a network: $u_\theta : (x, t) \mapsto \hat{u}$
2. Sample 1000 random $(x_i, t_i)$ inside $[0,1]^2$ (collocation points)
3. For each point, use autograd to compute $\hat{u}_t$ and $\hat{u}_{xx}$
4. PDE residual at that point: $r_i = \hat{u}_t - \nu \hat{u}_{xx}$
5. $\mathcal{L}_r = \frac{1}{N}\sum r_i^2$ (this should → 0 if the PDE is satisfied)
6. Similarly compute $\mathcal{L}_b$ at boundary points and $\mathcal{L}_i$ at $t=0$ points
7. Minimize $w_r \mathcal{L}_r + w_b \mathcal{L}_b + w_i \mathcal{L}_i$ with Adam (then optionally L-BFGS)

---

## Part 2: The PINN Hyperparameter Problem

### 2.1 Why PINNs Are Hard to Train

PINNs look elegant on paper but are **notoriously difficult to train** in practice. The core issue: the different loss components compete with each other.

**The Loss Balancing Problem:**
- $\mathcal{L}_r$ (PDE residual) involves second derivatives of the network — its gradients tend to be large and noisy
- $\mathcal{L}_b$ (boundary) is a simple pointwise MSE — its gradients are clean and small
- $\mathcal{L}_i$ (initial condition) sits somewhere in between

When you add these together, $\mathcal{L}_r$ typically **dominates** the gradient signal, drowning out the boundary and initial conditions. The network learns to approximately satisfy the PDE interior but violates the boundary conditions — producing garbage solutions.

### 2.2 What Hyperparameters Control PINN Training?

| Hyperparameter | Why It Matters | Typical Range |
|:---|:---|:---|
| **$w_r, w_b, w_i$** (loss weights) | Balance PDE residual vs. boundary vs. initial condition enforcement | $[0.01, 100]$ each |
| **Learning rate** | Controls optimization step size; too large → diverge, too small → stagnate | $[10^{-5}, 10^{-2}]$ |
| **Network depth** | Deeper = more expressive, but harder to train | $[2, 8]$ layers |
| **Network width** | Wider = better function approximation | $[20, 256]$ neurons |
| **Activation function** | `tanh` is standard; `sin` (Fourier features) helps for oscillatory solutions | Categorical |
| **Number of collocation points** | More points = better PDE coverage, but slower training | $[500, 10000]$ |

### 2.3 Current Solutions Are Ad-Hoc

The research community has proposed several **heuristic** methods to adaptively balance loss weights:

- **SoftAdapt**: Adjusts weights based on relative rates of loss decrease
- **ReLoBRaLo**: Uses random lookbacks to compute relative loss changes  
- **NTK weighting**: Uses Neural Tangent Kernel eigenvalues to equalize convergence rates
- **Inverse Dirichlet**: Weights inversely proportional to gradient magnitudes
- **Manual tuning**: The most common approach — trial and error

**Every single one of these is a hand-designed heuristic with no formal dynamical-systems grounding.** They have no notion of momentum, inertia, or energy conservation. They can oscillate wildly, overshoot, or get stuck.

---

## Part 3: Why HHD Is a Natural Fit for PINNs

### 3.1 The Core Insight

Your HHD algorithm treats hyperparameters as **dynamical variables** in a phase space with positions and momenta, evolving under Hamilton's equations via symplectic integration. This is *exactly* what PINN loss weights need:

| HHD Concept | PINN Application |
|:---|:---|
| Hyperparameter position $\lambda$ | Loss weights $(\log w_r, \log w_b, \log w_i)$ and learning rate $\log \text{lr}$ |
| Hyperparameter momentum $p_\lambda$ | Rate of change of loss weights — provides **inertia** that prevents oscillatory instability |
| Augmented Hamiltonian $H = T + V$ | $T = \frac{p_\theta^2}{2m_\theta} + \frac{p_w^2}{2m_w}$, $V = w_r\mathcal{L}_r + w_b\mathcal{L}_b + w_i\mathcal{L}_i$ |
| Symplectic leapfrog | Structure-preserving evolution of weights — bounded energy error prevents weight blow-up |
| Metropolis-Hastings accept/reject | Rejects weight proposals that *worsen* the total PINN loss |
| Adam warmup → HMC handoff | Maps to standard PINN practice of Adam warmup → fine-tuning, but now with co-evolving weights |
| Reflection boundary conditions | Keeps $\log w \in [\log w_{\min}, \log w_{\max}]$ — no weight explosion |

### 3.2 What HHD Does That Heuristics Can't

1. **Momentum and inertia**: When a weight is moving in a good direction, HHD's momentum keeps it going. SoftAdapt/ReLoBRaLo have no memory between steps.

2. **Energy conservation**: The symplectic integrator ensures $H' \approx H$ over arbitrarily long trajectories. Loss weights can't diverge — they're bounded by energy conservation. Heuristic methods have no such guarantee.

3. **Formal accept/reject**: If a proposed weight change makes things worse, HMC's Metropolis step rejects it and reverts. Heuristics blindly apply their update rule regardless.

4. **Joint evolution**: Network weights $\theta$ and loss weights $w$ evolve *simultaneously* in a single dynamical system, capturing their interactions. Heuristics treat weight adjustment as a separate, decoupled process.

5. **No additional hyperparameters**: Ironically, methods like SoftAdapt introduce *their own* hyperparameters (lookback window, smoothing factor). HHD's "hyperparameters" are physical: mass, step size, temperature — with clear physical interpretations.

### 3.3 The Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                    HHD-PINN Training Pipeline                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Phase 1: Adam Warmup (N_warmup epochs)                         │
│  ├── Fixed loss weights (w_r=1, w_b=1, w_i=1)                  │
│  ├── Standard Adam on θ only                                     │
│  └── Purpose: get θ to a reasonable basin                        │
│                                                                  │
│  Phase 2: HMC Co-Evolution (N_hmc epochs)                       │
│  ├── Sample momenta: p_θ ~ N(0, m_θ), p_w ~ N(0, m_w)         │
│  ├── Leapfrog L steps:                                           │
│  │   ├── Half-step momenta (θ gradients via backprop)            │
│  │   ├── Full-step positions (θ += εp_θ/m_θ, w += εp_w/m_w)    │
│  │   ├── Half-step momenta (w gradients via finite-diff)         │
│  │   └── Gradient clipping + NaN guards                          │
│  ├── Metropolis accept/reject on ΔH                              │
│  └── Best-validation checkpointing                               │
│                                                                  │
│  Output: Trained PINN θ* + optimal loss weights w*               │
└──────────────────────────────────────────────────────────────────┘
```

---

## Part 4: What's In This Folder

### Source Code (`src/`)

| File | Description |
|:---|:---|
| `pinn_model.py` | PINN neural network with `tanh` activation and autograd-based derivative computation |
| `pinn_loss.py` | Multi-component PINN loss ($\mathcal{L}_r + \mathcal{L}_b + \mathcal{L}_i$) with configurable weights |
| `pde_problems.py` | Three canonical PDE benchmarks: 1D Heat, 1D Burgers', 2D Poisson |
| `hhd_pinn_trainer.py` | HHD-driven PINN trainer adapting Method A for PINN-specific loss weights |

### Notebooks (`notebooks/`)

| Notebook | Description |
|:---|:---|
| `01_hhd_pinn_demo.ipynb` | Step-by-step educational notebook: teaches PINNs, runs baseline, runs HHD-PINN, compares results |

### How to Run

```bash
cd pinn_overlap
# Quick import test
python -c "from src.pde_problems import HeatEquation1D; print('OK')"

# Or open the notebook in Jupyter/Colab
jupyter notebook notebooks/01_hhd_pinn_demo.ipynb
```

---

## References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks. *Journal of Computational Physics*, 378, 686-707.
2. Wang, S., Teng, Y., & Perdikaris, P. (2021). Understanding and mitigating gradient flow pathologies in physics-informed neural networks. *SIAM Journal on Scientific Computing*.
3. Duruisseaux, V., Schmitt, J., & Leok, M. (2021). Adaptive Hamiltonian variational integrators and applications to symplectic accelerated optimization. *SIAM J. Sci. Comput.*
4. McClenny, L. D. & Braga-Neto, U. M. (2023). Self-adaptive physics-informed neural networks. *Journal of Computational Physics*.
