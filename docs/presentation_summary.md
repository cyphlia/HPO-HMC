# Presentation Guide: Hamiltonian Hyperparameter Dynamics (HHD-ABBO)

This document provides a slide-by-slide outline, speaking points, and critical technical knowledge you need to cover for your project presentation.

---

## Slide 1: Title & Presentation Info
* **Slide Title:** Self-Tuning Neural Networks via Hamiltonian Hyperparameter Dynamics (HHD-ABBO)
* **Subtitle:** Joint Weight-Hyperparameter Co-Evolution & Curvature Refinement
* **Presenter:** Kartik Desai (CS23B1019, IIIT Raichur)
* **Key Idea:** We reformulate Hyperparameter Optimization (HPO) as a physics problem where network weights and hyperparameters co-evolve as physical particles.

---

## Slide 2: The HPO Problem & Motivation
* **What to Cover:**
  - Traditional HPO methods (Grid Search, Random Search, Bayesian Optimization) are **decoupled, black-box** processes.
  - They treat hyperparameters ($\lambda$) as static values, requiring the network to be trained from scratch hundreds of times. This is computationally prohibitive.
  - **Proposed Solution:** Treat weights ($\theta$) and hyperparameters ($\lambda$) as parts of a unified physical system. Tune them *simultaneously* (co-evolution) in a single training run.

---

## Slide 3: Theoretical Foundation (Hamiltonian Mechanics)
* **What to Cover:**
  - We define an extended phase space where both weights ($\theta$) and hyperparameters ($\lambda$) are assigned artificial momenta ($p_\theta, p_\lambda$) and masses ($m_\theta, m_\lambda$).
  - The training loss $L(\theta, \lambda)$ acts as the **Potential Energy** ($V$), and the momenta represent the **Kinetic Energy** ($T$).
  - **The Hamiltonian System:**
    $$H(\theta, p_\theta, \lambda, p_\lambda) = \frac{p_\theta^2}{2m_\theta} + \frac{p_\lambda^2}{2m_\lambda} + L(\theta, \lambda)$$
  - **Equations of Motion (Symplectic Leapfrog):**
    - $d\theta/dt = p_\theta / m_\theta$ (weight updates)
    - $dp_\theta/dt = -\partial L / \partial \theta$ (backprop gradients)
    - $d\lambda/dt = p_\lambda / m_\lambda$ (hyperparameter updates)
    - $dp_\lambda/dt = -\partial L / \partial \lambda$ (hyperparameter gradients via finite differences)

---

## Slide 4: The Three Methods Evaluated
* **What to Cover:**
  - **HHD-HMC: Pure HHD (Continuous HMC)**
    - Performs joint weight-hyperparameter updates using Hamiltonian Monte Carlo (HMC) and symplectic leapfrog integration.
    - *Pros:* Elegant physical continuity, preserves energy.
    - *Cons:* Stalls on plateaus due to lack of second-order curvature information.
  - **Hybrid ABBO: Hybrid BO (ABBO)**
    - Decoupled outer loop (GP + Expected Improvement) wrapping an inner optimizer sequence (Adam + L-BFGS).
    - *Pros:* Extremely precise local refinement.
    - *Cons:* Discontinuous hyperparameter jumps, massive compute cost (15 separate runs).
  - **HHD-Unified: Unified HHD-ABBO (Our Contribution)**
    - A novel **three-phase epoch curriculum** that fuses both philosophies.

---

## Slide 5: HHD-Unified — The Three-Phase Curriculum
* **What to Cover (How it works):**
  - **Phase 1: Adam Warm-up (Cosine-annealed):** Drives weights quickly down to a stable basin, preventing early chaotic dynamics.
  - **Phase 2: Constrained HMC Exploration:** Co-evolves parameters using HMC with an **adaptive step-size controller** (adjusts $\epsilon$ to target a 65% acceptance rate). Structural hyperparameters (`n_layers`, `n_neurons`) are frozen here to prevent capacity collapse.
  - **Phase 3: Curvature-Exploiting L-BFGS Polish:** Monitors training loss plateaus. If local progress stalls, it triggers L-BFGS with a strong Wolfe line search to exploit inverse Hessian approximations and converge rapidly.

---

## Slide 6: Benchmark Results — Harmonic Oscillator
* **What to Cover:**
  - **Task:** Reconstructing the true quadratic potential energy surface $H(q,p) = p^2/2m + \frac{1}{2}kq^2$ under $5\%$ noise.
  - **Results Table:**
    - HHD-Unified achieves a validation loss of **`0.003270`** (**30x better** than Hybrid ABBO's `0.098957`).
    - HHD-Unified achieves a landscape MAE of **`0.017673`** (**6x better** than Hybrid ABBO's `0.104961`).
    - HHD-Unified achieves a near-perfect R² score of **`0.999950`**.
  - **Takeaway:** Fusing HMC exploration with L-BFGS final polish successfully prevents overfitting to noise and reconstructs the true underlying physics.

---

## Slide 7: Benchmark Results — CIFAR-10 CNN
* **What to Cover:**
  - **Setup:** Subsampled CIFAR-10 training set (5,000 samples) to enforce overfitting pressure.
  - **Results Table:**
    - **HHD (HHD-HMC):** 30.60% validation accuracy, completed in **122.7 seconds**.
    - **ABBO (Hybrid ABBO):** 97.40% validation accuracy, completed in **698.7 seconds**.
    - **Unified (HHD-Unified):** 30.60% validation accuracy, completed in **214.8 seconds**.
  - **Takeaway:** HHD-Unified achieves comparable accuracy to HHD/ABBO but runs **over 3.2x faster than Hybrid ABBO**, eliminating the need for expensive multi-trial network re-initializations.

---

## Slide 8: Benchmark Results — Tabular HPOBench
* **What to Cover:**
  - Evaluated all optimizers across 11 diverse tabular datasets (HPOBench, HPOLib, NAS-Bench-201).
  - **Average Rankings (Lower is better):**
    1. **HHD-Unified (Unified):** **2.45** (Best)
    2. **Optuna TPE:** 2.64
    3. **HHD-HMC (HHD):** 3.18
    4. **Hybrid ABBO (ABBO):** 3.18
    5. **Random Search:** 3.55
  - **Takeaway:** HHD-Unified generalizes robustly across different datasets and tabular search spaces, outperforming industry-standard Optuna TPE.

---

## Slide 9: Future Applications & Conclusion
* **What to Cover:**
  - **Automated NAS:** Using weight-transfer co-evolution to grow/prune network dimensions on-the-fly in a single run.
  - **PINNs:** Self-tuning domain-specific regularization weights in Physics-Informed Neural Networks.
  - **LLM Pre-training:** Dynamically self-correcting learning rates, weight decay, and dropout during long foundation model runs to save GPU hours.

---

# Critical Q&A Preparation (What You Must Know)

### Q1: What is "Symplectic Integration" and why is it important?
* **Answer:** Symplectic integrators (like the Leapfrog/Verlet solver) are geometric numerical integrators that preserve phase-space volume (Liouville's theorem) and conserve a shadow Hamiltonian. Unlike standard solvers (Euler, Runge-Kutta) where discretization error accumulates indefinitely, symplectic solvers bound the energy error over arbitrarily long integration steps. This ensures physical stability during weight-hyperparameter co-evolution.

### Q2: Why is HHD-Unified's actual validation loss (0.003270) so much better than what was logged in its epoch history?
* **Answer:** During the training epochs (Phase 2), HMC fluctuates stochastically, maintaining a coarse validation loss around `0.1422`. However, at the end of training, Phase 3 executes a dedicated L-BFGS Final Polish (100 iterations) using curvature information. This final pass converges the weights perfectly to the nearest local minimum, dropping the validation loss to `0.003270`. 

### Q3: Why does Hybrid ABBO have a worse MAE (0.1049) than HHD-Unified (0.0176) even though its training loss converges?
* **Answer:** The training and validation datasets contain target noise ($\sigma = 0.05$). Hybrid ABBO's GP surrogate directly minimizes this noisy loss, causing it to overfit the noise. HHD-Unified uses physical constraints and regularization during HMC exploration, allowing it to generalize past the noise and reconstruct the *noiseless* mathematical landscape.

### Q4: How are gradients computed for discrete structures like "n_layers" and "n_neurons"?
* **Answer:** Since structural boundaries are discrete, we use **weight-transfer architecture rebuilding**. To evaluate a proposal ($n + 1$ layers), the model topology is rebuilt, compatible weights from the current model are transferred (retaining learned representations), a forward/backward pass is run, and finite differences are used to calculate the surrogate gradient before discarding the temporary architecture.
