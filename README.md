# Hamiltonian Hyperparameter Dynamics (HHD)
## Self-Tuning Neural Networks via Symplectic Integration

A novel framework for **automatic hyperparameter optimization** that treats
hyperparameters as dynamical variables in an extended Hamiltonian system,
enabling joint co-evolution of network weights and hyperparameters through
symplectic integration.

---

## Overview

This project implements and compares three hyperparameter optimization methods:

| Method | Name | Approach |
|--------|------|----------|
| **A** | Pure HHD | Hamiltonian Monte Carlo co-evolution of weights + hyperparameters |
| **B** | Hybrid BO | Gaussian Process Bayesian Optimization + Adam + L-BFGS |
| **C** | Unified HHD-ABBO | Three-phase curriculum: Adam → HMC → L-BFGS *(Novel)* |

**Method C** is the novel contribution, combining the theoretical guarantees
of symplectic integration (from A) with the practical convergence benefits
of second-order optimization (from B).

### Key Innovation
The extended Hamiltonian:

```
H(θ, p_θ, λ, p_λ) = T(p_θ)/m_θ + T(p_λ)/m_λ + L(θ, λ)
```

enables **continuous**, **physics-based** hyperparameter trajectories with
provable energy conservation guarantees (Theorem 1 in `validation/`).

---

## Project Structure

```
current/
├── config.py                    # All configuration & search spaces
├── data_generator.py            # Harmonic oscillator data generation
├── hamiltonian.py               # Core: NN architecture + HP state + Hamiltonian system
├── symplectic_solver.py         # Leapfrog integrator + HMC sampler
├── train_hamiltonian.py         # Method A: Pure HHD trainer
├── hybrid_adam_bfgs.py          # Method B: Hybrid BO trainer
├── hybrid_hhd_abbo_improved.py  # Method C: Unified HHD-ABBO trainer (Novel)
├── cnn_benchmark.py             # CNN/MNIST benchmark for all 3 methods
├── hpobench_benchmark.py        # Tabular HPOBench benchmark runner for all methods
├── evaluate.py                  # Unified evaluation & results tables
├── main.py                      # CLI entry point
├── requirements.txt             # Python dependencies
├── plot_results.py              # Generates Figures 1-5 (Harmonic & CNN details)
├── plot_hpobench.py             # Generates Figures 6-8 (HPOBench regret & rankings)
├── plot_extra.py                # Generates Figures 9-10 (Val loss & CNN HHD vs ABBO)
├── generate_report_pdf.py       # Compiles cs23b1019_report.pdf
├── validation/                  # Theoretical validation framework
│   ├── validate.py              # Empirical validation tests
│   └── THEORETICAL_FOUNDATIONS.md
├── results/                     # Harmonic oscillator results
├── results_cnn/                 # CNN benchmark results
└── results_hpobench/            # HPOBench tabular lookup results
```

---

## Quick Start & Result Reproduction

### 1. Installation & Environment Setup
First, install all necessary dependencies (including PyTorch, SciPy, Optuna, scikit-learn, and the HPOBench tabular library):
```bash
pip install -r requirements.txt
```

---

### 2. How to Reproduce Harmonic Oscillator Results
1.  **Run the Benchmark:** Train all three methods (HHD, ABBO, and Unified) on the Harmonic Oscillator grid dataset:
    ```bash
    python main.py --task harmonic --compare
    ```
2.  **Generate Plots:** Update the landscape surface overlays and training histories (Figures 1-5, and Figure 9):
    ```bash
    python plot_results.py
    python plot_extra.py
    ```

---

### 3. How to Reproduce MNIST CNN Benchmark Results
1.  **Run the Benchmark:** Evaluate all three methods on the subsampled MNIST digits dataset:
    ```bash
    python main.py --task cnn
    ```
    This trains HHD, BO, and Unified on the CNN and writes result logs to `results_cnn/benchmark_results.json`.
2.  **Generate Plots:** Update the CNN accuracy progression curves (Figure 10 comparing HHD vs ABBO):
    ```bash
    python plot_extra.py
    ```

---

### 4. How to Reproduce HPOBench & NAS-Bench-201 Tabular Results
1.  **Run the Benchmark:** Execute the tabular lookups across all 11 datasets and 5 seeds:
    ```bash
    python main.py --task hpobench
    ```
    *Alternative (direct execution to customize parameters):*
    ```bash
    python hpobench_benchmark.py --trials 100 --seeds 0,1,2,3,4
    ```
    This writes lookup trajectories to the `results_hpobench/` directory.
2.  **Generate Plots:** Generate regret curves and average rank heatmaps (Figures 6, 6b, 7, and 8):
    ```bash
    python plot_hpobench.py
    ```

---

### 5. How to Compile the PDF Report
To compile the final two-page PDF report (`cs23b1019_report.pdf`) with the latest tables and figures, run:
```bash
python generate_report_pdf.py
```

---

### 6. Run Theoretical Validation
To verify Hamiltonian conservation metrics and error bounds:
```bash
cd validation
python validate.py
```

---

## Methods in Detail & Tuning Walkthroughs

### 1. Algorithm A: Pure Hamiltonian Hyperparameter Dynamics (HHD)
Method A co-evolves the network weights $\theta$ and the hyperparameters $\lambda$ inside a joint continuous phase space.

*   **Setup Formulation:** Hyperparameters are continuously relaxed variables: $\lambda = [\log(\text{lr}), \text{dropout}]$ (LR is log-scaled). Loss acts as potential energy; weights and HPs have momenta ($p_\theta, p_\lambda$) and masses ($m_\theta = 1.0, m_\lambda = 1.0$).
*   **Step-by-Step Walkthrough:**
    1.  **Warm-up:** Train model weights using Adam for 5 epochs while keeping hyperparameters frozen to stabilize gradients.
    2.  **Momentum Sampling:** At the start of each of the 15 epochs, retrieve a training batch and draw random momentum vectors: $p_\theta, p_\lambda \sim \mathcal{N}(0, I)$.
    3.  **Leapfrog Integration:** Simulate joint trajectories using a **symplectic Leapfrog solver** (4 steps, step size $\epsilon = 0.005$). Gradients with respect to weights ($\nabla_\theta L$) are computed via backpropagation; gradients with respect to HPs ($\nabla_\lambda L$) are computed via central finite differences.
    4.  **Metropolis-Hastings:** Evaluate the proposed coordinates ($\theta_{\text{prop}}, \lambda_{\text{prop}}$). Accept the new state with Metropolis probability ($T = 1e9$, optimization mode).
    5.  **Propagation & Epoch Training:** Set the network's dropout rate to the proposed dropout, update the Adam optimizer with the proposed learning rate, and train the weights for one full epoch (keeping HPs static). Evaluate test accuracy at the end of each epoch and cache the best checkpoints.

### 2. Algorithm B: Hybrid BO (ABBO)
Method B treats HPO as a decoupled black-box problem, separating outer-loop hyperparameter queries from inner-loop model training.

*   **Setup Formulation:** A Gaussian Process (GP) surrogate model is constructed over the 2D search space ($\log(\text{lr}), \text{dropout}$). An Expected Improvement (EI) acquisition function is maximized using L-BFGS-B (running 5 restarts) to select hyperparameter query points.
*   **Step-by-Step Walkthrough:**
    1.  **HP Selection:** Loop for 10 trials. Suggest hyperparameters randomly (trials 1–3) or by maximizing EI (trials 4–10).
    2.  **Model Reset:** Fully discard the previous model. Instantiate a fresh neural network from scratch, initializing weights randomly, and setting the proposed dropout rate.
    3.  **Inner Weight Training:** Train the new model from scratch for 10 epochs using Adam with the proposed learning rate.
    4.  **GP Update:** Evaluate accuracy on validation data. Define the cost feedback as $y = -\text{Accuracy}_{\text{val}}$. Feed $(\lambda_t, y_t)$ back to update the GP surrogate prior for the next trial, caching the best overall trial.

### 3. Method C: Unified HHD-ABBO (Novel Contribution)
Method C merges HHD's physical exploration with second-order L-BFGS curvature refinement in a single, unified curriculum:
*   **Phase 1 (Adam Warm-up):** Quick 30-epoch training to guide weights to a stable local basin.
*   **Phase 2 (Co-Evolution):** Joint HMC updates over 80 epochs with an **adaptive step-size controller** (adjusts $\epsilon$ to target a 65% acceptance rate). Structural parameters (`n_layers`, `n_neurons`) are frozen to prevent capacity collapse.
*   **Phase 3 (Plateau L-BFGS Polish):** If a training loss plateau is detected, Method C pauses HMC and runs a sequence of L-BFGS steps. A dedicated 100-step L-BFGS polish is run at the very end of training, locking in HPs and converging weights.

---

## Experimental Setup & Benchmarks

### 1. Simple Harmonic Oscillator (Landscape Regression)
*   **Setup:** Samples $2,500$ points uniformly from $q \in [-4.0, 4.0]$ and $p \in [-4.0, 4.0]$. Target values $H(q,p) = p^2/2m + \frac{1}{2}kq^2$ (with $m = 1.0, k = 1.0$) are corrupted with Gaussian noise ($\sigma = 0.05$).
*   **Split:** 80% train (2,000 samples) and 20% validation (500 samples). Validation and plots are evaluated on a noiseless $50 \times 50$ evaluation grid.
*   **Network:** MLP mapping inputs $[q, p]$ to scalar energy $H$. Tunes learning rate, dropout, depth, width, and batch size.

### 2. MNIST CNN Classification
*   **Setup:** Subsamples MNIST training set to 5,000 images (down from 60,000) to introduce overfitting pressure; test/validation set size is 1,000 images. Batch size is fixed at 64.
*   **Architecture:** Conv1 (16 output channels, kernel 3) $\rightarrow$ ReLU $\rightarrow$ MaxPool($2 \times 2$) $\rightarrow$ Conv2 (32 output channels, kernel 3) $\rightarrow$ ReLU $\rightarrow$ MaxPool($2 \times 2$) $\rightarrow$ Flatten $\rightarrow$ FC1 (128) $\rightarrow$ ReLU $\rightarrow$ Dropout ($p$) $\rightarrow$ FC2 (10 output classes). Tunes learning rate and dropout rate.

### 3. Tabular HPOBench Evaluation
*   **Setup:** 11 diverse datasets from HPOBench, HPOLib, and NAS-Bench-201 (e.g., Random Forest Tabular Australian, Blood Transfusion, Segment, Naval Propulsion, Parkinsons, Cifar10, ImageNet).
*   **Comparison:** Evaluates Method C against standard HPO baselines (Optuna TPE, Random Search) over a budget of 100 trials across 5 seeds.

---

## Empirical Results

### 1. Harmonic Oscillator Performance Comparison

| Metric | Method A (Pure HHD) | Method B (Hybrid BO) | Method C (Unified HHD-ABBO) | Improvement vs Method B |
|:---|:---:|:---:|:---:|:---:|
| **Best Val Loss** | 0.151283 | 0.098957 | **0.003270** | **~30x Better** |
| **Landscape MAE** | 0.699041 | 0.104961 | **0.017673** | **~6x Better** |
| **Landscape RMSE**| 0.786251 | 0.139415 | **0.024848** | **~5.6x Better** |
| **R² Score** | 0.949784 | 0.998421 | **0.999950** | **Near-Perfect Fit** |

*Method C's dedicated L-BFGS polish reduces final validation loss to `0.003270` and fits the noiseless true potential energy surface perfectly without overfitting the dataset noise.*

### 2. MNIST CNN Classification Results

| Method | Best Validation Acc | Final Validation Acc | Wall-clock Time (s) | Optimized LR | Optimized Dropout |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Method A (HHD)** | **97.80%** | **97.80%** | **122.7** | 0.001062 | 0.2059 |
| **Method B (BO)** | 97.40% | 97.40% | 698.7 | 0.000619 | 0.2236 |
| **Method C (Unified)**| **97.70%** | **97.70%** | **214.8** | 0.000789 | 0.2440 |

*Method C achieves comparable classification accuracy to HHD/BO while running **over 3.2x faster than Method B**, eliminating the need for expensive multi-trial network re-initializations.*

### 3. Tabular HPOBench Average Rankings (Lower is better)

Evaluated across 11 tabular datasets from HPOBench, HPOLib, and NAS-Bench-201:
1.  **Method C (Unified HHD-ABBO):** **2.45** (Best)
2.  **Optuna TPE:** 2.64
3.  **Method A (Pure HHD):** 3.18
4.  **Method B (Hybrid BO):** 3.18
5.  **Random Search:** 3.55

---

## Theoretical Validation

The `validation/` folder provides:

1. **Theorem 1: Symplectic Conservation** — Verifies |ΔH| = O(ε²) scaling
2. **Theorem 2: Detailed Balance** — Verifies HMC acceptance rates and ergodicity
3. **Theorem 3: Convergence Rates** — Compares epochs-to-threshold across methods
4. **Standard Optimizer Benchmark** — Compares against SGD, Adam, AdamW

See `validation/THEORETICAL_FOUNDATIONS.md` for full mathematical proofs.

---

## Configuration

All parameters are centralized in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `STEP_SIZE` | 0.01 | Leapfrog step size (ε) |
| `N_LEAPFROG_STEPS` | 5 | Leapfrog steps per HMC proposal (L) |
| `TEMPERATURE` | 1e9 | Boltzmann temperature (1e9 = optimization mode) |
| `MASS_THETA` | 1.0 | Weight momentum inertia |
| `MASS_LAMBDA` | 0.1 | Hyperparameter momentum inertia |
| `N_EPOCHS` | 60 | HMC co-evolution epochs |
| `N_WARMUP_EPOCHS` | 20 | Adam warmup epochs |
| `N_SAMPLES` | 2500 | Training data points |

---

## CLI Reference

```
python main.py [OPTIONS]

Options:
  --task {harmonic,cnn,both}   Benchmark task (default: harmonic)
  --method {pure,hybrid,improved}  Method to run (default: improved)
  --compare                    Run 3-way comparison
  --full                       Full pipeline: both tasks + evaluation
  --evaluate-only              Only evaluate saved results
  --epochs N                   HMC co-evolution epochs
  --warmup N                   Adam warmup epochs
  --n-samples N                Training data points
  --step-size F                Leapfrog step size
  --n-leapfrog N               Leapfrog steps per proposal
  --temperature F              Boltzmann temperature
  --seed N                     Random seed
  --device {cpu,cuda,mps}      Compute device
```

---

## References

1. Kingma & Ba (2015). "Adam: A Method for Stochastic Optimization." ICLR.
2. Liu & Nocedal (1989). "On the Limited Memory BFGS Method." Math. Programming.
3. Duane et al. (1987). "Hybrid Monte Carlo." Physics Letters B.
4. Neal (2011). "MCMC using Hamiltonian Dynamics." Handbook of MCMC.
5. Betancourt (2017). "A Conceptual Introduction to HMC." arXiv:1701.02434.
6. Leimkuhler & Reich (2004). "Simulating Hamiltonian Dynamics." Cambridge.
7. Hairer, Lubich & Wanner (2006). "Geometric Numerical Integration." Springer.

---

## License

Academic use. See individual file headers for details.
