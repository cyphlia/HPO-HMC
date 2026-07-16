# Hamiltonian Hyperparameter Dynamics (HHD)
## Self-Tuning Neural Networks via Symplectic Integration

A novel framework for **automatic hyperparameter optimization** that treats hyperparameters as dynamical variables in an extended Hamiltonian system, enabling joint co-evolution of network weights and hyperparameters through symplectic integration.

---

## Overview

This project implements and compares three hyperparameter optimization methods:

| Method | Name | Approach |
|--------|------|----------|
| **A** | Pure HHD | Hamiltonian Monte Carlo co-evolution of weights + hyperparameters |
| **B** | Hybrid BO | Gaussian Process Bayesian Optimization + Adam + L-BFGS |
| **C** | Unified HHD-ABBO | Three-phase curriculum: Adam → HMC → L-BFGS *(Novel)* |

**Method C** is the novel contribution, combining the theoretical guarantees of symplectic integration (from A) with the practical convergence benefits of second-order optimization (from B).

### Key Innovation
The extended Hamiltonian:
```
H(θ, p_θ, λ, p_λ) = T(p_θ)/m_θ + T(p_λ)/m_λ + L(θ, λ)
```
enables **continuous**, **physics-based** hyperparameter trajectories with provable energy conservation guarantees (see [validate.py](file:///c:/Minor%20Project/Model/current/validation/validate.py)).

---

## Project Structure

The project directory is structured as follows:

```
current/
├── main.py                       # CLI entry point
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── src/                          # Core implementation files
│   ├── config.py                 # Centralized configuration & search spaces
│   ├── data_generator.py         # Data generator for physical systems
│   ├── hamiltonian.py            # NN architecture, HP state, and Hamiltonian system
│   ├── symplectic_solver.py      # Leapfrog integrator & HMC sampler (with NaN guards)
│   ├── train_hamiltonian.py      # Method A: Pure HHD trainer (with best checkpointing)
│   ├── hybrid_adam_bfgs.py       # Method B: Hybrid BO trainer
│   └── hybrid_hhd_abbo_improved.py # Method C: Unified HHD-ABBO trainer
│
├── scripts/                      # Experiment and benchmark runners
│   ├── cnn_benchmark.py          # CIFAR-10 CNN benchmark for all 3 methods
│   ├── hpobench_benchmark.py     # Tabular HPOBench benchmark runner
│   ├── performance_testbed.py    # Optimization testbed comparison
│   ├── run_physics_benchmarks.py # Multi-system physics benchmark (Harmonic, Double-Well, Hénon-Heiles)
│   ├── ablation_study.py         # Ablation study for Method C variants
│   ├── sensitivity_analysis.py   # Meta-hyperparameter sensitivity sweeps
│   └── statistical_tests.py      # Friedman and Wilcoxon statistical tests
│
├── evaluation/                   # Evaluation & plotting utilities
│   ├── evaluate.py               # Unified evaluation & results printer
│   ├── plot_results.py           # Generates Figures 1-5 (Harmonic oscillator details)
│   ├── plot_hpobench.py          # Generates Figures 6-8 (HPOBench regret & rankings)
│   ├── plot_extra.py             # Generates Figures 9-10 (Val loss & CNN histories)
│   └── plot_all_results.py       # Comprehensive script generating all paper figures
│
├── validation/                   # Theoretical validation framework
│   ├── validate.py               # Main validation script
│   ├── energy_stability.py       # Leapfrog stability and energy error scaling
│   ├── convergence_ergodicity.py # Markov chain ergodicity and acceptance bounds
│   └── gradient_generalization.py# Generalization error scaling tests
│
├── docs/                         # LaTeX papers, drafts, and project writeups
│   ├── compile_latex.py          # Helper script to compile LaTeX documents
│   ├── cs23b1019_report.pdf      # Final compiled paper PDF
│   └── PROJECT_OVERVIEW.md       # High-level overview and mathematical details
│
├── results/                      # Consolidated experiment logs & JSONs
│   ├── ablation/                 # Ablation study JSONs
│   ├── cnn/                      # CIFAR-10 CNN benchmark JSONs
│   ├── harmonic_oscillator/      # Harmonic oscillator runs (Method A, B, C)
│   ├── hpobench/                 # Tabular HPOBench trajectories
│   ├── physics_benchmarks/       # Hénon-Heiles & Double-Well JSONs
│   └── validation/               # Mathematical validation output metrics
│
└── plots/                        # Generated figures and charts
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
1. **Run the Benchmark:** Train all three methods (HHD, ABBO, and Unified) on the Harmonic Oscillator grid dataset:
   ```bash
   python main.py --task harmonic --compare
   ```
2. **Generate Plots:** Generate the landscape surface overlays and training histories:
   ```bash
   python evaluation/plot_results.py
   python evaluation/plot_extra.py
   ```

---

### 3. How to Reproduce CIFAR-10 CNN Benchmark Results
The CNN benchmark has been upgraded to a CPU-feasible ResNet model running CIFAR-10 classification, tuning 4 HPs (`log_lr`, `dropout`, `log_wd`, `log_batch_size`).
1. **Run the Benchmark:** Evaluate all three methods:
   ```bash
   python main.py --task cnn
   ```
   This trains HHD, BO, and Unified on the CNN and writes logs to `results/cnn/benchmark_results.json`.
2. **Generate Plots:** Generate accuracy progression comparison curves:
   ```bash
   python evaluation/plot_extra.py
   ```

---

### 4. How to Reproduce HPOBench & NAS-Bench-201 Tabular Results
1. **Run the Benchmark:** Execute the tabular lookups across the datasets and seeds:
   ```bash
   python main.py --task hpobench
   ```
   *Or direct runner command:*
   ```bash
   python scripts/hpobench_benchmark.py --trials 100 --seeds 0,1,2,3,4
   ```
   This writes lookup trajectories to `results/hpobench/`.
2. **Generate Plots & Statistical Diagrams:**
   ```bash
   python evaluation/plot_hpobench.py
   python scripts/statistical_tests.py
   ```

---

### 5. Run Other Scientific Studies
1. **Run Multi-System Physics Benchmarks:** Run training across Harmonic Oscillator, Hénon-Heiles (4D), and Double-Well systems:
   ```bash
   python main.py --task physics_all
   ```
2. **Run Sensitivity Sweep:** Sweep meta-hyperparameters (such as Leapfrog step size $\epsilon$):
   ```bash
   python scripts/sensitivity_analysis.py
   ```
3. **Run Ablation Studies:** Measure the contribution of individual components of the Method C curriculum:
   ```bash
   python scripts/ablation_study.py
   ```
4. **Generate All Figures at Once:** Compile all project figures into the `plots/` directory:
   ```bash
   python evaluation/plot_all_results.py
   ```

---

### 6. Run Theoretical Validation
To verify Hamiltonian conservation metrics, ergodicity bounds, and generalization scaling:
```bash
python validation/validate.py
```

---

## Methods in Detail & Tuning Walkthroughs

### 1. Algorithm A: Pure Hamiltonian Hyperparameter Dynamics (HHD)
Method A co-evolves the network weights $\theta$ and the hyperparameters $\lambda$ inside a joint continuous phase space.
* **Warm-up:** Train model weights using Adam for 20 epochs while keeping hyperparameters frozen to stabilize gradients.
* **Momentum Sampling:** Retrieve a training batch and draw random momentum vectors: $p_\theta, p_\lambda \sim \mathcal{N}(0, I)$.
* **Leapfrog Integration:** Simulate joint trajectories using a **symplectic Leapfrog solver** (Leapfrog step size $\epsilon = 0.01$). Gradients with respect to weights ($\nabla_\theta L$) are computed via backpropagation; gradients with respect to HPs ($\nabla_\lambda L$) are computed via central finite differences.
* **Metropolis-Hastings:** Evaluate the proposed coordinates ($\theta_{\text{prop}}, \lambda_{\text{prop}}$). Accept the new state with Metropolis probability ($T = 1e9$, optimization mode).
* **Model Checkpointing (Fixed):** The trainer tracks the best parameter set seen across the HMC trajectory and restores it, preventing the dynamics from wandering away from optimal solutions in later epochs.

### 2. Algorithm B: Hybrid BO (ABBO)
Method B treats HPO as a decoupled black-box problem, separating outer-loop hyperparameter queries from inner-loop model training.
* **HP Selection:** Loop for 15 trials. Suggest hyperparameters randomly (trials 1–3) or by maximizing Expected Improvement (EI) using L-BFGS-B over a Gaussian Process (GP) surrogate.
* **Model Reset:** Fully discard the previous model. Instantiate a fresh neural network from scratch, initializing weights randomly.
* **Inner Weight Training:** Train the new model from scratch for 60 epochs using Adam.
* **GP Update:** Feed validation performance back to update the GP surrogate prior.

### 3. Method C: Unified HHD-ABBO
Method C merges HHD's physical exploration with second-order L-BFGS curvature refinement in a single, unified curriculum:
* **Phase 1 (Adam Warm-up):** Quick 20-epoch training to guide weights to a stable local basin.
* **Phase 2 (Co-Evolution):** Joint HMC updates over 60 epochs with an **adaptive step-size controller** (adjusts $\epsilon$ to target a 65% acceptance rate).
* **Phase 3 (Plateau L-BFGS Polish):** If a training loss plateau is detected, Method C pauses HMC and runs a sequence of L-BFGS steps. A final 100-step L-BFGS polish is run at the very end of training to lock in hyperparameters and quickly converge weights.

---

## Empirical Results

### 1. Harmonic Oscillator Performance Comparison

| Metric | Method A (Pure HHD) | Method B (Hybrid BO) | Method C (Unified HHD-ABBO) | Improvement vs Method B |
|:---|:---:|:---:|:---:|:---:|
| **Best Val Loss** | 0.151283 | 0.098957 | **0.003270** | **~30.3x Better** |
| **Landscape MAE** | 0.699041 | 0.104961 | **0.017673** | **~5.9x Better** |
| **Landscape RMSE**| 0.786251 | 0.139415 | **0.024848** | **~5.6x Better** |
| **R² Score** | 0.949784 | 0.998421 | **0.999950** | **Near-Perfect Fit** |

### 2. CIFAR-10 CNN Classification Results

| Method | Best Validation Acc | Final Validation Acc | Wall-clock Time (s) | Optimized LR | Optimized Dropout |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Method A (HHD)** | **30.90%** | 28.70% | 42.0 | 0.000955 | 0.2017 |
| **Method B (BO)** | 28.50% | 28.50% | **31.9** | 0.002574 | 0.2727 |
| **Method C (Unified)**| 30.60% | **29.70%** | 43.2 | 0.001094 | 0.1986 |

---

## Theoretical Validation

The `validation/` suite verifies the following mathematical principles:
1. **Symplectic Conservation (Theorem 1)** — Verifies $| \Delta H | = \mathcal{O}(\epsilon^2)$ energy error scaling for the Leapfrog integrator.
2. **Detailed Balance (Theorem 2)** — Verifies HMC transition ergodicity and correct distribution sampling.
3. **Convergence Rates (Theorem 3)** — Empirically validates training loss optimization rates.

Run `python validation/validate.py` to compile metrics and output charts to `results/validation/`.

---

## License

Academic use. See individual file headers for details.
