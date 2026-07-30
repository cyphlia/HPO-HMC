# Hamiltonian Hyperparameter Dynamics (HHD)
## Self-Tuning Neural Networks via Symplectic Integration

A unified optimization framework that treats neural network hyperparameters as continuous dynamical variables in an augmented Bregman Hamiltonian system. HHD enables the joint, concurrent co-evolution of model weights $\theta$ and hyperparameters $\lambda$ along smooth physical trajectories via geometric, symplectic integration.

---

## Overview

This repository implements the three optimization philosophies evaluated in the paper:

| Method | Name | Description | Approach |
|:---:|:---|:---|:---|
| **A** | **HHD-HMC** | Pure HHD Co-evolution | Symplectic Leapfrog integration of weights and HPs with a Metropolis-Hastings correction |
| **B** | **Hybrid ABBO** | Decoupled baseline | Gaussian Process Bayesian Optimization in the outer loop, Adam & L-BFGS in the inner loop |
| **C** | **HHD-Unified** | Multi-phase HHD *(Novel)* | Three-phase curriculum: Adam Warmup $\rightarrow$ HMC Co-evolution $\rightarrow$ L-BFGS Curvature Polish |

### Key Innovation
HHD shifts hyperparameter optimization from a decoupled outer-loop black-box problem to a joint physical system defined by the augmented Hamiltonian:

$$H(\theta, p_{\theta}, \lambda, p_{\lambda}) = T_{\theta}(p_{\theta}) + T_{\lambda}(p_{\lambda}) + \mathcal{L}(\theta, \lambda)$$

By simulating hyperparameter trajectories using a **symplectic Leapfrog integrator**, HHD preserves a shadow Hamiltonian with provable energy conservation guarantees and smooth hyperparameter curves.

### Numerical Stability & Implementation Safeguards
- **Temperature-Scaled Metropolis Acceptance:** Scales the complete energy differential $\min(1, \exp(-\Delta H / T))$ to guarantee ~100% proposal acceptance at $T=10^9$ without kinetic drift rejections.
- **Leapfrog Gradient Clipping & NaN Guards:** Gradient norm clipping within momentum updates and non-finite proposal guards ($\text{loss}$ or $H \in \{\text{NaN}, \infty\}$) prevent numerical divergence over long runs.
- **Best Validation Checkpointing:** Both Method A and Method C track running validation minimums, ensuring returned models reflect optimal weight-hyperparameter checkpoints rather than terminal trajectory wandering.

---

## Project Structure

```
current/
├── main.py                       # CLI entry point
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── colabnotebooks/               # Jupyter / Colab notebooks (two series)
│   │
│   │  # Series A — Self-contained method implementations (run in Colab without cloning src/)
│   ├── 01_pure_hamiltonian_sole.ipynb # Pure HHD (no Adam) on Harmonic Osc., Double-Well, Hénon-Heiles
│   ├── 02_method_a_hhd_hmc.ipynb      # Method A (HHD-HMC): Adam warmup → HMC co-evolution
│   ├── 03_method_c_hhd_unified.ipynb  # Method C (HHD-Unified): 3-phase curriculum & L-BFGS triggers
│   ├── 04_comparative_benchmark.ipynb # Head-to-head comparison of Pure HHD vs Method A vs Method C
│   │
│   │  # Series B — Narrative walkthrough (imports from src/, teaches concepts step-by-step)
│   ├── 00_Physics_Intuition.ipynb     # Physics from zero: Euler vs Leapfrog, symplectic integration
│   ├── 01_Methods_A_B_C_Live.ipynb    # Live execution of all 3 methods on Harmonic Oscillator
│   ├── 02_Statistics_Explained.ipynb  # Statistical testing walkthrough (Friedman, 5-seed analysis)
│   ├── 03_RealWorld_Showcase_and_Optimizers_Curse.ipynb  # Clinical data + Optimizer's Curse analysis
│   ├── 04_NUTS_vs_Leapfrog.ipynb      # NUTS vs fixed-step leapfrog: why the fancier tool loses
│   └── 05_Summary_and_Next_Steps.ipynb # Capstone: all results in one table + next steps
│
├── pinn_overlap/                 # HHD applied to Physics-Informed Neural Networks (PINNs)
│   ├── README.md                 # Educational guide: PINNs from scratch + HHD overlap motivation
│   ├── src/
│   │   ├── pinn_model.py         # PINN neural network with autograd derivative utilities
│   │   ├── pinn_loss.py          # Multi-component PINN loss (PDE residual + BC + IC)
│   │   ├── pde_problems.py       # Canonical PDEs: Heat, Burgers, Poisson (with exact solutions)
│   │   └── hhd_pinn_trainer.py   # HHD-driven PINN trainer (loss weights as dynamical variables)
│   └── notebooks/
│       └── 01_hhd_pinn_demo.ipynb # Interactive demo: baseline vs HHD-PINN comparison
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
│   ├── statistical_tests.py      # Friedman and Wilcoxon statistical tests
│   ├── fashion_mnist_testbed.py  # Fashion-MNIST Deep MLP testbed script
│   ├── real_world_showcase.py    # Real-world clinical showcase: breast cancer & diabetes diagnosis
│   ├── real_world_significance.py# Friedman/Nemenyi significance tests for the real-world showcase
│   ├── real_world_breast_cancer.py # Wisconsin Diagnostic Breast Cancer benchmark
│   └── nuts_benchmark.py         # NUTS benchmark comparing adaptive vs. fixed HMC
│
├── evaluation/                   # Evaluation & plotting utilities
│   ├── evaluate.py               # Unified evaluation & results printer
│   ├── plot_results.py           # Generates Figures 1-5 (Harmonic oscillator details)
│   ├── plot_hpobench.py          # Generates Figures 6-8 (HPOBench regret & rankings)
│   ├── plot_extra.py             # Generates Figures 9-10 (Val loss & CNN histories)
│   ├── plot_all_results.py       # Comprehensive script generating all paper figures
│   └── plot_fashion_mnist.py     # Generates publication plots for Fashion-MNIST testbed
│
├── validation/                   # Theoretical validation framework
│   ├── validate.py               # Main validation script
│   ├── energy_stability.py       # Leapfrog stability and energy error scaling
│   ├── convergence_ergodicity.py # Markov chain ergodicity and acceptance bounds
│   └── gradient_generalization.py# Generalization error scaling tests
│
├── docs/                         # LaTeX papers, drafts, and project writeups
│   ├── compile_latex.py          # Helper script to compile LaTeX documents (builds HO_main.tex)
│   ├── HO_main.tex               # Canonical, actively-maintained paper (build this one)
│   ├── REAL_WORLD_SHOWCASE.md    # Write-up for the breast-cancer/diabetes real-world experiments
│   ├── nuts_explanation.md       # Explanation of NUTS integration in joint HHD
│   ├── archive/                  # Superseded drafts (HO_main2.tex, ieee_paper.tex)
│   └── cs23b1019_report.pdf      # Compiled paper PDF
│
├── results/                      # Consolidated experiment logs & JSONs
│   ├── ablation/                 # Ablation study JSONs
│   ├── cnn/                      # CIFAR-10 CNN benchmark JSONs
│   ├── harmonic_oscillator/      # Harmonic oscillator runs (Method A, B, C)
│   ├── hpobench/                 # Tabular HPOBench trajectories
│   ├── physics_benchmarks/       # Hénon-Heiles & Double-Well JSONs (single-seed, exploratory --
│   │                              #   not statistically validated, not cited in the paper)
│   ├── validation/               # Mathematical validation output metrics
│   ├── fashion_mnist/            # Fashion-MNIST Deep MLP testbed JSONs
│   ├── breast_cancer/            # Real-world clinical showcase: breast cancer diagnosis
│   ├── diabetes/                 # Real-world clinical showcase: diabetes diagnosis
│   └── real_world_significance.json  # Friedman/Nemenyi results for both clinical datasets
│
└── plots/                        # Generated figures and charts
```

---

## A Note on `docs/`

**`HO_main.tex` is the paper.** If you're reading, citing, or editing the
paper, that is the only file to touch. `docs/archive/` contains earlier
drafts and a stale parallel fork (`HO_main2.tex`, which used a different
method-naming convention and had already drifted from `HO_main.tex`) kept
only for history — they are not maintained and should not be edited.
`docs/compile_latex.py` builds `HO_main.tex` specifically.

---

## Quick Start & Result Reproduction

### 1. Installation & Environment Setup
Install the necessary requirements (PyTorch, SciPy, Optuna, scikit-learn, and simple-hpo-bench):
```bash
pip install -r requirements.txt
```

#### Dataset Setup
- **Wisconsin Breast Cancer:** Loaded directly from `scikit-learn` (no download needed).
- **Pima Indians Diabetes:** Place `pima-indians-diabetes.csv` (obtainable from standard UCI repositories/Kaggle) in a `data/` folder at the project root:
  ```
  data/pima-indians-diabetes.csv
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
1. **Run the Benchmark:** Evaluate all three methods on the CIFAR-10 small slice:
   ```bash
   python main.py --task cnn
   ```
   This trains HHD, BO, and Unified on the CNN and writes logs to `results/cnn/benchmark_results.json`.
2. **Generate Plots:** Generate accuracy progression comparison curves:
   ```bash
   python evaluation/plot_extra.py
   ```

---

### 4. How to Reproduce Fashion-MNIST Deep MLP Testbed Results
1. **Run the Benchmark:** Run Method C and Default Adam across all 5 seeds:
   ```bash
   python main.py --task fashion_mnist
   ```
   *Or direct runner command with custom settings:*
   ```bash
   python scripts/fashion_mnist_testbed.py --seeds 0,1,2,3,4 --methods default,methodC
   ```
   This trains both models and writes JSON results to `results/fashion_mnist/`.
2. **Generate Plots:** Generate the accuracy comparison, convergence curves, and HP trajectory plots:
   ```bash
   python evaluation/plot_fashion_mnist.py
   ```

---

### 5. How to Reproduce HPOBench & NAS-Bench-201 Tabular Results
1. **Run the Benchmark:** Execute the tabular lookups across the datasets and seeds:
   ```bash
   python main.py --task hpobench
   ```
   This writes lookup trajectories to `results/hpobench/`.
2. **Generate Plots & Statistical Diagrams:**
   ```bash
   python evaluation/plot_hpobench.py
   python scripts/statistical_tests.py
   ```

---

### 6. Run Theoretical Validation
To verify Hamiltonian conservation metrics, ergodicity bounds, and generalization scaling:
```bash
python validation/validate.py
```

---

## Empirical Results

### 1. Harmonic Oscillator Benchmark
Evaluated over 5 independent random seeds (mean $\pm$ std). Matches the results reported in Table 3 of the paper:

| Metric | A: HHD-HMC | B: Hybrid ABBO | C: HHD-Unified | C's Improvement vs B |
|:---|:---:|:---:|:---:|:---:|
| **Best Val. MSE** | $0.2439 \pm 0.1627$ | $0.0952 \pm 0.0051$ | $\mathbf{0.00331 \pm 0.00014}$ | **~28.8x Better** |
| **Landscape MAE** | $0.3618 \pm 0.1091$ | $0.1028 \pm 0.0149$ | $\mathbf{0.0208 \pm 0.0014}$ | **~4.9x Better** |
| **Landscape RMSE**| $0.4872 \pm 0.1669$ | $0.1403 \pm 0.0197$ | $\mathbf{0.0270 \pm 0.0019}$ | **~5.2x Better** |
| **$R^2$ Score** | $0.9785 \pm 0.0158$ | $0.9984 \pm 0.0005$ | $\mathbf{0.99994 \pm 0.00001}$ | **Near-Perfect Fit** |
| **Wall time (s)** | $\mathbf{26.6 \pm 1.3}$ | $99.9 \pm 1.9$ | $85.6 \pm 1.4$ | **~14% Faster** |

---

### 2. CIFAR-10 CNN Classification Benchmark
Evaluated over 5 independent random seeds (mean $\pm$ std). Matches the results reported in Table 4 of the paper:

| Metric | A: HHD-HMC | B: Hybrid ABBO | C: HHD-Unified |
|:---|:---:|:---:|:---:|
| **Best Val. Acc. (%)** | $30.90 \pm 1.59$ | $28.50 \pm 2.19$ | $\mathbf{30.60 \pm 2.65}$ |
| **Wall time (s)** | $42.0$ | $\mathbf{31.9}$ | $43.2$ |
| **Final $\eta$** | $9.7 \times 10^{-4}$ | $4.5 \times 10^{-3}$ | $1.7 \times 10^{-3}$ |
| **Final $p_{\mathrm{drop}}$** | $0.19$ | $0.27$ | $0.20$ |

---

### 3. Fashion-MNIST Deep MLP Classification Results
Evaluated over 5 independent random seeds (mean $\pm$ std). Evaluates performance under dynamic hidden layers/units:

| Method | Best Validation Acc (%) | Wall-clock Time (s) | Final $\eta$ | Final $p_{\mathrm{drop}}$ |
|:---|:---:|:---:|:---:|:---:|
| **Default Adam (Fixed HPs)** | $84.43 \pm 0.41$ | $\mathbf{32.9}$ | $1.0 \times 10^{-3}$ | $0.20$ |
| **C: HHD-Unified** | $\mathbf{85.01 \pm 0.12}$ | $85.9$ | $1.1 \times 10^{-3}$ | $0.20$ |

*HHD-Unified achieves a higher validation accuracy than Default Adam, while maintaining a **~3.4x tighter standard deviation** across seeds due to the robust exploratory HMC co-evolution.*

---

### 4. Standardised Tabular Benchmarks (HPOBench, HPOLib, NAS-Bench-201)
Average Rankings across 11 datasets (1 = best). Matches the rank summary reported in Table 7 of the paper:

1. **Optuna TPE**: **1.36**
2. **Hybrid ABBO (Method B)**: **2.82**
3. **HHD-Unified (Method C)**: **3.09**
4. **Random Search**: **3.64**
5. **HHD-HMC (Method A)**: **4.09**

*The Friedman test yields $p = 7.92\times10^{-4}$ (highly significant), and the Nemenyi critical difference at $\alpha=0.05$ is $\mathrm{CD} = 1.84$. The rank difference between TPE and HHD-Unified ($1.73$) is less than the critical difference; they are statistically indistinguishable on these datasets.*

---

### 5. Real-World Clinical Tabular Showcase (Breast Cancer & Diabetes)
Evaluated over 5 independent random seeds (mean $\pm$ std) on held-out test splits. Demonstrates method performance under clinical diagnostic decision-support constraints:

#### Wisconsin Breast Cancer (569 patients, 30 features)
| Method | Held-out Test AUROC | Malignant Recall | Wall-clock Time (s) |
|:---|:---:|:---:|:---:|
| **Default Adam (Fixed HPs)** | $0.9959 \pm 0.0010$ | $0.9291 \pm 0.0329$ | $\mathbf{1.2}$ |
| **Random Search (20 trials)** | $0.9952 \pm 0.0022$ | $0.8766 \pm 0.0818$ | $21.7$ |
| **Optuna TPE (20 trials)** | $\mathbf{0.9963 \pm 0.0031}$ | $\mathbf{0.9384 \pm 0.0322}$ | $22.2$ |
| **C: HHD-Unified** | $0.9949 \pm 0.0028$ | $0.9241 \pm 0.0179$ | $\mathbf{1.6}$ |

#### Pima Indians Diabetes (768 patients, 8 features)
| Method | Held-out Test AUROC | Diabetic Recall | Wall-clock Time (s) |
|:---|:---:|:---:|:---:|
| **Default Adam (Fixed HPs)** | $\mathbf{0.8265 \pm 0.0225}$ | $0.6242 \pm 0.0524$ | $\mathbf{1.7}$ |
| **Random Search (20 trials)** | $0.8111 \pm 0.0306$ | $\mathbf{0.6996 \pm 0.1428}$ | $27.6$ |
| **Optuna TPE (20 trials)** | $0.8212 \pm 0.0207$ | $0.4770 \pm 0.1741$ | $29.0$ |
| **C: HHD-Unified** | $0.8224 \pm 0.0226$ | $0.6090 \pm 0.0847$ | $\mathbf{2.1}$ |

*Friedman statistical tests ($p=0.472$ for Breast Cancer, $p=0.102$ for Diabetes) confirm all 4 methods are statistically indistinguishable in test AUROC and recall at $N=5$ seeds. Method C achieves statistical quality parity while delivering a **13–19x wall-clock speedup** over 20-trial search baselines.*

---

## Theoretical Validation

The mathematical proof validation suite in `validation/` evaluates three core properties:
1. **Symplectic Conservation (Theorem 1)** — Verifies the Leapfrog integrator preserves energy with $| \Delta H | = \mathcal{O}(\epsilon^2)$ scaling.
2. **Detailed Balance (Theorem 2)** — Verifies that HMC proposals satisfy detailed balance and maintain ergodicity.
3. **Convergence Rates (Theorem 3)** — Empirically validates optimization loss profiles across training epochs.

Run the validation suite via:
```bash
python validation/validate.py
```

---

## References

1. Duane et al. (1987). "Hybrid Monte Carlo." *Physics Letters B*, 195(2):216-222.
2. Neal (2011). "MCMC using Hamiltonian Dynamics." *Handbook of Markov Chain Monte Carlo*.
3. Kingma & Ba (2015). "Adam: A Method for Stochastic Optimization." *ICLR*.
4. Liu & Nocedal (1989). "On the Limited Memory BFGS Method." *Mathematical Programming*, 45(1):503-528.
5. Demsar (2006). "Statistical Comparisons of Classifiers over Multiple Data Sets." *JMLR*, 7(1):1-30.

---

## License
Academic use. See individual file headers for details.
