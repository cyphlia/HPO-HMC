# HPOBench Benchmarking Pipeline — Flow Documentation

This document describes the end-to-end architecture and data flow of the HPOBench
benchmarking pipeline implemented in `hpobench_benchmark.py`, which evaluates five
hyperparameter optimisation strategies across 11 standardised tabular benchmark
datasets.

---

## 1. Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      HPOBench Benchmarking Pipeline                        │
│                                                                             │
│  ┌──────────┐   ┌──────────────┐   ┌───────────┐   ┌──────────────────┐   │
│  │ Benchmark │──▶│  Optimizer    │──▶│ Trajectory│──▶│  Results JSON    │   │
│  │  Suites   │   │  Dispatcher   │   │  Recorder │   │  (per seed)      │   │
│  └──────────┘   └──────────────┘   └───────────┘   └──────────────────┘   │
│       │                                                     │              │
│       ▼                                                     ▼              │
│  ┌──────────┐                                       ┌──────────────────┐   │
│  │  Search   │                                       │  summary.json    │   │
│  │  Space    │                                       │  (aggregated)    │   │
│  └──────────┘                                       └──────────────────┘   │
│                                                             │              │
│                                                             ▼              │
│                                                     ┌──────────────────┐   │
│                                                     │  plot_hpobench   │   │
│                                                     │  (visualisation) │   │
│                                                     └──────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Execution Scale

| Parameter         | Value                                         |
|-------------------|-----------------------------------------------|
| Benchmark suites  | 3 (HPOBench, HPOLib, NAS-Bench-201)           |
| Total datasets    | 11                                            |
| Optimizers        | 5 (Random, Optuna TPE, Method A/B/C)          |
| Seeds per run     | 5 (seeds 0–4)                                 |
| Trials per seed   | 100                                           |
| **Total evaluations** | **11 × 5 × 100 × 5 = 27,500**            |

---

## 2. Benchmark Suites

### 2.1 HPOBench (Tabular Classification)

The HPOBench suite uses pre-computed lookup tables from the `simple-hpo-bench`
library. Each dataset defines a discrete hyperparameter search space for
standard ML models (multi-layer perceptrons, SVMs) and stores the validation
accuracy for every possible configuration.

| Dataset           | Task           | Metric    | Direction |
|-------------------|----------------|-----------|-----------|
| Australian        | Classification | Accuracy  | Maximize  |
| Blood Transfusion | Classification | Accuracy  | Maximize  |
| Vehicle           | Classification | Accuracy  | Maximize  |
| Segment           | Classification | Accuracy  | Maximize  |

### 2.2 HPOLib (Tabular Regression)

The HPOLib suite stores pre-computed validation MSE for neural network regression
tasks across different hyperparameter configurations.

| Dataset                  | Task       | Metric | Direction |
|--------------------------|------------|--------|-----------|
| Naval Propulsion         | Regression | MSE    | Minimize  |
| Parkinsons Telemonitoring| Regression | MSE    | Minimize  |
| Protein Structure        | Regression | MSE    | Minimize  |
| Slice Localization       | Regression | MSE    | Minimize  |

### 2.3 NAS-Bench-201 (Architecture Search)

NAS-Bench-201 defines a cell-based neural architecture search space with 15,625
unique architectures. Each architecture is a Directed Acyclic Graph (DAG) with
4 nodes and 5 operation choices per edge: zero, skip-connection, 1×1 convolution,
3×3 convolution, and 3×3 average pooling.

| Dataset          | Task           | Metric   | Direction |
|------------------|----------------|----------|-----------|
| CIFAR-10         | Classification | Accuracy | Maximize  |
| CIFAR-100        | Classification | Accuracy | Maximize  |
| ImageNet-16-120  | Classification | Accuracy | Maximize  |

---

## 3. Optimizer Implementations

### 3.1 Random Search

The simplest baseline. Each trial independently samples a uniformly random
configuration from the discrete search space using `numpy.random.RandomState(seed)`.

```
for each trial:
    config ← uniform_random(search_space)
    cost   ← benchmark.evaluate(config)
    update best_cost
```

**Complexity:** O(1) per trial. No state carried between trials.

### 3.2 Optuna TPE (Tree-structured Parzen Estimator)

Uses the Optuna library's TPE sampler, which models the conditional distribution
P(hyperparameters | cost < threshold) using kernel density estimation. Discrete
hyperparameters are handled via `suggest_int` over their index space.

```
for each trial:
    for each parameter:
        index ← TPE.suggest_int(0, len(choices) - 1)
        config[param] ← choices[index]
    cost ← benchmark.evaluate(config)
    update TPE model
```

**Key properties:**
- Handles categorical/discrete spaces natively via index-based suggestions.
- Amortised O(n log n) per trial where n = number of completed trials.
- Seed is passed to `TPESampler(seed=seed)` for determinism.

### 3.3 Method A — Hamiltonian HHD

Adapts the Hamiltonian Monte Carlo framework to discrete index spaces using
continuous relaxation:

```
Phase 1 (Warm-up, trials 1–20):
    Uniform random sampling (same as Random Search)
    → Identifies a low-cost starting region
    
Phase 2 (HMC Co-evolution, trials 21–100):
    Transition to best configuration found during warm-up
    for each trial:
        Sample momentum: p ~ N(0, I)
        Run leapfrog integration (L=3 steps, ε=0.5):
            Half-step momentum with finite-difference gradient
            Full-step position with clipping to bounds
            Full-step momentum (or half for last step)
        Metropolis-Hastings accept/reject:
            Δcost = proposed_cost - current_cost
            Accept with probability min(1, exp(-Δcost))
```

**Gradient computation:** Since no computational graph exists for lookup-table
benchmarks, hyperparameter gradients are approximated via one-sided finite
differences with δ=1.0 (one index step):

```
∂cost/∂λ_d ≈ (cost(λ + δ·e_d) - cost(λ)) / δ
```

### 3.4 Method B — ABBO (Bayesian Optimisation with GP-EI)

Applies GP-based Bayesian Optimisation over a normalised continuous index space:

```
Phase 1 (Random Exploration, trials 1–20):
    Sample random integer indices for each parameter
    
Phase 2 (GP-Guided, trials 21–100):
    Normalise observed indices to [0, 1] (divide by bounds_high)
    Fit GP with RBF kernel to (normalised_X, observed_costs)
    Generate 1000 random candidate configurations
    Select candidate maximising Expected Improvement:
        EI(x) = (y_best - μ(x)) · Φ(Z) + σ(x) · φ(Z)
        Z = (y_best - μ(x)) / σ(x)
    Evaluate selected configuration
```

**Key properties:**
- GP refitted at every trial (kernel: ConstantKernel × RBF, α=10⁻⁶).
- Candidate pool size: 1000 random configurations.
- Falls back to random if GP prediction fails.

### 3.5 Method C — Unified Three-Phase Curriculum

Combines random exploration, adaptive HMC, and local refinement in three phases:

```
Phase 1 (Random Warm-up, trials 1–20):
    Uniform random sampling to cover the search space
    
Phase 2 (Adaptive HMC, trials 21–80):
    Transition to best configuration from Phase 1
    for each trial:
        Run leapfrog HMC (same as Method A)
        Adapt step-size ε based on acceptance rate:
            if acceptance_rate > 0.8: ε *= 1.05 (explore more)
            if acceptance_rate < 0.4: ε *= 0.95 (be more conservative)
            Clamp: 0.01 ≤ ε ≤ 5.0
        
Phase 3 (Local Refinement, trials 81–100):
    Start from best configuration found so far
    for each trial:
        Perturb random subset of parameters by ±1 index
        Evaluate perturbed configuration
        Accept if improved
```

**Adaptive step-size** is the key differentiator from Method A: by monitoring the
HMC acceptance rate and adjusting ε, Method C avoids both overly conservative
exploration (ε too small) and excessive rejections (ε too large).

---

## 4. Data Flow

### 4.1 Per-Trial Output

Each optimizer call returns a trajectory: a list of trial records:

```json
{
    "trial": 42,
    "cost": -0.957201,
    "best_cost": -0.957201,
    "config": {"alpha": 0.001, "batch_size": 32, ...},
    "time": 0.000123
}
```

- `cost`: The raw objective value for this trial (negated for maximisation tasks).
- `best_cost`: The best (lowest) cost seen so far (monotonically non-increasing).
- `config`: The actual hyperparameter values evaluated.
- `time`: Wall-clock time for this single evaluation (seconds).

### 4.2 File Output Structure

```
results_hpobench/
├── summary.json                          # Aggregated statistics
├── hpobench/
│   ├── australian/
│   │   ├── RandomSearch_seed0.json       # Full trajectory
│   │   ├── RandomSearch_seed1.json
│   │   ├── ...
│   │   ├── MethodC_Unified_seed4.json
│   ├── blood_transfusion/
│   │   └── ...
│   ├── vehicle/
│   └── segment/
├── hpolib/
│   ├── naval_propulsion/
│   ├── parkinsons_telemonitoring/
│   ├── protein_structure/
│   └── slice_localization/
└── nasbench201/
    ├── cifar10/
    ├── cifar100/
    └── imagenet/
```

### 4.3 Summary Aggregation

The pipeline aggregates per-seed results into `summary.json`:

```json
{
  "hpobench": {
    "australian": {
      "RandomSearch": {
        "mean_best": -0.957201,
        "std_best": 0.0,
        "mean_time": 0.0177
      },
      ...
    }
  }
}
```

- `mean_best`: Mean of `final_best_cost` across 5 seeds.
- `std_best`: Standard deviation of `final_best_cost` across 5 seeds.
- `mean_time`: Mean total wall-clock time across 5 seeds.

---

## 5. Reproducibility Guarantees

### 5.1 Deterministic Lookups

All three benchmark suites (HPOBench, HPOLib, NAS-Bench-201) use **static lookup
tables**. Given the same configuration, the returned metric is always identical
regardless of hardware, OS, GPU, or execution timing. This eliminates:

- Training variance (no stochastic gradient descent involved)
- Hardware-dependent floating-point differences
- Random initialisation effects

### 5.2 Seed Control

Every optimizer is initialised with an explicit `numpy.random.RandomState(seed)`.
The seed controls:

- Random Search: configuration sampling order
- Optuna TPE: sampler initialisation via `TPESampler(seed=seed)`
- Method A: initial configuration, momentum sampling, accept/reject draws
- Method B: initial random exploration, GP random state, candidate generation
- Method C: all of Method A's sources + local refinement perturbation

### 5.3 Full Reproduction Commands

To reproduce all results from scratch:

```bash
# Install dependencies
pip install simple-hpo-bench numpy optuna scikit-learn scipy

# Run the full pipeline (default: 100 trials × 5 seeds × 11 datasets × 5 optimizers)
python hpobench_benchmark.py

# Run a specific subset
python hpobench_benchmark.py --trials 100 --seeds 0,1,2,3,4 --datasets australian,cifar10
python hpobench_benchmark.py --optimizers RandomSearch,MethodC_Unified

# Generate plots from results
python plot_hpobench.py
```

### 5.4 Verification Checklist

| Property              | Mechanism                                |
|-----------------------|------------------------------------------|
| Same config → same cost | Tabular lookup (no training involved)   |
| Same seed → same trajectory | `np.random.RandomState(seed)` per optimizer |
| Cross-platform consistency | Pure Python + NumPy, no GPU dependencies |
| Result persistence    | Full trajectory JSON per seed per dataset |
| Statistical robustness| 5 independent seeds, mean ± std reported |

---

## 6. Visualisation Pipeline

The `plot_hpobench.py` script generates four publication-quality figures from the
results:

| Figure | Content | File |
|--------|---------|------|
| Fig. 6 | HPOBench 2×2 regret curves (Australian, Blood, Vehicle, Segment) | `fig6_hpobench_regret.png` |
| Fig. 6b | HPOLib 2×2 regret curves (Naval, Parkinsons, Protein, Slice) | `fig6b_hpolib_regret.png` |
| Fig. 7 | NAS-Bench-201 1×3 regret curves (CIFAR-10, CIFAR-100, ImageNet) | `fig7_nasbench201_regret.png` |
| Fig. 8 | Cross-benchmark summary (bar chart + rank heatmap) | `fig8_hpobench_summary.png` |

**Regret curves** plot the mean best-so-far objective cost ± 1 standard deviation
across 5 seeds, as a function of the number of trials. Lower is better for all
curves (accuracy-based benchmarks are negated).

**Summary figure** has two panels:
1. **Bar chart**: Mean final best cost per optimizer, averaged across all 11 benchmarks.
2. **Rank heatmap**: Per-benchmark ranking (1 = best, 5 = worst) with coolwarm colourmap.

---

## 7. Key Design Decisions

### 7.1 Why Index-Based Continuous Relaxation?

Methods A, B, and C were originally designed for continuous hyperparameter spaces.
To apply them to tabular/discrete benchmarks, hyperparameter values are mapped to
a continuous index space:

```
Parameter: batch_size with choices [16, 32, 64, 128]
Index space: [0, 1, 2, 3] (continuous)
Mapping: round(clip(index, 0, 3)) → choices[rounded_index]
```

For GP-based Method B, indices are further normalised to [0, 1] by dividing by
the maximum index. This ensures the RBF kernel operates on a unit-scaled space.

### 7.2 Why 20% Warm-up?

All three physics-inspired methods (A, B, C) allocate the first 20% of the trial
budget to random exploration. This ensures:

1. The GP surrogate (Method B) has sufficient diversity before fitting.
2. The HMC starting point (Methods A, C) is in a reasonably low-cost region.
3. The initial phase is identical to Random Search, providing a fair baseline.

### 7.3 Why Adaptive Step-Size in Method C?

Method A uses a fixed leapfrog step-size ε = 0.5 throughout. In discrete index
spaces where valid configurations are at integer coordinates, a fixed ε can
lead to either:

- **Too many rejections** (ε too large → proposals land far from valid regions)
- **Insufficient exploration** (ε too small → proposals stay near the current point)

Method C's adaptive controller monitors the running acceptance rate and adjusts ε
to maintain the target range [0.4, 0.8], achieving more efficient exploration.
