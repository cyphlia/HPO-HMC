# Validation Tests for *Hamiltonian Hyperparameter Dynamics* Paper

> **Purpose:** Every numerical claim in `HO_main.tex` / `results.tex` must be
> reproducible from the fixed codebase. This document lists the exact test runs
> required, the commands that execute them, the expected values with tolerances,
> and the paper section each test validates.
>
> **Canonical result files referenced below:**
> - `results/harmonic_multiseed/physics_multiseed_summary.json`
> - `results/method_a_fixed_results.json`
> - `results/method_c_fixed_results.json`
> - `results/cnn_results.txt`
> - `results/hpobench/` (per-dataset JSON)

---

## Test Group 1 — Harmonic Oscillator Physics Benchmark

*Paper sections: §7 (Setup), §8 (Results / tab:results), §9 (Discussion), §10 (Conclusion)*

### T1.1 · Method A (Pure HHD) — 5-seed physics run

**What it validates:** `tab:results` rows for Method A; Conclusion wall-time claim ($26.6\,\mathrm{s}$).

```bash
python main.py --task harmonic --method pure --seeds 0 1 2 3 4 --compare
```

| Metric | Expected (mean ± std) | Tolerance |
|---|---|---|
| Best Val MSE | 0.2439 ± 0.1627 | ±5% on mean |
| Landscape MAE | 0.3618 ± 0.1091 | ±5% |
| Landscape RMSE | 0.4872 ± 0.1669 | ±5% |
| R² | 0.9785 ± 0.0158 | R² ≥ 0.95 |
| Wall time | 26.6 ± 1.3 s | ±15% |

**Pass condition:** All metrics within tolerance across all 5 seeds (0–4).

---

### T1.2 · Method B (Hybrid ABBO) — 5-seed physics run

**What it validates:** `tab:results` rows for Method B; Discussion §11.3 timing (99.9 s).

```bash
python main.py --task harmonic --method hybrid --seeds 0 1 2 3 4 --compare
```

| Metric | Expected | Tolerance |
|---|---|---|
| Best Val MSE | 0.0952 ± 0.0051 | ±5% |
| MAE | 0.1028 ± 0.0149 | ±5% |
| RMSE | 0.1403 ± 0.0197 | ±5% |
| R² | 0.9984 ± 0.0005 | R² ≥ 0.997 |
| Wall time | 99.9 ± 1.9 s | ±20% |

---

### T1.3 · Method C (Unified HHD-ABBO) — 5-seed physics run

**What it validates:** `tab:results` rows for Method C; Abstract MSE/R² claims; Discussion §11.3
timing (85.6 s); Abstract "~29× reduction" claim.

```bash
python main.py --task harmonic --method unified --seeds 0 1 2 3 4 --compare
```

| Metric | Expected | Tolerance |
|---|---|---|
| Best Val MSE | 0.00331 ± 0.00014 | ±5% |
| MAE | 0.0208 ± 0.0014 | ±5% |
| RMSE | 0.0270 ± 0.0019 | ±5% |
| R² | 0.99994 ± 0.00001 | R² ≥ 0.9998 |
| Wall time | 85.6 ± 1.4 s | ±20% |
| MSE ratio B/C | ≈ 28.8× | ≥ 25× |

**Post-run sanity check:**
```python
import json
a = json.load(open("results/method_c_fixed_results.json"))
mse_c = sum(r["best_val_loss"] for r in a) / len(a)
mse_b = 0.0952   # from T1.2
print(f"MSE ratio: {mse_b / mse_c:.1f}x")   # should be ~29x
```

---

### T1.4 · Three-Way Comparison Table Reproduction

**What it validates:** That `tab:results` in `results.tex` is self-consistent with underlying JSON files.

```bash
python main.py --task harmonic --compare
```

Expected table (abbreviated):

| Method | Best Val MSE | R² | Time (s) |
|---|---|---|---|
| A | 0.2439 ± 0.1627 | 0.9785 | 26.6 |
| B | 0.0952 ± 0.0051 | 0.9984 | 99.9 |
| C | 0.0033 ± 0.0001 | 0.99994 | 85.6 |

**Pass condition:** All values match to 3 significant figures.

---

## Test Group 2 — CIFAR-10 CNN Classification (Small Slice)

*Paper sections: §7.4 (tab:cnn_setup), §8 (tab:cnn_results), §9.3 (Discussion), §10 (Conclusion)*

### T2.1 · All three methods — 5-seed CNN run

**What it validates:** `tab:cnn_results`; Abstract/Conclusion accuracy claim; Discussion §11.3 CIFAR timing.

```bash
python main.py --task cnn --seeds 101 102 103 104 105
```

| Method | Expected Acc (%) | Wall Time (s) |
|---|---|---|
| A | 30.90 ± 1.59 | 42.0 |
| B | 28.50 ± 2.19 | 31.9 |
| C | 30.60 ± 2.65 | 43.2 |

**Key assertions:**
- Method A accuracy > Method B accuracy (30.90 > 28.50)
- Method A and C confidence intervals overlap (both include ~30%)
- Wall time of B < A ≈ C

**Pass condition:** Accuracies within ±2 pp; wall times within ±25%.

---

## Test Group 3 — Tabular HPO Benchmarks (HPOBench / HPOLib / NAS-Bench-201)

*Paper sections: §7.6 (tab:tabular_protocol), §8 (tab:hpobench_class, tab:hpolib_reg, tab:nasbench_results, tab:hpobench_rankings)*

### T3.1 · Full 11-dataset benchmark — 5 seeds each

**What it validates:** All entries in the ranking table; average rank claims.

```bash
python scripts/hpobench_benchmark.py --seeds 0 1 2 3 4
```

*Estimated runtime: 30–90 min CPU.*

**Expected average ranks** (from tab:hpobench_rankings):

| Method | Expected Avg Rank | Tolerance |
|---|---|---|
| Random | 3.64 | ±0.5 |
| Optuna TPE | 1.36 | ±0.5 |
| Method A | 4.09 | ±0.5 |
| Method B | 2.82 | ±0.5 |
| Method C | 3.09 | ±0.5 |

---

### T3.2 · HPOBench Classification sub-table

**What it validates:** `tab:hpobench_class` (Australian, Blood Transf., Vehicle, Segment).

Expected winners per dataset:
- Australian: Method B best (95.61%), Random search also high
- Blood Transfusion: Optuna best (74.72%), Method C second (74.60%)
- Vehicle: Method A best (97.26%)
- Segment: Random / Method C tied best (97.23%)

---

### T3.3 · HPOLib Regression sub-table

**What it validates:** `tab:hpolib_reg` (Naval, Parkinsons, Protein Str., Slice Loc.).

Expected winners:
- Naval Propulsion: Optuna best (3.1e-5 MSE), C second (3.4e-5)
- Parkinsons: A and C tied best (7.1e-3)
- Protein Structure: Optuna best (2.2e-1), B second (2.2e-1)
- Slice Localization: Method C best (2.1e-4)

---

### T3.4 · NAS-Bench-201 sub-table

**What it validates:** `tab:nasbench_results` (CIFAR-10, CIFAR-100, ImageNet-16).

Expected winners:
- All three datasets: Method B best (91.60%, 73.52%, 46.90%)
- Method C consistently second-best on all three

---

## Test Group 4 — Statistical Significance Tests

*Paper sections: §8.3.4 (subsec:stats), Appendix B (app:experimental_repro)*

### T4.1 · Friedman + Nemenyi tests

**What it validates:** Friedman p = 7.92 × 10⁻⁴; Nemenyi CD = 1.84 at α=0.05.

```bash
python scripts/statistical_tests.py
cat results/stats_summary.json
python evaluation/plot_hpobench.py    # generates cd_diagram.png
```

| Claim | Expected | Pass Condition |
|---|---|---|
| Friedman p | 7.92 × 10⁻⁴ | p < 10⁻³ |
| Nemenyi CD | 1.84 | 1.74 ≤ CD ≤ 1.94 |
| Optuna vs A rank gap | 2.73 > 1.84 — **significant** | gap > CD |
| Optuna vs Random rank gap | 2.27 > 1.84 — **significant** | gap > CD |
| Optuna vs B rank gap | 1.46 < 1.84 — **not significant** | gap < CD |
| Optuna vs C rank gap | 1.73 < 1.84 — **not significant** | gap < CD |

---

## Test Group 5 — Theoretical Property Validation

*Paper sections: §4 (Propositions 1–3), §8.1–§8.2, Appendix A*

### T5.1 · Shadow Hamiltonian Bound (Proposition 1)

**What it validates:** |ΔH| = O(ε²) empirically; acceptance rate stays > 0 for all finite ε.

```bash
python validation/energy_stability.py
```

**Expected outputs:**
- Energy drift plot showing O(ε²) scaling across ε ∈ {0.001, 0.003, 0.005, 0.01, 0.02}
- Relative Hamiltonian drift < 5% over 40 sub-steps (without gradient clipping)

| ε | Max |ΔH| | Expected scaling |
|---|---|---|
| 0.001 | ~10⁻⁵ | baseline |
| 0.003 | ~10⁻⁴ | ≈ 9× baseline |
| 0.01  | ~10⁻³ | ≈ 100× baseline |

**Pass condition:** log|ΔH| vs log(ε) slope ≈ 2.0 ± 0.3.

---

### T5.2 · Detailed Balance of Method C (Proposition 2)

**What it validates:** Markov chain satisfies detailed balance and is ergodic.

```bash
python validation/convergence_ergodicity.py
```

**Expected outputs:**
- Metropolis acceptance rate > 80% during HMC phase at T = 10⁹
- No NaN or Inf in momentum/position states over 60 epochs
- Time-reversal test: leapfrog backward from (q', p') returns within O(ε²) of (q, p)

---

### T5.3 · Gradient Generalisation (Finite Difference Approximation)

**What it validates:** Remark 1 (§4.1.3) — truncation error O(δ) for discrete HP gradients.

```bash
python validation/gradient_generalization.py
```

**Expected:** Finite-difference HP gradient error vs perturbation δ follows O(δ) scaling;
central difference gives O(δ²) improvement.

---

### T5.4 · Energy Conservation Live Validation (Figure energy_conservation_live)

**What it validates:** §8.2 — unclipped +2.35% drift vs clipped +263.4% drift.

| Condition | Expected max drift | Tolerance |
|---|---|---|
| Without gradient clipping | +2.35% over 40 sub-steps | ±1 pp |
| With gradient clipping (clip=10.0) | +263.4% | ±20 pp |

> **Note:** The key qualitative result is that unclipped leapfrog preserves energy
> well (< 5% drift) while aggressive clipping breaks conservation.

---

## Test Group 6 — Ablation Study

*Paper sections: §8.9 (subsec:ablation, tab:ablation)*

### T6.1 · Five ablation variants on harmonic oscillator

**What it validates:** `tab:ablation` — contribution of each Method C component.

```bash
python main.py --task harmonic --method ablation --seeds 0 1 2 3 4
```

| Variant | Expected MSE (↓) | Tabular Avg Rank |
|---|---|---|
| Method C (Full) | 0.00331 ± 0.00014 | 3.09 ± 0.45 |
| C-noAdam | 0.00475 ± 0.00180 | 3.15 ± 0.66 |
| C-noHMC | 0.00326 ± 0.00013 | 2.20 ± 0.37 |
| **C-noLBFGS** | **0.24041 ± 0.04827** | **4.73 ± 0.72** |
| C-noPlateauDetect | 0.00323 ± 0.00015 | 3.00 ± 0.45 |
| C-fixedStep | 0.00355 ± 0.00014 | 3.98 ± 0.61 |

**Key claim to verify:** Removing L-BFGS causes ≥ 30× MSE degradation
(0.24041 / 0.00331 ≈ 72× — well above the conservative "36×" stated in the text).

---

## Test Group 7 — Sensitivity Analysis

*Paper sections: §8.10 (subsec:sensitivity), Appendix C (app:sensitivity)*

### T7.1 · Leapfrog step size sweep (ε)

```bash
python scripts/sensitivity_analysis.py --param epsilon --values 0.001 0.003 0.005 0.01 0.015 0.02
```

**Expected:** Stable MSE (within 2× of optimum) for ε ∈ [0.003, 0.01];
rising rejection rates for ε > 0.015.

### T7.2 · Leapfrog sub-steps sweep (L)

```bash
python scripts/sensitivity_analysis.py --param L --values 1 3 5 10 20
```

**Expected:** Performance flat from L=3 to L=20.

### T7.3 · Plateau patience sweep (P)

```bash
python scripts/sensitivity_analysis.py --param patience --values 2 4 8 16
```

**Expected:** P ∈ [2, 8] comparable; P=16 modest degradation.

### T7.4 · Mass ratio sweep (m_λ / m_θ)

```bash
python scripts/sensitivity_analysis.py --param mass_ratio --values 0.1 0.5 1.0 2.0
```

**Expected:** All mass ratios in [0.1, 2.0] robust; controls speed not final accuracy.

---

## Test Group 8 — Extended Benchmarks

### T8.1 · NUTS vs Fixed-Step Leapfrog (tab:nuts_comparison)

**What it validates:** §8.6 — fixed-step leapfrog achieves MSE 0.206 in 28.4 s; NUTS 0.304 in 142.8 s.

```bash
python main.py --task harmonic --method nuts-comparison --seeds 0 1 2 3 4
```

| Integrator | Best Val MSE | Wall Time (s) | Acceptance Rate |
|---|---|---|---|
| Fixed-Step Leapfrog | 0.206 ± 0.012 | 28.4 ± 1.1 | 84.2% ± 3.1% |
| NUTS | 0.304 ± 0.025 | 142.8 ± 5.4 | 91.5% ± 2.4% |

**Key claim:** Fixed-step leapfrog ≥ 5× faster than NUTS with better MSE.

---

### T8.2 · Fashion-MNIST Deep MLP (tab:fmnist_results)

```bash
python main.py --task fashion-mnist --method unified --seeds 0 1 2 3 4
```

| Method | Val Acc (%) | Wall Time (s) |
|---|---|---|
| Default Adam | 84.43 ± 0.41 | 32.9 |
| Method C | 85.01 ± 0.12 | 85.9 |

**Key claims:** Method C improves accuracy by +0.58 pp; variance is 3.4× lower (0.12 vs 0.41 std).

---

### T8.3 · Real-World Clinical Benchmarks

```bash
python main.py --task clinical --dataset breast-cancer --seeds 0 1 2 3 4
python main.py --task clinical --dataset diabetes --seeds 0 1 2 3 4
```

**Breast Cancer key claim:** Method C achieves statistical parity with 20-trial Optuna TPE
at **13–19× wall-clock speedup** (1.6 s vs 22.2 s).

| Method | Test AUROC | Positive Recall | Time (s) |
|---|---|---|---|
| Default Adam | 0.9959 ± 0.0010 | 0.9291 ± 0.0329 | 1.2 |
| Method C | 0.9949 ± 0.0028 | 0.9241 ± 0.0179 | 1.6 |
| Optuna TPE | 0.9963 ± 0.0031 | 0.9384 ± 0.0322 | 22.2 |

---

### T8.4 · Optimizer's Curse / Validation-Overfitting Gap (tab:optimizers_curse)

```bash
python main.py --task clinical --dataset diabetes --method all --compute-val-gap --seeds 0 1 2 3 4
```

**Expected validation AUROC gaps (val − test):**

| Method | Expected Gap |
|---|---|
| Default Adam | −0.003 ± 0.015 (no overfit) |
| Method C | +0.008 ± 0.016 (mild) |
| Optuna TPE | +0.025 ± 0.018 (3× larger than C) |
| Random Search | +0.032 ± 0.022 (4× larger than C) |

---

## Quick Smoke-Test Suite (< 5 minutes)

For rapid pre-commit / CI validation:

```bash
# 1. Physics sanity check (1 seed only)
python main.py --task harmonic --method unified --seeds 0
# Expected: MSE < 0.01, R2 > 0.999

# 2. Energy conservation check
python validation/energy_stability.py --quick
# Expected: drift < 5% over 10 sub-steps

# 3. Statistical tests (from cached results)
python scripts/statistical_tests.py --from-cache
# Expected: p < 0.001, CD ≈ 1.84

# 4. Shadow Hamiltonian scaling
python validation/validate.py --theorem 1 --quick
# Expected: slope ≈ 2.0 in log-log plot
```

---

## Reproducibility Checklist

Before claiming full reproducibility of the paper, confirm all pass:

- [ ] **T1.1** — Method A 5-seed HO: MSE 0.2439, R² 0.9785, wall 26.6 s
- [ ] **T1.2** — Method B 5-seed HO: MSE 0.0952, R² 0.9984, wall 99.9 s
- [ ] **T1.3** — Method C 5-seed HO: MSE 0.00331, R² 0.99994, wall 85.6 s; ratio ≥ 25×
- [ ] **T1.4** — Three-way table matches to 3 significant figures
- [ ] **T2.1** — CNN: A=30.90%, B=28.50%, C=30.60% (within ±2 pp)
- [ ] **T3.1** — 11-dataset average ranks: Optuna=1.36, B=2.82, C=3.09, RS=3.64, A=4.09 (±0.5)
- [ ] **T4.1** — Friedman p < 10⁻³; CD ∈ [1.74, 1.94]; Optuna significantly beats A and Random only
- [ ] **T5.1** — Shadow Hamiltonian slope ≈ 2.0 ± 0.3 in log-log plot
- [ ] **T5.2** — Detailed balance confirmed; no NaN/Inf crashes over 60 epochs
- [ ] **T6.1** — C-noLBFGS MSE ≥ 30× worse than full Method C
- [ ] **T7.1–T7.4** — Sensitivity plots match Appendix C narrative
- [ ] **T8.1** — Fixed leapfrog ≥ 5× faster than NUTS with lower MSE
- [ ] **T8.3** — Clinical benchmarks: 13–19× speedup vs Optuna TPE

---

## Environment Requirements

```bash
pip install simple-hpo-bench numpy optuna scikit-learn scipy scikit-posthocs autorank torch
```

| Requirement | Value |
|---|---|
| Python | 3.9+ |
| PyTorch | ≥ 2.0 (CPU) |
| Seeds (physics/tabular) | 0–4 |
| Seeds (CNN) | 101–105 |
| Device | CPU only (all wall-time claims are CPU-based) |
| Full suite estimated runtime | 2–4 hours |
| Smoke test estimated runtime | < 5 minutes |

---

## Full Reproducibility Run

```bash
pip install simple-hpo-bench numpy optuna scikit-learn scipy scikit-posthocs autorank
python scripts/hpobench_benchmark.py --seeds 0 1 2 3 4
python evaluation/plot_hpobench.py
python scripts/statistical_tests.py
python main.py --task harmonic --compare --seeds 0 1 2 3 4
python main.py --task cnn --seeds 101 102 103 104 105
```
