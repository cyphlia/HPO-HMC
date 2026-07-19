# Real-World Showcase: HHD-ABBO on Breast Cancer Diagnosis

`scripts/real_world_breast_cancer.py`

## Why this task

Every benchmark previously in this repo is either a synthetic Hamiltonian
system (harmonic oscillator, Hénon-Heiles, double-well) or an academic
image-classification set (Fashion-MNIST, CIFAR-10). None of them show the
algorithm doing the job it's actually pitched for in the paper: tuning a
model for someone with a real, small, high-stakes dataset who can't afford
a large HPO budget.

The Wisconsin Diagnostic Breast Cancer dataset (569 patients, 30 real-valued
features from digitized FNA images, binary malignant/benign target) is a
believable stand-in for that use case, and it stresses HHD-ABBO in ways the
rest of the repo doesn't:

- **Classification, not regression.** `BCEWithLogitsLoss`, not `MSELoss`.
  The finite-difference HP-gradient machinery and the L-BFGS closures all
  had to be re-verified against this loss, not just re-imported.
- **A clinically meaningful metric, not just loss.** Best-checkpoint
  selection is done on validation ROC-AUC, and the report also tracks
  **recall on the malignant class** specifically, because a missed cancer
  is a far more costly error than a false alarm — accuracy alone would
  hide that asymmetry.
- **Small-N tabular data.** 569 rows total, ~340 for training. This is a
  completely different regime from the 2,500-sample synthetic systems or
  the thousands of images in the CNN/Fashion-MNIST benchmarks.

## Setup

- 60/20/20 stratified train/val/test split, `StandardScaler` fit on train only.
- 5-D search space: `log_lr, dropout, log_wd, n_hidden, n_layers`.
- Baselines get an equal-opportunity budget: 20 trials each (Random Search,
  Optuna TPE), 40 Adam epochs per trial.
- Method C: 8-epoch Adam warmup + 15-epoch HMC co-evolution (3 Adam
  micro-epochs each) + plateau-triggered/final L-BFGS polish — the same
  three-phase curriculum used everywhere else in this repo, unmodified in
  structure.
- 5 seeds, held-out test set scored only after HP selection is finalized on
  the validation split (never used for model choice).

## Results (mean ± std over 5 seeds, held-out test set)

| Method | Test AUROC | Test Accuracy | Malignant Recall | Time (s) |
|---|---|---|---|---|
| Default Adam | 0.9969 ± 0.0044 | 0.9702 ± 0.0153 | 0.9441 ± 0.0455 | 1.2 |
| Random Search (20 trials) | 0.9988 ± 0.0019 | 0.9754 ± 0.0129 | 0.9674 ± 0.0315 | 21.5 |
| Optuna TPE (20 trials) | 0.9972 ± 0.0036 | 0.9825 ± 0.0096 | 0.9721 ± 0.0271 | 22.9 |
| **Method C (HHD-ABBO)** | 0.9962 ± 0.0045 | 0.9737 ± 0.0184 | 0.9487 ± 0.0558 | **1.5** |

(`plots/breast_cancer_comparison.png` gives the per-seed boxplots behind
these numbers; `results/breast_cancer/{results,summary}.json` has the raw
per-seed data.)

## Honest read of these numbers

This dataset is easy — every method clears 0.99 AUROC, so there is very
little headroom for any HPO method to differentiate on quality here, and
that's worth saying plainly rather than dressing up small differences as a
win. Within that ceiling:

- **Optuna TPE and Random Search have a small, consistent edge in raw
  quality** (higher mean AUROC and malignant recall) over both Default Adam
  and Method C. This is the same finding already documented in the main
  paper for general tabular HPO — Optuna TPE is the strongest method on
  tabular benchmarks, and this real-world dataset doesn't overturn that.
- **Method C is essentially tied with a single untuned Adam run** on this
  task, and slightly behind the 20-trial search baselines. It is not the
  most accurate option here.
- **Method C's actual advantage in this experiment is wall-clock cost.**
  It reaches comparable AUROC to Optuna TPE (0.9962 vs 0.9972) in **~15x
  less time** (1.5s vs 22.9s), because it evaluates one continuously-evolving
  model trajectory instead of training 20 independent models from scratch.
  For a practitioner with a small clinical dataset and no compute budget for
  a 20-trial search, that speed/quality tradeoff — not raw leaderboard
  position — is the actual value proposition demonstrated here.

This mirrors, on real data, exactly the conclusion the paper's tabular
HPOBench results already report: don't claim Method C beats TPE at general
tabular HPO, because on this evidence it doesn't. What it does show, and
what wasn't previously demonstrated anywhere in this repo, is that the
three-phase curriculum degrades gracefully — not catastrophically — outside
its synthetic-physics comfort zone, and does so at a fraction of the
compute cost of a from-scratch multi-trial search.

## Reproducing

```bash
python scripts/real_world_breast_cancer.py --seeds 0,1,2,3,4 --trials 20 --hmc-epochs 15
```

Outputs:
- `results/breast_cancer/results.json` — raw per-seed results
- `results/breast_cancer/summary.json` — aggregated summary stats
- `plots/breast_cancer_comparison.png` — AUROC / malignant-recall boxplots
