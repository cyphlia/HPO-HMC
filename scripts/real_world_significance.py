"""
Statistical significance testing for the real-world showcase (breast_cancer
and diabetes datasets), following the Demsar (2006) procedure already used
elsewhere in this repo for the HPOBench tabular benchmarks:

  1. Friedman test (blocked by seed) -- is there a significant difference
     in ranks among the 4 methods at all?
  2. Nemenyi post-hoc pairwise test + critical difference, if the Friedman
     test is significant.

This is run separately per dataset (blocking by seed within a dataset),
rather than pooling seeds across datasets into one Friedman test, because
the seeds are repeated-measures blocks within a dataset, not comparable
units across datasets with different sample sizes and difficulty.

Usage:
  python scripts/real_world_significance.py
"""
import json
import os
import sys

import numpy as np
import scipy.stats as ss

try:
    import scikit_posthocs as sp
    _POSTHOCS = True
except ImportError:
    _POSTHOCS = False

METHODS = ["Default Adam", "Random Search", "Optuna TPE", "Method C (HHD-ABBO)"]
METRICS = ["auroc", "positive_recall"]


def load_matrix(dataset: str, metric: str):
    path = os.path.join("results", dataset, "results.json")
    results = json.load(open(path))
    seeds = sorted(set(r["seed"] for r in results))
    # rows = seeds, cols = methods (Friedman requires a complete block design)
    mat = np.full((len(seeds), len(METHODS)), np.nan)
    for r in results:
        if not r.get("test_metrics"):
            continue
        i = seeds.index(r["seed"])
        j = METHODS.index(r["method"])
        mat[i, j] = r["test_metrics"][metric]
    if np.isnan(mat).any():
        missing = np.argwhere(np.isnan(mat))
        print(f"    WARNING: incomplete block design, missing cells: {missing.tolist()}")
    return mat, seeds


def run_for(dataset: str, metric: str):
    print(f"\n{'=' * 70}\n  {dataset} | metric = {metric}\n{'=' * 70}")
    mat, seeds = load_matrix(dataset, metric)
    print(f"  seeds (blocks): {seeds}")
    for j, m in enumerate(METHODS):
        col = mat[:, j]
        print(f"    {m:<24} mean={np.nanmean(col):.4f}  std={np.nanstd(col):.4f}")

    if mat.shape[0] < 3:
        print("  Too few seeds/blocks for a meaningful Friedman test (need >=3). Skipping.")
        return None

    # scipy's friedmanchisquare wants one array per treatment (method)
    stat, p = ss.friedmanchisquare(*[mat[:, j] for j in range(mat.shape[1])])
    print(f"\n  Friedman chi2 = {stat:.4f}, p = {p:.5f}")
    alpha = 0.05
    if p >= alpha:
        print(f"  Not significant at alpha={alpha}: cannot reject H0 that all methods "
              f"have the same rank distribution. Any observed mean differences are "
              f"consistent with noise at this sample size.")
        return {"dataset": dataset, "metric": metric, "friedman_chi2": float(stat),
                "friedman_p": float(p), "significant": False, "n_blocks": mat.shape[0]}

    print(f"  Significant at alpha={alpha}: proceeding to Nemenyi post-hoc test.")
    if _POSTHOCS:
        nemenyi = sp.posthoc_nemenyi_friedman(mat)
        nemenyi.columns = METHODS
        nemenyi.index = METHODS
        print("\n  Nemenyi pairwise p-values:")
        print(nemenyi.round(4).to_string())
        sig_pairs = []
        for i in range(len(METHODS)):
            for j in range(i + 1, len(METHODS)):
                pv = nemenyi.iloc[i, j]
                if pv < alpha:
                    sig_pairs.append((METHODS[i], METHODS[j], float(pv)))
        if sig_pairs:
            print("\n  Significant pairwise differences (p < 0.05):")
            for a, b, pv in sig_pairs:
                print(f"    {a} vs {b}: p={pv:.4f}")
        else:
            print("\n  No individual pairwise comparison survives Nemenyi correction, "
                  "despite the omnibus Friedman test being significant. This can happen "
                  "-- the Nemenyi test is conservative. Treat pairwise claims cautiously.")
        return {"dataset": dataset, "metric": metric, "friedman_chi2": float(stat),
                "friedman_p": float(p), "significant": True, "n_blocks": mat.shape[0],
                "nemenyi_matrix": nemenyi.round(4).to_dict(),
                "significant_pairs": sig_pairs}
    else:
        print("  scikit-posthocs not available; skipping Nemenyi post-hoc.")
        return {"dataset": dataset, "metric": metric, "friedman_chi2": float(stat),
                "friedman_p": float(p), "significant": True, "n_blocks": mat.shape[0]}


def main():
    all_out = {}
    for dataset in ["breast_cancer", "diabetes"]:
        path = os.path.join("results", dataset, "results.json")
        if not os.path.exists(path):
            print(f"[SKIP] {path} not found")
            continue
        for metric in METRICS:
            out = run_for(dataset, metric)
            if out is not None:
                all_out[f"{dataset}_{metric}"] = out

    out_path = os.path.join("results", "real_world_significance.json")
    with open(out_path, "w") as f:
        json.dump(all_out, f, indent=2, default=str)
    print(f"\n\nSaved full results to {out_path}")


if __name__ == "__main__":
    main()
