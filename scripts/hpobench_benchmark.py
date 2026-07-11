#!/usr/bin/env python
"""
Comprehensive HPOBench Benchmarking Pipeline
=============================================

Tests 5 optimizers across 11 datasets from HPOBench, HPOLib, and NAS-Bench-201
using the simple-hpo-bench library.

Optimizers:
    1. RandomSearch          – uniform random sampling
    2. OptunaTPE             – Tree-structured Parzen Estimator via Optuna
    3. MethodA_HHD           – Hamiltonian Monte Carlo adapted for discrete spaces
    4. MethodB_ABBO          – Bayesian Optimization with GP-EI over index space
    5. MethodC_Unified       – Three-phase curriculum (random → HMC → local refine)

Usage:
    python hpobench_benchmark.py
    python hpobench_benchmark.py --trials 50 --seeds 0,1,2 --datasets australian,cifar10
    python hpobench_benchmark.py --optimizers RandomSearch,OptunaTPE
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Graceful dependency checks
# ---------------------------------------------------------------------------

try:
    from hpo_benchmarks import HPOBench, HPOLib, NASBench201
except ImportError:
    print(
        "ERROR: 'simple-hpo-bench' is not installed.\n"
        "       Install it with:  pip install simple-hpo-bench"
    )
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print(
        "ERROR: 'numpy' is not installed.\n"
        "       Install it with:  pip install numpy"
    )
    sys.exit(1)

_OPTUNA_AVAILABLE = True
try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    _OPTUNA_AVAILABLE = False

_SKLEARN_AVAILABLE = True
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel
except ImportError:
    _SKLEARN_AVAILABLE = False

_HPBANDSTER_AVAILABLE = True
try:
    import hpbandster.core.nameserver as hpns
    from hpbandster.core.worker import Worker as HPB_Worker
    from hpbandster.optimizers import BOHB as HPB_BOHB
except ImportError:
    _HPBANDSTER_AVAILABLE = False

_SMAC_AVAILABLE = True
try:
    from smac import HyperparameterOptimizationFacade as SMAC_Facade
    from smac import Scenario as SMAC_Scenario
except ImportError:
    _SMAC_AVAILABLE = False

_CONFIGSPACE_AVAILABLE = True
try:
    import ConfigSpace as CS
except ImportError:
    _CONFIGSPACE_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_TRIALS = 100
SEEDS = [0, 1, 2, 3, 4]

DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "hpobench": {
        "class": HPOBench,
        "datasets": ["australian", "blood_transfusion", "vehicle", "segment"],
    },
    "hpolib": {
        "class": HPOLib,
        "datasets": [
            "naval_propulsion",
            "parkinsons_telemonitoring",
            "protein_structure",
            "slice_localization",
        ],
    },
    "nasbench201": {
        "class": NASBench201,
        "datasets": ["cifar10", "cifar100", "imagenet"],
    },
}

ALL_OPTIMIZERS = [
    "RandomSearch",
    "OptunaTPE",
    "MethodA_HHD",
    "MethodB_ABBO",
    "MethodC_Unified",
    "BOHB",
    "SMAC3",
]

RESULTS_DIR = Path("results_hpobench")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_evaluate(bench, params: dict) -> Optional[dict]:
    """Evaluate a configuration, returning None on failure."""
    try:
        return bench(params)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"Benchmark evaluation failed for {params}: {exc}")
        return None


def _to_cost(result: Optional[dict], metric_name: str, direction: str) -> float:
    """Convert a raw benchmark result to a minimisation cost."""
    if result is None:
        return float("inf")
    val = result.get(metric_name, float("inf"))
    if direction == "maximize":
        return -val
    return val


def _random_config(search_space: dict, rng: np.random.RandomState) -> dict:
    """Sample a uniformly random configuration."""
    return {
        name: choices[rng.randint(0, len(choices))]
        for name, choices in search_space.items()
    }


def _indices_to_config(indices: np.ndarray, param_names: list, search_space: dict) -> dict:
    """Map continuous index array → valid discrete configuration."""
    config: dict = {}
    for i, name in enumerate(param_names):
        choices = search_space[name]
        idx = int(np.clip(np.round(indices[i]), 0, len(choices) - 1))
        config[name] = choices[idx]
    return config


def _config_to_indices(config: dict, param_names: list, search_space: dict) -> np.ndarray:
    """Map a discrete configuration → index array."""
    indices = np.zeros(len(param_names), dtype=float)
    for i, name in enumerate(param_names):
        choices = search_space[name]
        try:
            indices[i] = float(choices.index(config[name]))
        except ValueError:
            indices[i] = 0.0
    return indices


# ---------------------------------------------------------------------------
# Optimizer implementations
# ---------------------------------------------------------------------------


def run_random_search(
    bench,
    search_space: dict,
    metric_name: str,
    direction: str,
    max_trials: int,
    seed: int,
) -> List[dict]:
    """Optimizer 1: Uniform random search."""
    rng = np.random.RandomState(seed)
    trajectory: List[dict] = []
    best_cost = float("inf")

    for t in range(1, max_trials + 1):
        cfg = _random_config(search_space, rng)
        t0 = time.perf_counter()
        result = _safe_evaluate(bench, cfg)
        elapsed = time.perf_counter() - t0
        cost = _to_cost(result, metric_name, direction)
        best_cost = min(best_cost, cost)
        trajectory.append(
            {
                "trial": t,
                "cost": cost,
                "best_cost": best_cost,
                "config": cfg,
                "time": round(elapsed, 6),
            }
        )
    return trajectory


def run_optuna_tpe(
    bench,
    search_space: dict,
    metric_name: str,
    direction: str,
    max_trials: int,
    seed: int,
) -> List[dict]:
    """Optimizer 2: Optuna TPE via index-based suggest_int."""
    if not _OPTUNA_AVAILABLE:
        raise RuntimeError(
            "Optuna is not installed. Install with:  pip install optuna"
        )

    param_names = list(search_space.keys())
    trajectory: List[dict] = []
    best_cost = float("inf")

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_cost
        cfg: dict = {}
        for name in param_names:
            choices = search_space[name]
            idx = trial.suggest_int(name + "_index", 0, len(choices) - 1)
            cfg[name] = choices[idx]

        t0 = time.perf_counter()
        result = _safe_evaluate(bench, cfg)
        elapsed = time.perf_counter() - t0
        cost = _to_cost(result, metric_name, direction)
        best_cost = min(best_cost, cost)
        trajectory.append(
            {
                "trial": len(trajectory) + 1,
                "cost": cost,
                "best_cost": best_cost,
                "config": cfg,
                "time": round(elapsed, 6),
            }
        )
        return cost

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=max_trials, show_progress_bar=False)
    return trajectory


def run_method_a_hhd(
    bench,
    search_space: dict,
    metric_name: str,
    direction: str,
    max_trials: int,
    seed: int,
) -> List[dict]:
    """Optimizer 3: Hamiltonian Monte Carlo adapted for discrete (index) spaces."""
    rng = np.random.RandomState(seed)
    param_names = list(search_space.keys())
    dims = len(param_names)
    bounds_high = np.array([len(search_space[n]) - 1 for n in param_names], dtype=float)

    trajectory: List[dict] = []
    best_cost = float("inf")
    best_config: dict = {}

    warmup_trials = max(1, int(0.20 * max_trials))
    epsilon = 0.5
    leapfrog_steps = 3

    # Initialise with a random config
    current_indices = np.array(
        [rng.randint(0, len(search_space[n])) for n in param_names], dtype=float
    )
    current_cfg = _indices_to_config(current_indices, param_names, search_space)
    t0 = time.perf_counter()
    result = _safe_evaluate(bench, current_cfg)
    elapsed = time.perf_counter() - t0
    current_cost = _to_cost(result, metric_name, direction)
    best_cost = current_cost
    best_config = deepcopy(current_cfg)
    trajectory.append(
        {
            "trial": 1,
            "cost": current_cost,
            "best_cost": best_cost,
            "config": current_cfg,
            "time": round(elapsed, 6),
        }
    )

    def _cost_at(indices: np.ndarray) -> float:
        cfg = _indices_to_config(indices, param_names, search_space)
        res = _safe_evaluate(bench, cfg)
        return _to_cost(res, metric_name, direction)

    def _finite_diff_grad(pos: np.ndarray, cost_at_pos: float) -> np.ndarray:
        grad = np.zeros(dims)
        delta = 1.0
        for d in range(dims):
            pos_plus = pos.copy()
            pos_plus[d] += delta
            pos_plus[d] = np.clip(pos_plus[d], 0, bounds_high[d])
            c_plus = _cost_at(pos_plus)
            grad[d] = (c_plus - cost_at_pos) / delta
        return grad

    for t in range(2, max_trials + 1):
        t0 = time.perf_counter()

        if t <= warmup_trials:
            # Warm-up: random sampling
            cfg = _random_config(search_space, rng)
            result = _safe_evaluate(bench, cfg)
            cost = _to_cost(result, metric_name, direction)
            current_indices = _config_to_indices(cfg, param_names, search_space)
            current_cost = cost
        else:
            # Transition from warm-up to HMC: start from the best configuration found so far
            if t == warmup_trials + 1:
                current_cfg = deepcopy(best_config)
                current_indices = _config_to_indices(current_cfg, param_names, search_space)
                current_cost = best_cost

            # HMC leapfrog
            momentum = rng.randn(dims)
            pos = current_indices.copy()
            pos_cost = current_cost

            # Half-step momentum
            grad = _finite_diff_grad(pos, pos_cost)
            momentum = momentum - 0.5 * epsilon * grad

            for _step in range(leapfrog_steps):
                # Full-step position
                pos = pos + epsilon * momentum
                pos = np.clip(pos, 0, bounds_high)

                # Recompute gradient at new position
                pos_cost_new = _cost_at(pos)
                grad = _finite_diff_grad(pos, pos_cost_new)

                # Full-step momentum (except last which is half)
                if _step < leapfrog_steps - 1:
                    momentum = momentum - epsilon * grad
                else:
                    momentum = momentum - 0.5 * epsilon * grad

            proposed_cost = _cost_at(pos)

            # Metropolis-Hastings acceptance
            delta_cost = proposed_cost - current_cost
            if delta_cost < 0 or rng.rand() < math.exp(-max(delta_cost, -700)):
                current_indices = pos
                current_cost = proposed_cost
                cfg = _indices_to_config(pos, param_names, search_space)
                cost = proposed_cost
            else:
                cfg = _indices_to_config(current_indices, param_names, search_space)
                cost = current_cost

        elapsed = time.perf_counter() - t0
        best_cost = min(best_cost, cost)
        if cost <= best_cost:
            best_config = deepcopy(cfg)
        trajectory.append(
            {
                "trial": t,
                "cost": cost,
                "best_cost": best_cost,
                "config": cfg,
                "time": round(elapsed, 6),
            }
        )

    return trajectory


def run_method_b_abbo(
    bench,
    search_space: dict,
    metric_name: str,
    direction: str,
    max_trials: int,
    seed: int,
) -> List[dict]:
    """Optimizer 4: Bayesian Optimisation with GP-EI over normalised index space."""
    if not _SKLEARN_AVAILABLE:
        raise RuntimeError(
            "scikit-learn is not installed. Install with:  pip install scikit-learn"
        )

    rng = np.random.RandomState(seed)
    param_names = list(search_space.keys())
    dims = len(param_names)
    bounds_high = np.array([len(search_space[n]) - 1 for n in param_names], dtype=float)

    trajectory: List[dict] = []
    best_cost = float("inf")

    warmup_trials = max(1, int(0.20 * max_trials))
    n_candidates = 1000

    X_observed: List[np.ndarray] = []
    y_observed: List[float] = []

    def _normalise(x: np.ndarray) -> np.ndarray:
        """Normalise indices to [0, 1]."""
        with np.errstate(divide="ignore", invalid="ignore"):
            normed = np.where(bounds_high > 0, x / bounds_high, 0.0)
        return normed

    def _expected_improvement(
        X_cand: np.ndarray, gp: GaussianProcessRegressor, y_best: float
    ) -> np.ndarray:
        mu, sigma = gp.predict(X_cand, return_std=True)
        sigma = np.maximum(sigma, 1e-9)
        imp = y_best - mu
        Z = imp / sigma
        from scipy.stats import norm

        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        return ei

    for t in range(1, max_trials + 1):
        t0 = time.perf_counter()

        if t <= warmup_trials:
            # Random exploration
            raw_indices = np.array(
                [rng.randint(0, len(search_space[n])) for n in param_names],
                dtype=float,
            )
        else:
            # Fit GP and optimise EI
            X_train = np.array([_normalise(x) for x in X_observed])
            y_train = np.array(y_observed)

            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
            gp = GaussianProcessRegressor(
                kernel=kernel, alpha=1e-6, normalize_y=True, random_state=seed
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp.fit(X_train, y_train)

            # Generate candidates
            cand_raw = np.column_stack(
                [
                    rng.randint(0, max(1, len(search_space[n])), size=n_candidates).astype(float)
                    for n in param_names
                ]
            )
            cand_normed = np.array([_normalise(c) for c in cand_raw])

            try:
                ei = _expected_improvement(cand_normed, gp, min(y_observed))
                best_cand_idx = int(np.argmax(ei))
            except Exception:
                best_cand_idx = rng.randint(0, n_candidates)

            raw_indices = cand_raw[best_cand_idx]

        cfg = _indices_to_config(raw_indices, param_names, search_space)
        result = _safe_evaluate(bench, cfg)
        elapsed = time.perf_counter() - t0
        cost = _to_cost(result, metric_name, direction)

        X_observed.append(raw_indices.copy())
        y_observed.append(cost)

        best_cost = min(best_cost, cost)
        trajectory.append(
            {
                "trial": t,
                "cost": cost,
                "best_cost": best_cost,
                "config": cfg,
                "time": round(elapsed, 6),
            }
        )

    return trajectory


def run_method_c_unified(
    bench,
    search_space: dict,
    metric_name: str,
    direction: str,
    max_trials: int,
    seed: int,
    use_adam_warmup: bool = True,
    use_hmc: bool = True,
    use_lbfgs: bool = True,
    use_adaptive_step: bool = True,
) -> List[dict]:
    """Optimizer 5: Three-phase curriculum (random → adaptive HMC → local refine)."""
    rng = np.random.RandomState(seed)
    param_names = list(search_space.keys())
    dims = len(param_names)
    bounds_high = np.array([len(search_space[n]) - 1 for n in param_names], dtype=float)

    phase1_end = max(1, int(0.20 * max_trials)) if use_adam_warmup else 0
    if not use_lbfgs:
        phase2_end = max_trials if use_hmc else phase1_end
    else:
        phase2_end = max(phase1_end + 1, int(0.80 * max_trials)) if use_hmc else phase1_end

    trajectory: List[dict] = []
    best_cost = float("inf")
    best_config: dict = {}

    # HMC state
    epsilon = 0.5
    leapfrog_steps = 3
    accept_count = 0
    hmc_trial_count = 0

    current_indices = np.array(
        [rng.randint(0, len(search_space[n])) for n in param_names], dtype=float
    )
    current_cfg = _indices_to_config(current_indices, param_names, search_space)
    t0 = time.perf_counter()
    result = _safe_evaluate(bench, current_cfg)
    elapsed = time.perf_counter() - t0
    current_cost = _to_cost(result, metric_name, direction)
    best_cost = current_cost
    best_config = deepcopy(current_cfg)
    trajectory.append(
        {
            "trial": 1,
            "cost": current_cost,
            "best_cost": best_cost,
            "config": current_cfg,
            "time": round(elapsed, 6),
        }
    )

    def _cost_at(indices: np.ndarray) -> float:
        cfg = _indices_to_config(indices, param_names, search_space)
        res = _safe_evaluate(bench, cfg)
        return _to_cost(res, metric_name, direction)

    def _finite_diff_grad(pos: np.ndarray, cost_at_pos: float) -> np.ndarray:
        grad = np.zeros(dims)
        delta = 1.0
        for d in range(dims):
            pos_plus = pos.copy()
            pos_plus[d] += delta
            pos_plus[d] = np.clip(pos_plus[d], 0, bounds_high[d])
            c_plus = _cost_at(pos_plus)
            grad[d] = (c_plus - cost_at_pos) / delta
        return grad

    for t in range(2, max_trials + 1):
        t0 = time.perf_counter()

        if t <= phase1_end:
            # Phase 1: Random warm-up
            cfg = _random_config(search_space, rng)
            result = _safe_evaluate(bench, cfg)
            cost = _to_cost(result, metric_name, direction)
            current_indices = _config_to_indices(cfg, param_names, search_space)
            current_cost = cost

        elif t <= phase2_end:
            # Transition from warm-up to HMC: start from the best configuration found so far
            if t == phase1_end + 1:
                current_cfg = deepcopy(best_config)
                current_indices = _config_to_indices(current_cfg, param_names, search_space)
                current_cost = best_cost

            # Phase 2: Adaptive HMC
            hmc_trial_count += 1
            momentum = rng.randn(dims)
            pos = current_indices.copy()

            # Half-step momentum
            grad = _finite_diff_grad(pos, current_cost)
            momentum = momentum - 0.5 * epsilon * grad

            for _step in range(leapfrog_steps):
                pos = pos + epsilon * momentum
                pos = np.clip(pos, 0, bounds_high)
                pos_c = _cost_at(pos)
                grad = _finite_diff_grad(pos, pos_c)
                if _step < leapfrog_steps - 1:
                    momentum = momentum - epsilon * grad
                else:
                    momentum = momentum - 0.5 * epsilon * grad

            proposed_cost = _cost_at(pos)
            delta_cost = proposed_cost - current_cost
            accepted = delta_cost < 0 or rng.rand() < math.exp(-max(delta_cost, -700))

            if accepted:
                current_indices = pos
                current_cost = proposed_cost
                cfg = _indices_to_config(pos, param_names, search_space)
                cost = proposed_cost
                accept_count += 1
            else:
                cfg = _indices_to_config(current_indices, param_names, search_space)
                cost = current_cost

            # Adaptive step size
            if use_adaptive_step:
                acceptance_rate = accept_count / hmc_trial_count
                if acceptance_rate > 0.8:
                    epsilon *= 1.05
                elif acceptance_rate < 0.4:
                    epsilon *= 0.95
                epsilon = max(0.01, min(epsilon, 5.0))  # clamp

        else:
            # Phase 3: Local refinement around best configuration
            if use_lbfgs:
                best_indices = _config_to_indices(best_config, param_names, search_space)
                perturbed = best_indices.copy()

                # Perturb random subset of parameters by ±1
                n_perturb = max(1, rng.randint(1, dims + 1))
                perturb_dims = rng.choice(dims, size=n_perturb, replace=False)
                for d in perturb_dims:
                    delta = rng.choice([-1, 1])
                    perturbed[d] = np.clip(perturbed[d] + delta, 0, bounds_high[d])

                cfg = _indices_to_config(perturbed, param_names, search_space)
                result = _safe_evaluate(bench, cfg)
                cost = _to_cost(result, metric_name, direction)
            else:
                # Fallback to random configuration
                cfg = _random_config(search_space, rng)
                result = _safe_evaluate(bench, cfg)
                cost = _to_cost(result, metric_name, direction)

        elapsed = time.perf_counter() - t0
        best_cost = min(best_cost, cost)
        if cost <= best_cost:
            best_config = deepcopy(cfg)
        trajectory.append(
            {
                "trial": t,
                "cost": cost,
                "best_cost": best_cost,
                "config": cfg,
                "time": round(elapsed, 6),
            }
        )

    return trajectory

def run_bohb(
    bench,
    search_space: dict,
    metric_name: str,
    direction: str,
    max_trials: int,
    seed: int,
) -> List[dict]:
    """Optimizer: BOHB-style KDE-based Bayesian Optimisation (sequential, no Pyro4).

    Implements the core BOHB idea (Falkner et al. 2018):
      - Random warm-up phase
      - Fit two KDE models: l(x) on the good configs (top quantile) and
        g(x) on the bad configs (bottom quantile)
      - Sample candidates, pick the one with highest l(x)/g(x) ratio
    This is the single-fidelity variant (budget=max), appropriate for
    tabular benchmarks where evaluation cost is constant.
    """
    rng = np.random.RandomState(seed)
    param_names = list(search_space.keys())
    # Convert all choice lists to native Python types to avoid serialization issues
    native_space = {
        name: [_make_json_serialisable(v) for v in choices]
        for name, choices in search_space.items()
    }

    trajectory: List[dict] = []
    best_cost = float("inf")

    # --- Warm-up: pure random sampling ---
    n_warmup = max(4, int(0.20 * max_trials))

    for t in range(1, n_warmup + 1):
        cfg = {name: native_space[name][rng.randint(len(native_space[name]))]
               for name in param_names}
        t0 = time.perf_counter()
        result = _safe_evaluate(bench, cfg)
        elapsed = time.perf_counter() - t0
        cost = float(_to_cost(result, metric_name, direction))
        best_cost = min(best_cost, cost)
        trajectory.append({
            "trial": t, "cost": cost, "best_cost": best_cost,
            "config": cfg, "time": round(elapsed, 6),
        })

    # --- KDE-BO phase ---
    # We represent each config as an index vector for KDE purposes
    n_candidates = 64
    good_quantile = 0.15  # top 15% are "good"

    for t in range(n_warmup + 1, max_trials + 1):
        costs_so_far = np.array([e["cost"] for e in trajectory])
        threshold = np.quantile(costs_so_far, good_quantile)
        good_idx = np.where(costs_so_far <= threshold)[0]
        bad_idx = np.where(costs_so_far > threshold)[0]

        # Fallback to random if not enough observations for KDE
        if len(good_idx) < 2 or len(bad_idx) < 2:
            cfg = {name: native_space[name][rng.randint(len(native_space[name]))]
                   for name in param_names}
        else:
            # Build index representations
            def cfg_to_indices(entry_cfg):
                return np.array([
                    native_space[n].index(entry_cfg[n]) if entry_cfg[n] in native_space[n]
                    else 0
                    for n in param_names
                ], dtype=float)

            good_vecs = np.array([cfg_to_indices(trajectory[i]["config"]) for i in good_idx])
            bad_vecs  = np.array([cfg_to_indices(trajectory[i]["config"]) for i in bad_idx])

            # KDE bandwidth (Silverman's rule)
            bw_good = max(1.0, np.std(good_vecs, axis=0).mean())
            bw_bad  = max(1.0, np.std(bad_vecs,  axis=0).mean())

            # Sample candidates and score by l(x)/g(x) ratio
            best_cand_cfg = None
            best_ratio = -float("inf")

            for _ in range(n_candidates):
                # Perturb a random good config
                base = good_vecs[rng.randint(len(good_vecs))].copy()
                noise = rng.randn(len(param_names)) * bw_good * 0.5
                cand_indices = np.clip(np.round(base + noise), 0,
                                       [len(native_space[n]) - 1 for n in param_names])
                cand_cfg = {n: native_space[n][int(cand_indices[i])]
                            for i, n in enumerate(param_names)}
                cand_vec = cand_indices

                # Gaussian kernel density estimate
                def kde_density(vecs, query, bw):
                    dists = np.sum(((vecs - query) / (bw + 1e-8)) ** 2, axis=1)
                    return float(np.mean(np.exp(-0.5 * dists)))

                l_x = kde_density(good_vecs, cand_vec, bw_good)
                g_x = kde_density(bad_vecs,  cand_vec, bw_bad) + 1e-10
                ratio = l_x / g_x

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_cand_cfg = cand_cfg

            cfg = best_cand_cfg if best_cand_cfg is not None else {
                name: native_space[name][rng.randint(len(native_space[name]))]
                for name in param_names
            }

        t0 = time.perf_counter()
        result = _safe_evaluate(bench, cfg)
        elapsed = time.perf_counter() - t0
        cost = float(_to_cost(result, metric_name, direction))
        best_cost = min(best_cost, cost)
        trajectory.append({
            "trial": t, "cost": cost, "best_cost": best_cost,
            "config": cfg, "time": round(elapsed, 6),
        })

    return trajectory


def run_smac3(
    bench,
    search_space: dict,
    metric_name: str,
    direction: str,
    max_trials: int,
    seed: int,
) -> List[dict]:
    """Optimizer: SMAC3 (Random Forest surrogate + EI acquisition)."""
    if not (_SMAC_AVAILABLE and _CONFIGSPACE_AVAILABLE):
        raise RuntimeError(
            "smac or ConfigSpace is not installed. Install with: pip install smac configspace"
        )

    # Convert all choice lists to native Python types
    native_space = {
        name: [_make_json_serialisable(v) for v in choices]
        for name, choices in search_space.items()
    }

    configspace = CS.ConfigurationSpace(seed=seed)
    for name, choices in native_space.items():
        configspace.add(CS.CategoricalHyperparameter(name, choices))

    trajectory: List[dict] = []
    best_cost = float("inf")
    trial_counter = 0

    def target_function(config, seed: int = 0) -> float:
        nonlocal best_cost, trial_counter
        t0 = time.perf_counter()
        cfg_dict = _make_json_serialisable(dict(config))
        result = _safe_evaluate(bench, cfg_dict)
        elapsed = time.perf_counter() - t0
        cost = float(_to_cost(result, metric_name, direction))
        best_cost = min(best_cost, cost)
        trial_counter += 1
        trajectory.append({
            "trial": trial_counter,
            "cost": cost,
            "best_cost": best_cost,
            "config": cfg_dict,
            "time": round(elapsed, 6),
        })
        return cost

    import logging
    logging.getLogger("smac").setLevel(logging.WARNING)

    scenario = SMAC_Scenario(
        configspace=configspace,
        deterministic=True,
        n_trials=max_trials,
        seed=seed,
    )
    smac = SMAC_Facade(scenario=scenario, target_function=target_function)
    smac.optimize()

    return trajectory[:max_trials]


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

OPTIMIZER_DISPATCH = {
    "RandomSearch": run_random_search,
    "OptunaTPE": run_optuna_tpe,
    "MethodA_HHD": run_method_a_hhd,
    "MethodB_ABBO": run_method_b_abbo,
    "MethodC_Unified": run_method_c_unified,
    "BOHB": run_bohb,
    "SMAC3": run_smac3,
}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def _make_json_serialisable(obj: Any) -> Any:
    """Recursively convert numpy types to Python natives for JSON."""
    if isinstance(obj, dict):
        return {k: _make_json_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serialisable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def run_pipeline(
    max_trials: int = MAX_TRIALS,
    seeds: List[int] | None = None,
    datasets: List[str] | None = None,
    optimizers: List[str] | None = None,
) -> dict:
    """Run the full benchmarking pipeline and write results to disk."""
    if seeds is None:
        seeds = list(SEEDS)
    if optimizers is None:
        optimizers = list(ALL_OPTIMIZERS)

    # Validate optimizers
    for opt in optimizers:
        if opt not in OPTIMIZER_DISPATCH:
            raise ValueError(
                f"Unknown optimizer '{opt}'. Choose from {ALL_OPTIMIZERS}"
            )

    # Build list of (suite_name, dataset_name) to run
    tasks: List[Tuple[str, str]] = []
    for suite_name, suite_info in DATASET_REGISTRY.items():
        for ds in suite_info["datasets"]:
            if datasets is None or ds in datasets:
                tasks.append((suite_name, ds))

    if not tasks:
        print("No matching datasets found. Nothing to do.")
        return {}

    summary: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}

    for suite_name, ds_name in tasks:
        suite_info = DATASET_REGISTRY[suite_name]
        bench_cls = suite_info["class"]

        try:
            bench = bench_cls(dataset_name=ds_name)
        except Exception as exc:
            print(f"WARNING: Could not load {suite_name}/{ds_name}: {exc}")
            continue

        search_space = bench.search_space
        metric_name = bench.metric_names[0]
        original_direction = bench.directions.get(metric_name, "minimize")

        for opt_name in optimizers:
            # Check availability
            if opt_name == "OptunaTPE" and not _OPTUNA_AVAILABLE:
                print(f"  SKIP {opt_name} (optuna not installed)")
                continue
            if opt_name == "MethodB_ABBO" and not _SKLEARN_AVAILABLE:
                print(f"  SKIP {opt_name} (scikit-learn not installed)")
                continue

            seed_results: List[float] = []
            seed_times: List[float] = []

            for seed in seeds:
                print(
                    f"Running {opt_name} on {suite_name}/{ds_name} seed={seed} ..."
                )

                wall_start = time.perf_counter()
                opt_fn = OPTIMIZER_DISPATCH[opt_name]
                try:
                    trajectory = opt_fn(
                        bench,
                        search_space,
                        metric_name,
                        original_direction,
                        max_trials,
                        seed,
                    )
                except Exception as exc:
                    print(f"  ERROR: {opt_name} failed on {ds_name} seed={seed}: {exc}")
                    continue
                wall_time = time.perf_counter() - wall_start

                final_best_cost = trajectory[-1]["best_cost"] if trajectory else float("inf")
                final_best_config = {}
                for entry in trajectory:
                    if entry["cost"] == final_best_cost:
                        final_best_config = entry["config"]

                result_record = _make_json_serialisable(
                    {
                        "optimizer": opt_name,
                        "benchmark_suite": suite_name,
                        "dataset": ds_name,
                        "seed": seed,
                        "max_trials": max_trials,
                        "direction": "minimize",
                        "original_direction": original_direction,
                        "trajectory": trajectory,
                        "final_best_cost": final_best_cost,
                        "final_best_config": final_best_config,
                        "wall_time_seconds": round(wall_time, 4),
                    }
                )

                # Write individual JSON
                out_dir = RESULTS_DIR / suite_name / ds_name
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{opt_name}_seed{seed}.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(result_record, f, indent=2)

                seed_results.append(final_best_cost)
                seed_times.append(wall_time)

            # Aggregate into summary
            if seed_results:
                summary.setdefault(suite_name, {}).setdefault(ds_name, {})[opt_name] = {
                    "mean_best": round(float(np.mean(seed_results)), 6),
                    "std_best": round(float(np.std(seed_results)), 6),
                    "mean_time": round(float(np.mean(seed_times)), 4),
                }

    # Print results table
    print("\n" + "=" * 110)
    print("  HPOBENCH BENCHMARK RESULTS: mean ± std")
    print("=" * 110)
    header = f"{'Dataset':<35}"
    for opt_name in optimizers:
        header += f"  {opt_name:>14}"
    print(header)
    print("-" * 110)
    for suite_name, ds_name in tasks:
        row = f"{suite_name}/{ds_name:<30}"
        for opt_name in optimizers:
            opt_data = summary.get(suite_name, {}).get(ds_name, {}).get(opt_name)
            if opt_data:
                mean = opt_data["mean_best"]
                std = opt_data["std_best"]
                row += f"  {mean:.4f} ± {std:.4f}"
            else:
                row += f"  {'N/A':>14}"
        print(row)
    print("=" * 110 + "\n")

    # Write summary
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_make_json_serialisable(summary), f, indent=2)
    print(f"\nSummary written to {summary_path}")
    return summary


def run_full_hpobench_pipeline() -> None:
    """Wrapper function to run the full HPOBench benchmark pipeline with default settings."""
    run_pipeline()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HPOBench Benchmarking Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=MAX_TRIALS,
        help=f"Number of optimisation trials (default: {MAX_TRIALS})",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in SEEDS),
        help="Comma-separated list of random seeds (default: 0,1,2,3,4)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help="Comma-separated dataset names or 'all' (default: all)",
    )
    parser.add_argument(
        "--optimizers",
        type=str,
        default="all",
        help="Comma-separated optimizer names or 'all' (default: all)",
    )
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    datasets = None if args.datasets.strip().lower() == "all" else [
        d.strip() for d in args.datasets.split(",")
    ]
    optimizers = None if args.optimizers.strip().lower() == "all" else [
        o.strip() for o in args.optimizers.split(",")
    ]

    run_pipeline(
        max_trials=args.trials,
        seeds=seeds,
        datasets=datasets,
        optimizers=optimizers,
    )


if __name__ == "__main__":
    main()
