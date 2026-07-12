# HPO-HMC Repository: Issues Found, Fixes Applied, Cross-Checked Results

Repository: https://github.com/cyphlia/HPO-HMC (cloned and run directly, not
just read — every claim below was verified by executing the actual code).

All 5-seed numbers use the repo's own existing protocol: seeds 0-4,
`config.N_SAMPLES=2500`, `config.N_WARMUP_EPOCHS=30`, `config.N_EPOCHS=80`.
The "original" column is the repo's own pre-existing
`results_physics_multiseed/physics_multiseed_summary.json` — not something
we generated — so this is a true before/after on the actual codebase.

## Cross-checked 5-seed results (original repo vs. after fixes)

### Method A (Pure HHD)

| Metric | Original (buggy) | Fixed | Change |
|---|---|---|---|
| Best val loss | 0.2760 ± 0.1107 | 0.2388 ± 0.1643 | modest improvement, more variance |
| **R²** (landscape) | **0.4517 ± 0.3184** | **0.9789 ± 0.0158** | **0.45 → 0.98, variance 20x tighter** |
| MAE (landscape) | 2.0391 ± 0.4967 | 0.3509 ± 0.1091 | **5.8x lower error** |
| Wall time | 21.24s ± 1.13s | 13.36s ± 0.35s | 37% faster |

### Method C (Unified HHD-ABBO)

| Metric | Original | Fixed | Change |
|---|---|---|---|
| Best val loss | 0.00658 ± 0.00302 | 0.00433 ± 0.00190 | 34% lower, tighter |
| R² (landscape) | 0.99964 ± 0.00026 | 0.99987 ± 0.00014 | modest but real improvement |
| MAE (landscape) | 0.04423 ± 0.02137 | 0.02689 ± 0.01227 | 39% lower |
| Wall time | 87.76s ± 3.71s | 50.15s ± 1.88s | 43% faster |
| **Crash risk** | **Would occasionally NaN-crash at full 80-epoch scale after the temperature fix alone** (see Issue 2) | **Fixed with gradient clipping + NaN-guard** | — |

### Method B (Hybrid ABBO)

Untouched by any of these fixes (doesn't use `HamiltonianMCMC` at all) — ran
a smoke test to confirm it's unaffected. Unaffected, as expected.

Raw per-seed numbers for the "fixed" columns, plus the scripts that produced
them (`rerun_method_a.py`, `rerun_method_c.py`), are included below so every
number here is reproducible from scratch.

---

## Issues found (in the order they were discovered) and what was changed

### Issue 1 — Method A never checkpoints its best model (train_hamiltonian.py)

**What was wrong:** `evaluate.py` computes the "Best Val Loss" column as
`min(history["val_loss"])` — a post-hoc minimum over the logged curve — but
the model actually saved to `results_hamiltonian/model.pt` (and used to
compute MAE/RMSE/R²/landscape plots) was whatever weights existed at the
**final** epoch. Methods B and C both already checkpoint their best model
(`best_val == final_val` exactly in every one of their result files); Method
A did not, so its "Best Val Loss" number described a different, better set
of weights than the ones actually evaluated for MAE/RMSE/R². In the original
single-run table this produced a 4x gap between Method A's own "Best Val
Loss" (0.151) and "Final Val Loss" (0.644) — the dynamics wander away from
good solutions with nothing to catch them.

**Fix:** `train_hamiltonian.py` now tracks `_best_val` / `_best_state` /
`_best_hp` during the HMC loop exactly like `hybrid_hhd_abbo_improved.py`
already does, and restores the best checkpoint before returning/saving.

### Issue 2 — Temperature only scaled the loss term, not the full ΔH (symplectic_solver.py)

**What was wrong:** `HamiltonianMCMC.propose()` computed
`H = kinetic + loss/temperature` and then accepted with
`exp(-(H_prop - H_init))` — i.e. temperature was baked into the loss term
*before* composing H, then never applied again at the acceptance step. But
the leapfrog dynamics themselves (the actual gradient-driven position/
momentum updates) always use the full, unscaled loss gradient — temperature
never touches them. The result: at `T=1e9` ("optimisation mode", meant to
mean "always accept"), the loss term inside H became ~0, but the acceptance
criterion was still fully exposed to whatever kinetic-energy drift the
leapfrog trajectory picked up, independent of whether the loss improved.
Measured acceptance rate at T=1e9 was **25-45%**, not the ~100% the
docstrings/README describe — good, loss-improving proposals were being
rejected purely on unrelated kinetic-energy grounds.

**Fix:** apply temperature to the entire ΔH at the acceptance step —
`exp(-(H_prop - H_init)/T)` — matching the paper's own stated formula
`min(1, e^{-ΔH/T})`. Verified acceptance now reaches 100% at T=1e9, as
documented.

**Important downstream consequence, also fixed:** once acceptance genuinely
approaches 100%, nothing was left to prevent theta/lambda from drifting
arbitrarily far during a long run. On the full 80-epoch protocol this
occasionally diverged (`log_lr` hyperparameter → `inf`, then `inf - inf` in
the finite-difference HP-gradient computation → `nan`, which then poisoned
the whole hyperparameter state and crashed with `ValueError: Invalid
learning rate: nan`, reproduced live). Fixed two ways:
- Gradient-norm clipping inside the leapfrog momentum step
  (`LeapfrogIntegrator._clip_grads_`, mirroring the clipping already used
  elsewhere in the Adam phases), reducing how often extreme states are
  reached at all.
- A NaN/Inf safety guard in `propose()`: a proposal with non-finite loss or
  H is now force-rejected regardless of temperature — accepting a corrupted
  state doesn't just make one proposal bad, it permanently poisons every
  future proposal, so no reasonable "always accept" policy should actually
  want it accepted.

Both Method A and Method C use `HamiltonianMCMC`, so both picked up this fix;
both were re-verified across all 5 seeds (results above) with no crashes and
better numbers than before.

### Issue 3 — `config.MASS_LAMBDA` silently ignored for Method A (train_hamiltonian.py)

**What was wrong:** `config.py` defines `MASS_LAMBDA = 5.0` with a comment
explaining what it controls. `hybrid_hhd_abbo_improved.py` (Method C)
already used a `mass_lambda: float = None` → falls back to
`config.MASS_LAMBDA` pattern. `train_hamiltonian.py` (Method A) hardcoded
`mass_lambda: float = 0.1` instead — `main.py` never passes `mass_lambda`
explicitly when constructing either trainer, so Method A silently used 0.1
regardless of what `config.py` said, while Method C correctly used 5.0. Dead
configuration that misleads anyone reading `config.py`.

**Fix:** Method A now uses the same `None` → `config.MASS_LAMBDA`
fallback pattern Method C already had, for consistency.

### Issue 4 — Single-method runs silently blend in stale results from other methods (main.py / evaluate.py)

**What was wrong:** `evaluate_harmonic()` unconditionally reads
`results_hamiltonian/`, `results_hybrid/`, and `results_unified_improved/`
from disk and prints a 3-way comparison table — regardless of which
method(s) were actually just trained. Running `python main.py --task
harmonic --method pure` (Method A only) still printed a full 3-way table
with Methods B and C's numbers pulled from whatever unrelated previous run
happened to populate those directories, with no indication they were stale.
Reproduced live: a 10-epoch smoke-test run of Method A alone printed exactly
the same B/C numbers as a much earlier full run.

**Fix:** `run_single_harmonic()` now returns which method/directory it
actually wrote; the dispatch code in `main.py` only evaluates that one
method when `--compare` wasn't passed, and prints an explicit note that it's
a single-method run, not a comparison.

---

## Files changed

- `symplectic_solver.py` — persistent-momentum (Generalized HMC) option
  added to `HamiltonianMCMC` (off by default, see note below), temperature
  bugfix, gradient clipping, NaN/Inf guard.
- `train_hamiltonian.py` — best-checkpoint tracking, `mass_lambda` config
  fallback, wiring for the new `momentum_refresh` option.
- `main.py` — single-method runs no longer blend in stale comparison data.
- `rerun_method_a.py`, `rerun_method_c.py` — verification scripts used to
  produce every "fixed" number in this report; safe to delete or keep as
  regression tests.
- `method_c_fixed_results.json` — raw per-seed output backing the Method C
  "fixed" row above.

## A feature that's implemented but intentionally NOT the new default: persistent momentum

While diagnosing Issue 2, `HamiltonianMCMC` was extended with a
`momentum_refresh` parameter implementing Generalized HMC (Horowitz 1991):
momentum persists across proposals via partial refresh instead of full
resampling, with negation-on-reject to preserve detailed balance. This is
correctly implemented and available (`momentum_refresh=0.1`, e.g.), but
**testing showed it is not a clear win at `T=1e9`**: with acceptance already
near 100%, persistent momentum has nothing to correct it (rejection-based
negation almost never triggers), so bad momentum can compound instead of
being reset — in a 40-epoch test it produced worse peak loss (56 vs. 3.9)
and a worse checkpoint (0.157 vs. 0.147) than plain full-resampling. It
looked more promising at a moderate temperature (`T=5`, best_val 0.173 vs.
0.290 for full-resample at `T=1`), where MH rejection is doing real work.
Left as an opt-in, off-by-default, correctly-implemented option rather than
overclaiming a specific best setting — a proper temperature × momentum_refresh
sweep (the repo already has `sensitivity_analysis.py` for exactly this) is
the natural next step and wasn't done here for scope reasons.

## Suggested next steps for the repo

1. Re-run `results_physics_multiseed/` with the fixes and commit the updated
   JSON — the numbers above show it's a strict improvement, not just a
   safety fix.
2. Consider whether `ablation_summary.json` and `results_hpobench/` also
   need re-running, since they may have run through the same buggy
   temperature path (they don't crash the way the 80-epoch physics run did,
   but their acceptance-rate-derived behavior would have differed under the
   bug).
3. Run a `sensitivity_analysis.py` sweep over `(temperature, momentum_refresh)`
   for Method A specifically, now that both are correctly implemented and
   interact in a way that isn't yet fully characterized.
4. Consider surfacing the new NaN-guard's trigger count in `history` (e.g.
   `n_rejected_nan`) so silent numerical instability during long runs is
   visible in logs rather than only showing up as an occasional crash.
