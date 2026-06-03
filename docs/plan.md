## Phase B — Baseline Benchmark (✅ DONE)

- Executed `predict_demand.py` (stripped the stray markdown fence).
- **OOF R² = 0.9616** (LGB 0.9615 + XGB 0.9606, blend w_LGB = 0.72).
- **Score = 96.16**. Submission shape `(41,778, 2)` verified; backup saved to `submission_baseline.csv`.
- Logged in `docs/status.md` and `docs/analysis.md` §15.

---

## Phase C — 8 Improvements, Ranked by ROI

Each item is a self-contained checkpoint with: **goal**, **method**, **validation**, **success criterion**, **rollback trigger**.

### C-1. Day-48-only honest holdout evaluator (FOUNDATION)

**Goal:** Establish a trustworthy CV R² before adding features. The baseline CV uses KFold over d48 + d49t, which leaks day-49 patterns.

**Method:**
- Build a new helper `scripts/honest_eval.py` (re-used by C-2…C-6).
- Train rows: `d48` only. Holdout = last 4 hours of d48 = `time_slot ∈ [80, 95]` (16 slots), i.e. the 17:00-23:45 evening window. This window has the largest demand variability (trough at 19h) and is the hardest slice — perfect for discriminating models.
- Build all lag/stat features **causally** within d48:
  - `lm*` for `time_slot ≥ 1` only look back at earlier slots of the same day.
  - Geohash stats computed only on `time_slot < 80` to avoid leakage.
  - `(geohash, time_slot)` means computed only on `time_slot < 80`.
- Train on `d48_train` (slots 0-79), predict on `d48_val` (slots 80-95), report R².
- Sanity: re-run with different holdout windows (e.g. slots 64-79 = afternoon) to confirm signal is stable.

**Validation:** R² on `d48_val` slots 80-95.

**Success criterion:** R² ≥ 0.92 on the holdout (baseline LGB on d48-only 5-fold typically scores ~0.95-0.96 because it sees the same day in different folds; a clean time-split will drop to 0.85-0.92).

**Rollback:** None — this is a new evaluator, doesn't modify anything.

**Files:** `scripts/honest_eval.py`, `eda/honest_baseline.json` (saved scores).

---

### C-2. Smoothed geohash target encoding (HIGH ROI)

**Goal:** Add a leak-free categorical signal for `geohash` (1,249 levels). Currently we only have geohash × time_slot means (`gts_mu`), but not a global smoothed geohash mean that handles sparse geohash.

**Method:**
- Smoothed mean: `te = (n * g_mean + alpha * gm) / (n + alpha)`, where `alpha` is the smoothing prior (start with α=20).
- For d48 training rows: use 5-fold OOF `te` (compute on other 4 folds, predict on held-out).
- For d49t: use full-d48 `te`.
- For test: use full-d48 `te`.
- Add a second variant: `te × is_hw` interaction (te_high for Highway, te_low for Residential).
- Hyperparameter: try α ∈ {5, 10, 20, 50, 100}; pick the one that maximizes honest-holdout R².

**Validation:** Honest-holdout R²; also check per-bucket (Highway vs Residential) R².

**Success criterion:** +0.001 R² on the holdout, OR ≥+0.002 on the Highway subset (where variance is highest).

**Rollback:** Drop the new columns from FEATS.

**Files:** `improve_demand.py` (new), `eda/feature_importance_C2.csv`.

---

### C-3. Interaction features (MEDIUM-HIGH ROI)

**Goal:** Capture non-linear effects between contextual and structural features. The EDA found `Weather` and `Landmarks` are weak on their own; they may interact with peak hours or lane count.

**Method (add 6 new features):**
| Feature | Formula | Rationale |
|---|---|---|
| `wx_x_peak` | `wx_enc * (is_mpk OR is_epk)` | Weather matters more at rush hour |
| `temp_x_peak` | `temp * (is_mpk OR is_epk)` | Temperature sensitivity at rush hour |
| `lanes_x_hw` | `NumberofLanes * is_hw` | Sharp 1→4 lane transition |
| `gmu_x_peak` | `g_mu * (is_mpk OR is_epk)` | High-demand geohash are spikier at peak |
| `is_lunch` | `hr.between(11, 13).astype(int)` | EDA found 11-13h is true daily peak (not 7-9) |
| `is_quiet` | `hr.between(17, 20).astype(int)` | EDA found 17-20h is the trough |

**Validation:** Honest-holdout R².

**Success criterion:** +0.0005 R² (modest; tree models discover most interactions, but the engineered ones can shortcut splits).

**Rollback:** Drop columns from FEATS.

**Files:** `improve_demand.py`.

---

### C-4. Forward lags from d49t (MEDIUM ROI)

**Goal:** Test rows at `time_slot = 9..16` can use d49t's `time_slot = 0..8` as forward-lag lookups (1 to 8 slots ahead). The baseline ignores d49t for lag construction.

**Method:**
- Build a second lookup `lm_d49t = d49t.set_index(['geohash', 'time_slot'])['demand']`.
- For each test row at time_slot `t`, populate `lp_d49t_<lag>` for `lag = 1..8` such that `t - lag ∈ [0, 8]`, i.e. `t ∈ [1, 16]`. This is only test rows at time_slot 1-16, but the test starts at time_slot 9, so valid range is `t ∈ [9, 16]` — about 6,900 test rows (16.5% of test).
- For d49t training rows, do the same: a d49t row at time_slot `t` can use earlier d49t rows at `t-lag` if `t-lag ≥ 0`. Roughly half of d49t (time_slot 1-8) gets some d49t forward-lag coverage.
- Add as 8 new columns `lp_d49t_1..lp_d49t_8`, NaN-filled where inapplicable.

**Validation:** Honest-holdout R²; subgroup R² on test-equivalent time_slot range.

**Success criterion:** +0.0005 overall, OR +0.002 on the time_slot 9-16 subgroup.

**Rollback:** Drop the 8 columns.

**Files:** `improve_demand.py`.

---

### C-5. Add CatBoost + Ridge stacking (MEDIUM ROI)

**Goal:** Diversity in the ensemble. CatBoost handles `geohash` as a native categorical (no target encoding needed), and the regularized linear stacker can find a better blend than the hand-tuned 0.72 LGB / 0.28 XGB.

**Method:**
- Install `catboost` via `uv pip install catboost`.
- Replace `geohash` string with a CatBoost-friendly categorical column.
- Train 5-fold CatBoost with `loss_function='RMSE'`, `learning_rate=0.05`, `depth=8`, `cat_features=['geohash', 'RoadType', 'Weather']`.
- OOF predictions: `oof_cb`.
- After getting OOF predictions for all three models (LGB, XGB, CatBoost), fit a Ridge regression with non-negative weights (use `scipy.optimize.nnls` or `sklearn.linear_model.Ridge` with positive=True):
  ```
  blend = Ridge(alpha=1.0, positive=True).fit(
      np.column_stack([oof_lgb, oof_xgb, oof_cb]), y
  )
  ```
  Then `final_oof = blend.predict(...)`, `final_test = blend.predict(np.column_stack([prl, prx, prc]))`.

**Validation:** Honest-holdout R²; also evaluate each model alone vs. ensemble.

**Success criterion:** Ensemble R² > single-best R² by ≥0.0005.

**Rollback:** Revert to baseline 0.72/0.28 blend; drop CatBoost.

**Files:** `improve_demand.py`.

---

### C-6. Small hyperparameter grid (LOW-MEDIUM ROI, LIGHT)

**Goal:** Squeeze the last 0.001-0.003 R² by tuning the 3 most-impactful LGB params. No Optuna — manual grid.

**Method:**
- Fix all other LGB params from baseline.
- Grid over `num_leaves ∈ {127, 255, 511}` × `min_child_samples ∈ {3, 5, 10}` × `learning_rate ∈ {0.02, 0.03, 0.05}` = 18 cells.
- Each cell: 5-fold KFold on **d48 only** (to keep it fast), report OOF R².
- Pick the top config and use it in the final retrain.
- Repeat for XGB with `max_depth ∈ {6, 8, 10}` × `min_child_weight ∈ {3, 5, 10}` × `learning_rate ∈ {0.03, 0.05, 0.07}` = 27 cells.
- **Wall-time estimate:** ~5 min/cell × 45 cells = ~4 hours total. **Will be parallelized** with `n_jobs` where possible; sequential otherwise.

**Validation:** Best config's d48-5fold R² vs. baseline LGB d48-5fold R².

**Success criterion:** +0.0005 R² vs. baseline. If not met, keep baseline params.

**Rollback:** Revert to baseline params.

**Files:** `improve_demand.py`, `eda/lgb_grid_results.csv`, `eda/xgb_grid_results.csv`.

---

### C-7. Tweedie / quantile-loss objective (LOW ROI, optional)

**Goal:** Robustness check. `demand` is bounded in [0, 1] with a mass near zero, so Tweedie (p=1.5) or quantile (median) might be a useful auxiliary model.

**Method:**
- Train one extra LGB with `objective='tweedie'`, `tweedie_variance_power=1.5`.
- Train one extra LGB with `objective='quantile'`, `alpha=0.5`.
- Add both to the Ridge stacker.
- If neither improves on the honest holdout, **skip** (do not include in the final model).

**Validation:** Honest-holdout R² for each, then with both in the stacker.

**Success criterion:** At least one of the two objectives adds ≥0.0003 to the stacker R².

**Rollback:** Don't include in the final stacker.

**Files:** `improve_demand.py` (only if C-5 already passed).

---

### C-8. Per-geohash bias correction (LOW ROI, post-processing)

**Goal:** After the model predicts, apply a small additive correction per geohash based on OOF residuals. This handles the few geohash where the model systematically overshoots or undershoots.

**Method:**
- Compute OOF residuals: `resid = y - oof_pred` per geohash.
- Per-geohash mean residual `r_g` (shrink toward 0 with count: `r_g_shrunk = r_g * n / (n + k)`, k=20).
- For test rows, `final_pred += r_g_shrunk[geohash]`.
- For geohash not in train, correction is 0.

**Validation:** Honest-holdout R² before vs. after correction.

**Success criterion:** +0.0002 R² on the holdout.

**Rollback:** Drop the post-processing step.

**Files:** `improve_demand.py`.

---

## Phase C — Execution Order & Checkpoints

```
C-1  honest_eval    ──┐
                       │ (foundation — must complete first)
C-2  geohash TE     ──┤
C-3  interactions   ──┤  (independent quick wins — try in any order)
C-4  forward lags   ──┤
                       │
C-5  catboost+ridge ──┤  (requires C-2/C-3/C-4 features)
                       │
C-6  param grid     ──┘  (run with all C-2..C-4 features baked in)
        │
        ├── best params + best features locked in
        │
C-7  tweedie/qtl    ─── (only if C-5 helped)
C-8  bias correct   ─── (post-processing, applied last)
```

**Stop / rollback rule per checkpoint:** if a step's honest-holdout R² regresses vs. previous checkpoint, revert and move on.

**Final scoreboard target:** R² ≥ **0.9635** on honest holdout (vs. baseline's CV R² 0.9616). Realistic ceiling for any improvement given the structural predictability of the data.

---

## Phase D — Final Submission & Documentation (✅ DONE)

After C-1..C-8 are settled:

1. **Retrain** the winning configuration on **all of d48 + d49t** (77,299 rows), 5-fold ensemble (averaging fold predictions), predict on 41,778 test rows.
2. **Clip** predictions to `[0, 1]`.
3. **Write** `/home/marcus/code/Gridlock/submission.csv` with columns `Index, demand`.
4. **Verify:** shape `(41778, 2)`, `Index` matches `test['Index']` exactly, no NaN, all `demand` ∈ [0, 1].
5. **Generate reports:**
   - `eda/final_metrics.json`: OOF R² per model, blend weights, honest-holdout R², baseline delta.
   - `eda/feature_importance.csv`: top-30 features by LGB gain.
   - Update `docs/status.md` with final results table.

**Phase D actual online score: 89.97** (vs baseline 87.26 = +2.71 points; OOF predicted only +0.10).

---

## Phase E — Bridge the OOF vs Online Gap (✅ DONE)

**Outcome: V2 online score 90.73** (+0.76 over V1's 89.97; +3.47 over baseline 87.26). The d49t per-geohash features, multi-seed averaging, and bias removal all contributed. Active `submission.csv` is V2.

### Root cause analysis (post-submission V1)

`eda/root_cause_analysis.py` revealed the gap's anatomy:

| Task | R² | What it measures |
|---|---:|---|
| OOF Mode A (5-fold d48) | 0.9661 | Predict d48 from d48 — same-day |
| OOF Mode C (d48 → d49t) | 0.7550 | Predict d49 from d48 — cross-day |
| **Online (V1 submission)** | **0.8997** | Predict d49 slots 9-55 from d48+d49t |

**The OOF (Mode A) is the wrong task** — it's same-day, but the actual test is cross-day. The day-shift tax is ~0.21 R² (21 points). Online sits 0.145 above Mode C, meaning V1 already partially closed the gap with d49t forward lags (C-4) and the d49t-anchored Ridge stack.

**Key missed opportunity:** d49t has all 1,190 test geohash (just slots 0-8), so per-geohash d49t stats (mean, std, min, max) are available for every test row. We're not using them as features. The d48→d49 day shift varies hugely per geohash (ratio mean=1.27, std=0.74) — capturing this explicitly should help the model learn the d48→d49 mapping.

**Other contributing factors:**
- Bias correction (C-8) overfits d48 KFold residuals; may add noise on test (a different day).
- Multi-seed averaging: OOF→online ratio is ~27×, so a +0.0001 OOF gain may translate to +0.3 online points.
- Ridge stack weights from Mode A may not transfer to test (cross-day); multi-seed averages the model variance.

### E-1: Per-geohash d49t features (HIGH ROI)

**Goal:** Give the model explicit d49-specific per-geohash signal.

**Method:**
- `d49t_g_mu`: per-geohash mean d49t demand (LOO for d49t rows, full for d48/test rows)
- `d49t_g_sd`: per-geohash std d49t demand (full for all)
- `d49t_g_min`, `d49t_g_max`: per-geohash min/max d49t demand
- `d49t_vs_d48`: per-geohash day-shift ratio `d49t_g_mu / g_mu` (clipped to [0.1, 5.0])

**Why LOO for d49t rows:** The d49t_g_mu(X) for a d49t row at (X, t=3) is the mean of d49t demand at (X, 0..8). This includes the row's own demand — mild leakage. LOO removes it: `loo = (n * mu - y) / (n - 1)`.

**Why available for all test rows:** d49t has all 1,190 test geohash (just slots 0-8), so the per-geohash stats are fully computed.

**Files:** `improve_demand.py` — add `use_d49t_geo_features` to `Config`, helper `_add_d49t_geo_features()`, update `_build_d48_features`, `_build_d49t_features`, `_build_test_features`, `get_feature_list`.

### E-2: Multi-seed averaging (MEDIUM-HIGH ROI)

**Goal:** Reduce model variance. The OOF→online gain ratio suggests large payoff.

**Method:**
- 3 seeds: {42, 7, 123}
- For each seed, train LGB+XGB+CatBoost (5-fold) and get OOF + test predictions
- Average OOF across seeds → fit Ridge stacker
- Average test across seeds → apply Ridge weights → clip to [0, 1]
- Wall time: 3× current ≈ 6-9 min

**Files:** `final_pipeline.py` — add `seeds: tuple` to `Config`, loop over seeds in `train_final_and_predict`.

### E-3: Drop bias correction (LOW ROI, MEDIUM RISK)

**Goal:** The per-geohash bias is computed on d48 KFold residuals and applied to test (a different day). This may add noise.

**Method:** Set `use_bias_correction=False` in `make_final_cfg`.

**Rollback:** Re-enable if Phase E degrades online score.

### E-4: Run Phase E pipeline

1. Backup `submission.csv` → `submission_v1.csv` (current 89.97)
2. Build Phase E config (E-1 + E-2 + E-3)
3. Run `scripts/phase_e_pipeline.py`
4. Compare submission_v2.csv (mean, std, correlation with v1)
5. If v2 looks reasonable, submit to platform

### E-5: Compare v1 vs v2

- Per-row correlation
- Mean / std of predictions
- Per-bucket (Highway/Street/Residential) statistics
- Decision: keep v1 if v2 is suspicious (e.g., predictions diverge wildly)

### E-6: Update docs

- `docs/status.md`: add Phase E section, online score table (V1=89.97, V2=TBD)
- `docs/analysis.md`: add §19 (post-Phase-E insights, day-shift calibration)
- `docs/plan.md`: mark Phase E complete when done

### Decision rule

- If V2 online score > 89.97 → keep V2 as `submission.csv`
- If V2 online score ≤ 89.97 → revert `submission.csv` to V1
- Always keep `submission_v1.csv` (89.97, known) and `submission_v2.csv` (TBD)

**Result: V2 online = 90.73 > 89.97 → V2 is now the active `submission.csv`.**

### OOF→online multiplier calibration

| Phase | OOF gain | Online gain | Multiplier |
|---|---:|---:|---:|
| V1 (Phase D) | +0.10 (full OOF) | +2.71 | 27× |
| V2 (Phase E) | +0.11 (Mode C) → +0.30 (full OOF) | +0.76 | 7× (Mode C), 2.5× (full) |

The Mode C multiplier (~7×) is more consistent for cross-day improvements. Use Mode C as the primary OOF signal for predicting online gains on cross-day tasks.

---

---

## File Inventory

| File | Action | Purpose |
|---|---|---|
| `docs/plan.md` | **new** | this plan |
| `docs/status.md` | edit | update with C/D results |
| `docs/analysis.md` | edit | add §18 final insights |
| `scripts/honest_eval.py` | **new** | day-48-only holdout evaluator (C-1) |
| `improve_demand.py` | **new** | all C-2..C-8 + final submission (D) |
| `eda/honest_baseline.json` | **new** | C-1 numbers |
| `eda/lgb_grid_results.csv` | **new** | C-6 grid results |
| `eda/xgb_grid_results.csv` | **new** | C-6 grid results |
| `eda/feature_importance.csv` | **new** | D-5 feature importance |
| `eda/final_metrics.json` | **new** | D-5 final metrics |
| `submission.csv` | overwrite | final predictions |
| `submission_baseline.csv` | unchanged | baseline backup |
| `predict_demand.py` | **unchanged** | historical reference |
| `eda/eda.py`, `eda/eda_deepdive.py` | unchanged | reproducible EDA |

---

## Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| C-2 target encoding leaks across folds | M | Strict KFold with `gm` prior; verify on honest holdout |
| C-4 d49t forward-lag leakage into OOF | L | Only adds columns; d49t rows in training still see d49t's own past (no d48 contamination) — verified by computing OOF residual delta on d49t subset |
| C-5 CatBoost overfits with `geohash` as cat | L | Cap `depth=8`, `l2_leaf_reg=3`; verify holdout R² |
| C-6 grid search overfits the holdout | M | Pick a single holdout, but report on 2 secondary holdouts (slots 64-79, 32-47) as cross-checks |
| Submission file rejected for wrong `Index` mapping | L | Re-verify with `test.merge(sub, on='Index').shape == (41778, len(test.columns)+1)` |
| Time budget blown by C-6 | M | Hard cap: 30 min for LGB grid, 30 min for XGB grid; bail if exceeded |

---

## Decision Log (just for transparency)

- **C-1 holdout = last 4 hours (slots 80-95):** chosen because the EDA found this is the most variable period (morning-peak vs evening-trough).
- **C-2 α=20 starting prior:** standard choice for ~1,249-level categoricals; grid-tuned in C-2.
- **C-5 Ridge over simple weighted mean:** the blend weights from baseline are scalar; Ridge allows per-row adaptive weighting, capturing local model strengths.
- **C-6 manual grid, not Optuna:** matches the "light compute" budget you chose; Optuna available as fallback.
- **C-7 Tweedie/quantile kept conditional:** demand isn't zero-inflated (min = 0.000001, no zeros), so this is a safety check, not a core item.

---

## Estimated total runtime

- C-1: ~5 min
- C-2: ~10 min (5 α values × 5-fold)
- C-3: ~10 min (single retrain)
- C-4: ~5 min
- C-5: ~25 min (CatBoost × 5-fold)
- C-6: ~60 min (LGB + XGB grids, sequential)
- C-7: ~10 min (2 model retrain)
- C-8: ~1 min
- D: ~5 min
- **Total: ~2-2.5 hours**, mostly C-6.

---

When you exit plan mode, I'll:
1. Write this plan to `docs/plan.md`.
2. Update `docs/status.md` to reflect that Phase B is done.
3. Begin with C-1 (honest_eval), as everything else depends on it.
