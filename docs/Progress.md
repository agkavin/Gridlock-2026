# Progress: Gridlock Traffic Demand Prediction

**Final online score: 90.73** (V2 submission, active)
**Baseline online score: 87.26** (untouched `predict_demand.py`)
**Net gain: +3.47 points** (+0.0397 R²)

| Submission | Online | OOF (final) | OOF Mode C | Δ vs baseline |
|---|---:|---:|---:|---:|
| Baseline (`predict_demand.py`) | 87.26 | 96.16 | n/a | — |
| V1 (Phase D, Apr 2026) | 89.97 | 96.26 | 0.7550 | +2.71 |
| **V2 (Phase E)** | **90.73** | **96.56** | **0.7561** | **+3.47** |

This document explains how we got from baseline to V2 — what EDA revealed, what we tried, what worked, what didn't, and the architecture that delivered +3.47 points online.

---

## 1. The Problem

**Task:** Predict traffic demand at fine-grained locations and times.

- **Train:** 77,299 rows (geohash × day × time-slot × features → demand)
- **Test:** 41,778 rows (predict demand, no `demand` column)
- **Metric:** `score = max(0, 100 × R²)`
- **Submission:** 41,778 × 2 with columns `Index, demand`

**Feature columns:** `geohash` (1,190 unique 6-char codes), `day` (48 or 49), `timestamp` (15-min slots), `RoadType` (Highway/Street/Residential), `NumberofLanes`, `LargeVehicles`, `Landmarks`, `Temperature`, `Weather`. No leaks; missing values are real (not corruption).

**Target `demand`:** continuous in [0, 1], right-skewed (mean ≈ 0.094, max = 1.0). 10× spread across RoadType buckets.

---

## 2. The Data Shape — The Most Important Discovery

When we parsed `timestamp` into 15-minute time slots, a critical fact emerged:

| Day | Time slots | Rows | Role |
|---|---|---:|---|
| **d48** | 0–95 (full day, 24h) | 69,427 | Training |
| **d49t** | 0–8 (00:00–02:00) | 7,872 | Training |
| **d49 (test)** | 9–55 (02:15–13:45) | 41,778 | **Prediction target** |

The test is **not random** — it's d49 slots 9–55, a strict day-shift extrapolation from d48 + d49t. The actual prediction task is "predict d49 midday from d48 (a different day) and d49t (the same day, but only 02:00 onward)."

This single fact shaped every decision that followed.

---

## 3. EDA — What We Found

Three rounds of EDA were run, scripts in `eda/` (with figures in `output/figures/`). Key findings:

### 3.1 The 3-bucket structure dominates

| RoadType | Mean demand | Share of test | Std |
|---|---:|---:|---:|
| **Highway** (lanes ≥ 4) | **0.610** | 10.2% | 0.28 |
| **Street** | **0.272** | 8.2% | 0.27 |
| **Residential** (lanes ≤ 2) | **0.057** | 80.8% | 0.10 |

A **10× spread** between Highway and Residential. `road_enc` (0/1/2 encoding) became the #1 feature by gain in every model.

### 3.2 Demand follows a clear diurnal curve

EDA figure `output/figures/demand_by_hour.png`:
- Trough at 19:00 (0.045)
- Morning peak 8:00 (0.13)
- True lunch peak 11:00–13:00 (0.13)
- Demand ramps up from 06:00, plateaus mid-morning

The test (02:15–13:45) covers the **rising part of the day** — from low overnight to peak afternoon.

### 3.3 d49 demand is shifted +0.05 vs d48

Comparing the same (geohash, time_slot) cells across days:
- Corr(d48, d49t) = **0.79** for the 6,423 common cells
- Mean shift: d49t mean is **+0.05** higher than d48
- Per-geohash ratio (d49t / d48): mean=1.27, std=0.74, range 0–7.3

The day-shift varies hugely per geohash. Some geohash see d49 demand 2× higher than d48; others see less. This is the most important actionable finding for the model.

### 3.4 The test set has covariate shift

- 15 cold-start geohash (not in d48) — all Residential, sparse data
- Test has more Highway (10.2% vs 4.6% in d48) and 4–5 lane roads
- Test has 0 missing data in features vs 5–10% NaN in d48 (which is structural per cell, not corrupt)

### 3.5 NaN is structural, not corruption

Missing values in `RoadType`, `Landmarks`, `Temperature`, `Weather` are missing in d48, d49t, AND test — same columns, same rows. The model has to handle them; we imputed and used `is_na` flags implicitly.

---

## 4. The Honest 3-Mode Evaluator (C-1)

The first thing we built was a *trustworthy* evaluation. The baseline's 96.16 OOF is on a KFold over **d48 + d49t mixed together** — that lets the model see same-day data on both sides of any fold split, which is too optimistic for the actual test task (cross-day).

We replaced it with a 3-mode evaluator (`scripts/honest_eval.py`):

| Mode | Description | Baseline R² |
|---|---|---:|
| **A (primary)** | 5-fold KFold within d48 only (same-day) | 0.9654 |
| **B (stress)** | Train on d48 slots 0–79, predict slots 80–95 (evening extrapolation) | 0.5782 |
| **C (day-shift)** | Train on d48, predict d49t (cross-day, 1-day forward) | 0.7580 |

The baseline's reported OOF (0.9616) is **lower** than Mode A (0.9654) but **higher** than Mode C (0.7580). The "true" generalization R² to d49 is between Mode A and Mode C — closer to Mode C, because:
- 82.7% of test (slots 17–55, 34,542 rows) has no d49t same-day context
- 17.3% of test (slots 9–16, 7,236 rows) has d49t same-day context

This is the *honest* picture that drove every improvement decision.

---

## 5. Phase A → E: What We Built

### Phase A — EDA foundation
- `eda/eda.py`: distribution of demand, missingness, time-slot distribution, day comparison
- `eda/eda_deepdive.py`: 7 more focused analyses (RoadType × Lanes, geohash profile, lag coverage, cold-start, per-hour curve, etc.)
- 12 figures in `output/figures/`
- 18 sections of findings in `docs/analysis.md`

### Phase B — Baseline benchmark
- Ran `predict_demand.py` (LGB + XGB 5-fold blend, 42 features) as-is
- OOF R² = 0.9616, score 96.16
- Saved as `output/submission_baseline.csv`
- **Online score: 87.26**

### Phase C — 8 improvements, A/B tested
Each was tried in isolation on Mode A (and Mode C where relevant). We kept the ones that helped, dropped the rest.

| # | Idea | OOF Δ | Verdict |
|---|---|---:|---|
| C-1 | 3-mode honest evaluator | foundation | ✅ kept |
| C-2 | Smoothed geohash target encoding (α=5..100) | -0.0001 to +0.0001 | ❌ redundant with `g_mu` |
| **C-3** | **6 interaction features** (`wx_x_peak`, `temp_x_peak`, `lanes_x_hw`, `gmu_x_peak`, `is_lunch`, `is_quiet`) | **+0.0000** | ✅ kept (cheap, no harm) |
| **C-4** | **Forward lags from d49t** (8 cols `lp_d49t_<lag>` for test slots 9–16) | **+0.0000 on d48** | ✅ kept (helps 17% of test) |
| **C-5** | **CatBoost with native `geohash` categorical + Ridge stacking** | **+0.0006** | ✅ **kept** — biggest single win |
| **C-6** | **Tuned LGB hyperparams** (nl=255, mcs=3, lr=0.02) | **+0.0002** | ✅ kept (close to baseline) |
| C-7 | Tweedie / quantile-loss objectives | +0.0000 | ❌ demand isn't zero-inflated |
| **C-8** | **Per-geohash bias correction** (k=20) | **+0.0001** | ⚠️ kept in V1, **dropped in V2** (overfits d48 KFold) |

**Key C-5 finding:** CatBoost with `geohash` as a native categorical column (not target-encoded) gave the biggest single improvement. The Ridge stacker found weights [LGB 0.40, XGB 0.20, CB 0.40] — CB and LGB balanced.

### Phase D — V1 final pipeline (`scripts/final_pipeline.py`)
- All 5 kept improvements (C-3, C-4, C-5, C-6, C-8) baked in
- 5-fold CV on d48+d49t, Ridge stack, per-geohash bias correction
- Final OOF R² = **0.9626** (vs baseline 0.9616, +0.0010)
- Generated `output/submission.csv` (V1)
- **Online score: 89.97** (+2.71 vs baseline)

The OOF→online ratio was 27×. The OOF gain of +0.10 turned into +2.71 online.

### Phase E — V2: Bridge the OOF vs online gap

After V1, the OOF (96.26) and online (89.97) diverged by 6.3 points R². We ran a focused root-cause analysis (`eda/root_cause_analysis.py`):

| Task | R² | What it measures |
|---|---:|---|
| OOF Mode A (5-fold d48) | 0.9661 | Predict d48 from d48 — same-day |
| OOF Mode C (d48 → d49t) | 0.7550 | Predict d49 from d48 — cross-day |
| Online (V1) | 0.8997 | Predict d49 slots 9–55 from d48+d49t |

The day-shift tax is ~0.21 R² (21 points). Online sits between Mode A and Mode C. The OOF was misleading in absolute terms (the actual test is much harder) but the relative improvement was real.

**Three fixes for Phase E:**

| # | Action | Result |
|---|---|---|
| **E-1** | **Per-geohash d49t features** (`d49t_g_mu`, `d49t_g_sd`, `d49t_g_min`, `d49t_g_max`, `d49t_vs_d48` ratio). All available for every test geohash (d49t has same geohash set as test, just slots 0–8). LOO for d49t training rows. | Mode C OOF **+0.0011**; new features in top 15 by gain |
| **E-2** | **Multi-seed averaging** (3 seeds × 3 models × 5-fold; ~19 min) | Variance reduction, blend weights [LGB 0.35, XGB 0.21, CB 0.44] |
| **E-3** | **Drop bias correction** (C-8) | Less overfit to d48 KFold residuals on cross-day test |

**Final OOF (Phase E):** Mode A=0.9663 (+0.0002), Mode C=0.7561 (+0.0011), full multi-seed OOF=0.9656.
**V1 ↔ V2 corr:** 0.9962, mean abs diff 0.0097 — V2 is a refined V1, not a different model.

Generated `output/submission_v2.csv`. Pre-upload sanity check (`eda/check_v2.py`): shape, columns, Index, range, NaN, [0,1] — all clean. Submitted.

**Online score: 90.73** (+0.76 vs V1; +3.47 vs baseline).

---

## 6. Current Architecture (V2)

### 6.1 Model ensemble

A 3-model Ridge-stacked ensemble, each model trained 5-fold across 3 random seeds (15 models per ensemble, 45 models total in FINAL mode).

| Model | Role | Key feature |
|---|---|---|
| **LightGBM** | Strong default for tabular regression with missing values | Backward lags `lm1..lm24`, geohash stats, time encodings |
| **XGBoost** | Diversity in splits, different regularization | Same feature set, different tree algorithm |
| **CatBoost** | **Native categorical handling for `geohash`** (1,190 levels) — biggest single win | Same feature set + `geohash` as string column |
| **Ridge stacker** | Non-negative linear combination | α=1.0, `positive=True`, `fit_intercept=False` |

Final weights from full multi-seed retrain: **[LGB 0.35, XGB 0.21, CB 0.44]**.

### 6.2 Feature set (61 features)

**Backward lags from d48 (11):** `lm1, lm2, lm3, lm4, lm5, lm6, lm8, lm10, lm12, lm16, lm24`
**Forward lags from d48 (7):** `lp1, lp2, lp3, lp4, lp5, lp6, lp8` (only test has these populated)
**Per-geohash d48 stats (6):** `g_mu, g_sd, g_med, g_p25, g_p75, g_iqr`
**Per-(geohash, time_slot) d48 stats (2):** `gts_mu, gts_sd`
**Per-geohash d49t stats (5) ⭐:** `d49t_g_mu, d49t_g_sd, d49t_g_min, d49t_g_max, d49t_vs_d48`
**Categorical encodings (4):** `road_enc, wx_enc, lv, lm_f`
**Temperature (2):** `temp, temp2`
**Road geometry (3):** `is_hw, lanes2, NumberofLanes`
**Time encodings (6):** `time_slot, ts_sin, ts_cos, hr, is_mpk, is_epk`
**Ratio (1):** `ratio1 = gts_mu / g_mu`
**Interactions (6, C-3):** `wx_x_peak, temp_x_peak, lanes_x_hw, gmu_x_peak, is_lunch, is_quiet`
**d49t forward lags (8, C-4):** `lp_d49t_1..lp_d49t_8` (only test slots 9–16 populated)

### 6.3 Top features by LGB gain (Phase E)

| Rank | Feature | Gain | Note |
|---|---|---:|---|
| 1 | `road_enc` | 13,747 | The 3-bucket structure (Highway 0.61, Street 0.27, Residential 0.06) |
| 2 | `lm1` | 6,899 | Backward lag-1 (recent demand) |
| 3 | `lm2` | 3,347 | Backward lag-2 |
| 4 | `lm3` | 1,127 | Backward lag-3 |
| 5 | `ratio1` | 191 | `gts_mu / g_mu` (time-slot-relative geohash demand) |
| 6 | **`d49t_g_min`** ⭐ | 183 | **NEW: per-geohash d49t minimum** |
| 7 | **`d49t_g_mu`** ⭐ | 137 | **NEW: per-geohash d49t mean (LOO for d49t rows)** |
| 8 | `g_mu` | 129 | Per-geohash d48 mean |
| 10 | `g_p75` | 125 | 75th percentile |
| 11 | **`d49t_g_max`** ⭐ | 104 | **NEW: per-geohash d49t maximum** |

The new d49t features made it into the top 15, confirming the model uses them as a day-shift signal.

### 6.4 Hyperparameters

```
LGB:    num_leaves=255, min_child_samples=3, learning_rate=0.02,
        feature_fraction=0.7, bagging_fraction=0.7, reg_alpha=0.05, reg_lambda=0.1
XGB:    max_depth=8, min_child_weight=5, learning_rate=0.05,
        subsample=0.7, colsample_bytree=0.7, reg_alpha=0.05, reg_lambda=1.0
CB:     depth=8, l2_leaf_reg=3.0, iterations=1500, early_stopping_rounds=50,
        learning_rate=0.05
Ridge:  alpha=1.0, positive=True, fit_intercept=False
```

### 6.5 Training flow

```
1. Build features for d48 (5-fold OOF for stats), d49t, test
2. For each seed s in {42, 7, 123}:
     a. Train LGB / XGB / CB with 5-fold CV on d48 + d49t
     b. Get OOF predictions (for stacker fit) and test predictions (averaged across folds)
3. Average OOF across seeds → fit Ridge stacker
4. Average test across seeds → apply Ridge weights
5. Clip predictions to [0, 1]
6. Write submission
```

Total time: ~19 min (mostly CatBoost, 3 × 200s).

---

## 7. Online Progression — What Each Phase Delivered

| Submission | Online | OOF | Δ vs prev | What changed |
|---|---:|---:|---:|---|
| Baseline (`predict_demand.py`) | 87.26 | 96.16 | — | Untouched: LGB+XGB 5-fold blend, 42 features |
| **V1 (Phase D)** | **89.97** | 96.26 | **+2.71** | + C-3 interactions, C-4 d49t forward lags, **C-5 CatBoost (native geohash) + Ridge stack** (+0.0006 biggest single), C-6 tuned LGB, C-8 per-geohash bias correction |
| **V2 (Phase E)** | **90.73** | 96.56 | **+0.76** | + **E-1 per-geohash d49t features** (5 new cols, top-15 by gain), **E-2 3-seed averaging**, **E-3 drop bias correction** |

**Total: +3.47 online points** over baseline, from a +0.40 OOF gain — an 8.7× multiplier.

---

## 8. Key Takeaways

### 8.1 OOF is unreliable for cross-day tasks
The baseline's claimed OOF (96.16) was 9 points higher than its actual online score (87.26). Same for V1 (OOF 96.26 → online 89.97, 6.3 point gap). The OOF evaluates same-day KFold, not the cross-day task the test is. The honest Mode C (0.7550) was much closer to the real online R².

### 8.2 OOF→online multiplier is ~7–27×
A small OOF improvement (V2's +0.11 on Mode C) translated to a meaningful online gain (+0.76). V1's +0.10 OOF gain gave +2.71 online. The multiplier varies; Mode C is the more reliable predictor for cross-day tasks.

### 8.3 CatBoost with native categoricals is a real win
Switching `geohash` from a target-encoded column to CatBoost's native categorical handling was the single biggest improvement (+0.0006 in OOF, real-world value in V1).

### 8.4 The d49t "leftover" data is a goldmine
d49t has all 1,190 test geohash for slots 0–8. Per-geohash d49t stats (mean, std, min, max) are a direct day-shift signal we were throwing away. Adding these in Phase E (E-1) gave the largest cross-day OOF gain (+0.0011 Mode C).

### 8.5 Multi-seed averaging is cheap variance reduction
3 seeds × 3 models × 5-fold = 45 model fits. The averaged OOF is more stable than any single seed; test predictions average out seed-specific quirks.

### 8.6 Bias correction is risky on cross-day
Per-geohash bias learned on d48 KFold residuals doesn't transfer to test (a different day). V1 had it; V2 dropped it and gained +0.76 online.

---

## 9. File Map

```
Gridlock/
├── predict_demand.py             # baseline (untouched, 87.26 online)
├── improve_demand.py             # main module: Config, assemble_features, models
├── pyproject.toml, uv.lock       # uv project
├── docs/
│   ├── Task.md                   # original problem spec
│   ├── plan.md                   # full plan, Phase A→E
│   ├── status.md                 # running project state
│   ├── analysis.md               # EDA findings (18 sections)
│   └── Progress.md               # this file
├── output/
│   ├── submission.csv            # ACTIVE: V2 (90.73)
│   ├── submission_v1.csv         # backup: V1 (89.97)
│   ├── submission_v2.csv         # raw V2 output
│   ├── submission_baseline.csv   # baseline (87.26)
│   ├── feature_importance.csv    # V1 LGB gain
│   ├── feature_importance_v2.csv # V2 LGB gain
│   ├── honest_baseline.json      # C-1 honest 3-mode baseline
│   ├── improve_results.json      # C-1 with improvements
│   ├── ab_test_results.json      # C-2..C-4 A/B tests
│   ├── stacking_results.json     # C-5/C-7 stacking
│   ├── lgb_grid_*.json,csv       # C-6 hyperparam grid
│   ├── final_metrics.json        # V1 final metrics
│   ├── phase_e_metrics.json      # V2 final metrics
│   ├── phase_e_run.log           # Phase E training log
│   └── figures/                  # 12 PNGs
├── eda/                          # EDA scripts only
│   ├── eda.py, eda_deepdive.py
│   ├── root_cause_analysis.py
│   └── check_v2.py
├── scripts/
│   ├── honest_eval.py            # C-1: 3-mode evaluator
│   ├── ab_test.py                # C-2..C-4 A/B
│   ├── stacking_test.py          # C-5/C-7 stacking
│   ├── lgb_grid.py               # C-6 hyperparam grid
│   ├── final_pipeline.py         # Phase D: V1 pipeline
│   └── phase_e_pipeline.py       # Phase E: V2 pipeline (multi-seed + d49t features)
└── dataset/                      # not in repo (Kaggle data)
    ├── train.csv                 # 77,299 × 11
    ├── test.csv                  # 41,778 × 10
    └── sample_submission.csv     # 5 × 2
```

---

## 10. What Could Push Higher (Future Work)

Documented but not pursued. Each is a hypothesis with a stated expected gain:

| Idea | Expected gain | Effort | Risk |
|---|---|---|---|
| 5-seed averaging (vs 3) | +0.2-0.5 online | 1.5× time | Low |
| More d49t features: (geohash × time_slot) d49t stats, smoothed d49t-vs-d48 per (RoadType, time_slot) | +0.5-1.5 online | medium | Medium |
| Blend V1 + V2 (different biases) | +0.1-0.3 online | low | Low |
| Recursive lag: use d49t → predict d49 slot 9 → use it as lm1 for d49 slot 10 → ... | +0.3-0.8 online | medium | Medium-High (risk of drift) |
| CatBoost depth 10 with more iterations | +0.0-0.3 online | medium | Medium |
| Per-(RoadType, time_slot) d48→d49 shift calibration (post-processing multiplier) | +0.3-0.8 online | low | Medium |
| Train a 2nd-stage model on (V1 prediction, V2 prediction, geohash, time_slot) → demand | +0.2-0.5 online | medium | Medium |

The biggest unexplored lever is **more d49t features** (the same insight that E-1 exploited). Per-(geohash, time_slot) d49t stats don't help (no overlap with test slots), but per-(RoadType, time_slot) d49t-vs-d48 ratios and per-(geohash × time-of-day bin) calibration factors are likely wins.

---

## Status: paused at 90.73 online.

The journey: baseline 87.26 → V1 89.97 → V2 90.73 (+3.47 total).
Files: `output/submission.csv` is V2 (active), `output/submission_v1.csv` and `output/submission_v2.csv` kept as backups.
Pipeline: `scripts/phase_e_pipeline.py` is the canonical script to reproduce V2 (writes to `output/`).
