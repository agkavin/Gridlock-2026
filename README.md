# Gridlock — Traffic Demand Prediction

Predicting traffic demand at fine-grained geohash × time-slot resolution for the Gridlock Kaggle task. Final online score: **90.73** (+3.47 over the baseline).

| Submission | Online | Source |
|---|---:|---|
| Baseline (`predict_demand.py`, untouched) | 87.26 | `submission_baseline.csv` |
| V1 (Phase D) | 89.97 | `submission_v1.csv` |
| **V2 (Phase E, active)** | **90.73** | `submission.csv` |

See [`Progress.md`](Progress.md) for the full story — EDA findings, Phase A through E, the model architecture, what worked and what didn't, and where to push next.

## Quick start

```bash
# Setup
uv sync                    # install deps
uv run python -c "import pandas, lightgbm, xgboost, catboost, sklearn"

# Reproduce V2 (active submission, online 90.73)
uv run python scripts/phase_e_pipeline.py
# → output/submission_v2.csv (~19 min)

# Reproduce V1 (online 89.97)
uv run python scripts/final_pipeline.py
# → output/submission.csv (will overwrite V2 — back up first)

# Reproduce baseline (untouched)
uv run python predict_demand.py
# → output/submission_baseline.csv
```

## Repository structure

```
Gridlock/
├── README.md                     # this file
├── .gitignore
├── pyproject.toml, uv.lock       # uv project (Python 3.13.5)
│
├── predict_demand.py             # baseline (untouched, 87.26 online)
├── improve_demand.py             # main module: Config, feature assembly, models
│
├── docs/                         # all markdown documentation
│   ├── Task.md                   # original problem spec
│   ├── Progress.md               # full project journey: baseline → V2 (90.73)
│   ├── plan.md                   # full plan, Phase A → E
│   ├── status.md                 # running project state
│   └── analysis.md               # 18 sections of EDA findings
│
├── scripts/                      # pipeline stages
│   ├── honest_eval.py            # 3-mode honest evaluator (C-1)
│   ├── ab_test.py                # A/B tests for C-2..C-4
│   ├── stacking_test.py          # C-5 / C-7 stacking experiments
│   ├── lgb_grid.py               # C-6 hyperparam grid
│   ├── final_pipeline.py         # Phase D: V1 pipeline
│   └── phase_e_pipeline.py       # Phase E: V2 pipeline (active)
│
├── eda/                          # EDA scripts only
│   ├── eda.py                    # main EDA
│   ├── eda_deepdive.py           # 7 deep-dive analyses
│   ├── root_cause_analysis.py    # OOF vs online gap analysis
│   └── check_v2.py               # V2 sanity check
│
└── output/                       # all generated outputs
    ├── submission.csv            # ACTIVE: V2 (90.73)
    ├── submission_v1.csv         # backup: V1 (89.97)
    ├── submission_v2.csv         # raw V2 (also at submission.csv)
    ├── submission_baseline.csv   # baseline (87.26)
    ├── feature_importance.csv    # V1 LGB gain
    ├── feature_importance_v2.csv # V2 LGB gain
    ├── honest_baseline.json      # C-1 3-mode baseline
    ├── improve_results.json
    ├── ab_test_results.json
    ├── stacking_results.json
    ├── lgb_grid_best.json
    ├── lgb_grid_results.csv
    ├── final_metrics.json        # V1 metrics
    ├── phase_e_metrics.json      # V2 metrics
    ├── phase_e_run.log           # Phase E training log
    └── figures/                  # 12 EDA PNGs
```

## Dataset

Not included in this repo (Kaggle competition data). Place the following files in `dataset/`:
- `train.csv` (77,299 × 11)
- `test.csv` (41,778 × 10)
- `sample_submission.csv` (5 × 2)

The data has 1,190 unique geohash codes, two days (d48 with all 96 time slots + d49t with slots 0–8), and a target `demand` ∈ [0, 1].

## Architecture (V2)

A 3-model **Ridge-stacked ensemble**, each model trained 5-fold across 3 random seeds:

| Model | Role | Weight |
|---|---|---:|
| LightGBM | Backward lags, geohash stats, time encodings | 0.35 |
| XGBoost | Different tree algorithm, regularization | 0.21 |
| **CatBoost (native geohash categorical)** | Biggest single win — handles 1,190-level categorical natively | 0.44 |
| Ridge stacker | α=1.0, positive=True, fit_intercept=False | — |

**61 features** including:
- 11 backward lags from d48 (`lm1..lm24`)
- 7 forward lags from d48
- 6 per-geohash d48 stats + 2 per-(geohash, time_slot) stats
- **5 per-geohash d49t stats** (E-1: `d49t_g_mu` LOO, `d49t_g_sd`, `d49t_g_min`, `d49t_g_max`, `d49t_vs_d48`)
- 6 interaction features (C-3)
- 8 d49t forward lags for test slots 9–16 (C-4)

See `Progress.md §6` for the full feature table and top-15 by gain.

## Key findings

1. **The OOF (96.16) is the wrong task** — same-day KFold, but the actual test is a cross-day prediction. Mode C (true day-shift) gives 0.7550; the online R² is 0.8997.
2. **d49t data is the missing signal** — it has all 1,190 test geohash for slots 0–8. Per-geohash d49t stats are fully available for every test row and they help.
3. **OOF→online multiplier is ~7–27×** — a small OOF gain can give a meaningful online improvement.
4. **CatBoost with native categoricals** is a real win — switching `geohash` to a string column gave the biggest single improvement.
5. **Bias correction is risky on cross-day** — V1 had it; V2 dropped it and gained +0.76 online.

## Online progression

| Submission | Online | Δ vs prev | What changed |
|---|---:|---:|---|
| Baseline | 87.26 | — | Untouched LGB+XGB blend |
| V1 (Phase D) | 89.97 | +2.71 | + interactions, d49t lags, **CatBoost (native cat) + Ridge stack**, tuned LGB, bias correction |
| **V2 (Phase E)** | **90.73** | **+0.76** | + **per-geohash d49t features**, **3-seed averaging**, drop bias correction |

## Hyperparameters

```
LGB:    num_leaves=255, min_child_samples=3, learning_rate=0.02,
        feature_fraction=0.7, bagging_fraction=0.7, reg_alpha=0.05, reg_lambda=0.1
XGB:    max_depth=8, min_child_weight=5, learning_rate=0.05,
        subsample=0.7, colsample_bytree=0.7
CB:     depth=8, l2_leaf_reg=3.0, iterations=1500, early_stopping_rounds=50,
        learning_rate=0.05
Ridge:  alpha=1.0, positive=True, fit_intercept=False
Seeds:  42, 7, 123
```

## Dependencies

Python 3.13.5, managed with [uv](https://docs.astral.sh/uv/). Main packages:
- `pandas`, `numpy`
- `scikit-learn`
- `lightgbm`, `xgboost`, `catboost`
- `matplotlib`, `seaborn` (for EDA)
- `scipy`

## Future work

Documented in `Progress.md §10`. Top candidates:
- More d49t features (per-(RoadType, time_slot) calibration, smoothed d49t-vs-d48 ratios)
- 5-seed averaging (vs 3)
- Recursive lag: use d49t → predict d49 slot 9 → use it as lm1 for slot 10, etc.
- Blend V1 + V2
- Per-(RoadType, time_slot) d48→d49 shift calibration (post-processing multiplier)
