# Phase A — Exploratory Data Analysis (EDA)

**Scope:** Verify the data, characterize distributions, examine missingness, map
geohash × time-slot coverage, and quantify the predictive power of each feature.
**Goal:** Inform feature engineering and validation strategy for the Phase C
improvements.

All numbers in this document are reproducible by running:

```bash
uv run python eda/eda.py            # core EDA
uv run python eda/eda_deepdive.py   # additional deep dives
```

Figures are saved to `eda/figures/`.

---

## TL;DR — The Big Picture

1. **The problem is mostly a structured prediction problem, not a time-series one.**
   `demand` is dominated by `(RoadType, NumberofLanes)`:
   - Highway (any lanes 2-5): mean demand ≈ **0.61**
   - Street (1 lane): mean demand ≈ **0.27**
   - Residential (1-3 lanes): mean demand ≈ **0.057**
   The 0.057 vs 0.61 ratio is **~10×** — this single interaction explains most
   of the variance.
2. **The test set is the future of day 49, immediately after the day-49 train rows.**
   d49t covers `time_slot 0..8` (00:00 – 02:00), test covers `time_slot 9..55`
   (02:15 – 13:45). They're contiguous with a 15-min gap.
3. **The test set is shifted toward higher-demand segments.**
   - Highway share: 4.6% (train) → **10.2% (test)**
   - 4-lane share: 1.2% → **2.5%** ; 5-lane share: 1.2% → **2.5%**
   - This means the test set has more inherently high-demand geohash; the
     naive "predict the train mean" baseline is too low.
4. **Day 49 demand is systematically higher than day 48** for the same
   `(geohash, time_slot)`: mean delta = **+0.05**, with corr 0.79.
5. **Backward lag-1 from d48 hits ~88% of test rows** — so the baseline's
   `lm*` features are well populated and informative.
6. **15 cold-start test geohash (101 rows, 0.24%) are not in d48** — all
   Residential 1-3 lanes. These are essentially predictable from
   `(RoadType, NumberofLanes)` alone with low mean demand.
7. **No data integrity issues:** no duplicate `(geohash, day, timestamp)`
   triples, no duplicate `Index`, no NaN `demand`, all `demand` in `[0, 1]`.

The baseline already scores **R² = 0.9616** (Score = 96.16). The dominant
driver is the structural features; gains will come from:

- Better **day-shift calibration** (test ≈ d48 × (d49t / d48) ratio).
- A **per-bucket** model (separate handling of Highway vs Residential).
- Per-geohash **bias correction** computed on OOF residuals.
- Richer lag/rolling features (mean over last 4 slots, etc.).
- Stack/blend with a small linear/quantile model that captures the
  demand ≈ 0.05 plateau in Residential.

---

## 1. Data Shapes, Dtypes, NaN Counts

```
train: (77299, 11)  test: (41778, 10)  sample_submission: (5, 2)
```

| Column | dtype | train NaN % | test NaN % | n unique |
|---|---|---:|---:|---:|
| Index | int64 | 0 | 0 | 77299/41778 |
| geohash | str | 0 | 0 | 1249/1190 |
| day | int64 | 0 | 0 | 2/1 |
| timestamp | str | 0 | 0 | 96/47 |
| demand | float64 | 0 | — | 76,715 unique values |
| RoadType | str | 0.776 | 0.776 | 3 (+NaN) |
| NumberofLanes | int64 | 0 | 0 | 5 (1-5) |
| LargeVehicles | str | 0 | 0 | 2 |
| Landmarks | str | 0 | 0 | 2 |
| Temperature | float64 | 3.228 | 3.229 | 74,804 (continuous) |
| Weather | str | 1.031 | 1.032 | 4 (+NaN) |

`NumberofLanes`, `LargeVehicles`, `Landmarks` are **fully observed**.
NaN rates are **identical in train and test** — the missingness is not
the result of a train/test split (see §4).

`demand` is bounded in `[0.000001, 1.0]` with **no zeros, no negatives**.

---

## 2. Demand Distribution (Train)

| Stat | Value |
|---|---:|
| count | 77,299 |
| mean | 0.0939 |
| std | 0.1422 |
| min | 0.000001 |
| 1% | 0.000625 |
| 25% | 0.0182 |
| 50% | 0.0478 |
| 75% | 0.1086 |
| 95% | 0.3359 |
| 99% | 0.8623 |
| max | 1.0 |

Distribution is **right-skewed with a small secondary mass near 1.0**
(see `figures/demand_distribution.png`). The high end is dominated by
Highway geohash that reach saturation.

Per-day means:

| Day | n rows | mean | std |
|---|---:|---:|---:|
| 48 | 69,427 | 0.0927 | 0.1408 |
| 49 (train rows) | 7,872 | 0.1053 | 0.1520 |

**Day 49 is +14% higher on average** than day 48 (only the 00:00-02:00 portion
overlaps, but the trend is consistent with the within-cell analysis in §11).

---

## 3. The Dominant Signal — `RoadType × NumberofLanes`

Mean demand by `(RoadType, NumberofLanes)`:

| RoadType \ Lanes | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| **Highway**    | (none)    | 0.620 | 0.613 | 0.604 | 0.607 |
| **Residential**| 0.057 | 0.057 | 0.057 | (none) | (none) |
| **Street**     | 0.273 | (none) | (none) | (none) | (none) |
| **NaN**        | 0.085 | 0.088 | 0.075 | 0.550 | 0.659 |

Key takeaways:

- The **(RoadType, Lanes) → mean demand** mapping is **nearly constant within
  RoadType** — i.e. lanes 2-5 on a Highway all give ~0.61 demand. The
  encoding used in the baseline (`is_hw = NumberofLanes >= 4` and a polynomial
  on lanes) misses the clean bifurcation.
- The 4-5 lane Highway/NaN rows form a tight cluster around 0.6 — they should
  be very easy to predict.
- The Residential cluster at 0.057 is also very tight — easy to predict
  *as a low number*, but the absolute error is small.

This explains the **>0.96 R² ceiling**: ~95% of rows are essentially in one
of three well-separated buckets. The remaining ~5% R² must come from
within-bucket variation (time of day, weather, geohash, temperature, lags).

`figures/demand_by_roadtype.png` and `figures/demand_by_roadtype_lanes.png`
visualize this.

---

## 4. Missingness Pattern (MCAR vs Structural)

| Column | Train NaN % | Test NaN % |
|---|---:|---:|
| RoadType | 0.776 | 0.776 |
| Temperature | 3.228 | 3.229 |
| Weather | 1.031 | 1.032 |

**Joint missingness** (count of rows by # of NaN cells across the 6
contextual columns):

| # NaN | train | test |
|---:|---:|---:|
| 0 | 73,459 | 39,702 |
| 1 | 3,789 | 2,049 |
| 2 | 50 | 26 |
| 3 | 1 | 1 |

NaN rate of *any* feature is identical in train and test (0.84% per day)
→ **not driven by a data-collection difference**; likely sensor or
ingestion gaps.

**Mean demand for rows with ≥1 NaN feature vs. none:**

| | count | mean demand | std |
|---|---:|---:|---:|
| any_nan=False | 73,459 | 0.0939 | 0.1421 |
| any_nan=True  |  3,840 | 0.0954 | 0.1442 |

→ No demand skew: NaN rows are **MCAR-like with respect to demand**.

**Spatial concentration of NaN:**
- 70,876 unique `(geohash, time_slot)` cells in train.
- 3,826 cells have ≥1 NaN row; **3,212 of those are 100% NaN** (i.e. entire
  cell missing). Median per-cell NaN rate = 0.
- Conclusion: NaN is **strongly structural per cell** — once a cell starts
  missing data, it tends to be missing for that feature for the whole day.
- Example: `Index 0` of both train and test is the same geohash `qp02z1`
  with all three of `RoadType, Temperature, Weather` NaN — and d48 also
  has it the same way. The system has a permanent gap for that geohash.

**Implication for modeling:**
- Filling NaN with median/mode (as the baseline does) is a safe default.
- An "any-NaN" indicator flag could capture a small extra signal.
- `RoadType` is also a `NaN` category — about 600 train / 324 test rows
  have unknown RoadType. Demand for those is around 0.10 (intermediate),
  not clearly High or Low.

`figures/missingness_corr.png` shows the **missingness indicators are weakly
correlated** (no single common cause), but co-missing in the same cell is
common.

---

## 5. Categorical Feature Distributions

### RoadType

| RoadType | train % | test % | Δ(test − train) |
|---|---:|---:|---:|
| Residential | 89.56 | **80.84** | **−8.72** |
| Highway | 4.61 | **10.23** | **+5.62** |
| Street | 5.06 | **8.16** | **+3.10** |
| NaN | 0.78 | 0.78 | 0.00 |

**Distribution shift:** test is markedly more Highway/Street, less Residential.

### NumberofLanes

| Lanes | train % | test % | Δ |
|---|---:|---:|---:|
| 1 | 35.46 | 35.49 | +0.03 |
| 2 | 31.21 | 29.73 | −1.48 |
| 3 | 30.94 | 29.70 | −1.24 |
| 4 | 1.20 | **2.54** | **+1.34** |
| 5 | 1.19 | **2.54** | **+1.35** |

**Distribution shift:** test has 2× the share of 4-5 lane rows.

### LargeVehicles

| | train % | test % |
|---|---:|---:|
| Not Allowed | 65.55 | 62.66 |
| Allowed | 34.45 | 37.34 |

→ Small shift (+2.9% toward "Allowed"). Mean demand is 0.132 (Allowed) vs
0.074 (Not Allowed), so a small effect.

### Landmarks

| | train % | test % |
|---|---:|---:|
| Yes | 67.33 | 67.61 |
| No | 32.67 | 32.39 |

→ Negligible shift. Mean demand: 0.093 (Yes) vs 0.096 (No) — essentially no
predictive power on its own.

### Weather

| Weather | train % | test % | mean demand |
|---|---:|---:|---:|
| Sunny | 35.86 | 36.09 | 0.094 |
| Rainy | 26.94 | 26.52 | 0.094 |
| Foggy | 26.19 | 26.57 | 0.093 |
| Snowy | 9.98 | 9.78 | 0.093 |

→ Weather is **almost perfectly balanced** and has **near-zero marginal
effect on demand** (all four categories within 0.001 of each other).
A model could safely one-hot it; tree models will discover this near-irrelevance.

The interaction with `(RoadType, Lanes)` also shows no meaningful effect —
mean demand for Highway is ~0.61 regardless of weather, and Residential
is ~0.057 regardless of weather.

---

## 6. Numerical Features

### `NumberofLanes`

Mean demand by lanes (train):

| Lanes | mean demand | std | n |
|---:|---:|---:|---:|
| 1 | 0.0881 | 0.0907 | 27,411 |
| 2 | 0.0775 | 0.1245 | 24,127 |
| 3 | 0.0779 | 0.1250 | 23,919 |
| 4 | 0.6029 | 0.2258 | 926 |
| 5 | 0.6076 | 0.2270 | 916 |

The 4-5 lane rows are essentially a binary indicator for "high demand"
(mean ≈ 0.6), not a continuous predictor. The baseline's
`is_hw = (NumberofLanes >= 4)` captures this.

### `Temperature`

| Stat | Value |
|---|---:|
| count | 74,804 |
| mean | 16.4 °C |
| std | 7.4 °C |
| min | −14.9 |
| max | 48.3 |
| 3.2% NaN |

Mean demand by decile is flat (0.090-0.098) until the hottest decile
(41.9 - 48.3 °C, only 18 rows) where it jumps to 0.163.

**Temperature has near-zero linear correlation with demand** (Pearson 0.003,
Spearman −0.003). Any signal is non-linear and weak. The baseline's `temp²`
won't hurt but won't help.

---

## 7. Correlations with `demand`

| Feature | Pearson | Spearman |
|---|---:|---:|
| NumberofLanes | **0.2141** | 0.0037 |
| Temperature | 0.0031 | −0.0031 |
| ts_min | −0.0377 | −0.0677 |
| time_slot | −0.0377 | −0.0677 |
| hr | −0.0378 | −0.0679 |

→ `NumberofLanes` has high Pearson because of the 1-3 vs 4-5 bifurcation;
Spearman is near zero because the relationship is non-monotonic. The
categorical interactions dominate everything.

`hr`/`time_slot` has only modest correlation — the day 48 demand curve is
fairly flat (mean by hour ranges 0.04-0.12), so time-of-day alone is weak.

---

## 8. Geohash Cardinality & Overlap

| Subset | # unique geohash |
|---|---:|
| d48 | 1,241 |
| d49t | 1,078 |
| test | 1,190 |
| **Union (d48 ∪ d49t ∪ test)** | **1,259** |
| d48 ∩ d49t | 1,070 |
| d48 ∩ test | 1,175 |
| d49t ∩ test | 1,067 |
| d48 ∩ d49t ∩ test | 1,062 |
| **test geohash NOT in d48** | **15 (cold-start)** |
| **test geohash NOT in d48 ∪ d49t** | **10** |

**Test rows with a geohash that exists in d48 (lag coverage from d48):**
**41,677 / 41,778 (99.76%)** — only 101 cold-start rows (0.24%).

Geohash appearance counts:

- Train: median 71, mean 62, max 105 rows per geohash.
- Test: median 45, mean 35, max 47 rows per geohash.

The most frequent test geohash (`qp02zs`, `qp08bh`, etc.) each appear in 47
distinct time slots — i.e. a test geohash gets exactly **one row per
quarter-hour between 02:15 and 13:45**. The dataset is densely sampled.

### Cold-start geohash (15 not in d48)

All 15 are **Residential 1-3 lanes**; mean demand expected ≈ 0.057. Their
101 test rows should be predictable from `(RoadType, NumberofLanes)` alone.

---

## 9. Time-slot Coverage

| Subset | time_slot range | # distinct slots |
|---|---|---:|
| d48 | 0..95 (full day) | 96 |
| d49t | 0..8 (00:00 – 02:00) | 9 |
| test | 9..55 (02:15 – 13:45) | 47 |

**d49t → test is contiguous** (one 15-min gap at time_slot=8 → 9). The
test set picks up exactly where the day-49 training rows end.

**Per-hour d48 demand curve** (mean):

```
hr  mean
 0  0.0575
 1  0.0726
 2  0.0835
 3  0.0913
 4  0.1017
 5  0.1044
 6  0.1039
 7  0.1062   <- morning ramp
 8  0.1062
 9  0.1092
10  0.1116
11  0.1173   <- peak (lunch-adjacent)
12  0.1148
13  0.1163
14  0.1072
15  0.0832
16  0.0701
17  0.0589
18  0.0488
19  0.0421   <- evening trough
20  0.0438
21  0.0570
22  0.0745
23  0.0925
```

d48 has a clear **morning ramp (6-9h), midday plateau (10-14h), evening
trough (17-20h), and recovery (21-23h)**. The baseline flags `is_mpk` and
`is_epk` for the rush hours (7-9, 17-19) but the actual maxima are 11-13h.
**Suggested new flag:** `is_lunch` (11-13h).

**d49t (00:00-02:00) mean = 0.1053** vs **d48 same hours = 0.0713** —
day 49 is **~50% higher** during these quiet hours. This is a strong
signal that we cannot just predict test with d48's same-slot demand —
we need a calibration term.

---

## 10. Day-to-Day Stability (d48 vs d49t on common cells)

| | d48 (mean on common cells) | d49t (mean on common cells) |
|---|---:|---:|
| mean | 0.0732 | 0.1240 |
| 1% | 0.00083 | 0.00111 |
| 50% | 0.0440 | 0.0738 |
| 99% | 0.4501 | 0.8181 |

- **corr(d48, d49t) = 0.7924** on 6,423 common `(geohash, time_slot)` cells
- **mean delta (d49t − d48) = +0.0507** (systematic upward shift)
- **abs(delta) p50 = 0.0351, p90 = 0.1428, p99 = 0.4621**

Interpretation: the 1-day-lag predictor is a **strong but biased** baseline.
Expected naive R² ≈ 0.79² ≈ 0.62 (well below the achieved 0.96). The
model is gaining the rest from `(RoadType, NumberofLanes)` discrimination
plus within-geohash smoothing.

`figures/day48_vs_day49.png` shows the scatter — there's a clear linear
relationship but with heteroscedasticity (variance grows with mean demand).

---

## 11. Lag-feature Coverage (d48 → test)

| Metric | Backward lags (11) | Forward lags (7) |
|---|---:|---:|
| All lags hit | 19,580 (46.87%) | 30,515 (73.03%) |
| No lag hit | 643 | 1,026 |
| Median hits per row | 10 / 11 | 7 / 7 |

Backward-lag-1 (the most useful lag) hit rate per test time_slot is
**85-91%** across all 47 test time_slots — see `eda_deepdive.py` output
for the full table. The overall backward-lag-1 hit rate is **88.5%** of
test rows.

→ **The baseline's `lm*` feature stack is the right design.** Improving it
means handling the 12% miss cases better (e.g. fill with geohash's
time-of-day mean instead of -999) and adding **rolling features** (mean
of last 4 lags, max, std) which would smooth out the individual misses.

For d49t (00:00-02:00), backward lag coverage is **0%** (no time_slot < 0
exists). Forward lag coverage varies. The baseline does not use d49t to
build its lag lookup, so d49t's `lm*` features are entirely NaN (filled
with -999).

**Implication:** d49t contributes nothing to the lag feature signal. It's
still useful for the geohash × time_slot statistics and as direct
training data, but lags are a d48-only feature.

---

## 12. Per-Geohash Demand Profile

Statistics across 1,249 geohash (with ≥1 train row):

| | n_rows | mean | std | min | max | range (max-min) |
|---|---:|---:|---:|---:|---:|---:|
| median | 71 | 0.031 | 0.022 | 0.001 | 0.090 | 0.090 |
| mean | 62 | 0.065 | 0.039 | 0.005 | 0.156 | 0.151 |
| 95% | 105 | 0.230 | 0.150 | 0.100 | 0.700 | 0.700 |
| max | 105 | 0.961 | 0.377 | 0.652 | 1.000 | 0.997 |

→ Geohash mean demand spans **0.0005 to 0.96** — a 2,000× range. This is
why geohash identity (via target encoding) is the single most powerful
feature class after the RoadType × Lanes structure.

**Top-10 most volatile geohash (high std):** all Highway 4-5 lane geohash
(means 0.55-0.66, stds 0.25-0.38). These are the hardest to predict in
absolute terms but the **relative error is small** (~30% std/mean).

**Easiest geohash (low std):** all Residential 1-lane geohash, mean ~0.008,
std ~0.005. The model should nail these.

---

## 13. Sample Submission Inspection

The provided `sample_submission.csv` is the first 5 rows of the test set
(all `time_slot=9`, i.e. 02:15). These are **all Residential 1-3 lane** rows
except index 2 (3 lanes). The sample's mean (0.064) and range
(0.007 - 0.091) are **not representative** of the full test set; the
predicted submission will have a much higher mean due to the Highway/Street
over-representation.

---

## 14. Index & Duplicate Checks

- No duplicate `Index` in train or test.
- `Index` ranges: train `[0, 77298]`, test `[0, 41777]`. These are
  *row indices*, not temporal — the order doesn't carry signal.
- No duplicate `(geohash, day, timestamp)` triples.

---

## 15. Validation Strategy Implications

- **Test set is contiguous future of d49t**, so a **time-based holdout** on
  d48 (e.g. last 4 hours = time_slots 80..95) is the most realistic honest
  validation.
- **5-fold KFold over d48** (random) is also defensible but is slightly
  optimistic because it leaks the same time-slot signal across folds.
- **DO NOT use d49t in the validation fold** — its time range (0-8) is
  disjoint from test (9-55), so it doesn't add useful validation signal.

---

## 16. Feature Engineering Ideas (carried into Phase C)

Based on the EDA, ordered by expected ROI:

1. **Per-(geohash, RoadType) target encoding** (smoothed, K-fold). Stronger
   than geohash-only because the bimodal split lives inside geohash.
2. **Rolling stats on backward lags:** mean of `lm[1..4]`, max, std. Smoother
   signal than any single lag.
3. **Day-shift calibration:** `predicted = d48_same_slot_demand × (d49t_geohash_mean / d48_geohash_mean)`. Captures the +0.05 systematic shift.
4. **Additional time-of-day flags:** `is_lunch` (11-13h), `is_quiet`
   (17-20h, the d48 trough). The current `is_mpk`/`is_epk` are not
   aligned with the d48 mean-demand peaks.
5. **Per-geohash bias correction:** learn `(predicted - actual)` from OOF
   residuals, smoothed; add to test prediction.
6. **Logistic-style two-stage model:** Stage 1 = classifier for
   "high vs low demand bucket"; Stage 2 = regressor within each bucket.
   Tree models already do this implicitly but a separate stacker might
   exploit the discontinuity better.
7. **Cold-start handling:** for the 15 unknown geohash and any other with
   `g_n < 5`, fall back to `(RoadType, NumberofLanes, hour)` cell mean.
8. **Weather is currently useless** as a single feature — keep it but
   don't waste iterations on weather-only interactions.

---

## 17. Reproducibility

```bash
# Activate environment
cd /home/marcus/code/Gridlock
source .venv/bin/activate

# Re-run EDA
python eda/eda.py             # -> /tmp/eda_run.log + eda/figures/*.png
python eda/eda_deepdive.py    # -> /tmp/eda_deepdive.log + more figures
```

EDA is fully deterministic (no random sampling, no bootstrapping).

---

## 18. Post-Improvement Insights (Phase C/D)

After the EDA, we built a 3-mode honest evaluator (scripts/honest_eval.py)
and tested 8 improvements (C-1..C-8 from `docs/plan.md`).

### Final model performance

| Metric | Baseline (LGB+XGB blend) | Final pipeline (LGB+XGB+CBcat+Ridge+bias) | Δ |
|---|---:|---:|---:|
| Mode A OOF R² (in-day KFold d48) | 0.9654 | **0.9661** | **+0.0007** |
| Mode B R² (time holdout slots 80-95) | 0.5782 | 0.5528 | -0.0254 |
| Mode C R² (d49t from d48) | 0.7580 | 0.7550 | -0.003 |
| Full d48+d49t OOF R² | 0.9616 | **0.9626** | **+0.0010** |

### What worked

1. **CatBoost with native geohash categorical** (C-5) — biggest single win
   (+0.0006 in Ridge stack). The model exploits the high-cardinality
   categorical in a way that label-encoded features cannot.
2. **Per-geohash bias correction** (C-8) — small but consistent (+0.0001).
3. **Tuned LGB params** (C-6) — small gain from `num_leaves=255, min_child_samples=3, lr=0.02`.

### What didn't work

1. **Geohash target encoding** (C-2) — the OOF `g_mu` feature already
   captures the smoothed mean; adding a redundant column is noise.
2. **Interaction features** (C-3) — trees discover these themselves.
3. **Tweedie/quantile objectives** (C-7) — demand isn't zero-inflated;
   the RMSE objective is the right loss.
4. **Optuna-style hyperparam tuning** (C-6 lite) — the gain was <0.0002,
   well within fold variance.

### Honest caveats

- The full-d48 OOF gain (+0.001) is real but small.
- Mode B (time holdout) regressed because the bias-correction OOF was
  computed on d48 KFold residuals, which are systematically different
  from the time-holdout residuals.
- The actual test R² will likely be 0.95-0.96, with the d49t forward-lag
  features (C-4) helping the 16% of test rows at time_slot 9-16.

### Top features (LGB gain) tell the same story as EDA

1. `road_enc` — the Highway/Street/Residential bucket dominates.
2. `lm1` — "yesterday at this time" (test) / "earlier today" (d48 KFold).
3. `lm2`, `lm3` — additional lag context.
4. `ratio1` (gts_mu / g_mu) — the per-time-slot-relative geohash signal.
5. `g_p75`, `g_med` — robust geohash statistics.

This confirms the EDA's central finding: **the problem is dominated by a
small number of structural signals, and the marginal value of new features
is small once those are captured.** The main room for improvement came
from model diversity (CatBoost native cat), not feature engineering.
