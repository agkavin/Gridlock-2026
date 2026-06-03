"""
Phase A — Verify & EDA for the Gridlock traffic-demand dataset.

Outputs:
  - prints a structured report to stdout
  - writes all numeric tables to eda/eda_summary.json
  - saves figures to eda/figures/*.png

Usage (from Gridlock/):
    uv run python eda/eda.py
"""

import json
import os
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

ROOT = Path("/home/marcus/code/Gridlock")
DATA = ROOT / "dataset"
OUTPUT = ROOT / "output"
FIGS = OUTPUT / "figures"
OUTPUT.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)


def parse_time(df):
    s = df["timestamp"].astype(str).str.strip().str.split(":", expand=True)
    h = pd.to_numeric(s[0], errors="coerce").fillna(0).astype(int)
    m = pd.to_numeric(s[1], errors="coerce").fillna(0).astype(int)
    return h * 60 + m, (h * 60 + m) // 15


def section(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def save_json(obj, name):
    out = OUTPUT / name
    with open(out, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"  -> wrote {out.relative_to(ROOT)}")


# =================================================================
# LOAD
# =================================================================
section("1. LOAD DATA")
train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
sample_sub = pd.read_csv(DATA / "sample_submission.csv")

print(f"train: {train.shape}")
print(f"test : {test.shape}")
print(f"sample_submission: {sample_sub.shape}")
print(f"train cols : {list(train.columns)}")
print(f"test  cols : {list(test.columns)}")
print(f"sample cols: {list(sample_sub.columns)}")

# =================================================================
# DTYPES & NaN COUNTS
# =================================================================
section("2. DTYPES & NaN COUNTS")


def dtype_nan_table(df, name):
    d = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "n_nan": df.isna().sum(),
            "pct_nan": (df.isna().mean() * 100).round(3),
            "n_unique": df.nunique(),
        }
    )
    print(f"\n--- {name} ---")
    print(d.to_string())
    return d


t_dt = dtype_nan_table(train, "TRAIN")
v_dt = dtype_nan_table(test, "TEST")

# =================================================================
# DEMAND DISTRIBUTION
# =================================================================
section("3. DEMAND DISTRIBUTION (TRAIN)")

y = train["demand"].astype(float)
desc = y.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).round(6)
print(desc)

print(f"\nNegative values: {(y < 0).sum()}")
print(f"Zero values   : {(y == 0).sum()}  ({(y == 0).mean()*100:.3f}%)")
print(f"demand > 1    : {(y > 1).sum()}")
print(f"NaN demand    : {y.isna().sum()}")

# Compare to sample_submission distribution
sub_dem = sample_sub["demand"].astype(float)
print(f"\nSample submission demand range: [{sub_dem.min():.6f}, {sub_dem.max():.6f}]")
print(f"Sample submission demand mean : {sub_dem.mean():.6f}, std: {sub_dem.std():.6f}")

# Histogram
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
ax[0].hist(y.dropna(), bins=80, color="steelblue", edgecolor="white")
ax[0].set_title("Train demand — full distribution")
ax[0].set_xlabel("demand")
ax[0].set_ylabel("count")
ax[1].hist(y.dropna(), bins=80, color="steelblue", edgecolor="white")
ax[1].set_yscale("log")
ax[1].set_title("Train demand — log-y")
ax[1].set_xlabel("demand")
ax[1].set_ylabel("count (log)")
fig.tight_layout()
fig.savefig(FIGS / "demand_distribution.png", dpi=110)
plt.close(fig)

# Boxplot per day
fig, ax = plt.subplots(figsize=(6, 4))
train.boxplot(column="demand", by="day", ax=ax)
ax.set_title("Demand by day")
ax.set_ylabel("demand")
fig.suptitle("")
fig.tight_layout()
fig.savefig(FIGS / "demand_by_day.png", dpi=110)
plt.close(fig)

# =================================================================
# GEOHASH CARDINALITY & OVERLAP
# =================================================================
section("4. GEOHASH CARDINALITY & OVERLAP")

# Apply same splits as baseline
train["ts_min"], train["time_slot"] = parse_time(train)
test["ts_min"], test["time_slot"] = parse_time(test)
train["hr"] = train["ts_min"] // 60
test["hr"] = test["ts_min"] // 60
d48 = train[train["day"] == 48].copy()
d49t = train[train["day"] == 49].copy()

g_d48 = set(d48["geohash"].unique())
g_d49t = set(d49t["geohash"].unique())
g_test = set(test["geohash"].unique())
g_all = g_d48 | g_d49t | g_test

print(f"unique geohash in d48   : {len(g_d48)}")
print(f"unique geohash in d49t  : {len(g_d49t)}")
print(f"unique geohash in test  : {len(g_test)}")
print(f"union (d48|d49t|test)   : {len(g_all)}")
print(f"d48 ∩ d49t              : {len(g_d48 & g_d49t)}")
print(f"d48 ∩ test              : {len(g_d48 & g_test)}")
print(f"d49t ∩ test             : {len(g_d49t & g_test)}")
print(f"d48 ∩ d49t ∩ test       : {len(g_d48 & g_d49t & g_test)}")
print(f"test geohash NOT in d48 : {len(g_test - g_d48)}  (cold-start geohash in test)")
print(f"test geohash NOT in d48∪d49t: {len(g_test - g_d48 - g_d49t)}")

# How many test rows have a geohash NOT seen in d48 (lag features will be NaN)
test_in_d48 = test["geohash"].isin(g_d48)
print(f"\nTest rows whose geohash exists in d48 (lag coverage): "
      f"{test_in_d48.sum()} / {len(test)} ({test_in_d48.mean()*100:.2f}%)")

# Geohash sample frequency
g_freq_train = train["geohash"].value_counts()
print(f"\nTrain: geohash row-count distribution")
print(g_freq_train.describe().round(2))
g_freq_test = test["geohash"].value_counts()
print(f"\nTest : geohash row-count distribution")
print(g_freq_test.describe().round(2))

# Top-N most-frequent geohash
print("\nTop-10 most-frequent geohash in TRAIN:")
print(g_freq_train.head(10))
print("\nTop-10 most-frequent geohash in TEST:")
print(g_freq_test.head(10))

# =================================================================
# TIME-SLOT COVERAGE
# =================================================================
section("5. TIME-SLOT & TIMESTAMP COVERAGE")

print("\nTrain timestamp (string) unique values (top 20):")
print(train["timestamp"].value_counts().head(20))

print("\nTrain time_slot range :", train["time_slot"].min(), "to", train["time_slot"].max())
print("Test  time_slot range :", test["time_slot"].min(), "to", test["time_slot"].max())

# How many time slots exist per day
ts_d48 = sorted(d48["time_slot"].unique())
ts_d49t = sorted(d49t["time_slot"].unique())
ts_test = sorted(test["time_slot"].unique())
print(f"\n# distinct time_slots in d48   : {len(ts_d48)} (range {ts_d48[0]}..{ts_d48[-1]})")
print(f"# distinct time_slots in d49t  : {len(ts_d49t)} (range {ts_d49t[0]}..{ts_d49t[-1]})")
print(f"# distinct time_slots in test  : {len(ts_test)} (range {ts_test[0]}..{ts_test[-1]})")

# Coverage: do we have all 96 15-min slots in each subset?
expected = set(range(96))
print(f"\nd48  missing time_slots  : {len(expected - set(ts_d48))} of 96")
print(f"d49t missing time_slots  : {len(expected - set(ts_d49t))} of 96")
print(f"test missing time_slots  : {len(expected - set(ts_test))} of 96")

# Plot of time-slot vs mean demand
ds = train.groupby("time_slot")["demand"].agg(["mean", "std", "count"])
fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
ax[0].plot(ds.index, ds["mean"], color="darkorange")
ax[0].set_ylabel("mean demand")
ax[0].set_title("Mean demand by 15-min time slot (all train)")
ax[0].grid(True, alpha=0.3)
ax[1].bar(ds.index, ds["count"], color="steelblue")
ax[1].set_xlabel("time_slot (0=00:00, 95=23:45)")
ax[1].set_ylabel("# rows")
ax[1].grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIGS / "demand_by_timeslot.png", dpi=110)
plt.close(fig)

# Demand by hour
dh = train.groupby("hr")["demand"].agg(["mean", "std", "count"])
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(dh.index, dh["mean"], yerr=dh["std"], color="teal", alpha=0.7)
ax.set_xlabel("hour of day")
ax.set_ylabel("mean demand (± std)")
ax.set_title("Demand by hour of day (train)")
ax.set_xticks(range(0, 24))
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIGS / "demand_by_hour.png", dpi=110)
plt.close(fig)

# Day 48 vs Day 49 demand shape per hour
hr_d48 = d48.groupby("hr")["demand"].mean()
hr_d49t = d49t.groupby("hr")["demand"].mean()
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(hr_d48.index, hr_d48.values, marker="o", label="day 48", color="darkorange")
ax.plot(hr_d49t.index, hr_d49t.values, marker="s", label="day 49 (train rows)", color="steelblue")
ax.set_xlabel("hour of day")
ax.set_ylabel("mean demand")
ax.set_title("Mean demand by hour: day 48 vs day 49 (train portion)")
ax.set_xticks(range(0, 24))
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIGS / "demand_by_hour_day_compare.png", dpi=110)
plt.close(fig)

print("\nMean demand by hour (d48 vs d49t):")
cmp_df = pd.DataFrame({"d48": hr_d48, "d49t": hr_d49t}).round(5)
print(cmp_df.to_string())

# =================================================================
# MISSINGNESS
# =================================================================
section("6. MISSINGNESS PATTERN")

miss_cols = ["RoadType", "NumberofLanes", "LargeVehicles", "Landmarks", "Temperature", "Weather"]
for col in miss_cols:
    tr_nan = train[col].isna().mean() * 100
    te_nan = test[col].isna().mean() * 100
    print(f"  {col:<16} train: {tr_nan:6.3f}% NaN  |  test: {te_nan:6.3f}% NaN")

# Joint missingness — is the missingness clustered?
print("\nJoint missingness in TRAIN (top combos):")
train_miss = train[miss_cols].isna()
train_miss["n_missing"] = train_miss.sum(axis=1)
print(train_miss["n_missing"].value_counts().sort_index())
print("\nJoint missingness in TEST (top combos):")
test_miss = test[miss_cols].isna()
test_miss["n_missing"] = test_miss.sum(axis=1)
print(test_miss["n_missing"].value_counts().sort_index())

# Are NaNs clustered in specific (geohash, time) cells?  i.e. structural vs random
# Quick test: distribution of demand for rows that have any NaN vs none
train["any_nan"] = train[miss_cols].isna().any(axis=1)
print("\nMean demand — rows with any missing feature vs none:")
print(train.groupby("any_nan")["demand"].agg(["count", "mean", "std"]).round(5))

# How does the NaN pattern look across (day, time_slot)?
print("\nNaN rate of ANY feature by day (train):")
print(train.groupby("day")[miss_cols].apply(lambda d: d.isna().mean().mean()).round(4))

# Specifically: for the first 30 test rows, what does the data look like?
print("\nFirst 5 test rows where Weather is NaN:")
print(test[test["Weather"].isna()].head())
print("\nFirst 5 train rows where RoadType is NaN:")
print(train[train["RoadType"].isna()].head())

# Is the pattern structural?  Check (geohash, time_slot) clusters
print("\nIs missingness concentrated in specific geohash-time cells?")
nan_cells = train[train["any_nan"]].groupby(["geohash", "time_slot"]).size()
total_cells = train.groupby(["geohash", "time_slot"]).size()
ratio = (nan_cells / total_cells).fillna(0)
print(f"  # unique (geohash, time_slot) cells in train: {len(total_cells)}")
print(f"  # cells with >=1 NaN row                     : {len(nan_cells)}")
print(f"  cells with 100% NaN rows                    : {(ratio == 1.0).sum()}")
print(f"  cells with 0% NaN rows (none of the 6 cols NaN): {(ratio == 0.0).sum()}")
print(f"  median NaN-rate per cell                    : {ratio.median():.4f}")

# Save missingness summary figure
fig, ax = plt.subplots(figsize=(8, 4))
train_miss_only = train[miss_cols].isna().astype(int)
sns.heatmap(train_miss_only.corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
ax.set_title("Correlation of missingness indicators (TRAIN)")
fig.tight_layout()
fig.savefig(FIGS / "missingness_corr.png", dpi=110)
plt.close(fig)

# =================================================================
# CATEGORICAL FEATURE ANALYSIS
# =================================================================
section("7. CATEGORICAL FEATURE DISTRIBUTIONS & MEAN DEMAND")

cat_cols = ["RoadType", "LargeVehicles", "Landmarks", "Weather"]
for col in cat_cols:
    print(f"\n--- {col} (train) ---")
    vc = train[col].value_counts(dropna=False)
    pct = train[col].value_counts(dropna=False, normalize=True).round(4) * 100
    print(pd.DataFrame({"count": vc, "pct": pct}))

    mean_d = train.groupby(col, dropna=False)["demand"].agg(["mean", "std", "count"]).round(5)
    print(f"\nMean demand by {col} (train):")
    print(mean_d)

# Test distribution differences
print("\n\nCategorical distributions: TRAIN vs TEST (relative %)")
for col in cat_cols:
    tr_pct = train[col].value_counts(dropna=False, normalize=True)
    te_pct = test[col].value_counts(dropna=False, normalize=True)
    cmb = pd.DataFrame({"train_pct": tr_pct, "test_pct": te_pct}).fillna(0)
    cmb["delta_pct"] = (cmb["test_pct"] - cmb["train_pct"]).round(4) * 100
    print(f"\n--- {col} ---")
    print(cmb.round(4))

# Cross-table: weather x road
print("\n\nCross-tab: RoadType x Weather (TRAIN)")
ct = pd.crosstab(train["RoadType"], train["Weather"], dropna=False, normalize="all").round(4)
print(ct)

# Mean demand by combined category
print("\n\nMean demand by (RoadType, Weather):")
print(train.groupby(["RoadType", "Weather"], dropna=False)["demand"].agg(["mean", "std", "count"]).round(5))

# =================================================================
# NUMERICAL FEATURE ANALYSIS
# =================================================================
section("8. NUMERICAL FEATURES")

for col in ["NumberofLanes", "Temperature"]:
    print(f"\n--- {col} (train) ---")
    print(train[col].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).round(3))
    print(f"NaN count: {train[col].isna().sum()}")

# Lanes distribution
print("\nNumberofLanes value counts (train):")
print(train["NumberofLanes"].value_counts(dropna=False).sort_index())

print("\nNumberofLanes value counts (test):")
print(test["NumberofLanes"].value_counts(dropna=False).sort_index())

# Mean demand by NumberofLanes
print("\nMean demand by NumberofLanes (train):")
print(train.groupby("NumberofLanes", dropna=False)["demand"].agg(["mean", "std", "count"]).round(5))

# Mean demand by Temperature bin
train["temp_bin"] = pd.cut(train["Temperature"], bins=10)
print("\nMean demand by Temperature decile (train):")
print(train.groupby("temp_bin", observed=True)["demand"].agg(["mean", "std", "count"]).round(5))

# =================================================================
# CORRELATIONS & FEATURE RELATIONSHIPS
# =================================================================
section("9. CORRELATIONS WITH demand (numeric features)")

num_feats = ["NumberofLanes", "Temperature", "ts_min", "time_slot", "hr"]
corrs = {}
for col in num_feats:
    s = train[[col, "demand"]].dropna()
    if s[col].nunique() > 1:
        c = s["demand"].corr(s[col])
    else:
        c = float("nan")
    corrs[col] = round(c, 4)
print("Pearson correlations with demand:")
print(pd.Series(corrs))

# Spearman (rank-based)
scorrs = {}
for col in num_feats:
    s = train[[col, "demand"]].dropna()
    if s[col].nunique() > 1:
        c = s["demand"].corr(s[col], method="spearman")
    else:
        c = float("nan")
    scorrs[col] = round(c, 4)
print("\nSpearman correlations with demand:")
print(pd.Series(scorrs))

# Group-mean demand at extremes
print("\nMean demand for top-10 hottest geohash:")
top_g = g_freq_train.head(10).index
print(train[train["geohash"].isin(top_g)].groupby("geohash")["demand"].agg(["mean", "std", "count"]).round(5))

# =================================================================
# LAG FEATURE COVERAGE — important for feasibility
# =================================================================
section("10. LAG-FEATURE COVERAGE ON TEST (using d48 lookup)")

# Build d48 lookup as the baseline does
lm = d48.set_index(["geohash", "time_slot"])["demand"]

BACK = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24]
FWD = [1, 2, 3, 4, 5, 6, 8]

# For each test row, count how many of the BACK and FWD lag lookups hit
test_lm = []
test_lp = []
for _, r in test.iterrows():
    cnt_lm = sum(1 for L in BACK if not np.isnan(lm.get((r["geohash"], r["time_slot"] - L), np.nan)))
    cnt_lp = sum(1 for L in FWD if not np.isnan(lm.get((r["geohash"], r["time_slot"] + L), np.nan)))
    test_lm.append(cnt_lm)
    test_lp.append(cnt_lp)

test["n_lm_hits"] = test_lm
test["n_lp_hits"] = test_lp
print(f"Test rows with ALL {len(BACK)} backward lags hit : {(test['n_lm_hits'] == len(BACK)).sum()}  ({(test['n_lm_hits'] == len(BACK)).mean()*100:.2f}%)")
print(f"Test rows with NO backward lag hit               : {(test['n_lm_hits'] == 0).sum()}")
print(f"Test rows with ALL {len(FWD)} forward lags hit  : {(test['n_lp_hits'] == len(FWD)).sum()}")
print(f"Test rows with NO forward lag hit                : {(test['n_lp_hits'] == 0).sum()}")
print(f"\nDistribution of backward-lag hit counts:")
print(test["n_lm_hits"].value_counts().sort_index())
print(f"\nDistribution of forward-lag hit counts:")
print(test["n_lp_hits"].value_counts().sort_index())

# Same for d49t
d49t_lm = []
d49t_lp = []
for _, r in d49t.iterrows():
    cnt_lm = sum(1 for L in BACK if not np.isnan(lm.get((r["geohash"], r["time_slot"] - L), np.nan)))
    cnt_lp = sum(1 for L in FWD if not np.isnan(lm.get((r["geohash"], r["time_slot"] + L), np.nan)))
    d49t_lm.append(cnt_lm)
    d49t_lp.append(cnt_lp)
d49t["n_lm_hits"] = d49t_lm
d49t["n_lp_hits"] = d49t_lp
print(f"\nd49t rows with ALL {len(BACK)} backward lags hit : {(d49t['n_lm_hits'] == len(BACK)).sum()}  ({(d49t['n_lm_hits'] == len(BACK)).mean()*100:.2f}%)")
print(f"d49t rows with NO backward lag hit               : {(d49t['n_lm_hits'] == 0).sum()}")

# For test rows: how many lag features will be non-NaN after the baseline pipeline?
# The baseline fills NaN with -999, so model learns "no lag" as a separate signal.

# =================================================================
# DUPLICATES & INDEX INTEGRITY
# =================================================================
section("11. INDEX & DUPLICATE CHECKS")

print(f"train.Index duplicates: {train['Index'].duplicated().sum()}")
print(f"test.Index duplicates : {test['Index'].duplicated().sum()}")
print(f"train.Index range     : [{train['Index'].min()}, {train['Index'].max()}]")
print(f"test.Index range      : [{test['Index'].min()}, {test['Index'].max()}]")
print(f"sample.Index range    : [{sample_sub['Index'].min()}, {sample_sub['Index'].max()}]")

# Composite key duplicates?
key_cols = ["geohash", "day", "timestamp"]
dup_train = train.duplicated(subset=key_cols).sum()
dup_test = test.duplicated(subset=key_cols).sum()
print(f"\nTRAIN duplicate (geohash, day, timestamp): {dup_train}")
print(f"TEST  duplicate (geohash, day, timestamp): {dup_test}")

# =================================================================
# GEOHASH DEMAND PROFILE
# =================================================================
section("12. GEOHASH DEMAND PROFILE")

g_stats = (
    train.groupby("geohash")["demand"]
    .agg(g_n="count", g_mean="mean", g_std="std", g_min="min", g_max="max")
    .round(5)
)
print(f"\nGeohash demand-stats summary (across {len(g_stats)} geohash):")
print(g_stats.describe().round(5))
print(f"\nTop-10 highest-demand geohash (by mean):")
print(g_stats.sort_values("g_mean", ascending=False).head(10))
print(f"\nTop-10 lowest-demand geohash (by mean):")
print(g_stats.sort_values("g_mean", ascending=True).head(10))
print(f"\nGeohash demand-std distribution (signal of variability):")
print(g_stats["g_std"].describe().round(5))
print(f"# geohash with std==0 (constant demand): {(g_stats['g_std'] == 0).sum()}")

# Demand range per geohash (max - min)
g_stats["range"] = g_stats["g_max"] - g_stats["g_min"]
print(f"\nGeohash demand range (max-min) distribution:")
print(g_stats["range"].describe().round(5))

# =================================================================
# TEMPORAL STABILITY: same geohash, same time_slot, d48 vs d49t
# =================================================================
section("13. DAY-TO-DAY STABILITY: same (geohash, time_slot) on d48 vs d49t")

# Inner join on (geohash, time_slot)
a = d48.groupby(["geohash", "time_slot"])["demand"].mean().rename("d48")
b = d49t.groupby(["geohash", "time_slot"])["demand"].mean().rename("d49t")
common = pd.concat([a, b], axis=1, join="inner").dropna()
common["delta"] = common["d49t"] - common["d48"]
print(f"Common (geohash, time_slot) cells across d48 & d49t: {len(common)}")
print(f"  corr(d48, d49t) = {common['d48'].corr(common['d49t']):.4f}")
print(f"  mean delta      = {common['delta'].mean():.5f}")
print(f"  std  delta      = {common['delta'].std():.5f}")
print(f"  abs(delta) p50  = {common['delta'].abs().median():.5f}")
print(f"  abs(delta) p90  = {common['delta'].abs().quantile(0.9):.5f}")
print(f"  abs(delta) p99  = {common['delta'].abs().quantile(0.99):.5f}")
print("\nQuantiles of demand on d48 vs d49t (common cells):")
print(common.describe(percentiles=[0.01, 0.5, 0.99]).round(5))

# Save scatter
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(common["d48"], common["d49t"], s=4, alpha=0.3, color="steelblue")
mx = max(common["d48"].max(), common["d49t"].max())
ax.plot([0, mx], [0, mx], "r--", alpha=0.6, label="y=x")
ax.set_xlabel("mean demand — day 48")
ax.set_ylabel("mean demand — day 49 (train portion)")
ax.set_title(f"Day-48 vs day-49 demand (n={len(common)} cells)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIGS / "day48_vs_day49.png", dpi=110)
plt.close(fig)

# =================================================================
# ALL FILES SAVED
# =================================================================
section("14. SAVED FILES")
print(f"Figures written to: {FIGS}")
for f in sorted(FIGS.iterdir()):
    print(f"  {f.name}")

print("\nEDA complete.")
