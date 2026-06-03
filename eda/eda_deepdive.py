"""
Additional deep-dive EDA analyses:

- RoadType x NumberofLanes interaction (the dominant demand driver)
- Test-set geohash profile vs train (covariate shift)
- Distribution of demand by (geohash, time_slot) and by time-of-day
- Sample submission comparison
- Lag-feature coverage maps
- Per-geohash variability clusters
- Cold-start geohash analysis (the 15 not in d48)
- Identify if test set is "future" of train in time
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

ROOT = Path("/home/marcus/code/Gridlock")
DATA = ROOT / "dataset"
FIGS = ROOT / "output" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)


def parse_time(df):
    s = df["timestamp"].astype(str).str.strip().str.split(":", expand=True)
    h = pd.to_numeric(s[0], errors="coerce").fillna(0).astype(int)
    m = pd.to_numeric(s[1], errors="coerce").fillna(0).astype(int)
    return h * 60 + m, (h * 60 + m) // 15


def section(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
sample_sub = pd.read_csv(DATA / "sample_submission.csv")
train["ts_min"], train["time_slot"] = parse_time(train)
test["ts_min"], test["time_slot"] = parse_time(test)
train["hr"] = train["ts_min"] // 60
test["hr"] = test["ts_min"] // 60
d48 = train[train["day"] == 48].copy()
d49t = train[train["day"] == 49].copy()

# =================================================================
# 1. RoadType x NumberofLanes — the dominant demand signal
# =================================================================
section("A. RoadType x NumberofLanes — joint demand distribution")
ct = train.groupby(["RoadType", "NumberofLanes"], dropna=False)["demand"].agg(
    ["count", "mean", "std"]
).round(5)
print(ct)
print()
# Pretty pivot of mean
pv = train.pivot_table(
    values="demand", index="RoadType", columns="NumberofLanes", aggfunc="mean"
).round(4)
print("Mean demand pivot (RoadType x Lanes):")
print(pv)

# =================================================================
# 2. Test-set distribution shift
# =================================================================
section("B. Test vs Train demand-driver distribution")
print("\n#1 RoadType distribution:")
for d, name in [(train, "train"), (test, "test")]:
    pct = d["RoadType"].value_counts(dropna=False, normalize=True).round(4) * 100
    print(f"  {name}: {pct.to_dict()}")

print("\n#2 NumberofLanes distribution:")
for d, name in [(train, "train"), (test, "test")]:
    pct = d["NumberofLanes"].value_counts(dropna=False, normalize=True).round(4) * 100
    print(f"  {name}: {pct.to_dict()}")

print("\n#3 (RoadType, NumberofLanes) joint distribution:")
ct_tr = train.groupby(["RoadType", "NumberofLanes"], dropna=False).size()
ct_te = test.groupby(["RoadType", "NumberofLanes"], dropna=False).size()
both = pd.DataFrame({"train_n": ct_tr, "test_n": ct_te}).fillna(0).astype(int)
both["total"] = both["train_n"] + both["test_n"]
both["train_pct"] = (both["train_n"] / both["train_n"].sum() * 100).round(3)
both["test_pct"] = (both["test_n"] / both["test_n"].sum() * 100).round(3)
both["delta_pct"] = (both["test_pct"] - both["train_pct"]).round(3)
both = both.sort_values("test_pct", ascending=False)
print(both)

# =================================================================
# 3. Demand histogram by RoadType
# =================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, rt in enumerate(["Residential", "Street", "Highway"]):
    sub = train[train["RoadType"] == rt]["demand"].dropna()
    axes[i].hist(sub, bins=40, color=["steelblue", "darkorange", "seagreen"][i], edgecolor="white")
    axes[i].set_title(f"RoadType = {rt}  (n={len(sub):,})")
    axes[i].set_xlabel("demand")
    axes[i].set_ylabel("count")
    axes[i].axvline(sub.mean(), color="red", linestyle="--", alpha=0.6, label=f"mean={sub.mean():.3f}")
    axes[i].legend()
fig.tight_layout()
fig.savefig(FIGS / "demand_by_roadtype.png", dpi=110)
plt.close(fig)

# =================================================================
# 4. Demand by (RoadType, Lanes) — boxplots
# =================================================================
fig, ax = plt.subplots(figsize=(12, 5))
train.boxplot(column="demand", by=["RoadType", "NumberofLanes"], ax=ax, rot=45)
ax.set_title("Demand by (RoadType, NumberofLanes)")
ax.set_ylabel("demand")
fig.suptitle("")
fig.tight_layout()
fig.savefig(FIGS / "demand_by_roadtype_lanes.png", dpi=110)
plt.close(fig)

# =================================================================
# 5. Sample submission vs train demand
# =================================================================
section("C. Sample submission pattern")
print(sample_sub)
print(f"\nMean sample demand: {sample_sub['demand'].mean():.5f}")
print(f"Std  sample demand: {sample_sub['demand'].std():.5f}")

# The sample submission has 5 rows. What do those indices correspond to in test?
print("\nFirst 5 test rows (matching sample_submission Index=0..4):")
print(test.head().to_string())

# Demand stats of test rows at the matching (geohash, time_slot) from train d48
print("\nFor those 5 (geohash, time_slot) on d48, what is the historical demand?")
for idx in range(5):
    r = test.iloc[idx]
    gh, ts = r["geohash"], r["time_slot"]
    hist = d48[(d48["geohash"] == gh) & (d48["time_slot"] == ts)]["demand"]
    print(f"  Index {idx}  geohash={gh}  time_slot={ts}  d48 demand = "
          f"{hist.values if len(hist) else 'NOT FOUND'}")

# =================================================================
# 6. Time-slot test layout
# =================================================================
section("D. Test time-slot distribution")
ts_counts = test["time_slot"].value_counts().sort_index()
print(f"Test time_slot range: {ts_counts.index.min()} to {ts_counts.index.max()}")
print(f"Test time_slot unique count: {len(ts_counts)}")
print("\nTest time_slot counts:")
print(ts_counts)

# Visualize
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(ts_counts.index, ts_counts.values, color="steelblue", edgecolor="white")
ax.set_xlabel("time_slot (0=00:00, 95=23:45)")
ax.set_ylabel("# test rows")
ax.set_title("Test row count by 15-min time slot")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIGS / "test_timeslot_distribution.png", dpi=110)
plt.close(fig)

# Same for d48
ts_counts_d48 = d48["time_slot"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(ts_counts_d48.index, ts_counts_d48.values, color="darkorange", edgecolor="white")
ax.set_xlabel("time_slot")
ax.set_ylabel("# d48 rows")
ax.set_title("d48 row count by 15-min time slot")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIGS / "d48_timeslot_distribution.png", dpi=110)
plt.close(fig)

# =================================================================
# 7. Lag coverage on TEST — d48 lookup detail
# =================================================================
section("E. Test lag coverage — d48 lookup")
lm = d48.set_index(["geohash", "time_slot"])["demand"]

BACK = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24]
FWD = [1, 2, 3, 4, 5, 6, 8]

# For test, how many of the lags hit per (geohash, time_slot) "neighborhood"?
# E.g. test rows at time_slot=9 need lags 8,7,6,5,4,3,1,-2,-7,-15,-23 -- some negative
# Most test rows are at time_slot 9-55, so lags 1-8 backward should hit easily

# Time slot distribution of test:
test_ts_dist = test["time_slot"].value_counts().sort_index()
hit_pct = []
for ts in test_ts_dist.index:
    n = test_ts_dist[ts]
    n_hit = sum(
        1
        for _, r in test[test["time_slot"] == ts].iterrows()
        if not np.isnan(lm.get((r["geohash"], r["time_slot"] - 1), np.nan))
    )
    hit_pct.append((ts, n, n_hit, n_hit / n * 100))
print("\nTest rows: backward-lag-1 hit rate by time_slot:")
print(f"  {'time_slot':>10} {'n_rows':>8} {'n_hit':>8} {'pct':>8}")
for ts, n, h, p in hit_pct:
    print(f"  {ts:>10} {n:>8} {h:>8} {p:>7.1f}%")

# =================================================================
# 8. Per-geohash mean demand vs count — does the rare-geohash get noisy means?
# =================================================================
section("F. Geohash count vs demand mean")
g_stats = train.groupby("geohash")["demand"].agg(g_n="count", g_mean="mean").reset_index()
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(g_stats["g_n"], g_stats["g_mean"], s=8, alpha=0.4, color="steelblue")
ax.set_xlabel("# rows per geohash")
ax.set_ylabel("mean demand")
ax.set_title("Geohash: row count vs mean demand")
ax.set_xscale("log")
ax.axhline(g_stats["g_mean"].mean(), color="red", linestyle="--", alpha=0.6, label="overall mean")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIGS / "geohash_count_vs_mean.png", dpi=110)
plt.close(fig)

# =================================================================
# 9. Test geohash cold-start — what features do they have?
# =================================================================
section("G. Test cold-start geohash (not in d48)")
g_d48 = set(d48["geohash"].unique())
g_test = set(test["geohash"].unique())
cold = g_test - g_d48
print(f"# cold-start geohash: {len(cold)}")
print(f"They are: {sorted(cold)}")
cold_test = test[test["geohash"].isin(cold)]
print(f"\n# test rows for cold-start geohash: {len(cold_test)}  ({len(cold_test)/len(test)*100:.2f}%)")
print("\nCold-start geohash feature distribution:")
for col in ["RoadType", "NumberofLanes", "LargeVehicles", "Landmarks", "Weather"]:
    print(f"  {col}: {cold_test[col].value_counts(dropna=False).to_dict()}")

# =================================================================
# 10. Day-49 training rows: what are they like?
# =================================================================
section("H. Day 49 train rows (d49t) — quick look")
print(f"d49t shape: {d49t.shape}")
print(f"d49t time_slot range: {d49t['time_slot'].min()} to {d49t['time_slot'].max()}")
print(f"d49t unique geohash: {d49t['geohash'].nunique()}")
print(f"d49t mean demand: {d49t['demand'].mean():.5f}")
print(f"d48  mean demand: {d48['demand'].mean():.5f}")
print(f"test mean demand estimate (from d48 same time_slot+geohash):")
# For each test row, what is the d48 same (geohash, time_slot) demand?
test["d48_demand"] = test.apply(
    lambda r: lm.get((r["geohash"], r["time_slot"]), np.nan), axis=1
)
print(f"  mean: {test['d48_demand'].mean():.5f}  (NaN count: {test['d48_demand'].isna().sum()})")
print(f"  median: {test['d48_demand'].median():.5f}")
print(f"  std: {test['d48_demand'].std():.5f}")

# =================================================================
# 11. Is test "future" of d49t?
# =================================================================
section("I. Time contiguity: d49t -> test")
print(f"d49t max time_slot : {d49t['time_slot'].max()}  ({d49t['time_slot'].max()*15//60}h{(d49t['time_slot'].max()*15)%60:02d})")
print(f"test  min time_slot : {test['time_slot'].min()}    ({test['time_slot'].min()*15//60}h{(test['time_slot'].min()*15)%60:02d})")
print(f"test  max time_slot : {test['time_slot'].max()}  ({test['time_slot'].max()*15//60}h{(test['time_slot'].max()*15)%60:02d})")
print(f"\n=> d49t covers 00:00-02:00, test covers 02:15-13:45 — contiguous (with 15-min gap).")

# =================================================================
# 12. Test geohash — high-demand vs low-demand mix
# =================================================================
section("J. Test geohash demand-anchor (using d48 mean as proxy)")
g_means = d48.groupby("geohash")["demand"].mean()
test_g_mean = test["geohash"].map(g_means)
print(f"Test rows: d48 geohash mean demand distribution (rows where geohash is in d48):")
print(test_g_mean.describe().round(5))
print(f"  fraction of test rows with d48 geohash mean > 0.5: {(test_g_mean > 0.5).mean():.4f}")
print(f"  fraction of test rows with d48 geohash mean > 0.1: {(test_g_mean > 0.1).mean():.4f}")

# =================================================================
# 13. What fraction of test demand is "easy" (low baseline residual)?
# =================================================================
section("K. Naive baseline: predict with d48 same (geohash, time_slot)")
# Build a simple OLS-naive prediction
test["naive_d48"] = test["d48_demand"]
# Compare naive to d49t same-(geohash, time_slot) on common cells
# (We saw corr = 0.79 earlier, so naive is a strong baseline)
print(f"If we predicted test = d48 (geohash, time_slot) demand directly:")
print(f"  -> this is essentially a 1-day-lag baseline")
print(f"  -> expected R² ~ corr(d48, d49t)^2 = {0.79**2:.4f} (Naive correlation baseline)")

# =================================================================
# 14. Geohash with HIGH temporal variability (likely hard to predict)
# =================================================================
section("L. Hard geohash — high temporal std")
g_stats = train.groupby("geohash")["demand"].agg(g_n="count", g_mean="mean", g_std="std").reset_index()
g_stats = g_stats[g_stats["g_n"] >= 20]  # require some data
hard = g_stats.sort_values("g_std", ascending=False).head(15)
print("Top-15 most volatile geohash (std of demand):")
print(hard.round(4).to_string())
print("\nEasy geohash (low std, high count):")
easy = g_stats[g_stats["g_n"] >= 50].sort_values("g_std", ascending=True).head(15)
print(easy.round(4).to_string())

print("\nDeep-dive EDA complete.")
