"""Sanity check submission_v2.csv before platform upload."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/marcus/code/Gridlock")
DATA = ROOT / "dataset"

test = pd.read_csv(DATA / "test.csv")
v1 = pd.read_csv(ROOT / "submission_v1.csv")
v2 = pd.read_csv(ROOT / "submission_v2.csv")
baseline = pd.read_csv(ROOT / "submission_baseline.csv")

print("=" * 70)
print("V2 SPEC CHECK")
print("=" * 70)
print(f"Shape: {v2.shape}  (expected (41778, 2))")
print(f"Columns: {list(v2.columns)}  (expected ['Index', 'demand'])")
print(f"Index monotonic: {v2['Index'].is_monotonic_increasing}")
print(f"Index matches test['Index'] exactly: {(v2['Index'].values == test['Index'].values).all()}")
print(f"Index range: {v2['Index'].min()} .. {v2['Index'].max()}  (expected 0 .. 41777)")
print(f"NaN in demand: {v2['demand'].isna().any()}")
print(f"Demand range: [{v2['demand'].min():.6f}, {v2['demand'].max():.6f}]  (must be in [0, 1])")
print(f"Demand in [0,1]: {v2['demand'].between(0, 1).all()}")

print("\n" + "=" * 70)
print("V2 DEMAND STATS")
print("=" * 70)
print(f"mean:  {v2['demand'].mean():.6f}")
print(f"std:   {v2['demand'].std():.6f}")
print(f"min:   {v2['demand'].min():.6f}")
print(f"p25:   {v2['demand'].quantile(0.25):.6f}")
print(f"p50:   {v2['demand'].quantile(0.50):.6f}")
print(f"p75:   {v2['demand'].quantile(0.75):.6f}")
print(f"p99:   {v2['demand'].quantile(0.99):.6f}")
print(f"max:   {v2['demand'].max():.6f}")
print(f"frac > 0.5: {(v2['demand'] > 0.5).mean():.4f}")
print(f"frac > 0.9: {(v2['demand'] > 0.9).mean():.4f}")
print(f"frac == 1: {(v2['demand'] == 1.0).mean():.4f}")
print(f"frac < 0.01: {(v2['demand'] < 0.01).mean():.4f}")

print("\n" + "=" * 70)
print("V2 vs V1 (89.97) COMPARISON")
print("=" * 70)
print(f"V1 mean: {v1['demand'].mean():.6f}, std: {v1['demand'].std():.6f}")
print(f"V2 mean: {v2['demand'].mean():.6f}, std: {v2['demand'].std():.6f}")
print(f"V1→V2 mean diff:  {(v2['demand'] - v1['demand']).mean():+.6f}")
print(f"V1↔V2 correlation: {v2['demand'].corr(v1['demand']):.6f}")
print(f"V1↔V2 mean abs diff: {(v2['demand'] - v1['demand']).abs().mean():.6f}")
print(f"V1↔V2 max abs diff:  {(v2['demand'] - v1['demand']).abs().max():.6f}")
print(f"V1→V2 RMSE: {np.sqrt(((v2['demand'] - v1['demand']) ** 2).mean()):.6f}")
# Per-quintile of v1: how much does v2 differ?
v1_q = pd.qcut(v1['demand'], 5, labels=False, duplicates='drop')
print("\nV1↔V2 mean abs diff by V1 quintile:")
for q in range(5):
    mask = v1_q == q
    print(f"  Q{q} (V1 demand {v1['demand'][mask].min():.3f}..{v1['demand'][mask].max():.3f}): "
          f"V1 mean={v1['demand'][mask].mean():.4f}, "
          f"V2 mean={v2['demand'][mask].mean():.4f}, "
          f"diff={(v2['demand'][mask].mean() - v1['demand'][mask].mean()):+.4f}")

print("\n" + "=" * 70)
print("V2 vs BASELINE (87.26) COMPARISON")
print("=" * 70)
print(f"Baseline mean: {baseline['demand'].mean():.6f}, std: {baseline['demand'].std():.6f}")
print(f"V1 mean: {v1['demand'].mean():.6f}, std: {v1['demand'].std():.6f}")
print(f"V2 mean: {v2['demand'].mean():.6f}, std: {v2['demand'].std():.6f}")
print(f"V2↔baseline corr: {v2['demand'].corr(baseline['demand']):.6f}")
print(f"V2↔baseline mean abs diff: {(v2['demand'] - baseline['demand']).abs().mean():.6f}")

print("\n" + "=" * 70)
print("PER-BUCKET (RoadType) — join with test.csv")
print("=" * 70)
test_with_v2 = test.merge(v2, on="Index").merge(v1.rename(columns={"demand": "demand_v1"}), on="Index")
test_with_v2["time_slot"] = test_with_v2["timestamp"].apply(
    lambda ts: (int(ts.split(":")[0]) * 60 + int(ts.split(":")[1])) // 15)
test_with_v2["d49t_ctx"] = ((test_with_v2["time_slot"] >= 9) & (test_with_v2["time_slot"] <= 16)).astype(int)

print("Per RoadType:")
for rt, sub in test_with_v2.groupby("RoadType", dropna=False):
    print(f"  {str(rt):>15s}: n={len(sub):>5d}, V1 mean={sub['demand_v1'].mean():.4f}, "
          f"V2 mean={sub['demand'].mean():.4f}, "
          f"diff={(sub['demand'].mean() - sub['demand_v1'].mean()):+.4f}")
print("\nPer test context (slots 9-16 have d49t context, slots 17+ don't):")
for ctx, sub in test_with_v2.groupby("d49t_ctx"):
    label = "WITH d49t ctx (slots 9-16)" if ctx == 1 else "NO d49t ctx (slots 17-55)"
    print(f"  {label}: n={len(sub):>5d}, V1 mean={sub['demand_v1'].mean():.4f}, "
          f"V2 mean={sub['demand'].mean():.4f}, "
          f"diff={(sub['demand'].mean() - sub['demand_v1'].mean()):+.4f}")

print("\n" + "=" * 70)
print("V2 vs TEST FEATURES — sanity")
print("=" * 70)
print(f"Test rows: {len(test):,}")
print(f"V2 rows:  {len(v2):,}")
print(f"Test geohash unique: {test['geohash'].nunique():,}")
print(f"Test RoadType counts:")
print(test['RoadType'].value_counts(dropna=False))
print(f"\nTest time_slot distribution:")
print(test_with_v2['time_slot'].value_counts().sort_index().head(20))
print("  ...")
print(test_with_v2['time_slot'].value_counts().sort_index().tail(10))

print("\n" + "=" * 70)
print("ANOMALY CHECK")
print("=" * 70)
# Suspicious if many predictions are very close to 0 or 1
n_very_low = (v2['demand'] < 0.005).sum()
n_very_high = (v2['demand'] > 0.95).sum()
n_at_1 = (v2['demand'] == 1.0).sum()
print(f"V2 predictions < 0.005: {n_very_low:,} ({n_very_low/len(v2):.2%})")
print(f"V2 predictions > 0.95:  {n_very_high:,} ({n_very_high/len(v2):.2%})")
print(f"V2 predictions == 1.0:  {n_at_1:,} ({n_at_1/len(v2):.2%})")
print(f"V1 predictions < 0.005: {(v1['demand'] < 0.005).sum():,} "
      f"({(v1['demand'] < 0.005).mean():.2%})")
print(f"V1 predictions > 0.95:  {(v1['demand'] > 0.95).sum():,} "
      f"({(v1['demand'] > 0.95).mean():.2%})")
print(f"V1 predictions == 1.0:  {(v1['demand'] == 1.0).sum():,} "
      f"({(v1['demand'] == 1.0).mean():.2%})")
print(f"Baseline predictions < 0.005: {(baseline['demand'] < 0.005).sum():,}")
print(f"Baseline predictions > 0.95:  {(baseline['demand'] > 0.95).sum():,}")
print(f"Baseline predictions == 1.0:  {(baseline['demand'] == 1.0).sum():,}")

# Check for test rows with very different V1 vs V2 (potential issues)
test_with_v2["v1_v2_diff"] = (test_with_v2["demand"] - test_with_v2["demand_v1"]).abs()
print(f"\nV1↔V2 abs diff stats:")
print(f"  median: {test_with_v2['v1_v2_diff'].median():.6f}")
print(f"  p99:    {test_with_v2['v1_v2_diff'].quantile(0.99):.6f}")
print(f"  p99.9:  {test_with_v2['v1_v2_diff'].quantile(0.999):.6f}")
print(f"  max:    {test_with_v2['v1_v2_diff'].max():.6f}")
print(f"  frac > 0.05: {(test_with_v2['v1_v2_diff'] > 0.05).mean():.4f}")
print(f"  frac > 0.10: {(test_with_v2['v1_v2_diff'] > 0.10).mean():.4f}")

print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
ok = (
    v2.shape == (41778, 2)
    and list(v2.columns) == ['Index', 'demand']
    and (v2['Index'].values == test['Index'].values).all()
    and v2['demand'].notna().all()
    and v2['demand'].between(0, 1).all()
    and abs(v2['demand'].mean() - v1['demand'].mean()) < 0.01
    and v2['demand'].corr(v1['demand']) > 0.99
)
print(f"All sanity checks pass: {ok}")
if ok:
    print("V2 is ready for upload.")
else:
    print("V2 has issues — DO NOT upload.")
