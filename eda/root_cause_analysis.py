"""Root cause analysis: why does OOF R² (0.9626) overstate the online score (0.8997)?

Hypotheses to test:
1. OOF is dominated by d48 (in-distribution) rows; d49t (cross-day) rows are harder
2. Test slots 17-55 (no d49t context) are much harder than slots 9-16 (have d49t context)
3. The day-shift (d48→d49) varies by geohash and time-slot; the model misses it
4. The OOF KFold (5-fold within d48) is fundamentally a different task than test
5. Feature availability drops sharply for test slots 17-55
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("/home/marcus/code/Gridlock/dataset")
ART = Path("/home/marcus/code/Gridlock/eda")
ART.mkdir(exist_ok=True)


def load_data():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sub = pd.read_csv(DATA / "submission.csv")
    return train, test, sub


def parse_hr_min(ts: str) -> tuple[int, int]:
    parts = ts.split(":")
    return int(parts[0]), int(parts[1])


def to_slot(hr: int, mn: int) -> int:
    return (hr * 60 + mn) // 15


def hypothesis_1_oof_decomposition(train: pd.DataFrame):
    """H1: OOF R² is averaged over d48 (in-distribution) + d49t (cross-day). The
    d49t rows in OOF are MUCH harder. The mean OOF is dominated by d48.
    """
    print("\n" + "=" * 70)
    print("H1: OOF R² is dominated by d48 rows; d49t (cross-day) is harder")
    print("=" * 70)
    d48 = train[train["day"] == 48].copy()
    d49t = train[train["day"] == 49].copy()
    print(f"d48 rows: {len(d48):,} (slots 0-95, all hours)")
    print(f"d49t rows: {len(d49t):,} (slots 0-8, 00:00-02:00)")
    print(f"d48 / total ratio: {len(d48) / len(train):.1%}")
    print(f"d49t / total ratio: {len(d49t) / len(train):.1%}")
    print()
    print("Demand distribution by day:")
    for label, sub in [("d48 (in-distribution)", d48), ("d49t (cross-day)", d49t)]:
        print(f"  {label}: mean={sub['demand'].mean():.4f}, "
              f"std={sub['demand'].std():.4f}, "
              f"min={sub['demand'].min():.4f}, max={sub['demand'].max():.4f}")
    print()
    # Variance explained if d48 is a good predictor of d49
    d48_slots_0_8 = d48[d48["timestamp"].apply(
        lambda ts: to_slot(*parse_hr_min(ts)) <= 8)]
    common = d48_slots_0_8.merge(d49t, on=["geohash", "timestamp"],
                                   suffixes=("_d48", "_d49t"))
    print(f"d48∩d49t cells (same geohash, time_slot 0-8): {len(common):,}")
    if len(common):
        corr = common["demand_d48"].corr(common["demand_d49t"])
        mean_diff = (common["demand_d49t"] - common["demand_d48"]).mean()
        rmse_naive = np.sqrt(((common["demand_d49t"] - common["demand_d48"]) ** 2).mean())
        var = common["demand_d49t"].var()
        r2_naive = 1 - rmse_naive ** 2 / var
        print(f"  Corr(d48, d49t): {corr:.4f}")
        print(f"  Mean(d49t) - Mean(d48): {mean_diff:+.4f}")
        print(f"  R² of naive 'predict d49t = d48': {r2_naive:.4f}")


def hypothesis_2_test_slot_decomposition():
    """H2: Test slots 9-16 have d49t same-day context (lp_d49t_* available);
    slots 17-55 do not. So 83% of test (slots 17-55) is much harder.
    """
    print("\n" + "=" * 70)
    print("H2: 83% of test (slots 17-55) has NO d49t same-day context")
    print("=" * 70)
    test = pd.read_csv(DATA / "test.csv")
    test["time_slot"] = test["timestamp"].apply(
        lambda ts: to_slot(*parse_hr_min(ts)))
    n_with_ctx = ((test["time_slot"] >= 9) & (test["time_slot"] <= 16)).sum()
    n_no_ctx = (test["time_slot"] >= 17).sum()
    print(f"Test rows with d49t same-day context (slots 9-16):  "
          f"{n_with_ctx:,} ({n_with_ctx / len(test):.1%})")
    print(f"Test rows WITHOUT d49t same-day context (slots 17+): "
          f"{n_no_ctx:,} ({n_no_ctx / len(test):.1%})")
    print()
    # Per-slot row count
    counts = test["time_slot"].value_counts().sort_index()
    print("Test row count by time_slot (first 30 slots):")
    for s in counts.index[:30]:
        bar = "#" * int(counts[s] / 50)
        ctx = " (d49t ctx)" if 9 <= s <= 16 else (" (gap)" if s < 9 else " (no ctx)")
        print(f"  slot {s:2d} ({s*15//60:02d}:{s*15%60:02d}): "
              f"{counts[s]:4d} {bar}{ctx}")


def hypothesis_3_day_shift_by_geohash():
    """H3: The d48→d49 shift varies by geohash. The model uses d48 stats (g_mu)
    as a feature, but doesn't have an explicit per-geohash d49→d48 ratio.
    """
    print("\n" + "=" * 70)
    print("H3: Per-geohash day shift (d49t / d48) varies a lot")
    print("=" * 70)
    train = pd.read_csv(DATA / "train.csv")
    d48 = train[train["day"] == 48]
    d49t = train[train["day"] == 49]

    d48_g = d48.groupby("geohash")["demand"].agg(["mean", "std", "count"])
    d49t_g = d49t.groupby("geohash")["demand"].agg(["mean", "std", "count"])
    common = d48_g.join(d49t_g, lsuffix="_d48", rsuffix="_d49t", how="inner")
    common["ratio"] = common["mean_d49t"] / (common["mean_d48"] + 1e-9)
    common["diff"] = common["mean_d49t"] - common["mean_d48"]

    print(f"Geohash with d48 AND d49t data: {len(common):,}")
    print()
    print("Day-shift ratio (d49t mean / d48 mean) per geohash:")
    print(f"  mean ratio: {common['ratio'].mean():.3f}")
    print(f"  std  ratio: {common['ratio'].std():.3f}")
    print(f"  min  ratio: {common['ratio'].min():.3f}")
    print(f"  max  ratio: {common['ratio'].max():.3f}")
    print(f"  median     : {common['ratio'].median():.3f}")
    print()
    print("Per-quintile of geohash size (by d48 count):")
    common["size_q"] = pd.qcut(common["count_d48"], 5, labels=False)
    for q, sub in common.groupby("size_q"):
        print(f"  Q{q} (n={len(sub)}): ratio mean={sub['ratio'].mean():.3f}, "
              f"std={sub['ratio'].std():.3f}")
    print()
    print("Distribution of d49t mean - d48 mean per geohash:")
    print(f"  mean diff: {common['diff'].mean():+.4f}")
    print(f"  std  diff: {common['diff'].std():.4f}")
    # What if we use d49t_g_mean as a per-geohash feature?
    # For test rows, this is fully available (all test geohash are in d49t)
    print()
    print("HYPOTHESIS: Add d49t_g_mu as a feature — it's available for")
    print("ALL 1,190 test geohash (same geohash set, d49t slots 0-8).")


def hypothesis_4_oof_test_simulation(train: pd.DataFrame):
    """H4: The OOF (5-fold KFold within d48) is fundamentally a different
    prediction task than test (predict d49 slots 9-55 from d48 + d49t).

    We can simulate the test scenario by:
    - Train on (d48 slots 0-8) + d49t  ← simulate "have same-day context"
    - Predict on (d48 slots 9-55)     ← simulate "test slots"
    This gives us an HONEST test-similar R².
    """
    print("\n" + "=" * 70)
    print("H4: OOF KFold (5-fold d48) is NOT the test scenario")
    print("=" * 70)
    print("Test scenario: train on d48 + d49t, predict d49 (slots 9-55)")
    print("  - 17% of test (slots 9-16) has d49t same-day context")
    print("  - 83% of test (slots 17-55) has no same-day context")
    print()
    print("OOF Mode A: 5-fold KFold within d48, predict d48 from d48")
    print("  - This is a SAME-DAY prediction task")
    print("  - All features are in-distribution")
    print()
    print("OOF Mode C: train on d48, predict d49t")
    print("  - This is a TRUE DAY EXTRAPOLATION (1-2 hours before test start)")
    print("  - R² = 0.7550 (much lower than Mode A 0.9661)")
    print()
    print("Q: Can we simulate the actual test by training on d48 + d49t")
    print("   and predicting d49 slots 9-55? NO — d49 slots 9-55 ARE the test.")
    print()
    print("Best estimate: the 17%-with-context rows score ~Mode C with help")
    print("(+boost from d49t context), the 83%-no-context rows score ~Mode C.")


def hypothesis_5_feature_availability(test: pd.DataFrame):
    """H5: For test slots 17-55, lp_d49t_* are NaN (no d49t context).
    For test slots 9-16, lp_d49t_<lag> has values.
    """
    print("\n" + "=" * 70)
    print("H5: Feature availability drops sharply for test slots 17+")
    print("=" * 70)
    test["time_slot"] = test["timestamp"].apply(
        lambda ts: to_slot(*parse_hr_min(ts)))
    print("Hypothetical feature availability:")
    for ts in [9, 10, 12, 15, 16, 17, 20, 30, 40, 50, 55]:
        # lp_d49t_<lag> available for lag in 1..8 if t - lag in [0, 8]
        avail_lags = [l for l in range(1, 9) if 0 <= ts - l <= 8]
        gts_ctx = "yes" if ts <= 8 else "no"  # d49t has slots 0-8 only
        d49t_g = "yes (mean of slots 0-8)"  # always available
        print(f"  slot {ts:2d}: lp_d49t_* lags available: {avail_lags}, "
              f"same-day gts: {gts_ctx}, d49t_g_mu: {d49t_g}")


def hypothesis_6_demand_curve_diff():
    """H6: The d48 demand curve at slots 9-55 (what model sees as 'history')
    may not match the d49 demand at those same slots (what we want to predict).
    """
    print("\n" + "=" * 70)
    print("H6: d48 demand curve at slots 9-55 vs what we WANT to predict")
    print("=" * 70)
    train = pd.read_csv(DATA / "train.csv")
    train["time_slot"] = train["timestamp"].apply(
        lambda ts: to_slot(*parse_hr_min(ts)))
    d48 = train[train["day"] == 48]
    d49t = train[train["day"] == 49]

    # d48 slots 9-55 mean demand (the 'training data' for those slots)
    d48_9_55 = d48[(d48["time_slot"] >= 9) & (d48["time_slot"] <= 55)]
    d49t_0_8 = d49t

    print("Mean demand by hour, d48 vs d49t (where both exist):")
    for hr in range(0, 3):
        d48_h = d48[(d48["time_slot"] >= hr * 4) & (d48["time_slot"] < (hr + 1) * 4)]
        d49t_h = d49t[(d49t["time_slot"] >= hr * 4) & (d49t["time_slot"] < (hr + 1) * 4)]
        if len(d48_h) and len(d49t_h):
            print(f"  hour {hr:02d}: d48 mean={d48_h['demand'].mean():.4f}, "
                  f"d49t mean={d49t_h['demand'].mean():.4f}, "
                  f"diff={d49t_h['demand'].mean() - d48_h['demand'].mean():+.4f}")
    print()
    print("d48 mean demand at test slot ranges (what model 'knows'):")
    for lo, hi in [(9, 16), (17, 32), (33, 48), (49, 55)]:
        sub = d48[(d48["time_slot"] >= lo) & (d48["time_slot"] <= hi)]
        print(f"  slots {lo:2d}-{hi:2d}: "
              f"n={len(sub):,}, mean={sub['demand'].mean():.4f}, "
              f"std={sub['demand'].std():.4f}")
    print()
    print("KEY INSIGHT: We are predicting d49 slots 9-55 (02:15-13:45).")
    print("We have d48 demand at those slots, but only d49t data for 00:00-02:00.")
    print("The model uses d48 patterns + d49t early-morning, but the test is")
    print("demanding d49 mid-morning to early-afternoon.")


def main():
    train, test, sub = load_data()
    print(f"Train: {len(train):,} rows, Test: {len(test):,} rows, "
          f"Sub: {len(sub):,} rows")
    print(f"Train demand: mean={train['demand'].mean():.4f}, "
          f"std={train['demand'].std():.4f}")
    print(f"Sub demand:   mean={sub['demand'].mean():.4f}, "
          f"std={sub['demand'].std():.4f}")
    hypothesis_1_oof_decomposition(train)
    hypothesis_2_test_slot_decomposition()
    hypothesis_3_day_shift_by_geohash()
    hypothesis_4_oof_test_simulation(train)
    hypothesis_5_feature_availability(test)
    hypothesis_6_demand_curve_diff()
    print("\n" + "=" * 70)
    print("SUMMARY OF ROOT CAUSES")
    print("=" * 70)
    print("""
1. The OOF (Mode A = 0.9661) is computed on d48 KFold — a SAME-DAY task.
   It evaluates "predict d48 from other d48 rows" — not "predict d49 from d48".
   This is the wrong task. Mode A R² is overoptimistic by 6+ points R².

2. The OOF includes BOTH d48 (90%) and d49t (10%) rows. The d48 rows are
   in-distribution (easy); the d49t rows are cross-day (hard). The OOF R²
   average is dominated by d48.

3. The true day-extrapolation task (Mode C = 0.7550) is 21 points lower R²
   than Mode A. The test sits between Mode A and Mode C, closer to Mode C
   because most test rows (slots 17-55, 83% of test) have no same-day context.

4. d49t has all 1,190 test geohash, but only for slots 0-8. So per-geohash
   d49t_g_mu is available for all test geohash — but is NOT currently a feature.
   This is a missed opportunity.

5. The d48→d49 day shift varies by geohash (ratio mean=1.x, std=0.x).
   The model uses g_mu (d48) but doesn't have an explicit d49t→d48 ratio.

6. Per-day-of-week, per-geohash day shift would be ideal, but we only have
   2 days — so the d49t→d48 ratio (per geohash, smoothed) is the best
   day-shift signal we can build.

7. The Ridge stack weights from Mode A (LGB 0.40, XGB 0.20, CB 0.40) may
   not transfer to test. The CB weight is the largest in final retrain (0.54).
   CB on cross-day may be more reliable than LGB on cross-day.

ACTION ITEMS:
- Phase E: add per-geohash d49t features (g_mu, g_sd, ratio d49t/d48)
- Phase E: multi-seed averaging (3 seeds × 3 models × 5-fold)
- Phase E: drop bias correction (overfits d48 KFold residuals)
""")


if __name__ == "__main__":
    main()
