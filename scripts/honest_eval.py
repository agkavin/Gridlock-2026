"""
C-1 — Honest day-48 holdout evaluator (3 evaluation modes).

Three complementary honest metrics, all leak-free:

A) PRIMARY: 5-fold KFold within d48 (RANDOM).
   - Mirrors the actual test scenario: the (geohash, time_slot) stat for
     val rows is computed OOF from the other 4 folds (which contain all 96
     slots, just like d48 contains all 96 slots for the test).
   - This is the metric we'll use to measure C-2..C-8 improvements.

B) SECONDARY: Time-based holdout (slots 80-95 of d48).
   - The extreme stress test. Val rows at slots 80-95 have NO
     (geohash, time_slot) stat available (those slots are excluded from
     training). The model must extrapolate in time.

C) TERTIARY: Predict d49t (day 49 morning) from d48 only.
   - Strict "predict day N+1 from day N" test. The most pessimistic
     scenario — no d49t context at all. Lower bound for test R².

All features are computed causally so val rows never influence their own
predictions. Lag features are looked up only in the "past" portion.

Run from /home/marcus/code/Gridlock:
    uv run python scripts/honest_eval.py
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

ROOT = Path("/home/marcus/code/Gridlock")
DATA = ROOT / "dataset"
EDA = ROOT / "eda"
EDA.mkdir(exist_ok=True)

HOLDOUT_START = 80


def parse_time(df):
    s = df["timestamp"].astype(str).str.strip().str.split(":", expand=True)
    h = pd.to_numeric(s[0], errors="coerce").fillna(0).astype(int)
    m = pd.to_numeric(s[1], errors="coerce").fillna(0).astype(int)
    return h * 60 + m, (h * 60 + m) // 15


def load_data():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    for df in (train, test):
        df["ts_min"], df["time_slot"] = parse_time(df)
    return train, test


# =================================================================
# FEATURE SET
# =================================================================
BACK = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24]
FWD = [1, 2, 3, 4, 5, 6, 8]

FEATS = (
    [f"lm{l}" for l in BACK]
    + [f"lp{l}" for l in FWD]
    + [
        "g_mu", "g_sd", "g_med", "g_p25", "g_p75", "g_iqr",
        "gts_mu", "gts_sd",
        "road_enc", "wx_enc", "lv", "lm_f",
        "temp", "temp2", "is_hw", "lanes2",
        "time_slot", "ts_sin", "ts_cos", "hr",
        "is_mpk", "is_epk", "ratio1", "NumberofLanes",
    ]
)


# =================================================================
# CAUSAL FEATURE BUILDER
# =================================================================
def build_causal_features(d48, source_idx, fill_value_idx=None):
    """
    Build features for d48. `source_idx` = boolean mask of rows that may be
    used as 'past' for lag lookups and stat aggregations.

    For training rows in source_idx: 5-fold OOF for stat features.
    For rows outside source_idx: stats computed from ALL of source_idx (no OOF
    needed because they're not in source_idx).
    """
    df = d48.copy()
    src = df[source_idx].copy()
    if fill_value_idx is None:
        fill_value_idx = source_idx  # default: fill with mean of source

    # Global mean (source only)
    gm = src["demand"].mean()

    # Backward lags from source only
    lm_idx = src.set_index(["geohash", "time_slot"])["demand"]
    for lag in BACK:
        df[f"lm{lag}"] = df.apply(
            lambda r: lm_idx.get((r["geohash"], r["time_slot"] - lag), np.nan),
            axis=1,
        )

    # Forward lags (set to NaN — strict causal)
    for lag in FWD:
        df[f"lp{lag}"] = np.nan

    # Geohash statistics — 5-fold OOF within source, then assign to non-source
    # from the full source aggregate
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    gc = ["g_mu", "g_sd", "g_med", "g_p25", "g_p75", "g_iqr"]
    for c in gc:
        df[c] = np.nan

    for ti, vi in kf.split(src):
        fs = (
            src.iloc[ti]
            .groupby("geohash")["demand"]
            .agg(
                g_mu="mean", g_sd="std", g_med="median",
                p25=lambda x: x.quantile(0.25),
                p75=lambda x: x.quantile(0.75),
            )
            .reset_index()
            .rename(columns={"p25": "g_p25", "p75": "g_p75"})
        )
        fs["g_iqr"] = fs["g_p75"] - fs["g_p25"]
        va = src.iloc[vi][["geohash"]].merge(fs, on="geohash", how="left")
        src_idx = src.index[vi]
        for c in gc:
            df.loc[src_idx, c] = va[c].values

    # Non-source rows: assign the FULL source aggregate (no leakage because
    # these rows are not in source)
    full_fs = (
        src.groupby("geohash")["demand"]
        .agg(
            g_mu="mean", g_sd="std", g_med="median",
            p25=lambda x: x.quantile(0.25),
            p75=lambda x: x.quantile(0.75),
        )
        .reset_index()
        .rename(columns={"p25": "g_p25", "p75": "g_p75"})
    )
    full_fs["g_iqr"] = full_fs["g_p75"] - full_fs["g_p25"]
    non_src = df[~source_idx][["geohash"]].merge(full_fs, on="geohash", how="left")
    non_src_idx = df[~source_idx].index
    for c in gc:
        df.loc[non_src_idx, c] = non_src[c].values

    for c in gc:
        df[c] = df[c].fillna(gm)

    # (geohash, time_slot) statistics — OOF within source, then assign to
    # non-source from full source aggregate
    for c in ["gts_mu", "gts_sd"]:
        df[c] = np.nan
    for ti, vi in kf.split(src):
        fgt = (
            src.iloc[ti]
            .groupby(["geohash", "time_slot"])["demand"]
            .agg(gts_mu="mean", gts_sd="std")
            .reset_index()
        )
        va = src.iloc[vi][["geohash", "time_slot"]].merge(
            fgt, on=["geohash", "time_slot"], how="left"
        )
        src_idx = src.index[vi]
        df.loc[src_idx, "gts_mu"] = va["gts_mu"].values
        df.loc[src_idx, "gts_sd"] = va["gts_sd"].values
    full_fgt = (
        src.groupby(["geohash", "time_slot"])["demand"]
        .agg(gts_mu="mean", gts_sd="std")
        .reset_index()
    )
    non_src_fgt = df[~source_idx][["geohash", "time_slot"]].merge(
        full_fgt, on=["geohash", "time_slot"], how="left"
    )
    df.loc[non_src_idx, "gts_mu"] = non_src_fgt["gts_mu"].values
    df.loc[non_src_idx, "gts_sd"] = non_src_fgt["gts_sd"].values

    df["gts_mu"] = df["gts_mu"].fillna(df["g_mu"])
    df["gts_sd"] = df["gts_sd"].fillna(df["g_sd"])

    # Categorical / derived features (constants)
    gt = src["Temperature"].median()
    wx = {"Sunny": 0, "Rainy": 1, "Foggy": 2, "Snowy": 3}
    rd = {"Residential": 0, "Street": 1, "Highway": 2}

    df["road_enc"] = df["RoadType"].map(rd).fillna(0).astype(int)
    df["wx_enc"] = df["Weather"].map(wx).fillna(0).astype(int)
    df["lv"] = (df["LargeVehicles"] == "Allowed").astype(int)
    df["lm_f"] = (df["Landmarks"] == "Yes").astype(int)
    df["temp"] = df["Temperature"].fillna(gt)
    df["temp2"] = df["temp"] ** 2
    df["is_hw"] = (df["NumberofLanes"] >= 4).astype(int)
    df["lanes2"] = df["NumberofLanes"] ** 2
    df["ts_sin"] = np.sin(2 * np.pi * df["time_slot"] / 96)
    df["ts_cos"] = np.cos(2 * np.pi * df["time_slot"] / 96)
    df["hr"] = df["ts_min"] // 60
    df["is_mpk"] = df["hr"].between(7, 9).astype(int)
    df["is_epk"] = df["hr"].between(17, 19).astype(int)
    df["ratio1"] = df["gts_mu"] / (df["g_mu"] + 1e-8)

    return df


# =================================================================
# MODELS
# =================================================================
def train_lgb(X_tr, y_tr, X_va, y_va):
    params = {
        "objective": "regression", "metric": "rmse",
        "learning_rate": 0.03, "num_leaves": 255, "min_child_samples": 5,
        "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 1,
        "reg_alpha": 0.05, "reg_lambda": 0.1,
        "n_jobs": 4, "verbose": -1, "seed": 42,
    }
    dt = lgb.Dataset(X_tr, y_tr)
    dv = lgb.Dataset(X_va, y_va, reference=dt)
    m = lgb.train(
        params, dt, num_boost_round=2000, valid_sets=[dv],
        callbacks=[lgb.early_stopping(150, verbose=False)],
    )
    return m.predict(X_va, num_iteration=m.best_iteration), m


def train_xgb(X_tr, y_tr, X_va, y_va):
    params = {
        "objective": "reg:squarederror", "eval_metric": "rmse",
        "learning_rate": 0.05, "max_depth": 8, "min_child_weight": 5,
        "subsample": 0.7, "colsample_bytree": 0.7,
        "reg_alpha": 0.05, "reg_lambda": 1.0,
        "n_jobs": 4, "seed": 42, "verbosity": 0,
    }
    dtr = xgb.DMatrix(X_tr, y_tr)
    dva = xgb.DMatrix(X_va, y_va)
    m = xgb.train(
        params, dtr, num_boost_round=2000,
        evals=[(dva, "val")], early_stopping_rounds=100, verbose_eval=False,
    )
    return m.predict(dva), m


def fit_predict_kfold(X, y, kf, model_fn):
    """Run 5-fold OOF with the given model factory."""
    oof = np.zeros(len(X))
    iters = []
    for fold, (ti, vi) in enumerate(kf.split(X), 1):
        X_tr, X_va = X.iloc[ti], X.iloc[vi]
        y_tr, y_va = y.iloc[ti], y.iloc[vi]
        pred, m = model_fn(X_tr, y_tr, X_va, y_va)
        oof[vi] = pred
        iters.append(m.best_iteration if hasattr(m, "best_iteration") else 0)
    r2 = r2_score(y, oof)
    return oof, r2, iters


# =================================================================
# EVALUATION MODES
# =================================================================
def mode_a_kfold_d48(d48, label="A) KFold d48 (random)"):
    """Random 5-fold KFold within d48. Most representative of the test scenario."""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # Build features with all of d48 as source (so gts stats are available
    # for every (geohash, time_slot) combination).
    d48f = build_causal_features(d48, source_idx=pd.Series(True, index=d48.index))
    X = d48f[FEATS].astype(float).fillna(-999)
    y = d48f["demand"].astype(float)

    print(f"\n[{label}] d48 shape: {X.shape}")
    lgb_oof, lgb_r2, lgb_iters = fit_predict_kfold(X, y, kf, train_lgb)
    xgb_oof, xgb_r2, xgb_iters = fit_predict_kfold(X, y, kf, train_xgb)

    bw, br = -1, -1
    for w in np.arange(0.0, 1.01, 0.02):
        r = r2_score(y, w * lgb_oof + (1 - w) * xgb_oof)
        if r > br:
            br, bw = r, w
    print(f"  LGB R²  : {lgb_r2:.4f}  (mean best_iter={np.mean(lgb_iters):.0f})")
    print(f"  XGB R²  : {xgb_r2:.4f}  (mean best_iter={np.mean(xgb_iters):.0f})")
    print(f"  Blend w_LGB={bw:.2f}  -> R²={br:.4f}  (Score={max(0, 100*br):.2f})")
    return {
        "label": label,
        "n": int(len(X)),
        "lgb_r2": float(lgb_r2),
        "xgb_r2": float(xgb_r2),
        "blend_w_lgb": float(bw),
        "blend_r2": float(br),
        "score": float(max(0, 100 * br)),
    }


def mode_b_time_holdout(d48, label="B) Time holdout (slots 80-95)"):
    """Time-based holdout: predict last 4 hours of d48 from first 20 hours."""
    source_idx = d48["time_slot"] < HOLDOUT_START
    d48f = build_causal_features(d48, source_idx=source_idx)
    tr = d48f[source_idx]
    va = d48f[~source_idx]
    X_tr = tr[FEATS].astype(float).fillna(-999)
    y_tr = tr["demand"].astype(float)
    X_va = va[FEATS].astype(float).fillna(-999)
    y_va = va["demand"].astype(float)

    print(f"\n[{label}] train: {len(X_tr)}  |  val: {len(X_va)}")
    lgb_pred, lgb_m = train_lgb(X_tr, y_tr, X_va, y_va)
    xgb_pred, xgb_m = train_xgb(X_tr, y_tr, X_va, y_va)
    lgb_r2 = r2_score(y_va, lgb_pred)
    xgb_r2 = r2_score(y_va, xgb_pred)

    bw, br = -1, -1
    for w in np.arange(0.0, 1.01, 0.02):
        r = r2_score(y_va, w * lgb_pred + (1 - w) * xgb_pred)
        if r > br:
            br, bw = r, w
    print(f"  LGB R²  : {lgb_r2:.4f}  (best_iter={lgb_m.best_iteration})")
    print(f"  XGB R²  : {xgb_r2:.4f}  (best_iter={xgb_m.best_iteration})")
    print(f"  Blend w_LGB={bw:.2f}  -> R²={br:.4f}  (Score={max(0, 100*br):.2f})")
    return {
        "label": label,
        "n_train": int(len(X_tr)),
        "n_val": int(len(X_va)),
        "lgb_r2": float(lgb_r2),
        "xgb_r2": float(xgb_r2),
        "blend_w_lgb": float(bw),
        "blend_r2": float(br),
        "score": float(max(0, 100 * br)),
    }


def mode_c_d49t_from_d48(d48, d49t, label="C) Predict d49t from d48 only"):
    """Predict day 49 morning using only day 48. Strict day-extrapolation test."""
    source_idx = pd.Series(True, index=d48.index)
    d48f = build_causal_features(d48, source_idx=source_idx)
    # Build features for d49t using d48 as source (full d48)
    d49f = build_features_for_external(d48, d49t, source_idx)

    X_tr = d48f[FEATS].astype(float).fillna(-999)
    y_tr = d48f["demand"].astype(float)
    X_va = d49f[FEATS].astype(float).fillna(-999)
    y_va = d49f["demand"].astype(float)

    print(f"\n[{label}] train: d48 ({len(X_tr)})  |  val: d49t ({len(X_va)})")
    lgb_pred, lgb_m = train_lgb(X_tr, y_tr, X_va, y_va)
    xgb_pred, xgb_m = train_xgb(X_tr, y_tr, X_va, y_va)
    lgb_r2 = r2_score(y_va, lgb_pred)
    xgb_r2 = r2_score(y_va, xgb_pred)

    bw, br = -1, -1
    for w in np.arange(0.0, 1.01, 0.02):
        r = r2_score(y_va, w * lgb_pred + (1 - w) * xgb_pred)
        if r > br:
            br, bw = r, w
    print(f"  LGB R²  : {lgb_r2:.4f}  (best_iter={lgb_m.best_iteration})")
    print(f"  XGB R²  : {xgb_r2:.4f}  (best_iter={xgb_m.best_iteration})")
    print(f"  Blend w_LGB={bw:.2f}  -> R²={br:.4f}  (Score={max(0, 100*br):.2f})")
    return {
        "label": label,
        "n_train": int(len(X_tr)),
        "n_val": int(len(X_va)),
        "lgb_r2": float(lgb_r2),
        "xgb_r2": float(xgb_r2),
        "blend_w_lgb": float(bw),
        "blend_r2": float(br),
        "score": float(max(0, 100 * br)),
    }


def build_features_for_external(d48, ext, source_idx):
    """Build features for an external dataframe (e.g. d49t) using d48 as source."""
    df = ext.copy()
    src = d48[source_idx].copy()
    gm = src["demand"].mean()

    lm_idx = src.set_index(["geohash", "time_slot"])["demand"]
    for lag in BACK:
        df[f"lm{lag}"] = df.apply(
            lambda r: lm_idx.get((r["geohash"], r["time_slot"] - lag), np.nan),
            axis=1,
        )
    for lag in FWD:
        df[f"lp{lag}"] = np.nan

    full_fs = (
        src.groupby("geohash")["demand"]
        .agg(g_mu="mean", g_sd="std", g_med="median",
             p25=lambda x: x.quantile(0.25), p75=lambda x: x.quantile(0.75))
        .reset_index()
        .rename(columns={"p25": "g_p25", "p75": "g_p75"})
    )
    full_fs["g_iqr"] = full_fs["g_p75"] - full_fs["g_p25"]
    gc = ["g_mu", "g_sd", "g_med", "g_p25", "g_p75", "g_iqr"]
    tmp = df[["geohash"]].merge(full_fs, on="geohash", how="left")
    for c in gc:
        df[c] = tmp[c].fillna(gm).values

    full_fgt = (
        src.groupby(["geohash", "time_slot"])["demand"]
        .agg(gts_mu="mean", gts_sd="std")
        .reset_index()
    )
    tmp = df[["geohash", "time_slot"]].merge(
        full_fgt, on=["geohash", "time_slot"], how="left"
    )
    df["gts_mu"] = tmp["gts_mu"].fillna(df["g_mu"]).values
    df["gts_sd"] = tmp["gts_sd"].fillna(df["g_sd"]).values

    gt = src["Temperature"].median()
    wx = {"Sunny": 0, "Rainy": 1, "Foggy": 2, "Snowy": 3}
    rd = {"Residential": 0, "Street": 1, "Highway": 2}

    df["road_enc"] = df["RoadType"].map(rd).fillna(0).astype(int)
    df["wx_enc"] = df["Weather"].map(wx).fillna(0).astype(int)
    df["lv"] = (df["LargeVehicles"] == "Allowed").astype(int)
    df["lm_f"] = (df["Landmarks"] == "Yes").astype(int)
    df["temp"] = df["Temperature"].fillna(gt)
    df["temp2"] = df["temp"] ** 2
    df["is_hw"] = (df["NumberofLanes"] >= 4).astype(int)
    df["lanes2"] = df["NumberofLanes"] ** 2
    df["ts_sin"] = np.sin(2 * np.pi * df["time_slot"] / 96)
    df["ts_cos"] = np.cos(2 * np.pi * df["time_slot"] / 96)
    df["hr"] = df["ts_min"] // 60
    df["is_mpk"] = df["hr"].between(7, 9).astype(int)
    df["is_epk"] = df["hr"].between(17, 19).astype(int)
    df["ratio1"] = df["gts_mu"] / (df["g_mu"] + 1e-8)

    return df


# =================================================================
# MAIN
# =================================================================
def main():
    train_df, _ = load_data()
    d48 = train_df[train_df["day"] == 48].reset_index(drop=True)
    d49t = train_df[train_df["day"] == 49].reset_index(drop=True)
    print(f"d48 shape: {d48.shape}    d49t shape: {d49t.shape}")

    res_a = mode_a_kfold_d48(d48)
    res_b = mode_b_time_holdout(d48)
    res_c = mode_c_d49t_from_d48(d48, d49t)

    # Persist
    out = EDA / "honest_baseline.json"
    with open(out, "w") as f:
        json.dump(
            {"mode_A_kfold_d48": res_a, "mode_B_time_holdout": res_b,
             "mode_C_d49t_from_d48": res_c},
            f, indent=2,
        )
    print(f"\nSaved: {out.relative_to(ROOT)}")

    # Summary
    print("\n" + "=" * 60)
    print("  HONEST BASELINE SUMMARY (baseline features only)")
    print("=" * 60)
    print(f"  A) KFold d48 (PRIMARY — most like test scenario) : "
          f"R²={res_a['blend_r2']:.4f}  (Score={res_a['score']:.2f})")
    print(f"  B) Time holdout slots 80-95 (stress test)        : "
          f"R²={res_b['blend_r2']:.4f}  (Score={res_b['score']:.2f})")
    print(f"  C) d49t from d48 (true day extrapolation)        : "
          f"R²={res_c['blend_r2']:.4f}  (Score={res_c['score']:.2f})")
    print(f"\n  Baseline (full KFold on d48+d49t) claims         : "
          f"R²=0.9616  (Score=96.16)  [optimistic]")
    print(f"\n  Use Mode A as the primary reference for C-2..C-8.")
    print("=" * 60)


if __name__ == "__main__":
    main()
