"""
improve_demand.py — Main improvement pipeline for Gridlock.

Implements C-2..C-8 from docs/plan.md. Each feature is toggleable so we can
A/B test the impact of every improvement on the honest metrics (Mode A/B/C
from scripts/honest_eval.py).

Usage:
    uv run python improve_demand.py                    # full pipeline
    uv run python improve_demand.py --quick            # quick mode (skip grid)
    uv run python improve_demand.py --config <name>    # use a specific config
"""

import argparse
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

ROOT = Path("/home/marcus/code/Gridlock")
DATA = ROOT / "dataset"
EDA = ROOT / "eda"
EDA.mkdir(exist_ok=True)


# =================================================================
# CONFIGURATION
# =================================================================
@dataclass
class Config:
    # C-2
    use_target_encoding: bool = False
    te_alpha: float = 20.0
    # C-3
    use_interactions: bool = False
    # C-4
    use_d49t_forward_lags: bool = False
    # C-5
    use_catboost: bool = False
    use_ridge_stacking: bool = False
    # C-6
    lgb_params: dict = field(default_factory=lambda: {
        "objective": "regression", "metric": "rmse",
        "learning_rate": 0.03, "num_leaves": 255, "min_child_samples": 5,
        "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 1,
        "reg_alpha": 0.05, "reg_lambda": 0.1,
        "n_jobs": 4, "verbose": -1, "seed": 42,
    })
    xgb_params: dict = field(default_factory=lambda: {
        "objective": "reg:squarederror", "eval_metric": "rmse",
        "learning_rate": 0.05, "max_depth": 8, "min_child_weight": 5,
        "subsample": 0.7, "colsample_bytree": 0.7,
        "reg_alpha": 0.05, "reg_lambda": 1.0,
        "n_jobs": 4, "seed": 42, "verbosity": 0,
    })
    cb_params: dict = field(default_factory=lambda: {
        "loss_function": "RMSE", "learning_rate": 0.05, "depth": 8,
        "l2_leaf_reg": 3.0, "iterations": 2000, "early_stopping_rounds": 100,
        "thread_count": 4, "random_seed": 42, "verbose": False,
    })
    # C-7
    use_tweedie: bool = False
    use_quantile: bool = False
    # C-8
    use_bias_correction: bool = False
    bias_correction_k: int = 20
    # E-1: per-geohash d49t features
    use_d49t_geo_features: bool = False
    # Misc
    n_folds: int = 5
    seed: int = 42
    seeds: tuple = (42,)  # for multi-seed averaging
    # Run mode
    run_modes: tuple = ("A", "B", "C")  # which honest modes to evaluate


# =================================================================
# TIME PARSING
# =================================================================
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
# FEATURE ENGINEERING
# =================================================================
BACK = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24]
FWD = [1, 2, 3, 4, 5, 6, 8]

ENCODED_CATS = {
    "wx": {"Sunny": 0, "Rainy": 1, "Foggy": 2, "Snowy": 3},
    "rd": {"Residential": 0, "Street": 1, "Highway": 2},
}


def add_categorical_encodings(df, src_for_temp_med):
    """Add road_enc, wx_enc, lv, lm_f, temp, temp2, is_hw, lanes2, ts_sin/cos, hr, peak flags, ratio1."""
    gt = src_for_temp_med["Temperature"].median()
    df["road_enc"] = df["RoadType"].map(ENCODED_CATS["rd"]).fillna(0).astype(int)
    df["wx_enc"] = df["Weather"].map(ENCODED_CATS["wx"]).fillna(0).astype(int)
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
    if "gts_mu" in df.columns and "g_mu" in df.columns:
        df["ratio1"] = df["gts_mu"] / (df["g_mu"] + 1e-8)
    return df


def add_interactions(df):
    """C-3: 6 interaction / extra time-of-day features."""
    if "wx_enc" in df.columns:
        df["wx_x_peak"] = df["wx_enc"] * ((df["is_mpk"] | df["is_epk"]).astype(int))
    if "temp" in df.columns:
        df["temp_x_peak"] = df["temp"] * ((df["is_mpk"] | df["is_epk"]).astype(int))
    if "NumberofLanes" in df.columns:
        df["lanes_x_hw"] = df["NumberofLanes"] * df["is_hw"]
    if "g_mu" in df.columns:
        df["gmu_x_peak"] = df["g_mu"] * ((df["is_mpk"] | df["is_epk"]).astype(int))
    df["is_lunch"] = df["hr"].between(11, 13).astype(int)
    df["is_quiet"] = df["hr"].between(17, 20).astype(int)
    return df


def add_target_encoding(df, src, alpha, kf, fold_assignments=None):
    """C-2: K-fold smoothed target encoding for geohash.

    For OOF (training): use 5-fold mean + prior.
    For non-OOF (test/d49t): use full src mean + prior.
    """
    gm = src["demand"].mean()
    n_src = src.groupby("geohash").size()
    g_mean = src.groupby("geohash")["demand"].mean()

    # Smoothed
    te = (n_src * g_mean + alpha * gm) / (n_src + alpha)
    te = te.reindex(df["geohash"]).values

    if fold_assignments is not None:
        # OOF: compute te per fold from the OTHER folds
        df["te_g"] = np.nan
        for fold_id in range(kf.get_n_splits()):
            tr_mask = fold_assignments != fold_id
            tr_src = src[tr_mask] if tr_mask.sum() > 0 else src
            n_tr = tr_src.groupby("geohash").size()
            g_mean_tr = tr_src.groupby("geohash")["demand"].mean()
            te_tr = (n_tr * g_mean_tr + alpha * gm) / (n_tr + alpha)
            in_fold = fold_assignments == fold_id
            df.loc[in_fold, "te_g"] = df.loc[in_fold, "geohash"].map(te_tr).fillna(gm).values
    else:
        df["te_g"] = pd.Series(te, index=df.index).fillna(gm).values

    # Interaction: te × is_hw
    df["te_g_x_hw"] = df["te_g"] * df["is_hw"]
    return df


def add_d49t_forward_lags(df, d49t_lookup):
    """C-4: 8 forward-lag features from d49t for test rows at time_slot 9-55."""
    for lag in range(1, 9):
        col = f"lp_d49t_{lag}"
        df[col] = df.apply(
            lambda r: d49t_lookup.get((r["geohash"], r["time_slot"] - lag), np.nan),
            axis=1,
        )
    return df


def add_d49t_geo_features(df, d49t, is_d49t_col=None):
    """E-1: 5 per-geohash d49t features.

    For d48 / test rows: use full d49t stats (no overlap with target).
    For d49t rows: use LOO for the mean (avoid the row's own demand in the mean);
    for std/min/max use full d49t (mild leakage is OK for trees).

    Adds columns:
      d49t_g_mu:   per-geohash d49t mean demand (LOO for d49t rows)
      d49t_g_sd:   per-geohash d49t std demand
      d49t_g_min:  per-geohash d49t min demand
      d49t_g_max:  per-geohash d49t max demand
      d49t_vs_d48: per-geohash day-shift ratio d49t_g_mu / g_mu (clipped to [0.1, 5.0])
    """
    if len(d49t) == 0:
        for c in ("d49t_g_mu", "d49t_g_sd", "d49t_g_min", "d49t_g_max", "d49t_vs_d48"):
            df[c] = np.nan
        return df
    g = d49t.groupby("geohash")["demand"].agg(
        d49t_g_mu="mean",
        d49t_g_sd="std",
        d49t_g_min="min",
        d49t_g_max="max",
        d49t_g_n="count",
    ).reset_index()
    df = df.merge(g, on="geohash", how="left")
    # LOO for d49t rows: (n * mu - y) / (n - 1)
    if is_d49t_col is not None and is_d49t_col in df.columns:
        mask = df[is_d49t_col].astype(bool) & (df["d49t_g_n"] > 1)
        if mask.any():
            loo = (df.loc[mask, "d49t_g_n"] * df.loc[mask, "d49t_g_mu"] -
                   df.loc[mask, "demand"]) / (df.loc[mask, "d49t_g_n"] - 1)
            df.loc[mask, "d49t_g_mu"] = loo.values
    # Fallback: if a row's geohash isn't in d49t, use the global d49t mean
    gm = d49t["demand"].mean()
    df["d49t_g_mu"] = df["d49t_g_mu"].fillna(gm)
    df["d49t_g_sd"] = df["d49t_g_sd"].fillna(d49t["demand"].std())
    df["d49t_g_min"] = df["d49t_g_min"].fillna(d49t["demand"].min())
    df["d49t_g_max"] = df["d49t_g_max"].fillna(d49t["demand"].max())
    # Day-shift ratio: d49t_g_mu / g_mu, clipped
    if "g_mu" in df.columns:
        df["d49t_vs_d48"] = (df["d49t_g_mu"] / (df["g_mu"] + 1e-8)).clip(0.1, 5.0)
    else:
        df["d49t_vs_d48"] = 1.0
    df = df.drop(columns=["d49t_g_n"], errors="ignore")
    return df


# =================================================================
# FEATURE ASSEMBLY
# =================================================================
def assemble_features(d48, d49t, test, cfg, mode="A", holdout_slot=80):
    """
    Build features for d48 (and optionally d49t, test) under the given config.

    mode: "A" (KFold d48), "B" (time holdout slots >= holdout_slot),
          "C" (predict d49t from d48), "FINAL" (full pipeline on d48+d49t -> test).
    Returns: (d48f, d49f_or_None, testf, gm, kf, fold_assignments_d48)
    """
    kf = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)

    if mode == "A":
        # All of d48 is "source"
        source_idx = pd.Series(True, index=d48.index)
        d48f = _build_d48_features(d48, source_idx, kf, cfg, d49t=d49t, with_oof=True)
        d49f, testf = None, None
        gm = d48["demand"].mean()
        return d48f, d49f, testf, gm, kf, np.zeros(len(d48), dtype=int)

    if mode == "B":
        # source = time_slot < holdout_slot
        source_idx = d48["time_slot"] < holdout_slot
        d48f = _build_d48_features(d48, source_idx, kf, cfg, d49t=d49t, with_oof=True)
        d49f, testf = None, None
        gm = d48[source_idx]["demand"].mean()
        return d48f, d49f, testf, gm, kf, np.zeros(len(d48), dtype=int)

    if mode == "C":
        # source = all of d48
        source_idx = pd.Series(True, index=d48.index)
        d48f = _build_d48_features(d48, source_idx, kf, cfg, d49t=d49t, with_oof=True)
        d49f = _build_d49t_features(d48, d49t, source_idx, cfg)
        testf = None
        gm = d48["demand"].mean()
        return d48f, d49f, testf, gm, kf, np.zeros(len(d48), dtype=int)

    if mode == "FINAL":
        # Train on d48+d49t, predict test
        source_idx = pd.Series(True, index=d48.index)
        d48f = _build_d48_features(d48, source_idx, kf, cfg, d49t=d49t, with_oof=True)
        d49f = _build_d49t_features(d48, d49t, source_idx, cfg)
        testf = _build_test_features(d48, d49t, test, cfg)
        gm = d48["demand"].mean()
        return d48f, d49f, testf, gm, kf, np.zeros(len(d48), dtype=int)

    raise ValueError(f"Unknown mode: {mode}")


def _build_d48_features(d48, source_idx, kf, cfg, d49t=None, with_oof=True):
    """Build all features for d48 (used as both training and validation)."""
    df = d48.copy()
    src = df[source_idx].copy()
    gm = src["demand"].mean()

    # Backward lags from source only
    lm_idx = src.set_index(["geohash", "time_slot"])["demand"]
    for lag in BACK:
        df[f"lm{lag}"] = df.apply(
            lambda r: lm_idx.get((r["geohash"], r["time_slot"] - lag), np.nan),
            axis=1,
        )
    for lag in FWD:
        df[f"lp{lag}"] = np.nan

    # Geohash stats — OOF within source for source rows, full source for non-source
    gc = ["g_mu", "g_sd", "g_med", "g_p25", "g_p75", "g_iqr"]
    for c in gc:
        df[c] = np.nan
    for ti, vi in kf.split(src):
        fs = _agg_geohash(src.iloc[ti])
        va = src.iloc[vi][["geohash"]].merge(fs, on="geohash", how="left")
        for c in gc:
            df.loc[src.index[vi], c] = va[c].values
    # Non-source rows: full source aggregate
    non_src = df[~source_idx]
    if len(non_src) > 0:
        full_fs = _agg_geohash(src)
        tmp = non_src[["geohash"]].merge(full_fs, on="geohash", how="left")
        for c in gc:
            df.loc[non_src.index, c] = tmp[c].values
    for c in gc:
        df[c] = df[c].fillna(gm)

    # (geohash, time_slot) stats — OOF + non-source from full
    for c in ["gts_mu", "gts_sd"]:
        df[c] = np.nan
    for ti, vi in kf.split(src):
        fgt = (
            src.iloc[ti].groupby(["geohash", "time_slot"])["demand"]
            .agg(gts_mu="mean", gts_sd="std").reset_index()
        )
        va = src.iloc[vi][["geohash", "time_slot"]].merge(
            fgt, on=["geohash", "time_slot"], how="left"
        )
        df.loc[src.index[vi], "gts_mu"] = va["gts_mu"].values
        df.loc[src.index[vi], "gts_sd"] = va["gts_sd"].values
    if len(non_src) > 0:
        full_fgt = (
            src.groupby(["geohash", "time_slot"])["demand"]
            .agg(gts_mu="mean", gts_sd="std").reset_index()
        )
        tmp = non_src[["geohash", "time_slot"]].merge(
            full_fgt, on=["geohash", "time_slot"], how="left"
        )
        df.loc[non_src.index, "gts_mu"] = tmp["gts_mu"].values
        df.loc[non_src.index, "gts_sd"] = tmp["gts_sd"].values
    df["gts_mu"] = df["gts_mu"].fillna(df["g_mu"])
    df["gts_sd"] = df["gts_sd"].fillna(df["g_sd"])

    df = add_categorical_encodings(df, src)
    if cfg.use_interactions:
        df = add_interactions(df)
    if cfg.use_target_encoding:
        # OOF target encoding
        fold_assignments = np.zeros(len(src), dtype=int)
        for fold_id, (_, vi) in enumerate(kf.split(src)):
            fold_assignments[vi] = fold_id
        full_fold = np.full(len(df), -1, dtype=int)
        full_fold[source_idx.values] = fold_assignments
        df = add_target_encoding(
            df, src, cfg.te_alpha, kf,
            fold_assignments=full_fold if with_oof else None,
        )
    # Add d49t forward lag columns as NaN (only test has them populated)
    if cfg.use_d49t_forward_lags:
        for lag in range(1, 9):
            df[f"lp_d49t_{lag}"] = np.nan

    # E-1: per-geohash d49t features (no leak: d48 rows are not in d49t)
    if cfg.use_d49t_geo_features and d49t is not None:
        df = add_d49t_geo_features(df, d49t, is_d49t_col=None)

    return df


def _agg_geohash(d):
    fs = (
        d.groupby("geohash")["demand"]
        .agg(g_mu="mean", g_sd="std", g_med="median",
             p25=lambda x: x.quantile(0.25), p75=lambda x: x.quantile(0.75))
        .reset_index()
        .rename(columns={"p25": "g_p25", "p75": "g_p75"})
    )
    fs["g_iqr"] = fs["g_p75"] - fs["g_p25"]
    return fs


def _build_d49t_features(d48, d49t, source_idx, cfg):
    """Build features for d49t using d48 (or d48[source_idx]) as source."""
    df = d49t.copy()
    src = d48[source_idx].copy()
    gm = src["demand"].mean()

    # Backward lags from d48 (full d48 or d48 source)
    lm_idx = src.set_index(["geohash", "time_slot"])["demand"]
    for lag in BACK:
        df[f"lm{lag}"] = df.apply(
            lambda r: lm_idx.get((r["geohash"], r["time_slot"] - lag), np.nan),
            axis=1,
        )
    for lag in FWD:
        df[f"lp{lag}"] = np.nan

    # Geohash stats (full source)
    full_fs = _agg_geohash(src)
    gc = ["g_mu", "g_sd", "g_med", "g_p25", "g_p75", "g_iqr"]
    tmp = df[["geohash"]].merge(full_fs, on="geohash", how="left")
    for c in gc:
        df[c] = tmp[c].fillna(gm).values

    # (geohash, time_slot) stats from source
    full_fgt = (
        src.groupby(["geohash", "time_slot"])["demand"]
        .agg(gts_mu="mean", gts_sd="std").reset_index()
    )
    tmp = df[["geohash", "time_slot"]].merge(
        full_fgt, on=["geohash", "time_slot"], how="left"
    )
    df["gts_mu"] = tmp["gts_mu"].fillna(df["g_mu"]).values
    df["gts_sd"] = tmp["gts_sd"].fillna(df["g_sd"]).values

    df = add_categorical_encodings(df, src)
    if cfg.use_interactions:
        df = add_interactions(df)
    if cfg.use_target_encoding:
        df = add_target_encoding(df, src, cfg.te_alpha, kf=None)
    # Add d49t forward lag columns as NaN
    if cfg.use_d49t_forward_lags:
        for lag in range(1, 9):
            df[f"lp_d49t_{lag}"] = np.nan
    # E-1: per-geohash d49t features. These rows ARE d49t, so LOO the mean.
    if cfg.use_d49t_geo_features:
        df["_is_d49t"] = True
        df = add_d49t_geo_features(df, d49t, is_d49t_col="_is_d49t")
        df = df.drop(columns=["_is_d49t"], errors="ignore")
    return df


def _build_test_features(d48, d49t, test, cfg):
    """Build features for test using d48 (full) and d49t (for forward lags)."""
    df = test.copy()
    src = d48.copy()  # full d48 for test
    gm = src["demand"].mean()

    # Backward lags from d48 (full)
    lm_idx = src.set_index(["geohash", "time_slot"])["demand"]
    for lag in BACK:
        df[f"lm{lag}"] = df.apply(
            lambda r: lm_idx.get((r["geohash"], r["time_slot"] - lag), np.nan),
            axis=1,
        )
    # Forward lags from d48 (mirrors baseline behavior)
    for lag in FWD:
        df[f"lp{lag}"] = df.apply(
            lambda r: lm_idx.get((r["geohash"], r["time_slot"] + lag), np.nan),
            axis=1,
        )

    # Geohash stats from d48 (full)
    full_fs = _agg_geohash(src)
    gc = ["g_mu", "g_sd", "g_med", "g_p25", "g_p75", "g_iqr"]
    tmp = df[["geohash"]].merge(full_fs, on="geohash", how="left")
    for c in gc:
        df[c] = tmp[c].fillna(gm).values

    # (geohash, time_slot) stats from d48
    full_fgt = (
        src.groupby(["geohash", "time_slot"])["demand"]
        .agg(gts_mu="mean", gts_sd="std").reset_index()
    )
    tmp = df[["geohash", "time_slot"]].merge(
        full_fgt, on=["geohash", "time_slot"], how="left"
    )
    df["gts_mu"] = tmp["gts_mu"].fillna(df["g_mu"]).values
    df["gts_sd"] = tmp["gts_sd"].fillna(df["g_sd"]).values

    # C-4: forward lags from d49t
    if cfg.use_d49t_forward_lags:
        d49_lookup = d49t.set_index(["geohash", "time_slot"])["demand"]
        df = add_d49t_forward_lags(df, d49_lookup)
    else:
        # Add NaN columns so feature list is consistent across frames
        for lag in range(1, 9):
            df[f"lp_d49t_{lag}"] = np.nan

    df = add_categorical_encodings(df, src)
    if cfg.use_interactions:
        df = add_interactions(df)
    if cfg.use_target_encoding:
        df = add_target_encoding(df, src, cfg.te_alpha, kf=None)
    # E-1: per-geohash d49t features. Test rows are not in d49t, so no LOO.
    if cfg.use_d49t_geo_features:
        df = add_d49t_geo_features(df, d49t, is_d49t_col=None)
    return df


# =================================================================
# FEATURE LIST
# =================================================================
def get_feature_list(cfg):
    feats = (
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
    if cfg.use_interactions:
        feats += ["wx_x_peak", "temp_x_peak", "lanes_x_hw", "gmu_x_peak",
                  "is_lunch", "is_quiet"]
    if cfg.use_target_encoding:
        feats += ["te_g", "te_g_x_hw"]
    if cfg.use_d49t_forward_lags:
        feats += [f"lp_d49t_{l}" for l in range(1, 9)]
    if cfg.use_d49t_geo_features:
        feats += ["d49t_g_mu", "d49t_g_sd", "d49t_g_min", "d49t_g_max", "d49t_vs_d48"]
    return feats


# =================================================================
# MODEL TRAINING
# =================================================================
def train_lgb(X_tr, y_tr, X_va, y_va, params):
    dt = lgb.Dataset(X_tr, y_tr)
    dv = lgb.Dataset(X_va, y_va, reference=dt)
    m = lgb.train(
        params, dt, num_boost_round=2000, valid_sets=[dv],
        callbacks=[lgb.early_stopping(150, verbose=False)],
    )
    return m.predict(X_va, num_iteration=m.best_iteration), m


def train_xgb(X_tr, y_tr, X_va, y_va, params):
    dtr = xgb.DMatrix(X_tr, y_tr)
    dva = xgb.DMatrix(X_va, y_va)
    m = xgb.train(
        params, dtr, num_boost_round=2000,
        evals=[(dva, "val")], early_stopping_rounds=100, verbose_eval=False,
    )
    return m.predict(dva), m


def fit_oof_kfold(X, y, kf, train_fn, params):
    """Returns OOF predictions array."""
    oof = np.zeros(len(X))
    iters = []
    for fold, (ti, vi) in enumerate(kf.split(X), 1):
        X_tr, X_va = X.iloc[ti], X.iloc[vi]
        y_tr, y_va = y.iloc[ti], y.iloc[vi]
        pred, m = train_fn(X_tr, y_tr, X_va, y_va, params)
        oof[vi] = pred
        iters.append(getattr(m, "best_iteration", 0) or 0)
    return oof, iters


# =================================================================
# EVALUATION
# =================================================================
def evaluate_mode(d48f, d49f, cfg, mode, holdout_slot=80, train_test_split=None):
    """Run all models on a given mode and return R² metrics."""
    FEATS = get_feature_list(cfg)
    results = {}

    if mode == "A":
        X = d48f[FEATS].astype(float).fillna(-999)
        y = d48f["demand"].astype(float)
        kf = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    elif mode == "B":
        tr_mask = d48f["time_slot"] < holdout_slot
        X_tr = d48f[tr_mask][FEATS].astype(float).fillna(-999)
        y_tr = d48f[tr_mask]["demand"].astype(float)
        X_va = d48f[~tr_mask][FEATS].astype(float).fillna(-999)
        y_va = d48f[~tr_mask]["demand"].astype(float)
        X, y, kf = (X_tr, y_tr, None), None, None
    elif mode == "C":
        X = d48f[FEATS].astype(float).fillna(-999)
        y = d48f["demand"].astype(float)
        X_va = d49f[FEATS].astype(float).fillna(-999)
        y_va = d49f["demand"].astype(float)
        kf = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    else:
        raise ValueError(mode)

    # LGB
    if mode == "B":
        lgb_pred, _ = train_lgb(X_tr, y_tr, X_va, y_va, cfg.lgb_params)
        lgb_oof = None
        lgb_r2 = r2_score(y_va, lgb_pred)
    else:
        lgb_oof, _ = fit_oof_kfold(X, y, kf, train_lgb, cfg.lgb_params)
        lgb_r2 = r2_score(y, lgb_oof) if mode == "A" else r2_score(y_va, lgb_oof)

    # XGB
    if mode == "B":
        xgb_pred, _ = train_xgb(X_tr, y_tr, X_va, y_va, cfg.xgb_params)
        xgb_oof = None
        xgb_r2 = r2_score(y_va, xgb_pred)
    else:
        xgb_oof, _ = fit_oof_kfold(X, y, kf, train_xgb, cfg.xgb_params)
        xgb_r2 = r2_score(y, xgb_oof) if mode == "A" else r2_score(y_va, xgb_oof)

    # Tweedie / quantile (only on mode A for speed)
    tw_r2 = None
    qt_r2 = None
    if (cfg.use_tweedie or cfg.use_quantile) and mode == "A":
        if cfg.use_tweedie:
            tw_params = dict(cfg.lgb_params)
            tw_params["objective"] = "tweedie"
            tw_params["tweedie_variance_power"] = 1.5
            tw_oof, _ = fit_oof_kfold(X, y, kf, train_lgb, tw_params)
            tw_r2 = r2_score(y, tw_oof)
        if cfg.use_quantile:
            qt_params = dict(cfg.lgb_params)
            qt_params["objective"] = "quantile"
            qt_params["alpha"] = 0.5
            qt_oof, _ = fit_oof_kfold(X, y, kf, train_lgb, qt_params)
            qt_r2 = r2_score(y, qt_oof)

    # CatBoost
    cb_r2 = None
    cb_oof = None
    if cfg.use_catboost and mode == "A":
        try:
            from catboost import CatBoostRegressor
            cb_oof_arr = np.zeros(len(X))
            for fold, (ti, vi) in enumerate(kf.split(X), 1):
                params = dict(cfg.cb_params)
                m = CatBoostRegressor(**params)
                m.fit(X.iloc[ti], y.iloc[ti],
                      eval_set=(X.iloc[vi], y.iloc[vi]),
                      verbose=False)
                cb_oof_arr[vi] = m.predict(X.iloc[vi])
            cb_oof = cb_oof_arr
            cb_r2 = r2_score(y, cb_oof)
        except ImportError:
            print("  ! catboost not installed; skipping")
            cfg.use_catboost = False

    # Blend
    if mode == "A":
        if cfg.use_ridge_stacking and cb_oof is not None:
            # Stack with Ridge, positive weights
            cols = [lgb_oof, xgb_oof, cb_oof]
            if tw_oof is not None:
                cols.append(tw_oof)
            if qt_oof is not None:
                cols.append(qt_oof)
            Xstack = np.column_stack(cols)
            blend = Ridge(alpha=1.0, positive=True, fit_intercept=False).fit(Xstack, y)
            oof_blend = blend.predict(Xstack)
            blend_r2 = r2_score(y, oof_blend)
            blend_w = blend.coef_
        else:
            bw, br = -1, -1
            for w in np.arange(0.0, 1.01, 0.02):
                r = r2_score(y, w * lgb_oof + (1 - w) * xgb_oof)
                if r > br:
                    br, bw = r, w
            blend_r2 = br
            blend_w = np.array([bw, 1 - bw])
    else:
        if mode == "B":
            bw, br = -1, -1
            for w in np.arange(0.0, 1.01, 0.02):
                r = r2_score(y_va, w * lgb_pred + (1 - w) * xgb_pred)
                if r > br:
                    br, bw = r, w
            blend_r2 = br
            blend_w = np.array([bw, 1 - bw])
            lgb_oof, xgb_oof = lgb_pred, xgb_pred
        else:  # C
            bw, br = -1, -1
            for w in np.arange(0.0, 1.01, 0.02):
                r = r2_score(y_va, w * lgb_oof + (1 - w) * xgb_oof)
                if r > br:
                    br, bw = r, w
            blend_r2 = br
            blend_w = np.array([bw, 1 - bw])

    # Bias correction
    if cfg.use_bias_correction and mode == "A":
        # Compute per-geohash mean residual and apply
        resid = y.values - (blend_w[0] * lgb_oof + blend_w[1] * xgb_oof)
        df_for_bias = d48f.copy()
        df_for_bias["resid"] = resid
        g_bias = df_for_bias.groupby("geohash")["resid"].agg(g_bias_sum="sum", g_n="count")
        g_bias["g_bias_mean"] = g_bias["g_bias_sum"] / g_bias["g_n"]
        k = cfg.bias_correction_k
        g_bias["g_bias_shrunk"] = g_bias["g_bias_mean"] * g_bias["g_n"] / (g_bias["g_n"] + k)
        bias_lookup = g_bias["g_bias_shrunk"].to_dict()
        d48f["_bias"] = d48f["geohash"].map(bias_lookup).fillna(0).values
        corrected_pred = (blend_w[0] * lgb_oof + blend_w[1] * xgb_oof) + d48f["_bias"].values
        bias_r2 = r2_score(y, corrected_pred)
        # Update blend score
        results["bias_correction_r2"] = float(bias_r2)
        # Use bias-corrected score as primary
        blend_r2 = bias_r2

    return {
        "mode": mode,
        "lgb_r2": float(lgb_r2),
        "xgb_r2": float(xgb_r2),
        "tweedie_r2": float(tw_r2) if tw_r2 is not None else None,
        "quantile_r2": float(qt_r2) if qt_r2 is not None else None,
        "catboost_r2": float(cb_r2) if cb_r2 is not None else None,
        "blend_weights": blend_w.tolist() if hasattr(blend_w, "tolist") else list(blend_w),
        "blend_r2": float(blend_r2),
        "score": float(max(0, 100 * blend_r2)),
    }


# =================================================================
# MAIN
# =================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Skip CatBoost, tweedie, quantile")
    parser.add_argument("--mode", default="all", help="A, B, C, all, or final")
    parser.add_argument("--config", default="default", help="named config preset")
    args = parser.parse_args()

    train_df, test_df = load_data()
    d48 = train_df[train_df["day"] == 48].reset_index(drop=True)
    d49t = train_df[train_df["day"] == 49].reset_index(drop=True)
    test = test_df.copy().reset_index(drop=True)
    print(f"d48: {d48.shape}    d49t: {d49t.shape}    test: {test.shape}")

    # Build a default config (all improvements enabled)
    cfg = Config(
        use_target_encoding=True,
        te_alpha=20.0,
        use_interactions=True,
        use_d49t_forward_lags=True,
        use_catboost=not args.quick,
        use_ridge_stacking=True,
        use_tweedie=not args.quick,
        use_quantile=not args.quick,
        use_bias_correction=True,
    )

    all_results = {}

    modes = ["A", "B", "C"] if args.mode == "all" else [args.mode]
    for mode in modes:
        print(f"\n{'=' * 70}\n  Mode {mode}\n{'=' * 70}")
        d48f, d49f, _, _, _, _ = assemble_features(d48, d49t, test, cfg, mode=mode)
        res = evaluate_mode(d48f, d49f, cfg, mode)
        print(f"  LGB R²  : {res['lgb_r2']:.4f}")
        print(f"  XGB R²  : {res['xgb_r2']:.4f}")
        if res.get("tweedie_r2"):
            print(f"  Tweedie : {res['tweedie_r2']:.4f}")
        if res.get("quantile_r2"):
            print(f"  Quantile: {res['quantile_r2']:.4f}")
        if res.get("catboost_r2"):
            print(f"  CatBoost: {res['catboost_r2']:.4f}")
        print(f"  Blend weights: {[round(w, 3) for w in res['blend_weights']]}")
        print(f"  Blend R² : {res['blend_r2']:.4f}  (Score={res['score']:.2f})")
        if "bias_correction_r2" in res:
            print(f"  After bias correction: {res['bias_correction_r2']:.4f}  (Score={res['bias_correction_r2']*100:.2f})")
        all_results[mode] = res

    out = EDA / "improve_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out.relative_to(ROOT)}")

    # Final summary
    print("\n" + "=" * 70)
    print("  FULL PIPELINE SUMMARY")
    print("=" * 70)
    for m, r in all_results.items():
        print(f"  Mode {m}: Blend R²={r['blend_r2']:.4f}  Score={r['score']:.2f}")
    print(f"  Baseline (Mode A, no improvements): R²=0.9654, Score=96.54")


if __name__ == "__main__":
    main()
