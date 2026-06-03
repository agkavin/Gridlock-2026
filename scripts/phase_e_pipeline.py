"""
Phase E — Bridge the OOF vs online gap.

Builds on the Phase D pipeline with three changes:
  E-1: per-geohash d49t features (d49t_g_mu, d49t_g_sd, d49t_g_min/max, d49t_vs_d48)
  E-2: multi-seed averaging (default 3 seeds: 42, 7, 123)
  E-3: bias correction OFF (overfits d48 KFold residuals; may hurt cross-day)

Generates submission_v2.csv (does NOT overwrite submission.csv).
Saves metrics and feature importance for comparison with V1.
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

import lightgbm as lgb
import xgboost as xgb

sys.path.insert(0, str(Path("/home/marcus/code/Gridlock")))
from improve_demand import (
    Config, assemble_features, load_data, get_feature_list,
    train_lgb, train_xgb, fit_oof_kfold,
)

warnings.filterwarnings("ignore")

ROOT = Path("/home/marcus/code/Gridlock")
DATA = ROOT / "dataset"
EDA = ROOT / "eda"
EDA.mkdir(exist_ok=True)


def make_phase_e_cfg():
    """Phase E config: E-1 ON, E-2 multi-seed, E-3 bias off, C-3/C-4/C-5/C-6 ON."""
    return Config(
        use_target_encoding=False,
        use_interactions=True,
        use_d49t_forward_lags=True,
        use_catboost=True,
        use_ridge_stacking=True,
        use_tweedie=False,
        use_quantile=False,
        use_bias_correction=False,   # E-3
        use_d49t_geo_features=True,  # E-1
        # E-2: 3 seeds
        seeds=(42, 7, 123),
        lgb_params={
            "objective": "regression", "metric": "rmse",
            "learning_rate": 0.02, "num_leaves": 255, "min_child_samples": 3,
            "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 1,
            "reg_alpha": 0.05, "reg_lambda": 0.1,
            "n_jobs": 4, "verbose": -1, "seed": 42,
        },
        xgb_params={
            "objective": "reg:squarederror", "eval_metric": "rmse",
            "learning_rate": 0.05, "max_depth": 8, "min_child_weight": 5,
            "subsample": 0.7, "colsample_bytree": 0.7,
            "reg_alpha": 0.05, "reg_lambda": 1.0,
            "n_jobs": 4, "seed": 42, "verbosity": 0,
        },
        cb_params={
            "loss_function": "RMSE", "learning_rate": 0.05, "depth": 8,
            "l2_leaf_reg": 3.0, "iterations": 1500, "early_stopping_rounds": 50,
            "thread_count": 4, "random_seed": 42, "verbose": False,
        },
    )


def add_catboost_columns(df):
    df = df.copy()
    df["geohash_cat"] = df["geohash"].astype(str)
    df["RoadType_cat"] = df["RoadType"].fillna("NaN").astype(str)
    df["Weather_cat"] = df["Weather"].fillna("NaN").astype(str)
    df["LargeVehicles_cat"] = df["LargeVehicles"].astype(str)
    df["Landmarks_cat"] = df["Landmarks"].astype(str)
    return df


CAT_COLS = ["geohash_cat", "RoadType_cat", "Weather_cat",
            "LargeVehicles_cat", "Landmarks_cat"]


def train_cb_with_cat(X_tr, y_tr, X_va, y_va, params, cat_cols):
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(**params)
    m.fit(X_tr, y_tr, eval_set=(X_va, y_va),
          cat_features=cat_cols, verbose=False)
    return m.predict(X_va), m


def fit_oof_kfold_cat(X, y, kf, params, cat_cols):
    oof = np.zeros(len(X))
    for ti, vi in kf.split(X):
        _, m = train_cb_with_cat(X.iloc[ti], y.iloc[ti], X.iloc[vi], y.iloc[vi],
                                  params, cat_cols)
        oof[vi] = m.predict(X.iloc[vi])
    return oof


def fit_test_kfold_cat(X, y, X_test, kf, params, cat_cols):
    test_preds = np.zeros(len(X_test))
    for ti, vi in kf.split(X):
        _, m = train_cb_with_cat(X.iloc[ti], y.iloc[ti], X.iloc[vi], y.iloc[vi],
                                  params, cat_cols)
        test_preds += m.predict(X_test) / kf.get_n_splits()
    return test_preds


def run_mode_a(cfg, d48, d49t, test, FEATS):
    """Mode A honest eval (single seed for speed)."""
    print(f"\n[Mode A] KFold d48 — sanity check on Phase E features", flush=True)
    d48f, _, _, _, _, _ = assemble_features(d48, d49t, test, cfg, mode="A")
    X = d48f[FEATS].astype(float).fillna(-999)
    y = d48f["demand"].astype(float)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    lgb_oof, _ = fit_oof_kfold(X, y, kf, train_lgb, cfg.lgb_params)
    xgb_oof, _ = fit_oof_kfold(X, y, kf, train_xgb, cfg.xgb_params)
    d48f_cb = add_catboost_columns(d48f)
    X_cb = d48f_cb[FEATS + CAT_COLS]
    cb_oof = fit_oof_kfold_cat(X_cb, y, kf, cfg.cb_params, CAT_COLS)
    Xstack = np.column_stack([lgb_oof, xgb_oof, cb_oof])
    blend = Ridge(alpha=1.0, positive=True, fit_intercept=False).fit(Xstack, y)
    oof_blend = blend.predict(Xstack)
    blend_r2 = r2_score(y, oof_blend)
    print(f"    Mode A weights: {blend.coef_.round(3).tolist()}", flush=True)
    print(f"    Mode A  R² (blend)        : {blend_r2:.4f}  (Score={blend_r2*100:.2f})", flush=True)
    print(f"    Mode A  LGB  R²: {r2_score(y, lgb_oof):.4f}", flush=True)
    print(f"    Mode A  XGB  R²: {r2_score(y, xgb_oof):.4f}", flush=True)
    print(f"    Mode A  CB   R²: {r2_score(y, cb_oof):.4f}", flush=True)
    return {
        "blend_r2": float(blend_r2),
        "blend_weights": blend.coef_.tolist(),
        "lgb_r2": float(r2_score(y, lgb_oof)),
        "xgb_r2": float(r2_score(y, xgb_oof)),
        "cb_r2": float(r2_score(y, cb_oof)),
    }


def run_mode_c(cfg, d48, d49t, test, FEATS):
    """Mode C: train on d48, predict d49t (true day shift)."""
    print(f"\n[Mode C] Predict d49t from d48 — true day shift", flush=True)
    d48f, d49f, _, _, _, _ = assemble_features(d48, d49t, test, cfg, mode="C")
    X = d48f[FEATS].astype(float).fillna(-999)
    y = d48f["demand"].astype(float)
    X_va = d49f[FEATS].astype(float).fillna(-999)
    y_va = d49f["demand"].astype(float)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    lgb_oof, _ = fit_oof_kfold(X, y, kf, train_lgb, cfg.lgb_params)
    xgb_oof, _ = fit_oof_kfold(X, y, kf, train_xgb, cfg.xgb_params)
    d48f_cb = add_catboost_columns(d48f)
    d49f_cb = add_catboost_columns(d49f)
    X_cb = d48f_cb[FEATS + CAT_COLS]
    X_va_cb = d49f_cb[FEATS + CAT_COLS]
    cb_oof = fit_oof_kfold_cat(X_cb, y, kf, cfg.cb_params, CAT_COLS)
    # Train on full d48, predict d49t
    lgb_pred, _ = train_lgb(X, y, X_va, y_va, cfg.lgb_params)
    xgb_pred, _ = train_xgb(X, y, X_va, y_va, cfg.xgb_params)
    _, cb_final = train_cb_with_cat(X_cb, y, X_va_cb, y_va, cfg.cb_params, CAT_COLS)
    cb_pred = cb_final.predict(X_va_cb)
    # Ridge stack: use OOF (in d48) weights
    Xstack = np.column_stack([lgb_oof, xgb_oof, cb_oof])
    blend = Ridge(alpha=1.0, positive=True, fit_intercept=False).fit(Xstack, y)
    pred_stack = np.column_stack([lgb_pred, xgb_pred, cb_pred])
    blend_pred = blend.predict(pred_stack)
    blend_r2_C = r2_score(y_va, blend_pred)
    print(f"    Mode C  R² (blend)        : {blend_r2_C:.4f}  (Score={blend_r2_C*100:.2f})", flush=True)
    print(f"    Mode C  LGB  R²: {r2_score(y_va, lgb_pred):.4f}", flush=True)
    print(f"    Mode C  XGB  R²: {r2_score(y_va, xgb_pred):.4f}", flush=True)
    print(f"    Mode C  CB   R²: {r2_score(y_va, cb_pred):.4f}", flush=True)
    return {
        "blend_r2": float(blend_r2_C),
        "blend_weights": blend.coef_.tolist(),
        "lgb_r2": float(r2_score(y_va, lgb_pred)),
        "xgb_r2": float(r2_score(y_va, xgb_pred)),
        "cb_r2": float(r2_score(y_va, cb_pred)),
    }


def run_final_multi_seed(cfg, d48, d49t, test, FEATS):
    """Final training on d48+d49t, multi-seed averaging."""
    print(f"\n[FINAL] Multi-seed training on d48 + d49t → test", flush=True)
    d48f, d49f, testf, gm, _, _ = assemble_features(d48, d49t, test, cfg, mode="FINAL")
    d48f["_origin"] = "d48"
    d49f["_origin"] = "d49t"
    testf["_origin"] = "test"
    combined = pd.concat([d48f, d49f], ignore_index=True)
    print(f"  combined train: {combined.shape}", flush=True)
    print(f"  test: {testf.shape}", flush=True)

    combined_cb = add_catboost_columns(combined)
    testf_cb = add_catboost_columns(testf)
    X = combined[FEATS].astype(float).fillna(-999)
    y = combined["demand"].astype(float)
    X_test = testf[FEATS].astype(float).fillna(-999)
    X_cb = combined_cb[FEATS + CAT_COLS]
    X_test_cb = testf_cb[FEATS + CAT_COLS]

    seeds = list(cfg.seeds)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    lgb_oof_acc = np.zeros(len(X))
    xgb_oof_acc = np.zeros(len(X))
    cb_oof_acc = np.zeros(len(X))
    lgb_test_acc = np.zeros(len(X_test))
    xgb_test_acc = np.zeros(len(X_test))
    cb_test_acc = np.zeros(len(X_test))

    for s in seeds:
        t0 = time.time()
        print(f"  [seed {s}] training LGB...", flush=True)
        lgb_params = dict(cfg.lgb_params); lgb_params["seed"] = s
        xgb_params = dict(cfg.xgb_params); xgb_params["seed"] = s
        cb_params = dict(cfg.cb_params); cb_params["random_seed"] = s

        lgb_oof, _ = fit_oof_kfold(X, y, kf, train_lgb, lgb_params)
        lgb_test = np.zeros(len(X_test))
        for ti, vi in kf.split(X):
            _, m = train_lgb(X.iloc[ti], y.iloc[ti], X.iloc[vi], y.iloc[vi], lgb_params)
            lgb_test += m.predict(X_test, num_iteration=m.best_iteration) / 5
        lgb_oof_acc += lgb_oof / len(seeds)
        lgb_test_acc += lgb_test / len(seeds)
        print(f"    LGB seed {s} OOF R²: {r2_score(y, lgb_oof):.4f}  "
              f"({time.time()-t0:.1f}s)", flush=True)

        t1 = time.time()
        print(f"  [seed {s}] training XGB...", flush=True)
        xgb_oof, _ = fit_oof_kfold(X, y, kf, train_xgb, xgb_params)
        xgb_test = np.zeros(len(X_test))
        for ti, vi in kf.split(X):
            _, m = train_xgb(X.iloc[ti], y.iloc[ti], X.iloc[vi], y.iloc[vi], xgb_params)
            xgb_test += m.predict(xgb.DMatrix(X_test)) / 5
        xgb_oof_acc += xgb_oof / len(seeds)
        xgb_test_acc += xgb_test / len(seeds)
        print(f"    XGB seed {s} OOF R²: {r2_score(y, xgb_oof):.4f}  "
              f"({time.time()-t1:.1f}s)", flush=True)

        t2 = time.time()
        print(f"  [seed {s}] training CatBoost (native cat)...", flush=True)
        cb_oof = fit_oof_kfold_cat(X_cb, y, kf, cb_params, CAT_COLS)
        cb_test = fit_test_kfold_cat(X_cb, y, X_test_cb, kf, cb_params, CAT_COLS)
        cb_oof_acc += cb_oof / len(seeds)
        cb_test_acc += cb_test / len(seeds)
        print(f"    CB seed {s} OOF R²: {r2_score(y, cb_oof):.4f}  "
              f"({time.time()-t2:.1f}s)", flush=True)

    # Ridge stack on averaged OOF
    Xstack = np.column_stack([lgb_oof_acc, xgb_oof_acc, cb_oof_acc])
    blend = Ridge(alpha=1.0, positive=True, fit_intercept=False).fit(Xstack, y)
    print(f"  Final Ridge weights: {blend.coef_.round(3).tolist()}", flush=True)

    oof_blend = blend.predict(Xstack)
    final_oof_r2 = r2_score(y, oof_blend)
    print(f"  Final OOF R²: {final_oof_r2:.4f}  (Score={final_oof_r2*100:.2f})", flush=True)

    # Build test predictions
    test_stack = np.column_stack([lgb_test_acc, xgb_test_acc, cb_test_acc])
    test_blend = blend.predict(test_stack)
    test_blend = np.clip(test_blend, 0, 1)

    sub = pd.DataFrame({"Index": test["Index"], "demand": test_blend})
    sub.to_csv(ROOT / "submission_v2.csv", index=False)
    print(f"\n  Saved submission: submission_v2.csv  ({sub.shape})", flush=True)

    # Per-model OOF (averaged across seeds) for diagnostics
    print(f"\n  Per-model averaged OOF R²:", flush=True)
    print(f"    LGB: {r2_score(y, lgb_oof_acc):.4f}", flush=True)
    print(f"    XGB: {r2_score(y, xgb_oof_acc):.4f}", flush=True)
    print(f"    CB : {r2_score(y, cb_oof_acc):.4f}", flush=True)
    print(f"    Blend: {final_oof_r2:.4f}", flush=True)

    # Feature importance (LGB gain) on first seed
    print(f"\n  Computing feature importance (LGB gain)...", flush=True)
    dt = lgb.Dataset(X, y)
    dv = lgb.Dataset(X.iloc[:1000], y.iloc[:1000], reference=dt)
    m_imp = lgb.train(cfg.lgb_params, dt, num_boost_round=300, valid_sets=[dv],
                      callbacks=[lgb.log_evaluation(0)])
    imp = pd.DataFrame({
        "feature": X.columns,
        "gain": m_imp.feature_importance(importance_type="gain"),
        "split": m_imp.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    imp.to_csv(EDA / "feature_importance_v2.csv", index=False)
    print(f"  Saved feature importance: eda/feature_importance_v2.csv", flush=True)
    print(f"  Top 15 features:", flush=True)
    for _, row in imp.head(15).iterrows():
        print(f"    {row['feature']:25s}  gain={row['gain']:>10.0f}", flush=True)

    return sub, final_oof_r2, blend.coef_.tolist(), imp


def main():
    print("=" * 70, flush=True)
    print("  PHASE E — Bridge OOF vs Online gap", flush=True)
    print("=" * 70, flush=True)
    t_total = time.time()

    cfg = make_phase_e_cfg()
    print(f"\n  Phase E config:", flush=True)
    print(f"    use_interactions     = {cfg.use_interactions}", flush=True)
    print(f"    use_d49t_forward_lags= {cfg.use_d49t_forward_lags}", flush=True)
    print(f"    use_catboost         = {cfg.use_catboost}", flush=True)
    print(f"    use_ridge_stacking   = {cfg.use_ridge_stacking}", flush=True)
    print(f"    use_bias_correction  = {cfg.use_bias_correction}  (E-3: OFF)", flush=True)
    print(f"    use_d49t_geo_features= {cfg.use_d49t_geo_features}  (E-1: ON)", flush=True)
    print(f"    seeds                = {cfg.seeds}  (E-2: multi-seed)", flush=True)
    print(f"    LGB params           = nl={cfg.lgb_params['num_leaves']} "
          f"mcs={cfg.lgb_params['min_child_samples']} lr={cfg.lgb_params['learning_rate']}", flush=True)

    train_df, test_df = load_data()
    d48 = train_df[train_df["day"] == 48].reset_index(drop=True)
    d49t = train_df[train_df["day"] == 49].reset_index(drop=True)
    test = test_df.copy().reset_index(drop=True)
    print(f"\n  d48: {d48.shape}    d49t: {d49t.shape}    test: {test.shape}", flush=True)

    FEATS = get_feature_list(cfg)
    print(f"  Features ({len(FEATS)}): {FEATS}", flush=True)

    # Step 1: Honest eval Mode A and C
    print(f"\n{'#' * 70}\n# HONEST EVAL — Phase E\n{'#' * 70}", flush=True)
    mode_a = run_mode_a(cfg, d48, d49t, test, FEATS)
    mode_c = run_mode_c(cfg, d48, d49t, test, FEATS)

    # Step 2: Final training (multi-seed) → submission_v2.csv
    sub, final_oof_r2, final_weights, imp = run_final_multi_seed(
        cfg, d48, d49t, test, FEATS)

    # Step 3: Verify submission
    print(f"\n{'=' * 70}\n# SUBMISSION V2 VERIFICATION\n{'=' * 70}", flush=True)
    assert sub.shape == (41778, 2), f"shape {sub.shape}"
    assert sub["Index"].is_monotonic_increasing, "Index not monotonic"
    assert sub["demand"].notna().all(), "NaN in demand"
    assert sub["demand"].between(0, 1).all(), "demand outside [0, 1]"
    assert (sub["Index"].values == test["Index"].values).all(), "Index mismatch with test"
    print(f"  ✓ Shape: {sub.shape}", flush=True)
    print(f"  ✓ Index matches test['Index'] exactly", flush=True)
    print(f"  ✓ No NaN", flush=True)
    print(f"  ✓ All demand ∈ [0, 1]", flush=True)
    print(f"  ✓ demand stats: mean={sub['demand'].mean():.4f}, "
          f"std={sub['demand'].std():.4f}, "
          f"min={sub['demand'].min():.4f}, max={sub['demand'].max():.4f}", flush=True)

    # Step 4: Compare to V1
    v1 = pd.read_csv(ROOT / "submission_v1.csv")
    print(f"\n  V1 (current 89.97) stats: mean={v1['demand'].mean():.4f}, "
          f"std={v1['demand'].std():.4f}, min={v1['demand'].min():.4f}, "
          f"max={v1['demand'].max():.4f}", flush=True)
    corr = sub["demand"].corr(v1["demand"])
    mean_diff = (sub["demand"] - v1["demand"]).mean()
    abs_diff = (sub["demand"] - v1["demand"]).abs().mean()
    print(f"  V1 ↔ V2: corr={corr:.4f}, mean diff={mean_diff:+.4f}, "
          f"mean abs diff={abs_diff:.4f}", flush=True)

    # Save metrics
    summary = {
        "config": {
            "use_interactions": cfg.use_interactions,
            "use_d49t_forward_lags": cfg.use_d49t_forward_lags,
            "use_catboost": cfg.use_catboost,
            "use_ridge_stacking": cfg.use_ridge_stacking,
            "use_bias_correction": cfg.use_bias_correction,
            "use_d49t_geo_features": cfg.use_d49t_geo_features,
            "seeds": list(cfg.seeds),
        },
        "mode_a": mode_a,
        "mode_c": mode_c,
        "final_oof_r2": float(final_oof_r2),
        "final_blend_weights": final_weights,
        "submission_shape": list(sub.shape),
        "submission_demand_mean": float(sub["demand"].mean()),
        "submission_demand_std": float(sub["demand"].std()),
        "v1_v2_corr": float(corr),
        "v1_v2_mean_diff": float(mean_diff),
        "v1_v2_mean_abs_diff": float(abs_diff),
    }
    with open(EDA / "phase_e_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 70}", flush=True)
    print(f"  PHASE E SUMMARY", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"  Mode A: blend R²={mode_a['blend_r2']:.4f}  "
          f"(Score={mode_a['blend_r2']*100:.2f})", flush=True)
    print(f"  Mode C: blend R²={mode_c['blend_r2']:.4f}  "
          f"(Score={mode_c['blend_r2']*100:.2f})", flush=True)
    print(f"  Full multi-seed OOF R²: {final_oof_r2:.4f}  "
          f"(Score={final_oof_r2*100:.2f})", flush=True)
    print(f"  Baseline (V0) online: 87.26  |  V1 online: 89.97  |  V2 online: TBD", flush=True)
    print(f"\n  Total time: {time.time()-t_total:.1f}s", flush=True)
    print(f"  → /home/marcus/code/Gridlock/submission_v2.csv", flush=True)
    print(f"  → /home/marcus/code/Gridlock/eda/phase_e_metrics.json", flush=True)
    print(f"  → /home/marcus/code/Gridlock/eda/feature_importance_v2.csv", flush=True)


if __name__ == "__main__":
    main()
