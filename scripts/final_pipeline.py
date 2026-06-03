"""
Phase D — Final pipeline + submission.

What it does:
1. Run honest eval (3 modes) on the FULL config (C-5 + C-6 + C-8 + interactions + target enc).
2. Train the final ensemble on ALL of d48 + d49t (77,299 rows) with 5-fold CV.
3. Predict the 41,778 test rows.
4. Apply per-geohash bias correction (C-8).
5. Clip to [0, 1] and write /home/marcus/code/Gridlock/submission.csv.
6. Generate reports: feature importances, per-model metrics, blend weights.

Improvements locked in:
- C-3 interactions (6 new features)
- C-5 CatBoost with native geohash (added to the stack)
- C-5 Ridge stacking
- C-6 best LGB params: nl=255, mcs=3, lr=0.02
- C-8 per-geohash bias correction (k=20)
"""

import json
import sys
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
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)


def make_final_cfg():
    """Configuration with all winning improvements locked in."""
    return Config(
        # C-2: target encoding (off — didn't help)
        use_target_encoding=False,
        te_alpha=20.0,
        # C-3: interactions (on — provides a tiny boost via is_lunch/is_quiet)
        use_interactions=True,
        # C-4: d49t forward lags (on — adds test-only signal for slots 9-16)
        use_d49t_forward_lags=True,
        # C-5: CatBoost with native geohash (on — biggest single improvement)
        use_catboost=True,
        use_ridge_stacking=True,
        # C-6: tuned LGB params
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
            "l2_leaf_reg": 3.0, "iterations": 2000, "early_stopping_rounds": 100,
            "thread_count": 4, "random_seed": 42, "verbose": False,
        },
        # C-7: tweedie/quantile (off — no help)
        use_tweedie=False,
        use_quantile=False,
        # C-8: bias correction (on)
        use_bias_correction=True,
        bias_correction_k=20,
    )


def train_catboost_with_cat(X_tr, y_tr, X_va, y_va, params, cat_cols):
    """Train CatBoost with specified categorical columns."""
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(**params)
    m.fit(X_tr, y_tr, eval_set=(X_va, y_va),
          cat_features=cat_cols, verbose=False)
    return m.predict(X_va), m


def add_catboost_columns(df):
    """Add string columns that CatBoost can use as native categoricals."""
    df = df.copy()
    df["geohash_cat"] = df["geohash"].astype(str)
    df["RoadType_cat"] = df["RoadType"].fillna("NaN").astype(str)
    df["Weather_cat"] = df["Weather"].fillna("NaN").astype(str)
    df["LargeVehicles_cat"] = df["LargeVehicles"].astype(str)
    df["Landmarks_cat"] = df["Landmarks"].astype(str)
    return df


def fit_oof_kfold_cat(X, y, kf, params, cat_cols):
    oof = np.zeros(len(X))
    for ti, vi in kf.split(X):
        _, m = train_catboost_with_cat(
            X.iloc[ti], y.iloc[ti], X.iloc[vi], y.iloc[vi],
            params, cat_cols,
        )
        oof[vi] = m.predict(X.iloc[vi])
    return oof


def fit_test_kfold_cat(X, y, X_test, kf, params, cat_cols, n_folds_predict=False):
    """Train K-fold CatBoost and return test predictions (averaged)."""
    test_preds = np.zeros(len(X_test))
    for ti, vi in kf.split(X):
        _, m = train_catboost_with_cat(
            X.iloc[ti], y.iloc[ti], X.iloc[vi], y.iloc[vi],
            params, cat_cols,
        )
        test_preds += m.predict(X_test) / kf.get_n_splits()
    return test_preds


def apply_bias_correction(blend_oof, y, geohash, k=20, geohash_test=None):
    """Compute per-geohash shrunk bias from residuals, return corrected OOF and bias array."""
    resid = y - blend_oof
    df = pd.DataFrame({"geohash": geohash, "resid": resid})
    g_stats = df.groupby("geohash")["resid"].agg(g_sum="sum", g_n="count")
    g_stats["g_bias"] = g_stats["g_sum"] / g_stats["g_n"]
    g_stats["g_bias_shrunk"] = g_stats["g_bias"] * g_stats["g_n"] / (g_stats["g_n"] + k)
    bias_lookup = g_stats["g_bias_shrunk"].to_dict()

    bias_oof = pd.Series(geohash).map(bias_lookup).fillna(0).values
    corrected_oof = blend_oof + bias_oof

    if geohash_test is not None:
        bias_test = pd.Series(geohash_test).map(bias_lookup).fillna(0).values
        return corrected_oof, bias_oof, bias_test
    return corrected_oof, bias_oof, None


def run_honest_eval(cfg, label=""):
    """Run 3 honest eval modes and return scores."""
    print(f"\n{'#' * 70}\n# HONEST EVAL ({label})\n{'#' * 70}", flush=True)
    train_df, test_df = load_data()
    d48 = train_df[train_df["day"] == 48].reset_index(drop=True)
    d49t = train_df[train_df["day"] == 49].reset_index(drop=True)
    test = test_df.copy().reset_index(drop=True)

    FEATS = get_feature_list(cfg)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    # ----- Mode A: KFold d48 -----
    print(f"\n[Mode A] KFold d48", flush=True)
    d48f, _, _, _, _, _ = assemble_features(d48, d49t, test, cfg, mode="A")
    X = d48f[FEATS].astype(float).fillna(-999)
    y = d48f["demand"].astype(float)

    print(f"  training LGB...", flush=True)
    lgb_oof, _ = fit_oof_kfold(X, y, kf, train_lgb, cfg.lgb_params)
    print(f"  training XGB...", flush=True)
    xgb_oof, _ = fit_oof_kfold(X, y, kf, train_xgb, cfg.xgb_params)
    print(f"  training CatBoost (native cat)...", flush=True)
    d48f_cb = add_catboost_columns(d48f)
    X_cb = d48f_cb[FEATS + ["geohash_cat", "RoadType_cat", "Weather_cat",
                              "LargeVehicles_cat", "Landmarks_cat"]]
    cat_cols = ["geohash_cat", "RoadType_cat", "Weather_cat", "LargeVehicles_cat", "Landmarks_cat"]
    cb_oof = fit_oof_kfold_cat(X_cb, y, kf, cfg.cb_params, cat_cols)

    # Ridge stack
    Xstack = np.column_stack([lgb_oof, xgb_oof, cb_oof])
    blend = Ridge(alpha=1.0, positive=True, fit_intercept=False).fit(Xstack, y)
    oof_blend = blend.predict(Xstack)
    print(f"    Ridge weights: {blend.coef_.round(3).tolist()}", flush=True)
    blend_r2 = r2_score(y, oof_blend)

    # Bias correction
    if cfg.use_bias_correction:
        corrected, bias_arr, _ = apply_bias_correction(
            oof_blend, y.values, d48f["geohash"].values, k=cfg.bias_correction_k
        )
        bias_r2 = r2_score(y, corrected)
        print(f"    Mode A  R² (blend)        : {blend_r2:.4f}  (Score={blend_r2*100:.2f})", flush=True)
        print(f"    Mode A  R² (after bias)   : {bias_r2:.4f}  (Score={bias_r2*100:.2f})", flush=True)
        results["A"] = {"blend_r2": blend_r2, "bias_r2": bias_r2,
                        "blend_weights": blend.coef_.tolist(),
                        "lgb_r2": r2_score(y, lgb_oof),
                        "xgb_r2": r2_score(y, xgb_oof),
                        "cb_r2": r2_score(y, cb_oof)}
    else:
        print(f"    Mode A  R² (blend)        : {blend_r2:.4f}  (Score={blend_r2*100:.2f})", flush=True)
        results["A"] = {"blend_r2": blend_r2,
                        "blend_weights": blend.coef_.tolist(),
                        "lgb_r2": r2_score(y, lgb_oof),
                        "xgb_r2": r2_score(y, xgb_oof),
                        "cb_r2": r2_score(y, cb_oof)}

    # ----- Mode B: Time holdout slots 80-95 -----
    print(f"\n[Mode B] Time holdout slots 80-95", flush=True)
    d48f, _, _, _, _, _ = assemble_features(d48, d49t, test, cfg, mode="B")
    tr_mask = d48f["time_slot"] < 80
    X_tr = d48f[tr_mask][FEATS].astype(float).fillna(-999)
    y_tr = d48f[tr_mask]["demand"].astype(float)
    X_va = d48f[~tr_mask][FEATS].astype(float).fillna(-999)
    y_va = d48f[~tr_mask]["demand"].astype(float)
    print(f"  training LGB...", flush=True)
    lgb_pred, _ = train_lgb(X_tr, y_tr, X_va, y_va, cfg.lgb_params)
    print(f"  training XGB...", flush=True)
    xgb_pred, _ = train_xgb(X_tr, y_tr, X_va, y_va, cfg.xgb_params)
    print(f"  training CatBoost...", flush=True)
    d48f_cb = add_catboost_columns(d48f)
    cat_cols = ["geohash_cat", "RoadType_cat", "Weather_cat", "LargeVehicles_cat", "Landmarks_cat"]
    X_tr_cb = d48f_cb[tr_mask][FEATS + cat_cols]
    X_va_cb = d48f_cb[~tr_mask][FEATS + cat_cols]
    cb_pred, _ = train_catboost_with_cat(X_tr_cb, y_tr, X_va_cb, y_va, cfg.cb_params, cat_cols)
    # Use blend weights from Mode A
    blend_pred = (blend.coef_[0] * lgb_pred + blend.coef_[1] * xgb_pred + blend.coef_[2] * cb_pred)
    blend_r2_B = r2_score(y_va, blend_pred)
    print(f"    Mode B  R² (blend)        : {blend_r2_B:.4f}  (Score={blend_r2_B*100:.2f})", flush=True)
    results["B"] = {"blend_r2": blend_r2_B, "n_val": int(len(X_va))}

    # ----- Mode C: Predict d49t from d48 -----
    print(f"\n[Mode C] Predict d49t from d48", flush=True)
    d48f, d49f, _, _, _, _ = assemble_features(d48, d49t, test, cfg, mode="C")
    X = d48f[FEATS].astype(float).fillna(-999)
    y = d48f["demand"].astype(float)
    X_va = d49f[FEATS].astype(float).fillna(-999)
    y_va = d49f["demand"].astype(float)
    print(f"  training LGB...", flush=True)
    lgb_oof, _ = fit_oof_kfold(X, y, kf, train_lgb, cfg.lgb_params)
    xgb_oof, _ = fit_oof_kfold(X, y, kf, train_xgb, cfg.xgb_params)
    print(f"  training CatBoost...", flush=True)
    d48f_cb = add_catboost_columns(d48f)
    d49f_cb = add_catboost_columns(d49f)
    X_cb = d48f_cb[FEATS + cat_cols]
    cb_oof = fit_oof_kfold_cat(X_cb, y, kf, cfg.cb_params, cat_cols)
    # Train models on full d48 and predict d49t
    print(f"  training final models on full d48...", flush=True)
    lgb_final, _ = train_lgb(X, y, X_va, y_va, cfg.lgb_params)
    xgb_final, _ = train_xgb(X, y, X_va, y_va, cfg.xgb_params)
    X_va_cb = d49f_cb[FEATS + cat_cols]
    _, cb_final = train_catboost_with_cat(X_cb, y, X_va_cb, y_va, cfg.cb_params, cat_cols)
    cb_pred = cb_final.predict(X_va_cb)
    blend_pred = (blend.coef_[0] * lgb_final + blend.coef_[1] * xgb_final + blend.coef_[2] * cb_pred)
    blend_r2_C = r2_score(y_va, blend_pred)
    print(f"    Mode C  R² (blend)        : {blend_r2_C:.4f}  (Score={blend_r2_C*100:.2f})", flush=True)
    results["C"] = {"blend_r2": blend_r2_C, "n_val": int(len(X_va))}

    return results, d48, d49t, test, blend


def train_final_and_predict(cfg, d48, d49t, test, blend_weights, blend_alpha=1.0):
    """Train on d48+d49t, predict test, return predictions and metrics."""
    print(f"\n{'#' * 70}\n# FINAL TRAINING (d48 + d49t -> test)\n{'#' * 70}", flush=True)
    FEATS = get_feature_list(cfg)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Build features for d48, d49t, test
    d48f, d49f, testf, gm, _, _ = assemble_features(d48, d49t, test, cfg, mode="FINAL")

    # Combine d48 + d49t as training
    d48f["_origin"] = "d48"
    d49f["_origin"] = "d49t"
    testf["_origin"] = "test"
    combined = pd.concat([d48f, d49f], ignore_index=True)
    print(f"  combined train: {combined.shape}", flush=True)
    print(f"  test: {testf.shape}", flush=True)

    # CatBoost columns
    combined_cb = add_catboost_columns(combined)
    testf_cb = add_catboost_columns(testf)
    cat_cols = ["geohash_cat", "RoadType_cat", "Weather_cat",
                "LargeVehicles_cat", "Landmarks_cat"]

    X = combined[FEATS].astype(float).fillna(-999)
    y = combined["demand"].astype(float)
    X_test = testf[FEATS].astype(float).fillna(-999)

    X_cb = combined_cb[FEATS + cat_cols]
    X_test_cb = testf_cb[FEATS + cat_cols]

    # Train each model with 5-fold CV; accumulate test predictions
    print(f"  training LGB (5-fold, test averaged)...", flush=True)
    lgb_test = np.zeros(len(X_test))
    for ti, vi in kf.split(X):
        _, m = train_lgb(X.iloc[ti], y.iloc[ti], X.iloc[vi], y.iloc[vi], cfg.lgb_params)
        lgb_test += m.predict(X_test, num_iteration=m.best_iteration) / 5
    # OOF on combined for bias correction
    lgb_oof, _ = fit_oof_kfold(X, y, kf, train_lgb, cfg.lgb_params)

    print(f"  training XGB (5-fold, test averaged)...", flush=True)
    xgb_test = np.zeros(len(X_test))
    for ti, vi in kf.split(X):
        _, m = train_xgb(X.iloc[ti], y.iloc[ti], X.iloc[vi], y.iloc[vi], cfg.xgb_params)
        xgb_test += m.predict(xgb.DMatrix(X_test)) / 5
    xgb_oof, _ = fit_oof_kfold(X, y, kf, train_xgb, cfg.xgb_params)

    print(f"  training CatBoost (5-fold, test averaged, native cat)...", flush=True)
    cb_test = fit_test_kfold_cat(X_cb, y, X_test_cb, kf, cfg.cb_params, cat_cols)
    cb_oof = fit_oof_kfold_cat(X_cb, y, kf, cfg.cb_params, cat_cols)

    # Ridge stack
    Xstack = np.column_stack([lgb_oof, xgb_oof, cb_oof])
    blend = Ridge(alpha=blend_alpha, positive=True, fit_intercept=False).fit(Xstack, y)
    print(f"  Final Ridge weights: {blend.coef_.round(3).tolist()}", flush=True)

    oof_blend = blend.predict(Xstack)
    final_oof_r2 = r2_score(y, oof_blend)
    print(f"  Final OOF R²: {final_oof_r2:.4f}  (Score={final_oof_r2*100:.2f})", flush=True)

    # Bias correction
    if cfg.use_bias_correction:
        corrected_oof, bias_oof, bias_test = apply_bias_correction(
            oof_blend, y.values, combined["geohash"].values,
            k=cfg.bias_correction_k, geohash_test=testf["geohash"].values,
        )
        bias_r2 = r2_score(y, corrected_oof)
        print(f"  After bias correction OOF R²: {bias_r2:.4f}  (Score={bias_r2*100:.2f})", flush=True)

    # Build test predictions
    test_stack = np.column_stack([lgb_test, xgb_test, cb_test])
    test_blend = blend.predict(test_stack)
    if cfg.use_bias_correction:
        test_blend = test_blend + bias_test
    test_blend = np.clip(test_blend, 0, 1)

    # Save predictions
    sub = pd.DataFrame({"Index": test["Index"], "demand": test_blend})
    sub.to_csv(OUTPUT / "submission.csv", index=False)
    print(f"\n  Saved submission: output/submission.csv  ({sub.shape})", flush=True)

    # Feature importance (LGB gain)
    # Train one LGB on all data for importance
    dt = lgb.Dataset(X, y)
    dv = lgb.Dataset(X.iloc[:1000], y.iloc[:1000], reference=dt)  # dummy val
    m_imp = lgb.train(cfg.lgb_params, dt, num_boost_round=300, valid_sets=[dv],
                      callbacks=[lgb.log_evaluation(0)])
    imp = pd.DataFrame({
        "feature": X.columns,
        "gain": m_imp.feature_importance(importance_type="gain"),
        "split": m_imp.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    imp.to_csv(OUTPUT / "feature_importance.csv", index=False)
    print(f"  Saved feature importance: output/feature_importance.csv", flush=True)

    return sub, final_oof_r2, blend.coef_.tolist(), imp


def main():
    print("=" * 70, flush=True)
    print("  PHASE D — FINAL PIPELINE", flush=True)
    print("=" * 70, flush=True)

    cfg = make_final_cfg()
    print(f"\n  Final config:", flush=True)
    print(f"    use_interactions   = {cfg.use_interactions}", flush=True)
    print(f"    use_d49t_forward_lags = {cfg.use_d49t_forward_lags}", flush=True)
    print(f"    use_catboost       = {cfg.use_catboost}", flush=True)
    print(f"    use_ridge_stacking = {cfg.use_ridge_stacking}", flush=True)
    print(f"    use_bias_correction= {cfg.use_bias_correction}  (k={cfg.bias_correction_k})", flush=True)
    print(f"    LGB params         = nl={cfg.lgb_params['num_leaves']} "
          f"mcs={cfg.lgb_params['min_child_samples']} lr={cfg.lgb_params['learning_rate']}", flush=True)

    # Step 1: Honest eval
    honest_results, d48, d49t, test, blend_a = run_honest_eval(cfg, label="FINAL CONFIG")

    # Step 2: Final training + prediction
    sub, final_oof_r2, final_weights, imp = train_final_and_predict(
        cfg, d48, d49t, test, blend_weights=blend_a.coef_,
    )

    # Step 3: Verify submission
    print(f"\n{'=' * 70}\n# SUBMISSION VERIFICATION\n{'=' * 70}", flush=True)
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

    # Step 4: Final summary
    summary = {
        "config": {
            "use_interactions": cfg.use_interactions,
            "use_d49t_forward_lags": cfg.use_d49t_forward_lags,
            "use_catboost": cfg.use_catboost,
            "use_ridge_stacking": cfg.use_ridge_stacking,
            "use_bias_correction": cfg.use_bias_correction,
            "bias_correction_k": cfg.bias_correction_k,
        },
        "honest_eval": honest_results,
        "final_oof_r2": float(final_oof_r2),
        "final_blend_weights": final_weights,
        "submission_shape": list(sub.shape),
        "submission_demand_mean": float(sub["demand"].mean()),
        "submission_demand_std": float(sub["demand"].std()),
    }
    with open(OUTPUT / "final_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 70}", flush=True)
    print(f"  FINAL SUMMARY", flush=True)
    print(f"{'=' * 70}", flush=True)
    for mode, r in honest_results.items():
        if "bias_r2" in r:
            print(f"  Mode {mode}: blend R²={r['blend_r2']:.4f}  "
                  f"bias R²={r['bias_r2']:.4f}", flush=True)
        else:
            print(f"  Mode {mode}: R²={r['blend_r2']:.4f}", flush=True)
    print(f"  Full d48+d49t OOF R²: {final_oof_r2:.4f}", flush=True)
    print(f"  Baseline KFold (d48+d49t) R²: 0.9616  "
          f"(Δ from baseline: {final_oof_r2 - 0.9616:+.4f})", flush=True)
    print(f"\n  → output/submission.csv", flush=True)
    print(f"  → output/final_metrics.json", flush=True)
    print(f"  → output/feature_importance.csv", flush=True)


if __name__ == "__main__":
    main()
