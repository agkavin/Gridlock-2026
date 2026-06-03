"""
Test the impact of CatBoost (C-5) and Ridge stacking (C-5) on Mode A.

CatBoost handles geohash as a native categorical, which is fundamentally
different from label-encoded. Combined with Ridge stacking, this should
add real model diversity.

Also tests Tweedie/quantile (C-7) as auxiliary models.
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
    Config, assemble_features, evaluate_mode, load_data,
    train_lgb, train_xgb, fit_oof_kfold, get_feature_list,
)

warnings.filterwarnings("ignore")

ROOT = Path("/home/marcus/code/Gridlock")
OUTPUT = ROOT / "output"


def make_cfg(**overrides):
    kw = dict(
        use_target_encoding=False,
        use_interactions=False,
        use_d49t_forward_lags=False,
        use_catboost=False,
        use_ridge_stacking=False,
        use_tweedie=False,
        use_quantile=False,
        use_bias_correction=False,
    )
    kw.update(overrides)
    return Config(**kw)


def train_catboost(X_tr, y_tr, X_va, y_va, params):
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(**params)
    m.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    return m.predict(X_va), m


def fit_oof_kfold_cb(X, y, kf, params, cat_features=None):
    oof = np.zeros(len(X))
    for fold, (ti, vi) in enumerate(kf.split(X), 1):
        X_tr, X_va = X.iloc[ti], X.iloc[vi]
        y_tr, y_va = y.iloc[ti], y.iloc[vi]
        m = None
        if cat_features:
            from catboost import CatBoostRegressor
            m = CatBoostRegressor(**params)
            m.fit(X_tr, y_tr, eval_set=(X_va, y_va),
                  cat_features=cat_features, verbose=False)
        else:
            _, m = train_catboost(X_tr, y_tr, X_va, y_va, params)
        oof[vi] = m.predict(X_va)
    return oof


def main():
    train_df, test_df = load_data()
    d48 = train_df[train_df["day"] == 48].reset_index(drop=True)
    d49t = train_df[train_df["day"] == 49].reset_index(drop=True)
    test = test_df.copy().reset_index(drop=True)

    print(f"d48: {d48.shape}    d49t: {d49t.shape}")

    cfg = make_cfg()
    d48f, d49f, _, _, _, _ = assemble_features(d48, d49t, test, cfg, mode="A")
    FEATS = get_feature_list(cfg)
    X = d48f[FEATS].astype(float).fillna(-999)
    y = d48f["demand"].astype(float)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Run base models
    print("\n" + "=" * 70)
    print("  BASE MODELS (OOF on d48)")
    print("=" * 70)
    lgb_oof, lgb_iters = fit_oof_kfold(X, y, kf, train_lgb, cfg.lgb_params)
    xgb_oof, xgb_iters = fit_oof_kfold(X, y, kf, train_xgb, cfg.xgb_params)
    lgb_r2 = r2_score(y, lgb_oof)
    xgb_r2 = r2_score(y, xgb_oof)
    print(f"  LGB R²  : {lgb_r2:.4f}  (mean best_iter={np.mean(lgb_iters):.0f})")
    print(f"  XGB R²  : {xgb_r2:.4f}  (mean best_iter={np.mean(xgb_iters):.0f})")

    # CatBoost — standard (numeric features only, like LGB/XGB)
    print("\nRunning CatBoost (numeric features only)...")
    cb_params = cfg.cb_params
    cb_oof_num = fit_oof_kfold_cb(X, y, kf, cb_params)
    cb_r2_num = r2_score(y, cb_oof_num)
    print(f"  CB  R²  : {cb_r2_num:.4f}  (numeric features)")

    # CatBoost — with geohash as native categorical
    print("\nRunning CatBoost (with geohash as native categorical)...")
    X_cb = X.copy()
    X_cb["geohash"] = d48f["geohash"].values
    X_cb["RoadType"] = d48f["RoadType"].fillna("NaN").values
    X_cb["Weather"] = d48f["Weather"].fillna("NaN").values
    X_cb["LargeVehicles"] = d48f["LargeVehicles"].values
    X_cb["Landmarks"] = d48f["Landmarks"].values
    cat_feats = ["geohash", "RoadType", "Weather", "LargeVehicles", "Landmarks"]
    cb_oof_cat = fit_oof_kfold_cb(X_cb, y, kf, cb_params, cat_features=cat_feats)
    cb_r2_cat = r2_score(y, cb_oof_cat)
    print(f"  CB  R²  : {cb_r2_cat:.4f}  (with native categoricals)")

    # Tweedie
    print("\nRunning Tweedie LGB...")
    tw_params = dict(cfg.lgb_params)
    tw_params["objective"] = "tweedie"
    tw_params["tweedie_variance_power"] = 1.5
    tw_oof, _ = fit_oof_kfold(X, y, kf, train_lgb, tw_params)
    tw_r2 = r2_score(y, tw_oof)
    print(f"  Tweedie R²  : {tw_r2:.4f}")

    # Quantile (median)
    print("\nRunning Quantile LGB (alpha=0.5)...")
    qt_params = dict(cfg.lgb_params)
    qt_params["objective"] = "quantile"
    qt_params["alpha"] = 0.5
    qt_oof, _ = fit_oof_kfold(X, y, kf, train_lgb, qt_params)
    qt_r2 = r2_score(y, qt_oof)
    print(f"  Quantile R² : {qt_r2:.4f}")

    # Stacking experiments
    print("\n" + "=" * 70)
    print("  STACKING EXPERIMENTS")
    print("=" * 70)
    # Simple 50/50
    s50 = r2_score(y, 0.5 * lgb_oof + 0.5 * xgb_oof)
    print(f"  50/50 LGB+XGB R²  : {s50:.4f}")
    # Hand-tuned best blend
    bw, br = -1, -1
    for w in np.arange(0.0, 1.01, 0.02):
        r = r2_score(y, w * lgb_oof + (1 - w) * xgb_oof)
        if r > br:
            br, bw = r, w
    print(f"  Best blend (LGB={bw:.2f}) R²  : {br:.4f}")

    # 3-way: LGB + XGB + CatBoost
    best3, bw3 = -1, None
    for w_l in np.arange(0.0, 1.01, 0.05):
        for w_x in np.arange(0.0, 1.01 - w_l, 0.05):
            w_c = 1 - w_l - w_x
            if w_c < 0:
                continue
            r = r2_score(y, w_l * lgb_oof + w_x * xgb_oof + w_c * cb_oof_num)
            if r > best3:
                best3, bw3 = r, (w_l, w_x, w_c)
    print(f"  Best 3-way (LGB+XGB+CB-num)  R²  : {best3:.4f}  weights={tuple(round(x,2) for x in bw3)}")

    # 3-way with native cat
    best3c, bw3c = -1, None
    for w_l in np.arange(0.0, 1.01, 0.05):
        for w_x in np.arange(0.0, 1.01 - w_l, 0.05):
            w_c = 1 - w_l - w_x
            if w_c < 0:
                continue
            r = r2_score(y, w_l * lgb_oof + w_x * xgb_oof + w_c * cb_oof_cat)
            if r > best3c:
                best3c, bw3c = r, (w_l, w_x, w_c)
    print(f"  Best 3-way (LGB+XGB+CB-cat)  R²  : {best3c:.4f}  weights={tuple(round(x,2) for x in bw3c)}")

    # Ridge stacking (positive, no intercept) over LGB + XGB + CB-num
    cols = [lgb_oof, xgb_oof, cb_oof_num]
    Xstack = np.column_stack(cols)
    for alpha in [0.01, 0.1, 1.0, 10.0]:
        blend = Ridge(alpha=alpha, positive=True, fit_intercept=False).fit(Xstack, y)
        oof_blend = blend.predict(Xstack)
        r = r2_score(y, oof_blend)
        print(f"  Ridge α={alpha:>5} (LGB+XGB+CB-num) R²  : {r:.4f}  weights={blend.coef_.round(3)}")

    # Ridge with native cat
    cols_c = [lgb_oof, xgb_oof, cb_oof_cat]
    Xstack_c = np.column_stack(cols_c)
    blend_c = Ridge(alpha=1.0, positive=True, fit_intercept=False).fit(Xstack_c, y)
    oof_blend_c = blend_c.predict(Xstack_c)
    r_c = r2_score(y, oof_blend_c)
    print(f"  Ridge α=1.0  (LGB+XGB+CB-cat) R²  : {r_c:.4f}  weights={blend_c.coef_.round(3)}")

    # Ridge with native cat + tweedie + quantile
    cols_all = [lgb_oof, xgb_oof, cb_oof_cat, tw_oof, qt_oof]
    Xstack_all = np.column_stack(cols_all)
    blend_all = Ridge(alpha=1.0, positive=True, fit_intercept=False).fit(Xstack_all, y)
    oof_all = blend_all.predict(Xstack_all)
    r_all = r2_score(y, oof_all)
    print(f"  Ridge α=1.0  (LGB+XGB+CB-cat+Tw+Qt) R²  : {r_all:.4f}  weights={blend_all.coef_.round(3)}")

    # Save
    out = OUTPUT / "stacking_results.json"
    results = {
        "lgb_r2": lgb_r2, "xgb_r2": xgb_r2,
        "cb_num_r2": cb_r2_num, "cb_cat_r2": cb_r2_cat,
        "tweedie_r2": tw_r2, "quantile_r2": qt_r2,
        "blend_50_50_r2": s50, "blend_best_lgb_xgb_r2": br,
        "blend_3way_num_r2": best3, "blend_3way_num_w": list(bw3),
        "blend_3way_cat_r2": best3c, "blend_3way_cat_w": list(bw3c),
        "ridge_cat_r2": r_c, "ridge_cat_w": blend_c.coef_.tolist(),
        "ridge_all_r2": r_all, "ridge_all_w": blend_all.coef_.tolist(),
    }
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out.relative_to(ROOT)}")

    # Summary
    print("\n" + "=" * 70)
    print("  STACKING SUMMARY (sorted by R²)")
    print("=" * 70)
    all_results = [
        ("LGB only", lgb_r2),
        ("XGB only", xgb_r2),
        ("CB numeric", cb_r2_num),
        ("CB native cat", cb_r2_cat),
        ("Tweedie LGB", tw_r2),
        ("Quantile LGB", qt_r2),
        ("LGB+XGB best blend", br),
        ("3-way LGB+XGB+CB-num best", best3),
        ("3-way LGB+XGB+CB-cat best", best3c),
        ("Ridge LGB+XGB+CB-cat", r_c),
        ("Ridge LGB+XGB+CB-cat+Tw+Qt", r_all),
    ]
    for name, r in sorted(all_results, key=lambda x: -x[1]):
        marker = "★" if r >= 0.966 else " "
        print(f"  {marker} {name:<40}  R²={r:.4f}  Score={r*100:.2f}")


if __name__ == "__main__":
    main()
