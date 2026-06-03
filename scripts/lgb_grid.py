"""
C-6 — FAST hyperparam grid (8 cells, with progressive screening).

Strategy to stay under ~15 min:
  1. First, 1-fold validation on each of 8 cells (fast: ~30s each = 4 min).
  2. Pick top 3, run 5-fold OOF on each (~2-3 min each = 9 min).
  3. Pick the winner.

Output is flushed after every cell so progress is visible.
"""

import json
import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

import lightgbm as lgb

sys.path.insert(0, str(Path("/home/marcus/code/Gridlock")))
from improve_demand import (
    Config, assemble_features, load_data, get_feature_list,
)

warnings.filterwarnings("ignore")

ROOT = Path("/home/marcus/code/Gridlock")
OUTPUT = ROOT / "output"


def make_cfg(**overrides):
    kw = dict(
        use_target_encoding=False, use_interactions=False,
        use_d49t_forward_lags=False, use_catboost=False,
        use_ridge_stacking=False, use_tweedie=False, use_quantile=False,
        use_bias_correction=False,
    )
    kw.update(overrides)
    return Config(**kw)


def fit_oof_lgb(X, y, kf, params, num_round=2000, early_stop=100, verbose=False):
    oof = np.zeros(len(X))
    iters = []
    for fi, (ti, vi) in enumerate(kf.split(X), 1):
        X_tr, X_va = X.iloc[ti], X.iloc[vi]
        y_tr, y_va = y.iloc[ti], y.iloc[vi]
        dt = lgb.Dataset(X_tr, y_tr)
        dv = lgb.Dataset(X_va, y_va, reference=dt)
        m = lgb.train(
            params, dt, num_boost_round=num_round, valid_sets=[dv],
            callbacks=[lgb.early_stopping(early_stop, verbose=False)],
        )
        oof[vi] = m.predict(X_va, num_iteration=m.best_iteration)
        iters.append(m.best_iteration)
        if verbose:
            print(f"      fold {fi}: best_iter={m.best_iteration}", flush=True)
    return oof, iters


def main():
    train_df, test_df = load_data()
    d48 = train_df[train_df["day"] == 48].reset_index(drop=True)
    d49t = train_df[train_df["day"] == 49].reset_index(drop=True)
    test = test_df.copy().reset_index(drop=True)
    print(f"d48: {d48.shape}    d49t: {d49t.shape}", flush=True)

    cfg = make_cfg()
    d48f, _, _, _, _, _ = assemble_features(d48, d49t, test, cfg, mode="A")
    FEATS = get_feature_list(cfg)
    X = d48f[FEATS].astype(float).fillna(-999)
    y = d48f["demand"].astype(float)

    base_params = cfg.lgb_params

    # Smaller grid: 8 cells, focused on most-impactful params
    grid = {
        "num_leaves": [255, 511],
        "min_child_samples": [3, 5],
        "learning_rate": [0.02, 0.05],
    }
    cells = list(product(grid["num_leaves"], grid["min_child_samples"], grid["learning_rate"]))
    print(f"\nScreening {len(cells)} configs with 1-fold validation (~4 min total)", flush=True)
    print("=" * 70, flush=True)

    kf1 = KFold(n_splits=5, shuffle=True, random_state=42)
    # 1-fold = train on fold-0 of a different split (4 folds used as train, 1 as val)
    # Quicker approach: just use a single train/val split
    from sklearn.model_selection import train_test_split
    idx_train, idx_val = train_test_split(np.arange(len(X)), test_size=0.2, random_state=42)

    screen_results = []
    for nl, mcs, lr in cells:
        params = dict(base_params)
        params["num_leaves"] = nl
        params["min_child_samples"] = mcs
        params["learning_rate"] = lr
        dt = lgb.Dataset(X.iloc[idx_train], y.iloc[idx_train])
        dv = lgb.Dataset(X.iloc[idx_val], y.iloc[idx_val], reference=dt)
        m = lgb.train(
            params, dt, num_boost_round=1500, valid_sets=[dv],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        pred = m.predict(X.iloc[idx_val], num_iteration=m.best_iteration)
        r2 = r2_score(y.iloc[idx_val], pred)
        screen_results.append({
            "num_leaves": nl, "min_child_samples": mcs,
            "learning_rate": lr, "screen_r2": float(r2),
            "best_iter": int(m.best_iteration),
        })
        print(f"  nl={nl:>3}  mcs={mcs:>2}  lr={lr:.2f}  R²={r2:.5f}  iter={m.best_iteration}",
              flush=True)

    # Pick top 3
    screen_results.sort(key=lambda r: -r["screen_r2"])
    top3 = screen_results[:3]
    print(f"\nTop 3 after screening:", flush=True)
    for r in top3:
        print(f"  nl={r['num_leaves']:>3}  mcs={r['min_child_samples']:>2}  lr={r['learning_rate']:.2f}  "
              f"screen_R²={r['screen_r2']:.5f}", flush=True)

    # 5-fold OOF on top 3
    print(f"\nRunning 5-fold OOF on top 3 configs (~9 min)...", flush=True)
    kf5 = KFold(n_splits=5, shuffle=True, random_state=42)
    final_results = []
    for r in top3:
        params = dict(base_params)
        params["num_leaves"] = r["num_leaves"]
        params["min_child_samples"] = r["min_child_samples"]
        params["learning_rate"] = r["learning_rate"]
        oof, iters = fit_oof_lgb(X, y, kf5, params, num_round=2000, early_stop=150)
        r2 = r2_score(y, oof)
        r["oof_r2"] = float(r2)
        r["mean_iter"] = int(np.mean(iters))
        final_results.append(r)
        print(f"  nl={r['num_leaves']:>3}  mcs={r['min_child_samples']:>2}  lr={r['learning_rate']:.2f}  "
              f"5-fold R²={r2:.5f}  mean_iter={r['mean_iter']}", flush=True)

    final_results.sort(key=lambda r: -r["oof_r2"])
    best = final_results[0]
    print("\n" + "=" * 70, flush=True)
    print(f"  BEST LGB CONFIG:  R²={best['oof_r2']:.5f}", flush=True)
    print(f"  num_leaves={best['num_leaves']}, min_child_samples={best['min_child_samples']}, "
          f"learning_rate={best['learning_rate']}", flush=True)
    print(f"  Baseline LGB params: R²=0.9652  (Δ={best['oof_r2'] - 0.9652:+.5f})", flush=True)
    print("=" * 70, flush=True)

    # Save
    out_csv = OUTPUT / "lgb_grid_results.csv"
    pd.DataFrame(screen_results).to_csv(out_csv, index=False)
    out_json = OUTPUT / "lgb_grid_best.json"
    with open(out_json, "w") as f:
        json.dump({"best": best, "all": screen_results, "final_top3": final_results}, f, indent=2)
    print(f"\nSaved: {out_csv.relative_to(ROOT)} and {out_json.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
