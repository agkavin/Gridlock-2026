"""
A/B test individual improvements against the honest Mode A baseline.

For each candidate improvement, run Mode A with it on, then with it off,
and report the delta. This isolates the marginal value of each change.

Usage: uv run python scripts/ab_test.py
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path("/home/marcus/code/Gridlock")))
from improve_demand import (
    Config, assemble_features, evaluate_mode, load_data,
)

warnings.filterwarnings("ignore")

ROOT = Path("/home/marcus/code/Gridlock")
OUTPUT = ROOT / "output"


def run_mode_a(cfg):
    train_df, test_df = load_data()
    d48 = train_df[train_df["day"] == 48].reset_index(drop=True)
    d49t = train_df[train_df["day"] == 49].reset_index(drop=True)
    test = test_df.copy().reset_index(drop=True)
    d48f, d49f, _, _, _, _ = assemble_features(d48, d49t, test, cfg, mode="A")
    return evaluate_mode(d48f, d49f, cfg, "A")


# Pre-baked test matrix
# Each test: (name, base_cfg_kwargs, change_kwargs)
BASE_KW = dict(
    use_target_encoding=False,
    use_interactions=False,
    use_d49t_forward_lags=False,
    use_catboost=False,
    use_ridge_stacking=False,
    use_tweedie=False,
    use_quantile=False,
    use_bias_correction=False,
)


def make_cfg(**overrides):
    kw = dict(BASE_KW)
    kw.update(overrides)
    return Config(**kw)


def main():
    train_df, test_df = load_data()
    d48 = train_df[train_df["day"] == 48].reset_index(drop=True)
    d49t = train_df[train_df["day"] == 49].reset_index(drop=True)
    test = test_df.copy().reset_index(drop=True)

    print(f"d48: {d48.shape}    d49t: {d49t.shape}")

    # Reference: pure baseline (no improvements)
    print("\n" + "=" * 70)
    print("  REFERENCE — baseline features only")
    print("=" * 70)
    cfg_ref = make_cfg()
    d48f, d49f, _, _, _, _ = assemble_features(d48, d49t, test, cfg_ref, mode="A")
    res_ref = evaluate_mode(d48f, d49f, cfg_ref, "A")
    print(f"  Blend R²  : {res_ref['blend_r2']:.4f}  (Score={res_ref['score']:.2f})")
    ref_r2 = res_ref["blend_r2"]

    # Test matrix
    tests = [
        ("C-2 target_encoding α=20",  dict(use_target_encoding=True, te_alpha=20.0)),
        ("C-2 target_encoding α=5",   dict(use_target_encoding=True, te_alpha=5.0)),
        ("C-2 target_encoding α=50",  dict(use_target_encoding=True, te_alpha=50.0)),
        ("C-2 target_encoding α=100", dict(use_target_encoding=True, te_alpha=100.0)),
        ("C-3 interactions",          dict(use_interactions=True)),
        ("C-2 + C-3 combined",        dict(use_target_encoding=True, te_alpha=20.0, use_interactions=True)),
        ("C-4 d49t forward lags (only effective on test, expect 0 delta on d48)", dict(use_d49t_forward_lags=True)),
    ]

    results = []
    for name, overrides in tests:
        print(f"\n{'-' * 70}\n  TEST: {name}\n{'-' * 70}")
        cfg = make_cfg(**overrides)
        d48f, d49f, _, _, _, _ = assemble_features(d48, d49t, test, cfg, mode="A")
        res = evaluate_mode(d48f, d49f, cfg, "A")
        delta = res["blend_r2"] - ref_r2
        print(f"  Blend R²  : {res['blend_r2']:.4f}  (Δ={delta:+.4f}, Score={res['score']:.2f})")
        results.append({"name": name, "blend_r2": res["blend_r2"], "delta": delta,
                        "lgb_r2": res["lgb_r2"], "xgb_r2": res["xgb_r2"]})

    out = OUTPUT / "ab_test_results.json"
    with open(out, "w") as f:
        json.dump({"ref_r2": ref_r2, "tests": results}, f, indent=2)
    print(f"\nSaved: {out.relative_to(ROOT)}")

    # Summary
    print("\n" + "=" * 70)
    print("  A/B TEST SUMMARY (Mode A — KFold d48)")
    print("=" * 70)
    print(f"  Reference:  R²={ref_r2:.4f}  Score={ref_r2*100:.2f}")
    print()
    for r in sorted(results, key=lambda x: -x["delta"]):
        marker = "↑" if r["delta"] > 0 else ("↓" if r["delta"] < 0 else "·")
        print(f"  {marker} {r['name'][:55]:<55}  R²={r['blend_r2']:.4f}  Δ={r['delta']:+.4f}")


if __name__ == "__main__":
    main()
