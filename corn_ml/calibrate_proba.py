#!/usr/bin/env python3
"""概率校准对比：Platt Scaling vs Isotonic Regression vs 未校准"""

import os
import warnings

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, accuracy_score

warnings.filterwarnings("ignore")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
HORIZONS = [5, 10, 20]

from build_features import FEATURE_COLS


def load(horizon: int):
    d = np.load(os.path.join(DATA_DIR, f"features_h{horizon}.npz"))
    return d["X_train"], d["X_val"], d["X_test"], d["y_train"], d["y_val"], d["y_test"]


def base_model():
    return LGBMClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=4,
        num_leaves=15, min_child_samples=50, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbose=-1,
    )


def brier_decomposition(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    resolution = 0.0
    reliability = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        n_b = mask.sum()
        if n_b == 0:
            continue
        f_b = y_prob[mask].mean()
        o_b = y_true[mask].mean()
        resolution += n_b * (o_b - y_true.mean()) ** 2
        reliability += n_b * (f_b - o_b) ** 2
    N = len(y_true)
    uncertainty = y_true.mean() * (1 - y_true.mean())
    resolution /= N
    reliability /= N
    return uncertainty, reliability, resolution


def main():
    print("=" * 62)
    print("  LightGBM 概率校准对比：Platt / Isotonic / None")
    print("=" * 62)

    for horizon in HORIZONS:
        print(f"\n{'─'*62}")
        print(f"  ◆ T+{horizon}日")
        print(f"{'─'*62}")

        X_tr, X_val, X_te, y_tr, y_val, y_te = load(horizon)

        # train base model
        raw = base_model()
        raw.fit(X_tr, y_tr)

        # -- uncalibrated --
        prob_raw = raw.predict_proba(X_te)[:, 1]
        brier_raw = brier_score_loss(y_te, prob_raw)
        auc_raw = roc_auc_score(y_te, prob_raw)
        unc, rel, res = brier_decomposition(y_te, prob_raw)

        # -- Platt scaling (sigmoid) --
        from sklearn.calibration import _SigmoidCalibration
        platt_cal = _SigmoidCalibration()
        raw_val_prob = raw.predict_proba(X_val)[:, 1]
        platt_cal.fit(raw_val_prob, y_val)
        raw_te_prob = raw.predict_proba(X_te)[:, 1]
        prob_platt = platt_cal.predict(raw_te_prob)
        prob_platt = np.clip(prob_platt, 0, 1)
        brier_platt = brier_score_loss(y_te, prob_platt)
        auc_platt = roc_auc_score(y_te, prob_platt)
        unc_p, rel_p, res_p = brier_decomposition(y_te, prob_platt)

        # -- Isotonic regression --
        from sklearn.isotonic import IsotonicRegression
        iso_cal = IsotonicRegression(out_of_bounds="clip")
        iso_cal.fit(raw_val_prob, y_val)
        prob_iso = iso_cal.predict(raw_te_prob)
        prob_iso = np.clip(prob_iso, 0, 1)
        brier_iso = brier_score_loss(y_te, prob_iso)
        auc_iso = roc_auc_score(y_te, prob_iso)
        unc_i, rel_i, res_i = brier_decomposition(y_te, prob_iso)

        # ---- results ----
        print(f"\n  {'方法':<16s} {'Brier':>8s} {'AUC':>8s} {'可靠度':>8s} {'分辨度':>8s} {'不确定性':>8s}")
        print(f"  {'─'*58}")
        for name, br, au, u, rl, rs in [
            ("未校准", brier_raw, auc_raw, unc, rel, res),
            ("Platt", brier_platt, auc_platt, unc_p, rel_p, res_p),
            ("Isotonic", brier_iso, auc_iso, unc_i, rel_i, res_i),
        ]:
            print(f"  {name:<16s} {br:>7.4f} {au:>7.4f} {rl:>7.4f} {rs:>7.4f} {u:>7.4f}")

        # best
        best_name = min(
            [("未校准", brier_raw), ("Platt", brier_platt), ("Isotonic", brier_iso)],
            key=lambda x: x[1],
        )[0]
        print(f"  → 最佳: {best_name} (Brier最低)")

        # reliability bins table
        best_prob = {"未校准": prob_raw, "Platt": prob_platt, "Isotonic": prob_iso}[best_name]

        print(f"\n  {best_name} 可靠性分桶 (每桶 ≈{len(y_te)//10} 样本):")
        print(f"  {'概率区间':<14s} {'样本数':>6s} {'预测均值':>8s} {'实际比例':>8s} {'偏差':>8s}")
        print(f"  {'─'*46}")
        bins = np.linspace(0, 1, 11)
        for i in range(10):
            mask = (best_prob >= bins[i]) & (best_prob < bins[i + 1])
            n = mask.sum()
            if n == 0:
                continue
            pred_mean = best_prob[mask].mean()
            actual = y_te[mask].mean()
            bias = pred_mean - actual
            print(f"  [{bins[i]:.1f}-{bins[i+1]:.1f}) {n:>6d} {pred_mean:>7.3f} {actual:>7.3f} {bias:>+8.3f}")

    print(f"\n{'='*62}")
    print("  完成")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()
