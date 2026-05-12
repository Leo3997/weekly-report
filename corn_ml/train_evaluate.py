#!/usr/bin/env python3
"""LightGBM v5：超参搜索 + 三分类 + 概率校准 + 滚动回测"""

import json
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FORECAST_HORIZONS = [5, 10, 20]
N_HYPER_SEARCH = 50


def load_features(horizon: int) -> tuple:
    data = np.load(os.path.join(DATA_DIR, f"features_h{horizon}.npz"))
    return (data["X_train"], data["X_val"], data["X_test"],
            data["y_train"], data["y_val"], data["y_test"])


def random_hyper_search(X_train, y_train, X_val, y_val, n_iter=50):
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    from sklearn.metrics import roc_auc_score

    best_score, best_params = 0, None
    results = []

    for i in range(n_iter):
        params = {
            "n_estimators": np.random.choice([200, 300, 400, 500, 600]),
            "learning_rate": np.random.choice([0.01, 0.02, 0.03, 0.05, 0.08]),
            "max_depth": np.random.choice([3, 4, 5, 6]),
            "num_leaves": np.random.choice([8, 12, 16, 24, 31]),
            "min_child_samples": np.random.choice([20, 30, 50, 80, 120]),
            "subsample": np.random.choice([0.6, 0.7, 0.8, 0.9]),
            "colsample_bytree": np.random.choice([0.5, 0.6, 0.7, 0.8]),
            "reg_alpha": np.random.choice([0.0, 0.1, 0.3, 0.5]),
            "reg_lambda": np.random.choice([0.0, 0.5, 1.0, 2.0]),
            "random_state": 42,
            "verbose": -1,
        }
        try:
            model = LGBMClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                      eval_metric="auc",
                      callbacks=[early_stopping(30, verbose=False), log_evaluation(0)])
            y_prob = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_prob)
            results.append({"auc": auc, **params})
            if auc > best_score:
                best_score = auc
                best_params = {k: v for k, v in params.items()
                               if k not in ["random_state", "verbose"]}
        except Exception:
            continue

    if not results:
        return {"n_estimators": 400, "learning_rate": 0.03, "max_depth": 4,
                "num_leaves": 16, "min_child_samples": 50, "subsample": 0.8,
                "colsample_bytree": 0.7, "reg_alpha": 0.1, "reg_lambda": 1.0}

    results.sort(key=lambda x: x["auc"], reverse=True)
    top5_auc = np.mean([r["auc"] for r in results[:5]])
    print(f"  搜索{n_iter}轮 | best AUC={best_score:.4f} | top5均值={top5_auc:.4f}")
    print(f"  best: lr={best_params['learning_rate']}, depth={best_params['max_depth']}, "
          f"leaves={best_params['num_leaves']}, min_child={best_params['min_child_samples']}")
    return best_params


def train_lightgbm(X_train, y_train, X_val, y_val, params=None):
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation

    if params is None:
        params = {"n_estimators": 400, "learning_rate": 0.03, "max_depth": 4,
                  "num_leaves": 16, "min_child_samples": 50, "subsample": 0.8,
                  "colsample_bytree": 0.7, "reg_alpha": 0.1, "reg_lambda": 1.0}

    model = LGBMClassifier(**params, random_state=42, verbose=-1)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="auc",
              callbacks=[early_stopping(50, verbose=False), log_evaluation(0)])
    return model


def calibrate_predict(model, X_val, y_val, X_te):
    from sklearn.isotonic import IsotonicRegression
    raw_val = model.predict_proba(X_val)[:, 1]
    raw_te = model.predict_proba(X_te)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    try:
        iso.fit(raw_val, y_val)
        return np.clip(iso.predict(raw_te), 0, 1)
    except Exception:
        return raw_te


def evaluate(model, X, y, proba=None) -> dict:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    if proba is None:
        proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    return {
        "n": len(y), "acc": accuracy_score(y, pred),
        "prec": precision_score(y, pred, zero_division=0),
        "rec": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "auc": roc_auc_score(y, proba), "up": y.mean(),
    }


def rolling_backtest(X_all, y_all, dates, train_months=60):
    from lightgbm import LGBMClassifier
    results = []
    unique_months = sorted(set((d.year, d.month) for d in dates))
    if len(unique_months) < train_months + 1:
        return pd.DataFrame()

    for si in range(len(unique_months) - train_months):
        tm = unique_months[si + train_months]
        te_m = unique_months[si + train_months - 1]
        ts = unique_months[si]
        tr_mask = dates.apply(lambda d: (d.year, d.month) >= ts and (d.year, d.month) <= te_m).values
        te_mask = dates.apply(lambda d: (d.year, d.month) == tm).values
        X_tr, y_tr = X_all[tr_mask], y_all[tr_mask]
        X_te, y_te = X_all[te_mask], y_all[te_mask]
        if len(X_tr) < 50 or len(X_te) < 5: continue
        m = LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=4,
                           num_leaves=15, random_state=42, verbose=-1)
        m.fit(X_tr, y_tr)
        from sklearn.metrics import accuracy_score
        results.append({"month": f"{tm[0]}-{tm[1]:02d}", "n": len(y_te),
                        "acc": accuracy_score(y_te, m.predict(X_te))})
    return pd.DataFrame(results)


def main():
    from build_features import FEATURE_COLS, CORE_FEATURES, PARTIAL_FEATURES, NOISE_THRESHOLD

    df = pd.read_csv(os.path.join(DATA_DIR, "features_full_v2.csv"), parse_dates=["date"])

    print("=" * 66)
    print("  LightGBM v5 — 精简特征 + 超参搜索 + 三分类 + 校准")
    print("=" * 66)
    print(f"特征: {len(FEATURE_COLS)} (核心{len(CORE_FEATURES)} + 部分{len(PARTIAL_FEATURES)})")
    print(f"三分类阈值: |return| > {NOISE_THRESHOLD:.1%}")
    print(f"超参搜索: {N_HYPER_SEARCH}轮")

    summary_rows = []

    for horizon in FORECAST_HORIZONS:
        print(f"\n{'='*66}")
        print(f"  ◆ T+{horizon}d")
        print(f"{'='*66}")

        X_train, X_val, X_test, y_train, y_val, y_test = load_features(horizon)
        print(f"样本: train={len(X_train)} val={len(X_val)} test={len(X_test)}")
        print(f"涨跌比: train={y_train.mean():.1%} val={y_val.mean():.1%} test={y_test.mean():.1%}")

        # 超参搜索
        print("\n  [超参搜索...]")
        best_params = random_hyper_search(X_train, y_train, X_val, y_val, N_HYPER_SEARCH)

        # 训练
        print("  [训练...]")
        model = train_lightgbm(X_train, y_train, X_val, y_val, best_params)
        print(f"  最佳迭代: {model.best_iteration_}")

        # 概率校准
        proba_cal = calibrate_predict(model, X_val, y_val, X_test)

        # 评估
        print(f"\n  {'集合':<6s} {'样本':>6s} {'准确率':>7s} {'精确率':>7s} {'召回率':>7s} {'F1':>7s} {'AUC':>7s}")
        print(f"  {'─'*52}")
        for X, y, name in [(X_train, y_train, "训练"), (X_val, y_val, "验证"), (X_test, y_test, "测试")]:
            m = evaluate(model, X, y)
            print(f"  {name:<6s} {m['n']:>6d} {m['acc']:>6.2%} {m['prec']:>6.2%} {m['rec']:>6.2%} {m['f1']:>7.4f} {m['auc']:>7.4f}")
            if name == "测试":
                summary_rows.append({"Horizon": f"T+{horizon}d", **{k: m[k] for k in ["n", "acc", "auc", "prec", "rec", "f1"]}})

        # 特征重要性
        imp = model.feature_importances_
        imp = imp / imp.sum()
        top = np.argsort(imp)[::-1][:8]
        print(f"\n  特征重要性:")
        for rank, idx in enumerate(top, 1):
            bar = "█" * int(imp[idx] * 35 + 1)
            print(f"    {rank}. {FEATURE_COLS[idx]:<24s} {imp[idx]:.1%}  {bar}")

        # 校准效果
        from sklearn.metrics import roc_auc_score, brier_score_loss
        raw_proba = model.predict_proba(X_test)[:, 1]
        print(f"\n  概率校准: raw AUC={roc_auc_score(y_test, raw_proba):.4f} Brier={brier_score_loss(y_test, raw_proba):.4f} -> cal AUC={roc_auc_score(y_test, proba_cal):.4f} Brier={brier_score_loss(y_test, proba_cal):.4f}")

        # 不确定度区间
        uncertain = (proba_cal > 0.4) & (proba_cal < 0.6)
        confident_mask = ~uncertain
        if confident_mask.sum() > 5:
            from sklearn.metrics import accuracy_score
            conf_acc = accuracy_score(y_test[confident_mask], (proba_cal[confident_mask] > 0.5).astype(int))
            print(f"  置信区间: {uncertain.sum()}/{len(y_test)} = {uncertain.mean():.0%} 不确定 -> 剔除后准确率 {conf_acc:.1%}")

        # 滚动回测
        print(f"\n  [滚动回测...]")
        label_col = f"label_binary_{horizon}d"
        mask = df[label_col].notna() & (df[label_col] != -1)
        for col in CORE_FEATURES:
            mask &= df[col].notna()
        clean = df[mask].copy()
        for col in PARTIAL_FEATURES:
            clean[col] = clean[col].fillna(0.0)
        X_all = clean[FEATURE_COLS].values.astype(np.float64)
        y_all = clean[label_col].values.astype(np.int64)
        bt = rolling_backtest(X_all, y_all, clean["date"])
        if not bt.empty:
            r12 = bt.tail(12)
            print(f"  {len(bt)}个月 | 均值={bt['acc'].mean():.2%} ±{bt['acc'].std():.3f} | 近12月={r12['acc'].mean():.2%}")

    # 汇总
    print(f"\n{'='*66}")
    print(f"  多目标汇总 (三分类 + 校准)")
    print(f"{'='*66}")
    print(f"  {'目标':<10s} {'样本':>6s} {'准确率':>7s} {'AUC':>7s} {'精确率':>7s} {'召回率':>7s} {'F1':>7s}")
    print(f"  {'─'*52}")
    for s in summary_rows:
        print(f"  {s['Horizon']:<10s} {s['n']:>6d} {s['acc']:>6.2%} {s['auc']:>7.4f} {s['prec']:>6.2%} {s['rec']:>6.2%} {s['f1']:>7.4f}")
    print(f"\n  完成 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*66}")


if __name__ == "__main__":
    main()
