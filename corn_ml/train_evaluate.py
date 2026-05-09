#!/usr/bin/env python3
"""LightGBM 多目标训练 + 概率输出 + 回测评估 (阶段二)"""

import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FORECAST_HORIZONS = [5, 10, 20]


def load_features(horizon: int) -> tuple:
    data = np.load(os.path.join(DATA_DIR, f"features_h{horizon}.npz"))
    return (
        data["X_train"], data["X_val"], data["X_test"],
        data["y_train"], data["y_val"], data["y_test"],
    )


def train_lightgbm(X_train, y_train, X_val, y_val):
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation

    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=3,
        num_leaves=12,
        min_child_samples=80,
        subsample=0.7,
        colsample_bytree=0.6,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42,
        verbose=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[
            early_stopping(stopping_rounds=50, verbose=False),
            log_evaluation(0),
        ],
    )
    return model


def evaluate(model, X, y) -> dict:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    y_prob = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)
    return {
        "n": len(y),
        "acc": accuracy_score(y, y_pred),
        "prec": precision_score(y, y_pred, zero_division=0),
        "rec": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "auc": roc_auc_score(y, y_prob),
        "up_ratio": y.mean(),
    }


def rolling_backtest(X_all, y_all, dates, train_months=36):
    from lightgbm import LGBMClassifier

    results = []
    unique_months = sorted(set((d.year, d.month) for d in dates))
    if len(unique_months) < train_months + 1:
        return pd.DataFrame()

    for start_idx in range(len(unique_months) - train_months):
        test_month = unique_months[start_idx + train_months]
        train_end = unique_months[start_idx + train_months - 1]
        train_start = unique_months[start_idx]

        train_mask = dates.apply(
            lambda d: (d.year, d.month) >= train_start and (d.year, d.month) <= train_end
        ).values
        test_mask = dates.apply(lambda d: (d.year, d.month) == test_month).values

        X_tr, y_tr = X_all[train_mask], y_all[train_mask]
        X_te, y_te = X_all[test_mask], y_all[test_mask]
        if len(X_tr) < 50 or len(X_te) < 5:
            continue

        m = LGBMClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            num_leaves=15, random_state=42, verbose=-1,
        )
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_te)

        from sklearn.metrics import accuracy_score
        results.append({
            "month": f"{test_month[0]}-{test_month[1]:02d}",
            "n": len(y_te),
            "acc": accuracy_score(y_te, y_pred),
        })
    return pd.DataFrame(results)


def main() -> None:
    from build_features import FEATURE_COLS, CORE_FEATURES, PARTIAL_FEATURES

    df = pd.read_csv(os.path.join(DATA_DIR, "features_full_v2.csv"), parse_dates=["date"])

    print("=" * 60)
    print("  LightGBM 玉米期货多目标涨跌预测 (阶段二)")
    print("=" * 60)
    print(f"特征数: {len(FEATURE_COLS)}")
    print()

    summary_rows = []

    for horizon in FORECAST_HORIZONS:
        print(f"\n{'='*60}")
        print(f"  ◆ T+{horizon}日预测")
        print(f"{'='*60}")

        X_train, X_val, X_test, y_train, y_val, y_test = load_features(horizon)
        print(f"训练: {len(X_train):,}  验证: {len(X_val):,}  测试: {len(X_test):,}")
        print(f"上涨比: {y_train.mean():.1%} / {y_val.mean():.1%} / {y_test.mean():.1%}")

        model = train_lightgbm(X_train, y_train, X_val, y_val)
        print(f"最佳迭代: {model.best_iteration_}")

        print(f"\n  {'数据集':<8s} {'样本':>6s} {'准确率':>7s} {'精确率':>7s} {'召回率':>7s} {'F1':>7s} {'AUC':>7s}")
        print(f"  {'-'*52}")
        for X, y, name in [(X_train, y_train, "训练"), (X_val, y_val, "验证"), (X_test, y_test, "测试")]:
            m = evaluate(model, X, y)
            print(f"  {name:<8s} {m['n']:>6d} {m['acc']:>6.2%} {m['prec']:>6.2%} {m['rec']:>6.2%} {m['f1']:>7.4f} {m['auc']:>7.4f}")
            if name == "测试":
                summary_rows.append({
                    "Horizon": f"T+{horizon}d",
                    "n_test": m["n"],
                    "acc": m["acc"],
                    "auc": m["auc"],
                    "prec": m["prec"],
                    "rec": m["rec"],
                    "f1": m["f1"],
                })

        # 特征重要性
        importances = model.feature_importances_
        total = importances.sum()
        if total > 0:
            importances = importances / total
        sorted_idx = np.argsort(importances)[::-1]
        print(f"\n  特征重要性 (T+{horizon}d):")
        for rank, idx in enumerate(sorted_idx[:6], 1):
            bar = "█" * int(importances[idx] * 40)
            print(f"    {rank}. {FEATURE_COLS[idx]:<18s} {importances[idx]:.4f}  {bar}")

        # 最近预测
        label_col = f"label_{horizon}d"
        mask = df["date"] >= "2025-01-01"
        for col in CORE_FEATURES:
            mask &= df[col].notna()
        mask &= df[label_col].notna()
        recent = df[mask].tail(10)
        for col in PARTIAL_FEATURES:
            recent[col] = recent[col].fillna(0.0)
        if len(recent) > 0:
            X_rec = recent[FEATURE_COLS].values.astype(np.float64)
            probs = model.predict_proba(X_rec)[:, 1]
            print(f"\n  最近10日预测 (T+{horizon}d):")
            print(f"    {'日期':<12s} {'涨概率':>7s}  {'预测':>4s}  {'实际':>4s}")
            correct = 0
            for _, row in recent.iterrows():
                i = recent.index.get_loc(row.name)
                pred = "↑" if probs[i] > 0.5 else "↓"
                actual = "↑" if row[label_col] == 1 else "↓"
                if pred == actual:
                    correct += 1
                print(f"    {row['date'].strftime('%Y-%m-%d'):<12s} {probs[i]:>6.1%}  {pred:>4s}  {actual:>4s}")
            print(f"    {'─'*29}")
            print(f"    正确率: {correct}/{len(recent)} = {correct/len(recent):.0%}")

        # 滚动回测
        mask_bt = df[label_col].notna()
        for col in CORE_FEATURES:
            mask_bt &= df[col].notna()
        clean = df[mask_bt].copy()
        for col in PARTIAL_FEATURES:
            clean[col] = clean[col].fillna(0.0)
        X_all = clean[FEATURE_COLS].values.astype(np.float64)
        y_all = clean[label_col].values.astype(np.int64)
        bt = rolling_backtest(X_all, y_all, clean["date"], train_months=60)
        if not bt.empty:
            print(f"\n  滚动回测 ({len(bt)} 月): 均准确率 {bt['acc'].mean():.2%} ±{bt['acc'].std():.3f}  近12月 {bt['acc'].tail(12).mean():.2%}")

    # ==================== 汇总 ====================
    print(f"\n{'='*60}")
    print("  多目标汇总")
    print(f"{'='*60}")
    print(f"  {'目标':<10s} {'样本':>6s} {'准确率':>7s} {'AUC':>7s} {'精确率':>7s} {'召回率':>7s} {'F1':>7s}")
    print(f"  {'-'*52}")
    for s in summary_rows:
        print(f"  {s['Horizon']:<10s} {s['n_test']:>6d} {s['acc']:>6.2%} {s['auc']:>7.4f} {s['prec']:>6.2%} {s['rec']:>6.2%} {s['f1']:>7.4f}")
    print(f"\n  完成 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
