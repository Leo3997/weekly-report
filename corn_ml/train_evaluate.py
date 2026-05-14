#!/usr/bin/env python3
"""LightGBM v5：超参搜索 + AI权重接入 + 温度缩放校准 + 前向预测"""

import json
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FORECAST_HORIZONS = [5, 10, 20, 30, 60]
N_HYPER_SEARCH = 20

SAMPLE_WEIGHT_RANGE = (0.4, 2.5)
SAMPLE_WEIGHT_STEEPNESS = 3.0
DAILY_HORIZONS = list(range(1, 21))


def load_features(horizon: int) -> tuple:
    data = np.load(os.path.join(DATA_DIR, f"features_h{horizon}.npz"))
    return (data["X_train"], data["X_val"], data["X_test"],
            data["y_train"], data["y_val"], data["y_test"])


def load_ai_weights() -> dict:
    weights_path = os.path.join(DATA_DIR, "ai_weights.json")
    if not os.path.exists(weights_path):
        print("  [AI权重] ai_weights.json 不存在，使用均匀权重")
        return {}
    with open(weights_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    weights = data.get("weights", {})
    source = data.get("source", "unknown")
    wdate = data.get("date", "unknown")
    print(f"  [AI权重] 已加载 ({source}, {wdate}), {len(weights)} 个特征")
    return weights


def compute_ai_sample_weight(X, feature_names, ai_weights,
                              ref_median=None, ref_mad=None):
    if not ai_weights:
        return np.ones(len(X)), None, None

    col_indices = []
    col_weights = []
    for j, name in enumerate(feature_names):
        w = ai_weights.get(name, 0.0)
        if w > 0.001:
            col_indices.append(j)
            col_weights.append(w)
    if not col_indices:
        return np.ones(len(X)), None, None

    if ref_median is None:
        ref_median = np.nanmedian(X[:, col_indices], axis=0)
    if ref_mad is None:
        ref_mad = np.nanmedian(np.abs(X[:, col_indices] - ref_median), axis=0)
        ref_mad = np.where(ref_mad < 1e-8, 1e-8, ref_mad)

    col_weights = np.array(col_weights)
    col_weights = col_weights / col_weights.sum()

    deviations = np.abs(X[:, col_indices] - ref_median) / ref_mad
    scores = deviations @ col_weights

    lo, hi = SAMPLE_WEIGHT_RANGE
    k = SAMPLE_WEIGHT_STEEPNESS
    x0 = np.median(scores)
    sample_weights = lo + (hi - lo) / (1.0 + np.exp(-k * (scores - x0)))

    return sample_weights, ref_median, ref_mad


def find_optimal_threshold(y_true, y_prob):
    from sklearn.metrics import balanced_accuracy_score
    p_lo = max(0.30, np.percentile(y_prob, 5))
    p_hi = min(0.70, np.percentile(y_prob, 95))
    if p_hi - p_lo < 0.05:
        p_lo = max(0.40, np.median(y_prob) - 0.05)
        p_hi = min(0.60, np.median(y_prob) + 0.05)
    thresholds = np.linspace(p_lo, p_hi, 81)
    best_thresh = 0.5
    best_score = -1.0
    for t in thresholds:
        score = balanced_accuracy_score(y_true, (y_prob >= t).astype(int))
        if score > best_score:
            best_score = score
            best_thresh = t
    return best_thresh


def random_hyper_search(X_train, y_train, X_val, y_val, n_iter=50, sample_weight_tr=None):
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    from sklearn.metrics import roc_auc_score

    best_score, best_params = 0, None
    results = []
    sw = sample_weight_tr

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
            "is_unbalance": True,
            "random_state": 42,
            "verbose": -1,
        }
        try:
            model = LGBMClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                      eval_metric="auc", sample_weight=sw,
                      callbacks=[early_stopping(30, verbose=False), log_evaluation(0)])
            y_prob = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_prob)
            results.append({"auc": auc, **params})
            if auc > best_score:
                best_score = auc
                best_params = {k: v for k, v in params.items()
                               if k not in ["random_state", "verbose", "is_unbalance"]}
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


def train_lightgbm(X_train, y_train, X_val, y_val, params=None, sample_weight_tr=None):
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation

    if params is None:
        params = {"n_estimators": 400, "learning_rate": 0.03, "max_depth": 4,
                  "num_leaves": 16, "min_child_samples": 50, "subsample": 0.8,
                  "colsample_bytree": 0.7, "reg_alpha": 0.1, "reg_lambda": 1.0}

    params["is_unbalance"] = True
    model = LGBMClassifier(**params, random_state=42, verbose=-1)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="auc",
              sample_weight=sample_weight_tr,
              callbacks=[early_stopping(50, verbose=False), log_evaluation(0)])
    return model


def calibrate_predict(model, X_val, y_val, X_te, sample_weight_val=None):
    from scipy.special import expit, logit
    from scipy.optimize import minimize_scalar
    from sklearn.metrics import brier_score_loss

    raw_val = model.predict_proba(X_val)[:, 1]
    raw_te = model.predict_proba(X_te)[:, 1]

    if len(raw_te) < 20:
        return raw_te

    raw_val_c = np.clip(raw_val, 1e-6, 1 - 1e-6)
    raw_te_c = np.clip(raw_te, 1e-6, 1 - 1e-6)

    raw_spread = raw_te.std()
    if raw_spread < 0.03:
        print(f"  跳过校准 (原始概率过于集中, std={raw_spread:.4f})")
        return raw_te

    def _brier_t(T):
        p = expit(logit(raw_val_c) / T)
        return brier_score_loss(y_val, p)

    try:
        res = minimize_scalar(_brier_t, bounds=(0.2, 5.0), method="bounded")
        T_opt = float(res.x)
    except Exception:
        T_opt = 1.0

    cal_te = expit(logit(raw_te_c) / T_opt)
    cal_te = np.clip(cal_te, 0.001, 0.999)

    cal_spread = cal_te.std()
    if cal_spread / max(raw_spread, 1e-6) < 0.3 or cal_spread < 0.02:
        print(f"  跳过校准 (校准后分布过窄, T={T_opt:.2f} cal_std={cal_spread:.4f} raw_std={raw_spread:.4f})")
        return raw_te

    direction = "锐化" if T_opt < 1.0 else "平滑"
    print(f"  温度缩放: T={T_opt:.2f}({direction}) raw_std={raw_spread:.3f} -> cal_std={cal_te.std():.3f}")

    return cal_te


def evaluate(model, X, y, proba=None, threshold=0.5) -> dict:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    if proba is None:
        proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "n": len(y), "acc": accuracy_score(y, pred),
        "prec": precision_score(y, pred, zero_division=0),
        "rec": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "auc": roc_auc_score(y, proba), "up": y.mean(),
        "thresh": threshold, "pred_up": pred.mean(),
    }


def rolling_backtest(X_all, y_all, dates, train_months=60,
                     feature_names=None, ai_weights=None):
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

        sw_tr = None
        if ai_weights and feature_names:
            sw_tr, _, _ = compute_ai_sample_weight(X_tr, feature_names, ai_weights)

        m = LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=4,
                           num_leaves=15, is_unbalance=True, random_state=42, verbose=-1)
        m.fit(X_tr, y_tr, sample_weight=sw_tr)
        from sklearn.metrics import accuracy_score
        results.append({"month": f"{tm[0]}-{tm[1]:02d}", "n": len(y_te),
                        "acc": accuracy_score(y_te, m.predict(X_te))})
    return pd.DataFrame(results)


def get_latest_features(df, feature_cols, core_features):
    mask = pd.Series(True, index=df.index)
    for col in core_features:
        mask &= df[col].notna()
    clean = df[mask].copy()
    latest = clean.iloc[-1]
    X_latest = latest[feature_cols].values.astype(np.float64).reshape(1, -1)
    return X_latest, latest["date"], latest["close"]


def extract_test_dates_and_actuals(df, horizon, core_features):
    label_col = f"label_binary_{horizon}d"
    return_col = f"future_return_{horizon}d"
    mask = df[label_col].notna() & (df[label_col] != -1)
    for col in core_features:
        mask &= df[col].notna()
    clean = df[mask].copy()
    test_mask = clean["date"] >= "2025-06-01"
    te = clean[test_mask]
    return te["date"].values, te[label_col].values.astype(np.int64), te[return_col].values.astype(np.float64)


def plot_prediction_curves(all_dates, all_probas, all_actuals, all_returns, save_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except Exception:
        print("  [曲线] matplotlib 不可用，跳过")
        return

    try:
        plt.rcParams["font.sans-serif"] = ["SimHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    colors = ["#2196F3", "#FF9800", "#E91E63", "#4CAF50", "#9C27B0"]
    color_light = ["#BBDEFB", "#FFE0B2", "#F8BBD0", "#C8E6C9", "#E1BEE7"]

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35,
                          height_ratios=[2.5, 1.5, 1.5])

    # =================== 图A: 多周期预测概率时序 ===================
    ax_ts = fig.add_subplot(gs[0, :])
    plot_days = 120
    n_total = len(all_dates[5])
    start_idx = max(0, n_total - plot_days)

    for h_idx, h in enumerate(FORECAST_HORIZONS):
        d = all_dates[h][start_idx:]
        p = all_probas[h][start_idx:]
        ax_ts.plot(d, p, color=colors[h_idx], linewidth=1.5,
                   label=f"T+{h}天", alpha=0.85)
        ax_ts.fill_between(d, 0.5, p, color=color_light[h_idx], alpha=0.35)

    ax_ts.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_ts.set_ylabel("上涨概率", fontsize=11)
    ax_ts.set_title("多周期预测概率时序（测试集）", fontsize=13, fontweight="bold")
    ax_ts.legend(loc="upper left", fontsize=9, ncol=5)
    ax_ts.set_ylim(0, 1)
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax_ts.tick_params(axis="both", labelsize=9)
    ax_ts.grid(True, alpha=0.3)

    for h_idx, h in enumerate(FORECAST_HORIZONS):
        latest_p = all_probas[h][-1]
        latest_d = all_dates[h][-1]
        ax_ts.annotate(f"  {latest_p:.0%}",
                       xy=(mdates.date2num(latest_d), latest_p),
                       fontsize=8, fontweight="bold", color=colors[h_idx],
                       va="center")

    # =================== 图B: 最新预测周期分布 ===================
    ax_horizon = fig.add_subplot(gs[1, :])
    latest_probas = [all_probas[h][-1] for h in FORECAST_HORIZONS]
    latest_labels = [f"T+{h}天" for h in FORECAST_HORIZONS]

    bars = ax_horizon.bar(latest_labels, latest_probas, color=colors,
                          width=0.5, edgecolor="white", linewidth=1.2)
    for bar, proba in zip(bars, latest_probas):
        ax_horizon.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{proba:.0%}", ha="center", fontsize=12, fontweight="bold",
                        color="#333333")
    ax_horizon.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8)
    ax_horizon.set_ylim(0, 1)
    ax_horizon.set_ylabel("上涨概率", fontsize=11)
    latest_date_str = pd.Timestamp(all_dates[5][-1]).strftime("%Y-%m-%d")
    ax_horizon.set_title(f"最新预测周期分布（{latest_date_str}）", fontsize=12, fontweight="bold")
    ax_horizon.tick_params(axis="both", labelsize=10)
    ax_horizon.grid(True, alpha=0.2, axis="y")

    for h_idx, h in enumerate(FORECAST_HORIZONS):
        direction = "看涨" if latest_probas[h_idx] > 0.5 else "看跌"
        emoji = "▲" if latest_probas[h_idx] > 0.5 else "▼"
        ax_horizon.text(h_idx, 0.08, f"{emoji} {direction}",
                        ha="center", fontsize=9, color=colors[h_idx],
                        fontweight="bold")

    # =================== 图C: 近30日预测热度图 ===================
    ax_heat = fig.add_subplot(gs[2, :])
    heat_days = 30
    heat_data = []
    n_total = len(all_dates[5])
    h_start = max(0, n_total - heat_days)
    for h_idx, h in enumerate(FORECAST_HORIZONS):
        row = []
        for i in range(max(0, n_total - heat_days), n_total):
            if i < len(all_probas[h]):
                row.append(all_probas[h][i])
            else:
                row.append(np.nan)
        heat_data.append(row)
    heat_data = np.array(heat_data)
    d_subset = all_dates[5][h_start:]
    date_strs = [pd.Timestamp(d).strftime("%m-%d") for d in d_subset]

    y_labels = [f"T+{h}天" for h in FORECAST_HORIZONS]
    im = ax_heat.imshow(heat_data, aspect="auto", cmap="RdYlGn",
                        vmin=0, vmax=1, interpolation="nearest")
    ax_heat.set_yticks(range(len(y_labels)))
    ax_heat.set_yticklabels(y_labels, fontsize=10)
    xtick_step = max(1, len(date_strs) // 10)
    ax_heat.set_xticks(range(0, len(date_strs), xtick_step))
    ax_heat.set_xticklabels([date_strs[i] for i in range(0, len(date_strs), xtick_step)],
                            fontsize=8, rotation=45)
    ax_heat.set_title("近30日预测热度图", fontsize=12, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax_heat, shrink=0.85, pad=0.02)
    cbar.set_label("上涨概率", fontsize=9)

    fig.suptitle("玉米期货多周期预测仪表盘（测试集评估）",
                 fontsize=15, fontweight="bold", y=0.98)

    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  [评估仪表盘] 已保存 -> {save_path}")


def plot_forward_forecast(models, latest_date, close_price, X_latest, save_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("  [前向预测] matplotlib 不可用，跳过")
        return

    try:
        plt.rcParams["font.sans-serif"] = ["SimHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    probas = {}
    for h in FORECAST_HORIZONS:
        p = models[h].predict_proba(X_latest)[:, 1]
        probas[h] = float(p[0])

    forecast_dates = [latest_date + pd.Timedelta(days=h) for h in FORECAST_HORIZONS]
    forecast_date_labels = ["今天"] + [d.strftime("%m-%d") for d in forecast_dates]

    colors = ["#2196F3", "#FF9800", "#E91E63", "#4CAF50", "#9C27B0"]
    directions = ["▲ 看涨" if probas[h] > 0.5 else "▼ 看跌" for h in FORECAST_HORIZONS]

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.3,
                          height_ratios=[2.5, 1.5])

    # =================== 图A: 前向预测曲线 ===================
    ax_main = fig.add_subplot(gs[0, :])
    x_pos = [0, 5, 10, 20, 30, 60]
    probs_list = [0.5] + [probas[h] for h in FORECAST_HORIZONS]
    dir_list = [""] + directions

    ax_main.plot(x_pos, probs_list, "o-", color="#333333", linewidth=2.5,
                 markersize=10, markerfacecolor="white", markeredgewidth=2,
                 zorder=5)
    for i, (x, y) in enumerate(zip(x_pos, probs_list)):
        c = "#888888" if i == 0 else colors[i - 1]
        ax_main.plot(x, y, "o", color=c, markersize=14, zorder=6)
        if i > 0:
            ax_main.annotate(f"{y:.0%}",
                            xy=(x, y), xytext=(x, y + 0.08),
                            fontsize=14, fontweight="bold", color=c,
                            ha="center", va="bottom")
            ax_main.annotate(dir_list[i],
                            xy=(x, y), xytext=(x, y - 0.08),
                            fontsize=10, color=c,
                            ha="center", va="top")

    ax_main.fill_between(x_pos, 0.5, probs_list, alpha=0.12, color="#4CAF50")
    ax_main.axhline(y=0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.5,
                    label="多空分界线")
    ax_main.set_xticks(x_pos)
    ax_main.set_xticklabels(forecast_date_labels, fontsize=11)
    ax_main.set_ylim(0, 1)
    ax_main.set_ylabel("上涨概率", fontsize=13)
    ax_main.set_title(f"玉米期货前向预测曲线（基准日: {latest_date.strftime('%Y-%m-%d')}  当前价: {close_price:.0f}）",
                      fontsize=14, fontweight="bold")
    ax_main.legend(fontsize=9, loc="upper right")
    ax_main.grid(True, alpha=0.25)

    # =================== 图B: 周期柱状对比 ===================
    ax_bar = fig.add_subplot(gs[1, 0])
    bar_labels = [f"T+{h}天" for h in FORECAST_HORIZONS]
    bar_values = [probas[h] for h in FORECAST_HORIZONS]
    bar_colors_fill = [
        "#4CAF50" if v > 0.5 else "#F44336" for v in bar_values
    ]
    bars = ax_bar.bar(bar_labels, bar_values, color=bar_colors_fill,
                      edgecolor="white", linewidth=1.5, alpha=0.9)
    for bar, v, d in zip(bars, bar_values, directions):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{v:.0%}", ha="center", fontsize=13, fontweight="bold")
        ax_bar.text(bar.get_x() + bar.get_width() / 2, 0.05,
                    d.replace("▲ ", "").replace("▼ ", ""),
                    ha="center", fontsize=10, color="white", fontweight="bold")
    ax_bar.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_ylabel("上涨概率", fontsize=11)
    ax_bar.set_title("各周期预测概率", fontsize=12, fontweight="bold")
    ax_bar.grid(True, alpha=0.2, axis="y")

    # =================== 图C: 预测摘要表 ===================
    ax_table = fig.add_subplot(gs[1, 1])
    ax_table.axis("off")
    summary_data = []
    summary_data.append(["周期", "预测日期", "上涨概率", "方向", "置信度"])
    for h, d in zip(FORECAST_HORIZONS, forecast_dates):
        p = probas[h]
        conf_level = abs(p - 0.5) / 0.5
        if conf_level > 0.8:
            conf_text = "强"
        elif conf_level > 0.4:
            conf_text = "中"
        else:
            conf_text = "弱"
        summary_data.append([
            f"T+{h}天",
            d.strftime("%Y-%m-%d"),
            f"{p:.1%}",
            "▲ 看涨" if p > 0.5 else "▼ 看跌",
            conf_text,
        ])

    table = ax_table.table(
        cellText=summary_data[1:],
        colLabels=summary_data[0],
        cellLoc="center",
        loc="center",
        colWidths=[0.12, 0.22, 0.15, 0.12, 0.10],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)
    for key, cell in table.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        if key[0] == 0:
            cell.set_fontsize(10)
            cell.set_facecolor("#EEEEEE")
            cell.set_text_props(weight="bold")
        else:
            row_idx = key[0] - 1
            if probas[FORECAST_HORIZONS[row_idx]] > 0.5:
                cell.set_facecolor("#E8F5E9")
            else:
                cell.set_facecolor("#FFEBEE")
    ax_table.set_title("预测摘要", fontsize=12, fontweight="bold", y=1.02)

    fig.suptitle("玉米期货多周期前向预测",
                 fontsize=15, fontweight="bold", y=0.99)

    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [前向预测] 已保存 -> {save_path}")

    latest_date_str = latest_date.strftime("%Y-%m-%d")
    print(f"\n  {'='*54}")
    print(f"  玉米期货前向预测 ({latest_date_str})")
    print(f"  {'='*54}")


def train_and_predict_daily(df, feature_cols, core_features, partial_features,
                            ai_weights, X_latest):
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation

    train_cutoff = pd.Timestamp("2020-06-01")
    test_cutoff = pd.Timestamp("2025-06-01")

    base_params = {
        "n_estimators": 300, "learning_rate": 0.03, "max_depth": 4,
        "num_leaves": 16, "min_child_samples": 50, "subsample": 0.8,
        "colsample_bytree": 0.7, "reg_alpha": 0.1, "reg_lambda": 1.0,
        "is_unbalance": True, "random_state": 42, "verbose": -1,
    }

    daily_probas = {}
    models_daily = {}

    for h in DAILY_HORIZONS:
        label_col = f"future_return_{h}d"
        if label_col not in df.columns:
            df[label_col] = df["close"].shift(-h) / df["close"] - 1
        lb_col = f"label_daily_{h}d"
        df[lb_col] = (df[label_col] > 0).astype(int)

        mask = df[lb_col].notna()
        for col in core_features:
            mask &= df[col].notna()
        clean = df[mask].copy()
        for col in partial_features:
            clean[col] = clean[col].fillna(0.0)

        train_mask = clean["date"] < train_cutoff
        val_mask = (clean["date"] >= train_cutoff) & (clean["date"] < test_cutoff)

        X_tr = clean.loc[train_mask, feature_cols].values.astype(np.float64)
        y_tr = clean.loc[train_mask, lb_col].values.astype(np.int64)
        X_val = clean.loc[val_mask, feature_cols].values.astype(np.float64)
        y_val = clean.loc[val_mask, lb_col].values.astype(np.int64)

        if len(X_tr) < 50:
            daily_probas[h] = 0.5
            continue

        sw_tr, _, _ = compute_ai_sample_weight(X_tr, feature_cols, ai_weights)

        m = LGBMClassifier(**base_params)
        try:
            m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="auc",
                  sample_weight=sw_tr,
                  callbacks=[early_stopping(30, verbose=False), log_evaluation(0)])
        except Exception:
            m.fit(X_tr, y_tr, sample_weight=sw_tr)

        try:
            proba = m.predict_proba(X_latest)[0, 1]
            daily_probas[h] = float(proba)
        except Exception:
            daily_probas[h] = 0.5
        models_daily[h] = m

    return daily_probas, models_daily


def plot_daily_forecast_curves(daily_probas, latest_date, close_price, save_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("  [逐日预测] matplotlib 不可用，跳过")
        return

    try:
        plt.rcParams["font.sans-serif"] = ["SimHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    groups = [(5, "#2196F3", "T+1~5天 预测"), (10, "#FF9800", "T+1~10天 预测"), (20, "#E91E63", "T+1~20天 预测")]
    latest_pd = pd.Timestamp(latest_date)

    fig = plt.figure(figsize=(16, 10))

    # =================== 图A: 三条折线合并 ===================
    ax_all = fig.add_subplot(2, 1, 1)
    for max_days, color, label in groups:
        x_days = list(range(1, max_days + 1))
        y_probs = [daily_probas.get(d, 0.5) for d in x_days]
        dates = [latest_pd + pd.Timedelta(days=d) for d in x_days]
        date_labels = [d.strftime("%m-%d") for d in dates]
        ax_all.plot(x_days, y_probs, "o-", color=color, linewidth=2.2,
                    markersize=7, label=label, alpha=0.85)
        for dx, dy, dl in zip(x_days, y_probs, date_labels):
            offset = 1.2 if dx == max_days else 0
            if offset:
                ax_all.annotate(f"{dy:.0%}",
                               xy=(dx, dy), xytext=(dx + 0.3, dy),
                               fontsize=9, fontweight="bold", color=color,
                               va="center")
        if max_days == 20:
            for dx, dy, dl in zip(x_days, y_probs, date_labels):
                if dx % 5 == 0 and dx < 20:
                    ax_all.axvline(x=dx, color="#CCCCCC", linestyle=":", linewidth=0.6)
                    ax_all.text(dx, 0.04, dl, fontsize=7, color="#666666",
                               ha="center", rotation=90, va="bottom")

    ax_all.axhline(y=0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.5)
    ax_all.set_xticks(range(1, 21))
    ax_all.set_xlim(0.5, 20.5)
    ax_all.set_ylim(0, 1)
    ax_all.set_xlabel("预测天数", fontsize=12)
    ax_all.set_ylabel("上涨概率", fontsize=12)
    ax_all.set_title(f"玉米期货逐日预测折线（基准日: {latest_date.strftime('%Y-%m-%d')}  当前价: {close_price:.0f}）",
                     fontsize=13, fontweight="bold")
    ax_all.legend(fontsize=10, loc="upper right")
    ax_all.grid(True, alpha=0.2)

    # =================== 图B: 三面板分开展示 ===================
    gs = fig.add_gridspec(3, 1, hspace=0.55)
    for gi, (max_days, color, title) in enumerate(groups):
        ax = fig.add_subplot(gs[gi, 0])
        x_days = list(range(1, max_days + 1))
        y_probs = [daily_probas.get(d, 0.5) for d in x_days]
        dates = [latest_pd + pd.Timedelta(days=d) for d in x_days]

        is_up = [p > 0.5 for p in y_probs]
        bar_colors = ["#4CAF50" if u else "#F44336" for u in is_up]

        ax.bar(x_days, y_probs, color=bar_colors, width=0.6,
               edgecolor="white", linewidth=0.8, alpha=0.9)

        for dx, dy, d in zip(x_days, y_probs, dates):
            label = f"{dy:.0%}"
            ax.text(dx, dy + 0.03, label, ha="center", fontsize=8,
                    fontweight="bold", color=color)
            date_label = d.strftime("%m-%d")
            direction = "▲涨" if dy > 0.5 else "▼跌"
            ax.text(dx, 0.06, f"{date_label}\n{direction}", ha="center",
                    fontsize=7, color="#666666", linespacing=1.2)

        ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xticks(x_days)
        ax.set_xlim(0.5, max_days + 0.5)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("上涨概率", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold", color=color)
        ax.grid(True, alpha=0.15, axis="y")

        up_count = sum(1 for p in y_probs if p > 0.5)
        avg_prob = np.mean(y_probs)
        trend = "偏多" if avg_prob > 0.52 else ("偏空" if avg_prob < 0.48 else "中性")
        ax.text(0.98, 0.95, f"看涨 {up_count}/{max_days}  均值 {avg_prob:.0%}  {trend}",
                transform=ax.transAxes, fontsize=8, color="#666666",
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5", alpha=0.8))

    fig.suptitle("玉米期货逐日预测折线图",
                 fontsize=15, fontweight="bold", y=0.99)

    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [逐日预测] 已保存 -> {save_path}")

    latest_date_str = latest_date.strftime("%Y-%m-%d")
    print(f"\n  {'='*60}")
    print(f"  玉米期货逐日预测详情 ({latest_date_str})")
    print(f"  {'='*60}")
    for max_days, color, title in groups:
        print(f"\n  ┌ {title}")
        for d in range(1, max_days + 1):
            p = daily_probas.get(d, 0.5)
            dt = latest_pd + pd.Timedelta(days=d)
            bar = "█" * int(p * 20 + 1)
            print(f"  │ T+{d:>2d}天 ({dt.strftime('%m-%d')}): {p:.1%} {bar} {'▲看涨' if p > 0.5 else '▼看跌'}")
        up_n = sum(1 for d in range(1, max_days + 1) if daily_probas.get(d, 0.5) > 0.5)
        avg_p = np.mean([daily_probas.get(d, 0.5) for d in range(1, max_days + 1)])
        print(f"  └ {up_n}/{max_days}天看涨, 均值={avg_p:.0%}")
    print(f"  {'='*60}")


def save_predictions_csv(all_dates, all_probas, all_actuals, all_returns, thresholds, save_path):
    n_max = max(len(all_dates[h]) for h in FORECAST_HORIZONS)
    rows = []
    for i in range(n_max):
        row = {}
        for h in FORECAST_HORIZONS:
            d = all_dates[h]
            if i < len(d):
                row["date"] = pd.Timestamp(d[i]).strftime("%Y-%m-%d")
                break
        for h in FORECAST_HORIZONS:
            thresh = thresholds.get(h, 0.5)
            if i < len(all_probas[h]):
                row[f"prob_T+{h}d"] = round(float(all_probas[h][i]), 6)
                row[f"label_T+{h}d"] = int(all_actuals[h][i])
                row[f"return_T+{h}d"] = round(float(all_returns[h][i]), 6)
                pred_dir = 1 if all_probas[h][i] >= thresh else 0
                row[f"pred_T+{h}d"] = pred_dir
            else:
                row[f"prob_T+{h}d"] = np.nan
                row[f"label_T+{h}d"] = np.nan
                row[f"return_T+{h}d"] = np.nan
                row[f"pred_T+{h}d"] = np.nan
        rows.append(row)
    df_out = pd.DataFrame(rows)
    df_out.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"  [预测CSV] 已保存 -> {save_path}")


def main():
    from build_features import FEATURE_COLS, CORE_FEATURES, PARTIAL_FEATURES, NOISE_THRESHOLD

    fast_mode = any(a == "--fast" for a in sys.argv)
    if fast_mode:
        print("  [快速模式] 跳过滚动回测 + 逐日模型")

    if not any(a == "--skip-update" for a in sys.argv):
        from update_data import update_all
        update_all()
        local_build_path = os.path.join(DATA_DIR, "build_features.py")
        if os.path.exists(local_build_path):
            print("\n  [自动重建特征] 数据已更新，重新计算特征...")
            import subprocess
            subprocess.run([sys.executable, local_build_path, "--skip-update"], check=False)

    df = pd.read_csv(os.path.join(DATA_DIR, "features_full_v2.csv"), parse_dates=["date"])

    ai_weights = load_ai_weights()
    use_ai = len(ai_weights) > 0

    print("=" * 66)
    print("  LightGBM v5 — AI权重接入 + 超参搜索 + 概率校准")
    print("=" * 66)
    print(f"特征: {len(FEATURE_COLS)} (核心{len(CORE_FEATURES)} + 部分{len(PARTIAL_FEATURES)})")
    print(f"三分类阈值: |return| > {NOISE_THRESHOLD:.1%}")
    print(f"超参搜索: {N_HYPER_SEARCH}轮")
    print(f"AI权重: {'已接入 (sample_weight)' if use_ai else '未接入 (均匀权重)'}")
    if use_ai:
        top_weights = sorted(ai_weights.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"  Top5: {' | '.join(f'{k}={v:.2%}' for k, v in top_weights)}")

    summary_rows = []
    all_test_probas = {}
    all_test_dates = {}
    all_test_actuals = {}
    all_test_returns = {}
    all_thresholds = {}
    models = {}

    CORE_FEATURES_REF = CORE_FEATURES

    for horizon in FORECAST_HORIZONS:
        print(f"\n{'='*66}")
        print(f"  ◆ T+{horizon}d")
        print(f"{'='*66}")

        X_train, X_val, X_test, y_train, y_val, y_test = load_features(horizon)
        print(f"样本: train={len(X_train)} val={len(X_val)} test={len(X_test)}")
        print(f"涨跌比: train={y_train.mean():.1%} val={y_val.mean():.1%} test={y_test.mean():.1%}")

        sw_train, ref_med, ref_mad = compute_ai_sample_weight(
            X_train, FEATURE_COLS, ai_weights)
        if use_ai:
            sw_val, _, _ = compute_ai_sample_weight(
                X_val, FEATURE_COLS, ai_weights, ref_med, ref_mad)
            sw_test, _, _ = compute_ai_sample_weight(
                X_test, FEATURE_COLS, ai_weights, ref_med, ref_mad)
            print(f"AI sample_weight: train[{sw_train.min():.2f}~{sw_train.max():.2f}] "
                  f"val[{sw_val.min():.2f}~{sw_val.max():.2f}] "
                  f"test[{sw_test.min():.2f}~{sw_test.max():.2f}]")
        else:
            sw_train = sw_val = sw_test = None

        print("\n  [超参搜索...]")
        best_params = random_hyper_search(
            X_train, y_train, X_val, y_val, N_HYPER_SEARCH, sw_train)

        print("  [训练...]")
        model = train_lightgbm(X_train, y_train, X_val, y_val,
                               best_params, sw_train)
        models[horizon] = model
        print(f"  最佳迭代: {model.best_iteration_}")

        val_proba_raw = model.predict_proba(X_val)[:, 1]
        test_proba_raw = model.predict_proba(X_test)[:, 1]

        proba_cal = calibrate_predict(model, X_val, y_val, X_test, sw_val)

        opt_thresh = find_optimal_threshold(y_val, val_proba_raw)
        all_thresholds[horizon] = opt_thresh
        print(f"  最优阈值: {opt_thresh:.3f} (验证集均衡准确率最优)")

        t_dates, t_actuals, t_returns = extract_test_dates_and_actuals(df, horizon, CORE_FEATURES_REF)
        all_test_dates[horizon] = t_dates
        min_len = min(len(proba_cal), len(t_dates))
        all_test_probas[horizon] = proba_cal[:min_len]
        all_test_actuals[horizon] = t_actuals[:min_len]
        all_test_returns[horizon] = t_returns[:min_len]

        print(f"\n  {'集合':<6s} {'样本':>6s} {'阈值':>6s} {'准确率':>7s} {'精确率':>7s} {'召回率':>7s} {'F1':>7s} {'AUC':>7s} {'预测涨%':>8s}")
        print(f"  {'─'*66}")
        for X, y, name in [(X_train, y_train, "训练"), (X_val, y_val, "验证")]:
            m = evaluate(model, X, y, threshold=opt_thresh)
            print(f"  {name:<6s} {m['n']:>6d} {m['thresh']:>5.2f} {m['acc']:>6.2%} {m['prec']:>6.2%} {m['rec']:>6.2%} {m['f1']:>7.4f} {m['auc']:>7.4f} {m['pred_up']:>7.1%}")
        m = evaluate(model, X_test, y_test, proba=proba_cal[:min_len], threshold=opt_thresh)
        print(f"  {'测试':<6s} {m['n']:>6d} {m['thresh']:>5.2f} {m['acc']:>6.2%} {m['prec']:>6.2%} {m['rec']:>6.2%} {m['f1']:>7.4f} {m['auc']:>7.4f} {m['pred_up']:>7.1%}")
        summary_rows.append({"Horizon": f"T+{horizon}d", "thresh": opt_thresh, **{k: m[k] for k in ["n", "acc", "auc", "prec", "rec", "f1"]}})

        imp = model.feature_importances_
        imp = imp / imp.sum()
        top = np.argsort(imp)[::-1][:8]
        print(f"\n  特征重要性:")
        for rank, idx in enumerate(top, 1):
            bar = "█" * int(imp[idx] * 35 + 1)
            ai_w = ai_weights.get(FEATURE_COLS[idx], 0) if ai_weights else 0
            ai_tag = f" [AI={ai_w:.1%}]" if ai_w > 0 else ""
            print(f"    {rank}. {FEATURE_COLS[idx]:<24s} {imp[idx]:.1%}{ai_tag}  {bar}")

        from sklearn.metrics import roc_auc_score, brier_score_loss
        raw_proba = model.predict_proba(X_test)[:, 1]
        cal_auc = roc_auc_score(y_test, proba_cal)
        cal_brier = brier_score_loss(y_test, proba_cal)
        raw_auc = roc_auc_score(y_test, raw_proba)
        raw_brier = brier_score_loss(y_test, raw_proba)
        if not np.allclose(raw_proba, proba_cal, atol=1e-5):
            print(f"\n  概率校准: raw AUC={raw_auc:.4f} Brier={raw_brier:.4f} -> cal AUC={cal_auc:.4f} Brier={cal_brier:.4f}")
        else:
            print(f"\n  概率校准: 跳过 (AUC={raw_auc:.4f} Brier={raw_brier:.4f})")

        uncertain = (proba_cal > 0.4) & (proba_cal < 0.6)
        confident_mask = ~uncertain
        if confident_mask.sum() > 5:
            from sklearn.metrics import accuracy_score
            conf_acc = accuracy_score(y_test[confident_mask], (proba_cal[confident_mask] >= opt_thresh).astype(int))
            print(f"  置信区间: {uncertain.sum()}/{len(y_test)} = {uncertain.mean():.0%} 不确定 -> 剔除后准确率 {conf_acc:.1%}")

        if not fast_mode:
            print(f"\n  [滚动回测...]")
            label_col = f"label_binary_{horizon}d"
            mask = df[label_col].notna() & (df[label_col] != -1)
            for col in CORE_FEATURES_REF:
                mask &= df[col].notna()
            clean = df[mask].copy()
            for col in PARTIAL_FEATURES:
                clean[col] = clean[col].fillna(0.0)
            X_all = clean[FEATURE_COLS].values.astype(np.float64)
            y_all = clean[label_col].values.astype(np.int64)
            bt = rolling_backtest(X_all, y_all, clean["date"],
                                  feature_names=FEATURE_COLS if use_ai else None,
                                  ai_weights=ai_weights if use_ai else None)
            if not bt.empty:
                r12 = bt.tail(12)
                print(f"  {len(bt)}个月 | 均值={bt['acc'].mean():.2%} ±{bt['acc'].std():.3f} | 近12月={r12['acc'].mean():.2%}")

    print(f"\n{'='*66}")
    print(f"  多目标汇总 (AI权重 + 校准)")
    print(f"{'='*66}")
    print(f"  {'目标':<10s} {'阈值':>6s} {'样本':>6s} {'准确率':>7s} {'AUC':>7s} {'精确率':>7s} {'召回率':>7s} {'F1':>7s}")
    print(f"  {'─'*60}")
    for s in summary_rows:
        print(f"  {s['Horizon']:<10s} {s['thresh']:>5.2f} {s['n']:>6d} {s['acc']:>6.2%} {s['auc']:>7.4f} {s['prec']:>6.2%} {s['rec']:>6.2%} {s['f1']:>7.4f}")

    latest_summary = []
    for h in FORECAST_HORIZONS:
        thresh = all_thresholds.get(h, 0.5)
        if h in all_test_probas and len(all_test_probas[h]) > 0:
            latest_summary.append(
                f"T+{h}d={all_test_probas[h][-1]:.1%}"
                f"({'▲看涨' if all_test_probas[h][-1] >= thresh else '▼看跌'})")
    if latest_summary:
        print(f"\n  最新预测: {' | '.join(latest_summary)}")

    print(f"\n  [生成预测曲线...]")
    curve_path = os.path.join(DATA_DIR, "prediction_curve.png")
    plot_prediction_curves(all_test_dates, all_test_probas,
                           all_test_actuals, all_test_returns, curve_path)

    csv_path = os.path.join(DATA_DIR, "predictions_detail.csv")
    save_predictions_csv(all_test_dates, all_test_probas,
                         all_test_actuals, all_test_returns, all_thresholds, csv_path)

    print(f"\n  [生成前向预测...]")
    X_latest, latest_date, close_price = get_latest_features(
        df, FEATURE_COLS, CORE_FEATURES_REF)
    print(f"  基准日: {latest_date.strftime('%Y-%m-%d')}  收盘价: {close_price:.0f}")
    forward_path = os.path.join(DATA_DIR, "forward_forecast.png")
    plot_forward_forecast(models, latest_date, close_price, X_latest, forward_path)

    if not fast_mode:
        print(f"\n  [生成逐日预测折线（1~20天）...]")
        print(f"  正在训练 {len(DAILY_HORIZONS)} 个逐日模型...")
        daily_probas, _ = train_and_predict_daily(
            df, FEATURE_COLS, CORE_FEATURES_REF, PARTIAL_FEATURES,
            ai_weights, X_latest)
        daily_path = os.path.join(DATA_DIR, "daily_forecast.png")
        plot_daily_forecast_curves(daily_probas, latest_date, close_price, daily_path)

    print(f"\n  完成 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*66}")


if __name__ == "__main__":
    main()
