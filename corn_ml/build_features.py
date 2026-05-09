#!/usr/bin/env python3
"""特征工程v3：6站独立气候 + 5个交叉特征 + T+5/10/20多目标"""

import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

ANOMALY_WINDOW_YEARS = 5
FORECAST_HORIZONS = [5, 10, 20]

CLIMATE_STATIONS = [
    "哈尔滨", "长春", "沈阳",
    "呼和浩特", "济南", "郑州",
]

CLIMATE_REGIONS = {
    "东北": ["哈尔滨", "长春", "沈阳"],
    "华北黄淮": ["呼和浩特", "济南", "郑州"],
}


def load_and_align() -> pd.DataFrame:
    futures = pd.read_csv(
        os.path.join(DATA_DIR, "corn_futures_main.csv"), parse_dates=["date"]
    )
    cbot_corn = pd.read_csv(
        os.path.join(DATA_DIR, "corn_cbot.csv"), parse_dates=["date"]
    )
    cbot_wheat = pd.read_csv(
        os.path.join(DATA_DIR, "cbot_wheat.csv"), parse_dates=["date"]
    )
    spot_basis = pd.read_csv(
        os.path.join(DATA_DIR, "corn_spot_basis.csv"), parse_dates=["date"]
    )
    climate = pd.read_csv(
        os.path.join(DATA_DIR, "corn_climate_daily.csv"), dtype={"date": str}
    )
    hog = pd.read_csv(os.path.join(DATA_DIR, "hog_price_daily.csv"))

    futures["date"] = pd.to_datetime(futures["date"])
    cbot_corn["date"] = pd.to_datetime(cbot_corn["date"])
    cbot_wheat["date"] = pd.to_datetime(cbot_wheat["date"])
    spot_basis["date"] = pd.to_datetime(spot_basis["date"])
    hog["date"] = pd.to_datetime(hog["date"])

    climate["date_dt"] = pd.to_datetime(climate["date"], format="%Y%m%d")

    # 按站 pivot: 每站独立气温/降水列
    climate_pivoted = climate.pivot_table(
        index="date_dt", columns="station",
        values=["temp_C", "precip_mm"], aggfunc="mean",
    )
    climate_pivoted.columns = [
        f"{col[0]}_{col[1]}" for col in climate_pivoted.columns
    ]
    climate_pivoted = climate_pivoted.reset_index().rename(columns={"date_dt": "date"})

    merged = futures.merge(
        cbot_corn.rename(columns={c: f"cbot_{c}" for c in ["open", "high", "low", "close", "volume"]}),
        on="date", how="left",
    )
    merged = merged.merge(
        cbot_wheat.rename(columns={c: f"wheat_{c}" for c in ["open", "high", "low", "close", "volume"]}),
        on="date", how="left",
    )
    merged = merged.merge(spot_basis, on="date", how="left")
    merged = merged.merge(climate_pivoted, on="date", how="left")
    merged = merged.merge(hog[["date", "hog_price"]], on="date", how="left")

    merged = merged.sort_values("date").reset_index(drop=True)
    return merged


def _station_anomaly(df: pd.DataFrame, col_name: str) -> pd.Series:
    """对单个站的气候列计算过去 ANOMALY_WINDOW_YEARS 年均值异常，无未来函数"""
    df = df.copy()
    df["doy"] = df["date"].dt.dayofyear
    df["year"] = df["date"].dt.year

    pivot = df.dropna(subset=[col_name]).pivot_table(
        index="doy", columns="year", values=col_name, aggfunc="mean"
    )

    anomalies: dict[tuple, float] = {}
    for doy in pivot.index:
        years = sorted(pivot.columns)
        for i, yr in enumerate(years):
            val = pivot.loc[doy, yr]
            if pd.isna(val):
                continue
            past_vals = [
                pivot.loc[doy, y]
                for y in years[max(0, i - ANOMALY_WINDOW_YEARS):i]
                if pd.notna(pivot.loc[doy, y])
            ]
            if past_vals:
                anomalies[(doy, yr)] = val - (sum(past_vals) / len(past_vals))

    result = df.apply(
        lambda r: anomalies.get((r["doy"], r["year"]), np.nan), axis=1
    )
    return result


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ==================== 价格动量 ====================
    df["return_1d"] = df["close"].pct_change()
    df["momentum_5d"] = df["close"].pct_change(5)
    df["momentum_20d"] = df["close"].pct_change(20)
    df["momentum_60d"] = df["close"].pct_change(60)
    df["volatility"] = df["return_1d"].rolling(20).std()

    # ==================== 成交持仓 ====================
    df["oi_change"] = df["open_interest"].pct_change(5).replace([np.inf, -np.inf], np.nan)
    df["volume_change"] = df["volume"].pct_change(5).replace([np.inf, -np.inf], np.nan)

    # ==================== 技术指标 ====================
    df["ma_20"] = df["close"].rolling(20).mean()
    df["price_vs_ma20"] = df["close"] / df["ma_20"] - 1
    df["rsi_14"] = _compute_rsi(df["close"], 14)
    df["hl_ratio"] = (df["high"] - df["low"]) / df["close"]

    # ==================== 基差 ====================
    df["basis"] = df["spot_price"] - df["dominant_contract_price"]

    # ==================== CBOT 联动 ====================
    df["cbot_return_5d"] = df["cbot_close"].pct_change(5)
    df["corn_wheat_ratio"] = df["cbot_close"] / df["wheat_close"].replace(0, np.nan)

    # ==================== 生猪 ====================
    df["hog_price_lag1"] = df["hog_price"].shift(1)
    df["hog_mom"] = df["hog_price"].pct_change(20)
    df["hog_mom_lag1"] = df["hog_mom"].shift(1)

    # ==================== 按区域独立气候异常 ====================
    for region_name, stations in CLIMATE_REGIONS.items():
        t_cols = [f"temp_C_{s}" for s in stations if f"temp_C_{s}" in df.columns]
        p_cols = [f"precip_mm_{s}" for s in stations if f"precip_mm_{s}" in df.columns]
        if t_cols:
            df[f"temp_region_{region_name}"] = df[t_cols].mean(axis=1)
            df[f"t_anom_{region_name}"] = _station_anomaly(
                df[["date", f"temp_region_{region_name}"]].rename(
                    columns={f"temp_region_{region_name}": f"temp_region_{region_name}"}
                ), f"temp_region_{region_name}"
            )
        if p_cols:
            df[f"precip_region_{region_name}"] = df[p_cols].mean(axis=1)
            df[f"p_anom_{region_name}"] = _station_anomaly(
                df[["date", f"precip_region_{region_name}"]].rename(
                    columns={f"precip_region_{region_name}": f"precip_region_{region_name}"}
                ), f"precip_region_{region_name}"
            )

    # 前向填充
    region_anom_cols = []
    for region_name in CLIMATE_REGIONS:
        for prefix in ["t_anom_", "p_anom_"]:
            col = f"{prefix}{region_name}"
            if col in df.columns:
                df[col] = df[col].ffill()
                region_anom_cols.append(col)

    # 清理中间列
    drop_cols = [c for c in df.columns if c.startswith("temp_region_") or c.startswith("precip_region_")]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # ==================== 5个交叉特征 ====================
    # 1. 复合天气压力: 高温 + 少雨 同时发生 → 比任一单独更严重
    t_cols = [f"t_anom_{r}" for r in CLIMATE_REGIONS if f"t_anom_{r}" in df.columns]
    p_cols = [f"p_anom_{r}" for r in CLIMATE_REGIONS if f"p_anom_{r}" in df.columns]
    df["t_anom_avg"] = df[t_cols].mean(axis=1) if t_cols else 0
    df["p_anom_avg"] = df[p_cols].mean(axis=1) if p_cols else 0
    df["weather_stress"] = df["t_anom_avg"] * (-df["p_anom_avg"])  # 高温×少雨 = 干旱压力

    # 2. 天气 × 资金: 坏天气 + 资金撤退 = 加速下跌
    df["oi_change_lag1"] = df["oi_change"].shift(1)
    df["precip_x_oi"] = df["p_anom_avg"] * df["oi_change_lag1"]

    # 3. 波动率 × 基差: 高波动 + 高基差 = 逼仓
    df["basis_lag1"] = df["basis"].shift(1)
    df["vol_x_basis"] = df["volatility"] * df["basis_lag1"]

    # 4. 玉米/小麦比价趋势
    df["ratio_change_20d"] = df["corn_wheat_ratio"].pct_change(20)

    # 5. 趋势 × 持仓: 下跌 + 增仓 = 空头入场
    df["mom_x_oi"] = df["momentum_20d"] * df["oi_change"]

    # ==================== 清理临时列 ====================
    df.drop(columns=["t_anom_avg", "p_anom_avg", "oi_change_lag1", "basis_lag1"], inplace=True)

    # ==================== 多目标标签 ====================
    for horizon in FORECAST_HORIZONS:
        df[f"future_return_{horizon}d"] = (
            df["close"].shift(-horizon) / df["close"] - 1
        )
        df[f"label_{horizon}d"] = (df[f"future_return_{horizon}d"] > 0).astype(int)

    return df


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


STATION_T_ANOM = [f"t_anom_{r}" for r in CLIMATE_REGIONS]
STATION_P_ANOM = [f"p_anom_{r}" for r in CLIMATE_REGIONS]

FEATURE_COLS = [
    "momentum_5d",
    "momentum_20d",
    "momentum_60d",         # ★ 季度动量
    "volatility",
    "oi_change",
    "volume_change",
    "price_vs_ma20",
    "rsi_14",
    "hl_ratio",
    "basis",
    "cbot_return_5d",
    "corn_wheat_ratio",
    "hog_price_lag1",
    "hog_mom_lag1",
    *STATION_T_ANOM,
    *STATION_P_ANOM,
    "weather_stress",
    "precip_x_oi",
    "vol_x_basis",
    "ratio_change_20d",
    "mom_x_oi",
]

CORE_FEATURES = [
    "momentum_5d", "momentum_20d", "momentum_60d",
    "volatility", "oi_change", "volume_change",
    "price_vs_ma20", "rsi_14", "hl_ratio",
]
for r in CLIMATE_REGIONS:
    CORE_FEATURES.append(f"t_anom_{r}")
    CORE_FEATURES.append(f"p_anom_{r}")

PARTIAL_FEATURES = [
    "basis", "cbot_return_5d", "corn_wheat_ratio",
    "hog_price_lag1", "hog_mom_lag1",
    "weather_stress", "precip_x_oi", "vol_x_basis",
    "ratio_change_20d", "mom_x_oi",
]


def clean_and_split(
    df: pd.DataFrame,
    horizon: int,
    train_months: int = 60,
    test_cutoff: str = "2025-06-01",
) -> tuple:
    from datetime import datetime as dt
    test_dt = dt.strptime(test_cutoff, "%Y-%m-%d")
    train_dt = test_dt.replace(year=test_dt.year - train_months // 12)
    if train_dt.month != test_dt.month:
        train_dt = train_dt.replace(month=test_dt.month)
    train_cutoff = train_dt.strftime("%Y-%m-%d")

    label_col = f"label_{horizon}d"
    mask = df[label_col].notna()
    for col in CORE_FEATURES:
        mask &= df[col].notna()
    clean = df[mask].copy()

    for col in PARTIAL_FEATURES:
        clean[col] = clean[col].fillna(0.0)

    train_mask = clean["date"] < train_cutoff
    val_mask = (clean["date"] >= train_cutoff) & (clean["date"] < test_cutoff)
    test_mask = clean["date"] >= test_cutoff

    X_train = clean.loc[train_mask, FEATURE_COLS].values.astype(np.float64)
    y_train = clean.loc[train_mask, label_col].values.astype(np.int64)
    X_val = clean.loc[val_mask, FEATURE_COLS].values.astype(np.float64)
    y_val = clean.loc[val_mask, label_col].values.astype(np.int64)
    X_test = clean.loc[test_mask, FEATURE_COLS].values.astype(np.float64)
    y_test = clean.loc[test_mask, label_col].values.astype(np.int64)

    return (X_train, X_val, X_test, y_train, y_val, y_test,
            clean.loc[train_mask, "date"].values,
            clean.loc[val_mask, "date"].values,
            clean.loc[test_mask, "date"].values)


def main() -> None:
    print("=" * 60)
    print("  特征工程 v3：6站独立气候 + 5交叉特征")
    print("=" * 60)

    df = load_and_align()
    print(f"数据对齐: {len(df)} 行 ({df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()})")

    df = build_features(df)
    df.to_csv(os.path.join(DATA_DIR, "features_full_v2.csv"), index=False, encoding="utf-8-sig")

    print(f"特征列 ({len(FEATURE_COLS)}):")
    print(f"  技术面: {[c for c in FEATURE_COLS if c.startswith(('momentum','volatility','oi_','volume','price','rsi','hl'))]}")
    print(f"  基差/CBOT/生猪: {[c for c in FEATURE_COLS if c in ['basis','cbot_return_5d','corn_wheat_ratio','hog_price_lag1','hog_mom_lag1']]}")
    print(f"  气候(6站×2): {[c for c in FEATURE_COLS if c.startswith(('t_anom_','p_anom_'))]}")
    print(f"  交叉特征: weather_stress, precip_x_oi, vol_x_basis, ratio_change_20d, mom_x_oi")
    print(f"标签: T+{FORECAST_HORIZONS} 日")

    for horizon in FORECAST_HORIZONS:
        (X_tr, X_val, X_te, y_tr, y_val, y_te,
         _, _, _) = clean_and_split(df, horizon)
        print(f"\n  T+{horizon}d:")
        print(f"    训练: {len(X_tr):>5d}  上涨 {y_tr.mean():.1%}")
        print(f"    验证: {len(X_val):>5d}  上涨 {y_val.mean():.1%}")
        print(f"    测试: {len(X_te):>5d}  上涨 {y_te.mean():.1%}")

        np.savez(
            os.path.join(DATA_DIR, f"features_h{horizon}.npz"),
            X_train=X_tr, X_val=X_val, X_test=X_te,
            y_train=y_tr, y_val=y_val, y_test=y_te,
        )

    print(f"\n保存: features_full_v2.csv + features_h*.npz")
    print("=" * 60)


if __name__ == "__main__":
    main()
