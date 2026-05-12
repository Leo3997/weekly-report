#!/usr/bin/env python3
"""特征工程v5：精简特征 + 生长阶段交互 + 三分类标签 + 回归目标"""

import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

ANOMALY_WINDOW_YEARS = 5
FORECAST_HORIZONS = [5, 10, 20]
NOISE_THRESHOLD = 0.015

CLIMATE_STATIONS = [
    "哈尔滨", "长春", "沈阳",
    "呼和浩特", "济南", "郑州",
]

CLIMATE_REGIONS = {
    "东北": ["哈尔滨", "长春", "沈阳"],
    "华北黄淮": ["呼和浩特", "济南", "郑州"],
}


def load_and_align() -> pd.DataFrame:
    futures = pd.read_csv(os.path.join(DATA_DIR, "corn_futures_main.csv"), parse_dates=["date"])
    cbot_corn = pd.read_csv(os.path.join(DATA_DIR, "corn_cbot.csv"), parse_dates=["date"])
    cbot_wheat = pd.read_csv(os.path.join(DATA_DIR, "cbot_wheat.csv"), parse_dates=["date"])
    spot_basis = pd.read_csv(os.path.join(DATA_DIR, "corn_spot_basis.csv"), parse_dates=["date"])
    climate = pd.read_csv(os.path.join(DATA_DIR, "corn_climate_daily.csv"), dtype={"date": str})
    hog_futures = pd.read_csv(os.path.join(DATA_DIR, "hog_futures_main.csv"), parse_dates=["date"])
    satellite = pd.read_csv(os.path.join(DATA_DIR, "corn_satellite_daily.csv"), dtype={"date": str})

    futures["date"] = pd.to_datetime(futures["date"])
    cbot_corn["date"] = pd.to_datetime(cbot_corn["date"])
    cbot_wheat["date"] = pd.to_datetime(cbot_wheat["date"])
    spot_basis["date"] = pd.to_datetime(spot_basis["date"])

    climate["date_dt"] = pd.to_datetime(climate["date"], format="%Y%m%d")
    climate_pivoted = climate.pivot_table(
        index="date_dt", columns="station", values=["temp_C", "precip_mm"], aggfunc="mean",
    )
    climate_pivoted.columns = [f"{col[0]}_{col[1]}" for col in climate_pivoted.columns]
    climate_pivoted = climate_pivoted.reset_index().rename(columns={"date_dt": "date"})

    satellite["date_dt"] = pd.to_datetime(satellite["date"], format="%Y%m%d")
    sat_params = ["GWETROOT", "GWETTOP", "GWETPROF", "ALLSKY_SFC_SW_DWN", "RH2M"]
    sat_pivoted = satellite.pivot_table(
        index="date_dt", columns="station", values=sat_params, aggfunc="mean",
    )
    sat_pivoted.columns = [f"{col[0]}_{col[1]}" for col in sat_pivoted.columns]
    sat_pivoted = sat_pivoted.reset_index().rename(columns={"date_dt": "date"})

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
    merged = merged.merge(sat_pivoted, on="date", how="left")
    merged = merged.merge(
        hog_futures[["date", "close", "open_interest"]].rename(
            columns={"close": "hog_close", "open_interest": "hog_oi"}
        ), on="date", how="left",
    )

    merged = merged.sort_values("date").reset_index(drop=True)
    return merged


def _station_anomaly(df: pd.DataFrame, col_name: str) -> pd.Series:
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
                pivot.loc[doy, y] for y in years[max(0, i - ANOMALY_WINDOW_YEARS):i]
                if pd.notna(pivot.loc[doy, y])
            ]
            if past_vals:
                anomalies[(doy, yr)] = val - (sum(past_vals) / len(past_vals))

    return df.apply(lambda r: anomalies.get((r["doy"], r["year"]), np.nan), axis=1)


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ==================== 精简技术面 (6个, 原来9个) ====================
    df["return_1d"] = df["close"].pct_change()
    df["momentum_20d"] = df["close"].pct_change(20)
    df["momentum_60d"] = df["close"].pct_change(60)
    df["momentum_accel"] = df["momentum_20d"] - df["momentum_60d"]  # ★ 趋势加速
    df["volatility"] = df["return_1d"].rolling(20).std()
    df["oi_change"] = df["open_interest"].pct_change(5).replace([np.inf, -np.inf], np.nan)
    df["price_vs_ma20"] = df["close"] / df["close"].rolling(20).mean() - 1
    df["rsi_14"] = _compute_rsi(df["close"], 14)
    df["hl_ratio"] = (df["high"] - df["low"]) / df["close"]

    # ==================== 基差 (1个) ====================
    df["basis"] = df["spot_price"] - df["dominant_contract_price"]

    # ==================== CBOT (2个) ====================
    df["cbot_return_5d"] = df["cbot_close"].pct_change(5)
    df["corn_wheat_ratio"] = df["cbot_close"] / df["wheat_close"].replace(0, np.nan)

    # ==================== 生猪期货 (2个) ====================
    df["hog_return_20d"] = df["hog_close"].pct_change(20)
    df["corn_hog_corr"] = df["close"].rolling(60).corr(df["hog_close"])

    # ==================== 季节 (5个) ====================
    df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)
    df["is_critical"] = df["date"].dt.month.isin([7, 8]).astype(float)   # 抽穗+灌浆
    df["is_planting"] = df["date"].dt.month.isin([4, 5]).astype(float)    # 播种+出苗
    df["is_harvest"] = df["date"].dt.month.isin([9, 10]).astype(float)    # 收获

    # ==================== 按区域气候异常 (4个) ====================
    for region_name, stations in CLIMATE_REGIONS.items():
        t_cols = [f"temp_C_{s}" for s in stations if f"temp_C_{s}" in df.columns]
        p_cols = [f"precip_mm_{s}" for s in stations if f"precip_mm_{s}" in df.columns]
        if t_cols:
            df[f"_temp_{region_name}"] = df[t_cols].mean(axis=1)
            df[f"t_anom_{region_name}"] = _station_anomaly(
                df[["date", f"_temp_{region_name}"]].rename(columns={f"_temp_{region_name}": f"_temp_{region_name}"}),
                f"_temp_{region_name}",
            )
        if p_cols:
            df[f"_precip_{region_name}"] = df[p_cols].mean(axis=1)
            df[f"p_anom_{region_name}"] = _station_anomaly(
                df[["date", f"_precip_{region_name}"]].rename(columns={f"_precip_{region_name}": f"_precip_{region_name}"}),
                f"_precip_{region_name}",
            )

    for col in [c for c in df.columns if c.startswith("t_anom_") or c.startswith("p_anom_")]:
        df[col] = df[col].ffill()

    # ==================== 精简卫星特征: 根区水分 + 表层梯度 ====================
    for region_name, stations in CLIMATE_REGIONS.items():
        r_cols = [f"GWETROOT_{s}" for s in stations if f"GWETROOT_{s}" in df.columns]
        s_cols = [f"GWETTOP_{s}" for s in stations if f"GWETTOP_{s}" in df.columns]

        if r_cols:
            df[f"_gwetroot_{region_name}"] = df[r_cols].mean(axis=1)
            df[f"sm_root_anom_{region_name}"] = _station_anomaly(
                df[["date", f"_gwetroot_{region_name}"]].rename(
                    columns={f"_gwetroot_{region_name}": f"_gwetroot_{region_name}"}
                ), f"_gwetroot_{region_name}",
            )
        if r_cols and s_cols:
            df[f"_gwettop_{region_name}"] = df[s_cols].mean(axis=1)
            df[f"sm_grad_anom_{region_name}"] = _station_anomaly(
                df[["date", f"_gwettop_{region_name}"]].rename(
                    columns={f"_gwettop_{region_name}": f"_gwettop_{region_name}"}
                ), f"_gwettop_{region_name}",
            )
            df[f"sm_grad_anom_{region_name}"] = (
                df[f"sm_grad_anom_{region_name}"]
                - df[f"sm_root_anom_{region_name}"]
            )

    for col in [c for c in df.columns if c.startswith("sm_")]:
        df[col] = df[col].ffill()

    # radiation + humidity
    for param, short in [("ALLSKY_SFC_SW_DWN", "srad"), ("RH2M", "rh")]:
        for region_name, stations in CLIMATE_REGIONS.items():
            cols = [f"{param}_{s}" for s in stations if f"{param}_{s}" in df.columns]
            if cols:
                df[f"_{param}_{region_name}"] = df[cols].mean(axis=1)
                df[f"{short}_anom_{region_name}"] = _station_anomaly(
                    df[["date", f"_{param}_{region_name}"]].rename(
                        columns={f"_{param}_{region_name}": f"_{param}_{region_name}"}
                    ), f"_{param}_{region_name}",
                )
                df[f"{short}_anom_{region_name}"] = df[f"{short}_anom_{region_name}"].ffill()

    # 清理中间列
    mid_cols = [c for c in df.columns if c.startswith("_")]
    df.drop(columns=mid_cols, inplace=True, errors="ignore")

    # ==================== 生长阶段 × 气候交互 (6个) ★ ====================
    df["t_anom_avg"] = df[[c for c in df.columns if c.startswith("t_anom_")]].mean(axis=1)
    df["p_anom_avg"] = df[[c for c in df.columns if c.startswith("p_anom_")]].mean(axis=1)

    df["t_critical"] = df["t_anom_avg"] * df["is_critical"]
    df["t_planting"] = df["t_anom_avg"] * df["is_planting"]
    df["p_critical"] = df["p_anom_avg"] * df["is_critical"]
    df["p_planting"] = df["p_anom_avg"] * df["is_planting"]
    df["sm_planting_ne"] = df["sm_root_anom_东北"] * df["is_planting"]
    df["sm_planting_hb"] = df["sm_root_anom_华北黄淮"] * df["is_planting"]

    # ==================== 交叉特征 (3个) ====================
    df["precip_x_oi"] = df["p_anom_avg"] * df["oi_change"].shift(1)
    df["vol_x_basis"] = df["volatility"] * df["basis"].shift(1)
    df["mom_x_oi"] = df["momentum_20d"] * df["oi_change"]

    df.drop(columns=["t_anom_avg", "p_anom_avg"], inplace=True)

    # ==================== 多目标标签 ====================
    for horizon in FORECAST_HORIZONS:
        df[f"future_return_{horizon}d"] = df["close"].shift(-horizon) / df["close"] - 1
        df[f"label_{horizon}d"] = (df[f"future_return_{horizon}d"] > 0).astype(int)
        df[f"label_3class_{horizon}d"] = 1
        df.loc[df[f"future_return_{horizon}d"] > NOISE_THRESHOLD, f"label_3class_{horizon}d"] = 2
        df.loc[df[f"future_return_{horizon}d"] < -NOISE_THRESHOLD, f"label_3class_{horizon}d"] = 0
        df[f"label_binary_{horizon}d"] = 1
        df.loc[df[f"label_3class_{horizon}d"] == 0, f"label_binary_{horizon}d"] = 0
        df.loc[df[f"label_3class_{horizon}d"] == 1, f"label_binary_{horizon}d"] = -1

    return df


FEATURE_COLS = [
    "momentum_20d",
    "momentum_60d",
    "momentum_accel",
    "volatility",
    "oi_change",
    "price_vs_ma20",
    "rsi_14",
    "hl_ratio",
    "basis",
    "cbot_return_5d",
    "corn_wheat_ratio",
    "hog_return_20d",
    "corn_hog_corr",
    "month_sin",
    "month_cos",
    "is_critical",
    "is_planting",
    "is_harvest",
    "t_anom_东北",
    "t_anom_华北黄淮",
    "p_anom_东北",
    "p_anom_华北黄淮",
    "sm_root_anom_东北",
    "sm_root_anom_华北黄淮",
    "sm_grad_anom_东北",
    "sm_grad_anom_华北黄淮",
    "srad_anom_东北",
    "srad_anom_华北黄淮",
    "rh_anom_东北",
    "rh_anom_华北黄淮",
    "t_critical",
    "t_planting",
    "p_critical",
    "p_planting",
    "sm_planting_ne",
    "sm_planting_hb",
    "precip_x_oi",
    "vol_x_basis",
    "mom_x_oi",
]

CORE_FEATURES = [
    "momentum_20d", "momentum_60d", "momentum_accel",
    "volatility", "oi_change", "price_vs_ma20",
    "rsi_14", "hl_ratio",
    "month_sin", "month_cos",
    "is_critical", "is_planting", "is_harvest",
    "t_anom_东北", "t_anom_华北黄淮",
    "p_anom_东北", "p_anom_华北黄淮",
    "sm_root_anom_东北", "sm_root_anom_华北黄淮",
    "sm_grad_anom_东北", "sm_grad_anom_华北黄淮",
    "srad_anom_东北", "srad_anom_华北黄淮",
    "rh_anom_东北", "rh_anom_华北黄淮",
]

PARTIAL_FEATURES = [
    "basis", "cbot_return_5d", "corn_wheat_ratio",
    "hog_return_20d", "corn_hog_corr",
    "t_critical", "t_planting", "p_critical", "p_planting",
    "sm_planting_ne", "sm_planting_hb",
    "precip_x_oi", "vol_x_basis", "mom_x_oi",
]


def clean_and_split(df: pd.DataFrame, horizon: int,
                    train_cutoff: str = "2020-06-01",
                    test_cutoff: str = "2025-06-01",
                    use_3class: bool = True) -> tuple:
    label_col = f"label_binary_{horizon}d" if use_3class else f"label_{horizon}d"
    mask = df[label_col].notna()
    mask &= (df[label_col] != -1)
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

    return X_train, X_val, X_test, y_train, y_val, y_test


def main() -> None:
    print("=" * 60)
    print("  特征工程 v5：精简 + 交互 + 三分类")
    print("=" * 60)

    df = load_and_align()
    print(f"数据对齐: {len(df)} 行 ({df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()})")
    df = build_features(df)
    df.to_csv(os.path.join(DATA_DIR, "features_full_v2.csv"), index=False, encoding="utf-8-sig")

    print(f"特征: {len(FEATURE_COLS)}个 (核心{len(CORE_FEATURES)} + 部分{len(PARTIAL_FEATURES)})")
    print(f"新增: momentum_accel, sm_grad × 2, t_critical/planting, p_critical/planting, sm_planting × 2")
    print(f"删除: momentum_5d, volume_change, sm_surf, sm_prof, hog_oi_change, weather_stress, ratio_change_20d")
    print(f"三分类: |future_return|>{NOISE_THRESHOLD:.0%} -> 涨/跌, 中间段丢弃")

    for horizon in FORECAST_HORIZONS:
        X_tr, X_val, X_te, y_tr, y_val, y_te = clean_and_split(df, horizon)
        label_col = f"label_binary_{horizon}d"
        print(f"\n  T+{horizon}d:")
        print(f"    训练: {len(X_tr):>5d}  上涨 {y_tr.mean():.1%}")
        print(f"    验证: {len(X_val):>5d}  上涨 {y_val.mean():.1%}")
        print(f"    测试: {len(X_te):>5d}  上涨 {y_te.mean():.1%}")

        np.savez(os.path.join(DATA_DIR, f"features_h{horizon}.npz"),
                 X_train=X_tr, X_val=X_val, X_test=X_te,
                 y_train=y_tr, y_val=y_val, y_test=y_te)

    print(f"\n保存完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
