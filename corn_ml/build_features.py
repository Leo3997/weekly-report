#!/usr/bin/env python3
"""特征工程v5：精简特征 + 生长阶段交互 + 三分类标签 + 回归目标"""

import os
import sys

import numpy as np
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

ANOMALY_WINDOW_YEARS = 5
FORECAST_HORIZONS = [5, 10, 20, 30, 60]
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

    _load_inventory(merged)
    _load_usda_wasde(merged)
    _load_casde(merged)

    merged = merged.sort_values("date").reset_index(drop=True)
    return merged


def _load_inventory(merged: pd.DataFrame):
    inv_path = os.path.join(DATA_DIR, "corn_inventory_daily.csv")
    if os.path.exists(inv_path):
        inv = pd.read_csv(inv_path, parse_dates=["date"])
    else:
        try:
            import akshare
            df = akshare.futures_inventory_em(symbol="玉米")
            df = df.rename(columns={"日期": "date", "库存": "inventory", "增减": "inv_change"})
            df["date"] = pd.to_datetime(df["date"])
            inv = df[["date", "inventory", "inv_change"]].copy()
            inv.to_csv(inv_path, index=False, encoding="utf-8-sig")
        except Exception:
            merged["inventory"] = np.nan
            merged["inv_change"] = np.nan
            return

    inv = inv.sort_values("date")
    inv["date"] = pd.to_datetime(inv["date"])
    inv_dates = inv["date"].values
    merged_dates = merged["date"].values
    for col in ["inventory", "inv_change"]:
        if col not in inv.columns:
            continue
        vals = inv[col].values.astype(float)
        idx = np.searchsorted(inv_dates, merged_dates, side="right") - 1
        idx = np.clip(idx, 0, len(inv_dates) - 1)
        merged[col] = vals[idx]


def _load_usda_wasde(merged: pd.DataFrame):
    path = os.path.join(DATA_DIR, "usda_wasde_corn.csv")
    if not os.path.exists(path):
        return
    usda = pd.read_csv(path)
    usda["report_date"] = pd.to_datetime(usda["report_date"])
    usda = usda.sort_values("report_date")

    usda_cols = [
        "global_stocks_to_use", "china_production_mmt", "china_imports_mmt",
        "china_ending_stocks_mmt", "us_farm_price_usd_bu",
    ]
    usda_dates = usda["report_date"].values
    merged_dates = merged["date"].values
    for col in usda_cols:
        if col not in usda.columns:
            continue
        vals = usda[col].values.astype(float)
        idx = np.searchsorted(usda_dates, merged_dates, side="right") - 1
        idx = np.clip(idx, 0, len(usda_dates) - 1)
        merged[f"usda_{col}"] = vals[idx]


def _load_casde(merged: pd.DataFrame):
    path = os.path.join(DATA_DIR, "casde_corn_supply_demand.csv")
    if not os.path.exists(path):
        return
    casde = pd.read_csv(path)
    casde["report_date"] = pd.to_datetime(casde["report_date"])
    casde = casde.sort_values("report_date")

    casde_cols = [
        "corn_area_kha", "corn_yield_kg_ha", "corn_feed_consumption_mmt",
        "corn_industrial_consumption_mmt", "corn_total_consumption_mmt",
        "corn_imports_mmt", "corn_production_mmt",
    ]
    casde_dates = casde["report_date"].values
    merged_dates = merged["date"].values
    for col in casde_cols:
        if col not in casde.columns:
            continue
        vals = casde[col].values.astype(float)
        idx = np.searchsorted(casde_dates, merged_dates, side="right") - 1
        idx = np.clip(idx, 0, len(casde_dates) - 1)
        merged[f"casde_{col}"] = vals[idx]


def _station_anomaly(df: pd.DataFrame, col_name: str) -> pd.Series:
    """计算气候异常值（当前值 - 过去N年同一天的均值）。
    严格遵循时间序列规范，不使用未来信息。
    """
    df = df.copy()
    df["doy"] = df["date"].dt.dayofyear
    df["year"] = df["date"].dt.year
    
    # 预计算每一年的 DOY 均值（避免重复计算）
    # 注意：这里只按 doy 和 year 分组，不涉及跨年聚合，是安全的
    doy_means = df.groupby(["year", "doy"])[col_name].mean().reset_index()
    
    result = []
    # 使用字典加速查找：(year, doy) -> value
    val_map = doy_means.set_index(["year", "doy"])[col_name].to_dict()
    
    # 逐行/逐组计算异常，仅使用过去 N 年的数据
    for _, row in df.iterrows():
        curr_yr = row["year"]
        curr_doy = row["doy"]
        curr_val = row[col_name]
        
        if pd.isna(curr_val):
            result.append(np.nan)
            continue
            
        past_vals = []
        for y in range(curr_yr - ANOMALY_WINDOW_YEARS, curr_yr):
            v = val_map.get((y, curr_doy))
            if v is not None and pd.notna(v):
                past_vals.append(v)
        
        if past_vals:
            result.append(curr_val - np.mean(past_vals))
        else:
            result.append(np.nan)
            
    return pd.Series(result, index=df.index)


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

    # ★ 短期特征增强：保留波动率和成交量，删除冗余动量/RSI/均线
    df["volatility_5d"] = df["return_1d"].rolling(5).std()
    df["volume_ratio_5d"] = df["volume"] / df["volume"].rolling(5).mean().replace(0, np.nan)

    # ★ 中期特征增强：只保留低相关性指标
    df["ma5_ma20_ratio"] = df["close"].rolling(5).mean() / df["close"].rolling(20).mean().replace(0, np.nan) - 1
    df["price_position_20d"] = (df["close"] - df["low"].rolling(20).min()) / (df["high"].rolling(20).max() - df["low"].rolling(20).min()).replace(0, np.nan)

    # ★ 高区分度技术指标
    # MACD：只保留 macd_hist（已包含 macd 和 signal 的差值信息，避免共线）
    # adjust=False：递推形式，不使用未来数据
    _ema12 = df["close"].ewm(span=12, adjust=False).mean()
    _ema26 = df["close"].ewm(span=26, adjust=False).mean()
    _macd = _ema12 - _ema26
    _macd_signal = _macd.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = _macd - _macd_signal
    
    # 布林带
    ma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    df["bb_width"] = (2 * std20) / ma20.replace(0, np.nan)
    df["bb_position"] = (df["close"] - (ma20 - 2 * std20)) / (4 * std20).replace(0, np.nan)
    
    # 成交量动量
    df["volume_momentum"] = df["volume"].pct_change(5)
    df["volume_ma_ratio"] = df["volume"] / df["volume"].rolling(20).mean().replace(0, np.nan)
    
    # 价格动量变化率
    df["momentum_change"] = df["momentum_20d"].diff()
    df["volatility_change"] = df["volatility"].diff()

    # ==================== 市场状态检测 (波动率regime) ====================
    vol_20 = df["return_1d"].rolling(20).std()
    vol_60 = df["return_1d"].rolling(60).std()
    vol_120 = df["return_1d"].rolling(120).std()
    df["vol_regime_short"] = vol_20 / vol_60.replace(0, np.nan)
    df["vol_regime_long"] = vol_60 / vol_120.replace(0, np.nan)
    df["vol_percentile_60d"] = df["return_1d"].rolling(60).apply(lambda x: pd.Series(x).rank().iloc[-1] / len(x), raw=False)
    df["trend_regime"] = df["momentum_20d"] / vol_20.replace(0, np.nan)

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

    # ==================== 期货库存特征 (3个) ★ ====================
    if "inventory" in df.columns:
        df["inv_level"] = df["inventory"] / 10000
        df["inv_change_5d"] = df["inv_change"].rolling(5).mean()
        df["inv_level_vs_ma20"] = df["inventory"] / df["inventory"].rolling(20).mean() - 1
        for c in ["inventory", "inv_change"]:
            df.drop(columns=[c], inplace=True, errors="ignore")

    # ==================== USDA WASDE 供需特征 (4个) ★ ====================
    if "usda_global_stocks_to_use" in df.columns:
        df["usda_stocks_use"] = df["usda_global_stocks_to_use"]
        df["usda_stocks_use_yoy"] = df["usda_global_stocks_to_use"].pct_change(252)
        df["usda_china_imports"] = df["usda_china_imports_mmt"]
        df["usda_price"] = df["usda_us_farm_price_usd_bu"]
        raw_usda_cols = ["usda_global_stocks_to_use", "usda_china_production_mmt",
                         "usda_china_imports_mmt", "usda_china_ending_stocks_mmt",
                         "usda_us_farm_price_usd_bu"]
        for c in raw_usda_cols:
            df.drop(columns=[c], inplace=True, errors="ignore")

    # ==================== CASDE 供需特征 (5个) ★ ====================
    if "casde_corn_yield_kg_ha" in df.columns:
        df["casde_yield"] = df["casde_corn_yield_kg_ha"] / 1000
        df["casde_area"] = df["casde_corn_area_kha"] / 1000
        df["casde_feed_ratio"] = df["casde_corn_feed_consumption_mmt"] / df["casde_corn_total_consumption_mmt"]
        df["casde_imports"] = df["casde_corn_imports_mmt"]
        df["casde_surplus"] = df["casde_corn_total_consumption_mmt"] - df.get("casde_corn_production_mmt", 0)
        for c in [c for c in df.columns if c.startswith("casde_") and c not in ["casde_yield", "casde_area", "casde_feed_ratio", "casde_imports", "casde_surplus"]]:
            df.drop(columns=[c], inplace=True, errors="ignore")

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


# 删除了以下高度共线特征（每组只保留1~2个信息量最高的）：
# 动量组：删 return_3d, return_5d, momentum_10d（保留 momentum_20d/60d/accel）
# RSI组：删 rsi_7, rsi_21（保留 rsi_14）
# 均线组：删 price_vs_ma5, price_vs_ma10（保留 price_vs_ma20, ma5_ma20_ratio）
# 波动率组：删 volatility_ratio（保留 volatility, volatility_5d）
# MACD组：删 macd（保留 macd_hist，已包含 macd 和 signal 信息）
FEATURE_COLS = [
    # 基础技术面
    "momentum_20d",
    "momentum_60d",
    "momentum_accel",
    "volatility",
    "oi_change",
    "price_vs_ma20",
    "rsi_14",
    "hl_ratio",
    # 短期特征
    "volatility_5d",
    "volume_ratio_5d",
    # 中期特征
    "ma5_ma20_ratio",
    "price_position_20d",
    # 高区分度技术指标
    "macd_hist",
    "bb_width",
    "bb_position",
    "volume_momentum",
    "volume_ma_ratio",
    "momentum_change",
    "volatility_change",
    # 基差和CBOT
    "basis",
    "cbot_return_5d",
    "corn_wheat_ratio",
    # 生猪期货
    "hog_return_20d",
    "corn_hog_corr",
    # 季节
    "month_sin",
    "month_cos",
    "is_critical",
    "is_planting",
    "is_harvest",
    # 气候异常
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
    # 交互特征
    "t_critical",
    "t_planting",
    "p_critical",
    "p_planting",
    "sm_planting_ne",
    "sm_planting_hb",
    "precip_x_oi",
    "vol_x_basis",
    "mom_x_oi",
    # 库存
    "inv_level",
    "inv_change_5d",
    "inv_level_vs_ma20",
    # USDA/CASDE (去重后)
    "usda_stocks_use",
    "usda_stocks_use_yoy",
    "usda_price",
    "casde_imports",
    "casde_surplus",
]

CORE_FEATURES = [
    "momentum_20d", "momentum_60d", "momentum_accel",
    "volatility", "oi_change", "price_vs_ma20",
    "rsi_14", "hl_ratio",
    # 短期特征
    "volatility_5d",
    # 中期特征
    "ma5_ma20_ratio", "price_position_20d",
    # 高区分度技术指标
    "macd_hist", "bb_width", "bb_position",
    # 季节
    "month_sin", "month_cos",
    "is_critical", "is_planting", "is_harvest",
    # 气候异常
    "t_anom_东北", "t_anom_华北黄淮",
    "p_anom_东北", "p_anom_华北黄淮",
    "sm_root_anom_东北", "sm_root_anom_华北黄淮",
    "sm_grad_anom_东北", "sm_grad_anom_华北黄淮",
    "srad_anom_东北", "srad_anom_华北黄淮",
    "rh_anom_东北", "rh_anom_华北黄淮",
]

PARTIAL_FEATURES = [
    # 短期/中期特征
    "volume_ratio_5d",
    "volume_momentum", "volume_ma_ratio",
    "momentum_change", "volatility_change",
    # 市场状态检测
    "vol_regime_short", "vol_regime_long",
    "vol_percentile_60d", "trend_regime",
    # 基差和CBOT
    "basis", "cbot_return_5d", "corn_wheat_ratio",
    # 生猪期货
    "hog_return_20d", "corn_hog_corr",
    # 交互特征
    "t_critical", "t_planting", "p_critical", "p_planting",
    "sm_planting_ne", "sm_planting_hb",
    "precip_x_oi", "vol_x_basis", "mom_x_oi",
    # 库存
    "inv_level", "inv_change_5d", "inv_level_vs_ma20",
    # USDA/CASDE
    "usda_stocks_use", "usda_stocks_use_yoy", "usda_price",
    "casde_imports", "casde_surplus",
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

    if not any(a == "--skip-update" for a in sys.argv):
        from update_data import update_all
        update_all()

    df = load_and_align()
    print(f"数据对齐: {len(df)} 行 ({df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()})")
    df = build_features(df)
    df.to_csv(os.path.join(DATA_DIR, "features_full_v2.csv"), index=False, encoding="utf-8-sig")

    print(f"特征: {len(FEATURE_COLS)}个 (核心{len(CORE_FEATURES)} + 部分{len(PARTIAL_FEATURES)})")
    print(f"整改v6删除共线特征: return_3d/5d, rsi_7/21, price_vs_ma5/10, momentum_10d, ma10_ma60_ratio, volatility_ratio, macd（保留macd_hist）")
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
