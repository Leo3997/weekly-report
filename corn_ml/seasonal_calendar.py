#!/usr/bin/env python3
"""玉米季节性日历：返回不同日期的淡季/旺季/关键期权重系数"""

from datetime import date, datetime


# 月份 → 淡旺季
# 1月: 需求旺季(春节前), 2月: 淡季(节后), 3-4月: 播种预期, 5月: 出苗期
# 6月: 拔节期, 7月: 抽穗吐丝(最敏感), 8月: 灌浆期, 9月: 乳熟
# 10月: 收获上市压力, 11-12月: 需求旺季(冬季消费)

MONTH_SEASON = {
    1:  "需求旺季",
    2:  "节后淡季",
    3:  "播种预期",
    4:  "播种出苗",
    5:  "出苗生长期",
    6:  "拔节生长期",
    7:  "抽穗吐丝期 ★关键★",
    8:  "灌浆期 ★关键★",
    9:  "乳熟收获期",
    10: "集中上市期",
    11: "需求旺季",
    12: "需求旺季",
}


def get_season_info(dt: date = None) -> dict:
    if dt is None:
        dt = date.today()
    m = dt.month
    season_name = MONTH_SEASON.get(m, "未知")
    is_critical = m in [7, 8]

    return {
        "month": m,
        "season": season_name,
        "is_critical": is_critical,
        "is_planting": m in [4, 5],
        "is_growing": m in [6, 7, 8],
        "is_harvest": m in [9, 10],
        "is_off_season": m in [2],
    }


def get_feature_weight_modifiers(dt: date = None) -> dict:
    """
    返回 33 个特征的季节修正系数 (>1=加重, <1=减轻)
    结合玉米生长周期逻辑:
      - 生长期(6-8月): 气候/土壤因子权重翻倍
      - 收获期(9-10月): 基差/库存因子加强
      - 淡季(2月): 全部减弱
    """
    info = get_season_info(dt)

    base_modifiers = {
        "momentum_5d": 1.0, "momentum_20d": 1.0, "momentum_60d": 1.0,
        "volatility": 1.0, "oi_change": 1.0, "volume_change": 1.0,
        "price_vs_ma20": 1.0, "rsi_14": 1.0, "hl_ratio": 1.0,
        "basis": 1.0, "cbot_return_5d": 1.0, "corn_wheat_ratio": 1.0,
        "hog_price_lag1": 1.0, "hog_mom_lag1": 1.0,
        "ratio_change_20d": 1.0, "mom_x_oi": 1.0,
    }

    # 气候/卫星/交叉类 → 根据季节动态调整
    climate_mod = 1.0
    if info["is_critical"]:
        climate_mod = 2.5        # 抽穗灌浆期: 天气因子×2.5
    elif info["is_growing"]:
        climate_mod = 2.0        # 生长期: ×2.0
    elif info["is_planting"]:
        climate_mod = 1.5        # 播种期: ×1.5

    # 基差在收获期加强
    basis_mod = 2.0 if info["is_harvest"] else 1.0

    # 外盘全年稳定
    foreign_mod = 1.2

    # 生猪需求旺季加强
    hog_mod = 1.5 if info["month"] in [1, 11, 12] else 1.0

    # 淡季全减
    if info["is_off_season"]:
        for k in base_modifiers:
            base_modifiers[k] = 0.6
        climate_mod = 0.6
        basis_mod = 0.6
        foreign_mod = 0.6
        hog_mod = 0.6

    # 应用气候修正
    for prefix in ["t_anom_", "p_anom_", "sm_root_", "sm_surf_", "sm_prof_",
                   "srad_anom_", "rh_anom_", "weather_stress", "precip_x_oi"]:
        base_modifiers[prefix] = climate_mod

    base_modifiers["basis"] = basis_mod
    base_modifiers["vol_x_basis"] = basis_mod
    base_modifiers["cbot_return_5d"] = foreign_mod
    base_modifiers["corn_wheat_ratio"] = foreign_mod
    base_modifiers["ratio_change_20d"] = foreign_mod
    base_modifiers["hog_price_lag1"] = hog_mod
    base_modifiers["hog_mom_lag1"] = hog_mod

    return base_modifiers


def apply_seasonal_weights(
    feature_names: list[str],
    dt: date = None,
    base_weights: list[float] = None,
) -> list[float]:
    modifiers = get_feature_weight_modifiers(dt)
    if base_weights is None:
        base_weights = [1.0 / len(feature_names)] * len(feature_names)

    adjusted = []
    for name, w in zip(feature_names, base_weights):
        for prefix, mod in modifiers.items():
            if name.startswith(prefix) or name == prefix:
                w *= mod
                break
        adjusted.append(w)

    total = sum(adjusted)
    if total > 0:
        adjusted = [w / total for w in adjusted]
    return adjusted


if __name__ == "__main__":
    for m in range(1, 13):
        d = date(2025, m, 15)
        info = get_season_info(d)
        mods = get_feature_weight_modifiers(d)
        climate_m = mods.get("t_anom_", 1.0)
        print(f"{m:>2}月 {info['season']:<12s}  气候修正={climate_m:.1f}x  "
              f"关键期={info['is_critical']}  收获={info['is_harvest']}")
