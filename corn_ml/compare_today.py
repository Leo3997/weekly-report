#!/usr/bin/env python3
"""今日 vs 过去5年同日 气候+卫星要素对比报告 (自动适配2026)"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

CLIMATE_REGIONS = {
    "东北": ["哈尔滨", "长春", "沈阳"],
    "华北黄淮": ["呼和浩特", "济南", "郑州"],
}

ELEMENTS = {
    "temp_C": ("气温", "°C"),
    "precip_mm": ("降水", "mm"),
    "GWETROOT": ("根区土壤水分", "0-1"),
    "GWETTOP": ("表层土壤水分", "0-1"),
    "GWETPROF": ("剖面土壤水分", "0-1"),
    "ALLSKY_SFC_SW_DWN": ("太阳辐射", "MJ/m²"),
    "RH2M": ("相对湿度", "%"),
}


def load_data():
    climate = pd.read_csv(os.path.join(DATA_DIR, "corn_climate_daily.csv"), dtype={"date": str})
    satellite = pd.read_csv(os.path.join(DATA_DIR, "corn_satellite_daily.csv"), dtype={"date": str})

    for df in [climate, satellite]:
        df["month"] = df["date"].str[4:6].astype(int)
        df["day"] = df["date"].str[6:8].astype(int)
        df["year"] = df["date"].str[:4].astype(int)

    return climate, satellite


def get_day_values(climate, satellite, year, month, day):
    c_day = climate[(climate["year"] == year) & (climate["month"] == month) & (climate["day"] == day)]
    s_day = satellite[(satellite["year"] == year) & (satellite["month"] == month) & (satellite["day"] == day)]

    results = {}
    for elem_key in ELEMENTS:
        results[elem_key] = {}
        src = c_day if elem_key in ["temp_C", "precip_mm"] else s_day
        for station in ["哈尔滨", "长春", "沈阳", "呼和浩特", "济南", "郑州"]:
            row = src[src["station"] == station]
            raw = row[elem_key].values[0] if len(row) > 0 else None
            try:
                val = float(raw) if raw is not None and raw == raw else None
            except (ValueError, TypeError):
                val = None
            results[elem_key][station] = val
        for region_name, stations in CLIMATE_REGIONS.items():
            vals = [results[elem_key][s] for s in stations if results[elem_key][s] is not None]
            results[elem_key][region_name] = float(np.mean(vals)) if vals else None
    return results


def main():
    climate, satellite = load_data()

    # Auto-detect latest date
    latest = climate["date"].max()
    target_month = int(latest[4:6])
    target_day = int(latest[6:8])
    today_year = int(latest[:4])

    available_years = sorted(climate["year"].unique())
    past_years = [y for y in range(today_year - 5, today_year) if y in available_years]
    if not past_years:
        past_years = available_years[-6:-1] if len(available_years) >= 6 else available_years[:-1]

    all_years = past_years + [today_year]
    all_data = {}
    for yr in all_years:
        all_data[yr] = get_day_values(climate, satellite, yr, target_month, target_day)

    has_today = all_data[today_year]["temp_C"]["东北"] is not None

    today_display = f"{today_year}-{target_month:02d}-{target_day:02d}"
    if not has_today:
        # fallback: use latest available
        fallback = past_years[-1]
        today_display = f"⚠️ {today_year}数据尚未上线，用 {fallback} 代替"
        report_year = fallback
    else:
        report_year = today_year
        today_display = f"{report_year}-{target_month:02d}-{target_day:02d} (NASA最新)"

    def safe_val(v):
        return round(v, 2) if v is not None else None

    print("=" * 72)
    print(f"  玉米产区气候要素对比报告")
    print(f"  基准日: {today_display}")
    print(f"  对比期: {past_years[0]}-{past_years[-1]}（过去5年同日）")
    print("=" * 72)

    for elem_key, (elem_name, unit) in ELEMENTS.items():
        print(f"\n┌  {elem_name} ({unit})")
        print(f"│")
        today_vals = all_data[report_year][elem_key]

        header = f"│  {'站点/区域':<10s}"
        for yr in past_years:
            header += f"  {yr:>6d}"
        if report_year not in past_years:
            header += f"  {report_year:>6d}"
        header += f"  {'5年均值':>8s}  {'差值':>8s}  {'方向':>6s}"
        print(header)

        for stations_list, is_region in [
            (["哈尔滨", "长春", "沈阳", "呼和浩特", "济南", "郑州"], False),
            (["东北", "华北黄淮"], True),
        ]:
            for name in stations_list:
                prefix = "│  " if not is_region else "│  ★"
                today_v = today_vals.get(name)

                line = f"{prefix}{name:<10s}"
                past_vals_raw = []
                for yr in past_years:
                    v = all_data[yr][elem_key].get(name)
                    past_vals_raw.append(v)
                    line += f"  {safe_val(v) or '-':>6}"

                if report_year not in past_years:
                    tv_disp = safe_val(today_v)
                    line += f"  {tv_disp if isinstance(tv_disp, (int, float)) else tv_disp:>6.2f}" if tv_disp is not None else f"  {'-':>6}"

                valid_past = [v for v in past_vals_raw if v is not None]
                avg5 = np.mean(valid_past) if valid_past else None
                avg5_str = f"{avg5:>8.2f}" if avg5 is not None else "       -"

                if today_v is not None and avg5 is not None:
                    diff = today_v - avg5
                    diff_str = f"{diff:>+8.2f}"
                    threshold = {"temp_C": 1.0, "precip_mm": 1.0}.get(elem_key, 0.03)
                    if abs(diff) <= threshold:
                        direction = "正常"
                    elif elem_key == "precip_mm":
                        direction = "偏多↑" if diff > 0 else "偏少↓"
                    elif elem_key == "temp_C":
                        direction = "偏高↑" if diff > 0 else "偏低↓"
                    else:
                        direction = "偏高↑" if diff > 0 else "偏低↓"
                else:
                    diff_str = "       -"
                    direction = "无数据"
                line += f"  {avg5_str}  {diff_str}  {direction}"
                print(line)

    print(f"\n{'─'*72}")
    print(f"  ★ = 产区均值 | ↑偏高/偏多  ↓偏低/偏少")
    print(f"  数据: NASA POWER (气温/降水) + 卫星级土壤/辐射/湿度")
    print(f"{'─'*72}")

    # 摘要
    print(f"\n{'='*72}")
    print(f"  综合摘要")
    print(f"{'='*72}")
    for region_name in CLIMATE_REGIONS:
        summaries = []
        for elem_key, (elem_name, unit) in ELEMENTS.items():
            today_v = all_data[report_year][elem_key].get(region_name)
            past_vals = [all_data[yr][elem_key].get(region_name) for yr in past_years]
            past_vals = [v for v in past_vals if v is not None]
            if today_v is not None and past_vals:
                avg5 = np.mean(past_vals)
                diff = today_v - avg5
                th = {"temp_C": 1.0, "precip_mm": 1.0}.get(elem_key, 0.03)
                if abs(diff) > th:
                    d = "偏高" if diff > 0 else "偏低"
                    if elem_key == "precip_mm":
                        d = "偏多" if diff > 0 else "偏少"
                    summaries.append(f"{elem_name} {d} {abs(diff):.1f}{unit}")
        print(f"  {region_name}: {'; '.join(summaries) if summaries else '各要素接近5年均值'}")

    # 干旱信号
    print(f"\n  水分条件:")
    for region_name in CLIMATE_REGIONS:
        precip = all_data[report_year]["precip_mm"].get(region_name)
        sm_root = all_data[report_year]["GWETROOT"].get(region_name)
        sm_surf = all_data[report_year]["GWETTOP"].get(region_name)
        if all(v is not None for v in [precip, sm_root, sm_surf]):
            p_past = [v for v in [all_data[yr]["precip_mm"].get(region_name) for yr in past_years] if v is not None]
            s_past = [v for v in [all_data[yr]["GWETROOT"].get(region_name) for yr in past_years] if v is not None]
            p_lo = precip < np.percentile(p_past, 25) if p_past else False
            s_lo = sm_root < np.percentile(s_past, 25) if s_past else False
            if p_lo and s_lo:
                st = "⚠️ 降水+土壤双低"
            elif p_lo:
                st = "⚡ 降水偏少"
            elif s_lo:
                st = "🔍 土壤偏干"
            else:
                st = "✅ 正常"
            print(f"    {region_name}: {st} (降水={precip:.1f}mm, 根区={sm_root:.2f}, 表层={sm_surf:.2f})")

    print(f"\n{'─'*72}")
    print(f"  下次更新: python3 compare_today.py")
    print(f"  NASA 数据延迟约 3-4 天，自动取最新可用日期")
    print(f"{'─'*72}")


if __name__ == "__main__":
    main()
