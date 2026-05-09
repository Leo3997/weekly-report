#!/usr/bin/env python3
"""拉取真实 NASA POWER 气候 + akshare 生猪/小麦/CBOT 补充数据"""

import csv
import os
import sys
import time
from datetime import datetime
from typing import Any

import requests

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

CORN_BELT_STATIONS: list[dict[str, Any]] = [
    {"name": "哈尔滨", "lat": 45.75, "lon": 126.63},
    {"name": "长春",   "lat": 43.82, "lon": 125.32},
    {"name": "沈阳",   "lat": 41.80, "lon": 123.43},
    {"name": "呼和浩特", "lat": 40.82, "lon": 111.75},
    {"name": "济南",   "lat": 36.67, "lon": 116.98},
    {"name": "郑州",   "lat": 34.76, "lon": 113.65},
]

CLIMATE_YEARS = list(range(2010, 2027))


def _nasa_fetch(lat: float, lon: float, start: int, end: int) -> dict:
    resp = requests.get(
        "https://power.larc.nasa.gov/api/temporal/daily/point",
        params={
            "parameters": "T2M,PRECTOTCORR",
            "community": "AG",
            "longitude": round(lon, 4),
            "latitude": round(lat, 4),
            "start": start,
            "end": end,
            "format": "JSON",
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json().get("properties", {}).get("parameter", {})


def fetch_climate() -> str:
    path = os.path.join(OUTPUT_DIR, "corn_climate_daily.csv")
    rows: list[dict] = []

    for yi, year in enumerate(CLIMATE_YEARS):
        start_int = int(f"{year}0401")
        end_int = int(f"{year}1031")
        print(f"  [{yi + 1}/{len(CLIMATE_YEARS)}] {year} 年生长季 ...", end=" ", flush=True)
        try:
            for station in CORN_BELT_STATIONS:
                param = _nasa_fetch(station["lat"], station["lon"], start_int, end_int)
                t2m = param.get("T2M", {})
                precip = param.get("PRECTOTCORR", {})
                all_dates = sorted(set(t2m.keys()) | set(precip.keys()))
                for d in all_dates:
                    t_val = t2m.get(d, -999)
                    p_val = precip.get(d, -999)
                    if t_val != -999 or p_val != -999:
                        rows.append({
                            "date": d,
                            "station": station["name"],
                            "lat": station["lat"],
                            "lon": station["lon"],
                            "temp_C": t_val if t_val != -999 else "",
                            "precip_mm": p_val if p_val != -999 else "",
                        })
            print(f"OK ({sum(1 for r in rows if r['date'].startswith(str(year)))} 条)")
        except Exception as exc:
            print(f"FAIL: {exc}")
        if yi < len(CLIMATE_YEARS) - 1:
            time.sleep(0.3)

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "station", "lat", "lon", "temp_C", "precip_mm"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  → 共 {len(rows)} 行, 保存到 {path}")
    return path


def fetch_hog_price() -> str:
    import akshare as ak
    print("[生猪价格] 拉取搜猪网生猪均价 (日频) ...", end=" ", flush=True)
    df = ak.spot_hog_year_trend_soozhu()
    df = df.rename(columns={"日期": "date", "价格": "hog_price"})
    df["date"] = df["date"].astype(str)
    path = os.path.join(OUTPUT_DIR, "hog_price_daily.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"OK ({len(df)} 行, {df['date'].iloc[0]} ~ {df['date'].iloc[-1]})")
    return path


def fetch_cbot_wheat() -> str:
    import akshare as ak
    print("[CBOT小麦] 拉取 (W) ...", end=" ", flush=True)
    df = ak.futures_foreign_hist(symbol="W")
    path = os.path.join(OUTPUT_DIR, "cbot_wheat.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"OK ({len(df)} 行, {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()})")
    return path


def main() -> None:
    print("=" * 60)
    print("  玉米 ML 阶段二：真实气候 + 月频因子拉取")
    print("=" * 60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    path1 = fetch_climate()
    path2 = fetch_hog_price()
    path3 = fetch_cbot_wheat()

    print("\n新数据文件:")
    print(f"  气候(真实NASA): {path1}")
    print(f"  生猪均价:       {path2}")
    print(f"  CBOT小麦:       {path3}")
    print("=" * 60)


if __name__ == "__main__":
    main()
