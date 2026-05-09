#!/usr/bin/env python3
"""拉取玉米期货、现货、基差、CBOT、气候全部原始数据"""

import csv
import json
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


def fetch_futures_main() -> str:
    import akshare as ak
    print("[1/4] 拉取玉米期货主力连续 (akshare) ...", end=" ", flush=True)
    df = ak.futures_main_sina(symbol="C0")
    df = df.rename(columns={
        "日期": "date", "开盘价": "open", "最高价": "high",
        "最低价": "low", "收盘价": "close", "成交量": "volume",
        "持仓量": "open_interest", "动态结算价": "settle",
    })
    path = os.path.join(OUTPUT_DIR, "corn_futures_main.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"OK ({len(df)} rows, {df['date'].iloc[0]} ~ {df['date'].iloc[-1]})")
    return path


def fetch_spot_basis() -> str:
    import akshare as ak
    print("[2/4] 拉取玉米现货 + 基差 (akshare) ...", end=" ", flush=True)
    try:
        df = ak.futures_spot_price_daily(start_day="20220101", end_day="20260507", vars_list=["C"])
        if df.empty:
            raise ValueError("empty")
    except Exception:
        df = ak.futures_spot_price_daily(start_day="20220101", end_day="20260507", vars_list=["C"])
    df = df.rename(columns={
        "spot_price": "spot_price", "near_basis": "near_basis",
        "dom_basis": "dom_basis", "near_basis_rate": "near_basis_rate",
        "dom_basis_rate": "dom_basis_rate",
        "dominant_contract_price": "dom_contract_price",
    })
    path = os.path.join(OUTPUT_DIR, "corn_spot_basis.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"OK ({len(df)} rows, {df['date'].iloc[0]} ~ {df['date'].iloc[-1]})")
    return path


def fetch_cbot_corn() -> str:
    import akshare as ak
    print("[3/4] 拉取 CBOT 玉米期货 (akshare) ...", end=" ", flush=True)
    df = ak.futures_foreign_hist(symbol="C")
    df = df.rename(columns={
        "open": "cbot_open", "high": "cbot_high", "low": "cbot_low",
        "close": "cbot_close", "volume": "cbot_volume",
        "settlement": "cbot_settle",
    })
    path = os.path.join(OUTPUT_DIR, "corn_cbot.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"OK ({len(df)} rows, {df['date'].iloc[0]} ~ {df['date'].iloc[-1]})")
    return path


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


def fetch_climate(start_year: int = 2005, end_year: int = 2025) -> str:
    path = os.path.join(OUTPUT_DIR, "corn_climate_daily.csv")
    existing_dates: set[str] = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                existing_dates.add(row.get("date", ""))

    rows: list[dict] = []
    total_years = end_year - start_year + 1

    for yi, year in enumerate(range(start_year, end_year + 1)):
        start_int = int(f"{year}0401")
        end_int = int(f"{year}1031")
        print(f"  [{yi + 1}/{total_years}] 拉取 {year} 年生长季气候 (NASA POWER) ...",
              end=" ", flush=True)
        try:
            for station in CORN_BELT_STATIONS:
                key = f"{year}-{station['name']}"
                if key in existing_dates:
                    continue
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
            print(f"OK")
        except Exception as e:
            print(f"FAIL: {e}")
        if yi < total_years - 1:
            time.sleep(0.5)

    if rows:
        fieldnames = ["date", "station", "lat", "lon", "temp_C", "precip_mm"]
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(f"  气候数据共 {len(rows)} 行 -> {path}")
    return path


def main() -> None:
    print("=" * 60)
    print("  玉米 ML 数据拉取")
    print("=" * 60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    path1 = fetch_futures_main()
    path2 = fetch_spot_basis()
    path3 = fetch_cbot_corn()
    path4 = fetch_climate()

    print("\n" + "=" * 60)
    print("数据文件:")
    print(f"  期货主力:   {path1}")
    print(f"  现货+基差: {path2}")
    print(f"  CBOT玉米:   {path3}")
    print(f"  气候日数据: {path4}")
    print("=" * 60)


if __name__ == "__main__":
    main()
