#!/usr/bin/env python3
"""增量更新气候+卫星数据到最新日期  — 每次运行前自动调用"""

import csv
import os
import sys
import time
from datetime import date, datetime, timedelta

import requests

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
NASA_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

STATIONS = [
    {"name": "哈尔滨", "lat": 45.75, "lon": 126.63},
    {"name": "长春", "lat": 43.82, "lon": 125.32},
    {"name": "沈阳", "lat": 41.80, "lon": 123.43},
    {"name": "呼和浩特", "lat": 40.82, "lon": 111.75},
    {"name": "济南", "lat": 36.67, "lon": 116.98},
    {"name": "郑州", "lat": 34.76, "lon": 113.65},
]


def _nasa_fetch(lat, lon, start_str, end_str, params_str):
    resp = requests.get(
        NASA_URL,
        params={
            "parameters": params_str,
            "community": "AG",
            "longitude": round(lon, 4),
            "latitude": round(lat, 4),
            "start": start_str,
            "end": end_str,
            "format": "JSON",
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("properties", {}).get("parameter", {})


def _read_last_date(csv_path):
    if not os.path.exists(csv_path):
        return 20000401
    last = 20000401
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = int(row.get("date", "0"))
            if d > last:
                last = d
    return last


def update_climate():
    csv_path = os.path.join(DATA_DIR, "corn_climate_daily.csv")
    last_date = _read_last_date(csv_path)
    last_dt = date(last_date // 10000, (last_date % 10000) // 100, last_date % 100)
    today = date.today()

    if (today - last_dt).days <= 2:
        print(f"  [气候] 已最新 (截止 {last_date})")
        return

    start_str = (last_dt + timedelta(days=1)).strftime("%Y%m%d")
    end_str = (today + timedelta(days=3)).strftime("%Y%m%d")

    print(f"  [气候] 增量拉取 {start_str} ~ {end_str} (NASA POWER) ...", end=" ", flush=True)

    new_rows = []
    for station in STATIONS:
        try:
            param = _nasa_fetch(station["lat"], station["lon"], start_str, end_str, "T2M,PRECTOTCORR")
            t2m = param.get("T2M", {})
            precip = param.get("PRECTOTCORR", {})
            all_dates = sorted(set(t2m.keys()) | set(precip.keys()))
            for d in all_dates:
                t_val = t2m.get(d, -999)
                p_val = precip.get(d, -999)
                if t_val != -999 or p_val != -999:
                    new_rows.append({
                        "date": d,
                        "station": station["name"],
                        "lat": station["lat"],
                        "lon": station["lon"],
                        "temp_C": t_val if t_val != -999 else "",
                        "precip_mm": p_val if p_val != -999 else "",
                    })
        except Exception as e:
            print(f"\n  [气候] {station['name']} 拉取失败: {e}", file=sys.stderr)
            continue
        time.sleep(0.2)

    if new_rows:
        with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["date", "station", "lat", "lon", "temp_C", "precip_mm"])
            writer.writerows(new_rows)
        latest = max(int(r["date"]) for r in new_rows)
        print(f"OK (+{len(new_rows)}行, 最新={latest})")
    else:
        print("无新数据")


def update_satellite():
    csv_path = os.path.join(DATA_DIR, "corn_satellite_daily.csv")
    last_date = _read_last_date(csv_path)
    last_dt = date(last_date // 10000, (last_date % 10000) // 100, last_date % 100)
    today = date.today()

    if (today - last_dt).days <= 2:
        print(f"  [卫星] 已最新 (截止 {last_date})")
        return

    start_str = (last_dt + timedelta(days=1)).strftime("%Y%m%d")
    end_str = (today + timedelta(days=3)).strftime("%Y%m%d")
    params_str = "GWETROOT,GWETTOP,GWETPROF,ALLSKY_SFC_SW_DWN,RH2M"
    fieldnames = ["date", "station", "lat", "lon", "GWETROOT", "GWETTOP", "GWETPROF", "ALLSKY_SFC_SW_DWN", "RH2M"]

    print(f"  [卫星] 增量拉取 {start_str} ~ {end_str} (NASA POWER) ...", end=" ", flush=True)

    new_rows = []
    for station in STATIONS:
        try:
            param = _nasa_fetch(station["lat"], station["lon"], start_str, end_str, params_str)
            date_set = set()
            for pn in fieldnames[4:]:
                date_set.update(param.get(pn.upper(), {}).keys())
            for d in sorted(date_set):
                row = {"date": d, "station": station["name"], "lat": station["lat"], "lon": station["lon"]}
                for pn in fieldnames[4:]:
                    val = param.get(pn.upper(), {}).get(d, -999)
                    row[pn] = val if val != -999 else ""
                new_rows.append(row)
        except Exception as e:
            print(f"\n  [卫星] {station['name']} 拉取失败: {e}", file=sys.stderr)
            continue
        time.sleep(0.2)

    if new_rows:
        with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerows(new_rows)
        latest = max(int(r["date"]) for r in new_rows)
        print(f"OK (+{len(new_rows)}行, 最新={latest})")
    else:
        print("无新数据")


def update_all():
    print("=" * 50)
    print("  增量数据更新 (NASA POWER)")
    print("=" * 50)
    update_climate()
    update_satellite()
    print("=" * 50)


if __name__ == "__main__":
    update_all()
