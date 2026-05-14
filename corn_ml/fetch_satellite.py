#!/usr/bin/env python3
"""拉取 NASA POWER 卫星级参数：土壤水分、辐射、湿度、蒸散"""

import csv
import os
import time
from typing import Any

import requests

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

STATIONS: list[dict[str, Any]] = [
    {"name": "哈尔滨", "lat": 45.75, "lon": 126.63},
    {"name": "长春",   "lat": 43.82, "lon": 125.32},
    {"name": "沈阳",   "lat": 41.80, "lon": 123.43},
    {"name": "呼和浩特", "lat": 40.82, "lon": 111.75},
    {"name": "济南",   "lat": 36.67, "lon": 116.98},
    {"name": "郑州",   "lat": 34.76, "lon": 113.65},
]

NASA_PARAMS = (
    "GWETROOT,"         # 根区土壤水分 (0-1) = SMAP 等价
    "GWETTOP,"          # 表层土壤水分 (0-1)
    "GWETPROF,"         # 深层土壤剖面水分 (0-1)
    "ALLSKY_SFC_SW_DWN,"  # 太阳短波辐射 (MJ/m²/day) = NDVI 驱动
    "RH2M"              # 2m 相对湿度 (%)
)

PARAM_NAMES = ["GWETROOT", "GWETTOP", "GWETPROF", "ALLSKY_SFC_SW_DWN", "RH2M"]

YEARS = list(range(2010, 2027))


def _fetch(lat: float, lon: float, start: int, end: int) -> dict:
    resp = requests.get(
        "https://power.larc.nasa.gov/api/temporal/daily/point",
        params={
            "parameters": NASA_PARAMS,
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


def main():
    path = os.path.join(OUTPUT_DIR, "corn_satellite_daily.csv")
    rows: list[dict] = []

    for yi, year in enumerate(YEARS):
        start_int = int(f"{year}0401")
        end_int = int(f"{year}1031")
        print(f"  [{yi + 1}/{len(YEARS)}] {year} 生长季 ...", end=" ", flush=True)
        try:
            for station in STATIONS:
                param = _fetch(station["lat"], station["lon"], start_int, end_int)
                date_set = set()
                for pn in PARAM_NAMES:
                    date_set.update(param.get(pn.upper(), {}).keys())
                for d in sorted(date_set):
                    row = {
                        "date": d,
                        "station": station["name"],
                        "lat": station["lat"],
                        "lon": station["lon"],
                    }
                    for pn in PARAM_NAMES:
                        val = param.get(pn.upper(), {}).get(d, -999)
                        row[pn] = val if val != -999 else ""
                    rows.append(row)
            print(f"OK")
        except Exception as exc:
            print(f"FAIL: {exc}")
        if yi < len(YEARS) - 1:
            time.sleep(0.3)

    fieldnames = ["date", "station", "lat", "lon"] + PARAM_NAMES
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  → {len(rows)} 行 -> {path}")


if __name__ == "__main__":
    main()
