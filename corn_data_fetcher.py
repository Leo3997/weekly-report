#!/usr/bin/env python3
"""
全国玉米主要产区数据爬取工具
气候数据: NASA POWER API (逐日气温、降水)
种植面积: 国家统计局 API (优先) -> USDA FAS API -> 参考数据(回退)
"""

import csv
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Optional

import requests

CORRECTION_THRESHOLD = 0.2

MAJOR_CORN_REGIONS: list[dict[str, Any]] = [
    {"province": "黑龙江", "lat": 47.862, "lon": 127.761, "reg_code": "230000"},
    {"province": "吉林",   "lat": 43.897, "lon": 125.326, "reg_code": "220000"},
    {"province": "辽宁",   "lat": 41.804, "lon": 123.433, "reg_code": "210000"},
    {"province": "内蒙古", "lat": 44.091, "lon": 113.886, "reg_code": "150000"},
    {"province": "河北",   "lat": 38.043, "lon": 114.469, "reg_code": "130000"},
    {"province": "山东",   "lat": 36.670, "lon": 116.991, "reg_code": "370000"},
    {"province": "河南",   "lat": 33.882, "lon": 113.614, "reg_code": "410000"},
    {"province": "山西",   "lat": 37.570, "lon": 112.287, "reg_code": "140000"},
    {"province": "陕西",   "lat": 35.604, "lon": 110.006, "reg_code": "610000"},
    {"province": "四川",   "lat": 30.651, "lon": 104.076, "reg_code": "510000"},
    {"province": "云南",   "lat": 25.046, "lon": 102.710, "reg_code": "530000"},
    {"province": "贵州",   "lat": 26.600, "lon": 106.713, "reg_code": "520000"},
]

_CORN_AREA_REFERENCE: dict[str, float] = {
    "黑龙江": 6500.0,
    "吉林":   4400.0,
    "内蒙古": 4200.0,
    "山东":   3900.0,
    "河南":   3800.0,
    "河北":   3400.0,
    "辽宁":   2700.0,
    "山西":   1800.0,
    "四川":   1800.0,
    "云南":   1800.0,
    "陕西":   1200.0,
    "贵州":    800.0,
}


def _safe_request(
    url: str,
    params: Optional[dict[str, Any]] = None,
    headers: Optional[dict[str, str]] = None,
    timeout: int = 30,
    retries: int = 2,
) -> requests.Response:
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  [重试] {exc}，{wait}s后重试...", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("unreachable")


# ======================== NASA POWER 气候数据 ========================

def fetch_nasa_power_climate(
    lat: float,
    lon: float,
    start_date: int,
    end_date: int,
) -> dict[str, Any]:
    base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "T2M,T2M_MAX,T2M_MIN,PRECTOTCORR",
        "community": "AG",
        "longitude": round(lon, 4),
        "latitude": round(lat, 4),
        "start": start_date,
        "end": end_date,
        "format": "JSON",
    }
    resp = _safe_request(base_url, params=params, timeout=60)
    data = resp.json()
    return data.get("properties", {}).get("parameter", {})


def _aggregate(param_values: dict[str, float]) -> dict[str, Optional[float]]:
    if not param_values:
        return {"avg": None, "max": None, "min": None, "sum": None}
    filtered = {k: v for k, v in param_values.items() if v != -999}
    if not filtered:
        return {"avg": None, "max": None, "min": None, "sum": None}
    vals = list(filtered.values())
    return {
        "avg": round(sum(vals) / len(vals), 2),
        "max": round(max(vals), 2),
        "min": round(min(vals), 2),
        "sum": round(sum(vals), 2),
    }


# ======================== 国家统计局 API ========================

def _build_nbs_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Referer": "https://data.stats.gov.cn/",
    })
    try:
        s.get("https://data.stats.gov.cn/", timeout=10)
    except requests.RequestException:
        pass
    return s


def fetch_nbs_corn_area(year: int) -> dict[str, Optional[float]]:
    session = _build_nbs_session()
    url = "https://data.stats.gov.cn/easyquery.htm"
    params = {
        "m": "QueryData",
        "dbcode": "fsnd",
        "rowcode": "reg",
        "colcode": "sj",
        "wds": "[]",
        "dfwds": json.dumps([
            {"wdcode": "zb", "valuecode": "A0D0A"},
            {"wdcode": "reg", "valuecode": ",".join(r["reg_code"] for r in MAJOR_CORN_REGIONS)},
        ]),
    }
    resp = session.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    result: dict[str, Optional[float]] = {}
    datanodes = data.get("returndata", {}).get("datanodes", [])
    for node in datanodes:
        code_parts = node.get("code", "").split(".")
        if len(code_parts) < 3:
            continue
        region_code = code_parts[1]
        str_data = node.get("data", {}).get("strdata", "")
        try:
            val = float(str_data)
        except (ValueError, TypeError):
            val = None
        for region in MAJOR_CORN_REGIONS:
            if region["reg_code"] == region_code:
                result[region["province"]] = val
                break
    return result


# ======================== USDA FAS API ========================

_USDA_COMMODITY_CODE = "0440000"


def fetch_usda_corn_area(api_key: str, year: int) -> dict[str, Optional[float]]:
    headers = {"API_KEY": api_key, "Accept": "application/json"}
    url = f"https://apps.fas.usda.gov/OpenData/api/psd/commodity/{_USDA_COMMODITY_CODE}/country/CH"
    resp = _safe_request(url, headers=headers, timeout=60)
    records = resp.json()
    if not isinstance(records, list):
        raise ValueError(f"USDA 返回格式异常: {type(records)}")

    target_year = str(year)
    matched = [r for r in records if r.get("year") == target_year]
    if not matched:
        matched = sorted(records, key=lambda r: r.get("year", ""), reverse=True)[:1]

    result: dict[str, Optional[float]] = {}
    for r in matched:
        attr_name = r.get("attributeName", "")
        if "area harvested" in attr_name.lower():
            for region in MAJOR_CORN_REGIONS:
                result[region["province"]] = r.get("amount")
            break
    return result


# ======================== CSV 输出 ========================

def output_climate_csv(climate_data: list[dict[str, Any]], output_dir: str) -> str:
    path = os.path.join(output_dir, "corn_climate_data.csv")
    fieldnames = [
        "province", "lat", "lon",
        "temp_avg_C", "temp_max_C", "temp_min_C",
        "precip_avg_mm", "precip_max_mm", "precip_min_mm", "precip_sum_mm",
        "period_start", "period_end",
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in climate_data:
            writer.writerow(row)
    return path


def output_area_csv(
    area_data: dict[str, Optional[float]],
    output_dir: str,
    source_label: str,
) -> str:
    path = os.path.join(output_dir, "corn_area_data.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["province", "area_kha", "data_source", "data_note"])
        writer.writeheader()
        for region in MAJOR_CORN_REGIONS:
            province = region["province"]
            val = area_data.get(province)
            note = ""
            if val is None:
                note = "数据缺失"
            elif val < CORRECTION_THRESHOLD:
                note = "⚠ 数值异常偏低，可能为统计口径差异（疑似单位为千公顷而非万亩）"
            writer.writerow({
                "province": province,
                "area_kha": val,
                "data_source": source_label,
                "data_note": note,
            })
    return path


# ======================== 主流程 ========================

def _parse_args() -> dict[str, Any]:
    args: dict[str, Any] = {"usda_key": None}
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--usda-key" and i + 1 < len(sys.argv):
            args["usda_key"] = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--climate-year" and i + 1 < len(sys.argv):
            args["climate_year"] = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--area-year" and i + 1 < len(sys.argv):
            args["area_year"] = int(sys.argv[i + 1])
            i += 2
        else:
            i += 1
    return args


def main() -> None:
    cli_args = _parse_args()
    output_dir = os.path.dirname(os.path.abspath(__file__))
    current_year = datetime.now().year

    climate_year = cli_args.get("climate_year", current_year - 1)
    climate_start_int = int(f"{climate_year}0401")
    climate_end_int = int(f"{climate_year}1031")
    climate_start_display = f"{climate_year}-04-01"
    climate_end_display = f"{climate_year}-10-31"

    area_year = cli_args.get("area_year", current_year - 2)

    print("=" * 60)
    print("  全国玉米主要产区数据爬取工具")
    print("=" * 60)
    print(f"气候数据时段: {climate_start_display} ~ {climate_end_display}")
    print(f"气候数据来源: NASA POWER (逐日)")
    print(f"种植面积年份: {area_year}")
    print(f"产区数量:     {len(MAJOR_CORN_REGIONS)} 个\n")

    # ==================== 1. 气候数据 ====================
    print("[1/2] 正在获取气候数据 ...")
    climate_rows: list[dict[str, Any]] = []
    for i, region in enumerate(MAJOR_CORN_REGIONS, 1):
        name = region["province"]
        lat, lon = region["lat"], region["lon"]
        print(f"  [{i:2d}/{len(MAJOR_CORN_REGIONS)}] {name} ({lat}, {lon}) ...", end=" ", flush=True)
        try:
            param_data = fetch_nasa_power_climate(lat, lon, climate_start_int, climate_end_int)
            t2m_stats = _aggregate(param_data.get("T2M", {}))
            precip_stats = _aggregate(param_data.get("PRECTOTCORR", {}))
            climate_rows.append({
                "province": name,
                "lat": lat,
                "lon": lon,
                "temp_avg_C": t2m_stats["avg"],
                "temp_max_C": t2m_stats["max"],
                "temp_min_C": t2m_stats["min"],
                "precip_avg_mm": precip_stats["avg"],
                "precip_max_mm": precip_stats["max"],
                "precip_min_mm": precip_stats["min"],
                "precip_sum_mm": precip_stats["sum"],
                "period_start": climate_start_display,
                "period_end": climate_end_display,
            })
            print("OK")
        except Exception as exc:
            print(f"失败: {exc}")
            climate_rows.append({
                "province": name, "lat": lat, "lon": lon,
                "temp_avg_C": None, "temp_max_C": None, "temp_min_C": None,
                "precip_avg_mm": None, "precip_max_mm": None, "precip_min_mm": None, "precip_sum_mm": None,
                "period_start": climate_start_display, "period_end": climate_end_display,
            })

    climate_path = output_climate_csv(climate_rows, output_dir)
    print(f"\n  ✓ 气候数据已保存至: {climate_path}")

    # ==================== 2. 种植面积数据 ====================
    print("\n[2/2] 正在获取种植面积数据 ...")
    area_data: dict[str, Optional[float]] = {}
    area_source = ""

    # ---- 方案A: 国家统计局 ----
    try:
        print("  -> 尝试 国家统计局 API ...", end=" ", flush=True)
        area_data = fetch_nbs_corn_area(area_year)
        if area_data:
            area_source = f"国家统计局 ({area_year}年)"
            print(f"成功 ({len(area_data)} 省)")
        else:
            raise ValueError("空数据")
    except Exception as exc:
        print(f"失败 ({exc})")

    # ---- 方案B: USDA FAS ----
    if not area_data:
        usda_key = cli_args.get("usda_key")
        if usda_key:
            try:
                print("  -> 尝试 USDA FAS API ...", end=" ", flush=True)
                area_data = fetch_usda_corn_area(usda_key, area_year)
                if area_data:
                    area_source = f"USDA FAS PSD ({area_year}年)"
                    print(f"成功 ({len(area_data)} 省)")
                else:
                    raise ValueError("空数据")
            except Exception as exc:
                print(f"失败 ({exc})")
        else:
            print("  -> USDA API: 需要 --usda-key 参数（在 https://developer.fas.usda.gov 注册）")

    # ---- 方案C: 嵌入式参考数据 ----
    if not area_data:
        print("  -> 使用嵌入式参考数据 (中国统计年鉴公开数据)")
        area_data = dict(_CORN_AREA_REFERENCE)
        area_source = f"参考数据 (中国统计年鉴，约 {area_year} 年水平)"

    area_path = output_area_csv(area_data, output_dir, area_source)
    print(f"\n  ✓ 种植面积数据已保存至: {area_path}")

    # ==================== 3. 汇总打印 ====================
    print("\n" + "=" * 60)
    print("  数据汇总")
    print("=" * 60)
    print(f"{'省份':<8} {'均温°C':>7} {'最高°C':>7} {'最低°C':>7} {'降水mm':>8} {'面积kha':>9}")
    print("-" * 48)
    for row in climate_rows:
        name = row["province"]
        t_avg = row.get("temp_avg_C") or "-"
        t_max = row.get("temp_max_C") or "-"
        t_min = row.get("temp_min_C") or "-"
        p_sum = row.get("precip_sum_mm") or "-"
        area = area_data.get(name, "-")
        t_avg_s = f"{t_avg:>7}" if isinstance(t_avg, (int, float)) else f"{t_avg:>7}"
        t_max_s = f"{t_max:>7}" if isinstance(t_max, (int, float)) else f"{t_max:>7}"
        t_min_s = f"{t_min:>7}" if isinstance(t_min, (int, float)) else f"{t_min:>7}"
        p_s = f"{p_sum:>8}" if isinstance(p_sum, (int, float)) else f"{p_sum:>8}"
        a_s = f"{area:>9}" if isinstance(area, (int, float)) else f"{area:>9}"
        print(f"{name:<8} {t_avg_s} {t_max_s} {t_min_s} {p_s} {a_s}")
    print("-" * 48)
    print(f"面积数据来源: {area_source}")
    print(f"\n生成文件:")
    print(f"  {climate_path}")
    print(f"  {area_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
