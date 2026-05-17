#!/usr/bin/env python3
"""
生猪 & 鸡蛋 供需基本面数据获取 (akshare)
每周可运行，数据保存到 CSV，供周报生成器调用
"""
import os
import sys
import csv
from datetime import date, datetime, timedelta

import akshare as ak

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

HOG_CSV = os.path.join(DATA_DIR, "hog_fundamentals.csv")
EGG_CSV = os.path.join(DATA_DIR, "egg_fundamentals.csv")


def fetch_hog_fundamentals():
    """
    获取生猪供需基本面数据:
      - 全国均价 (搜猪网 年趋势)  → spot_hog_year_trend_soozhu
      - 头均养殖成本 → futures_hog_cost
      - 猪粮比 (core) → futures_hog_core
      - 供给指标 → futures_hog_supply
      - 玉米/豆粕/配合饲料价格 → soozhu spot
      - 期现基差 → futures_spot_price(LH)
    Returns: dict
    """
    today = date.today().isoformat()
    result = {"date": today}

    # 1. 全国均价 (元/公斤)
    try:
        df = ak.spot_hog_year_trend_soozhu()
        result["hog_price_kg"] = float(df.iloc[-1]["价格"])
        result["hog_price_date"] = str(df.iloc[-1]["日期"])
    except Exception:
        result["hog_price_kg"] = ""

    # 2. 头均养殖成本
    try:
        df = ak.futures_hog_cost()
        result["hog_cost"] = float(df.iloc[-1]["value"])
        result["hog_cost_date"] = str(df.iloc[-1]["date"])
    except Exception:
        result["hog_cost"] = ""

    # 3. 猪粮比 (core)
    try:
        df = ak.futures_hog_core()
        result["hog_core"] = float(df.iloc[-1]["value"])
        result["hog_core_date"] = str(df.iloc[-1]["date"])
    except Exception:
        result["hog_core"] = ""

    # 4. 供给指标
    try:
        df = ak.futures_hog_supply()
        result["hog_supply"] = float(df.iloc[-1]["value"])
        result["hog_supply_date"] = str(df.iloc[-1]["date"])
    except Exception:
        result["hog_supply"] = ""

    # 5. 期现基差 (主力合约)
    try:
        df = ak.futures_spot_price(date=today.replace("-", ""), vars_list=["LH"])
        row = df[df["symbol"] == "LH"]
        if not row.empty:
            r = row.iloc[0]
            result["hog_futures_spot"] = float(r["spot_price"])
            result["hog_futures_close"] = float(r["dominant_contract_price"])
            result["hog_futures_basis"] = float(r["dom_basis"])
            result["hog_futures_basis_rate"] = float(r["dom_basis_rate"])
    except Exception:
        pass

    # 6. 玉米价格 (元/公斤)
    try:
        df = ak.spot_corn_price_soozhu()
        result["corn_price_kg"] = float(df.iloc[-1]["价格"])
    except Exception:
        result["corn_price_kg"] = ""

    # 7. 豆粕价格 (元/公斤)
    try:
        df = ak.spot_soybean_price_soozhu()
        result["soybean_price_kg"] = float(df.iloc[-1]["价格"])
    except Exception:
        result["soybean_price_kg"] = ""

    # 8. 配合饲料价格 (元/公斤)
    try:
        df = ak.spot_mixed_feed_soozhu()
        result["feed_price_kg"] = float(df.iloc[-1]["价格"])
    except Exception:
        result["feed_price_kg"] = ""

    return result


def fetch_egg_fundamentals():
    """
    获取鸡蛋供需基本面数据:
      - 期现基差 → futures_spot_price(JD)
      - 玉米/豆粕成本 → soozhu spot (与 hog 共用)
      - 鸡蛋现货 → futures_spot_price 中的 spot_price
    Returns: dict
    """
    today = date.today().isoformat()
    result = {"date": today}

    # 1. 期现基差
    try:
        df = ak.futures_spot_price(date=today.replace("-", ""), vars_list=["JD"])
        row = df[df["symbol"] == "JD"]
        if not row.empty:
            r = row.iloc[0]
            result["egg_spot"] = float(r["spot_price"])
            result["egg_futures_close"] = float(r["dominant_contract_price"])
            result["egg_futures_basis"] = float(r["dom_basis"])
            result["egg_futures_basis_rate"] = float(r["dom_basis_rate"])
    except Exception:
        pass

    # 2. 玉米价格 (与饲料成本相关)
    try:
        df = ak.spot_corn_price_soozhu()
        result["corn_price_kg"] = float(df.iloc[-1]["价格"])
    except Exception:
        result["corn_price_kg"] = ""

    # 3. 豆粕价格
    try:
        df = ak.spot_soybean_price_soozhu()
        result["soybean_price_kg"] = float(df.iloc[-1]["价格"])
    except Exception:
        result["soybean_price_kg"] = ""

    # 4. 估算饲料成本: 玉米占65%, 豆粕占25%, 2.2斤饲料产1斤蛋
    try:
        corn_p = float(result.get("corn_price_kg", 0) or 0)
        soy_p = float(result.get("soybean_price_kg", 0) or 0)
        if corn_p > 0 and soy_p > 0:
            feed_per_kg = corn_p * 0.65 + soy_p * 0.25
            result["est_feed_cost"] = round(feed_per_kg * 1.1, 2)  # 约2.2斤料/斤蛋的一半成本
    except Exception:
        result["est_feed_cost"] = ""

    return result


def save_fundamentals_csv(csv_path, data: dict, columns: list):
    """追加一行数据到 CSV，同一天不重复写入"""
    existing = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing.append(row)

    # 同一天不重复写
    today_str = data.get("date", "")
    for row in existing:
        if row.get("date") == today_str:
            return  # 已存在

    existing.append(data)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(existing)


def update_all_fundamentals():
    """更新生猪和鸡蛋基本面数据 (供 generate_weekly_report 调用)"""
    results = {}

    print("  [生猪] 拉取供需基本面...", file=sys.stderr)
    try:
        hog = fetch_hog_fundamentals()
        hog_cols = [
            "date", "hog_price_kg", "hog_price_date",
            "hog_cost", "hog_cost_date",
            "hog_core", "hog_core_date",
            "hog_supply", "hog_supply_date",
            "hog_futures_spot", "hog_futures_close", "hog_futures_basis", "hog_futures_basis_rate",
            "corn_price_kg", "soybean_price_kg", "feed_price_kg",
        ]
        save_fundamentals_csv(HOG_CSV, hog, hog_cols)
        results["hog"] = hog
        print(f"    ✓ 生猪: 均价{hog.get('hog_price_kg','?')}元/kg, "
              f"猪粮比{hog.get('hog_core','?')}, 成本{hog.get('hog_cost','?')}元/头",
              file=sys.stderr)
    except Exception as e:
        print(f"    ✗ 生猪: {e}", file=sys.stderr)

    print("  [鸡蛋] 拉取供需基本面...", file=sys.stderr)
    try:
        egg = fetch_egg_fundamentals()
        egg_cols = [
            "date", "egg_spot", "egg_futures_close", "egg_futures_basis", "egg_futures_basis_rate",
            "corn_price_kg", "soybean_price_kg", "est_feed_cost",
        ]
        save_fundamentals_csv(EGG_CSV, egg, egg_cols)
        results["egg"] = egg
        print(f"    ✓ 鸡蛋: 现货{egg.get('egg_spot','?')}元/500kg, "
              f"基差{egg.get('egg_futures_basis','?')}, "
              f"饲料成本≈{egg.get('est_feed_cost','?')}元/kg",
              file=sys.stderr)
    except Exception as e:
        print(f"    ✗ 鸡蛋: {e}", file=sys.stderr)

    return results


if __name__ == "__main__":
    update_all_fundamentals()
