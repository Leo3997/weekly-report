#!/usr/bin/env python3
"""USDA WASDE + CASDE 供需报告半自动爬虫
用法:
  python3 fetch_supply_demand.py              # 搜索最新报告并更新CSV
  python3 fetch_supply_demand.py --list       # 仅列出最新可用报告链接
"""

import csv
import os
import re
import sys
import time
from datetime import date, datetime, timedelta

import requests
from bs4 import BeautifulSoup

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}

USDA_CSV = os.path.join(DATA_DIR, "usda_wasde_corn.csv")
CASDE_CSV = os.path.join(DATA_DIR, "casde_corn_supply_demand.csv")


def search_chinagrain_usda(days: int = 90) -> list[dict]:
    """搜索中华粮网最新 USDA 玉米供需报告"""
    results = []
    try:
        url = "http://www.chinagrain.cn/axfwnh/"
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", href=True):
            text = a.get_text(strip=True)
            href = a["href"]
            if ("USDA" in text or "usda" in text.lower()) and "玉米" in text:
                full_url = href if href.startswith("http") else f"http://www.chinagrain.cn{href}"
                results.append({"title": text, "url": full_url, "source": "chinagrain"})
    except Exception as e:
        print(f"  [chinagrain USDA] 搜索失败: {e}", file=sys.stderr)
    return results


def search_chinagrain_casde(days: int = 90) -> list[dict]:
    """搜索中华粮网最新 CASDE 供需报告"""
    results = []
    try:
        url = "http://www.chinagrain.cn/axfwnh/"
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", href=True):
            text = a.get_text(strip=True)
            href = a["href"]
            if "农业农村部" in text and "供需" in text:
                full_url = href if href.startswith("http") else f"http://www.chinagrain.cn{href}"
                results.append({"title": text, "url": full_url, "source": "casde"})
    except Exception as e:
        print(f"  [chinagrain CASDE] 搜索失败: {e}", file=sys.stderr)
    return results


def parse_usda_article(url: str) -> dict:
    """从 chinagrain USDA 文章提取关键数据"""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.encoding = "utf-8"
        text = BeautifulSoup(resp.text, "html.parser").get_text()

        data = {"url": url, "raw_text": text[:3000]}

        patterns = {
            "global_production_mmt": r"全球玉米产量[^0-9]*(\d+\.?\d*)\s*亿吨",
            "global_ending_stocks_mmt": r"全球.*期末库存[^0-9]*(\d+\.?\d*)\s*亿吨",
            "china_production_mmt": r"中国产量[^0-9]*(\d+\.?\d*)\s*亿吨",
            "china_imports_mmt": r"中国.*进口[^0-9]*(\d+)\s*万吨",
            "us_production_bu": r"美国玉米产量[^0-9]*(\d+\.?\d*)\s*亿蒲",
            "us_ending_stocks_bu": r"美国.*期末库存[^0-9]*(\d+\.?\d*)\s*亿蒲",
            "us_farm_price": r"均价[^0-9]*(\d+\.?\d*)\s*美元",
            "argentina_production_mmt": r"阿根廷[^0-9]*(\d+)\s*万吨",
            "brazil_production_mmt": r"巴西[^0-9]*(\d+)\s*万吨",
        }
        for key, pat in patterns.items():
            m = re.search(pat, text)
            if m:
                try:
                    val = float(m.group(1))
                    data[key] = val
                except ValueError:
                    pass
        return data
    except Exception as e:
        print(f"  [解析USDA] {e}", file=sys.stderr)
        return {}


def parse_casde_article(url: str) -> dict:
    """从 chinagrain CASDE 文章提取关键数据"""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.encoding = "utf-8"
        text = BeautifulSoup(resp.text, "html.parser").get_text()

        data = {"url": url, "raw_text": text[:3000]}

        patterns = {
            "corn_area_kha": r"玉米种植面积\s*(\d+)\s*千公顷",
            "corn_production_mmt": r"总产量\s*(\d+)\s*万吨",
            "corn_yield_kg_ha": r"单产每公顷\s*(\d+)\s*公斤",
            "corn_feed_consumption": r"饲用消费[^0-9]*(\d+)\s*万吨",
            "corn_industrial_consumption": r"工业消费[^0-9]*(\d+)\s*万吨",
            "corn_total_consumption": r"消费总量[^0-9]*(\d+)\s*万吨",
            "corn_imports": r"进口[^0-9]*(\d+)\s*万吨",
            "soybean_area_kha": r"大豆播种面积[^0-9]*(\d+)\s*千公顷",
            "soybean_production_mmt": r"大豆[^总]*产量[^0-9]*(\d+)\s*万吨",
            "soybean_imports": r"大豆进口量[^0-9]*(\d+)\s*万吨",
        }
        for key, pat in patterns.items():
            m = re.search(pat, text, re.DOTALL)
            if m:
                try:
                    val = float(m.group(1).replace(",", ""))
                    data[key] = val
                except ValueError:
                    pass
        return data
    except Exception as e:
        print(f"  [解析CASDE] {e}", file=sys.stderr)
        return {}


def auto_update_usda():
    """自动搜索最新 USDA 报告并追加到 CSV"""
    print("  [USDA] 搜索最新玉米供需报告...")
    articles = search_chinagrain_usda()
    if not articles:
        print("    ✗ 未找到新报告")
        return

    print(f"    找到 {len(articles)} 篇相关文章")
    latest = articles[0]
    print(f"    最新: {latest['title'][:80]}")

    existing_dates = set()
    if os.path.exists(USDA_CSV):
        with open(USDA_CSV, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                existing_dates.add(row.get("report_date", ""))

    article_date = _extract_date_from_url(latest["url"])
    if article_date and article_date in existing_dates:
        print(f"    → 已存在 ({article_date})，跳过")
        return

    parsed = parse_usda_article(latest["url"])
    if parsed:
        print(f"    提取到: { {k:v for k,v in parsed.items() if k != 'raw_text'} }")
        print(f"    → 请手动审核后添加到 {USDA_CSV}")
        print(f"    原文: {latest['url']}")


def auto_update_casde():
    """自动搜索最新 CASDE 报告并追加建议"""
    print("  [CASDE] 搜索最新中国供需报告...")
    articles = search_chinagrain_casde()
    if not articles:
        print("    ✗ 未找到新报告")
        return

    print(f"    找到 {len(articles)} 篇相关文章")
    latest = articles[0]
    print(f"    最新: {latest['title'][:80]}")

    existing_dates = set()
    if os.path.exists(CASDE_CSV):
        with open(CASDE_CSV, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                existing_dates.add(row.get("report_date", ""))

    article_date = _extract_date_from_url(latest["url"])
    if article_date and article_date in existing_dates:
        print(f"    → 已存在 ({article_date})，跳过")
        return

    parsed = parse_casde_article(latest["url"])
    if parsed:
        print(f"    提取到: { {k:v for k,v in parsed.items() if k != 'raw_text'} }")
        print(f"    → 请手动审核后添加到 {CASDE_CSV}")
        print(f"    原文: {latest['url']}")


def _extract_date_from_url(url: str) -> str:
    m = re.search(r"/(\d{4})/(\d{2})/(\d{2})/", url)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return ""


def list_reports():
    """列出所有可用的报告"""
    print("=== USDA WASDE 玉米供需报告 ===")
    usda = search_chinagrain_usda()
    for a in usda[:5]:
        print(f"  {a['title'][:100]}")
        print(f"    {a['url']}")

    print("\n=== CASDE 中国农产品供需报告 ===")
    casde = search_chinagrain_casde()
    for a in casde[:5]:
        print(f"  {a['title'][:100]}")
        print(f"    {a['url']}")

    print(f"\n=== 已保存数据 ===")
    for csv_path, name in [(USDA_CSV, "USDA"), (CASDE_CSV, "CASDE")]:
        if os.path.exists(csv_path):
            with open(csv_path, encoding="utf-8-sig") as f:
                reader = list(csv.DictReader(f))
                print(f"  {name}: {len(reader)} 条, 最新={reader[-1].get('report_date','?')}")


def main():
    if "--list" in sys.argv:
        list_reports()
        return

    print("=" * 60)
    print("  供需报告半自动爬虫 (USDA + CASDE)")
    print("=" * 60)

    auto_update_usda()
    print()
    auto_update_casde()

    print(f"\n{'='*60}")
    print("  提示: 爬虫提取数据后需人工审核再写入CSV")
    print("  下次运行: python3 fetch_supply_demand.py --list")
    print("=" * 60)


if __name__ == "__main__":
    main()
