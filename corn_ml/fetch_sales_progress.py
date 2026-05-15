#!/usr/bin/env python3
"""主产区售粮进度数据抓取 + 图表生成
数据源: 饲料行业信息网(feedtrade.com.cn) / 慧博投研 / 百度新闻搜索
"""

import os
import re
import csv
import sys
from datetime import date, datetime

import requests
from bs4 import BeautifulSoup

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(DATA_DIR, "regional_sales_progress.csv")
OUT_DIR = os.path.join(os.path.dirname(DATA_DIR), "corn_weekly_report", "output")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}

KEYWORDS_PATTERNS = {
    "黑龙江": r"黑龙江[^\d]*(\d+)[%％\s]*(?:售|已售|进度|见底|售罄)",
    "吉林": r"吉林[^\d]*(\d+)[%％\s]*(?:售|已售|进度|见底|售罄)",
    "辽宁": r"辽宁[^\d]*(\d+)[%％\s]*(?:售|已售|进度|见底|售罄)",
    "内蒙古": r"内蒙古[^\d]*(\d+)[%％\s]*(?:售|已售|进度|见底|售罄)",
    "山东": r"山东[^\d]*(\d+)[%％\s]*(?:售|已售|进度|见底|售罄)",
    "河北": r"河北[^\d]*(\d+)[%％\s]*(?:售|已售|进度|见底|售罄)",
    "河南": r"河南[^\d]*(\d+)[%％\s]*(?:售|已售|进度|见底|售罄)",
}

SELLOUT_KW = ["售罄", "见底", "基本售完", "接近售罄"]


def fetch_from_feedtrade():
    """从饲料行业信息网抓取玉米售粮进度"""
    results = []
    try:
        url = "https://www.feedtrade.com.cn/corn/"
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.encoding = "gb2312"
        soup = BeautifulSoup(resp.text, "html.parser")

        for a in soup.find_all("a", href=True):
            title = a.get_text(strip=True)
            if "玉米" in title and any(kw in title for kw in ["售粮", "市场", "日报", "周报", "分析"]):
                href = a["href"]
                if not href.startswith("http"):
                    href = "https://www.feedtrade.com.cn" + href
                try:
                    art_resp = requests.get(href, headers=HEADERS, timeout=15)
                    art_resp.encoding = "gb2312"
                    art_text = BeautifulSoup(art_resp.text, "html.parser").get_text()

                    for province, pat in KEYWORDS_PATTERNS.items():
                        m = re.search(pat, art_text)
                        if m:
                            pct = int(m.group(1))
                            results.append({
                                "province": province,
                                "sales_progress_pct": pct,
                                "source_url": href,
                            })

                    if len(results) >= 3:
                        break
                except Exception:
                    continue
    except Exception as e:
        print(f"  [feedtrade] {e}", file=sys.stderr)

    return results


def fetch_from_baidu_news():
    """从百度新闻搜索玉米售粮进度"""
    results = []
    queries = [
        "玉米售粮进度 2026",
        "东北玉米 余粮 2026年5月",
        "华北玉米 售粮 进度",
    ]
    for q in queries:
        try:
            url = f"https://www.baidu.com/s?wd={q}&tn=news&rtt=4"
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.encoding = "utf-8"
            text = resp.text
            for province, pat in KEYWORDS_PATTERNS.items():
                m = re.search(pat, text)
                if m:
                    pct = int(m.group(1))
                    results.append({
                        "province": province,
                        "sales_progress_pct": pct,
                        "source_url": url,
                    })
        except Exception:
            pass
    return results


def load_existing_data():
    if not os.path.exists(CSV_PATH):
        return []
    rows = []
    with open(CSV_PATH, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def save_data(rows):
    fieldnames = [
        "report_date", "region", "province", "sales_progress_pct",
        "remaining_grain", "spot_price_low", "spot_price_high",
        "price_unit", "price_vs_month", "notes",
    ]
    with open(CSV_PATH, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def build_markdown_table(rows):
    lines = ["## 主产区玉米售粮进度", ""]
    lines.append(f"**数据日期: {rows[0]['report_date']}**" if rows else "")
    lines.append("")
    lines.append("| 区域 | 省份 | 售粮进度 | 余粮情况 | 现货价格 | 较上月 |")
    lines.append("|------|------|----------|----------|----------|--------|")
    for r in rows:
        price = f"{r['spot_price_low']}~{r['spot_price_high']} {r['price_unit']}" if r.get("spot_price_low") else "—"
        progress = f"{r['sales_progress_pct']}%"
        lines.append(
            f"| {r['region']} | {r['province']} | {progress} | {r['remaining_grain']} | {price} | {r.get('price_vs_month', '—')} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print("=== 主产区售粮进度抓取工具 ===")
    print(f"  数据文件: {CSV_PATH}")

    print("\n[1/2] 尝试从饲料行业信息网抓取...")
    feedtrade_data = fetch_from_feedtrade()
    if feedtrade_data:
        print(f"  ✓ 获取到 {len(feedtrade_data)} 省数据")
    else:
        print("  ✗ 未获取到数据 (使用本地缓存)")

    print("\n[2/2] 尝试从百度新闻搜索...")
    baidu_data = fetch_from_baidu_news()
    if baidu_data:
        print(f"  ✓ 获取到 {len(baidu_data)} 条结果")
    else:
        print("  ✗ 未获取到数据")

    existing = load_existing_data()
    if existing:
        print(f"\n[OK] 本地已有 {len(existing)} 条记录")
        print(build_markdown_table(existing))
    else:
        print("\n[WARN] 无本地数据，请手动更新 CSV")
