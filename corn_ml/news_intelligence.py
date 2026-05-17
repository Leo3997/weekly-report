#!/usr/bin/env python3
"""
新闻情报收集 v4 — akshare stock_news_em 农业股新闻 (真实, 带时间戳)
"""

import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, date, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests

_ENV_FILES = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "corn_weekly_report", ".env"),
]
for _ef in _ENV_FILES:
    if os.path.exists(_ef):
        try:
            with open(_ef, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())
        except Exception:
            pass

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}

CLIMATE_REGIONS = {
    "东北": ["哈尔滨", "长春", "沈阳"],
    "华北黄淮": ["呼和浩特", "济南", "郑州"],
}

# akshare lazy import
_ak = None


def _get_ak():
    global _ak
    if _ak is None:
        import akshare as _ak_module
        _ak = _ak_module
    return _ak

# akshare 农业股代码 — 通过它们的关联新闻间接覆盖玉米市场
AGRI_STOCKS = [
    ("600598", "北大荒"), ("000713", "丰乐种业"),
    ("002385", "大北农"), ("000998", "隆平高科"),
    ("300189", "神农种业"), ("600313", "农发种业"),
]

CORN_KEYWORDS = [
    "玉米", "粮食", "谷物", "饲料", "种植", "大豆", "豆粕",
    "小麦", "收储", "CBOT", "cbot", "养殖", "生猪", "农产品",
    "中储粮", "进口", "出口", "关税", "补贴", "种业", "育种",
    "转基因", "天气", "干旱", "洪涝", "产量",
]

AGRI_SPECIFIC_KW = [
    "玉米", "粮食", "谷物", "饲料", "种植", "大豆", "豆粕",
    "小麦", "收储", "CBOT", "cbot", "养殖", "生猪", "农产品",
    "中储粮", "种业", "育种", "转基因", "干旱", "洪涝", "产量",
]

EXCLUDE_KW = [
    "铜", "铝", "锌", "锡", "镍", "原油", "石油", "黄金", "白银",
    "锂", "钴", "铁矿石", "螺纹钢", "甲醇", "PTA", "LPG", "碳酸锂",
    "芯片", "半导体", "新能源车", "光伏", "医药", "地产", "楼市",
    "人民币汇率", "央行", "MLF", "LPR", "降息", "降准", "A股",
    "上证", "深证", "创业板", "沙特", "中东冲突", "俄乌",
    "制造业", "工业", "轮胎", "锡价", "锡矿",
]


def _call_deepseek(prompt: str, max_tokens: int = 2000) -> str:
    if not DEEPSEEK_API_KEY:
        return ""
    for attempt in range(3):
        try:
            resp = requests.post(
                f"{DEEPSEEK_BASE_URL}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": "你是中国玉米期货市场分析助手。请用中文回答, 简洁专业。"},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.3,
                    "max_tokens": max_tokens,
                },
                timeout=60,
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            if resp.status_code == 429:
                import time; time.sleep(2)
                continue
            print(f"[DeepSeek] {resp.status_code}", file=sys.stderr)
            return ""
        except Exception as e:
            print(f"[DeepSeek] {e}", file=sys.stderr)
            if attempt < 2:
                import time; time.sleep(1)
            else:
                return ""
    return ""


# ============= 新闻抓取 (akshare stock_news_em, 真实+时间戳) =============

def _fetch_agri_news_akshare(max_per_stock: int = 20, max_age_days: int = 7) -> list[dict]:
    from datetime import timedelta
    cutoff_date = date.today() - timedelta(days=max_age_days)

    results: list[dict] = []
    seen = set()

    for code, name in AGRI_STOCKS:
        try:
            ak = _get_ak()
            df = ak.stock_news_em(symbol=code)
        except Exception as e:
            print(f"  [akshare {name}] {e}", file=sys.stderr)
            continue
        count = 0
        for _, row in df.iterrows():
            title = str(row.get("新闻标题", ""))
            if not any(kw in title for kw in CORN_KEYWORDS):
                continue
            pub_time = str(row.get("发布时间", ""))
            try:
                dt = datetime.strptime(pub_time[:10], "%Y-%m-%d").date()
            except ValueError:
                continue
            if dt < cutoff_date:
                continue
            key = title[:60]
            if key in seen:
                continue
            seen.add(key)
            summary = str(row.get("新闻内容", ""))[:200]
            source = str(row.get("文章来源", name))
            url = str(row.get("新闻链接", ""))
            results.append({
                "title": title, "abstract": summary,
                "time": pub_time, "source": f"{name}({source})",
                "url": url,
            })
            count += 1
            if count >= max_per_stock:
                break

    results.sort(key=lambda x: x.get("time", ""), reverse=True)
    return results


def _fetch_sina_corn_search(max_items: int = 40, max_age_days: int = 7) -> list[dict]:
    now_ts = int(datetime.now().timestamp())
    cutoff = now_ts - max_age_days * 86400
    results = []
    seen = set()
    for page in range(1, 4):
        try:
            r = requests.get(
                "https://feed.mix.sina.com.cn/api/roll/get",
                params={"pageid": 153, "lid": 2516, "num": 50, "page": page},
                headers=HEADERS, timeout=15,
            )
            if r.status_code != 200:
                continue
            data = r.json()
            for item in data.get("result", {}).get("data", []):
                title = item.get("title", "")
                intro = item.get("intro", "")
                combined = title + (intro or "")
                if not any(kw in combined for kw in AGRI_SPECIFIC_KW):
                    continue
                if any(kw in combined for kw in EXCLUDE_KW):
                    continue
                ctime = int(item.get("ctime", 0))
                if ctime < cutoff:
                    continue
                key = (title + (intro or "")[:40])
                if key in seen:
                    continue
                seen.add(key)
                dt = datetime.fromtimestamp(ctime)
                results.append({
                    "title": title, "abstract": intro[:150] if intro else "",
                    "time": dt.strftime("%Y-%m-%d %H:%M"), "source": "新浪财经",
                })
                if len(results) >= max_items:
                    break
            if len(results) >= max_items:
                break
        except Exception:
            pass
    return results


def _fetch_eastmoney_corn_search(max_items: int = 30, max_age_days: int = 7) -> list[dict]:
    from datetime import timedelta
    cutoff_date = date.today() - timedelta(days=max_age_days)
    results = []
    seen = set()
    search_keywords = ["玉米期货", "玉米价格", "玉米供需", "玉米进口", "玉米库存",
                       "玉米种植", "CBOT玉米", "农产品期货", "粮食安全"]
    for keyword in search_keywords:
        if len(results) >= max_items:
            break
        try:
            ak = _get_ak()
            df = ak.stock_news_em(symbol=keyword)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            title = str(row.get("新闻标题", ""))
            if any(kw in title for kw in EXCLUDE_KW):
                continue
            pub_time = str(row.get("发布时间", ""))
            try:
                dt = datetime.strptime(pub_time[:10], "%Y-%m-%d").date()
            except ValueError:
                continue
            if dt < cutoff_date:
                continue
            key = title[:60]
            if key in seen:
                continue
            seen.add(key)
            summary = str(row.get("新闻内容", ""))[:200]
            source = str(row.get("文章来源", "东方财富"))
            url = str(row.get("新闻链接", ""))
            results.append({
                "title": title, "abstract": summary,
                "time": pub_time, "source": source,
                "url": url,
            })
            if len(results) >= max_items:
                break

    results.sort(key=lambda x: x.get("time", ""), reverse=True)
    return results


def collect_corn_news(max_age_days: int = 7) -> str:
    import time as _time

    all_items: list[dict] = []
    seen_titles: set = set()

    print("  [东方财富] 玉米关键词直接搜索...", file=sys.stderr)
    em_items = _fetch_eastmoney_corn_search(30, max_age_days)
    for it in em_items:
        key = it["title"][:60]
        if key not in seen_titles:
            seen_titles.add(key)
            all_items.append(it)
    print(f"    ✓ 东方财富: {len(em_items)}条", file=sys.stderr)

    print("  [akshare] 农业股关联新闻...", file=sys.stderr)
    agri_items = _fetch_agri_news_akshare(20, max_age_days)
    for it in agri_items:
        key = it["title"][:60]
        if key not in seen_titles:
            seen_titles.add(key)
            all_items.append(it)
    print(f"    ✓ akshare农业股: {len(agri_items)}条", file=sys.stderr)

    print("  [新浪财经] 期货滚动新闻...", file=sys.stderr)
    sina_items = _fetch_sina_corn_search(40, max_age_days)
    for it in sina_items:
        key = (it["title"] + it.get("abstract", ""))[:60]
        if key not in seen_titles:
            seen_titles.add(key)
            all_items.append(it)
    print(f"    ✓ 新浪财经: {len(sina_items)}条", file=sys.stderr)

    if not all_items:
        return "[未能获取到新闻 — 各新闻源可能暂时不可用]"

    all_items.sort(key=lambda x: str(x.get("time", "")), reverse=True)

    today = date.today()
    lines = [
        f"以下为 {today} 从多源抓取的真实玉米相关新闻:",
        f"",
        f"数据来源: 东方财富关键词搜索 + akshare农业股新闻 + 新浪财经期货滚动",
        f"时效过滤: 仅保留{today}前{max_age_days}天内新闻",
        f"共 {len(all_items)} 条, 展示如下 (全部来自真实数据接口):",
        "",
    ]

    for i, item in enumerate(all_items[:40], 1):
        time_s = f"[{item.get('time','')}] " if item.get('time') else ""
        src_s = f" ({item.get('source','')})" if item.get('source') else ""
        lines.append(f"{i}. {time_s}{item['title'][:120]}{src_s}")
        if item.get("abstract"):
            abst = re.sub(r'<[^>]+>', '', item['abstract'])[:150]
            if abst:
                lines.append(f"   {abst}")

    lines.append(f"\n(以上为多源真实数据接口返回, 共{len(all_items)}条, 非AI生成)")
    return "\n".join(lines)


def classify_news_with_ai(news_text: str) -> str:
    """两步处理: 先让 DeepSeek 过滤不相关条目, 再做多空分类"""
    if not DEEPSEEK_API_KEY:
        return "DeepSeek API 不可用"

    prompt = f"""你是一个玉米期货分析师。请对以下新闻列表执行两步处理:

## 第一步: 相关性过滤
逐条检查每条新闻, 剔除以下类型的条目:
- 主要讲其他品种(玻璃/纯碱/螺纹钢/PVC/PTA/甲醇/有色金属/贵金属/原油)的新闻
- 纯粹讲A股大盘/央行货币政策/MLF/降息降准的宏观新闻
- 与玉米/饲料/养殖无关的制造业/工业/地产/芯片/新能源新闻
- 虽然提到"玉米"但实际上以其他品种为主的行情综述

对每条新闻标注 [相关] 或 [剔除], 剔除的需简要说明原因。

## 第二步: 多空分类
仅对标注为 [相关] 的新闻进行以下分类:

### 利多因素
(选出对玉米价格有利的因素, 标注影响程度: 强/中/弱, 并引用对应的新闻标题原文)

### 利空因素
(选出对玉米价格不利的因素, 标注影响程度: 强/中/弱, 并引用对应的新闻标题原文)

### 综合判断
(综合所有信息, 给出未来1-4周价格方向判断 + 置信度: 高/中/低)

### 做多信心指数
(0-100, 100表示极度看多)

注意: 只分析 [相关] 新闻中有实质内容的信息, 不要编造任何新闻。

{news_text[:3500]}
"""
    result = _call_deepseek(prompt, 2500)
    if not result:
        return "DeepSeek API 调用失败"
    return result


def get_live_anomaly_data() -> str:
    """读取最新实测气候/卫星数据"""
    climate = pd.read_csv(os.path.join(DATA_DIR, "corn_climate_daily.csv"), dtype={"date": str})
    satellite = pd.read_csv(os.path.join(DATA_DIR, "corn_satellite_daily.csv"), dtype={"date": str})

    for df in [climate, satellite]:
        df["month"] = df["date"].str[4:6].astype(int)
        df["day"] = df["date"].str[6:8].astype(int)
        df["year"] = df["date"].str[:4].astype(int)

    latest_c = climate["date"].max()
    m = int(latest_c[4:6])
    d = int(latest_c[6:8])
    this_year = climate["year"].max()
    past_years = [y for y in range(this_year - 5, this_year)
                  if y in sorted(climate["year"].unique())]
    if not past_years:
        past_years = sorted(climate["year"].unique())[-6:-1]

    lines = [f"实测数据截止: {latest_c} (NASA POWER, 延迟约3-4天)", ""]

    for region_name, stations in CLIMATE_REGIONS.items():
        lines.append(f"## {region_name}:")
        for st in stations:
            parts = []
            c_row = climate[(climate["year"] == this_year) & (climate["month"] == m) &
                            (climate["day"] == d) & (climate["station"] == st)]
            if len(c_row) == 0:
                continue
            temp_now = c_row["temp_C"].values[0]
            precip_now = c_row["precip_mm"].values[0]
            temp_past, precip_past = [], []
            for yr in past_years:
                cr = climate[(climate["year"] == yr) & (climate["month"] == m) &
                             (climate["day"] == d) & (climate["station"] == st)]
                if len(cr) > 0:
                    for arr, val in [(temp_past, cr["temp_C"].values[0]),
                                      (precip_past, cr["precip_mm"].values[0])]:
                        if pd.notna(val) and val == val:
                            arr.append(val)
            if temp_past:
                t_avg = np.mean(temp_past)
                t_diff = float(temp_now) - t_avg
                t_label = "偏高" if t_diff > 1 else ("偏低" if t_diff < -1 else "正常")
                parts.append(f"气温 {temp_now:.1f}°C (5年均 {t_avg:.1f}°C, {t_label}{t_diff:+.1f})")
            if precip_past:
                p_avg = np.mean(precip_past)
                p_diff = float(precip_now) - p_avg
                p_label = "偏多" if p_diff > 1 else ("偏少" if p_diff < -1 else "正常")
                parts.append(f"降水 {precip_now:.1f}mm (5年均 {p_avg:.1f}mm, {p_label}{p_diff:+.1f})")

            s_row = satellite[(satellite["year"] == this_year) & (satellite["month"] == m) &
                              (satellite["day"] == d) & (satellite["station"] == st)]
            if len(s_row) > 0:
                for elem, ename in [("GWETROOT", "根区土壤"), ("GWETTOP", "表层土壤")]:
                    if elem in s_row.columns:
                        v_now = s_row[elem].values[0]
                        if pd.isna(v_now) or v_now == "" or str(v_now) == "nan":
                            continue
                        v_now = float(v_now)
                        v_past = []
                        for yr in past_years:
                            sr = satellite[(satellite["year"] == yr) & (satellite["month"] == m) &
                                           (satellite["day"] == d) & (satellite["station"] == st)]
                            if len(sr) > 0:
                                sv = sr[elem].values[0]
                                if pd.notna(sv) and str(sv) != "" and str(sv) != "nan":
                                    try: v_past.append(float(sv))
                                    except: pass
                        if v_past:
                            v_avg = np.mean(v_past)
                            v_diff = v_now - v_avg
                            v_label = "偏高" if v_diff > 0.03 else ("偏低" if v_diff < -0.03 else "正常")
                            parts.append(f"{ename} {v_now:.2f}(5年均{v_avg:.2f}, {v_label}{v_diff:+.2f})")
            if parts:
                lines.append(f"  {st}: {'; '.join(parts)}")
        lines.append("")
    lines.append("(以上数据从 NASA POWER 数据库实时读取, 非 AI 推测)")
    return "\n".join(lines)


def fetch_news_brief() -> str:
    """全网搜集真实新闻 → DeepSeek 分类"""
    print("  [1/2] 从百度新闻抓取真实新闻...", file=sys.stderr)
    raw_news = collect_corn_news()
    if not raw_news or "未能获取" in raw_news:
        return raw_news

    print("  [2/2] DeepSeek 分析新闻 (利多/利空)...", file=sys.stderr)
    analysis = classify_news_with_ai(raw_news)
    if not analysis or "失败" in analysis:
        return raw_news + "\n\n(AI 分类暂不可用, 以上为原始新闻抓取结果)"

    return raw_news + "\n\n---\n## AI 智能分析 (DeepSeek)\n" + analysis


def analyze_supply_demand_current(live_data: str = "") -> str:
    if not DEEPSEEK_API_KEY:
        return "DeepSeek API 不可用"

    prompt = f"""请基于以下实测天气数据, 分析当前中国玉米供需格局:

{live_data[:1200]}

请回答:
1. 当前时点的供需平衡状态如何?
2. 未来1个月最大的上行风险是什么?
3. 未来1个月最大的下行风险是什么?
4. 做多信心指数(0-100)当前是多少? 为什么?
"""
    result = _call_deepseek(prompt, 1200)
    return result or "DeepSeek API 不可用"


def generate_weight_proposal(
    feature_names: list[str],
    seasonal_info: dict,
    live_data: str = "",
    news_brief: str = "",
    supply_demand_summary: str = "",
) -> tuple[str, dict[str, float], dict[str, str]]:
    features_str = "\n".join(f"  {i+1}. {n}" for i, n in enumerate(feature_names))

    prompt = f"""你是一个中国玉米期货量化分析师。根据以下实时信息, 给ML模型的每个特征分配权重。

## 当前背景
日期: {date.today()}
季节: {seasonal_info['season']}
关键生长期: {seasonal_info['is_critical']}

## 实测天气
{live_data[:1000]}

## 真实新闻 (百度搜索抓取 + DeepSeek分析)
{news_brief[:1200]}

## 供需库存数据 (USDA WASDE + CASDE + 期货库存)
{supply_demand_summary[:1500]}

## 特征列表 ({len(feature_names)}个)
{features_str}

## 任务
给每个特征分配权重(0-1, 总和=1), 并解释原因.
权重必须引用实测数据或新闻原文中的具体数字.
特别注意: 供需库存类特征(inv_, usda_, casde_开头)应根据当前库存水平和供需数据调整权重.

JSON格式要求 (输出纯JSON, 键名用特征原名):
{{"weights": {{"feat1": 0.05, "feat2": 0.08}}, "reasoning": {{"feat1": "理由"}}, "summary": "概述"}}

注意: 只需输出这段JSON, 不要其他文字。权重总和=1。"""
    result = _call_deepseek(prompt, 2500)

    weights, reasoning, summary = {}, {}, "AI不可用"
    if result:
        try:
            start = result.find("{")
            end = result.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(result[start:end])
            else:
                parsed = json.loads(result)
            raw = parsed.get("weights", {})
            reasoning = parsed.get("reasoning", {})
            summary = parsed.get("summary", "AI分析完成")
            matched = {}
            for n in feature_names:
                matched[n] = raw.get(n, raw.get(n.replace("_anom_", "_anom_"), 0))
            total = sum(matched.values())
            weights = {n: v / total for n, v in matched.items()} if total > 0 else {
                n: 1.0 / len(feature_names) for n in feature_names
            }
        except Exception as e:
            print(f"[权重解析] {e}", file=sys.stderr)
            weights = {n: 1.0 / len(feature_names) for n in feature_names}

    return summary, weights, reasoning


def rule_based_weights(feature_names, seasonal_info):
    from seasonal_calendar import get_feature_weight_modifiers
    modifiers = get_feature_weight_modifiers()
    w = {}
    for name in feature_names:
        base = 1.0 / len(feature_names)
        for prefix, mod in modifiers.items():
            if name.startswith(prefix) or name == prefix:
                base *= mod
                break
        w[name] = base
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}


def get_inventory_summary() -> str:
    """读取期货库存数据并生成可读摘要"""
    import numpy as np
    inv_path = os.path.join(DATA_DIR, "corn_inventory_daily.csv")
    if not os.path.exists(inv_path):
        try:
            ak = _get_ak()
            df = ak.futures_inventory_em(symbol="玉米")
            df = df.rename(columns={"日期": "date", "库存": "inventory", "增减": "inv_change"})
            df["date"] = pd.to_datetime(df["date"])
            df[["date", "inventory", "inv_change"]].to_csv(inv_path, index=False, encoding="utf-8-sig")
        except Exception:
            return "【玉米期货库存】数据暂不可用"

    df = pd.read_csv(inv_path, parse_dates=["date"])
    df = df.sort_values("date")
    latest = df.iloc[-1]
    week_ago_idx = max(0, len(df) - 6)
    week_ago = df.iloc[week_ago_idx]
    month_ago_idx = max(0, len(df) - 22)
    month_ago = df.iloc[month_ago_idx]

    inv_now = latest["inventory"]
    inv_wk = week_ago["inventory"]
    inv_mo = month_ago["inventory"]
    chg_now = latest.get("inv_change", np.nan)

    lines = [
        "=" * 68,
        "  【玉米期货库存 (大商所注册仓单)】",
        "=" * 68,
        f"  数据截止: {latest['date'].strftime('%Y-%m-%d')}",
        f"  最新库存: {inv_now:.0f} 吨  ({inv_now/10000:.2f} 万吨)",
        f"  日增减量: {chg_now:+.0f} 吨" if not np.isnan(chg_now) else "",
        f"  较一周前: {inv_now - inv_wk:+.0f} 吨 ({(inv_now/inv_wk - 1)*100:+.1f}%)",
        f"  较一月前: {inv_now - inv_mo:+.0f} 吨 ({(inv_now/inv_mo - 1)*100:+.1f}%)",
        "",
    ]
    trend = "↑ 库存累积（供应宽松，利空）" if inv_now > inv_mo * 1.05 else (
        "↓ 库存下降（供应收紧，利多）" if inv_now < inv_mo * 0.95 else "→ 库存平稳（供需平衡）")
    lines.append(f"  趋势判断: {trend}")

    recent = df.tail(10)
    lines.append("  近10日库存变化:")
    for _, r in recent.iterrows():
        chg_str = f" ({r['inv_change']:+.0f})" if not pd.isna(r.get("inv_change", np.nan)) else ""
        lines.append(f"    {r['date'].strftime('%m-%d')}: {r['inventory']:.0f}{chg_str}")

    lines.append("")
    lines.append("  数据来源: 东方财富 futures_inventory_em(symbol='玉米')")
    return "\n".join(lines)


def get_wasde_summary() -> str:
    """读取 USDA WASDE 玉米供需数据并生成可读摘要"""
    path = os.path.join(DATA_DIR, "usda_wasde_corn.csv")
    if not os.path.exists(path):
        return "【USDA WASDE 供需报告】数据暂不可用"

    df = pd.read_csv(path)
    df = df.sort_values("report_date")
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    def _diff(label, unit, curr, prev_val, fmt=".1f"):
        d = curr - prev_val
        sign = "+" if d > 0 else ""
        return f"  {label:<20s}: {curr:{fmt}} {unit}  (较上年 {sign}{d:{fmt}})"

    lines = [
        "=" * 68,
        f"  【USDA WASDE 玉米供需报告】",
        f"  报告日期: {latest['report_date']}  |  市场年度: {latest.get('market_year','?')}",
        "=" * 68,
        _diff("全球产量", "亿吨", latest["global_production_mmt"] / 100, prev["global_production_mmt"] / 100),
        _diff("全球消费", "亿吨", latest["global_consumption_mmt"] / 100, prev["global_consumption_mmt"] / 100),
        _diff("全球期末库存", "亿吨", latest["global_ending_stocks_mmt"] / 100, prev["global_ending_stocks_mmt"] / 100),
        f"  全球库存消费比: {latest['global_stocks_to_use']:.1f}%  (上年 {prev['global_stocks_to_use']:.1f}%)",
        "",
        _diff("中国产量", "亿吨", latest["china_production_mmt"] / 100, prev["china_production_mmt"] / 100),
        _diff("中国进口", "百万吨", latest["china_imports_mmt"] / 100, prev["china_imports_mmt"] / 100, ".2f"),
        _diff("中国期末库存", "亿吨", latest["china_ending_stocks_mmt"] / 100, prev["china_ending_stocks_mmt"] / 100),
        "",
        _diff("美国产量", "亿吨", latest["us_production_mmt"] / 100, prev["us_production_mmt"] / 100),
        f"  美国农场均价: ${latest['us_farm_price_usd_bu']:.2f}/蒲  (上年 ${prev['us_farm_price_usd_bu']:.2f})",
        _diff("巴西产量", "亿吨", latest["brazil_production_mmt"] / 100, prev["brazil_production_mmt"] / 100),
        _diff("阿根廷产量", "亿吨", latest["argentina_production_mmt"] / 100, prev["argentina_production_mmt"] / 100),
        "",
    ]

    stocks_change = latest["global_ending_stocks_mmt"] - prev["global_ending_stocks_mmt"]
    price_change = latest["us_farm_price_usd_bu"] - prev["us_farm_price_usd_bu"]
    if stocks_change < -5:
        bias = "★★★ 利多 — 全球库存显著下降，供应收紧"
    elif stocks_change < 0:
        bias = "★★☆ 偏多 — 全球库存小幅下降"
    elif stocks_change > 5:
        bias = "☆☆☆ 利空 — 全球库存显著上升"
    else:
        bias = "★☆☆ 中性 — 全球库存变化不大"
    lines.append(f"  综合判断: {bias}")

    if price_change != 0:
        lines.append(f"  美国农场均价同比: {price_change:+.2f} $/蒲 (预示新季价格方向)")
    lines.append("  数据来源: USDA FAS PSD + chinagrain.cn 转载")
    return "\n".join(lines)


def get_casde_summary() -> str:
    """读取 CASDE 中国玉米供需数据并生成可读摘要"""
    path = os.path.join(DATA_DIR, "casde_corn_supply_demand.csv")
    if not os.path.exists(path):
        return "【CASDE 中国供需报告】数据暂不可用"

    df = pd.read_csv(path)
    df = df.sort_values("report_date")
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    def _diff(label, unit, curr, prev_val, fmt=".1f", factor=1):
        d = curr - prev_val
        sign = "+" if d > 0 else ""
        return f"  {label:<18s}: {curr/factor:{fmt}} {unit}  (较上年 {sign}{d/factor:{fmt}})"

    area_cur = latest["corn_area_kha"]
    area_prev = prev["corn_area_kha"]
    yield_cur = latest["corn_yield_kg_ha"]

    lines = [
        "=" * 68,
        f"  【CASDE 中国玉米供需形势】",
        f"  报告日期: {latest['report_date']}  |  市场年度: {latest.get('market_year','?')}",
        "=" * 68,
        _diff("种植面积", "千公顷", area_cur, area_prev, ".0f"),
        _diff("产量", "万吨", latest["corn_production_mmt"] * 100, prev["corn_production_mmt"] * 100, ".0f", 1),
        f"  单产: {yield_cur:.0f} 公斤/公顷 ({yield_cur/15:.1f} 公斤/亩)",
        "",
        _diff("饲用消费", "万吨", latest["corn_feed_consumption_mmt"] * 100, prev["corn_feed_consumption_mmt"] * 100, ".0f", 1),
        _diff("工业消费", "万吨", latest["corn_industrial_consumption_mmt"] * 100, prev["corn_industrial_consumption_mmt"] * 100, ".0f", 1),
        _diff("总消费", "万吨", latest["corn_total_consumption_mmt"] * 100, prev["corn_total_consumption_mmt"] * 100, ".0f", 1),
        "",
        _diff("进口量", "万吨", latest["corn_imports_mmt"] * 100, prev["corn_imports_mmt"] * 100, ".0f", 1),
        "",
    ]

    surplus = (latest["corn_production_mmt"] - latest["corn_total_consumption_mmt"]) * 100
    if surplus > 500:
        bal = f"★★☆ 产大于需 {surplus:.0f}万吨 (供应偏宽松)"
    elif surplus < -1000:
        bal = f"☆☆☆ 产不足需 {abs(surplus):.0f}万吨 (供应偏紧)"
    else:
        bal = f"★☆☆ 产需基本平衡 ({surplus:+.0f}万吨)"
    lines.append(f"  产需平衡: {bal}")

    notes = latest.get("notes", "")
    if notes and not pd.isna(notes):
        lines.append(f"  备注: {notes}")
    lines.append("  数据来源: 农业农村部 CASDE + chinagrain.cn 转载")
    return "\n".join(lines)


def get_supply_demand_summary() -> str:
    parts = []
    try:
        parts.append(get_inventory_summary())
    except Exception as e:
        parts.append(f"【库存】读取失败: {e}")
    try:
        parts.append(get_wasde_summary())
    except Exception as e:
        parts.append(f"【USDA】读取失败: {e}")
    try:
        parts.append(get_casde_summary())
    except Exception as e:
        parts.append(f"【CASDE】读取失败: {e}")
    return "\n".join(parts)


def generate_final_report(feature_names, ai_weights, ai_reasoning, ai_summary,
                          seasonal_info, news_brief, sd_analysis, live_data, used_ai,
                          supply_demand_summary=""):
    lines = [
        "=" * 68,
        f"  玉米期货 AI 权重分析报告 (真实新闻版)",
        f"  生成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 68, "",
        f"【当前季节】{seasonal_info['month']}月 {seasonal_info['season']}",
        f"  关键期:{'是★' if seasonal_info['is_critical'] else '否'}",
        "", live_data, "",
    ]
    if supply_demand_summary:
        lines += [supply_demand_summary, ""]
    if used_ai:
        lines += ["【AI 权重摘要】", f"  {ai_summary}", "", "【权重分配】"]
        for name, w in sorted(ai_weights.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(w * 40 + 1)
            lines.append(f"  {name:<28s} {w:.3f} ({w:.1%}) {bar}")
            if name in ai_reasoning:
                lines.append(f"    → {ai_reasoning[name][:100]}")
    else:
        lines += ["【权重(规则引擎)】"]
        rw = rule_based_weights(feature_names, seasonal_info)
        for name, w in sorted(rw.items(), key=lambda x: x[1], reverse=True)[:12]:
            lines.append(f"  {name:<28s} {w:.3f} ({w:.1%})")
    if news_brief and "未能获取" not in news_brief:
        lines += ["", "【真实新闻情报 (akshare/东方财富)】", news_brief[:2000]]
    lines += ["", "─" * 68,
              f"  数据来源: akshare/东方财富(期货库存) + USDA FAS PSD(全球供需) + 农业农村部 CASDE(中国供需)",
              f"  新闻来源: akshare stock_news_em (6只农业股, 真实+时间戳) + 新浪财经",
              f"  分析引擎: {'DeepSeek API' if used_ai else '规则引擎 (离线模式)'}",
              "─" * 68]
    return "\n".join(lines)


if __name__ == "__main__":
    print("新闻情报 v2 — 真实抓取测试")
    if DEEPSEEK_API_KEY:
        print("[DeepSeek Key] 已设置")
    else:
        print("[DeepSeek Key] 未设置, 将用网页抓取")
    print("\n开始抓取新闻...")
    brief = fetch_news_brief()
    print(brief[:1000])
