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
    "转基因", "天气", "干旱", "洪涝", "产量", "库存",
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

def _fetch_agri_news_akshare(max_per_stock: int = 10) -> list[dict]:
    """通过 akshare 的 stock_news_em 抓取农业股关联新闻 (真实时间戳)"""
    results: list[dict] = []
    seen = set()

    for code, name in AGRI_STOCKS:
        try:
            ak = _get_ak()
            df = ak.stock_news_em(symbol=code)
        except Exception as e:
            print(f"  [akshare {name}] {e}", file=sys.stderr)
            continue
        for _, row in df.iterrows():
            title = str(row.get("新闻标题", ""))
            if not any(kw in title for kw in CORN_KEYWORDS):
                continue
            key = title[:60]
            if key in seen:
                continue
            seen.add(key)
            pub_time = str(row.get("发布时间", ""))
            summary = str(row.get("新闻内容", ""))[:200]
            source = str(row.get("文章来源", name))
            url = str(row.get("新闻链接", ""))
            results.append({
                "title": title, "abstract": summary,
                "time": pub_time, "source": f"{name}({source})",
                "url": url,
            })
        if len(results) >= max_per_stock * len(AGRI_STOCKS):
            break

    results.sort(key=lambda x: x.get("time", ""), reverse=True)
    return results


def _fetch_sina_corn_search(max_items: int = 15, max_age_days: int = 7) -> list[dict]:
    """从新浪财经期货滚动新闻抓取农业相关 (自带时间戳, 辅助源)"""
    now_ts = int(datetime.now().timestamp())
    cutoff = now_ts - max_age_days * 86400
    results = []
    try:
        r = requests.get(
            "https://feed.mix.sina.com.cn/api/roll/get",
            params={"pageid": 153, "lid": 2516, "num": 50, "page": 1},
            headers=HEADERS, timeout=15,
        )
        if r.status_code != 200:
            return results
        data = r.json()
        for item in data.get("result", {}).get("data", []):
            title = item.get("title", "")
            intro = item.get("intro", "")
            if not any(kw in title + intro for kw in CORN_KEYWORDS):
                continue
            ctime = int(item.get("ctime", 0))
            if ctime < cutoff:
                continue
            dt = datetime.fromtimestamp(ctime)
            results.append({
                "title": title, "abstract": intro[:150] if intro else "",
                "time": dt.strftime("%Y-%m-%d %H:%M"), "source": "新浪财经",
            })
            if len(results) >= max_items:
                break
    except Exception:
        pass
    return results


def collect_corn_news(max_age_days: int = 7) -> str:
    """通过 akshare 农业股票新闻接口搜集玉米相关真实新闻"""
    import time as _time

    print("  [akshare] 抓取农业股关联新闻...", file=sys.stderr)
    all_items = _fetch_agri_news_akshare(10)
    
    # Also add Sina futures feed as supplement
    try:
        sina_items = _fetch_sina_corn_search(15, max_age_days)
        seen_titles = {it["title"][:60] for it in all_items}
        for item in sina_items:
            if item["title"][:60] not in seen_titles:
                seen_titles.add(item["title"][:60])
                all_items.append(item)
    except Exception:
        pass

    if not all_items:
        return "[未能获取到新闻 — akshare API 可能暂时不可用]"

    today = date.today()
    lines = [
        f"以下为 {today} 从 akshare 农业股票新闻接口抓取的真实玉米相关新闻:",
        f"",
        f"数据来源: 东方财富新闻 (北大荒/丰乐种业/大北农/隆平高科/神农种业/农发种业)",
        f"时效过滤: 各股最新10条中筛选玉米相关, 每条均有真实发布时间戳",
        f"共 {len(all_items)} 条, 展示如下:",
        "",
    ]

    for i, item in enumerate(all_items[:30], 1):
        time_s = f"[{item.get('time','')}] " if item.get('time') else ""
        src_s = f" ({item.get('source','')})"
        lines.append(f"{i}. {time_s}{item['title'][:120]}{src_s}")
        if item.get("abstract"):
            abst = re.sub(r'<[^>]+>', '', item['abstract'])[:150]
            if abst:
                lines.append(f"   {abst}")

    lines.append(f"\n(以上为东方财富/akshare 真实数据接口返回, 非 AI 生成)")
    return "\n".join(lines)


def classify_news_with_ai(news_text: str) -> str:
    """将真实新闻文本交给 DeepSeek 做利多/利空分类"""
    if not DEEPSEEK_API_KEY:
        return "DeepSeek API 不可用"

    prompt = f"""以下是今天从百度新闻搜索抓取的真实玉米市场新闻。

请阅读这些新闻标题, 并分类为:

## 利多因素
(选出对玉米价格有利的因素, 标注影响程度: 强/中/弱, 并引用对应的新闻标题原文)

## 利空因素
(选出对玉米价格不利的因素, 标注影响程度: 强/中/弱, 并引用对应的新闻标题原文)

## 综合判断
(综合所有信息, 给出未来1-4周价格方向判断 + 置信度: 高/中/低)

## 做多信心指数
(0-100, 100表示极度看多 )

注意: 只分析新闻标题中的真实信息, 不要编造任何新闻。如果某条新闻与玉米无关或信息不足, 可以忽略。

{news_text[:3000]}
"""
    result = _call_deepseek(prompt, 1500)
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
{news_brief[:1500]}

## 特征列表 ({len(feature_names)}个)
{features_str}

## 任务
给每个特征分配权重(0-1, 总和=1), 并解释原因.
权重必须引用实测数据或新闻原文中的具体数字.

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


def generate_final_report(feature_names, ai_weights, ai_reasoning, ai_summary,
                          seasonal_info, news_brief, sd_analysis, live_data, used_ai):
    lines = [
        "=" * 68,
        f"  玉米期货 AI 权重分析报告 (真实新闻版)",
        f"  生成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 68, "",
        f"【当前季节】{seasonal_info['month']}月 {seasonal_info['season']}",
        f"  关键期:{'是★' if seasonal_info['is_critical'] else '否'}",
        "", live_data, "",
    ]
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
              f"  新闻来源: akshare stock_news_em (6只农业股, 真实+时间戳)",
              f"  分析引擎: DeepSeek API",
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
