#!/usr/bin/env python3
"""
新闻情报收集 + 利多/利空分类 + 实时天气异常注入
来源: DeepSeek API + NASA POWER 实际观测数据
"""

import json
import os
import sys
from datetime import datetime, date
from typing import Any, Optional

import numpy as np
import pandas as pd

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

CLIMATE_REGIONS = {
    "东北": ["哈尔滨", "长春", "沈阳"],
    "华北黄淮": ["呼和浩特", "济南", "郑州"],
}

ELEMENTS = {
    "temp_C": ("气温", "°C"),
    "precip_mm": ("降水", "mm"),
    "GWETROOT": ("根区土壤水分", "0-1"),
    "GWETTOP": ("表层土壤水分", "0-1"),
    "GWETPROF": ("剖面土壤水分", "0-1"),
    "ALLSKY_SFC_SW_DWN": ("太阳辐射", "MJ/m²"),
    "RH2M": ("相对湿度", "%"),
}


def _call_deepseek(prompt: str) -> str:
    if not DEEPSEEK_API_KEY:
        return ""
    try:
        import requests
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
                "max_tokens": 2000,
            },
            timeout=45,
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        print(f"[DeepSeek API] {resp.status_code}: {resp.text[:200]}", file=sys.stderr)
        return ""
    except Exception as e:
        print(f"[DeepSeek API] 调用失败: {e}", file=sys.stderr)
        return ""


def get_live_anomaly_data(today: Optional[date] = None) -> str:
    """读取最新实测气候/卫星数据, 与过去5年同日对比, 生成文本摘要"""
    if today is None:
        today = date.today()

    climate = pd.read_csv(os.path.join(DATA_DIR, "corn_climate_daily.csv"), dtype={"date": str})
    satellite = pd.read_csv(os.path.join(DATA_DIR, "corn_satellite_daily.csv"), dtype={"date": str})

    for df in [climate, satellite]:
        df["month"] = df["date"].str[4:6].astype(int)
        df["day"] = df["date"].str[6:8].astype(int)
        df["year"] = df["date"].str[:4].astype(int)

    latest_c = climate["date"].max()
    latest_s = satellite["date"].max()
    m = int(latest_c[4:6])
    d = int(latest_c[6:8])

    lines = [f"实测数据截止: {latest_c} (NASA延迟约3-4天)", ""]

    available_years = sorted(climate["year"].unique())
    this_year = max(available_years)
    past_years = [y for y in range(this_year - 5, this_year) if y in available_years]
    if not past_years:
        past_years = available_years[-6:-1]

    for region_name, stations in CLIMATE_REGIONS.items():
        lines.append(f"## {region_name}产区 ({'、'.join(stations)}):")
        for st in stations:
            parts = []
            # climate
            c_row = climate[(climate["year"] == this_year) & (climate["month"] == m) & (climate["day"] == d) & (climate["station"] == st)]
            if len(c_row) > 0:
                temp_now = c_row["temp_C"].values[0]
                precip_now = c_row["precip_mm"].values[0]
                temp_past = []
                precip_past = []
                for yr in past_years:
                    cr = climate[(climate["year"] == yr) & (climate["month"] == m) & (climate["day"] == d) & (climate["station"] == st)]
                    if len(cr) > 0:
                        tv = cr["temp_C"].values[0]
                        pv = cr["precip_mm"].values[0]
                        if pd.notna(tv) and tv == tv:
                            temp_past.append(tv)
                        if pd.notna(pv) and pv == pv:
                            precip_past.append(pv)
                t_avg = round(np.mean(temp_past), 1) if temp_past else "?"
                p_avg = round(np.mean(precip_past), 1) if precip_past else "?"
                t_diff = round(float(temp_now) - float(t_avg), 1) if temp_past else "?"
                p_diff = round(float(precip_now) - float(p_avg), 1) if precip_past else "?"
                if temp_past:
                    t_label = "偏高" if t_diff > 1 else ("偏低" if t_diff < -1 else "正常")
                    parts.append(f"气温 {temp_now:.1f}°C (5年均 {t_avg}°C, {t_label}{t_diff:+.1f})")
                if precip_past:
                    p_label = "偏多" if p_diff > 1 else ("偏少" if p_diff < -1 else "正常")
                    parts.append(f"降水 {precip_now:.1f}mm (5年均 {p_avg}mm, {p_label}{p_diff:+.1f})")

            # satellite
            s_row = satellite[(satellite["year"] == this_year) & (satellite["month"] == m) & (satellite["day"] == d) & (satellite["station"] == st)]
            if len(s_row) > 0:
                for elem, ename in [("GWETROOT", "根区土壤"), ("GWETTOP", "表层土壤"), ("RH2M", "相对湿度")]:
                    if elem in s_row.columns:
                        v_now = s_row[elem].values[0]
                        if pd.isna(v_now) or v_now == "" or v_now == "nan":
                            continue
                        v_now = float(v_now)
                        v_past = []
                        for yr in past_years:
                            sr = satellite[(satellite["year"] == yr) & (satellite["month"] == m) & (satellite["day"] == d) & (satellite["station"] == st)]
                            if len(sr) > 0:
                                sv = sr[elem].values[0]
                                if pd.notna(sv) and str(sv) != "" and str(sv) != "nan":
                                    try:
                                        v_past.append(float(sv))
                                    except:
                                        pass
                        if v_past:
                            v_avg = round(np.mean(v_past), 2)
                            v_diff = round(float(v_now) - v_avg, 2)
                            th = 3 if elem == "RH2M" else 0.03
                            v_label = "偏高" if v_diff > th else ("偏低" if v_diff < -th else "正常")
                            parts.append(f"{ename} {v_now:.2f} (5年均 {v_avg}, {v_label}{v_diff:+.2f})")

            if parts:
                lines.append(f"  {st}: {'; '.join(parts)}")
        lines.append("")

    drought_lines = []
    for region_name, stations in CLIMATE_REGIONS.items():
        precip_vals, sm_vals = [], []
        for st in stations:
            cr = climate[(climate["year"] == this_year) & (climate["month"] == m) & (climate["day"] == d) & (climate["station"] == st)]
            sr = satellite[(satellite["year"] == this_year) & (satellite["month"] == m) & (satellite["day"] == d) & (satellite["station"] == st)]
            if len(cr) > 0:
                pv = cr["precip_mm"].values[0]
                if pd.notna(pv) and pv == pv:
                    precip_vals.append(float(pv))
            if len(sr) > 0:
                if "GWETROOT" in sr.columns:
                    sv = sr["GWETROOT"].values[0]
                    if pd.notna(sv) and str(sv) != "" and str(sv) != "nan":
                        sm_vals.append(float(sv))
        if precip_vals and sm_vals:
            p_mean = np.mean(precip_vals)
            s_mean = np.mean(sm_vals)
            if p_mean < 1 and s_mean < 0.45:
                status = "⚠️ 干旱风险: 降水和土壤水分双低"
            elif p_mean < 1:
                status = "⚡ 降水偏少"
            elif s_mean < 0.45:
                status = "🔍 土壤偏干"
            else:
                status = "✅ 水分正常"
            drought_lines.append(f"  {region_name}: {status}")

    if drought_lines:
        lines.append("## 水分条件评估:")
        lines.extend(drought_lines)
        lines.append("")

    lines.append("(以上实测数据直接从 NASA POWER 数据库读取, 非AI推测)")
    return "\n".join(lines)


def fetch_news_brief(today: Optional[date] = None) -> str:
    if today is None:
        today = date.today()
    date_str = today.strftime("%Y年%m月%d日")

    prompt = f"""请以{date_str}为中心, 搜集并总结近期(最近1-2周)影响中国玉米期货市场的重要新闻和事件。

请按以下格式输出:

## 近期新闻概要
(用5-8条要闻概括, 每条100字以内)

## 利多因素
(列出所有对玉米价格上涨有利的因素, 每条附带影响程度: 强/中/弱)

## 利空因素  
(列出所有对玉米价格下跌有利的因素, 每条附带影响程度: 强/中/弱)

## 综合判断
(综合利多利空, 给出你对未来1-4周玉米价格方向的判断, 以及置信度: 高/中/低)
"""
    result = _call_deepseek(prompt)
    if not result:
        return "DeepSeek API 不可用 (请设置 DEEPSEEK_API_KEY 环境变量)"
    return result


def analyze_supply_demand_current(live_data: str = "") -> str:
    today = date.today()
    date_str = today.strftime("%Y年%m月%d日")

    prompt = f"""请分析{date_str}当前时点的中国玉米供需格局:

已知条件:
- 当前处于玉米生长季, 春玉米已播种, 夏玉米即将播种
- 生猪存栏是玉米最大需求方(占60%)
- 深加工(淀粉/酒精)是第二大需求方
- 当前国际玉米价格(CBOT)波动会影响进口成本和国内市场情绪
- 中国中储粮的收储和抛储政策是重要的价格调节手段

{live_data}

请回答:
1. 当前时点的供需平衡状态如何?
2. 未来1个月最大的上行风险是什么?
3. 未来1个月最大的下行风险是什么?
4. 如果让你给"做多信心指数"打分(0-100), 当前是多少? 为什么?
"""
    result = _call_deepseek(prompt)
    return result or "DeepSeek API 不可用"


def generate_weight_proposal(
    feature_names: list[str],
    seasonal_info: dict,
    live_data: str = "",
    today: Optional[date] = None,
) -> tuple[str, dict[str, float], dict[str, str]]:
    if today is None:
        today = date.today()

    news_brief = fetch_news_brief(today)
    sd_analysis = analyze_supply_demand_current(live_data)

    features_str = "\n".join(f"  {i+1}. {n}" for i, n in enumerate(feature_names))

    prompt = f"""你是一个中国玉米期货量化分析师。现在需要你根据以下信息, 给机器学习模型的每个特征分配权重。

## 当前市场背景
日期: {today.strftime("%Y-%m-%d")}
季节: {seasonal_info['season']}
是否关键生长期: {seasonal_info['is_critical']}

## 实测天气/土壤数据 (NASA POWER, 基于6个玉米带气象站实时观测)
{live_data[:1200]}

## 新闻情报
{news_brief[:1200]}

## 供需分析
{sd_analysis[:600]}

## 模型特征列表 (共{len(feature_names)}个)
{features_str}

## 任务
请为每个特征分配一个权重系数(0.0~1.0之间, 所有系数之和=1.0), 并解释为什么。

要求:
1. 权重调整必须引用上述实测数据中的具体数字——不能仅凭季节猜测
2. 如实测数据显示某产区气温/降水异常, 相应的 t_anom/p_anom 特征应加减权重
3. 如实测数据显示土壤水分偏低, sm_* 特征应加重
4. 季节关键期(7-8月)的气候类特征应给更高权重
5. 新闻中提到的供需变化(直播/进口/政策)应反映到相应特征权重
6. 每个特征的权重调整必须有理有据, 引用实测值或新闻原文

请严格按以下JSON格式输出(不要输出其他内容):

```json
{{
  "weights": {{
    "特征名1": 0.05,
    "特征名2": 0.08,
    ...
  }},
  "reasoning": {{
    "特征名1": "当前XX站实测气温=YY°C, 比5年均值偏高/偏低Z°C, 因此...",
    ...
  }},
  "summary": "整体权重分配逻辑概述"
}}
```

注意: 不需要为不在列表中的特征设置权重。所有权重之和必须等于1.0。"""
    result = _call_deepseek(prompt)

    weights: dict[str, float] = {}
    reasoning: dict[str, str] = {}
    summary = "AI 分析不可用，使用默认均权"

    if result:
        try:
            start = result.find("```json")
            end = result.find("```", start + 7) if start >= 0 else -1
            if start >= 0 and end > start:
                json_str = result[start + 7:end]
            else:
                json_str = result
            parsed = json.loads(json_str)
            raw_weights = parsed.get("weights", {})
            reasoning = parsed.get("reasoning", {})
            summary = parsed.get("summary", "AI 权重分析完成")

            matched = {n: raw_weights.get(n, 0) for n in feature_names}
            total = sum(matched.values())
            if total > 0:
                weights = {n: v / total for n, v in matched.items()}
            else:
                weights = {n: 1.0 / len(feature_names) for n in feature_names}
        except Exception as e:
            print(f"[权重解析] {e}", file=sys.stderr)
            weights = {n: 1.0 / len(feature_names) for n in feature_names}

    return summary, weights, reasoning


def rule_based_weights(
    feature_names: list[str],
    seasonal_info: dict,
) -> dict[str, float]:
    from seasonal_calendar import get_feature_weight_modifiers
    modifiers = get_feature_weight_modifiers()

    weights = {}
    for name in feature_names:
        w = 1.0 / len(feature_names)
        for prefix, mod in modifiers.items():
            if name.startswith(prefix) or name == prefix:
                w *= mod
                break
        weights[name] = w

    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def generate_final_report(
    feature_names: list[str],
    ai_weights: dict[str, float],
    ai_reasoning: dict[str, str],
    ai_summary: str,
    seasonal_info: dict,
    news_brief: str,
    sd_analysis: str,
    live_data: str,
    used_ai: bool,
) -> str:
    lines = [
        "=" * 68,
        f"  玉米期货 ML 模型 — AI 权重分析报告",
        f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 68,
        "",
        "【当前季节】",
        f"  月份: {seasonal_info['month']}月",
        f"  阶段: {seasonal_info['season']}",
        f"  关键生长期: {'是 ★' if seasonal_info['is_critical'] else '否'}",
        "",
        live_data,
        "",
    ]

    if used_ai:
        lines += [
            "【AI 权重分析摘要 (DeepSeek)】",
            f"  {ai_summary}",
            "",
            "【权重分配】",
        ]
        sorted_w = sorted(ai_weights.items(), key=lambda x: x[1], reverse=True)
        for name, w in sorted_w:
            reason = ai_reasoning.get(name, "")
            bar = "█" * int(w * 40)
            lines.append(f"  {name:<28s} {w:.3f} ({w:.1%}) {bar}")
            if reason:
                lines.append(f"    → {reason}")
    else:
        lines += ["【权重分配 (规则引擎, AI 不可用)】", ""]
        rule_w = rule_based_weights(feature_names, seasonal_info)
        sorted_w = sorted(rule_w.items(), key=lambda x: x[1], reverse=True)
        for name, w in sorted_w[:12]:
            bar = "█" * int(w * 40)
            lines.append(f"  {name:<28s} {w:.3f} ({w:.1%}) {bar}")

    if news_brief and news_brief != "DeepSeek API 不可用":
        lines += ["", "【近期新闻情报】", news_brief[:800]]

    lines += [
        "",
        "─" * 68,
        f"  权重来源: {'DeepSeek API (含实测天气数据)' if used_ai else '本地规则引擎'}",
        "─" * 68,
    ]

    return "\n".join(lines)


if __name__ == "__main__":
    from seasonal_calendar import get_season_info
    info = get_season_info()
    print(f"当前季节: {info['season']}")

    if DEEPSEEK_API_KEY:
        print("\n正在调用 DeepSeek 获取新闻情报...")
        brief = fetch_news_brief()
        print(brief[:500])
    else:
        print("\n未设置 DEEPSEEK_API_KEY")
        print("export DEEPSEEK_API_KEY=你的key 来启用AI分析")
