#!/usr/bin/env python3
"""
AI 权重分析器 — 整合季节性 + 新闻情报 + AI分析 → 输出权重报告
用法:
  python3 ai_weight_analyzer.py                    # 使用当前日期, DeepSeek API (如有)
  python3 ai_weight_analyzer.py --offline          # 仅规则引擎
  python3 ai_weight_analyzer.py --date 2025-07-15  # 指定日期
"""

import json
import os
import sys
from datetime import date, datetime

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_PATH = os.path.join(DATA_DIR, "ai_weight_report.txt")

from build_features import FEATURE_COLS, CORE_FEATURES
from seasonal_calendar import get_season_info, get_feature_weight_modifiers, apply_seasonal_weights
from news_intelligence import (
    generate_weight_proposal,
    rule_based_weights,
    generate_final_report,
    fetch_news_brief,
    analyze_supply_demand_current,
    get_live_anomaly_data,
    DEEPSEEK_API_KEY,
)


def main():
    offline = "--offline" in sys.argv
    report_date = date.today()

    for i, arg in enumerate(sys.argv):
        if arg == "--date" and i + 1 < len(sys.argv):
            try:
                report_date = datetime.strptime(sys.argv[i + 1], "%Y-%m-%d").date()
            except ValueError:
                pass

    seasonal_info = get_season_info(report_date)
    use_ai = bool(DEEPSEEK_API_KEY and not offline)

    print("=" * 68)
    print("  AI 权重分析器")
    print("=" * 68)
    print(f"  日期: {report_date}")
    print(f"  季节: {seasonal_info['season']}")
    print(f"  AI 分析: {'启用 (DeepSeek)' if use_ai else '离线 (规则引擎)'}")
    print(f"  特征数: {len(FEATURE_COLS)}")
    print()

    news_brief = ""
    sd_analysis = ""
    ai_weights = {}
    ai_reasoning = {}
    ai_summary = ""

    if use_ai:
        print("[0/3] 读取最新实测天气/土壤数据 ...")
        live_data = get_live_anomaly_data()
        print(f"  ✓ 已读取 {len(live_data)} 字符的实测数据")

        print("[1/3] 正在从百度搜索抓取真实新闻 ...")
        news_brief = fetch_news_brief()
        print(f"  {'✓' if news_brief else '✗'} {'已获取' if news_brief else '失败'}")

        print("[2/3] 正在调用 DeepSeek 分析供需格局 ...")
        sd_analysis = analyze_supply_demand_current(live_data)
        print(f"  {'✓' if sd_analysis else '✗'} {'已获取' if sd_analysis else '失败'}")

        print("[3/3] 正在调用 DeepSeek 生成特征权重建议 (基于实测+新闻)...")
        ai_summary, ai_weights, ai_reasoning = generate_weight_proposal(
            FEATURE_COLS, seasonal_info, live_data, news_brief,
        )
        print(f"  ✓ AI 分析了 {len(ai_weights)} 个特征的权重")
    else:
        print("[离线模式] 使用季节规则引擎分配权重")
        live_data = get_live_anomaly_data()
        print(f"  ✓ 已读取实测数据 (供报告展示)")
        ai_weights = rule_based_weights(FEATURE_COLS, seasonal_info)

    print()

    # 生成报告
    report = generate_final_report(
        FEATURE_COLS, ai_weights, ai_reasoning, ai_summary,
        seasonal_info, news_brief, sd_analysis, live_data, use_ai,
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    # 保存权重 JSON 供 ML 模型使用
    weights_json = os.path.join(DATA_DIR, "ai_weights.json")
    with open(weights_json, "w", encoding="utf-8") as f:
        json.dump({
            "date": report_date.isoformat(),
            "season": seasonal_info,
            "source": "deepseek" if use_ai else "rule_engine",
            "weights": {k: round(v, 6) for k, v in ai_weights.items()},
        }, f, ensure_ascii=False, indent=2)

    print(report)
    print(f"\n报告已保存: {REPORT_PATH}")
    print(f"权重JSON:   {weights_json}")


if __name__ == "__main__":
    main()
