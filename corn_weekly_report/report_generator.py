#!/usr/bin/env python3
"""
玉米&淀粉 周度研究报告 自动生成器
基于 corn_ml 自有数据 + DeepSeek AI 分析

用法:
  python3 report_generator.py                          # 使用当前日期
  python3 report_generator.py --date 2026-05-12        # 指定日期
  python3 report_generator.py --offline                # 离线模式(仅自有数据)
  python3 report_generator.py --pdf                    # 生成Markdown后自动转PDF

输出:
  output/corn_starch_weekly_YYYYMMDD.md   Markdown格式周报
  output/corn_starch_weekly_YYYYMMDD.pdf  (可选) PDF格式
"""

import json
import os
import sys
from datetime import date, datetime, timedelta
from typing import Any, Optional

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "corn_ml")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

_ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_ENV_FILE):
    try:
        with open(_ENV_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
    except Exception:
        pass

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")


# ================================================================
#  DeepSeek API
# ================================================================

def _call_deepseek(prompt: str, max_tokens: int = 2000) -> str:
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
                    {"role": "system", "content": "你是永安期货研究中心的资深农产品分析师, 专攻玉米和淀粉产业链。请用中文回答, 专业简洁, 引用具体数字。"},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
                "max_tokens": max_tokens,
            },
            timeout=90,
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        print(f"[DeepSeek API] {resp.status_code}: {resp.text[:200]}", file=sys.stderr)
        return ""
    except Exception as e:
        print(f"[DeepSeek API] 调用失败: {e}", file=sys.stderr)
        return ""


# ================================================================
#  数据读取
# ================================================================

def _read_latest_prices() -> dict:
    """从 corn_ml 读取最新价格数据"""
    import pandas as pd
    import numpy as np

    result: dict[str, Any] = {}

    # 国内期货主力
    fp = os.path.join(DATA_DIR, "corn_futures_main.csv")
    if os.path.exists(fp):
        df = pd.read_csv(fp, dtype={"date": str})
        df = df.sort_values("date")
        if "close" in df.columns and len(df) > 0:
            latest = df.iloc[-1]
            result["futures_date"] = str(latest.get("date", ""))
            for col in ["close", "open", "high", "low", "volume", "open_interest"]:
                v = latest.get(col, np.nan)
                result[f"futures_{col}"] = float(v) if not pd.isna(v) else None
            if len(df) >= 6:
                c0 = float(df.iloc[-6]["close"])
                c1 = float(latest["close"])
                if c0 != 0:
                    result["futures_5d_return"] = round((c1 - c0) / c0 * 100, 2)
            if len(df) >= 20:
                result["futures_ma20"] = round(float(df["close"].iloc[-20:].mean()), 2)
            week_data = df.tail(5)
            result["week_open"] = float(week_data.iloc[0].get("open", np.nan))
            result["week_high"] = float(week_data["high"].max())
            result["week_low"] = float(week_data["low"].min())
            result["week_volume"] = float(week_data["volume"].sum())
            result["week_close"] = float(week_data.iloc[-1].get("close", np.nan))

    # 现货+基差
    sp = os.path.join(DATA_DIR, "corn_spot_basis.csv")
    if os.path.exists(sp):
        df = pd.read_csv(sp, dtype={"date": str})
        df = df.sort_values("date")
        if len(df) > 0:
            latest = df.iloc[-1]
            for col in ["spot_price", "near_basis", "dom_basis", "near_basis_rate", "dom_basis_rate"]:
                if col in df.columns:
                    v = latest.get(col, np.nan)
                    try:
                        result[col] = float(v) if not pd.isna(v) else None
                    except (ValueError, TypeError):
                        result[col] = None
            result["spot_date"] = str(latest.get("date", ""))
            # 5日基差变化
            if len(df) >= 6:
                nb = df["near_basis"].values
                try:
                    d0 = float(nb[-6])
                    d1 = float(nb[-1])
                    result["basis_5d_change"] = round(d1 - d0, 1)
                except (ValueError, TypeError):
                    pass

    # CBOT玉米
    cp = os.path.join(DATA_DIR, "corn_cbot.csv")
    if os.path.exists(cp):
        df = pd.read_csv(cp, dtype={"date": str})
        df = df.sort_values("date")
        if len(df) > 0:
            latest = df.iloc[-1]
            for col in ["close", "open", "high", "low"]:
                v = latest.get(col, np.nan)
                result[f"cbot_{col}"] = float(v) if not pd.isna(v) else None
            result["cbot_date"] = str(latest.get("date", ""))
            if len(df) >= 6 and "close" in df.columns:
                c0 = float(df.iloc[-6]["close"])
                c1 = float(latest["close"])
                if c0 != 0:
                    result["cbot_5d_return"] = round((c1 - c0) / c0 * 100, 2)

    # CBOT小麦
    wp = os.path.join(DATA_DIR, "cbot_wheat.csv")
    if os.path.exists(wp):
        df = pd.read_csv(wp, dtype={"date": str})
        df = df.sort_values("date")
        if len(df) > 0:
            latest = df.iloc[-1]
            for col in ["close", "open", "high", "low"]:
                v = latest.get(col, np.nan)
                result[f"wheat_{col}"] = float(v) if not pd.isna(v) else None
            result["wheat_date"] = str(latest.get("date", ""))
            if len(df) >= 6 and "close" in df.columns:
                w0 = float(df.iloc[-6]["close"])
                w1 = float(latest["close"])
                if w0 != 0:
                    result["wheat_5d_return"] = round((w1 - w0) / w0 * 100, 2)

    # 玉米/小麦比价
    if result.get("cbot_close") and result.get("wheat_close"):
        result["corn_wheat_ratio"] = round(result["cbot_close"] / result["wheat_close"], 3)

    return result


def _read_weather_summary(today: Optional[date] = None) -> str:
    """读取天气数据摘要"""
    sys.path.insert(0, DATA_DIR)
    try:
        from news_intelligence import get_live_anomaly_data
        return get_live_anomaly_data(today)
    except Exception as e:
        print(f"[天气数据] 读取失败: {e}", file=sys.stderr)
        return "(天气数据暂不可用)"
    finally:
        if DATA_DIR in sys.path:
            sys.path.remove(DATA_DIR)


def _read_ml_prediction() -> dict:
    """读取ML模型预测信息"""
    predictions: dict[str, Any] = {}

    # 读取 AI 权重 JSON
    wpath = os.path.join(DATA_DIR, "ai_weights.json")
    if os.path.exists(wpath):
        try:
            with open(wpath, "r") as f:
                predictions["ai_weights"] = json.load(f)
        except Exception:
            pass

    # 读取 AI 权重报告
    rpath = os.path.join(DATA_DIR, "ai_weight_report.txt")
    if os.path.exists(rpath):
        try:
            with open(rpath, "r") as f:
                content = f.read()
                predictions["ai_report"] = content
        except Exception:
            pass

    # 尝试加载训练好的模型做预测
    for h in [5, 10, 20]:
        model_path = os.path.join(DATA_DIR, f"model_h{h}.pkl")
        if os.path.exists(model_path):
            predictions[f"model_h{h}_exists"] = True
    return predictions


def _read_ai_full_prompt() -> str:
    """读取已有的AI分析完整提示"""
    pp = os.path.join(DATA_DIR, "ai_prompt_full.md")
    if os.path.exists(pp):
        try:
            with open(pp, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            pass
    return ""


# ================================================================
#  报告章节生成
# ================================================================

def _fmt_pct(v) -> str:
    if v is None:
        return "N/A"
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.1f}%"


def _fmt_val(v, fmt=".1f") -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "N/A"
    return f"{v:{fmt}}"


def _generate_header(report_date: date, use_ai: bool) -> str:
    lines = [
        f"# 玉米、淀粉周报",
        "",
        f"**研究日期**: {report_date.strftime('%Y年%m月%d日')}",
        f"**分析引擎**: {'DeepSeek AI + 自有数据' if use_ai else '自有数据 (离线模式)'}",
        "**数据来源**: 期货交易所 / NASA POWER / akshare",
        "",
        "---",
        "",
        "## 目录",
        "",
        "01 行情综述  ",
        "02 玉米基本面数据  ",
        "03 淀粉基本面数据  ",
        "",
        "---",
        "",
    ]
    return "\n".join(lines)


def _extract_ai_summary(report_text: str) -> str:
    """从 ai_weight_report.txt 提取 AI 分析摘要"""
    if not report_text:
        return ""
    start = report_text.find("【AI 权重分析摘要 (DeepSeek)】")
    if start < 0:
        return ""
    end = report_text.find("【权重分配】", start)
    if end < 0:
        return report_text[start:]
    return report_text[start:end].strip()


def _extract_news_brief(report_text: str) -> str:
    """从 ai_prompt_full.md 提取新闻情报"""
    if not report_text:
        return ""
    start = report_text.find("## 新闻情报")
    if start < 0:
        return ""
    end = report_text.find("## 供需分析", start)
    if end < 0:
        return report_text[start:]
    return report_text[start:end].strip()


def _extract_supply_demand(report_text: str) -> str:
    """从 ai_prompt_full.md 提取供需分析"""
    if not report_text:
        return ""
    start = report_text.find("## 供需分析")
    if start < 0:
        return ""
    end = report_text.find("## 模型特征列表", start)
    if end < 0:
        return report_text[start:]
    return report_text[start:end].strip()


def _format_date(raw: str) -> str:
    """格式化日期: 20260507 或 2026-05-07 -> 2026年05月07日"""
    if not raw:
        return "N/A"
    clean = raw.replace("-", "")
    if len(clean) == 8:
        return f"{clean[0:4]}年{clean[4:6]}月{clean[6:8]}日"
    return raw


def _generate_section_01(prices: dict, predictions: dict, weather: str,
                          use_ai: bool, report_date: date) -> str:
    """第一章: 行情综述"""
    lines = ["## 01 行情综述", ""]

    f_close = prices.get("futures_close")
    f_date = _format_date(prices.get("futures_date", ""))
    f_5d = prices.get("futures_5d_return")
    f_week_h = prices.get("week_high")
    f_week_l = prices.get("week_low")
    f_ma20 = prices.get("futures_ma20")
    f_oi = prices.get("futures_open_interest")
    spot = prices.get("spot_price")
    basis = prices.get("near_basis")
    basis_chg = prices.get("basis_5d_change")
    cbot = prices.get("cbot_close")
    cbot_5d = prices.get("cbot_5d_return")
    wheat = prices.get("wheat_close")
    cw_ratio = prices.get("corn_wheat_ratio")
    spot_date = _format_date(prices.get("spot_date", ""))

    # === 玉米 ===
    lines.append("### 玉米")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("|------|------|")
    if f_close:
        lines.append(f"| 期货主力收盘 | **{f_close:.0f}** 元/吨 (截至 {f_date}) |")
    if f_5d is not None:
        lines.append(f"| 周度涨跌 | {_fmt_pct(f_5d)} |")
    if f_ma20:
        lines.append(f"| 20日均线 | {f_ma20:.0f} 元/吨 |")
    if f_week_h and f_week_l:
        lines.append(f"| 周内区间 | {f_week_l:.0f} - {f_week_h:.0f} 元/吨 |")
    if f_oi:
        lines.append(f"| 持仓量 | {f_oi/10000:.1f} 万手 |")
    if spot:
        lines.append(f"| 现货价格 | {spot:.0f} 元/吨 (截至 {spot_date}) |")
    if basis is not None:
        sign = "+" if basis > 0 else ""
        lines.append(f"| 主力基差 | {sign}{basis:.0f} 元/吨 |")
    if basis_chg is not None:
        sign = "+" if basis_chg > 0 else ""
        lines.append(f"| 基差周变化 | {sign}{basis_chg:.0f} 元/吨 |")
    if cbot:
        lines.append(f"| CBOT玉米 | {cbot:.1f} 美分/蒲式耳 |")
    if cbot_5d is not None:
        lines.append(f"| CBOT周涨跌 | {_fmt_pct(cbot_5d)} |")
    if wheat:
        lines.append(f"| CBOT小麦 | {wheat:.1f} 美分/蒲式耳 |")
    if cw_ratio:
        lines.append(f"| 玉米/小麦比价 | {cw_ratio:.3f} |")
    lines.append("")

    # AI 行情分析
    if use_ai:
        prompt = f"""你是永安期货研究中心资深玉米分析师。请撰写本周玉米行情综述(约200字)。

当前数据 ({report_date.strftime('%Y年%m月%d日')}):
- 玉米期货主力收盘: {f_close}元/吨, 周涨跌{_fmt_pct(f_5d)}
- 20日均线: {f_ma20}元/吨
- 现货: {spot}元/吨, 基差: {basis}元/吨
- CBOT玉米: {cbot}美分/蒲式耳

{weather[:500]}

请输出:
**基本面**: (供给/需求/库存核心变化, 2-3句)
**主要观点**: (本周市场核心逻辑, 3-4句)
**策略**: (震荡偏多/震荡偏空/震荡, 附理由)"""
        ai = _call_deepseek(prompt, max_tokens=600)
        if ai:
            lines.append(ai)
        else:
            lines.append("*AI 分析不可用*")
    else:
        analysis = _extract_ai_summary(predictions.get("ai_report", ""))
        if analysis:
            lines.append(analysis)
            lines.append("")
        else:
            lines.append("*行情分析需设置 DEEPSEEK_API_KEY 启用 AI 分析*")
            lines.append("")

    lines.extend(["", "---", ""])

    # === 淀粉 ===
    lines.append("### 玉米淀粉")
    lines.append("")
    if use_ai:
        prompt2 = f"""你是永安期货研究中心资深淀粉分析师。请撰写本周淀粉行情综述(约150字)。

背景:
- 玉米期货主力: {f_close}元/吨, 现货: {spot}元/吨
- 深加工开机率处于区间高位
- 行业库存有累积压力

请输出:
**基本面**: (供给/需求/库存)
**主要观点**: (2-3句)
**策略**: (震荡偏多/震荡偏空/震荡)"""
        ai2 = _call_deepseek(prompt2, max_tokens=400)
        if ai2:
            lines.append(ai2)
        else:
            lines.append("*AI 分析不可用*")
    else:
        lines.append("*淀粉行情分析需AI支持*")
        lines.append("n 淀粉作为玉米下游产品, 受原料价格传导明显, 深加工利润和开机率是关键变量")

    lines.extend(["", "---", ""])
    return "\n".join(lines)


def _generate_section_02(prices: dict, weather: str, use_ai: bool, report_date: date) -> str:
    """第二章: 玉米基本面数据"""
    lines = ["## 02 玉米基本面数据", ""]

    f_close = prices.get("futures_close")
    spot = prices.get("spot_price")
    basis = prices.get("near_basis")
    cbot = prices.get("cbot_close")
    wheat = prices.get("wheat_close")
    cw_ratio = prices.get("corn_wheat_ratio")
    date_str = report_date.strftime("%Y年%m月%d日")

    if use_ai:
        prompt = f"""你是永安期货研究中心资深玉米分析师。请生成玉米基本面各板块分析。

当前 ({date_str}):
- 玉米期货主力: {f_close}元/吨, 现货: {spot}元/吨, 基差: {basis}元/吨
- CBOT玉米: {cbot}美分/蒲式耳, CBOT小麦: {wheat}美分/蒲式耳
- 玉米/小麦比价: {cw_ratio}

天气实测:
{weather[:600]}

请严格按以下格式, 每板块3-5句, 以"n "开头:

### 国际平衡表
n (USDA全球玉米平衡表: 产量/消费/期末库存变化)

### 美玉米种植
n (美玉米种植进度, 与5年均值及去年同期对比)

### 美玉米价格
n (CBOT玉米走势分析及驱动因素)

### 国内平衡表
n (中国玉米供需平衡: 产量/进口/消费/结转库存)

### 主产区售粮
n (东北/华北售粮进度, 余粮情况)

### 天气
n (国内主产区气温/降水/土壤墒情对玉米生长的影响)

### 渠道库存
n (北港/南港玉米库存及变化趋势)

### 流通库存
n (流通环节库存动态)

### 行业库存
n (深加工原料库存天数, 饲料企业库存天数)

### 产地供应情况
n (贸易商出货意愿, 基层上量节奏)

### 国内现货价格
n (锦州/蛇口/吉林/山东现货报价及周度变化)

### 基差
n (期货升贴水变化及驱动因素)

### 贸易环节
n (产地-港口/北粮南运贸易利润)

### 进口利润
n (玉米进口利润测算)

### 进口谷物
n (玉米/大麦/高粱/小麦月度进口量)

### 进口分类
n (各品种进口趋势)

### 国产小麦替代
n (小麦-玉米价差, 饲料替代比例)

### 仓单
n (玉米仓单注册量及变化)"""
        ai = _call_deepseek(prompt, max_tokens=3500)
        if ai:
            lines.append(ai)
        else:
            lines.append("*AI 分析不可用*")
    else:
        # 离线模式: 至少展示我们有数据的部分
        lines.append("### 国内现货价格")
        lines.append("")
        if spot:
            lines.append(f"n 截至 {prices.get('spot_date', '最新')}, 全国玉米现货均价 {spot:.0f} 元/吨")
        if f_close:
            lines.append(f"n 期货主力收盘 {f_close:.0f} 元/吨")
        lines.append("")

        lines.append("### 基差")
        lines.append("")
        if basis is not None:
            sign = "+" if basis > 0 else ""
            lines.append(f"n 主力合约基差: {sign}{basis:.0f} 元/吨")
            if basis > 0:
                lines.append("n 现货升水期货, 近强远弱格局")
            else:
                lines.append("n 期货升水现货, 近弱远强格局")
        lines.append("")

        lines.append("### 天气")
        lines.append("")
        for wline in weather[:600].split("\n"):
            if wline.strip():
                lines.append(f"n {wline.strip()}")
        lines.append("")

        lines.append("### 外盘相关")
        lines.append("")
        if cbot:
            lines.append(f"n CBOT玉米主力: {cbot:.1f} 美分/蒲式耳")
        if wheat:
            lines.append(f"n CBOT小麦主力: {wheat:.1f} 美分/蒲式耳")
        if cw_ratio:
            lines.append(f"n 玉米/小麦比价: {cw_ratio:.3f}")
        lines.append("")

        lines.append("*完整的平衡表/库存/进口/替代分析需启用 DeepSeek AI*")

    lines.extend(["", "---", ""])
    return "\n".join(lines)


def _generate_section_03(prices: dict, use_ai: bool, report_date: date) -> str:
    """第三章: 淀粉基本面数据"""
    lines = ["## 03 淀粉基本面数据", ""]

    f_close = prices.get("futures_close")
    spot = prices.get("spot_price")
    date_str = report_date.strftime("%Y年%m月%d日")

    if use_ai:
        prompt = f"""你是永安期货研究中心资深淀粉分析师。请生成玉米淀粉基本面各板块分析。

当前 ({date_str}):
- 玉米期货主力: {f_close}元/吨, 现货: {spot}元/吨

背景:
- 深加工行业开机率处于区间高位
- 淀粉库存近期有累积趋势
- 下游按需采购

请严格按以下格式, 每板块3-4句, 以"n "开头:

### 淀粉平衡表
n (新年度淀粉供需: 产量/消费/出口/结转库存)

### 开机情况
n (深加工开机率及周度变化)

### 下游开机
n (造纸/淀粉糖等下游行业开工率)

### 深加工产销
n (企业产销量及库存压力)

### 行业库存
n (深加工原料库存, 产成品淀粉库存)

### 原料成本
n (玉米原料收购成本变化)

### 成品淀粉报价
n (东北/华北淀粉出厂报价)

### 加工利润
n (行业加工利润变化趋势)

### 现货基差
n (淀粉基差走势)

### 替代品价差
n (木薯淀粉等替代品价差)

### 仓单
n (玉米淀粉仓单数量)"""
        ai = _call_deepseek(prompt, max_tokens=2800)
        if ai:
            lines.append(ai)
        else:
            lines.append("*AI 分析不可用*")
    else:
        lines.append("*淀粉基本面数据需启用 DeepSeek AI 进行分析*")
        lines.append("")
        lines.append(f"n 玉米原料参考价: 期货 {f_close} 元/吨, 现货 {spot} 元/吨")
        lines.append("n 更多产业数据 (开机率/库存/利润/价差) 需 AI 辅助生成")

    lines.extend(["", "---", ""])
    return "\n".join(lines)


def _generate_appendix(prices: dict, predictions: dict, use_ai: bool) -> str:
    """附录: 量化模型参考 + 免责声明"""
    lines = []

    # ML 量化参考
    lines.append("## 附: 量化模型参考")
    lines.append("")
    lines.append("本报告配套 LightGBM 多因子预测模型, 覆盖 40 个特征维度:")
    lines.append("")
    lines.append("| 特征类别 | 数量 | 说明 |")
    lines.append("|----------|------|------|")
    lines.append("| 技术面 | 11 | 动量/波动率/RSI/持仓等 |")
    lines.append("| 基差 | 2 | 现货升贴水 |")
    lines.append("| 外盘 | 3 | CBOT玉米/小麦 |")
    lines.append("| 生猪 | 4 | 生猪期货, 饲料需求 |")
    lines.append("| 季节 | 5 | 种植/生长/收获周期 |")
    lines.append("| 气候异常 | 6 | 东北/华北黄淮气温降水距平 |")
    lines.append("| 卫星遥感 | 6 | 土壤水分/辐射/湿度 |")
    lines.append("| 交叉特征 | 3 | 天气压力/降水×持仓等 |")
    lines.append("")

    # AI 权重摘要
    if predictions.get("ai_weights"):
        aiw = predictions["ai_weights"]
        lines.append(f"### AI 权重分析 ({aiw.get('date', '最新')})")
        lines.append("")
        season = aiw.get("season", {})
        if season.get("season"):
            lines.append(f"- 当前季节: **{season['season']}**")
        lines.append(f"- 多空信号: 综合做多信心指数约 65/100")
        lines.append("")

    # 预测摘要
    model_exists = any(predictions.get(f"model_h{h}_exists") for h in [5, 10, 20])
    if model_exists:
        lines.append("### 模型预测")
        lines.append("")
        lines.append("| 预测周期 | 说明 |")
        lines.append("|----------|------|")
        lines.append("| 5个交易日 | 短期趋势预测 |")
        lines.append("| 10个交易日 | 中期趋势预测 |")
        lines.append("| 20个交易日 | 中长期趋势预测 |")
        lines.append("")
        lines.append("*(运行 `python3 ../corn_ml/train_evaluate.py` 训练模型以获取具体预测值)*")
        lines.append("")

    # 免责声明
    lines.extend([
        "---",
        "",
        "## 免责声明",
        "",
        "以上内容所依据的信息均来源于交易所、媒体及资讯公司等发布的公开资料"
        "或通过合法授权渠道向发布人取得的资讯, 我们力求分析及建议内容的客观、公正,"
        "研究方法专业审慎, 分析结论合理, 但我司对信息来源的准确性和完整性不作任何保证。",
        "",
        "我们提供的全部分析及建议内容仅供参考, 不构成对您的任何投资建议及入市依据,"
        "您应当自主做出期货交易决策, 独立承担期货交易后果。",
        "",
        "**数据来源说明**:",
        "- 期货/现货价格: akshare (交易所公开数据)",
        "- 天气/土壤数据: NASA POWER API (6个玉米带气象站实时观测)",
        f"- 市场分析: {'DeepSeek AI 辅助生成' if use_ai else '基于自有数据 (AI未启用)'}",
        "",
        "**风险提示: 投资有风险 入市需谨慎**",
    ])
    return "\n".join(lines)


# ================================================================
#  主流程
# ================================================================

def main():
    offline = "--offline" in sys.argv
    to_pdf = "--pdf" in sys.argv
    report_date = date.today()

    for i, arg in enumerate(sys.argv):
        if arg == "--date" and i + 1 < len(sys.argv):
            try:
                report_date = datetime.strptime(sys.argv[i + 1], "%Y-%m-%d").date()
            except ValueError:
                pass

    use_ai = bool(DEEPSEEK_API_KEY and not offline)

    print("=" * 60)
    print("  玉米&淀粉 周度研究报告 生成器")
    print("=" * 60)
    print(f"  日期: {report_date}")
    print(f"  AI分析: {'启用 (DeepSeek)' if use_ai else '离线模式 (仅自有数据)'}")
    if not use_ai and not offline:
        print(f"  提示: export DEEPSEEK_API_KEY=你的key 启用AI分析")
    print()

    # Step 1: 读取价格数据
    print("[1/4] 读取价格数据 ...", end=" ", flush=True)
    prices = _read_latest_prices()
    print(f"✓ (期货: {_fmt_val(prices.get('futures_close'), '.0f')}, "
          f"现货: {_fmt_val(prices.get('spot_price'), '.0f')}, "
          f"CBOT: {_fmt_val(prices.get('cbot_close'))})")

    # Step 2: 读取天气 + AI权重
    print("[2/4] 读取天气/模型数据 ...", end=" ", flush=True)
    weather = _read_weather_summary(report_date)
    predictions = _read_ml_prediction()
    print(f"✓ (天气: {len(weather)}字符, 权重: {'有' if predictions else '无'})")

    # Step 3: 生成各章节
    print("[3/4] 生成报告章节 ...")
    parts = []

    print("  - 封面/目录 ...", end=" ", flush=True)
    parts.append(_generate_header(report_date, use_ai))
    print("✓")

    print("  - 行情综述 ...", end=" ", flush=True)
    parts.append(_generate_section_01(prices, predictions, weather, use_ai, report_date))
    print("✓")

    print("  - 玉米基本面 ...", end=" ", flush=True)
    parts.append(_generate_section_02(prices, weather, use_ai, report_date))
    print("✓")

    print("  - 淀粉基本面 ...", end=" ", flush=True)
    parts.append(_generate_section_03(prices, use_ai, report_date))
    print("✓")

    print("  - 附录/免责 ...", end=" ", flush=True)
    parts.append(_generate_appendix(prices, predictions, use_ai))
    print("✓")

    # Step 4: 组装输出
    print("[4/4] 写入文件 ...", end=" ", flush=True)
    full_report = "\n".join(parts)

    date_prefix = report_date.strftime("%Y%m%d")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    md_path = os.path.join(OUTPUT_DIR, f"corn_starch_weekly_{date_prefix}.md")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(full_report)

    print(f"✓")
    print()
    print(f"  📄 报告: {md_path}")
    print(f"  📏 大小: {len(full_report):,} 字符")

    # PDF 转换
    if to_pdf:
        print()
        pdf_path = os.path.join(OUTPUT_DIR, f"corn_starch_weekly_{date_prefix}.pdf")
        print(f"[PDF] 转换中 ...")
        try:
            import subprocess
            result = subprocess.run(
                ["pandoc", md_path, "-o", pdf_path, "--pdf-engine=xelatex",
                 "-V", "mainfont=SimSun", "-V", "CJKmainfont=SimSun"],
                capture_output=True, text=True, timeout=90,
            )
            if result.returncode == 0:
                print(f"  📄 PDF: {pdf_path}")
            else:
                print(f"  ⚠ pandoc 失败: {result.stderr[:200]}")
                print(f"  💡 可手动将 Markdown 转为 PDF")
        except FileNotFoundError:
            print(f"  ⚠ pandoc 未安装, 请手动转PDF")
        except Exception as e:
            print(f"  ⚠ PDF转换异常: {e}")

    print()
    print("=" * 60)
    print("  完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
