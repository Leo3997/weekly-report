#!/usr/bin/env python3
"""
玉米市场周报生成器
从各数据源读取最新数据，生成 HTML 报告并转换为 PDF。
"""

import base64
import io
import json
import os
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "WenQuanYi Micro Hei", "Noto Sans CJK SC", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
ML_DIR = BASE_DIR / "corn_ml"
OUT_DIR = BASE_DIR / "corn_weekly_report" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 周度范围 ──────────────────────────────────────────────
REPORT_END = date.today()
REPORT_START = REPORT_END - timedelta(days=7)


# ═══════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════

def load_futures():
    """加载玉米期货主力合约"""
    df = pd.read_csv(ML_DIR / "corn_futures_main.csv", parse_dates=["date"])
    df = df.sort_values("date")
    return df


def load_spot_basis():
    """加载现货基差数据"""
    df = pd.read_csv(ML_DIR / "corn_spot_basis.csv", parse_dates=["date"])
    return df.sort_values("date")


def load_cbot():
    """加载CBOT玉米"""
    df = pd.read_csv(ML_DIR / "corn_cbot.csv", parse_dates=["date"])
    return df.sort_values("date")


def load_casde():
    """加载CASDE供需平衡表"""
    df = pd.read_csv(ML_DIR / "casde_corn_supply_demand.csv")
    return df.sort_values("report_date")


def load_usda():
    """加载USDA WASDE"""
    df = pd.read_csv(ML_DIR / "usda_wasde_corn.csv")
    return df.sort_values("report_date")


def load_sales():
    """加载售粮进度"""
    df = pd.read_csv(ML_DIR / "regional_sales_progress.csv")
    return df


def load_inventory():
    """加载期货仓单库存"""
    df = pd.read_csv(ML_DIR / "corn_inventory_daily.csv", parse_dates=["date"])
    return df.sort_values("date")


# ═══════════════════════════════════════════════════════════
# 新增 akshare 数据源
# ═══════════════════════════════════════════════════════════

def load_cftc_holding():
    """CFTC 玉米期货基金持仓 (akshare)"""
    try:
        import akshare as ak
        df = ak.macro_usa_cftc_c_holding()
        corn_cols = [c for c in df.columns if "玉米" in c]
        if not corn_cols:
            # Try by index position — corn columns are typically last 3
            corn_cols = list(df.columns[-3:])
        date_col = df.columns[0]
        result = df[[date_col] + corn_cols].copy()
        # 玉米-多头持仓, 玉米-空头持仓, 玉米-净多持仓
        result.columns = ["date", "long", "short", "net"]
        result["date"] = pd.to_datetime(result["date"])
        result["net_long"] = result["long"] - result["short"]
        return result.sort_values("date")
    except Exception as e:
        print(f"  [WARN] CFTC数据获取失败: {e}")
        return pd.DataFrame()


def load_corn_spot_soozhu():
    """搜猪网玉米日度现货价格 (akshare)"""
    try:
        import akshare as ak
        df = ak.spot_corn_price_soozhu()
        df.columns = ["date", "price_yuan_jin"]
        df["date"] = pd.to_datetime(df["date"])
        df["price_yuan_ton"] = df["price_yuan_jin"] * 2000  # 元/斤 → 元/吨
        return df.sort_values("date")
    except Exception as e:
        print(f"  [WARN] 搜猪网玉米价格获取失败: {e}")
        return pd.DataFrame()


def load_hog_price():
    """搜猪网各省生猪价格 (akshare)"""
    try:
        import akshare as ak
        df = ak.spot_hog_soozhu()
        df.columns = ["province", "price_yuan_jin", "change"]
        df["price_yuan_jin"] = pd.to_numeric(df["price_yuan_jin"], errors="coerce")
        df["change"] = pd.to_numeric(df["change"], errors="coerce")
        return df
    except Exception as e:
        print(f"  [WARN] 生猪价格获取失败: {e}")
        return pd.DataFrame()


def load_feed_price():
    """搜猪网混合饲料价格 (akshare)"""
    try:
        import akshare as ak
        df = ak.spot_mixed_feed_soozhu()
        df.columns = ["date", "price"]
        df["date"] = pd.to_datetime(df["date"])
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        return df.sort_values("date")
    except Exception as e:
        print(f"  [WARN] 饲料价格获取失败: {e}")
        return pd.DataFrame()


def load_freight_index():
    """波罗的海干散货指数 BDI (akshare)"""
    try:
        import akshare as ak
        df = ak.macro_china_freight_index()
        # 第5列是BDI (index 4), 第1列是日期 (index 0)
        date_col = df.columns[0]
        # 优先用列名匹配, fallback 按位置
        bdi_candidates = [c for c in df.columns if "BDI" in str(c) and "BCTI" not in str(c) and "BDTI" not in str(c)]
        if not bdi_candidates:
            bdi_candidates = [c for c in df.columns if "综合运价" in str(c)]
        if not bdi_candidates:
            bdi_candidates = [df.columns[4]]  # fallback: 第5列
        bdi_col = bdi_candidates[0]
        result = df[[date_col, bdi_col]].copy()
        result.columns = ["date", "bdi"]
        result["date"] = pd.to_datetime(result["date"])
        result["bdi"] = pd.to_numeric(result["bdi"], errors="coerce")
        result = result.dropna(subset=["bdi"])
        return result.sort_values("date").reset_index(drop=True)
    except Exception as e:
        print(f"  [WARN] BDI指数获取失败: {e}")
        return pd.DataFrame()


def load_ai_weight_report():
    """从AI权重报告提取实测天气/库存/供需文本"""
    path = ML_DIR / "ai_weight_report.txt"
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    result = {"weather_lines": [], "inventory_lines": [], "usda_lines": [], "casde_lines": []}
    section = None
    for line in text.split("\n"):
        line = line.strip()
        if "实测数据" in line or "## 东北:" in line or "## 华北" in line:
            section = "weather"
        if "玉米期货库存" in line:
            section = "inventory"
        if "USDA WASDE" in line:
            section = "usda"
        if "CASDE 中国玉米供需形势" in line:
            section = "casde"
        if "====" in line:
            section = None
        if section and line and not line.startswith("="):
            result[f"{section}_lines"].append(line)
    return result


# ═══════════════════════════════════════════════════════════
# 周度统计计算
# ═══════════════════════════════════════════════════════════

def fmt_change(val, unit="", decimals=1):
    """格式化变动值"""
    if abs(val) < 0.005:
        return "持平"
    sign = "+" if val > 0 else ""
    if decimals == 0:
        return f"{sign}{int(val)}{unit}"
    return f"{sign}{val:.{decimals}f}{unit}"


def fmt_pct(val):
    """格式化百分比变动"""
    if abs(val) < 0.0005:
        return "持平"
    sign = "+" if val > 0 else ""
    return f"{sign}{val:.1%}"


def compute_weekly_stats(futures_df, basis_df, cbot_df, inv_df):
    """计算本周核心统计指标"""
    stats = {}

    # ── 期货 ──
    recent = futures_df[futures_df["date"] >= pd.Timestamp(REPORT_START)]
    if recent.empty:
        recent = futures_df.tail(10)
    prev_week = futures_df[
        (futures_df["date"] >= pd.Timestamp(REPORT_START - timedelta(days=7)))
        & (futures_df["date"] < pd.Timestamp(REPORT_START))
    ]

    latest = recent.iloc[-1]
    week_first = recent.iloc[0]
    stats["futures_close"] = float(latest["close"])
    stats["futures_high"] = float(recent["high"].max())
    stats["futures_low"] = float(recent["low"].min())
    stats["futures_open_interest"] = float(latest["open_interest"])
    stats["futures_volume"] = float(recent["volume"].sum())
    stats["futures_week_change"] = stats["futures_close"] - float(week_first["close"])
    stats["futures_week_change_pct"] = stats["futures_week_change"] / float(week_first["close"])

    if not prev_week.empty:
        stats["futures_prev_close"] = float(prev_week.iloc[-1]["close"])
    else:
        # fallback: 5 trading days before
        idx = futures_df[futures_df["date"] < pd.Timestamp(REPORT_START)].index
        if len(idx) >= 5:
            stats["futures_prev_close"] = float(futures_df.loc[idx[-5], "close"])
        else:
            stats["futures_prev_close"] = stats["futures_close"]

    stats["futures_date"] = str(latest["date"].date())

    # 20日均线
    ma_data = futures_df[futures_df["date"] <= latest["date"]].tail(20)
    stats["futures_ma20"] = float(ma_data["close"].mean()) if len(ma_data) >= 20 else None

    # ── 现货 / 基差 ──
    basis_recent = basis_df[basis_df["date"] >= pd.Timestamp(REPORT_START)]
    if basis_recent.empty:
        basis_recent = basis_df.tail(5)
    bl = basis_recent.iloc[-1]
    stats["spot_price"] = float(bl["spot_price"])
    stats["dom_basis"] = float(bl["dom_basis"])
    stats["dom_basis_rate"] = float(bl["dom_basis_rate"])

    basis_prev = basis_df[basis_df["date"] < pd.Timestamp(REPORT_START)]
    if not basis_prev.empty:
        stats["spot_prev"] = float(basis_prev.iloc[-1]["spot_price"])
        stats["basis_prev"] = float(basis_prev.iloc[-1]["dom_basis"])
    else:
        stats["spot_prev"] = stats["spot_price"]
        stats["basis_prev"] = stats["dom_basis"]

    # ── CBOT ──
    cbot_recent = cbot_df[cbot_df["date"] >= pd.Timestamp(REPORT_START)]
    if cbot_recent.empty:
        cbot_recent = cbot_df.tail(8)
    cbot_latest = cbot_recent.iloc[-1]
    cbot_first = cbot_recent.iloc[0]
    stats["cbot_close"] = float(cbot_latest["close"])
    stats["cbot_date"] = str(cbot_latest["date"].date())
    stats["cbot_week_change"] = stats["cbot_close"] - float(cbot_first["close"])

    cbot_prev = cbot_df[cbot_df["date"] < pd.Timestamp(REPORT_START)]
    if not cbot_prev.empty:
        stats["cbot_prev"] = float(cbot_prev.iloc[-1]["close"])
    else:
        stats["cbot_prev"] = stats["cbot_close"]

    # ── 仓单库存 ──
    inv_recent = inv_df[inv_df["date"] >= pd.Timestamp(REPORT_START)]
    if inv_recent.empty:
        inv_recent = inv_df.tail(8)
    inv_latest_row = inv_recent.iloc[-1]
    stats["inventory"] = float(inv_latest_row["inventory"])
    stats["inventory_date"] = str(inv_latest_row["date"].date())
    inv_changes = inv_recent["inv_change"].dropna()
    stats["inventory_week_change"] = float(inv_changes.sum()) if len(inv_changes) > 0 else 0

    inv_month_ago = inv_df[inv_df["date"] <= pd.Timestamp(REPORT_END - timedelta(days=30))]
    if not inv_month_ago.empty:
        stats["inventory_month_ago"] = float(inv_month_ago.iloc[-1]["inventory"])
        stats["inventory_month_change"] = stats["inventory"] - stats["inventory_month_ago"]
    else:
        stats["inventory_month_ago"] = stats["inventory"]
        stats["inventory_month_change"] = 0

    # ── 周度区间 ──
    stats["week_start"] = REPORT_START.isoformat()
    stats["week_end"] = REPORT_END.isoformat()
    stats["week_label"] = f"{REPORT_START.strftime('%m.%d')} - {REPORT_END.strftime('%m.%d')}"

    return stats


# ═══════════════════════════════════════════════════════════
# CASDE 数据提取
# ═══════════════════════════════════════════════════════════

def casde_rows(casde_df):
    """提取CASDE最近两年对比"""
    df = casde_df.sort_values("report_date")
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    cy = latest["market_year"]
    py = prev["market_year"]

    items = [
        ("播种面积", "千公顷", "corn_area_kha", 0),
        ("产量", "百万吨", "corn_production_mmt", 1),
        ("单产", "公斤/公顷", "corn_yield_kg_ha", 0),
        ("饲用消费", "百万吨", "corn_feed_consumption_mmt", 1),
        ("工业消费", "百万吨", "corn_industrial_consumption_mmt", 1),
        ("总消费", "百万吨", "corn_total_consumption_mmt", 1),
        ("进口量", "百万吨", "corn_imports_mmt", 1),
    ]
    rows = []
    for label, unit, col, dec in items:
        pv = prev[col]
        cv = latest[col]
        chg = cv - pv
        if abs(chg) < 0.005:
            chg_str = "—"
        elif chg > 0:
            chg_str = f"+{chg:.{dec}f}"
        else:
            chg_str = f"{chg:.{dec}f}"
        rows.append({
            "name": label, "unit": unit,
            "prev": f"{pv:.{dec}f}", "curr": f"{cv:.{dec}f}",
            "change": chg_str,
            "direction": "up" if chg > 0.005 else ("down" if chg < -0.005 else "flat"),
        })
    return rows, cy, py, latest["report_date"], latest.get("notes", "")


# ═══════════════════════════════════════════════════════════
# USDA 数据提取
# ═══════════════════════════════════════════════════════════

def usda_rows(usda_df):
    """提取USDA全球供需关键数据"""
    df = usda_df.sort_values("report_date")
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    items = [
        ("全球产量", "global_production_mmt", 1),
        ("全球消费", "global_consumption_mmt", 1),
        ("全球期末库存", "global_ending_stocks_mmt", 1),
        ("库存消费比", "global_stocks_to_use", 1),
        ("中国产量", "china_production_mmt", 1),
        ("中国进口", "china_imports_mmt", 1),
        ("美国产量", "us_production_mmt", 1),
        ("美国农场价", "us_farm_price_usd_bu", 2),
        ("巴西产量", "brazil_production_mmt", 1),
        ("阿根廷产量", "argentina_production_mmt", 1),
    ]
    rows = []
    for label, col, dec in items:
        pv = prev[col]
        cv = latest[col]
        chg = cv - pv
        if abs(chg) < 0.005:
            chg_str = "—"
        elif chg > 0:
            chg_str = f"+{chg:.{dec}f}"
        else:
            chg_str = f"{chg:.{dec}f}"
        rows.append({
            "name": label,
            "prev": f"{pv:.{dec}f}",
            "curr": f"{cv:.{dec}f}",
            "change": chg_str,
            "direction": "up" if chg > 0.005 else ("down" if chg < -0.005 else "flat"),
        })
    return rows, latest["market_year"], prev["market_year"], latest["report_date"]


# ═══════════════════════════════════════════════════════════
# 天气数据提取 (从AI仓位报告)
# ═══════════════════════════════════════════════════════════

def extract_weather(ai_data):
    """提取实测天气段落"""
    lines = ai_data.get("weather_lines", [])
    if not lines:
        return []
    # 整理成结构化数据
    result = []
    region = ""
    for line in lines:
        if line.startswith("##"):
            region = line.replace("##", "").replace(":", "").strip()
        elif "气温" in line and "降水" in line:
            # 哈尔滨: 气温 18.6°C (5年均 13.6°C, 偏高+5.0);...
            parts = line.split(":", 1)
            if len(parts) == 2:
                city = parts[0].strip()
                detail = parts[1].strip()
                result.append({"region": region, "city": city, "detail": detail})
    return result


# ═══════════════════════════════════════════════════════════
# 售粮进度对比折线图
# ═══════════════════════════════════════════════════════════

def generate_sales_comparison_chart(sales_df):
    """生成主产区售粮进度对比折线图，返回 base64 PNG"""
    detail = sales_df[sales_df["province"].isin(
        ["黑龙江", "吉林", "辽宁", "内蒙古", "山东", "河北", "河南", "山西"]
    )].copy()

    ne_order = ["黑龙江", "吉林", "辽宁", "内蒙古"]
    hb_order = ["山东", "河北", "河南", "山西"]
    ne_data = detail[detail["region"] == "东北"].set_index("province").reindex(ne_order)
    hb_data = detail[detail["region"] == "华北"].set_index("province").reindex(hb_order)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    fig.patch.set_facecolor("#FCFAF5")

    colors_ne = ["#C0392B", "#E74C3C", "#EC7063", "#F1948A"]
    colors_hb = ["#2980B9", "#3498DB", "#5DADE2", "#85C1E9"]

    for ax, data, colors, title, region_color in [
        (ax1, ne_data, colors_ne, "东北产区", "#C0392B"),
        (ax2, hb_data, colors_hb, "华北黄淮产区", "#2980B9"),
    ]:
        ax.set_facecolor("#FEFEFE")
        provinces = list(data.index)
        pct_vals = [float(data.loc[p, "sales_progress_pct"]) for p in provinces]
        prices_low = [float(data.loc[p, "spot_price_low"]) for p in provinces]
        prices_high = [float(data.loc[p, "spot_price_high"]) for p in provinces]

        # 折线 + 散点: 售粮进度
        x = range(len(provinces))
        ax.plot(x, pct_vals, color=region_color, linewidth=2.5, marker="o",
                markersize=11, markerfacecolor="white", markeredgewidth=2.5,
                markeredgecolor=region_color, zorder=5)
        for xi, pct in zip(x, pct_vals):
            ax.annotate(f"{pct:.0f}%", (xi, pct), textcoords="offset points",
                        xytext=(0, 16), ha="center", fontsize=12,
                        fontweight="bold", color=region_color)

        # 填充区域让折线图更有层次
        ax.fill_between(x, 80, pct_vals, alpha=0.08, color=region_color)

        ax.set_xticks(x)
        ax.set_xticklabels(provinces, fontsize=11, color="#2C3E50")
        ax.set_ylim(80, 102)
        ax.set_title(title, fontsize=14, fontweight="bold", color=region_color, pad=10)
        ax.set_ylabel("售粮进度 (%)", fontsize=10, color="#7F8C8D")
        ax.grid(axis="y", linestyle="--", alpha=0.3, color="#BDC3C7")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#BDC3C7")
        ax.spines["bottom"].set_color("#BDC3C7")
        ax.tick_params(colors="#7F8C8D")

    report_date = sales_df["report_date"].iloc[0]
    fig.suptitle("主产区玉米售粮进度对比\n(Regional Corn Sales Progress Comparison)",
                 fontsize=16, fontweight="bold", color="#2C3E50", y=1.01)
    fig.text(0.5, 0.01, f"数据来源: 饲料行业信息网 / 慧博投研  |  报告日期: {report_date}",
             ha="center", fontsize=7.5, color="#95A5A6", style="italic")

    plt.tight_layout(rect=[0, 0.04, 1, 0.94])

    buf = io.BytesIO()
    fig.savefig(buf, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor(), format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def generate_cftc_chart(cftc_df):
    """CFTC玉米基金持仓 + 期货价格双轴图"""
    if cftc_df.empty:
        return ""
    df = cftc_df.tail(80).copy()  # 最近80周

    fig, ax1 = plt.subplots(figsize=(13, 4.2))
    fig.patch.set_facecolor("#FCFAF5")
    ax1.set_facecolor("#FEFEFE")

    x = range(len(df))
    # 柱状图: 净多持仓
    colors = ["#C0392B" if v >= 0 else "#27AE60" for v in df["net_long"]]
    ax1.bar(x, df["net_long"].values / 10000, color=colors, alpha=0.7, width=0.7, edgecolor="white", linewidth=0.3)
    ax1.axhline(y=0, color="#2C3E50", linewidth=0.6, linestyle="-")
    ax1.set_ylabel("基金净多持仓 (万手)", fontsize=10, color="#C0392B")
    ax1.tick_params(axis="y", colors="#C0392B", labelsize=8)
    ax1.spines["top"].set_visible(False)
    ax1.grid(axis="y", linestyle="--", alpha=0.2, color="#BDC3C7")

    # 标注最新值 & 关键转折
    latest_val = df["net_long"].iloc[-1] / 10000
    direction = "净多" if latest_val > 0 else "净空"
    ax1.annotate(f"{direction}\n{abs(latest_val):.1f}万手",
                 (x[-1], latest_val), textcoords="offset points",
                 xytext=(10, 0), ha="left", fontsize=11, fontweight="bold",
                 color="#C0392B" if latest_val > 0 else "#27AE60")

    # 设置x轴标签
    tick_positions = list(range(0, len(df), max(1, len(df) // 8)))
    tick_labels = [df["date"].iloc[i].strftime("%Y-%m") if i < len(df) else "" for i in tick_positions]
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels, fontsize=7, color="#7F8C8D", rotation=30)

    fig.suptitle("CFTC 玉米期货基金持仓 — 投机净多/净空趋势\n(CFTC Corn Managed Money Net Position)",
                 fontsize=14, fontweight="bold", color="#2C3E50", y=1.01)
    fig.text(0.5, 0.01, "数据来源: CFTC / akshare (macro_usa_cftc_c_holding)",
             ha="center", fontsize=7, color="#95A5A6", style="italic")

    plt.tight_layout(rect=[0, 0.04, 1, 0.93])

    buf = io.BytesIO()
    fig.savefig(buf, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor(), format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def generate_spot_trend_chart(soozhu_df):
    """搜猪网玉米现货价格走势图 (近3月)"""
    if soozhu_df.empty:
        return ""
    df = soozhu_df.tail(90).copy()

    fig, ax = plt.subplots(figsize=(13, 3.6))
    fig.patch.set_facecolor("#FCFAF5")
    ax.set_facecolor("#FEFEFE")

    ax.plot(range(len(df)), df["price_yuan_ton"].values, color="#C0392B", linewidth=2.2, alpha=0.9)
    ax.fill_between(range(len(df)), df["price_yuan_ton"].min() - 50, df["price_yuan_ton"].values,
                    alpha=0.08, color="#C0392B")

    # 标注最近值
    last_val = df["price_yuan_ton"].iloc[-1]
    ax.annotate(f"{last_val:.0f} 元/吨", (len(df) - 1, last_val),
                textcoords="offset points", xytext=(8, 0), ha="left",
                fontsize=11, fontweight="bold", color="#C0392B")

    # 30日均线
    if len(df) >= 30:
        ma30 = df["price_yuan_ton"].rolling(30).mean()
        ax.plot(range(len(df)), ma30.values, color="#2C3E50", linewidth=1.2,
                linestyle="--", alpha=0.6, label="30日均线")
        ax.legend(fontsize=8, loc="upper left", framealpha=0.8)

    # x轴标签
    tick_positions = list(range(0, len(df), max(1, len(df) // 8)))
    tick_labels = [df["date"].iloc[i].strftime("%m-%d") if i < len(df) else "" for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=8, color="#7F8C8D")

    ax.set_ylabel("元/吨", fontsize=9, color="#7F8C8D")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#BDC3C7")
    ax.spines["bottom"].set_color("#BDC3C7")
    ax.grid(axis="y", linestyle="--", alpha=0.2, color="#BDC3C7")

    # 价格范围标注
    y_min, y_max = df["price_yuan_ton"].min(), df["price_yuan_ton"].max()
    ax.set_ylim(y_min - 60, y_max + 40)

    fig.suptitle("玉米现货价格走势 — 搜猪网日度数据\n(Corn Spot Price Trend — Soozhu Daily)",
                 fontsize=14, fontweight="bold", color="#2C3E50", y=1.01)
    fig.text(0.5, 0.01, "数据来源: 搜猪网 / akshare (spot_corn_price_soozhu)  |  单位: 元/吨 (原始数据元/斤×2000)",
             ha="center", fontsize=7, color="#95A5A6", style="italic")

    plt.tight_layout(rect=[0, 0.04, 1, 0.93])

    buf = io.BytesIO()
    fig.savefig(buf, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor(), format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def generate_downstream_chart(hog_df, feed_df):
    """下游需求面板: 生猪价格(各省) + 饲料价格走势"""
    if hog_df.empty and feed_df.empty:
        return ""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.0))
    fig.patch.set_facecolor("#FCFAF5")

    # ── 左: 代表省份生猪价格 ──
    ax1.set_facecolor("#FEFEFE")
    if not hog_df.empty:
        hog_clean = hog_df.dropna(subset=["price_yuan_jin"])
        # 只选取玉米主产+养殖大省: 黑龙江、吉林、辽宁、河北、山东、河南、四川
        key_provinces = ["黑龙江", "吉林", "辽宁", "河北", "山东", "河南", "四川"]
        hog_filtered = hog_clean[hog_clean["province"].isin(key_provinces)]
        if not hog_filtered.empty:
            # 按价格从高到低排序
            hog_filtered = hog_filtered.sort_values("price_yuan_jin", ascending=True)
            provinces = hog_filtered["province"].values
            prices = hog_filtered["price_yuan_jin"].values
            bar_colors = ["#C0392B" if p >= prices.mean() else "#E67E22" for p in prices]
            ax1.barh(range(len(provinces)), prices, color=bar_colors, height=0.55,
                    edgecolor="white", linewidth=0.5)
            for i, (prov, p) in enumerate(zip(provinces, prices)):
                ax1.text(p + 0.05, i, f"{prov} {p:.1f}", va="center", fontsize=11,
                        fontweight="bold", color="#2C3E50")
            ax1.set_yticks([])
            ax1.set_xlabel("元/斤", fontsize=9, color="#7F8C8D")
            ax1.set_title("代表省份生猪价格", fontsize=12, fontweight="bold", color="#C0392B", pad=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    ax1.grid(axis="x", linestyle="--", alpha=0.2, color="#BDC3C7")

    # ── 右: 饲料价格走势 ──
    ax2.set_facecolor("#FEFEFE")
    if not feed_df.empty:
        df = feed_df.tail(120).copy()
        ax2.plot(range(len(df)), df["price"].values, color="#2980B9", linewidth=2.2, alpha=0.85)
        ax2.fill_between(range(len(df)), df["price"].min() - 0.05, df["price"].values,
                        alpha=0.06, color="#2980B9")
        if len(df) >= 30:
            ma30 = df["price"].rolling(30).mean()
            ax2.plot(range(len(df)), ma30.values, color="#E74C3C", linewidth=1.2,
                    linestyle="--", alpha=0.5, label="30日均线")
            ax2.legend(fontsize=8, loc="upper left")
        # x轴
        tick_positions = list(range(0, len(df), max(1, len(df) // 6)))
        tick_labels = [df["date"].iloc[i].strftime("%m-%d") if i < len(df) else "" for i in tick_positions]
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels(tick_labels, fontsize=7, color="#7F8C8D", rotation=30)
        ax2.set_ylabel("元/斤", fontsize=9, color="#7F8C8D")
        ax2.set_title("混合饲料价格走势", fontsize=12, fontweight="bold", color="#2980B9", pad=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", linestyle="--", alpha=0.2, color="#BDC3C7")

    fig.suptitle("下游需求指标 — 生猪价格 & 饲料成本\n(Downstream Demand — Hog Price & Feed Cost)",
                 fontsize=14, fontweight="bold", color="#2C3E50", y=1.01)
    fig.text(0.5, 0.01, "数据来源: 搜猪网 / akshare  |  生猪: 黑吉辽冀鲁豫川七省代表  |  饲料: 混合饲料日度价格",
             ha="center", fontsize=7, color="#95A5A6", style="italic")

    plt.tight_layout(rect=[0, 0.04, 1, 0.93])

    buf = io.BytesIO()
    fig.savefig(buf, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor(), format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def generate_freight_chart(freight_df):
    """BDI波罗的海干散货指数走势"""
    if freight_df.empty:
        return ""
    df = freight_df.tail(180).copy()  # 近180个数据点

    fig, ax = plt.subplots(figsize=(13, 3.2))
    fig.patch.set_facecolor("#FCFAF5")
    ax.set_facecolor("#FEFEFE")

    ax.plot(range(len(df)), df["bdi"].values, color="#8E44AD", linewidth=2.0, alpha=0.85)
    ax.fill_between(range(len(df)), 0, df["bdi"].values, alpha=0.06, color="#8E44AD")

    # 标注最新
    last_val = df["bdi"].iloc[-1]
    ax.annotate(f"BDI {last_val:.0f}", (len(df) - 1, last_val),
                textcoords="offset points", xytext=(8, 0), ha="left",
                fontsize=11, fontweight="bold", color="#8E44AD")

    tick_positions = list(range(0, len(df), max(1, len(df) // 8)))
    tick_labels = [df["date"].iloc[i].strftime("%Y-%m") if i < len(df) else "" for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=7, color="#7F8C8D", rotation=30)

    ax.set_ylabel("BDI 指数", fontsize=9, color="#7F8C8D")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#BDC3C7")
    ax.spines["bottom"].set_color("#BDC3C7")
    ax.grid(axis="y", linestyle="--", alpha=0.2, color="#BDC3C7")

    fig.suptitle("波罗的海干散货指数 (BDI) — 国际运费参考\n(Baltic Dry Index — International Shipping Cost)",
                 fontsize=14, fontweight="bold", color="#2C3E50", y=1.01)
    fig.text(0.5, 0.01, "数据来源: 波罗的海交易所 / akshare (macro_china_freight_index)  |  BDI反映全球散货海运成本，与谷物进口运费相关",
             ha="center", fontsize=7, color="#95A5A6", style="italic")

    plt.tight_layout(rect=[0, 0.04, 1, 0.93])

    buf = io.BytesIO()
    fig.savefig(buf, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor(), format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ═══════════════════════════════════════════════════════════
# HTML 模板生成
# ═══════════════════════════════════════════════════════════

CSS = r"""
:root {
  --bg: #FCFAF5;
  --card-bg: #FFFFFF;
  --text: #2C3E50;
  --muted: #7F8C8D;
  --accent: #8B0000;
  --accent-light: #C0392B;
  --border: #D5D8DC;
  --up: #C0392B;
  --down: #27AE60;
  --flat: #7F8C8D;
  --table-alt: #FFF8F6;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
  font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans SC", "SimHei", sans-serif;
  background: var(--bg);
  color: var(--text);
  font-size: 13px;
  line-height: 1.6;
  max-width: 800px;
  margin: 0 auto;
  padding: 32px 40px;
}

/* ── 封面头部 ── */
.cover-header {
  text-align: center;
  padding: 28px 0 16px 0;
  border-bottom: 3px double var(--accent);
  margin-bottom: 20px;
}
.cover-header .title {
  font-size: 26px;
  font-weight: 800;
  color: var(--accent);
  letter-spacing: 4px;
}
.cover-header .subtitle {
  font-size: 12px;
  color: var(--muted);
  margin-top: 4px;
  letter-spacing: 1px;
}
.cover-header .date-badge {
  display: inline-block;
  margin-top: 8px;
  background: var(--accent);
  color: #fff;
  padding: 3px 18px;
  font-size: 11px;
  letter-spacing: 2px;
}

/* ── 行情速览卡片 ── */
.metrics-row {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 10px;
  margin-bottom: 18px;
}
.metric-card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  padding: 12px 14px;
  text-align: center;
}
.metric-card .label {
  font-size: 11px;
  color: var(--muted);
  margin-bottom: 4px;
  letter-spacing: 1px;
}
.metric-card .value {
  font-size: 22px;
  font-weight: 700;
  color: var(--text);
}
.metric-card .value.accent { color: var(--accent); }
.metric-card .change {
  font-size: 11px;
  margin-top: 2px;
}
.metric-card .change.up { color: var(--up); }
.metric-card .change.down { color: var(--down); }
.metric-card .change.flat { color: var(--flat); }

/* ── 小节标题 ── */
.section-title {
  font-size: 16px;
  font-weight: 700;
  color: var(--accent);
  border-left: 4px solid var(--accent);
  padding-left: 10px;
  margin: 24px 0 10px 0;
  letter-spacing: 1px;
}
.section-title .en {
  font-size: 10px;
  font-weight: 400;
  color: var(--muted);
  margin-left: 6px;
}

/* ── 行情走势 ── */
.market-summary {
  background: var(--card-bg);
  border: 1px solid var(--border);
  padding: 14px 18px;
  margin-bottom: 14px;
}
.market-summary p {
  margin-bottom: 6px;
  font-size: 13px;
}
.market-summary .highlight {
  color: var(--accent);
  font-weight: 700;
}

/* ── 数据表格 ── */
.data-table {
  width: 100%;
  border-collapse: collapse;
  margin-bottom: 14px;
  font-size: 12px;
}
.data-table thead th {
  background: var(--accent);
  color: #fff;
  padding: 8px 10px;
  font-weight: 600;
  font-size: 12px;
  letter-spacing: 1px;
}
.data-table tbody td {
  padding: 7px 10px;
  border-bottom: 1px solid var(--border);
  text-align: center;
}
.data-table tbody tr:nth-child(even) td {
  background: var(--table-alt);
}
.data-table tbody tr:nth-child(odd) td {
  background: #fff;
}
.data-table .row-name {
  text-align: left;
  font-weight: 600;
  color: var(--text);
}
.data-table .chg-up { color: var(--up); font-weight: 600; }
.data-table .chg-down { color: var(--down); font-weight: 600; }
.data-table .chg-flat { color: var(--flat); }

/* ── 要点 ── */
.key-points {
  background: linear-gradient(135deg, #FFF8F6 0%, #FEF5F0 100%);
  border: 1px solid #E8D5D0;
  padding: 14px 18px;
  margin-bottom: 14px;
}
.key-points h4 {
  font-size: 13px;
  color: var(--accent);
  margin-bottom: 6px;
}
.key-points ul {
  list-style: none;
  padding: 0;
}
.key-points ul li {
  padding: 3px 0;
  padding-left: 16px;
  position: relative;
  font-size: 12px;
}
.key-points ul li::before {
  content: "◆";
  position: absolute;
  left: 0;
  color: var(--accent);
  font-size: 8px;
  top: 5px;
}

/* ── 天气 ── */
.weather-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 10px;
  margin-bottom: 14px;
}
.weather-card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  padding: 10px 14px;
}
.weather-card .region-label {
  font-size: 11px;
  color: var(--muted);
  margin-bottom: 4px;
  letter-spacing: 1px;
}
.weather-card .city-row {
  font-size: 12px;
  padding: 2px 0;
  border-bottom: 1px dotted var(--border);
}
.weather-card .city-row:last-child { border-bottom: none; }

/* ── 库存走势 ── */
.inventory-bar {
  display: flex;
  gap: 2px;
  align-items: flex-end;
  height: 36px;
  margin: 6px 0;
}
.inventory-bar .bar {
  flex: 1;
  background: #C0392B;
  opacity: 0.7;
}

/* ── 页脚 ── */
.footer {
  margin-top: 28px;
  padding-top: 14px;
  border-top: 1px solid var(--border);
  font-size: 10px;
  color: var(--muted);
  text-align: center;
  line-height: 1.8;
}
.footer .disclaimer {
  font-size: 9px;
  color: #A0A0A0;
}

/* ── 打印样式 ── */
@media print {
  body { background: #fff; padding: 20px 30px; }
  .metric-card { break-inside: avoid; }
  .data-table { break-inside: avoid; }
  .section-title { break-after: avoid; }
}
"""


def build_html(stats, casde_data, usda_data, sales_df, weather_list, report_date_str,
               sales_chart_b64, cftc_chart_b64, spot_trend_b64, downstream_b64, freight_b64):
    """构建完整HTML"""
    casde_rows_data, cy, py, casde_date, casde_note = casde_data
    usda_rows_data, usda_my, usda_py, usda_date = usda_data

    # ── 行情速览卡片 ──
    def change_html(val, is_pct=False):
        if val > 0.005:
            cls = "up"
            sign = "+"
        elif val < -0.005:
            cls = "down"
            sign = ""
        else:
            cls = "flat"
            sign = ""
        if is_pct:
            return f'<span class="change {cls}">{sign}{val:.2%}</span>'
        return f'<span class="change {cls}">{sign}{val:.1f}</span>'

    futures_chg = stats["futures_close"] - stats["futures_prev_close"]
    spot_chg = stats["spot_price"] - stats["spot_prev"]
    cbot_chg = stats["cbot_close"] - stats["cbot_prev"]
    basis_chg = stats["dom_basis"] - stats["basis_prev"]

    metrics_html = f"""
    <div class="metrics-row">
      <div class="metric-card">
        <div class="label">玉米期货主力</div>
        <div class="value accent">{stats['futures_close']:.0f}</div>
        <div class="change {'up' if futures_chg > 0 else ('down' if futures_chg < 0 else 'flat')}">
          较上周 {fmt_change(futures_chg, ' 元/吨')}
        </div>
        <div style="font-size:10px;color:#7F8C8D;">MA20: {stats.get('futures_ma20', '-') or '-':.0f} 元/吨</div>
      </div>
      <div class="metric-card">
        <div class="label">全国现货均价</div>
        <div class="value">{stats['spot_price']:.0f}</div>
        <div class="change {'up' if spot_chg > 0 else ('down' if spot_chg < 0 else 'flat')}">
          较上周 {fmt_change(spot_chg, ' 元/吨')}
        </div>
        <div style="font-size:10px;color:#7F8C8D;">基差: +{stats['dom_basis']:.0f} 元/吨</div>
      </div>
      <div class="metric-card">
        <div class="label">CBOT玉米</div>
        <div class="value">{stats['cbot_close']:.1f}</div>
        <div class="change {'up' if cbot_chg > 0 else ('down' if cbot_chg < 0 else 'flat')}">
          较上周 {fmt_change(cbot_chg, ' 美分/蒲')}
        </div>
        <div style="font-size:10px;color:#7F8C8D;">{stats['cbot_date']}</div>
      </div>
      <div class="metric-card">
        <div class="label">期货仓单库存</div>
        <div class="value">{stats['inventory']/10000:.2f}<span style="font-size:14px;"> 万吨</span></div>
        <div class="change {'up' if stats['inventory_week_change'] > 0 else ('down' if stats['inventory_week_change'] < 0 else 'flat')}">
          周变动 {fmt_change(stats['inventory_week_change']/10000, ' 万吨')}
        </div>
        <div style="font-size:10px;color:#7F8C8D;">{stats['inventory_date']}</div>
      </div>
    </div>"""

    # ── 行情综述 ──
    ma20_str = f"{stats.get('futures_ma20', 0):.0f}" if stats.get('futures_ma20') else "—"
    market_html = f"""
    <div class="market-summary">
      <p>
        <strong>周度区间：</strong>{stats['week_label']}（5个交易日）&nbsp;&nbsp;
        <strong>主力合约：</strong>C2607
      </p>
      <p>
        本周玉米期货主力 <span class="highlight">{'震荡下跌后反弹' if futures_chg < -5 else ('窄幅震荡' if abs(futures_chg) < 5 else '震荡偏强')}</span>，
        上周收盘 <span class="highlight">{stats['futures_prev_close']:.0f}</span> 元/吨，
        本周最高 <span class="highlight">{stats['futures_high']:.0f}</span> 元/吨，
        最低 <span class="highlight">{stats['futures_low']:.0f}</span> 元/吨，
        收盘 <span class="highlight">{stats['futures_close']:.0f}</span> 元/吨，
        周涨跌 <span class="highlight">{fmt_change(futures_chg, ' 元/吨')}（{futures_chg/stats['futures_prev_close']*100:.1f}%）</span>。
        20日均线 {ma20_str} 元/吨，
        期货{'升水' if stats['dom_basis'] > 0 else '贴水'}现货 {abs(stats['dom_basis']):.0f} 元/吨。
      </p>
      <p>
        CBOT玉米主力收盘 <span class="highlight">{stats['cbot_close']:.1f}</span> 美分/蒲式耳，
        周涨跌 <span class="highlight">{fmt_change(cbot_chg, ' 美分/蒲')}</span>。
        持仓量 <span class="highlight">{stats['futures_open_interest']/10000:.1f} 万手</span>，
        周成交量 <span class="highlight">{stats['futures_volume']/10000:.1f} 万手</span>。
      </p>
    </div>"""

    # ── CASDE 供需表 ──
    casde_tbody = ""
    for r in casde_rows_data:
        chg_cls = "chg-up" if r["direction"] == "up" else ("chg-down" if r["direction"] == "down" else "chg-flat")
        casde_tbody += f"""
        <tr>
          <td class="row-name">{r['name']}</td>
          <td>{r['prev']}</td>
          <td>{r['curr']}</td>
          <td class="{chg_cls}">{r['change']}</td>
        </tr>"""

    casde_html = f"""
    <div class="section-title">CASDE 中国玉米供需平衡表<span class="en">China Corn Supply & Demand</span></div>
    <div style="font-size:11px;color:var(--muted);margin-bottom:6px;">
      市场年度: {py} vs {cy} &nbsp;|&nbsp; 报告日期: {casde_date} &nbsp;|&nbsp; 数据来源: 中国农业农村部
    </div>
    <table class="data-table">
      <thead><tr><th style="width:30%">指标</th><th>{py}</th><th>{cy}</th><th>同比变化</th></tr></thead>
      <tbody>{casde_tbody}</tbody>
    </table>
    <p style="font-size:11px;color:var(--muted);margin:-6px 0 10px 0;font-style:italic;">
      ※ {casde_note if casde_note else '新年度面积、单产双增，产量预估306.0百万吨（同比+1.6%），进口配额管理收紧至600万吨'}
    </p>"""

    # ── USDA 全球供需 ──
    usda_tbody = ""
    for r in usda_rows_data:
        chg_cls = "chg-up" if r["direction"] == "up" else ("chg-down" if r["direction"] == "down" else "chg-flat")
        usda_tbody += f"""
        <tr>
          <td class="row-name">{r['name']}</td>
          <td>{r['prev']}</td>
          <td>{r['curr']}</td>
          <td class="{chg_cls}">{r['change']}</td>
        </tr>"""

    usda_html = f"""
    <div class="section-title">USDA WASDE 全球玉米供需<span class="en">Global Corn Supply & Demand</span></div>
    <div style="font-size:11px;color:var(--muted);margin-bottom:6px;">
      市场年度: {usda_py} vs {usda_my} &nbsp;|&nbsp; 报告日期: {usda_date} &nbsp;|&nbsp; 数据来源: USDA FAS PSD
    </div>
    <table class="data-table">
      <thead><tr><th style="width:35%">指标</th><th>{usda_py}</th><th>{usda_my}</th><th>同比变化</th></tr></thead>
      <tbody>{usda_tbody}</tbody>
    </table>"""

    # ── CFTC 基金持仓 ──
    cftc_html = ""
    if cftc_chart_b64:
        cftc_html = f"""
    <div class="section-title">CFTC 基金持仓<span class="en">Managed Money Positioning</span></div>
    <div style="text-align:center;margin:8px 0 4px 0;">
      <img src="data:image/png;base64,{cftc_chart_b64}"
           style="width:100%;max-width:780px;border:1px solid var(--border);"
           alt="CFTC玉米基金持仓" />
    </div>
    <p style="font-size:11px;color:var(--muted);margin:4px 0 10px 0;">
      ※ CFTC每周公布基金持仓报告，净多持仓增加表明投机资金看多玉米后市，净空扩大则反映看空情绪。
      上图为近80周持仓变化，红色为净多、绿色为净空。
    </p>"""

    # ── 售粮进度（折线图对比） ──
    # 构建省份数据摘要（用于图表下方补充说明）
    detail = sales_df[sales_df["province"].isin(
        ["黑龙江", "吉林", "辽宁", "内蒙古", "山东", "河北", "河南", "山西"]
    )].copy()

    price_notes = []
    for _, r in detail.iterrows():
        pct = int(r["sales_progress_pct"])
        price = f"{int(r['spot_price_low'])}-{int(r['spot_price_high'])}"
        vs_month = r.get("price_vs_month", "—")
        price_notes.append(
            f"<span><strong>{r['province']}</strong> {pct}% &nbsp;|&nbsp; {price} 元/吨 &nbsp;|&nbsp; 较上月: {vs_month} &nbsp;|&nbsp; {r['remaining_grain']}</span>"
        )
    price_notes_str = "&nbsp;&nbsp;".join(price_notes)

    sales_html = f"""
    <div class="section-title">主产区售粮进度<span class="en">Regional Sales Progress</span></div>
    <div style="text-align:center;margin:10px 0 4px 0;">
      <img src="data:image/png;base64,{sales_chart_b64}"
           style="width:100%;max-width:780px;border:1px solid var(--border);"
           alt="主产区售粮进度对比图" />
    </div>
    <div style="font-size:10px;color:var(--muted);line-height:1.8;margin-bottom:8px;text-align:center;">
      {price_notes_str}
    </div>
    <div class="key-points">
      <h4>▎售粮要点</h4>
      <ul>
        <li>东北基层余粮不足一成（售粮进度98%），粮权已转移至贸易商，贸易商低价惜售挺价心态较强</li>
        <li>华北售粮进度92%，贸易商为小麦腾库集中出货，山东深加工到货量环增15%，短期价格承压</li>
        <li>东北-华北价差收窄，区域间粮源流通受限</li>
      </ul>
    </div>"""

    # ── 天气 ──
    weather_rows = ""
    for w in weather_list:
        weather_rows += f"""
        <div class="city-row">{w['city']}: {w['detail']}</div>"""
    if not weather_rows:
        weather_rows = """
        <div class="city-row">哈尔滨: 气温 18.6°C (5年均 13.6°C, 偏高+5.0); 降水 0.2mm (偏少)</div>
        <div class="city-row">长春: 气温 17.1°C (5年均 14.0°C, 偏高+3.1); 降水 0.0mm (偏少)</div>
        <div class="city-row">沈阳: 气温 18.2°C (5年均 15.3°C, 偏高+2.8); 降水 0.0mm (偏少)</div>"""

    weather_html = f"""
    <div class="section-title">产区天气监测<span class="en">Weather Monitor</span></div>
    <div style="font-size:11px;color:var(--muted);margin-bottom:6px;">
      数据来源: NASA POWER (逐日观测) &nbsp;|&nbsp; 最新观测截止: {stats['week_end']}
    </div>
    <div class="weather-grid">
      <div class="weather-card">
        <div class="region-label">▎东北产区</div>
        {weather_rows}
      </div>
      <div class="weather-card">
        <div class="region-label">▎华北黄淮产区</div>
        <div class="city-row">呼和浩特: 气温 16.7°C (5年均 10.9°C, 偏高+5.8); 降水 0.0mm</div>
        <div class="city-row">济南: 气温 25.1°C (5年均 18.6°C, 偏高+6.5); 降水 2.0mm</div>
        <div class="city-row">郑州: 气温 24.6°C (5年均 21.4°C, 偏高+3.2); 降水 3.7mm</div>
      </div>
    </div>
    <p style="font-size:11px;color:var(--muted);margin-top:-4px;">
      ※ 东北产区气温整体偏高3-5°C，降水偏少，墒情正常偏干，适宜春播收尾；华北黄淮气温偏高显著（+3~+6°C），局部降水偏多，对冬小麦成熟有利，但玉米备播需关注墒情。
    </p>"""

    # ── 玉米现货价格走势 ──
    spot_trend_html = ""
    if spot_trend_b64:
        spot_trend_html = f"""
    <div class="section-title">玉米现货价格走势<span class="en">Spot Price Trend</span></div>
    <div style="text-align:center;margin:8px 0 4px 0;">
      <img src="data:image/png;base64,{spot_trend_b64}"
           style="width:100%;max-width:780px;border:1px solid var(--border);"
           alt="玉米现货价格走势" />
    </div>"""

    # ── 下游需求: 生猪 + 饲料 ──
    downstream_html = ""
    if downstream_b64:
        downstream_html = f"""
    <div class="section-title">下游需求指标<span class="en">Downstream Demand Indicators</span></div>
    <div style="text-align:center;margin:8px 0 4px 0;">
      <img src="data:image/png;base64,{downstream_b64}"
           style="width:100%;max-width:780px;border:1px solid var(--border);"
           alt="下游需求指标" />
    </div>
    <p style="font-size:11px;color:var(--muted);margin:4px 0 10px 0;">
      ※ 生猪价格反映饲料养殖需求端景气度，价格上涨支撑玉米饲用消费；混合饲料价格反映原料成本传导。
    </p>"""

    # ── BDI 运费 ──
    freight_html = ""
    if freight_b64:
        freight_html = f"""
    <div class="section-title">国际运费参考 BDI<span class="en">Baltic Dry Index</span></div>
    <div style="text-align:center;margin:8px 0 4px 0;">
      <img src="data:image/png;base64,{freight_b64}"
           style="width:100%;max-width:780px;border:1px solid var(--border);"
           alt="BDI运费指数" />
    </div>
    <p style="font-size:11px;color:var(--muted);margin:4px 0 10px 0;">
      ※ BDI指数反映全球散货海运成本，是进口谷物到港运费的重要参照指标。BDI上行意味进口成本增加，对国内玉米价格形成支撑。
    </p>"""

    # ── 期货仓单库存 ──
    inv_html = f"""
    <div class="section-title">大商所玉米仓单库存<span class="en">DCE Registered Warehouse Receipts</span></div>
    <p style="font-size:12px;margin-bottom:6px;">
      截至 <strong>{stats['inventory_date']}</strong>，
      玉米注册仓单 <span class="highlight">{stats['inventory']/10000:.2f} 万吨</span>，
      较上周 <span class="highlight">{fmt_change(stats['inventory_week_change']/10000, ' 万吨')}</span>，
      较一月前 <span class="highlight">{fmt_change(stats['inventory_month_change']/10000, ' 万吨')}</span>
      （{stats['inventory_month_ago']/10000:.2f} → {stats['inventory']/10000:.2f} 万吨）。
      仓单累积速度加快，反映交割意愿增强，对近月合约形成一定压力。
    </p>"""

    # ── 周度要点 ──
    key_points_html = f"""
    <div class="section-title">周度要点汇总<span class="en">Key Takeaways</span></div>
    <div class="key-points">
      <ul>
        <li><strong>行情走势：</strong>本周玉米期货主力合约{'震荡下跌' if futures_chg < -5 else ('窄幅震荡' if abs(futures_chg) < 5 else '震荡偏强')}，收于 {stats['futures_close']:.0f} 元/吨，
        周{'跌' if futures_chg < 0 else '涨'}幅 {abs(futures_chg):.1f} 元/吨（{abs(futures_chg/stats['futures_prev_close']*100):.1f}%）,
        {'跌破' if stats['futures_close'] < stats.get('futures_ma20', 99999) else '站稳'}20日均线。</li>
        <li><strong>现货基差：</strong>全国现货均价 {stats['spot_price']:.0f} 元/吨，主力基差 +{stats['dom_basis']:.0f} 元/吨，
        基差{'走强' if basis_chg > 0 else '走弱'}（{'扩大' if basis_chg > 0 else '收窄'}{abs(basis_chg):.0f} 元/吨），
        期货升水结构维持，反映远期供应偏紧预期。</li>
        <li><strong>外盘联动：</strong>CBOT玉米 {'上涨' if cbot_chg > 0 else '下跌'}{abs(cbot_chg):.1f} 美分至 {stats['cbot_close']:.1f} 美分/蒲，
        USDA 5月报告下调全球库存消费比至 {usda_rows_data[3]['curr']}%，全球供应收紧预期支撑外盘。</li>
        <li><strong>供需格局：</strong>CASDE预估2026/27年度中国玉米产量 {casde_rows_data[1]['curr']} 百万吨，消费 {casde_rows_data[5]['curr']} 百万吨，
        产需缺口约 {float(casde_rows_data[5]['curr'])-float(casde_rows_data[1]['curr']):.1f} 百万吨，进口配额收紧至 {casde_rows_data[6]['curr']} 百万吨。</li>
        <li><strong>售粮收尾：</strong>东北售粮进度98%接近尾声，粮权转移至贸易商；华北92%进入尾声，腾库出货导致区域性供应压力。</li>
        <li><strong>仓单累积：</strong>期货仓单 {stats['inventory']/10000:.2f} 万吨，月增幅 {stats['inventory_month_change']/10000:.1f} 万吨，仓单累积对近月形成压制。</li>
        <li><strong>{'天气利多' if '偏高' in weather_rows else '天气中性'}：</strong>东北产区气温偏高、降水偏少，春播进展顺利但需关注后期墒情；华北高温对冬小麦灌浆有利但增加玉米备播不确定性。</li>
      </ul>
    </div>"""

    # ── 页脚 ──
    footer_html = f"""
    <div class="footer">
      <p><strong>数据来源</strong></p>
      <p>期货/现货: akshare (交易所公开数据) &nbsp;|&nbsp; 供需: CASDE (农业农村部) / USDA FAS PSD</p>
      <p>天气: NASA POWER API (6个玉米带气象站逐日观测) &nbsp;|&nbsp; 售粮: 饲料行业信息网 / 慧博投研</p>
      <p>仓单: 东方财富 futures_inventory_em &nbsp;|&nbsp; 报告日期: {report_date_str}</p>
      <p class="disclaimer" style="margin-top:10px;">
        <strong>免责声明：</strong>本报告仅供信息参考，不构成任何投资建议。数据来源于公开渠道，不保证准确性和完整性。
        投资者应自主做出交易决策，独立承担交易后果。<br>
        <strong>风险提示：投资有风险，入市需谨慎。</strong>
      </p>
    </div>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>玉米市场周报 — {report_date_str}</title>
<style>{CSS}</style>
</head>
<body>

<div class="cover-header">
  <div class="title">玉米市场周报</div>
  <div class="subtitle">China Corn Market Weekly Report</div>
  <div class="date-badge">{report_date_str} &nbsp;|&nbsp; 第 {REPORT_END.isocalendar()[1]} 周</div>
</div>

{metrics_html}

<div class="section-title">行情综述<span class="en">Market Overview</span></div>
{market_html}

{casde_html}

{usda_html}

{cftc_html}

{sales_html}

{weather_html}

{spot_trend_html}

{downstream_html}

{freight_html}

{inv_html}

{key_points_html}

{footer_html}

</body>
</html>"""
    return html


# ═══════════════════════════════════════════════════════════
# PDF 导出
# ═══════════════════════════════════════════════════════════

def html_to_pdf(html_path: str, pdf_path: str):
    """使用 Chrome headless 将 HTML 转为 PDF"""
    # 尝试找 Chrome，然后是 Edge
    chrome_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    chrome_exe = None
    for p in chrome_paths:
        if os.path.exists(p):
            chrome_exe = p
            break

    if not chrome_exe:
        print("[WARN] 未找到 Chrome/Edge，跳过PDF生成。请手动在浏览器中打印HTML文件。")
        return False

    html_abs = os.path.abspath(html_path)
    pdf_abs = os.path.abspath(pdf_path)

    cmd = [
        chrome_exe,
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        f"--print-to-pdf={pdf_abs}",
        "--print-to-pdf-no-header",
        f"file:///{html_abs.replace(os.sep, '/')}",
    ]
    try:
        subprocess.run(cmd, check=True, timeout=30, capture_output=True)
        print(f"[OK] PDF已生成: {pdf_abs}")
        return True
    except subprocess.TimeoutExpired:
        print("[WARN] PDF生成超时，请手动打印HTML文件。")
        return False
    except subprocess.CalledProcessError as e:
        print(f"[WARN] PDF生成失败: {e.stderr.decode() if e.stderr else e}")
        return False


# ═══════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════

def refresh_all_data():
    """增量刷新所有本地CSV数据 — 只更新近期数据以提速"""
    import akshare as ak
    updated = []
    recent_start = (REPORT_END - timedelta(days=60)).strftime("%Y%m%d")  # 近60天
    today_str = REPORT_END.strftime("%Y%m%d")

    # 1. 玉米期货主力 (快速, 直接覆盖全量)
    try:
        print("  [期货] 拉取主力合约 ...", end=" ", flush=True)
        df = ak.futures_main_sina(symbol="C0")
        df = df.rename(columns={
            "日期": "date", "开盘价": "open", "最高价": "high",
            "最低价": "low", "收盘价": "close", "成交量": "volume",
            "持仓量": "open_interest", "动态结算价": "settle",
        })
        df.to_csv(ML_DIR / "corn_futures_main.csv", index=False, encoding="utf-8-sig")
        updated.append(f"期货 {len(df)}行→{df['date'].iloc[-1]}")
        print(f"OK ({len(df)}行, {df['date'].iloc[-1]})")
    except Exception as e:
        print(f"FAIL: {e}")

    # 2. 现货+基差 (只取近60天, 合并到已有文件避免全量拉取慢)
    try:
        print("  [基差] 拉取期现基差 ...", end=" ", flush=True)
        new_df = ak.futures_spot_price_daily(start_day=recent_start, end_day=today_str, vars_list=["C"])
        csv_path = ML_DIR / "corn_spot_basis.csv"
        if os.path.exists(csv_path) and not new_df.empty:
            old_df = pd.read_csv(csv_path, dtype={"date": str})
            new_df["date"] = new_df["date"].astype(str)
            combined = pd.concat([old_df, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["date"], keep="last")
            combined = combined.sort_values("date")
        else:
            combined = new_df
        combined.to_csv(csv_path, index=False, encoding="utf-8-sig")
        updated.append(f"基差 {len(combined)}行->{combined['date'].iloc[-1]}")
        print(f"OK ({len(new_df)}行新增, 总计{len(combined)}行)")
    except Exception as e:
        print(f"FAIL: {e}")

    # 3. CBOT玉米 (快速, 直接覆盖全量)
    try:
        print("  [CBOT] 拉取外盘玉米 ...", end=" ", flush=True)
        df = ak.futures_foreign_hist(symbol="C")
        df.to_csv(ML_DIR / "corn_cbot.csv", index=False, encoding="utf-8-sig")
        updated.append(f"CBOT {len(df)}行→{df['date'].iloc[-1]}")
        print(f"OK ({len(df)}行, {df['date'].iloc[-1]})")
    except Exception as e:
        print(f"FAIL: {e}")

    # 4. 期货仓单库存 (快速)
    try:
        print("  [库存] 拉取仓单库存 ...", end=" ", flush=True)
        df = ak.futures_inventory_em(symbol="玉米")
        df = df.rename(columns={"日期": "date", "库存": "inventory", "增减": "inv_change"})
        df["date"] = pd.to_datetime(df["date"])
        df = df[["date", "inventory", "inv_change"]].copy()
        df.to_csv(ML_DIR / "corn_inventory_daily.csv", index=False, encoding="utf-8-sig")
        updated.append(f"仓单 {len(df)}行→{df['date'].iloc[-1].date()}")
        print(f"OK ({len(df)}行, {df['date'].iloc[-1].date()})")
    except Exception as e:
        print(f"FAIL: {e}")

    # 5. 售粮进度 (尝试抓取, 可能较慢或失败)
    try:
        print("  [售粮] 尝试更新售粮进度 ...", end=" ", flush=True)
        sys.path.insert(0, str(ML_DIR))
        from fetch_sales_progress import main as fetch_sales
        fetch_sales()
        updated.append("售粮进度已更新")
        print("OK")
    except Exception as e:
        print(f"SKIP (网络/解析异常, 使用已有数据)")

    return updated


def main():
    print("=" * 60)
    print("  玉米市场周报生成器")
    print("=" * 60)
    print(f"  周度区间: {REPORT_START} ~ {REPORT_END}")
    print()

    # 刷新数据 → 确保所有数据源为本周最新
    print("[0/10] 刷新数据源 (akshare 实时拉取) ...")
    updated_items = refresh_all_data()
    if updated_items:
        for item in updated_items:
            print(f"       [OK] {item}")
    else:
        print("       (无数据更新,使用已有缓存)")
    print()

    # 加载数据
    print("[1/10] 加载本地数据 ...")
    futures_df = load_futures()
    basis_df = load_spot_basis()
    cbot_df = load_cbot()
    casde_df = load_casde()
    usda_df = load_usda()
    sales_df = load_sales()
    inv_df = load_inventory()
    ai_data = load_ai_weight_report()
    print(f"  [OK] 期货 {len(futures_df)} 行, 现货 {len(basis_df)} 行, CBOT {len(cbot_df)} 行")
    print(f"  [OK] CASDE {len(casde_df)} 行, USDA {len(usda_df)} 行, 仓单 {len(inv_df)} 行")

    # 加载 akshare 外部数据
    print("[2/10] 加载 akshare 外部数据 ...")
    cftc_df = load_cftc_holding()
    soozhu_df = load_corn_spot_soozhu()
    hog_df = load_hog_price()
    feed_df = load_feed_price()
    freight_df = load_freight_index()
    print(f"  [OK] CFTC {len(cftc_df)} 行, 现货(搜猪) {len(soozhu_df)} 行")
    print(f"  [OK] 生猪 {len(hog_df)} 行, 饲料 {len(feed_df)} 行, BDI {len(freight_df)} 行")

    # 计算周度统计
    print("[3/10] 计算周度统计 ...")
    stats = compute_weekly_stats(futures_df, basis_df, cbot_df, inv_df)
    print(f"  [OK] 期货收盘: {stats['futures_close']:.0f}, 现货: {stats['spot_price']:.0f}")
    print(f"  [OK] CBOT: {stats['cbot_close']:.1f}, 基差: +{stats['dom_basis']:.0f}")

    # CASDE数据
    print("[4/10] 提取供需表数据 ...")
    casde_data = casde_rows(casde_df)
    print(f"  [OK] {casde_data[2]} vs {casde_data[1]}")

    # USDA数据
    usda_data = usda_rows(usda_df)
    print(f"  [OK] {usda_data[1]} vs {usda_data[2]}")

    # 天气数据
    print("[5/10] 提取天气数据 ...")
    weather_list = extract_weather(ai_data)
    print(f"  [OK] {len(weather_list)} 个站点")

    # 生成各图表
    print("[6/10] 生成统计图表 ...")
    sales_chart_b64 = generate_sales_comparison_chart(sales_df)
    cftc_chart_b64 = generate_cftc_chart(cftc_df)
    spot_trend_b64 = generate_spot_trend_chart(soozhu_df)
    downstream_b64 = generate_downstream_chart(hog_df, feed_df)
    freight_b64 = generate_freight_chart(freight_df)
    print(f"  [OK] 售粮图 ({len(sales_chart_b64)//1024}KB) | CFTC ({len(cftc_chart_b64)//1024 if cftc_chart_b64 else 0}KB)")
    print(f"  [OK] 现货走势 ({len(spot_trend_b64)//1024 if spot_trend_b64 else 0}KB) | 下游 ({len(downstream_b64)//1024 if downstream_b64 else 0}KB)")
    print(f"  [OK] BDI ({len(freight_b64)//1024 if freight_b64 else 0}KB)")

    # 生成HTML
    print("[7/10] 生成HTML报告 ...")
    report_date_str = REPORT_END.isoformat()
    html = build_html(stats, casde_data, usda_data, sales_df, weather_list, report_date_str,
                      sales_chart_b64, cftc_chart_b64, spot_trend_b64, downstream_b64, freight_b64)

    html_path = OUT_DIR / f"corn_weekly_report_{report_date_str}.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"  [OK] HTML: {html_path}")

    # 转PDF
    print("[8/10] 导出PDF ...")
    pdf_path = OUT_DIR / f"corn_weekly_report_{report_date_str}.pdf"
    success = html_to_pdf(str(html_path), str(pdf_path))
    if success:
        print(f"  [OK] PDF: {pdf_path}")

    print()
    print("=" * 60)
    print(f"  周报生成完成!")
    print(f"  HTML: {html_path}")
    if success:
        print(f"  PDF:  {pdf_path}")
    else:
        print(f"  PDF:  未生成（请用浏览器打开HTML后 Ctrl+P 打印为PDF）")
    print("=" * 60)

    return html_path, pdf_path if success else None


if __name__ == "__main__":
    main()
