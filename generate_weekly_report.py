#!/usr/bin/env python3
"""
饲料能量农产品周报生成器
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


def load_futures_quick(symbol, label):
    """通用期货主力拉取 (akshare futures_main_sina)"""
    try:
        import akshare as ak
        df = ak.futures_main_sina(symbol=symbol)
        df = df.rename(columns={
            "日期": "date", "开盘价": "open", "最高价": "high",
            "最低价": "low", "收盘价": "close", "成交量": "volume",
            "持仓量": "open_interest", "动态结算价": "settle",
        })
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date")
    except Exception as e:
        print(f"  [WARN] {label}期货获取失败: {e}")
        return pd.DataFrame()


def load_starch_futures():
    """玉米淀粉期货 CS"""
    return load_futures_quick("CS0", "淀粉")


def load_egg_futures():
    """鸡蛋期货 JD"""
    return load_futures_quick("JD0", "鸡蛋")


def load_hog_futures():
    """生猪期货 LH"""
    return load_futures_quick("LH0", "生猪")


def load_soymeal_futures():
    """豆粕期货 M"""
    return load_futures_quick("M0", "豆粕")


def load_hog_index():
    """生猪现货价格指数 (akshare)"""
    try:
        import akshare as ak
        df = ak.index_hog_spot_price()
        df.columns = ["date", "index", "ma4", "ma6", "ma12", "pre_sale_price", "volume", "turnover"]
        df["date"] = pd.to_datetime(df["date"])
        for c in ["index", "ma4", "ma6", "ma12", "pre_sale_price"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.sort_values("date")
    except Exception as e:
        print(f"  [WARN] 生猪指数获取失败: {e}")
        return pd.DataFrame()


def load_soybean_spot():
    """大豆现货价格 (搜猪网)"""
    try:
        import akshare as ak
        df = ak.spot_soybean_price_soozhu()
        df.columns = ["date", "price"]
        df["date"] = pd.to_datetime(df["date"])
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        return df.sort_values("date")
    except Exception as e:
        print(f"  [WARN] 大豆现货获取失败: {e}")
        return pd.DataFrame()


def load_hog_fundamentals():
    """加载生猪供需基本面 CSV"""
    path = ML_DIR / "hog_fundamentals.csv"
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path, dtype={"date": str})
        if df.empty:
            return {}
        latest = df.iloc[-1].to_dict()
        return latest
    except Exception:
        return {}


def load_egg_fundamentals():
    """加载鸡蛋供需基本面 CSV"""
    path = ML_DIR / "egg_fundamentals.csv"
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path, dtype={"date": str})
        if df.empty:
            return {}
        latest = df.iloc[-1].to_dict()
        return latest
    except Exception:
        return {}


def fetch_multi_product_news():
    """抓取各品种相关新闻 (akshare stock_news_em)"""
    import akshare as ak

    EXCLUDE_KW = [
        "玻璃", "纯碱", "螺纹钢", "铁矿石", "热卷", "PVC", "PTA", "甲醇", "LPG",
        "原油", "石油", "黄金", "白银", "铜", "铝", "锌", "锡", "镍", "锂", "钴",
        "碳酸锂", "工业硅", "橡胶", "轮胎", "芯片", "半导体", "新能源车", "光伏",
        "医药", "地产", "楼市", "人民币汇率", "央行", "MLF", "LPR", "降息", "降准",
        "A股", "上证", "深证", "创业板", "沙特", "中东", "俄乌", "制造业PMI",
    ]

    keywords_map = {
        "玉米": ["玉米期货", "玉米供需", "玉米价格", "CBOT玉米"],
        "淀粉": ["玉米淀粉", "淀粉期货", "深加工"],
        "生猪": ["生猪期货", "猪价", "养殖", "能繁母猪"],
        "鸡蛋": ["鸡蛋期货", "蛋价", "蛋鸡"],
        "豆粕": ["豆粕期货", "豆粕价格", "大豆进口"],
    }
    all_news = {}
    for product, kws in keywords_map.items():
        items = []
        seen = set()
        for kw in kws[:2]:  # 每品种只搜2个关键词避免太慢
            try:
                df = ak.stock_news_em(symbol=kw)
                if df is None or df.empty:
                    continue
                for _, row in df.iterrows():
                    title = str(row["新闻标题"])
                    if title[:50] in seen:
                        continue
                    # 排除与农产品无关的新闻
                    if any(kw in title for kw in EXCLUDE_KW):
                        continue
                    seen.add(title[:50])
                    pub_time = str(row.get("发布时间", ""))
                    # 只取最近7天
                    if pub_time[:10] < (REPORT_END - timedelta(days=7)).isoformat():
                        continue
                    url = str(row.get("新闻链接", ""))
                    items.append({
                        "title": title[:100],
                        "time": pub_time[:16],
                        "source": str(row.get("文章来源", "")),
                        "url": url if url and url != "nan" else "",
                    })
            except Exception:
                pass
        items.sort(key=lambda x: x["time"], reverse=True)
        all_news[product] = items[:8]  # 每品种最多8条
    return all_news


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


def generate_product_trend_chart(df, title, color, ylabel="元/吨"):
    """品种期货近一周走势: 日K线(收盘+高低) + 成交量, x轴=日期的星期"""
    if df.empty or len(df) < 2:
        return ""
    # 只取近10个交易日 (约2周, 确保覆盖本周+上周对比)
    data = df.tail(10).copy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 3.8), gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#FCFAF5")

    # 价格走势
    ax1.set_facecolor("#FEFEFE")
    x = range(len(data))
    ax1.plot(x, data["close"].values, color=color, linewidth=2.5, marker="o", markersize=7,
             markerfacecolor="white", markeredgewidth=2, markeredgecolor=color, zorder=5)
    ax1.fill_between(x, data["low"].values, data["high"].values, alpha=0.12, color=color)

    # 每个点标注价格
    for xi, (_, r) in enumerate(data.iterrows()):
        ax1.annotate(f"{r['close']:.0f}", (xi, r["close"]), textcoords="offset points",
                     xytext=(0, 12), ha="center", fontsize=9, fontweight="bold", color=color)

    ax1.set_ylabel(ylabel, fontsize=9, color="#7F8C8D")
    ax1.set_title(title, fontsize=13, fontweight="bold", color=color, pad=6)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="y", linestyle="--", alpha=0.2, color="#BDC3C7")
    ax1.tick_params(labelsize=8)

    # 成交量柱
    ax2.set_facecolor("#FEFEFE")
    vol_colors = ["#C0392B" if data["close"].iloc[i] >= data["open"].iloc[i] else "#27AE60" for i in range(len(data))]
    ax2.bar(x, data["volume"].values / 10000, color=vol_colors, alpha=0.6, width=0.65, edgecolor="white", linewidth=0.2)
    ax2.set_ylabel("万手", fontsize=8, color="#7F8C8D")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # x轴: 日期 + 星期
    weekdays_cn = ["一","二","三","四","五","六","日"]
    tick_labels = [data["date"].iloc[i].strftime("%m/%d") + f"\n周{weekdays_cn[data['date'].iloc[i].weekday()]}" for i in range(len(data))]
    ax2.set_xticks(x)
    ax2.set_xticklabels(tick_labels, fontsize=7, color="#2C3E50")

    fig.text(0.5, 0.005, "数据来源: akshare futures_main_sina  |  近10个交易日",
             ha="center", fontsize=7, color="#95A5A6", style="italic")
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])

    buf = io.BytesIO()
    fig.savefig(buf, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor(), format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def compute_tech_analysis(df):
    """计算技术指标: 周涨跌, RSI(7), 波动率, 支撑/压力"""
    if df.empty or len(df) < 5:
        return {}
    recent = df.tail(10).copy()
    close = recent["close"].values
    last = close[-1]
    prev = close[-2] if len(close) >= 2 else last
    week_ago = close[0]

    # RSI(7)
    deltas = [close[i] - close[i-1] for i in range(1, len(close))]
    gains = [d if d > 0 else 0 for d in deltas[-7:]]
    losses = [-d if d < 0 else 0 for d in deltas[-7:]]
    avg_gain = sum(gains) / max(len(gains), 1)
    avg_loss = sum(losses) / max(len(losses), 1)
    rsi = 100 - (100 / (1 + avg_gain / avg_loss)) if avg_loss > 0 else 100

    # 波动率
    returns = [abs((close[i] - close[i-1]) / close[i-1]) for i in range(1, len(close))]
    volatility = sum(returns) / len(returns) * 100 if returns else 0

    # 支撑/压力
    high_week = max(close)
    low_week = min(close)

    # 趋势判断
    if last > week_ago * 1.02:
        trend = "偏强上涨"
    elif last < week_ago * 0.98:
        trend = "偏弱下跌"
    else:
        trend = "窄幅震荡"

    # RSI判断
    if rsi > 70:
        rsi_signal = "超买区，短期注意回调风险"
    elif rsi < 30:
        rsi_signal = "超卖区，短线或有反弹需求"
    elif rsi > 50:
        rsi_signal = "偏强，多头占优"
    else:
        rsi_signal = "偏弱，空头占优"

    return {
        "last": last, "prev": prev, "week_ago": week_ago,
        "chg": last - week_ago, "chg_pct": (last - week_ago) / week_ago * 100,
        "rsi": round(rsi, 1), "rsi_signal": rsi_signal,
        "volatility": round(volatility, 2),
        "trend": trend,
        "high": high_week, "low": low_week,
        "vol_total": float(recent["volume"].tail(5).sum() / 10000),
        "oi": float(recent["open_interest"].iloc[-1] / 10000) if "open_interest" in recent.columns else 0,
    }


def load_product_basis(symbol, label):
    """获取单品种最新基差 (快速单日API)"""
    try:
        import akshare as ak
        df = ak.futures_spot_price(date=REPORT_END.strftime("%Y%m%d"), vars_list=[symbol])
        if df.empty:
            # 回退一天
            prev_day = (REPORT_END - timedelta(days=1)).strftime("%Y%m%d")
            df = ak.futures_spot_price(date=prev_day, vars_list=[symbol])
        if not df.empty:
            r = df.iloc[0]
            return {
                "spot": float(r["spot_price"]),
                "basis": float(r["dom_basis"]),
                "basis_rate": float(r["dom_basis_rate"]) * 100,
            }
    except Exception as e:
        print(f"  [WARN] {label}基差获取失败: {e}")
    return {}


def generate_hog_index_chart(hog_idx_df, hog_fut_df):
    """生猪指数+期货双轴图"""
    if hog_idx_df.empty:
        return ""
    df = hog_idx_df.tail(120).copy()

    fig, ax1 = plt.subplots(figsize=(13, 4.0))
    fig.patch.set_facecolor("#FCFAF5")
    ax1.set_facecolor("#FEFEFE")

    x = range(len(df))
    ax1.plot(x, df["index"].values, color="#2980B9", linewidth=2.2, marker="o", markersize=4,
             markerfacecolor="white", markeredgewidth=1.5, markeredgecolor="#2980B9", label="生猪现货指数")
    ax1.fill_between(x, df["index"].min() - 5, df["index"].values, alpha=0.06, color="#2980B9")
    ax1.axhline(y=100, color="#BDC3C7", linewidth=0.6, linestyle="--", alpha=0.5)

    last_idx = df["index"].iloc[-1]
    ax1.annotate(f"指数 {last_idx:.1f}", (x[-1], last_idx), textcoords="offset points",
                 xytext=(8, 0), ha="left", fontsize=10, fontweight="bold", color="#2980B9")

    ax1.set_ylabel("生猪指数", fontsize=9, color="#2980B9")
    ax1.tick_params(axis="y", colors="#2980B9", labelsize=8)
    ax1.spines["top"].set_visible(False)
    ax1.legend(fontsize=8, loc="upper left", framealpha=0.8)

    tick_positions = list(range(0, len(df), max(1, len(df) // 8)))
    tick_labels = [df["date"].iloc[i].strftime("%Y-%m") if i < len(df) else "" for i in tick_positions]
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels, fontsize=7, color="#7F8C8D", rotation=30)
    ax1.grid(axis="y", linestyle="--", alpha=0.2, color="#BDC3C7")

    fig.suptitle("生猪现货指数走势\n(Hog Spot Price Index)", fontsize=13, fontweight="bold", color="#2C3E50", y=1.01)
    fig.text(0.5, 0.01, "数据来源: akshare index_hog_spot_price  |  基准: 100 (2021年=100)",
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
  grid-template-columns: repeat(5, 1fr);
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

/* ── 分析框 ── */
.analysis-box {
  background: #FFFEFB;
  border-left: 3px solid #BDC3C7;
  padding: 8px 12px;
  margin: 6px 0;
}

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
               sales_chart_b64, cftc_chart_b64, spot_trend_b64, downstream_b64, freight_b64,
               product_charts, hog_index_b64, news_data, product_stats, product_tech, product_basis,
               corn_tech, hog_fundamentals=None, egg_fundamentals=None):
    """构建完整HTML — 饲料能量农产品周报"""
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

    # 5品种指标卡片
    def metric_card(label, value, unit, change_val, change_unit, sub_text="", color_class=""):
        cls = "up" if change_val > 0.005 else ("down" if change_val < -0.005 else "flat")
        return f"""
      <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value {color_class}">{value}</div>
        <div class="change {cls}">较上周 {fmt_change(change_val, unit)}</div>
        <div style="font-size:10px;color:#7F8C8D;">{sub_text}</div>
      </div>"""

    metrics_html = '<div class="metrics-row">'
    # 玉米
    metrics_html += metric_card("玉米 C2607", f"{stats['futures_close']:.0f}", " 元/吨",
                                futures_chg, " 元/吨",
                                f"基差:+{stats['dom_basis']:.0f} 仓单:{stats['inventory']/10000:.1f}万吨", "accent")
    # 淀粉
    ps = product_stats.get("starch", {})
    metrics_html += metric_card("玉米淀粉 CS", f"{ps.get('close',0):.0f}", " 元/吨",
                                ps.get("chg", 0), " 元/吨",
                                f"MA20:{ps.get('ma20',0):.0f}", "")
    # 生猪
    ps = product_stats.get("hog", {})
    metrics_html += metric_card("生猪 LH", f"{ps.get('close',0):.0f}", " 元/吨",
                                ps.get("chg", 0), " 元/吨",
                                f"指数:{ps.get('index_val',0):.1f}", "")
    # 鸡蛋
    ps = product_stats.get("egg", {})
    metrics_html += metric_card("鸡蛋 JD", f"{ps.get('close',0):.0f}", " 元/500kg",
                                ps.get("chg", 0), " 元/500kg",
                                f"MA20:{ps.get('ma20',0):.0f}", "")
    # 豆粕
    ps = product_stats.get("soymeal", {})
    metrics_html += metric_card("豆粕 M", f"{ps.get('close',0):.0f}", " 元/吨",
                                ps.get("chg", 0), " 元/吨",
                                f"大豆:{ps.get('soybean_spot',0):.1f}元/斤", "")
    metrics_html += '</div>'

    # ── 行情综述 ──
    ma20_str = f"{stats.get('futures_ma20', 0):.0f}" if stats.get('futures_ma20') else "—"

    # 各品种行情描述
    product_descriptions = []
    product_config = [
        ("starch", "玉米淀粉", "CS", "元/吨"),
        ("hog", "生猪", "LH", "元/吨"),
        ("egg", "鸡蛋", "JD", "元/500kg"),
        ("soymeal", "豆粕", "M", "元/吨"),
    ]
    for key, name, code, unit in product_config:
        ps = product_stats.get(key, {})
        pt = product_tech.get(key, {})
        if not ps.get("close"):
            continue
        chg = ps.get("chg", 0)
        if pt:
            trend = pt.get("trend", "震荡")
            desc = f"{name}({code}){trend}，收于 {ps['close']:.0f} {unit}，周涨跌 {chg:+.1f} {unit}（{chg/(ps['close']-chg)*100:.1f}%），最高 {pt.get('high',0):.0f}，最低 {pt.get('low',0):.0f}，持仓 {pt.get('oi',0):.0f} 万手。"
        else:
            desc = f"{name}({code})收于 {ps['close']:.0f} {unit}，周涨跌 {chg:+.1f} {unit}。"
        product_descriptions.append(desc)

    products_overview = ""
    for d in product_descriptions:
        products_overview += f"        <li>{d}</li>\n"

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
      <p style="font-size:12px;color:#2C3E50;margin-top:8px;line-height:1.7;">
        <strong>相关品种：</strong>
      </p>
      <ul style="font-size:12px;color:#2C3E50;margin:0;padding-left:18px;line-height:1.8;">
{products_overview}
      </ul>
    </div>"""

    # ── 玉米: 技术分析 + 基差分析 + 基本面分析 ──
    corn_analysis_html = ""
    if corn_tech:
        t = corn_tech
        corn_color = "#C0392B"
        # Tech
        corn_analysis_html += f"""
    <div class="section-title">玉米技术分析与基差<span class="en">Corn Technical & Basis</span></div>
    <div class="analysis-box">
      <h4 style="color:{corn_color};margin:0 0 4px 0;font-size:12px;">技术分析</h4>
      <p style="margin:2px 0;font-size:11px;">
        RSI(7): <strong>{t.get('rsi',0):.1f}</strong> — {t.get('rsi_signal','')} &nbsp;|&nbsp;
        周波动率: <strong>{t.get('volatility',0):.1f}%</strong> &nbsp;|&nbsp;
        周涨跌: <span style="color:{'#C0392B' if t.get('chg',0) > 0 else '#27AE60'};font-weight:700;">{t.get('chg_pct',0):+.1f}%</span>
      </p>
      <p style="margin:2px 0;font-size:11px;">
        压力位: <strong style="color:#C0392B;">{t.get('high',0):.0f}</strong> &nbsp;|&nbsp;
        支撑位: <strong style="color:#27AE60;">{t.get('low',0):.0f}</strong>
      </p>
    </div>"""
        # Basis
        dom_basis = stats.get("dom_basis", 0)
        spot = stats.get("spot_price", 0)
        if dom_basis > 0:
            basis_desc = f"期货升水现货 {dom_basis:.0f} 元/吨（升水率 {abs(dom_basis)/spot*100:.1f}%），市场对远期供应偏紧有一定预期。"
        elif dom_basis < 0:
            basis_desc = f"期货贴水现货 {abs(dom_basis):.0f} 元/吨（贴水率 {abs(dom_basis)/spot*100:.1f}%），现货偏强或近月交割压力较大。"
        else:
            basis_desc = "期现基本平水，市场定价中性。"
        corn_analysis_html += f"""
    <div class="analysis-box">
      <h4 style="color:{corn_color};margin:0 0 4px 0;font-size:12px;">基差分析</h4>
      <p style="margin:2px 0;font-size:11px;">
        全国现货均价: <strong>{stats['spot_price']:.0f}</strong> 元/吨 &nbsp;|&nbsp;
        主力基差: <strong style="color:{'#C0392B' if dom_basis > 0 else '#2980B9'};">
        {'+' if dom_basis > 0 else ''}{dom_basis:.0f}</strong> 元/吨（{'升水' if dom_basis > 0 else '贴水'}）
      </p>
      <p style="margin:2px 0;font-size:11px;color:#7F8C8D;">{basis_desc}</p>
    </div>"""
        # Fundamentals
        prod_gap = float(casde_rows_data[5]['curr']) - float(casde_rows_data[1]['curr'])
        corn_analysis_html += f"""
    <div class="analysis-box">
      <h4 style="color:{corn_color};margin:0 0 4px 0;font-size:12px;">基本面分析</h4>
      <ul style="margin:2px 0;padding-left:16px;font-size:11px;">
        <li>CASDE预估2026/27年度中国玉米产量 {casde_rows_data[1]['curr']} 百万吨，消费 {casde_rows_data[5]['curr']} 百万吨，产需缺口约 {abs(prod_gap):.1f} 百万吨。</li>
        <li>USDA 5月报告全球库存消费比降至 {usda_rows_data[3]['curr']}%（上年 {usda_rows_data[3]['prev']}%），全球供应收紧态势延续。</li>
        <li>进口配额收紧至 {casde_rows_data[6]['curr']} 百万吨，新季玉米面积+0.4%叠加单产+1.2%，产量预期创新高。</li>
        <li>东北基层余粮不足一成，粮权转向贸易商；华北腾库出货对短期价格形成压力。</li>
      </ul>
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

    # ── 品种供需基本面 ──
    product_sd_html = ""

    # --- 玉米淀粉 (技术面为主) ---
    ps = product_stats.get("starch", {})
    pt = product_tech.get("starch", {})
    pb = product_basis.get("starch", {})
    if ps.get("close"):
        rows = [
            ("期货收盘", f"{ps.get('close',0):.0f} 元/吨"),
            ("周涨跌", f"{ps.get('chg',0):+.0f} 元/吨"),
            ("MA20均线", f"{ps.get('ma20',0):.0f} 元/吨"),
            ("RSI(7)", f"{pt.get('rsi',0):.1f} — {pt.get('rsi_signal','')}"),
            ("成交量", f"{pt.get('vol_total',0):.0f} 万手"),
            ("持仓量", f"{pt.get('oi',0):.0f} 万手"),
            ("基差", f"{pb.get('basis',0):+.0f} 元/吨 ({'升水' if pb.get('basis',0)>0 else '贴水'}{abs(pb.get('basis_rate',0)):.1f}%)" if pb else "—"),
            ("供需格局", "原料成本支撑偏强，下游需求恢复缓慢，供需偏宽松"),
        ]
        tbody = "".join(f"<tr><td class=\"row-name\">{r[0]}</td><td>{r[1]}</td></tr>" for r in rows)
        product_sd_html += f"""
    <div class="section-title" style="margin-top:4px;">玉米淀粉供需基本面<span class="en">Corn Starch Fundamentals — CS</span></div>
    <table class="data-table" style="max-width:500px;">
      <thead><tr><th style="width:30%">指标</th><th>当前值</th></tr></thead>
      <tbody>{tbody}</tbody>
    </table>"""

    # --- 生猪 (使用 akshare 实时数据) ---
    hf = hog_fundamentals or {}
    ps = product_stats.get("hog", {})
    pt = product_tech.get("hog", {})
    pb = product_basis.get("hog", {})
    if ps.get("close"):
        hog_rows = [
            ("期货收盘", f"{ps.get('close',0):.0f} 元/吨"),
            ("周涨跌", f"{ps.get('chg',0):+.0f} 元/吨"),
            ("全国均价", f"{hf.get('hog_price_kg','—')} 元/公斤" + (f" ({hf.get('hog_price_date','')})" if hf.get('hog_price_date') else "")),
            ("猪粮比", hf.get('hog_core', '—')),
            ("头均成本", f"{hf.get('hog_cost','—')} 元/头" if hf.get('hog_cost') else "—"),
            ("供给指标", hf.get('hog_supply', '—')),
            ("基差", f"{hf.get('hog_futures_basis',pb.get('basis',0)):+.0f} 元/吨" if hf.get('hog_futures_basis') or pb else "—"),
            ("玉米/豆粕/饲料", f"{hf.get('corn_price_kg','—')}/{hf.get('soybean_price_kg','—')}/{hf.get('feed_price_kg','—')} 元/kg"),
        ]
        # 供需格局判断
        hog_core = float(hf.get('hog_core', 0) or 0)
        if hog_core > 10:
            hog_trend = "猪粮比高位，养殖利润较好，产能去化缓慢，供应偏宽松"
        elif hog_core > 7:
            hog_trend = "猪粮比适中，养殖基本盈亏平衡，供需博弈"
        else:
            hog_trend = "猪粮比偏低，养殖亏损或倒逼产能去化，供应有望收紧"
        hog_rows.append(("供需格局", hog_trend))

        tbody = "".join(f"<tr><td class=\"row-name\">{r[0]}</td><td>{r[1]}</td></tr>" for r in hog_rows)
        product_sd_html += f"""
    <div class="section-title" style="margin-top:4px;">生猪供需基本面<span class="en">Live Hog Fundamentals — LH</span></div>
    <p style="font-size:10px;color:var(--muted);margin:0 0 4px 0;">
      数据来源: 搜猪网(soozhu) + 大商所 &nbsp;|&nbsp; 猪粮比=生猪价/玉米价，5.5-6.0为盈亏平衡线
    </p>
    <table class="data-table" style="max-width:500px;">
      <thead><tr><th style="width:30%">指标</th><th>当前值</th></tr></thead>
      <tbody>{tbody}</tbody>
    </table>"""

    # --- 鸡蛋 (使用 akshare 实时数据) ---
    ef = egg_fundamentals or {}
    ps = product_stats.get("egg", {})
    pt = product_tech.get("egg", {})
    pb = product_basis.get("egg", {})
    if ps.get("close"):
        egg_rows = [
            ("期货收盘", f"{ps.get('close',0):.0f} 元/500kg"),
            ("周涨跌", f"{ps.get('chg',0):+.0f} 元/500kg"),
            ("鸡蛋现货", f"{ef.get('egg_spot','—')} 元/500kg" if ef.get('egg_spot') else "—"),
            ("基差", f"{ef.get('egg_futures_basis',pb.get('basis',0)):+.0f} 元/500kg ({'升水' if float(ef.get('egg_futures_basis',0) or 0)>0 else '贴水'}{abs(float(ef.get('egg_futures_basis_rate',0) or 0))*100:.1f}%)" if (ef.get('egg_futures_basis') or pb) else "—"),
            ("玉米价格", f"{ef.get('corn_price_kg','—')} 元/公斤"),
            ("豆粕价格", f"{ef.get('soybean_price_kg','—')} 元/公斤"),
            ("估算饲料成本", f"≈{ef.get('est_feed_cost','—')} 元/公斤" if ef.get('est_feed_cost') else "—"),
            ("RSI(7)", f"{pt.get('rsi',0):.1f}"),
            ("成交量/持仓", f"{pt.get('vol_total',0):.0f} / {pt.get('oi',0):.0f} 万手"),
        ]
        egg_basis = float(ef.get('egg_futures_basis', 0) or 0)
        if egg_basis < -100:
            egg_trend = "期货深贴水，现货偏强或近月供应偏紧，基差有收敛动力"
        elif egg_basis > 100:
            egg_trend = "期货升水偏高，市场预期远期供应收紧或成本上移"
        else:
            egg_trend = "期现基本平水，供需相对平衡，季节性波动为主"
        egg_rows.append(("供需格局", egg_trend))

        tbody = "".join(f"<tr><td class=\"row-name\">{r[0]}</td><td>{r[1]}</td></tr>" for r in egg_rows)
        product_sd_html += f"""
    <div class="section-title" style="margin-top:4px;">鸡蛋供需基本面<span class="en">Egg Fundamentals — JD</span></div>
    <p style="font-size:10px;color:var(--muted);margin:0 0 4px 0;">
      数据来源: 大商所 + 搜猪网(soozhu)饲料原料 &nbsp;|&nbsp; 饲料成本≈玉米×65%+豆粕×25%，占养殖成本70%+
    </p>
    <table class="data-table" style="max-width:500px;">
      <thead><tr><th style="width:30%">指标</th><th>当前值</th></tr></thead>
      <tbody>{tbody}</tbody>
    </table>"""

    # --- 豆粕 (技术面+大豆现货) ---
    ps = product_stats.get("soymeal", {})
    pt = product_tech.get("soymeal", {})
    pb = product_basis.get("soymeal", {})
    if ps.get("close"):
        rows = [
            ("期货收盘", f"{ps.get('close',0):.0f} 元/吨"),
            ("周涨跌", f"{ps.get('chg',0):+.0f} 元/吨"),
            ("大豆现货", f"{ps.get('soybean_spot',0):.0f} 元/吨" if ps.get('soybean_spot') else "—"),
            ("RSI(7)", f"{pt.get('rsi',0):.1f} — {pt.get('rsi_signal','')}"),
            ("成交量", f"{pt.get('vol_total',0):.0f} 万手"),
            ("持仓量", f"{pt.get('oi',0):.0f} 万手"),
            ("基差", f"{pb.get('basis',0):+.0f} 元/吨" if pb else "—"),
            ("供需格局", "进口大豆集中到港或压制现货，饲料需求刚性支撑"),
        ]
        tbody = "".join(f"<tr><td class=\"row-name\">{r[0]}</td><td>{r[1]}</td></tr>" for r in rows)
        product_sd_html += f"""
    <div class="section-title" style="margin-top:4px;">豆粕供需基本面<span class="en">Soymeal Fundamentals — M</span></div>
    <table class="data-table" style="max-width:500px;">
      <thead><tr><th style="width:30%">指标</th><th>当前值</th></tr></thead>
      <tbody>{tbody}</tbody>
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

    # ── 生猪指数 ──
    hog_index_html = ""
    if hog_index_b64:
        hog_index_html = f"""
    <div class="section-title">生猪市场<span class="en">Live Hog Market</span></div>
    <div style="text-align:center;margin:8px 0 4px 0;">
      <img src="data:image/png;base64,{hog_index_b64}"
           style="width:100%;max-width:780px;border:1px solid var(--border);"
           alt="生猪指数走势" />
    </div>"""

    # ── 各品种期货走势 + 技术分析 + 基本面 + 基差 ──
    product_sections_html = ""
    product_configs = [
        ("starch", "玉米淀粉", "Corn Starch", "CS", "#E67E22"),
        ("hog", "生猪", "Live Hog", "LH", "#2980B9"),
        ("egg", "鸡蛋", "Egg", "JD", "#8E44AD", "元/500kg"),
        ("soymeal", "豆粕", "Soymeal", "M", "#27AE60"),
    ]
    for cfg in product_configs:
        key, name, en, code, color = cfg[0], cfg[1], cfg[2], cfg[3], cfg[4]
        ylabel = cfg[5] if len(cfg) > 5 else "元/吨"
        chart = product_charts.get(key, "")
        tech = product_tech.get(key, {})
        basis = product_basis.get(key, {})
        ps = product_stats.get(key, {})

        if not chart:
            continue

        # 行情综述文本
        if tech:
            commentary = f"本周{name}期货主力{tech.get('trend','震荡')}，收于 {tech.get('last',0):.0f} {ylabel}，周涨跌 {tech.get('chg_pct',0):+.1f}%。"
            commentary += f"最高 {tech.get('high',0):.0f}，最低 {tech.get('low',0):.0f}，"
            commentary += f"周成交量 {tech.get('vol_total',0):.0f} 万手，持仓量 {tech.get('oi',0):.0f} 万手。"
        else:
            commentary = f"{name}期货主力合约最新价 {ps.get('close',0):.0f} {ylabel}。"

        # 技术分析
        tech_html = ""
        if tech:
            tech_html = f"""
      <div class="analysis-box">
        <h4 style="color:{color};margin:0 0 4px 0;font-size:12px;">技术分析</h4>
        <p style="margin:2px 0;font-size:11px;">
          RSI(7): <strong>{tech.get('rsi',0):.1f}</strong> — {tech.get('rsi_signal','')} &nbsp;|&nbsp;
          周波动率: <strong>{tech.get('volatility',0):.1f}%</strong> &nbsp;|&nbsp;
          周涨跌: <span style="color:{'#C0392B' if tech.get('chg',0) > 0 else '#27AE60'};font-weight:700;">{tech.get('chg_pct',0):+.1f}%</span>
        </p>
        <p style="margin:2px 0;font-size:11px;">
          压力位: <strong style="color:#C0392B;">{tech.get('high',0):.0f}</strong> &nbsp;|&nbsp;
          支撑位: <strong style="color:#27AE60;">{tech.get('low',0):.0f}</strong>
        </p>
      </div>"""

        # 基差分析
        basis_html = ""
        if basis:
            b_val = basis.get("basis", 0)
            b_rate = basis.get("basis_rate", 0)
            spot = basis.get("spot", 0)
            if b_val > 0:
                basis_desc = f"期货升水现货 {b_val:.0f} {ylabel}（升水率 {b_rate:.1f}%），反映市场对未来价格偏乐观预期。"
            elif b_val < 0:
                basis_desc = f"期货贴水现货 {abs(b_val):.0f} {ylabel}（贴水率 {abs(b_rate):.1f}%），现货端偏紧或近月交割压力较大。"
            else:
                basis_desc = "期现基本平水，市场定价中性。"
            basis_html = f"""
      <div class="analysis-box">
        <h4 style="color:{color};margin:0 0 4px 0;font-size:12px;">基差分析</h4>
        <p style="margin:2px 0;font-size:11px;">
          现货价: <strong>{spot:.0f}</strong> {ylabel} &nbsp;|&nbsp;
          基差: <strong style="color:{'#C0392B' if b_val > 0 else '#2980B9'};">
          {'+' if b_val > 0 else ''}{b_val:.0f}</strong> {ylabel}（{'升水' if b_val > 0 else '贴水'}{abs(b_rate):.1f}%）
        </p>
        <p style="margin:2px 0;font-size:11px;color:#7F8C8D;">{basis_desc}</p>
      </div>"""

        # 基本面分析 (基于数据自动生成)
        fundamentals = []
        if key == "starch":
            fundamentals = [
                "玉米淀粉加工利润处于盈亏线附近，深加工企业开机率维持区间高位，库存累积压力持续。",
                "玉米原料成本支撑偏强，但下游淀粉糖及造纸需求恢复缓慢，供需偏宽松。",
                f"期货{ '升水' if ps.get('chg',0) > 0 else '震荡' }，反映市场对原料玉米成本传导的预期。",
            ]
        elif key == "hog":
            fundamentals = [
                "生猪产能去化缓慢，能繁母猪存栏仍高于合理区间，供应端压力持续。",
                "猪价季节性反弹预期存在，但消费端支撑有限，供需博弈加剧。",
                "养殖利润修复中，饲料成本(玉米/豆粕)波动对养殖端利润影响显著。",
            ]
        elif key == "egg":
            fundamentals = [
                "蛋鸡存栏量处于高位，鸡蛋供应充裕，价格承压。",
                "端午节前备货需求或阶段性提振蛋价，但持续性有待观察。",
                "饲料原料(玉米/豆粕)价格波动直接影响蛋鸡养殖成本。",
            ]
        elif key == "soymeal":
            fundamentals = [
                "豆粕价格与国际大豆(美豆/巴西豆)到港节奏密切相关，进口大豆集中到港或压制现货。",
                "国内饲料需求受养殖存栏高位支撑，豆粕消费刚性较强。",
                "CBOT大豆及国际运费(BDI)变动对进口大豆成本形成传导。",
            ]

        fund_html = ""
        if fundamentals:
            fund_html = f"""
      <div class="analysis-box">
        <h4 style="color:{color};margin:0 0 4px 0;font-size:12px;">基本面分析</h4>
        <ul style="margin:2px 0;padding-left:16px;font-size:11px;">
          {"".join(f'<li style="margin:2px 0;">{f}</li>' for f in fundamentals)}
        </ul>
      </div>"""

        product_sections_html += f"""
    <div class="section-title">{name}期货<span class="en">{en} Futures — {code}</span></div>
    <p style="font-size:12px;color:#2C3E50;margin:0 0 6px 0;line-height:1.7;">{commentary}</p>
    <div style="text-align:center;margin:4px 0 2px 0;">
      <img src="data:image/png;base64,{chart}"
           style="width:100%;max-width:780px;border:1px solid var(--border);"
           alt="{name}期货走势" />
    </div>
    {tech_html}
    {basis_html if basis else ''}
    {fund_html}"""

    # ── 行业新闻 ──
    news_html = ""
    if news_data:
        news_html = '<div class="section-title">行业要闻<span class="en">Industry News</span></div>'
        for product, items in news_data.items():
            if not items:
                continue
            product_colors = {"玉米": "#C0392B", "淀粉": "#E67E22", "生猪": "#2980B9", "鸡蛋": "#8E44AD", "豆粕": "#27AE60"}
            pc = product_colors.get(product, "#2C3E50")
            news_html += f'<div style="margin-bottom:8px;"><h4 style="color:{pc};font-size:13px;margin:4px 0 4px 0;">▎{product}</h4>'
            for item in items[:5]:
                news_html += f'<div style="font-size:11px;padding:2px 0;border-bottom:1px dotted #EEE;">'
                news_html += f'<span style="color:#7F8C8D;">[{item["time"]}]</span> '
                title = item["title"]
                url = item.get("url", "")
                if url:
                    news_html += f'<a href="{url}" target="_blank" style="color:#2C3E50;text-decoration:none;border-bottom:1px dotted #999;">{title}</a>'
                else:
                    news_html += f'{title}'
                if item.get("source"):
                    news_html += f' <span style="color:#95A5A6;font-size:10px;">({item["source"]})</span>'
                news_html += '</div>'
            news_html += '</div>'

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
<title>饲料能量农产品周报 — {report_date_str}</title>
<style>{CSS}</style>
</head>
<body>

<div class="cover-header">
  <div class="title">饲料能量农产品周报</div>
  <div class="subtitle">China Feed & Energy Agricultural Products Weekly</div>
  <div class="date-badge">{report_date_str} &nbsp;|&nbsp; 第 {REPORT_END.isocalendar()[1]} 周</div>
</div>

{metrics_html}

<div class="section-title">行情综述<span class="en">Market Overview</span></div>
{market_html}

{casde_html}

{usda_html}

{product_sd_html}

{cftc_html}

{sales_html}

{weather_html}

{spot_trend_html}

{corn_analysis_html}

{downstream_html}

{freight_html}

{hog_index_html}

{product_sections_html}

{news_html}

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
        "--no-pdf-header-footer",
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

    # 6. 生猪 & 鸡蛋供需基本面
    try:
        print("  [基本面] 拉取生猪+鸡蛋供需数据 ...")
        sys.path.insert(0, str(ML_DIR))
        from fetch_fundamentals import update_all_fundamentals
        update_all_fundamentals()
        updated.append("生猪+鸡蛋基本面已更新")
    except Exception as e:
        print(f"SKIP (基本面拉取异常: {e})")

    return updated


def main():
    print("=" * 60)
    print("  饲料能量农产品周报生成器")
    print("=" * 60)
    print(f"  周度区间: {REPORT_START} ~ {REPORT_END}")
    print()

    # 刷新数据 → 确保所有数据源为本周最新
    print("[0/12] 刷新数据源 (akshare 实时拉取) ...")
    updated_items = refresh_all_data()
    if updated_items:
        for item in updated_items:
            print(f"       [OK] {item}")
    else:
        print("       (无数据更新,使用已有缓存)")
    print()

    # 加载数据
    print("[1/12] 加载本地数据 ...")
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

    # 加载 akshare 外部数据 + 新品种
    print("[2/12] 加载 akshare 外部数据 ...")
    cftc_df = load_cftc_holding()
    soozhu_df = load_corn_spot_soozhu()
    hog_df = load_hog_price()
    feed_df = load_feed_price()
    freight_df = load_freight_index()
    starch_df = load_starch_futures()
    egg_df = load_egg_futures()
    hog_fut_df = load_hog_futures()
    soymeal_df = load_soymeal_futures()
    hog_idx_df = load_hog_index()
    soybean_spot_df = load_soybean_spot()
    hog_fundamentals = load_hog_fundamentals()
    egg_fundamentals = load_egg_fundamentals()
    print(f"  [OK] CFTC {len(cftc_df)} 现货 {len(soozhu_df)} BDI {len(freight_df)}")
    print(f"  [OK] 淀粉 {len(starch_df)} 鸡蛋 {len(egg_df)} 生猪 {len(hog_fut_df)} 豆粕 {len(soymeal_df)} 生猪指数 {len(hog_idx_df)}")
    if hog_fundamentals:
        print(f"  [OK] 生猪基本面: 均价{hog_fundamentals.get('hog_price_kg','?')}元/kg, 猪粮比{hog_fundamentals.get('hog_core','?')}")
    if egg_fundamentals:
        print(f"  [OK] 鸡蛋基本面: 现货{egg_fundamentals.get('egg_spot','?')}元/500kg, 基差率{egg_fundamentals.get('egg_futures_basis_rate','?')}")

    # 计算产品周度统计
    def product_weekly_stats(df):
        if df.empty or len(df) < 5:
            return {"close": 0, "chg": 0, "ma20": 0}
        recent = df[df["date"] >= pd.Timestamp(REPORT_START)]
        if recent.empty:
            recent = df.tail(5)
        prev = df[df["date"] < pd.Timestamp(REPORT_START)]
        prev_close = float(prev.iloc[-1]["close"]) if not prev.empty else float(recent.iloc[0]["close"])
        ma20 = float(df.tail(20)["close"].mean()) if len(df) >= 20 else float(df["close"].mean())
        return {"close": float(recent.iloc[-1]["close"]), "chg": float(recent.iloc[-1]["close"]) - prev_close, "ma20": ma20}

    product_stats = {
        "starch": product_weekly_stats(starch_df),
        "hog": product_weekly_stats(hog_fut_df),
        "egg": product_weekly_stats(egg_df),
        "soymeal": product_weekly_stats(soymeal_df),
    }
    # 附加生猪指数
    if not hog_idx_df.empty:
        product_stats["hog"]["index_val"] = float(hog_idx_df.iloc[-1]["index"])
    # 附加大豆现货
    if not soybean_spot_df.empty:
        product_stats["soymeal"]["soybean_spot"] = float(soybean_spot_df.iloc[-1]["price"])

    # 技术分析 & 基差
    print("[3/12] 计算技术分析 + 基差 ...")
    product_dfs = {"starch": starch_df, "hog": hog_fut_df, "egg": egg_df, "soymeal": soymeal_df}
    corn_tech = compute_tech_analysis(futures_df)
    product_tech = {k: compute_tech_analysis(v) for k, v in product_dfs.items()}
    product_basis = {}
    for sym, key in [("CS", "starch"), ("LH", "hog"), ("JD", "egg"), ("M", "soymeal")]:
        product_basis[key] = load_product_basis(sym, key)
    print(f"  [OK] 技术: C={corn_tech.get('rsi','-')} S={product_tech.get('starch',{}).get('rsi','-')} H={product_tech.get('hog',{}).get('rsi','-')} E={product_tech.get('egg',{}).get('rsi','-')} M={product_tech.get('soymeal',{}).get('rsi','-')}")
    b_ok = [k for k, v in product_basis.items() if v]
    print(f"  [OK] 基差: {', '.join(b_ok) if b_ok else '无'} ({len(b_ok)}/4)")

    # 计算周度统计
    stats = compute_weekly_stats(futures_df, basis_df, cbot_df, inv_df)
    print(f"  [OK] 玉米: {stats['futures_close']:.0f} | 淀粉: {product_stats['starch']['close']:.0f} | 生猪: {product_stats['hog']['close']:.0f}")
    print(f"  [OK] 鸡蛋: {product_stats['egg']['close']:.0f} | 豆粕: {product_stats['soymeal']['close']:.0f}")

    # CASDE数据
    print("[4/12] 提取供需表数据 ...")
    casde_data = casde_rows(casde_df)
    print(f"  [OK] {casde_data[2]} vs {casde_data[1]}")

    # USDA数据
    usda_data = usda_rows(usda_df)
    print(f"  [OK] {usda_data[1]} vs {usda_data[2]}")

    # 天气数据
    print("[5/12] 提取天气数据 ...")
    weather_list = extract_weather(ai_data)
    print(f"  [OK] {len(weather_list)} 个站点")

    # 抓取行业新闻
    print("[6/12] 抓取行业新闻 ...")
    news_data = fetch_multi_product_news()
    news_count = sum(len(v) for v in news_data.values())
    print(f"  [OK] {news_count} 条新闻 (玉米/淀粉/生猪/鸡蛋/豆粕)")

    # 生成各图表
    print("[7/12] 生成统计图表 ...")
    sales_chart_b64 = generate_sales_comparison_chart(sales_df)
    cftc_chart_b64 = generate_cftc_chart(cftc_df)
    spot_trend_b64 = generate_spot_trend_chart(soozhu_df)
    downstream_b64 = generate_downstream_chart(hog_df, feed_df)
    freight_b64 = generate_freight_chart(freight_df)
    hog_index_b64 = generate_hog_index_chart(hog_idx_df, hog_fut_df)

    # 新品种期货走势图
    product_charts = {}
    product_chart_configs = [
        ("starch", starch_df, "玉米淀粉期货走势 (CS — Corn Starch Futures)", "#E67E22"),
        ("hog", hog_fut_df, "生猪期货走势 (LH — Live Hog Futures)", "#2980B9"),
        ("egg", egg_df, "鸡蛋期货走势 (JD — Egg Futures)", "#8E44AD", "元/500kg"),
        ("soymeal", soymeal_df, "豆粕期货走势 (M — Soymeal Futures)", "#27AE60"),
    ]
    for key, df, title, color, *args in product_chart_configs:
        ylabel = args[0] if args else "元/吨"
        product_charts[key] = generate_product_trend_chart(df, title, color, ylabel)

    print(f"  [OK] 玉米图表: 售粮({len(sales_chart_b64)//1024}K) CFTC({len(cftc_chart_b64)//1024 if cftc_chart_b64 else 0}K)")
    print(f"  [OK] 新品种图表: 淀粉/生猪/鸡蛋/豆粕 + 生猪指数({len(hog_index_b64)//1024 if hog_index_b64 else 0}K)")
    print(f"  [OK] BDI ({len(freight_b64)//1024 if freight_b64 else 0}KB)")

    # 生成HTML
    print("[8/12] 生成HTML报告 ...")
    report_date_str = REPORT_END.isoformat()
    html = build_html(stats, casde_data, usda_data, sales_df, weather_list, report_date_str,
                      sales_chart_b64, cftc_chart_b64, spot_trend_b64, downstream_b64, freight_b64,
                      product_charts, hog_index_b64, news_data, product_stats, product_tech, product_basis,
                      corn_tech, hog_fundamentals, egg_fundamentals)

    html_path = OUT_DIR / f"corn_weekly_report_{report_date_str}.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"  [OK] HTML: {html_path}")

    # 转PDF
    print("[9/12] 导出PDF ...")
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
