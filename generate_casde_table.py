#!/usr/bin/env python3
"""生成CASDE中国玉米供需平衡表图片"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(DATA_DIR, "corn_ml", "casde_corn_supply_demand.csv")
OUT_DIR = os.path.join(DATA_DIR, "corn_weekly_report", "output")


def draw_table(ax, headers, data_rows, col_widths, header_color="#8B0000"):
    table = ax.table(
        cellText=data_rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.55)

    row_colors = ("#FFFFFF", "#FFF5F5")
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D5D8DC")
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", fontweight="bold", fontsize=9)
        else:
            bg = row_colors[(row - 1) % len(row_colors)]
            cell.set_facecolor(bg)
            if col == 0:
                cell.set_text_props(fontweight="bold", fontsize=8.5)
            else:
                cell.set_text_props(fontsize=8.5, color="#2C3E50")
    for key, cell in table.get_celld().items():
        cell.set_height(cell.get_height() * 0.85)


def generate_casde_overview():
    """生成CASDE中国玉米供需总览表 — 最近两年对比"""
    df = pd.read_csv(CSV_PATH)
    df = df.sort_values("report_date")
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    cy = latest["market_year"]
    py = prev["market_year"]

    items = [
        ("播种面积 (千公顷)", "corn_area_kha", 0),
        ("产量 (百万吨)", "corn_production_mmt", 1),
        ("单产 (公斤/公顷)", "corn_yield_kg_ha", 0),
        ("饲用消费 (百万吨)", "corn_feed_consumption_mmt", 1),
        ("工业消费 (百万吨)", "corn_industrial_consumption_mmt", 1),
        ("总消费 (百万吨)", "corn_total_consumption_mmt", 1),
        ("进口量 (百万吨)", "corn_imports_mmt", 1),
    ]

    headers = ["指标", py, cy, "同比变化"]
    data_rows = []
    for label, col, decimals in items:
        pv = prev[col]
        cv = latest[col]
        chg = cv - pv
        if abs(chg) < 0.005:
            chg_str = "—"
        elif chg > 0:
            chg_str = f"+{cv - pv:+.{decimals}f}"
        else:
            chg_str = f"{cv - pv:.{decimals}f}"
        data_rows.append([label, f"{pv:.{decimals}f}", f"{cv:.{decimals}f}", chg_str])

    if pd.notna(latest.get("notes")) and latest["notes"]:
        note = latest["notes"]
    else:
        note = ""

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor("#F5F0E8")
    ax.set_facecolor("#F5F0E8")
    ax.axis("off")

    ax.text(0.5, 0.93, "CASDE 中国玉米供需平衡表\n(China Corn Supply & Demand — 农业农村部)",
            transform=ax.transAxes, fontsize=15, fontweight="bold",
            ha="center", va="center", color="#2C3E50")

    draw_table(ax, headers, data_rows,
               col_widths=[0.38, 0.20, 0.20, 0.22],
               header_color="#C0392B")

    source = f"数据来源: 中国农业农村部 CASDE  |  报告日期: {latest['report_date']}  |  市场年度: {cy}"
    ax.text(0.5, 0.03, source, transform=ax.transAxes, fontsize=7,
            ha="center", va="center", color="#7F8C8D", style="italic")
    if note:
        ax.text(0.5, 0.08, f"备注: {note}", transform=ax.transAxes, fontsize=7.5,
                ha="center", va="center", color="#E74C3C", style="italic")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    out = os.path.join(OUT_DIR, "casde_china_corn_supply_demand.png")
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[OK] CASDE总览表 -> {out}")
    return out


def generate_casde_history():
    """生成CASDE中国玉米供需历史走势曲线图"""
    df = pd.read_csv(CSV_PATH)
    df = df.sort_values("report_date")

    years = [s.replace("/", "/\n") for s in df["market_year"]]
    x = range(len(df))

    panels = [
        ("播种面积", "千公顷", "corn_area_kha", "#2980B9"),
        ("产量 vs 总消费", "百万吨", "corn_production_mmt", "#27AE60"),
        ("单产", "公斤/公顷", "corn_yield_kg_ha", "#8E44AD"),
        ("饲用消费 vs 工业消费", "百万吨", "corn_feed_consumption_mmt", "#E67E22"),
        ("进口量", "百万吨", "corn_imports_mmt", "#C0392B"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(16, 13))
    fig.patch.set_facecolor("#F8F6F0")
    axes = axes.flatten()

    colors_panel = ["#2980B9", "#27AE60", "#8E44AD", "#E67E22", "#C0392B"]
    markers = ["o", "s", "D", "^", "v"]

    for idx, (title, unit, col, color) in enumerate(panels):
        ax = axes[idx]
        ax.set_facecolor("#FEFEFE")

        vals = df[col].values

        ax.plot(x, vals, color=color, linewidth=2.5, marker=markers[idx],
                markersize=8, markerfacecolor="white", markeredgewidth=2,
                markeredgecolor=color, zorder=5)

        for xi, vi in zip(x, vals):
            ax.annotate(f"{vi:.0f}" if vi < 1000 else f"{vi:.1f}",
                        (xi, vi), textcoords="offset points",
                        xytext=(0, 12), ha="center", fontsize=7.5,
                        color=color, fontweight="bold")

        if idx == 1:
            vals2 = df["corn_total_consumption_mmt"].values
            ax.plot(x, vals2, color="#E74C3C", linewidth=2.5, marker="s",
                    markersize=8, markerfacecolor="white", markeredgewidth=2,
                    markeredgecolor="#E74C3C", zorder=5)
            for xi, vi in zip(x, vals2):
                ax.annotate(f"{vi:.1f}", (xi, vi), textcoords="offset points",
                            xytext=(0, -16), ha="center", fontsize=7.5,
                            color="#E74C3C", fontweight="bold")
            ax.legend(["产量", "总消费"], loc="upper left", fontsize=8.5,
                      framealpha=0.9, edgecolor="#BDC3C7")
            ax.set_title("产量 vs 总消费", fontsize=13, fontweight="bold", color="#2C3E50", pad=10)
        elif idx == 3:
            vals2 = df["corn_industrial_consumption_mmt"].values
            ax.plot(x, vals2, color="#16A085", linewidth=2.5, marker="^",
                    markersize=8, markerfacecolor="white", markeredgewidth=2,
                    markeredgecolor="#16A085", zorder=5)
            for xi, vi in zip(x, vals2):
                ax.annotate(f"{vi:.1f}", (xi, vi), textcoords="offset points",
                            xytext=(0, -16), ha="center", fontsize=7.5,
                            color="#16A085", fontweight="bold")
            ax.legend(["饲用消费", "工业消费"], loc="upper left", fontsize=8.5,
                      framealpha=0.9, edgecolor="#BDC3C7")
            ax.set_title("饲用消费 vs 工业消费", fontsize=13, fontweight="bold", color="#2C3E50", pad=10)
        else:
            ax.set_title(title, fontsize=13, fontweight="bold", color="#2C3E50", pad=10)

        ax.set_xticks(x)
        ax.set_xticklabels(years, fontsize=8)
        ax.set_ylabel(unit, fontsize=9, color="#7F8C8D")
        ax.grid(True, linestyle="--", alpha=0.3, color="#BDC3C7")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#BDC3C7")
        ax.spines["bottom"].set_color("#BDC3C7")
        ax.tick_params(colors="#7F8C8D", labelsize=8)

    axes[5].axis("off")

    fig.suptitle("CASDE 中国玉米供需 — 历史走势\n(China Corn Supply & Demand Trend 2013/14–2026/27)",
                 fontsize=17, fontweight="bold", color="#2C3E50", y=0.99)

    fig.text(0.5, 0.01, "数据来源: 中国农业农村部 CASDE (历年5月报告)  |  单位: 产量/消费/进口为百万吨, 面积千公顷, 单产公斤/公顷",
             ha="center", fontsize=7.5, color="#95A5A6", style="italic")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out = os.path.join(OUT_DIR, "casde_china_corn_history.png")
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[OK] CASDE历史走势 -> {out}")
    return out


if __name__ == "__main__":
    generate_casde_overview()
    generate_casde_history()
    print("\n[DONE] CASDE中国玉米供需表已生成!")
