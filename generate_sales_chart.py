#!/usr/bin/env python3
"""生成主产区售粮进度表图片"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "corn_ml", "regional_sales_progress.csv")
OUT_DIR = os.path.join(BASE_DIR, "corn_weekly_report", "output")


def generate_sales_table():
    df = pd.read_csv(CSV_PATH)
    report_date = df["report_date"].iloc[0]

    detail_df = df[df["province"].isin(["黑龙江", "吉林", "辽宁", "内蒙古", "山东", "河北", "河南", "山西"])]

    northeast = detail_df[detail_df["region"] == "东北"]
    huabei = detail_df[detail_df["region"] == "华北"]

    def make_row(r):
        progress = int(r["sales_progress_pct"])
        price = f"{int(r['spot_price_low'])}-{int(r['spot_price_high'])}"
        bar = "█" * (progress // 5) + "░" * ((100 - progress) // 5)
        return [
            r["province"],
            f"{progress}%",
            bar,
            price,
            r.get("price_vs_month", "—"),
            r["remaining_grain"],
        ]

    headers = ["省份", "售粮进度", "进度条", "现货价格\n(元/吨)", "较上月", "余粮情况"]

    ne_rows = [make_row(r) for _, r in northeast.iterrows()]
    hb_rows = [make_row(r) for _, r in huabei.iterrows()]

    n_ne = len(ne_rows)
    n_hb = len(hb_rows)
    total_rows = 2 + n_ne + 1 + n_hb

    fig, ax = plt.subplots(figsize=(13, 5.5))
    fig.patch.set_facecolor("#F5F0E8")
    ax.set_facecolor("#F5F0E8")
    ax.axis("off")

    ax.text(0.5, 0.93, "中国玉米主产区售粮进度\n(China Corn Regional Sales Progress)",
            transform=ax.transAxes, fontsize=15, fontweight="bold",
            ha="center", va="center", color="#2C3E50")

    col_widths = [0.10, 0.12, 0.32, 0.18, 0.14, 0.14]

    table_data = []
    table_colors = []

    table_data.append(headers)
    table_colors.append(["#2C3E50"] * 6)

    for row in ne_rows:
        table_data.append(row)
        table_colors.append(["#FFFFFF"] * 6)

    section_sep = ["【东北产区】", "", "", "", "", ""]
    table_data.append(section_sep)
    table_colors.append(["#3498DB"] + ["#D6EAF8"] * 5)

    for row in hb_rows:
        table_data.append(row)
        table_colors.append(["#FFFFFF"] * 6)

    table = ax.table(
        cellText=table_data,
        cellLoc="center",
        loc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.55)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#BDC3C7")
        cell.set_linewidth(0.5)
        bg = table_colors[row][col]
        cell.set_facecolor(bg)

        if row == 0:
            cell.set_text_props(color="white", fontweight="bold", fontsize=9.5)
        elif table_data[row][0] == "【东北产区】" or table_data[row][0] == "【华北产区】":
            if col == 0:
                cell.set_text_props(color="white", fontweight="bold", fontsize=10)
            else:
                cell.set_text_props(color="#2C3E50", fontsize=8.5)
        else:
            if col == 0:
                cell.set_text_props(fontweight="bold", fontsize=9.5, color="#C0392B")

            elif col == 2:
                cell.set_text_props(fontsize=7, color="#7F8C8D", fontfamily="monospace")
            elif col == 1:
                cell.set_text_props(fontsize=10, fontweight="bold", color="#27AE60")
            else:
                cell.set_text_props(fontsize=9, color="#2C3E50")

    for key, cell in table.get_celld().items():
        cell.set_height(cell.get_height() * 0.85)

    source = f"数据来源: 饲料行业信息网 / 慧博投研 / 多地贸易商报价  |  报告日期: {report_date}"
    ax.text(0.5, 0.02, source, transform=ax.transAxes, fontsize=7,
            ha="center", va="center", color="#95A5A6", style="italic")

    points = (
        "【要点】"
        "东北基层余粮不足一成(售粮进度98%)，粮权已转移至贸易商，贸易商低价惜售挺价心态较强; "
        "华北售粮进度92%，贸易商为小麦腾库集中出货，山东深加工到货量环增15%，短期价格承压。"
    )

    note_height = 0.06 + 0.02 * (len(points) // 60)
    ax.text(0.5, 0.07, points, transform=ax.transAxes, fontsize=7.5,
            ha="center", va="center", color="#E74C3C", style="italic",
            wrap=True)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    out = os.path.join(OUT_DIR, "regional_corn_sales_progress.png")
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[OK] 售粮进度表 -> {out}")
    return out


def generate_sales_with_chart():
    """带可视化进度条的版本"""
    df = pd.read_csv(CSV_PATH)
    report_date = df["report_date"].iloc[0]

    detail_df = df[df["province"].isin(["黑龙江", "吉林", "辽宁", "内蒙古", "山东", "河北", "河南", "山西", "东北整体", "华北整体"])]

    ne_provinces = ["黑龙江", "吉林", "辽宁", "内蒙古", "东北整体"]
    hb_provinces = ["山东", "河北", "河南", "山西", "华北整体"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    fig.patch.set_facecolor("#F5F0E8")

    colors = ["#27AE60", "#2980B9", "#8E44AD", "#E67E22", "#C0392B"]

    for idx_ax, (ax, prov_list, region_name, overall_idx) in enumerate([
        (axes[0], ne_provinces, "东北产区", 4),
        (axes[1], hb_provinces, "华北黄淮产区", 4),
    ]):
        ax.set_facecolor("#FEFEFE")
        values = []
        labels = []
        for prov in prov_list:
            row = detail_df[detail_df["province"] == prov]
            if len(row) > 0:
                pct = float(row["sales_progress_pct"].iloc[0])
                values.append(pct)
                labels.append(prov)

        bars = ax.barh(labels, values, color=colors[:len(labels)], height=0.55, edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:.0f}%", va="center", fontsize=11, fontweight="bold", color="#2C3E50")

        ax.set_xlim(0, 105)
        ax.set_title(region_name, fontsize=13, fontweight="bold", color="#2C3E50", pad=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#BDC3C7")
        ax.spines["bottom"].set_color("#BDC3C7")
        ax.tick_params(labelsize=10, colors="#2C3E50")
        ax.grid(axis="x", linestyle="--", alpha=0.3, color="#BDC3C7")

    fig.suptitle("中国玉米主产区售粮进度\n(China Corn Regional Sales Progress)",
                 fontsize=16, fontweight="bold", color="#2C3E50", y=0.99)
    fig.text(0.5, 0.01, f"数据来源: 饲料行业信息网 / 慧博投研 / 多地贸易商报价  |  报告日期: {report_date}",
             ha="center", fontsize=7.5, color="#95A5A6", style="italic")

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])

    out = os.path.join(OUT_DIR, "regional_corn_sales_chart.png")
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[OK] 售粮进度图 -> {out}")
    return out


if __name__ == "__main__":
    generate_sales_table()
    generate_sales_with_chart()
    print("\n[DONE] 主产区售粮进度图表已生成!")
