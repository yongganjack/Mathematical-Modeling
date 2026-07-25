# ============================================================
# 双轴折线图绘图模板 — Dual-Axis Line Chart Template
# ============================================================
# 使用步骤：
#   1. 准备你的数据（numpy 数组或 Python list）
#   2. 修改下方 EXAMPLE 区域的 config 配置字典
#   3. 运行脚本即可出图
#
# 模板支持：
#   - 左/右双 Y 轴，每条轴可绘制多条曲线（实线/虚线/点线自由组合）
#   - 渐变背景色带（标注不同阶段）
#   - 竖直阶段分界线
#   - 箭头 + 文字标注
#   - 左轴科学计数法 / 右轴整数格式化
#   - 中文 / 英文混排，LaTeX 数学公式
#   - 同时导出 PNG + SVG
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FuncFormatter
from matplotlib.colors import to_rgba

# ============================================================
# 全局字体与样式预设
# ============================================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["SimHei", "Microsoft YaHei", "DejaVu Sans"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# ============================================================
# 预设配色方案（可按需增删）
# ============================================================
PALETTE = {
    "blue":    "#0B71C8",
    "orange":  "#E34B0A",
    "peach":   "#F4C7A1",
    "green":   "#A9D18E",
    "black":   "#111111",
    "red":     "#D73027",
    "purple":  "#7B2D8E",
    "teal":    "#008080",
}

# 线型映射
LINE_STYLES = {
    "solid":   (0, ()),          # ━━━━
    "dashed":  (0, (5, 4)),      # ━ ━ ━
    "dotted":  (0, (1, 3)),      # · · ·
    "dashdot": (0, (5, 4, 1, 4)),# ━ · ━ ·
}


# ============================================================
# 工具函数
# ============================================================
def _draw_gradient_band(axis, x0, x1, y_bottom, y_top,
                        color, alpha_start, alpha_end, zorder=0):
    """在指定区域绘制从 alpha_start 渐变到 alpha_end 的垂直色带。"""
    rgba = np.array(to_rgba(color))
    n = 512
    img = np.ones((2, n, 4), dtype=float)
    img[..., :3] = rgba[:3]
    img[..., 3] = np.linspace(alpha_start, alpha_end, n)[None, :]
    axis.imshow(
        img,
        extent=(x0, x1, y_bottom, y_top),
        origin="lower",
        aspect="auto",
        interpolation="bicubic",
        zorder=zorder,
    )


def _draw_annotations(ax, annotations):
    """批量绘制箭头 + 文字标注。"""
    for ann in annotations:
        ax.text(
            ann["text_xy"][0], ann["text_xy"][1],
            ann["text"],
            fontsize=ann.get("fontsize", 14),
            fontweight=ann.get("fontweight", "bold"),
            color=ann.get("color", "#111111"),
        )
        ax.annotate(
            "",
            xy=ann["arrow_to"],
            xytext=ann["arrow_from"],
            arrowprops=dict(
                arrowstyle="-|>",
                color=ann.get("arrow_color", "#111111"),
                lw=ann.get("arrow_lw", 5.0),
                mutation_scale=ann.get("arrow_scale", 18),
                shrinkA=0, shrinkB=0,
                capstyle="butt", joinstyle="miter",
            ),
            zorder=6,
        )


# ============================================================
# 核心绘图函数
# ============================================================
def create_dual_axis_chart(config):
    """
    根据配置字典绘制双轴折线图。

    参数
    ----
    config : dict，包含以下顶层键（标 * 为必填）：

        figure : dict           — figsize, dpi
        title  : dict           — text, fontsize, pad
        x_axis : dict           — *data, label, limits, ticks
        y_left : dict           — label, color, limits, ticks, scientific_notation
        y_right: dict           — label, color, limits, ticks
        lines_left  : list[dict]— 左轴曲线 [{y, label, color, style, lw}, ...]
        lines_right : list[dict]— 右轴曲线 [{y, label, color, style, lw}, ...]
        phase_lines : list[float]  （可选）竖虚线 x 坐标
        bands       : list[dict]   （可选）渐变背景 [{x0, x1, color, alpha_start, alpha_end}]
        annotations : list[dict]   （可选）标注
        legend      : dict         （可选）图例参数
        output      : dict         — png, svg, dpi, show
        layout      : dict         — left, right, bottom, top 边距

    返回
    ----
    fig, ax, ax2  — 可继续手动修改
    """
    # ---- 解包配置（带默认值）----
    fig_cfg     = config.get("figure", {})
    title_cfg   = config.get("title", {})
    x_cfg       = config["x_axis"]
    yl_cfg      = config.get("y_left", {})
    yr_cfg      = config.get("y_right", {})
    lines_l     = config.get("lines_left", [])
    lines_r     = config.get("lines_right", [])
    bands       = config.get("bands", [])
    phase_lines = config.get("phase_lines", [])
    annotations = config.get("annotations", [])
    legend_cfg  = config.get("legend", {})
    output_cfg  = config.get("output", {})
    layout_cfg  = config.get("layout", {})

    x_data = x_cfg["data"]

    # ---- 创建画布 ----
    fig, ax = plt.subplots(
        figsize=fig_cfg.get("figsize", (12.8, 6.12)),
        dpi=fig_cfg.get("dpi", 100),
    )
    ax2 = ax.twinx()

    yl_color = yl_cfg.get("color", "#0B71C8")
    yr_color = yr_cfg.get("color", "#E34B0A")

    # ---- 渐变背景色带 ----
    yl_lim = yl_cfg.get("limits", None)
    if yl_lim is None and (bands or phase_lines):
        # 如果未指定范围，尝试从数据推算
        pass

    for band in bands:
        _draw_gradient_band(
            ax,
            x0=band["x0"], x1=band["x1"],
            y_bottom=band.get("y_bottom", yl_lim[0]) if yl_lim else 0,
            y_top=band.get("y_top", yl_lim[1]) if yl_lim else 1,
            color=band["color"],
            alpha_start=band["alpha_start"],
            alpha_end=band["alpha_end"],
            zorder=band.get("zorder", 0),
        )

    # ---- 绘制曲线 ----
    # 左轴曲线
    left_handles = []
    for line in lines_l:
        ls = LINE_STYLES.get(line.get("style", "solid"), (0, ()))
        h, = ax.plot(
            x_data, line["y"],
            color=line.get("color", yl_color),
            lw=line.get("lw", 1.8),
            ls=ls,
            label=line.get("label", ""),
            zorder=line.get("zorder", 4),
        )
        left_handles.append(h)

    # 右轴曲线
    right_handles = []
    for line in lines_r:
        ls = LINE_STYLES.get(line.get("style", "solid"), (0, ()))
        h, = ax2.plot(
            x_data, line["y"],
            color=line.get("color", yr_color),
            lw=line.get("lw", 1.8),
            ls=ls,
            label=line.get("label", ""),
            zorder=line.get("zorder", 4),
        )
        right_handles.append(h)

    # ---- 阶段分界线 ----
    for xpos in phase_lines:
        ax.axvline(
            xpos,
            color=config.get("phase_line_color", "#111111"),
            lw=config.get("phase_line_lw", 1.6),
            ls=LINE_STYLES.get(config.get("phase_line_style", "dashed"), (0, (5, 4))),
            zorder=3,
        )

    # ---- 坐标轴范围、刻度 ----
    if "limits" in x_cfg:
        ax.set_xlim(*x_cfg["limits"])
    if yl_lim:
        ax.set_ylim(*yl_lim)
    yr_lim = yr_cfg.get("limits", None)
    if yr_lim:
        ax2.set_ylim(*yr_lim)

    if "ticks" in x_cfg:
        ax.set_xticks(x_cfg["ticks"])
    if "ticks" in yl_cfg:
        ax.set_yticks(yl_cfg["ticks"])
    if "ticks" in yr_cfg:
        ax2.set_yticks(yr_cfg["ticks"])

    # ---- 轴标签 ----
    ax.set_xlabel(
        x_cfg.get("label", ""),
        fontsize=x_cfg.get("label_fontsize", 17),
    )
    ax.set_ylabel(
        yl_cfg.get("label", ""),
        fontsize=yl_cfg.get("label_fontsize", 17),
        color=yl_color,
    )
    ax2.set_ylabel(
        yr_cfg.get("label", ""),
        fontsize=yr_cfg.get("label_fontsize", 17),
        color=yr_color,
    )

    # ---- 左轴科学计数法 ----
    if yl_cfg.get("scientific_notation", False):
        sf = ScalarFormatter(useMathText=True)
        sf.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(sf)
        ax.yaxis.get_offset_text().set_color(yl_color)
        ax.yaxis.get_offset_text().set_fontsize(13)

    # ---- 右轴格式化（去尾随零）----
    if yr_cfg.get("strip_trailing_zero", True):
        ax2.yaxis.set_major_formatter(
            FuncFormatter(lambda value, pos: f"{value:g}")
        )

    # ---- 刻度样式 ----
    ax.tick_params(
        axis="x", direction="in", length=7, width=1.5,
        labelsize=13, pad=7, colors="#111111",
    )
    ax.tick_params(
        axis="y", direction="in", length=7, width=1.6,
        labelsize=13, pad=8, colors=yl_color,
    )
    ax2.tick_params(
        axis="y", direction="in", length=7, width=1.6,
        labelsize=13, pad=8, colors=yr_color,
    )

    # ---- 坐标轴脊柱 ----
    ax.spines["left"].set_color(yl_color)
    ax.spines["left"].set_linewidth(1.8)
    ax.spines["bottom"].set_color("#111111")
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax2.spines["right"].set_color(yr_color)
    ax2.spines["right"].set_linewidth(1.8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)

    # ---- 标题 ----
    if title_cfg:
        ax.set_title(
            title_cfg.get("text", ""),
            fontsize=title_cfg.get("fontsize", 18),
            pad=title_cfg.get("pad", 10),
        )

    # ---- 标注 ----
    _draw_annotations(ax, annotations)

    # ---- 图例 ----
    all_handles = left_handles + right_handles
    if all_handles:
        ax.legend(
            handles=all_handles,
            loc=legend_cfg.get("loc", "lower right"),
            bbox_to_anchor=legend_cfg.get("bbox_to_anchor", (1.005, -0.005)),
            fontsize=legend_cfg.get("fontsize", 10.5),
            frameon=legend_cfg.get("frameon", True),
            fancybox=legend_cfg.get("fancybox", False),
            framealpha=legend_cfg.get("framealpha", 0.94),
            facecolor=legend_cfg.get("facecolor", "white"),
            edgecolor=legend_cfg.get("edgecolor", "#111111"),
            borderpad=legend_cfg.get("borderpad", 0.25),
            labelspacing=legend_cfg.get("labelspacing", 0.25),
            handlelength=legend_cfg.get("handlelength", 2.9),
            handletextpad=legend_cfg.get("handletextpad", 0.45),
        ).get_frame().set_linewidth(legend_cfg.get("frame_lw", 1.3))

    # ---- 布局 ----
    fig.subplots_adjust(
        left=layout_cfg.get("left", 0.105),
        right=layout_cfg.get("right", 0.905),
        bottom=layout_cfg.get("bottom", 0.14),
        top=layout_cfg.get("top", 0.89),
    )

    # ---- 导出 ----
    png_path = output_cfg.get("png", "chart.png")
    svg_path = output_cfg.get("svg", "chart.svg")
    save_dpi = output_cfg.get("dpi", 200)

    fig.savefig(png_path, dpi=save_dpi, facecolor="white")
    if svg_path:
        fig.savefig(svg_path, facecolor="white")

    if output_cfg.get("show", True):
        plt.show()

    print(f"PNG  已保存至: {png_path}")
    if svg_path:
        print(f"SVG  已保存至: {svg_path}")

    return fig, ax, ax2


# ============================================================
# ============================================================
# 使用示例：原"多群体演化模型"复现
# ============================================================
# ============================================================
if __name__ == "__main__":

    # ---------- 1) 生成合成数据（替换成你自己的数据即可） ----------
    rng = np.random.default_rng(20250308)
    x = np.arange(0, 601, 1, dtype=float)

    def correlated_noise(n, scale=1.0, window=9):
        """低通滤波随机噪声，模拟生态波动。"""
        z = rng.normal(0.0, 1.0, n)
        kernel = np.ones(window, dtype=float) / window
        return np.convolve(z, kernel, mode="same") * scale

    # 生物量 b1（考虑操作性别比）
    b1 = np.empty_like(x)
    m1 = x < 100
    b1[m1] = (4.17 + 0.24 * np.exp(-((x[m1] - 31) / 26) ** 2)
              + 0.055 * np.sin(x[m1] / 15)
              + correlated_noise(m1.sum(), scale=0.055, window=5))
    m2 = (x >= 100) & (x < 300)
    b1[m2] = (2.10 + (3.95 - 2.10) * np.exp(-(x[m2] - 100) / 17.0)
              + correlated_noise(m2.sum(), scale=0.035, window=7))
    m3 = x >= 300
    b1[m3] = (2.08 + (4.16 - 2.08) * (1 - np.exp(-(x[m3] - 300) / 31.0))
              + 0.065 * np.sin((x[m3] - 300) / 17)
              + 0.035 * np.sin((x[m3] - 300) / 7.5)
              + correlated_noise(m3.sum(), scale=0.065, window=5))

    # 生物量 b2（不考虑操作性别比）
    b2 = np.empty_like(x)
    m1 = x < 100
    b2[m1] = (4.18 + 0.025 * np.sin(x[m1] / 12) - 0.00015 * x[m1]
              + correlated_noise(m1.sum(), scale=0.045, window=5))
    m2 = (x >= 100) & (x < 300)
    b2[m2] = (2.07 + (3.92 - 2.07) * np.exp(-(x[m2] - 100) / 23.0)
              + correlated_noise(m2.sum(), scale=0.035, window=7))
    m3 = x >= 300
    b2[m3] = (2.06 + (4.08 - 2.06) * (1 - np.exp(-(x[m3] - 300) / 41.0))
              + 0.035 * np.sin((x[m3] - 300) / 18)
              + correlated_noise(m3.sum(), scale=0.055, window=5))

    # 多样性指数 d1（考虑操作性别比）
    d1 = (1.18 + 0.26 * (1 - np.exp(-x / 18.0))
          + 0.008 * np.sin(x / 11) + 0.010 * np.sin(x / 38)
          + correlated_noise(x.size, scale=0.010, window=7))
    d1 += -0.035 * np.exp(-((x - 300) / 6.5) ** 2)
    d1 +=  0.026 * np.exp(-((x - 330) / 22.0) ** 2)

    # 多样性指数 d2（不考虑操作性别比）
    d2 = (1.17 + 0.235 * (1 - np.exp(-x / 20.0))
          + 0.006 * np.sin(x / 12) + 0.009 * np.sin(x / 42)
          + correlated_noise(x.size, scale=0.009, window=7))
    d2 += -0.018 * np.exp(-((x - 300) / 8.0) ** 2)

    # 缩放生物量以显示 ×10^5
    b1 *= 1e5
    b2 *= 1e5

    # ---------- 2) 填写配置 ----------
    config = {
        "figure": {"figsize": (12.8, 6.12), "dpi": 100},

        "title": {"text": "多群体演化模型", "fontsize": 18, "pad": 10},

        # X 轴
        "x_axis": {
            "data":   x,
            "label":  "时间 (/年)",
            "limits": (0, 600),
            "ticks":  np.arange(0, 601, 100),
        },

        # 左 Y 轴（生物量）
        "y_left": {
            "label":                "生物量 (/kg)",
            "color":                PALETTE["blue"],
            "limits":               (2.0e5, 4.5e5),
            "ticks":                np.arange(2.0e5, 4.51e5, 0.5e5),
            "scientific_notation":  True,   # 显示 ×10^5
        },

        # 右 Y 轴（多样性指数）
        "y_right": {
            "label":   "香农-维纳多样性指数",
            "color":   PALETTE["orange"],
            "limits":  (0.0, 1.8),
            "ticks":   np.arange(0.0, 1.81, 0.2),
        },

        # 左轴曲线
        "lines_left": [
            {"y": b1, "label": "B (考虑操作性别比)",   "style": "solid",  "color": PALETTE["blue"]},
            {"y": b2, "label": "B (不考虑操作性别比)", "style": "dashed", "color": PALETTE["blue"]},
        ],

        # 右轴曲线
        "lines_right": [
            {"y": d1, "label": "D (考虑操作性别比)",   "style": "solid",  "color": PALETTE["orange"]},
            {"y": d2, "label": "D (不考虑操作性别比)", "style": "dashed", "color": PALETTE["orange"]},
        ],

        # 阶段分界线
        "phase_lines": [100, 300],

        # 渐变背景色带
        "bands": [
            {"x0": 100, "x1": 300, "color": PALETTE["peach"],
             "alpha_start": 0.26, "alpha_end": 0.05},
            {"x0": 300, "x1": 375, "color": PALETTE["green"],
             "alpha_start": 0.34, "alpha_end": 0.00},
        ],

        # 箭头 + 文字标注
        "annotations": [
            {
                "text": r"$K$ 下降50%",
                "text_xy": (109, 4.36e5),
                "arrow_from": (157, 4.28e5),
                "arrow_to":   (108, 4.06e5),
                "arrow_color": "#DE8C4F",
            },
            {
                "text": r"$K$恢复",
                "text_xy": (171, 2.47e5),
                "arrow_from": (248, 2.49e5),
                "arrow_to":   (289, 2.27e5),
                "arrow_color": "#00A651",
            },
        ],

        # 图例
        "legend": {
            "loc": "lower right",
            "bbox_to_anchor": (1.005, -0.005),
            "fontsize": 10.5,
        },

        # 输出
        "output": {
            "png": "reproduced_multiple_population_model.png",
            "svg": "reproduced_multiple_population_model.svg",
            "dpi": 200,
            "show": True,
        },
    }

    # ---------- 3) 一行出图 ----------
    create_dual_axis_chart(config)
