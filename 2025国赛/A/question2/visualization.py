"""问题2的PNG/SVG绘图。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt


def _save(fig: Any, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=130, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_trajectory(data: Any, plan: Any, derived: Any, output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(); start = data.uav_init[plan.uav_index]; end = derived.explosion_point
    ax.plot([start[0], end[0]], [start[1], end[1]], label="无人机航迹"); ax.scatter([end[0]], [end[1]], label="爆炸点"); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title("问题二：轨迹"); ax.legend()
    _save(fig, output_dir / "trajectory"); return {"png": output_dir / "trajectory.png", "svg": output_dir / "trajectory.svg"}


def plot_intervals(intervals: Any, output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir); fig, ax = plt.subplots()
    for missile, values in intervals.items():
        for start, end in values: ax.plot([start, end], [missile, missile], linewidth=6)
    ax.set_xlabel("时间 (秒)"); ax.set_ylabel("导弹编号"); ax.set_title("烟雾覆盖区间"); _save(fig, output_dir / "intervals"); return {"png": output_dir / "intervals.png", "svg": output_dir / "intervals.svg"}


def plot_optimizer_history(history: list[dict[str, Any]], output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir); fig, ax = plt.subplots()
    pso = [r for r in history if r.get("source", "pso") == "pso"]; de = [r for r in history if r.get("source") == "de"]
    if pso: ax.plot([r["evaluations"] for r in pso], [r["best"] for r in pso], label="PSO")
    if de: ax.plot([r["evaluations"] for r in de], [r["best"] for r in de], label="DE")
    ax.set_xlabel("评估次数"); ax.set_ylabel("最佳时长 (秒)"); ax.set_title("PSO与DE优化历程对比"); ax.legend(); _save(fig, output_dir / "optimization_history")
    with (output_dir / "optimization_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for row in history for k in row})); writer.writeheader(); writer.writerows(history)
    return {"png": output_dir / "optimization_history.png", "svg": output_dir / "optimization_history.svg", "csv": output_dir / "optimization_history.csv"}
