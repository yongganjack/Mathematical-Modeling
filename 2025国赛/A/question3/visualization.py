"""English-language diagnostic plots for Question 3."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from question1.data_processing import ProblemData
from question1.model import DerivedBomb, UAVPlan, direction_from_heading


def _save(fig: Any, output_dir: str | Path, stem: str) -> dict[str, Path]:
    directory = Path(output_dir); directory.mkdir(parents=True, exist_ok=True)
    png, svg = directory / f"{stem}.png", directory / f"{stem}.svg"
    fig.savefig(png, dpi=180, bbox_inches="tight"); fig.savefig(svg, bbox_inches="tight")
    return {"png": png, "svg": svg}


def plot_trajectory(data: ProblemData, plan: UAVPlan, derived: Sequence[DerivedBomb], output_dir: str | Path) -> dict[str, Path]:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5.2))
    direction = direction_from_heading(plan.heading_rad)
    end_time = max(item.explosion_time for item in derived) + 3.0
    times = np.linspace(0.0, end_time, 150)
    origin = np.asarray(data.uav_init[plan.uav_index])
    path = origin[None, :] + plan.speed * times[:, None] * direction[None, :]
    ax.plot(path[:, 0], path[:, 1], color="navy", label="FY1 flight path")
    colors = ("tab:red", "tab:green", "tab:orange")
    for item, color in zip(derived, colors):
        ax.scatter(*item.release_point[:2], marker="v", color=color, label=f"Bomb {item.bomb_index} release")
        ax.scatter(*item.explosion_point[:2], marker="*", s=100, color=color, label=f"Bomb {item.bomb_index} explosion")
    ax.scatter(0.0, 200.0, marker="s", color="black", label="Target")
    ax.set(title="Question 3: FY1 Three-Bomb Trajectory", xlabel="x (m)", ylabel="y (m)")
    ax.axis("equal"); ax.grid(alpha=.25); ax.legend(fontsize=8, ncol=2)
    paths = _save(fig, output_dir, "trajectory"); plt.close(fig); return paths


def plot_intervals(intervals: Mapping[int, Sequence[Sequence[float]]], output_dir: str | Path) -> dict[str, Path]:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 2.8))
    for start, end in intervals.get(0, ()):
        ax.broken_barh([(start, end - start)], (0.6, 0.8), facecolors="tab:blue")
    ax.set(title="M1 Joint Coverage Intervals", xlabel="Time (s)", yticks=[1.0], yticklabels=["M1"]); ax.grid(axis="x", alpha=.25)
    paths = _save(fig, output_dir, "intervals"); plt.close(fig); return paths


def plot_contributions(contributions: Mapping[str, Any], output_dir: str | Path) -> dict[str, Path]:
    import matplotlib.pyplot as plt
    labels = ["Bomb 1", "Bomb 2", "Bomb 3"]
    x = np.arange(3); width = .25
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for offset, key, label in [(-width, "standalone", "Standalone"), (0, "sequential_marginal", "Sequential marginal"), (width, "removal_marginal", "Removal marginal")]:
        ax.bar(x + offset, [contributions[key][index] for index in (1, 2, 3)], width, label=label)
    ax.set(title="Bomb Coverage Contributions", ylabel="Duration (s)", xticks=x, xticklabels=labels); ax.legend(); ax.grid(axis="y", alpha=.25)
    paths = _save(fig, output_dir, "coverage_contributions"); plt.close(fig); return paths


def plot_optimizer_history(history: Sequence[Mapping[str, Any]], output_dir: str | Path) -> dict[str, Path]:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for source in ("pso", "de"):
        rows = [row for row in history if row.get("source") == source]
        if rows: ax.plot([row["evaluations"] for row in rows], [row["best"] for row in rows], marker="o", label=source.upper())
    ax.set(title="Optimizer History (Fast Profile)", xlabel="Objective evaluations", ylabel="Best joint duration (s)"); ax.grid(alpha=.25); ax.legend()
    paths = _save(fig, output_dir, "optimizer_history"); plt.close(fig); return paths
