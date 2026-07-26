"""English diagnostic figures for the Question 5 solver."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from question1.data_processing import ProblemData
from question1.model import UAVPlan, derive_bomb, missile_hit_time, missile_position, uav_position


def _save(figure: plt.Figure, output_dir: Path, stem: str) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {suffix: output_dir / f"{stem}.{suffix}" for suffix in ("png", "svg")}
    for suffix, path in paths.items(): figure.savefig(path, dpi=180 if suffix == "png" else None, bbox_inches="tight")
    plt.close(figure)
    return paths


def plot_trajectories(data: ProblemData, plans: Sequence[UAVPlan], output_dir: Path) -> dict[str, Path]:
    figure = plt.figure(figsize=(10, 7)); axis = figure.add_subplot(111, projection="3d")
    for missile in range(3):
        times = np.linspace(0.0, missile_hit_time(missile, data), 120)
        points = np.asarray([missile_position(float(time), missile, data) for time in times])
        axis.plot(points[:, 0], points[:, 1], points[:, 2], label=f"M{missile + 1}", linewidth=2)
    for plan in plans:
        horizon = max((bomb.release_time + bomb.fuse_delay for bomb in plan.bombs), default=5.0)
        times = np.linspace(0.0, max(5.0, horizon), 80)
        points = np.asarray([uav_position(float(time), plan, data) for time in times])
        axis.plot(points[:, 0], points[:, 1], points[:, 2], linestyle="--", label=f"FY{plan.uav_index + 1}")
        for bomb in plan.bombs:
            derived = derive_bomb(plan, bomb, data)
            axis.scatter(*derived.explosion_point, marker="x", s=35)
    axis.set(xlabel="X (m)", ylabel="Y (m)", zlabel="Altitude (m)", title="UAV routes and missile trajectories")
    axis.legend(ncol=2, fontsize=8)
    return _save(figure, output_dir, "q5_trajectories")


def plot_duration_bars(durations: Sequence[float], output_dir: Path) -> dict[str, Path]:
    figure, axis = plt.subplots(figsize=(6, 4))
    values = np.asarray(durations, dtype=float); axis.bar(["M1", "M2", "M3"], values, color=["#3b82f6", "#f97316", "#22c55e"])
    axis.set(ylabel="Effective interference time (s)", title="Verified interference by missile")
    for index, value in enumerate(values): axis.text(index, value, f"{value:.3f}", ha="center", va="bottom")
    return _save(figure, output_dir, "q5_duration_bars")


def plot_stage_history(history: Sequence[Mapping[str, Any]], output_dir: Path) -> dict[str, Path]:
    figure, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=False)
    for stage in (1, 2):
        rows = [row for row in history if int(row["stage"]) == stage]
        iterations = [row["iteration"] for row in rows]
        axes[stage - 1].plot(iterations, [row["Jsum"] for row in rows], label="Jsum")
        axes[stage - 1].plot(iterations, [row["Jmin"] for row in rows], label="Jmin")
        axes[stage - 1].set(title=f"Integer PSO stage {stage}", ylabel="Coarse duration (s)")
        axes[stage - 1].legend()
    axes[-1].set_xlabel("Iteration")
    return _save(figure, output_dir, "q5_stage_history")


def plot_intervals(intervals: Mapping[int, Sequence[Sequence[float]]], output_dir: Path) -> dict[str, Path]:
    figure, axis = plt.subplots(figsize=(9, 3.8))
    for missile in range(3):
        for start, end in intervals.get(missile, ()):
            axis.broken_barh([(start, end - start)], (missile + 0.65, 0.7))
    axis.set(yticks=[1, 2, 3], yticklabels=["M1", "M2", "M3"], xlabel="Time (s)", title="Verified full-occlusion intervals")
    return _save(figure, output_dir, "q5_intervals")
