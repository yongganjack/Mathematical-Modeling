"""English-language figures for the fixed Question 1 strategy."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from question1.data_processing import ProblemData
from question1.model import (
    DerivedBomb,
    UAVPlan,
    missile_hit_time,
    missile_position,
    smoke_center,
    uav_position,
)


def _figure_paths(output_dir: str | Path, stem: str) -> dict[str, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return {
        "png": directory / f"{stem}.png",
        "svg": directory / f"{stem}.svg",
        "csv": directory / f"{stem}.csv",
    }


def _save_figure(figure: plt.Figure, paths: Mapping[str, Path]) -> None:
    figure.tight_layout()
    figure.savefig(paths["png"], dpi=200, bbox_inches="tight")
    figure.savefig(paths["svg"], bbox_inches="tight")
    plt.close(figure)


def plot_trajectory(
    data: ProblemData,
    uav_plan: UAVPlan,
    derived_bomb: DerivedBomb,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Plot missile, UAV, release, explosion, and smoke-center trajectories."""

    paths = _figure_paths(output_dir, "trajectory")
    missile_times = np.linspace(0.0, missile_hit_time(0, data), 180)
    missile_xyz = np.asarray(
        [missile_position(time, 0, data) for time in missile_times]
    )
    uav_end = derived_bomb.explosion_time + float(data.smoke_lifetime)
    uav_times = np.linspace(0.0, uav_end, 120)
    uav_xyz = np.asarray([uav_position(time, uav_plan, data) for time in uav_times])
    smoke_times = np.linspace(derived_bomb.explosion_time, uav_end, 100)
    smoke_samples = [
        (time, smoke_center(time, derived_bomb, 0, data)) for time in smoke_times
    ]
    smoke_samples = [(time, center) for time, center in smoke_samples if center is not None]
    smoke_xyz = np.asarray([center for _, center in smoke_samples])

    with paths["csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["entity", "time", "x", "y", "z"])
        for time, point in zip(missile_times, missile_xyz):
            writer.writerow(["missile", time, *point])
        for time, point in zip(uav_times, uav_xyz):
            writer.writerow(["uav", time, *point])
        writer.writerow(["release_point", derived_bomb.release_time, *derived_bomb.release_point])
        writer.writerow(["explosion_point", derived_bomb.explosion_time, *derived_bomb.explosion_point])
        for time, point in smoke_samples:
            writer.writerow(["smoke_center", time, *point])

    figure = plt.figure(figsize=(9, 6))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(*missile_xyz.T, label="Missile trajectory", color="tab:red")
    axis.plot(*uav_xyz.T, label="UAV straight-line path", color="tab:blue")
    axis.scatter(*derived_bomb.release_point, label="Release point", marker="o", s=55)
    axis.scatter(*derived_bomb.explosion_point, label="Explosion point", marker="*", s=90)
    if len(smoke_xyz):
        axis.plot(*smoke_xyz.T, label="Smoke-cloud center", color="tab:gray")
    axis.set_title("Question 1 Trajectories and Smoke-Cloud Motion")
    axis.set_xlabel("X coordinate (m)")
    axis.set_ylabel("Y coordinate (m)")
    axis.set_zlabel("Altitude (m)")
    axis.legend(loc="best")
    _save_figure(figure, paths)
    return paths


def plot_intervals(
    intervals_by_missile: Mapping[int, Sequence[Sequence[float]]],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Plot effective smoke-screen intervals as an English timeline."""

    paths = _figure_paths(output_dir, "intervals")
    rows: list[tuple[int, int, float, float, float]] = []
    for missile_index in sorted(intervals_by_missile):
        for interval_index, interval in enumerate(intervals_by_missile[missile_index], 1):
            start, end = map(float, interval)
            rows.append((int(missile_index), interval_index, start, end, end - start))
    with paths["csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["missile_index", "interval_index", "start_time", "end_time", "duration"])
        writer.writerows(rows)

    figure, axis = plt.subplots(figsize=(8, 3.8))
    for missile_index, _, start, end, _ in rows:
        axis.barh(
            f"Missile {missile_index + 1}",
            end - start,
            left=start,
            height=0.45,
            color="tab:blue",
        )
    axis.set_title("Effective Smoke-Screen Intervals")
    axis.set_xlabel("Time after launch (s)")
    axis.set_ylabel("Threat missile")
    axis.grid(axis="x", alpha=0.3)
    _save_figure(figure, paths)
    return paths


def plot_convergence(
    convergence_rows: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Compare fast and verification objective durations."""

    paths = _figure_paths(output_dir, "convergence")
    rows = [dict(row) for row in convergence_rows]
    fields = [
        "profile",
        "duration",
        "time_step",
        "target_surface_points",
        "absolute_difference",
    ]
    with paths["csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    labels = [str(row["profile"]) for row in rows]
    durations = [float(row["duration"]) for row in rows]
    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    bars = axis.bar(labels, durations, color=["tab:orange", "tab:green"][: len(rows)])
    axis.bar_label(bars, fmt="%.6f")
    axis.set_title("Fast and Verification Duration Comparison")
    axis.set_xlabel("Evaluation profile")
    axis.set_ylabel("Effective duration (s)")
    axis.grid(axis="y", alpha=0.3)
    _save_figure(figure, paths)
    return paths
