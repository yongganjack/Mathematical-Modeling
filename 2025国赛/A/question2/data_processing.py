"""问题2的小型确定性候选生成器。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.stats import qmc

from question1.data_processing import ProblemData
from question1.model import missile_hit_time


def q2_bounds(data: ProblemData) -> list[tuple[float, float]]:
    """返回 ``[heading, speed, release, fuse]`` 的边界。"""

    hit = missile_hit_time(0, data)
    max_fuse = min(float(data.uav_init[0, 2]) * 2.0 / float(data.gravity), hit)
    max_fuse = float(np.sqrt(max(0.0, 2.0 * float(data.uav_init[0, 2]) / float(data.gravity))))
    return [(-float(np.pi), float(np.pi)), tuple(map(float, data.uav_speed_bounds)), (0.0, hit), (0.0, min(max_fuse, hit))]


def latin_hypercube_candidates(
    data: ProblemData, count: int, rng: np.random.Generator
) -> np.ndarray:
    """生成一个小型的去重拉丁超立方候选集。"""

    if count <= 0:
        return np.empty((0, 4), dtype=float)
    bounds = np.asarray(q2_bounds(data), dtype=float)
    sampler = qmc.LatinHypercube(d=4, seed=rng)
    raw = qmc.scale(sampler.random(int(count)), bounds[:, 0], bounds[:, 1])
    hit = bounds[2, 1]
    raw[:, 3] = np.minimum(raw[:, 3], np.maximum(0.0, hit - raw[:, 2]))
    return np.unique(np.round(raw, decimals=12), axis=0)


def fixed_q1_candidate(data: ProblemData) -> np.ndarray:
    """返回以问题2坐标表示的问题1固定策略。"""

    return np.asarray([np.pi, 120.0, 1.5, 3.6], dtype=float)


def q2_config(config: Mapping[str, Any] | None) -> dict[str, int]:
    """解析有界的PSO/DE预算，保持快速运行的真实小规模。"""

    runtime = (config or {}).get("optimization", {}).get("q2_runtime", {})
    if runtime:
        return {
            "pso_particles": int(runtime["pso_particles"]),
            "pso_iterations": int(runtime["pso_iterations"]),
            "de_particles": int(runtime["de_particles"]),
            "de_iterations": int(runtime["de_iterations"]),
            "max_evaluations": int((config or {}).get("optimization", {}).get("budgets", {}).get("q2", {}).get("max_evaluations", 2000)),
        }
    profile = str((config or {}).get("profile", "quick")).lower()
    requested = int((config or {}).get("optimization", {}).get("budgets", {}).get("q2", {}).get("max_evaluations", 2000))
    if profile == "quick":
        return {"pso_particles": 8, "pso_iterations": 6, "de_particles": 4, "de_iterations": 2, "max_evaluations": min(requested, 2000)}
    # 正常/竞赛模式: PSO=30粒子×50代, DE=6粒子×50代（4维问题）
    return {"pso_particles": 30, "pso_iterations": 50, "de_particles": 6, "de_iterations": 50, "max_evaluations": requested}
