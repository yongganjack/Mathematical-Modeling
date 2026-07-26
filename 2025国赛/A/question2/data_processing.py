"""Small deterministic candidate generators for Question 2."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.stats import qmc

from question1.data_processing import ProblemData
from question1.model import missile_hit_time


def q2_bounds(data: ProblemData) -> list[tuple[float, float]]:
    """Return bounds for ``[heading, speed, release, fuse]``."""

    hit = missile_hit_time(0, data)
    max_fuse = min(float(data.uav_init[0, 2]) * 2.0 / float(data.gravity), hit)
    max_fuse = float(np.sqrt(max(0.0, 2.0 * float(data.uav_init[0, 2]) / float(data.gravity))))
    return [(-float(np.pi), float(np.pi)), tuple(map(float, data.uav_speed_bounds)), (0.0, hit), (0.0, min(max_fuse, hit))]


def latin_hypercube_candidates(
    data: ProblemData, count: int, rng: np.random.Generator
) -> np.ndarray:
    """Generate a small deduplicated Latin-hypercube candidate set."""

    if count <= 0:
        return np.empty((0, 4), dtype=float)
    bounds = np.asarray(q2_bounds(data), dtype=float)
    sampler = qmc.LatinHypercube(d=4, seed=rng)
    raw = qmc.scale(sampler.random(int(count)), bounds[:, 0], bounds[:, 1])
    hit = bounds[2, 1]
    raw[:, 3] = np.minimum(raw[:, 3], np.maximum(0.0, hit - raw[:, 2]))
    return np.unique(np.round(raw, decimals=12), axis=0)


def fixed_q1_candidate(data: ProblemData) -> np.ndarray:
    """Return the Q1 fixed strategy expressed in Q2 coordinates."""

    return np.asarray([np.pi, 120.0, 1.5, 3.6], dtype=float)


def q2_config(config: Mapping[str, Any] | None) -> dict[str, int]:
    """Resolve bounded PSO/DE budgets, keeping quick runs genuinely small."""

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
    particles = max(8, min(24, requested // 80))
    return {"pso_particles": particles, "pso_iterations": max(2, min(30, requested // max(particles, 1) - 1)), "de_particles": max(4, min(12, particles // 2)), "de_iterations": max(2, min(20, requested // max(particles * 2, 1) - 1)), "max_evaluations": requested}
