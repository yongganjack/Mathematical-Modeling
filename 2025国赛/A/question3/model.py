"""Candidate decoding and cooperative optimization for Question 3."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from question1.data_processing import ProblemData
from question1.evaluation import EvaluationResult, evaluate_solution
from question1.model import BombPlan, UAVPlan, max_fuse_delay, missile_hit_time
from question2.model import OptimizerResult, solve_de, solve_pso


def decode_q3_candidate(candidate: Sequence[float], data: ProblemData) -> UAVPlan:
    """Repair a finite eight-variable candidate into a feasible FY1 plan."""

    values = np.asarray(candidate, dtype=float)
    if values.shape != (8,) or not np.all(np.isfinite(values)):
        raise ValueError("Q3 candidate must be a finite 8-vector")
    theta, speed, base, slack2, slack3, tau1, tau2, tau3 = map(float, values)
    theta = float((theta + np.pi) % (2.0 * np.pi) - np.pi)
    speed = float(np.clip(speed, *map(float, data.uav_speed_bounds)))
    hit = missile_hit_time(0, data)
    gap = float(data.min_release_interval)
    if hit < 2.0 * gap:
        raise ValueError("M1 flight horizon cannot accommodate three releases")
    release1 = float(np.clip(base, 0.0, hit - 2.0 * gap))
    release2 = release1 + gap + float(np.clip(slack2, 0.0, hit - release1 - 2.0 * gap))
    release3 = release2 + gap + float(np.clip(slack3, 0.0, hit - release2 - gap))
    fuse_limit = max_fuse_delay(0, data)
    releases = (release1, release2, release3)
    fuses = tuple(float(np.clip(tau, 0.0, min(fuse_limit, hit - release))) for tau, release in zip((tau1, tau2, tau3), releases))
    bombs = tuple(BombPlan(index, release, fuse, 0) for index, (release, fuse) in enumerate(zip(releases, fuses), 1))
    return UAVPlan(0, theta, speed, bombs)


def encode_q3_plan(plan: UAVPlan, data: ProblemData) -> np.ndarray:
    """Encode a decoded plan back into the eight optimization coordinates."""

    if len(plan.bombs) != 3:
        raise ValueError("Q3 plan must contain exactly three bombs")
    gap = float(data.min_release_interval)
    releases = [float(bomb.release_time) for bomb in plan.bombs]
    return np.asarray([
        plan.heading_rad, plan.speed, releases[0],
        max(0.0, releases[1] - releases[0] - gap),
        max(0.0, releases[2] - releases[1] - gap),
        *(float(bomb.fuse_delay) for bomb in plan.bombs),
    ], dtype=float)


def q3_objective(candidate: Sequence[float], data: ProblemData, config: Mapping[str, Any], profile: str = "fast", return_evaluation: bool = False) -> float | tuple[float, EvaluationResult | None]:
    """Score M1's pointwise joint coverage produced by all three clouds."""

    result: EvaluationResult | None = None
    try:
        plan = decode_q3_candidate(candidate, data)
        result = evaluate_solution([plan], [0], data, config["sampling"][profile], config["numerical"], question_id=3)
        score = float(result.duration_by_missile[0]) if result.feasible else -1e15
    except Exception:
        score = -1e15
    return (score, result) if return_evaluation else score


def _seed_candidates(data: ProblemData) -> list[np.ndarray]:
    """Geometry-based starts only; no stored optimizer result is embedded."""

    hit = missile_hit_time(0, data)
    base = min(1.5, max(0.0, hit - 2.0))
    fuse = min(3.6, max_fuse_delay(0, data))
    return [
        np.asarray([np.pi, 120.0, base, 0.0, 0.0, fuse, fuse, fuse]),
        np.asarray([np.pi, 120.0, base, 1.0, 1.0, fuse, fuse, fuse]),
        np.asarray([np.pi, 100.0, 0.0, 2.0, 2.0, 2.0, 3.0, 4.0]),
    ]


def solve_question3(data: ProblemData, config: Mapping[str, Any], pso_rng: np.random.Generator | None = None, de_rng: np.random.Generator | None = None) -> OptimizerResult:
    """Run compact PSO and DE searches, then verify the best candidates."""

    from question3.data_processing import q3_bounds, q3_config

    if pso_rng is None or de_rng is None:
        pso_seed, de_seed = np.random.SeedSequence(2025).spawn(2)
        pso_rng = pso_rng or np.random.default_rng(pso_seed)
        de_rng = de_rng or np.random.default_rng(de_seed)
    runtime = q3_config(config)
    objective = lambda x: q3_objective(x, data, config, "fast")
    bounds = q3_bounds(data)
    pso = solve_pso(objective, bounds, pso_rng, runtime["pso_particles"], runtime["pso_iterations"])
    de = solve_de(objective, bounds, de_rng, runtime["de_particles"], runtime["de_iterations"])
    for optimizer in (pso, de):
        optimizer.best_position = encode_q3_plan(decode_q3_candidate(optimizer.best_position, data), data)

    seeds = _seed_candidates(data)
    seed_scores = [(float(objective(seed)), encode_q3_plan(decode_q3_candidate(seed, data), data)) for seed in seeds]
    fast_candidates = [(pso.best_score, pso.best_position, "pso"), (de.best_score, de.best_position, "de")]
    fast_candidates.extend((score, seed, f"seed_{index}") for index, (score, seed) in enumerate(seed_scores, 1))
    verified: list[tuple[float, np.ndarray, str, EvaluationResult | None]] = []
    for _, candidate, source in fast_candidates:
        score, evaluation = q3_objective(candidate, data, config, "verify", return_evaluation=True)
        verified.append((float(score), np.asarray(candidate, dtype=float), source, evaluation))
    verified.sort(key=lambda item: item[0], reverse=True)
    best_score, best_position, selected_source, best_evaluation = verified[0]
    return OptimizerResult(
        best_position, best_score,
        [{**row, "source": "pso"} for row in pso.history] + [{**row, "source": "de"} for row in de.history],
        pso.evaluations + de.evaluations + len(seed_scores) + len(verified),
        "pso_and_de_completed", "q3",
        {
            "pso": pso, "de": de, "verified_evaluation": best_evaluation,
            "selected_source": selected_source,
            "fast_best": {"pso": pso.best_score, "de": de.best_score},
            "seed_fast_scores": {f"seed_{index}": score for index, (score, _) in enumerate(seed_scores, 1)},
            "verified_candidates": verified,
            "evaluation_counts": {
                "pso": pso.evaluations, "de": de.evaluations,
                "seeds": len(seed_scores), "verification": len(verified),
                "total": pso.evaluations + de.evaluations + len(seed_scores) + len(verified),
            },
        },
    )
