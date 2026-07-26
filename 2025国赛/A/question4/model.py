"""Candidate decoding and cooperative optimization for Question 4."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from question1.data_processing import ProblemData
from question1.evaluation import EvaluationResult, evaluate_solution
from question1.model import BombPlan, UAVPlan, max_fuse_delay, missile_hit_time
from question2.model import OptimizerResult, solve_de, solve_pso


def decode_q4_candidate(candidate: Sequence[float], data: ProblemData) -> tuple[UAVPlan, UAVPlan, UAVPlan]:
    """Repair the 12 variables into independent feasible FY1/FY2/FY3 plans."""

    values = np.asarray(candidate, dtype=float)
    if values.shape != (12,) or not np.all(np.isfinite(values)):
        raise ValueError("Q4 candidate must be a finite 12-vector")
    hit = missile_hit_time(0, data)
    plans: list[UAVPlan] = []
    for uav_index, offset in enumerate(range(0, 12, 4)):
        theta, speed, release, tau = map(float, values[offset:offset + 4])
        theta = float((theta + np.pi) % (2.0 * np.pi) - np.pi)
        speed = float(np.clip(speed, *map(float, data.uav_speed_bounds)))
        release = float(np.clip(release, 0.0, hit))
        tau = float(np.clip(tau, 0.0, min(max_fuse_delay(uav_index, data), hit - release)))
        plans.append(UAVPlan(uav_index, theta, speed, (BombPlan(1, release, tau, 0),)))
    return tuple(plans)  # type: ignore[return-value]


def encode_q4_plans(plans: Sequence[UAVPlan]) -> np.ndarray:
    ordered = sorted(plans, key=lambda plan: plan.uav_index)
    if [plan.uav_index for plan in ordered] != [0, 1, 2] or any(len(plan.bombs) != 1 for plan in ordered):
        raise ValueError("Q4 encoding requires FY1/FY2/FY3 with one bomb each")
    return np.asarray([
        value
        for plan in ordered
        for value in (plan.heading_rad, plan.speed, plan.bombs[0].release_time, plan.bombs[0].fuse_delay)
    ], dtype=float)


def q4_objective(
    candidate: Sequence[float], data: ProblemData, config: Mapping[str, Any],
    profile: str = "fast", return_evaluation: bool = False,
) -> float | tuple[float, EvaluationResult | None]:
    """Score M1 pointwise joint coverage from all three simultaneous clouds."""

    result: EvaluationResult | None = None
    try:
        plans = decode_q4_candidate(candidate, data)
        result = evaluate_solution(plans, [0], data, config["sampling"][profile], config["numerical"], question_id=4)
        score = float(result.duration_by_missile[0]) if result.feasible else -1e15
    except Exception:
        score = -1e15
    return (score, result) if return_evaluation else score


def _heading(origin: np.ndarray, destination_xy: np.ndarray) -> float:
    delta = np.asarray(destination_xy, dtype=float) - np.asarray(origin[:2], dtype=float)
    return float(np.arctan2(delta[1], delta[0]))


def _seed_candidates(data: ProblemData) -> list[np.ndarray]:
    """Create staggered geometry starts; DE supplies Latin-hypercube exploration."""

    hit = missile_hit_time(0, data)
    destinations = (np.zeros(2), np.asarray(data.target_center_xy, dtype=float))
    seeds: list[np.ndarray] = []
    for variant, destination in enumerate((*destinations, np.array([0.0, 100.0]))):
        values: list[float] = []
        for uav_index in range(3):
            release = min(hit, 0.8 + 1.1 * uav_index + 0.35 * variant)
            fuse = min(max_fuse_delay(uav_index, data), hit - release, 3.0 + 0.7 * uav_index)
            values.extend([
                _heading(data.uav_init[uav_index], destination),
                float((120.0, 105.0, 90.0)[uav_index]), release, max(0.0, fuse),
            ])
        seeds.append(np.asarray(values, dtype=float))
    return seeds


def solve_question4(
    data: ProblemData, config: Mapping[str, Any],
    pso_rng: np.random.Generator | None = None, de_rng: np.random.Generator | None = None,
) -> OptimizerResult:
    """Run compact PSO/DE, then verify their best candidates and heuristics."""

    from question4.data_processing import q4_bounds, q4_config

    if pso_rng is None or de_rng is None:
        pso_seed, de_seed = np.random.SeedSequence(2025).spawn(2)
        pso_rng = pso_rng or np.random.default_rng(pso_seed)
        de_rng = de_rng or np.random.default_rng(de_seed)
    runtime = q4_config(config)
    objective = lambda x: q4_objective(x, data, config, "fast")
    bounds = q4_bounds(data)
    pso = solve_pso(objective, bounds, pso_rng, runtime["pso_particles"], runtime["pso_iterations"])
    de = solve_de(objective, bounds, de_rng, runtime["de_particles"], runtime["de_iterations"])
    for optimizer in (pso, de):
        optimizer.best_position = encode_q4_plans(decode_q4_candidate(optimizer.best_position, data))

    seeds = _seed_candidates(data)
    seed_scores = [(float(objective(seed)), encode_q4_plans(decode_q4_candidate(seed, data))) for seed in seeds]
    fast_candidates = [(pso.best_score, pso.best_position, "pso"), (de.best_score, de.best_position, "de")]
    fast_candidates.extend((score, seed, f"seed_{index}") for index, (score, seed) in enumerate(seed_scores, 1))
    verified: list[tuple[float, np.ndarray, str, EvaluationResult | None]] = []
    for _, candidate, source in fast_candidates:
        score, evaluation = q4_objective(candidate, data, config, "verify", return_evaluation=True)
        verified.append((float(score), np.asarray(candidate, dtype=float), source, evaluation))
    verified.sort(key=lambda item: item[0], reverse=True)
    best_score, best_position, selected_source, best_evaluation = verified[0]
    counts = {
        "pso": pso.evaluations, "de": de.evaluations, "seeds": len(seed_scores),
        "verification": len(verified), "total": pso.evaluations + de.evaluations + len(seed_scores) + len(verified),
    }
    return OptimizerResult(
        best_position, best_score,
        [{**row, "source": "pso"} for row in pso.history] + [{**row, "source": "de"} for row in de.history],
        counts["total"], "pso_and_de_completed", "q4",
        {
            "pso": pso, "de": de, "verified_evaluation": best_evaluation,
            "selected_source": selected_source, "fast_best": {"pso": pso.best_score, "de": de.best_score},
            "seed_fast_scores": {f"seed_{index}": score for index, (score, _) in enumerate(seed_scores, 1)},
            "verified_candidates": verified, "evaluation_counts": counts, "runtime_config": runtime,
        },
    )
