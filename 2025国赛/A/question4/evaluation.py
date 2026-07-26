"""Evaluation and per-UAV contribution accounting for Question 4."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from question1.data_processing import ProblemData
from question1.evaluation import EvaluationResult, evaluate_solution
from question1.model import UAVPlan
from question4.model import decode_q4_candidate


def evaluate_q4_candidate(candidate: Sequence[float], data: ProblemData, config: Mapping[str, Any], profile: str = "fast") -> EvaluationResult:
    return evaluate_solution(decode_q4_candidate(candidate, data), [0], data, config["sampling"][profile], config["numerical"], question_id=4)


def _duration(plans: Sequence[UAVPlan], data: ProblemData, config: Mapping[str, Any], profile: str) -> float:
    result = evaluate_solution(plans, [0], data, config["sampling"][profile], config["numerical"], question_id=5)
    if not result.feasible:
        raise ValueError(f"partial Q4 plan is infeasible: {dict(result.violations)}")
    return float(result.duration_by_missile[0])


def _nonnegative(value: float, label: str, tolerance: float = 1e-8) -> float:
    if value >= 0.0:
        return float(value)
    if value >= -tolerance:
        return 0.0
    raise ValueError(f"{label} is unexpectedly negative: {value}")


def uav_contributions(plans: Sequence[UAVPlan], data: ProblemData, config: Mapping[str, Any], profile: str = "verify") -> dict[str, Any]:
    """Compute standalone, stable sequential, and removal marginals for each UAV."""

    ordered_uavs = sorted(plans, key=lambda plan: plan.uav_index)
    if [plan.uav_index for plan in ordered_uavs] != [0, 1, 2] or any(len(plan.bombs) != 1 for plan in ordered_uavs):
        raise ValueError("Q4 contribution analysis requires FY1/FY2/FY3 with one bomb each")
    total = _duration(ordered_uavs, data, config, profile)
    standalone = {plan.uav_index: _duration([plan], data, config, profile) for plan in ordered_uavs}
    release_order = sorted(ordered_uavs, key=lambda plan: (plan.bombs[0].release_time, plan.uav_index, plan.bombs[0].bomb_index))
    sequential: dict[int, float] = {}
    previous = 0.0
    for count, plan in enumerate(release_order, 1):
        current = _duration(release_order[:count], data, config, profile)
        sequential[plan.uav_index] = _nonnegative(current - previous, f"sequential marginal for FY{plan.uav_index + 1}")
        previous = current
    removal = {
        plan.uav_index: _nonnegative(
            total - _duration([item for item in ordered_uavs if item.uav_index != plan.uav_index], data, config, profile),
            f"removal marginal for FY{plan.uav_index + 1}",
        )
        for plan in ordered_uavs
    }
    return {
        "total_duration": total, "standalone": standalone,
        "sequential_marginal": sequential, "removal_marginal": removal,
        "release_order": [plan.uav_index for plan in release_order],
    }
