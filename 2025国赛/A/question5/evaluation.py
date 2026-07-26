"""问题5的最终物理评估与贡献核算。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from question1.data_processing import ProblemData
from question1.evaluation import EvaluationResult, evaluate_solution
from question1.model import BombPlan, UAVPlan
from question5.model import Route, RouteLibrary, approximate_intervals, approximate_score, route_coverage


def routes_to_plans(routes: Sequence[Route]) -> tuple[UAVPlan, ...]:
    ordered = sorted(routes, key=lambda route: route.uav_index)
    if [route.uav_index for route in ordered] != list(range(5)):
        raise ValueError("Q5 需要五个 UAV 各一条选定路径")
    return tuple(route.to_plan() for route in ordered)


def evaluate_final_routes(
    routes: Sequence[Route], data: ProblemData, config: Mapping[str, Any]
) -> tuple[EvaluationResult, EvaluationResult]:
    """调用完整评估器各一次，分别使用快速模式和验证模式。"""

    plans = routes_to_plans(routes)
    fast = evaluate_solution(plans, [0, 1, 2], data, config["sampling"]["fast"], config["numerical"], question_id=5)
    verify = evaluate_solution(plans, [0, 1, 2], data, config["sampling"]["verify"], config["numerical"], question_id=5)
    return fast, verify


def _durations(plans: Sequence[UAVPlan], data: ProblemData, config: Mapping[str, Any]) -> np.ndarray:
    result = evaluate_solution(plans, [0, 1, 2], data, config["sampling"]["fast"], config["numerical"], question_id=5)
    if not result.feasible:
        raise ValueError(f"Q5 部分方案不可行: {dict(result.violations)}")
    return np.asarray(result.duration_by_missile, dtype=float)


def contribution_analysis(
    routes: Sequence[Route], data: ProblemData, config: Mapping[str, Any]
) -> dict[str, Any]:
    """使用快速评估计算序列炸弹边际贡献和路径移除边际贡献。"""

    plans = list(routes_to_plans(routes))
    ordered_bombs = sorted(
        ((bomb.release_time, plan.uav_index, bomb.bomb_index, bomb) for plan in plans for bomb in plan.bombs),
        key=lambda item: (item[0], item[1], item[2]),
    )
    prefix: dict[int, list[BombPlan]] = {index: [] for index in range(5)}
    previous = np.zeros(3, dtype=float)
    sequential: dict[tuple[int, int], float] = {}
    attached: dict[tuple[int, int], list[float]] = {}
    order: list[dict[str, Any]] = []
    negative_residuals: list[dict[str, Any]] = []
    for _, uav, bomb_index, bomb in ordered_bombs:
        prefix[uav].append(bomb)
        current_plans = [
            UAVPlan(plan.uav_index, plan.heading_rad, plan.speed, tuple(prefix[plan.uav_index]))
            for plan in plans
        ]
        current = _durations(current_plans, data, config)
        delta = current - previous
        cleaned = delta.copy()
        for missile, value in enumerate(delta):
            if -1e-6 <= value < 0.0:
                cleaned[missile] = 0.0
            elif value < -1e-6:
                negative_residuals.append({"uav_index": uav, "bomb_index": bomb_index, "missile_index": missile, "value": float(value)})
        label = int(bomb.assigned_missile if bomb.assigned_missile is not None else 0)
        sequential[(uav, bomb_index)] = float(max(0.0, cleaned[label]))
        attached[(uav, bomb_index)] = [float(value) for value in cleaned]
        order.append({"uav_index": uav, "bomb_index": bomb_index, "assigned_missile": label, "prefix_duration": current.tolist()})
        previous = current
    total = _durations(plans, data, config)
    removal: dict[int, list[float]] = {}
    for plan in plans:
        without = [other for other in plans if other.uav_index != plan.uav_index]
        difference = total - _durations(without, data, config)
        difference[(difference < 0.0) & (difference >= -1e-6)] = 0.0
        removal[plan.uav_index] = [float(max(0.0, value)) for value in difference]
    return {
        "sequential_marginal": {f"FY{uav + 1}_B{bomb}": value for (uav, bomb), value in sequential.items()},
        "sequential_lookup": sequential,
        "attached_all_missiles": {f"FY{uav + 1}_B{bomb}": value for (uav, bomb), value in attached.items()},
        "route_removal_marginal": {f"FY{uav + 1}": value for uav, value in removal.items()},
        "release_order": order,
        "fast_total_duration": total.tolist(),
        "negative_residuals": negative_residuals,
        "evaluation_count": len(ordered_bombs) + 1 + len(plans),
    }


def coarse_selected_summary(routes: Sequence[Route], library: RouteLibrary, data: ProblemData) -> dict[str, Any]:
    coverages = [route_coverage(route, library.grid, data) for route in routes]
    score = approximate_score(coverages, library.grid)
    return {"Jsum": score[0], "Jmin": score[1], "intervals": approximate_intervals(coverages, library.grid)}
