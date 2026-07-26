"""问题三的评估函数与单弹贡献核算。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from question1.data_processing import ProblemData
from question1.evaluation import EvaluationResult, evaluate_solution
from question1.model import UAVPlan
from question3.model import decode_q3_candidate


def evaluate_q3_candidate(candidate: Sequence[float], data: ProblemData, config: Mapping[str, Any], profile: str = "fast") -> EvaluationResult:
    plan = decode_q3_candidate(candidate, data)
    return evaluate_solution([plan], [0], data, config["sampling"][profile], config["numerical"], question_id=3)


def _duration(plan: UAVPlan, bombs: tuple, data: ProblemData, config: Mapping[str, Any], profile: str) -> float:
    subset = UAVPlan(plan.uav_index, plan.heading_rad, plan.speed, bombs)
    result = evaluate_solution([subset], [0], data, config["sampling"][profile], config["numerical"], question_id=5)
    if not result.feasible:
        raise ValueError(f"Q3 部分方案不可行: {dict(result.violations)}")
    return float(result.duration_by_missile[0])


def _nonnegative(value: float, label: str, tolerance: float = 1e-8) -> float:
    if value >= 0.0:
        return float(value)
    if value >= -tolerance:
        return 0.0
    raise ValueError(f"{label} 意外为负数: {value}")


def bomb_contributions(plan: UAVPlan, data: ProblemData, config: Mapping[str, Any], profile: str = "verify") -> dict[str, Any]:
    """计算单独覆盖时长、按释放顺序的序贯边际贡献以及去除边际贡献。"""

    if len(plan.bombs) != 3:
        raise ValueError("Q3 贡献分析需要恰好三枚炸弹")
    bombs = tuple(plan.bombs)
    total = _duration(plan, bombs, data, config, profile)
    standalone = {bomb.bomb_index: _duration(plan, (bomb,), data, config, profile) for bomb in bombs}
    ordered = tuple(sorted(bombs, key=lambda bomb: (bomb.release_time, bomb.bomb_index)))
    sequential: dict[int, float] = {}
    previous = 0.0
    for count, bomb in enumerate(ordered, 1):
        current = _duration(plan, ordered[:count], data, config, profile)
        sequential[bomb.bomb_index] = _nonnegative(current - previous, f"sequential marginal for bomb {bomb.bomb_index}")
        previous = current
    removal = {
        bomb.bomb_index: _nonnegative(total - _duration(plan, tuple(item for item in bombs if item is not bomb), data, config, profile), f"removal marginal for bomb {bomb.bomb_index}")
        for bomb in bombs
    }
    return {
        "total_duration": total,
        "standalone": standalone,
        "sequential_marginal": sequential,
        "removal_marginal": removal,
        "release_order": [bomb.bomb_index for bomb in ordered],
    }
