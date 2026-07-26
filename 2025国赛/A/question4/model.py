"""问题4的候选解码与协同优化。"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from question1.data_processing import ProblemData
from question1.evaluation import EvaluationResult, evaluate_solution
from question1.model import BombPlan, UAVPlan, max_fuse_delay, missile_hit_time
from question2.model import OptimizerResult, solve_de, solve_pso

logger = logging.getLogger(__name__)


def decode_q4_candidate(candidate: Sequence[float], data: ProblemData) -> tuple[UAVPlan, UAVPlan, UAVPlan]:
    """将12个变量修复为独立的可行FY1/FY2/FY3计划。"""

    values = np.asarray(candidate, dtype=float)
    logger.debug("decode_q4_candidate: 输入向量形状=%s", values.shape)
    if values.shape != (12,) or not np.all(np.isfinite(values)):
        raise ValueError("Q4 候选解必须为有限 12 维向量")
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
        raise ValueError("Q4 编码需要 FY1/FY2/FY3 各一枚炸弹")
    return np.asarray([
        value
        for plan in ordered
        for value in (plan.heading_rad, plan.speed, plan.bombs[0].release_time, plan.bombs[0].fuse_delay)
    ], dtype=float)


def q4_objective(
    candidate: Sequence[float], data: ProblemData, config: Mapping[str, Any],
    profile: str = "fast", return_evaluation: bool = False,
) -> float | tuple[float, EvaluationResult | None]:
    """计算三片同时烟雾云对M1的逐点联合覆盖得分。"""

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
    """创建交错几何起始点；DE提供拉丁超立方探索。"""

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
    """运行紧凑的PSO/DE优化，然后验证其最佳候选和启发式方案。"""

    from question4.data_processing import q4_bounds, q4_config

    if pso_rng is None or de_rng is None:
        pso_seed, de_seed = np.random.SeedSequence(2025).spawn(2)
        pso_rng = pso_rng or np.random.default_rng(pso_seed)
        de_rng = de_rng or np.random.default_rng(de_seed)
    runtime = q4_config(config)
    logger.info("solve_question4: 开始优化, 运行配置=%s", runtime)
    objective = lambda x: q4_objective(x, data, config, "fast")
    bounds = q4_bounds(data)
    logger.info("开始PSO优化, particles=%d, iterations=%d", runtime["pso_particles"], runtime["pso_iterations"])
    pso = solve_pso(objective, bounds, pso_rng, runtime["pso_particles"], runtime["pso_iterations"])
    logger.info("PSO完成, 最佳得分=%.4f, 评估次数=%d, 终止原因=%s", pso.best_score, pso.evaluations, pso.termination_reason)
    logger.info("开始DE优化, particles=%d, iterations=%d", runtime["de_particles"], runtime["de_iterations"])
    de = solve_de(objective, bounds, de_rng, runtime["de_particles"], runtime["de_iterations"])
    logger.info("DE完成, 最佳得分=%.4f, 评估次数=%d, 终止原因=%s", de.best_score, de.evaluations, de.termination_reason)
    for optimizer in (pso, de):
        optimizer.best_position = encode_q4_plans(decode_q4_candidate(optimizer.best_position, data))

    seeds = _seed_candidates(data)
    logger.info("评估 %d 个种子候选...", len(seeds))
    seed_scores = [(float(objective(seed)), encode_q4_plans(decode_q4_candidate(seed, data))) for seed in seeds]
    logger.info("种子候选快速评估分数: %s", {f"seed_{i}": s for i, (s, _) in enumerate(seed_scores, 1)})
    fast_candidates = [(pso.best_score, pso.best_position, "pso"), (de.best_score, de.best_position, "de")]
    fast_candidates.extend((score, seed, f"seed_{index}") for index, (score, seed) in enumerate(seed_scores, 1))
    verified: list[tuple[float, np.ndarray, str, EvaluationResult | None]] = []
    logger.info("开始验证模式评估 %d 个候选...", len(fast_candidates))
    for idx, (_, candidate, source) in enumerate(fast_candidates):
        score, evaluation = q4_objective(candidate, data, config, "verify", return_evaluation=True)
        verified.append((float(score), np.asarray(candidate, dtype=float), source, evaluation))
        logger.debug("候选 %d/%d (来源=%s): 验证得分=%.4f", idx + 1, len(fast_candidates), source, float(score))
    verified.sort(key=lambda item: item[0], reverse=True)
    best_score, best_position, selected_source, best_evaluation = verified[0]
    logger.info("最佳候选来源=%s, 验证得分=%.4f, PSO最优=%.4f, DE最优=%.4f", selected_source, best_score, pso.best_score, de.best_score)
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
