"""PSO/DE优化器与问题2单炸弹目标函数。"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import logging

import numpy as np
from scipy.optimize import differential_evolution

from question1.data_processing import ProblemData
from question1.evaluation import EvaluationResult, evaluate_solution
from question1.model import BombPlan, UAVPlan, missile_hit_time, max_fuse_delay

logger = logging.getLogger(__name__)


@dataclass
class OptimizerResult:
    best_position: np.ndarray
    best_score: float
    history: list[dict[str, float]]
    evaluations: int
    termination_reason: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.best_position = np.asarray(self.best_position, dtype=float).copy()
        self.best_score = float(self.best_score)
        self.evaluations = int(self.evaluations)


def _safe_score(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return -1e15
    return value if np.isfinite(value) else -1e15


def solve_pso(objective: Callable[[np.ndarray], float], bounds: Sequence[tuple[float, float]], rng: np.random.Generator, particles: int = 16, iterations: int = 20) -> OptimizerResult:
    """使用紧凑、可复现的PSO最大化 ``objective``。"""

    bounds_array = np.asarray(bounds, dtype=float)
    lower, upper = bounds_array[:, 0], bounds_array[:, 1]
    dim = len(bounds_array)
    particles, iterations = max(1, int(particles)), max(0, int(iterations))
    logger.info("开始PSO优化: particles=%d, iterations=%d, dim=%d", particles, iterations, dim)
    positions = rng.uniform(lower, upper, size=(particles, dim))
    velocities = rng.uniform(-(upper - lower), upper - lower, size=(particles, dim)) * 0.1
    pbest = positions.copy()
    pbest_scores = np.asarray([_safe_score(objective(x)) for x in positions], dtype=float)
    evaluations = particles
    best_index = int(np.argmax(pbest_scores))
    gbest = pbest[best_index].copy()
    gbest_score = float(pbest_scores[best_index])
    history: list[dict[str, float]] = []

    def record() -> None:
        nonlocal gbest_score, gbest
        history.append({"iteration": float(len(history)), "best": float(gbest_score), "mean": float(np.mean(pbest_scores)), "std": float(np.std(pbest_scores)), "diversity": float(np.mean(np.std(positions, axis=0))), "evaluations": float(evaluations)})

    record()
    report_interval = max(1, iterations // 5)
    vmax = 0.5 * (upper - lower)
    for _ in range(iterations):
        inertia = 0.9 - 0.5 * (len(history) / max(iterations, 1))
        r1, r2 = rng.random((2, particles, dim))
        velocities = inertia * velocities + 1.5 * r1 * (pbest - positions) + 1.5 * r2 * (gbest - positions)
        velocities = np.clip(velocities, -vmax, vmax)
        positions = np.clip(positions + velocities, lower, upper)
        scores = np.asarray([_safe_score(objective(x)) for x in positions], dtype=float)
        evaluations += particles
        improved = scores > pbest_scores
        pbest[improved] = positions[improved]
        pbest_scores[improved] = scores[improved]
        index = int(np.argmax(pbest_scores))
        if pbest_scores[index] > gbest_score:
            gbest_score, gbest = float(pbest_scores[index]), pbest[index].copy()
        record()
        if len(history) % report_interval == 0:
            logger.info("PSO迭代 %d/%d, 最佳=%.4f, 均值=%.4f, 评估次数=%d", len(history), iterations, gbest_score, np.mean(pbest_scores), evaluations)
    logger.info("PSO完成: 最佳得分=%.4f, 评估次数=%d, 终止原因=%s", gbest_score, evaluations, "iterations_exhausted")
    return OptimizerResult(gbest, gbest_score, history, evaluations, "iterations_exhausted", "pso")


def solve_de(objective: Callable[[np.ndarray], float], bounds: Sequence[tuple[float, float]], rng: np.random.Generator, particles: int = 8, iterations: int = 10) -> OptimizerResult:
    """通过SciPy差分进化（拉丁超立方初始化）进行最大化。"""

    seed = int(rng.integers(0, 2**32 - 1))
    logger.info("开始DE优化: particles=%d, iterations=%d, seed=%d", particles, iterations, seed)
    history: list[dict[str, float]] = []
    best_score = -1e15
    best_position = np.mean(np.asarray(bounds, dtype=float), axis=1)
    evaluations = 0

    def wrapped(x: np.ndarray) -> float:
        nonlocal evaluations, best_score, best_position
        score = _safe_score(objective(np.asarray(x, dtype=float)))
        evaluations += 1
        if score > best_score:
            best_score, best_position = score, np.asarray(x, dtype=float).copy()
        return -score

    def callback(xk: np.ndarray, convergence: float) -> bool:
        history.append({"iteration": float(len(history)), "best": float(best_score), "mean": float(best_score), "std": 0.0, "diversity": 0.0, "evaluations": float(evaluations), "convergence": float(convergence)})
        return False

    result = differential_evolution(wrapped, list(bounds), seed=seed, popsize=max(1, int(particles)), maxiter=max(0, int(iterations)), init="latinhypercube", polish=False, updating="immediate", callback=callback, workers=1, tol=-1.0, atol=-1.0)
    if not history:
        history.append({"iteration": 0.0, "best": float(best_score), "mean": float(best_score), "std": 0.0, "diversity": 0.0, "evaluations": float(evaluations)})
    logger.info("DE完成: 最佳得分=%.4f, 评估次数=%d, 终止原因=%s", best_score, evaluations, str(getattr(result, "message", "completed")))
    return OptimizerResult(best_position, best_score, history, evaluations, str(getattr(result, "message", "completed")), "de")


def decode_q2_candidate(candidate: Sequence[float], data: ProblemData) -> UAVPlan:
    values = np.asarray(candidate, dtype=float)
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise ValueError("Q2 候选解必须为有限向量 [theta, speed, release, tau]")
    theta, speed, release, tau = map(float, values)
    hit = missile_hit_time(0, data)
    theta = float((theta + np.pi) % (2.0 * np.pi) - np.pi)
    speed = float(np.clip(speed, *map(float, data.uav_speed_bounds)))
    release = float(np.clip(release, 0.0, hit))
    tau = float(np.clip(tau, 0.0, min(max_fuse_delay(0, data), hit - release)))
    return UAVPlan(0, theta, speed, (BombPlan(1, release, tau, 0),))


def q2_objective(candidate: Sequence[float], data: ProblemData, config: Mapping[str, Any] | None = None, profile: str = "fast", return_evaluation: bool = False) -> float | tuple[float, EvaluationResult]:
    try:
        plan = decode_q2_candidate(candidate, data)
        cfg = config or {"sampling": {"fast": {"target_surface_points": 72, "time_step": 0.05}}, "numerical": {"root_tolerance": 1e-6, "interval_merge_tolerance": 1e-8}}
        result = evaluate_solution([plan], [0], data, cfg["sampling"][profile], cfg["numerical"], question_id=2)
        score = float(result.duration_by_missile[0]) if result.feasible else -1e15
    except Exception:
        result = None
        score = -1e15
    return (score, result) if return_evaluation else score


def solve_question2(data: ProblemData, config: Mapping[str, Any], pso_rng: np.random.Generator | None = None, de_rng: np.random.Generator | None = None) -> OptimizerResult:
    from question2.data_processing import q2_bounds, q2_config

    if pso_rng is None or de_rng is None:
        pso_seed, de_seed = np.random.SeedSequence(2025).spawn(2)
        pso_rng = pso_rng or np.random.default_rng(pso_seed)
        de_rng = de_rng or np.random.default_rng(de_seed)
    budget = q2_config(config)
    logger.info("开始问题2优化: pso_particles=%d, pso_iterations=%d, de_particles=%d, de_iterations=%d, max_evaluations=%d",
                budget["pso_particles"], budget["pso_iterations"], budget["de_particles"], budget["de_iterations"], budget["max_evaluations"])
    objective = lambda x: q2_objective(x, data, config, "fast")
    pso = solve_pso(objective, q2_bounds(data), pso_rng, budget["pso_particles"], budget["pso_iterations"])
    logger.info("PSO最佳得分: %.4f, 评估次数: %d", pso.best_score, pso.evaluations)
    de = solve_de(objective, q2_bounds(data), de_rng, budget["de_particles"], budget["de_iterations"])
    logger.info("DE最佳得分: %.4f, 评估次数: %d", de.best_score, de.evaluations)
    for optimizer in (pso, de):
        plan = decode_q2_candidate(optimizer.best_position, data)
        bomb = plan.bombs[0]
        optimizer.best_position = np.asarray(
            [plan.heading_rad, plan.speed, bomb.release_time, bomb.fuse_delay],
            dtype=float,
        )
    candidates = [pso.best_position, de.best_position]
    logger.info("开始验证阶段: 对 %d 个候选解进行verify评估", len(candidates))
    verified: list[tuple[float, np.ndarray, EvaluationResult | None]] = []
    for candidate in candidates:
        score, result = q2_objective(candidate, data, config, "verify", return_evaluation=True)
        verified.append((float(score), np.asarray(candidate, dtype=float), result))
    verified.sort(key=lambda item: item[0], reverse=True)
    best_score, best_position, best_eval = verified[0]
    logger.info("验证完成: 最终最佳得分=%.4f, feasible=%s", best_score, best_eval.feasible if best_eval else "N/A")
    return OptimizerResult(best_position, best_score, pso.history + [{**row, "source": "de"} for row in de.history], pso.evaluations + de.evaluations, "pso_and_de_completed", "q2", {"pso": pso, "de": de, "verified_evaluation": best_eval, "fast_best": {"pso": pso.best_score, "de": de.best_score}, "verified_candidates": verified})
