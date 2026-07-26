"""Evaluation adapters that delegate all geometry to Question 1."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from question1.data_processing import ProblemData
from question1.evaluation import EvaluationResult, check_solution, evaluate_solution
from question2.model import decode_q2_candidate


def evaluate_q2_candidate(candidate: Sequence[float], data: ProblemData, config: Mapping[str, Any], profile: str = "fast") -> EvaluationResult:
    plan = decode_q2_candidate(candidate, data)
    return evaluate_solution([plan], [0], data, config["sampling"][profile], config["numerical"], question_id=2)


def candidate_summary(candidate: Sequence[float], data: ProblemData) -> dict[str, Any]:
    plan = decode_q2_candidate(candidate, data)
    bomb = plan.bombs[0]
    return {"heading_rad": float(plan.heading_rad), "heading_deg": float(np.degrees(plan.heading_rad)), "speed": float(plan.speed), "release_time": float(bomb.release_time), "fuse_delay": float(bomb.fuse_delay), "explosion_time": float(bomb.release_time + bomb.fuse_delay)}


def feasibility_summary(candidate: Sequence[float], data: ProblemData) -> dict[str, Any]:
    plan = decode_q2_candidate(candidate, data)
    report = check_solution([plan], 2, data)
    return {"feasible": bool(report.feasible), "violations": dict(report.violations)}


def compare_optimizers(pso: Any, de: Any) -> dict[str, Any]:
    return {"pso": {"best_score": float(pso.best_score), "evaluations": int(pso.evaluations)}, "de": {"best_score": float(de.best_score), "evaluations": int(de.evaluations)}, "winner": "pso" if pso.best_score >= de.best_score else "de"}
