from dataclasses import replace

import numpy as np
import pytest

from question1.evaluation import (
    EvaluationResult,
    check_solution,
    check_uav_plan,
    decode_ordered_release_times,
    evaluate_solution,
)
from question1.model import BombPlan, UAVPlan, max_fuse_delay


def _plan(problem_data, *, uav=0, speed=100.0, bombs=()):
    return UAVPlan(uav, 0.0, speed, tuple(bombs))


def test_uav_feasibility_reports_speed_and_timing_violations(problem_data):
    bomb = BombPlan(0, -1.0, 1.0, 0)
    report = check_uav_plan(_plan(problem_data, speed=69.0, bombs=(bomb,)), problem_data)
    assert not report.feasible
    assert report.violations["speed_low"] > 0
    assert report.violations["negative_release_time"] > 0


def test_release_gap_and_question_bomb_counts(problem_data):
    bombs = (BombPlan(0, 0.0, 1.0, 0), BombPlan(1, 1.0, 1.0, 0))
    assert check_uav_plan(_plan(problem_data, bombs=bombs), problem_data).feasible
    bad = check_solution([_plan(problem_data, bombs=bombs)], 1, problem_data)
    assert not bad.feasible and bad.violations["wrong_bomb_count"] > 0
    q5 = check_solution([], 5, problem_data)
    assert q5.feasible


def test_question_routes_duplicate_and_q3_counts(problem_data):
    q3 = [
        _plan(problem_data, bombs=tuple(BombPlan(i, float(i), 1.0, 0) for i in range(3)))
    ]
    assert check_solution(q3, 3, problem_data).feasible
    duplicate_routes = [_plan(problem_data), _plan(problem_data)]
    report = check_solution(duplicate_routes, 5, problem_data)
    assert not report.feasible and report.violations["multiple_routes_per_uav"] > 0


def test_decode_release_times_clips_and_enforces_gap():
    times = decode_ordered_release_times([-2.0, 0.1, 0.2], 5.0, 1.0)
    assert np.all(times >= 0.0)
    assert np.all(np.diff(times) >= 1.0 - 1e-12)
    with pytest.raises(ValueError):
        decode_ordered_release_times([0, 1, 2], 1.0, 1.0)


@pytest.mark.parametrize("speed", [70.0, 140.0])
def test_speed_boundaries_are_feasible(problem_data, speed):
    assert check_uav_plan(_plan(problem_data, speed=speed), problem_data).feasible


def test_late_and_below_ground_explosions_are_structured(problem_data):
    late = BombPlan(0, 100.0, 0.0, 0)
    below = BombPlan(0, 0.0, 100.0, 0)
    assert check_uav_plan(_plan(problem_data, bombs=(late,)), problem_data).violations["late_explosion"] > 0
    assert check_uav_plan(_plan(problem_data, bombs=(below,)), problem_data).violations["negative_explosion_height"] > 0


def test_exact_maximum_fuse_height_is_feasible(problem_data):
    delay = max_fuse_delay(0, problem_data)
    report = check_uav_plan(
        _plan(problem_data, bombs=(BombPlan(0, 0.0, delay, 0),)),
        problem_data,
    )
    assert "negative_explosion_height" not in report.violations


def test_q5_rejects_four_bombs_and_accepts_empty_route(problem_data):
    four = tuple(BombPlan(i, float(i), 0.0, 0) for i in range(4))
    assert "wrong_bomb_count" in check_solution([_plan(problem_data, bombs=four)], 5, problem_data).violations
    assert check_solution([_plan(problem_data)], 5, problem_data).feasible


def test_real_evaluation_returns_immutable_nonnegative_structure(problem_data):
    bomb = BombPlan(0, 1.5, 3.6, 0)
    plan = _plan(problem_data, speed=120.0, bombs=(bomb,))
    result = evaluate_solution(
        [plan], [0], problem_data,
        {"target_surface_points": 24, "time_step": 0.5},
        {"root_tolerance": 1e-6, "interval_merge_tolerance": 1e-8},
        question_id=1,
    )
    assert isinstance(result, EvaluationResult)
    assert result.feasible
    assert result.duration_by_missile.shape == (3,)
    assert result.duration_by_missile.dtype == np.float64
    assert np.all(result.duration_by_missile >= 0.0)
    assert result.sum_objective >= 0.0 and result.min_duration >= 0.0
    with pytest.raises(ValueError):
        result.duration_by_missile.setflags(write=True)
    with pytest.raises(TypeError):
        result.violations["new"] = 1.0


def test_infeasible_evaluation_returns_violations_without_crashing(problem_data):
    result = evaluate_solution(
        [_plan(problem_data, speed=60.0)], [0], problem_data,
        {"target_surface_points": 12, "time_step": 1.0},
        {"root_tolerance": 1e-6, "interval_merge_tolerance": 1e-8},
    )
    assert not result.feasible
    assert result.violations["speed_low"] > 0
    np.testing.assert_array_equal(result.duration_by_missile, np.zeros(3))


def test_empty_cloud_solution_is_zero_and_empty_missile_selection_is_rejected(problem_data):
    result = evaluate_solution(
        [], [0], problem_data,
        {"target_surface_points": 12, "time_step": 1.0},
        {"root_tolerance": 1e-6, "interval_merge_tolerance": 1e-8},
    )
    assert result.feasible and result.sum_objective == 0.0
    with pytest.raises(ValueError, match="missile"):
        evaluate_solution([], [], problem_data, {"target_surface_points": 12, "time_step": 1.0}, {})


def test_evaluator_uses_joint_pointwise_cloud_union(monkeypatch, problem_data):
    import question1.evaluation as evaluation

    bombs = (BombPlan(0, 0.0, 0.0, 0), BombPlan(1, 1.0, 0.0, 0))
    plan = _plan(problem_data, bombs=bombs)
    monkeypatch.setattr(evaluation, "missile_hit_time", lambda index, data: 1.0)
    monkeypatch.setattr(evaluation, "sample_cylinder_surface", lambda profile, data: np.zeros((2, 3)))
    monkeypatch.setattr(evaluation, "visible_target_points", lambda missile, points, data: points)
    monkeypatch.setattr(evaluation, "missile_position", lambda t, index, data: np.array([10.0, 0.0, 0.0]))
    monkeypatch.setattr(evaluation, "smoke_center", lambda t, bomb, index, data: np.array([0.0, 0.0, 0.0]))
    monkeypatch.setattr(
        evaluation,
        "point_to_segments_distance",
        lambda centers, missile, points: (np.array([[0.0, 20.0], [20.0, 0.0]]), np.zeros((2, 2))),
    )
    result = evaluation.evaluate_solution(
        [plan], [0], problem_data,
        {"target_surface_points": 2, "time_step": 0.25},
        {"root_tolerance": 1e-6, "interval_merge_tolerance": 1e-8},
    )
    assert result.duration_by_missile[0] == pytest.approx(1.0)
    assert result.coverage_ratio_summary[0]["maximum"] == 1.0
