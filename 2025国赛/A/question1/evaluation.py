"""联合烟雾覆盖计算、可行性检查和区间评估。"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from question1.data_processing import (
    ProblemData,
    sample_cylinder_surface,
    visible_target_points,
)
from question1.model import (
    BombPlan,
    DerivedBomb,
    UAVPlan,
    derive_bomb,
    missile_hit_time,
    missile_position,
    point_to_segments_distance,
    smoke_center,
)


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
Interval = tuple[float, float]


def _blocked_matrix(blocked: Any, *, require_points: bool = True) -> BoolArray:
    array = np.asarray(blocked, dtype=np.bool_)
    if array.ndim != 2:
        raise ValueError("blocked 必须为二维 (烟云数, 点数) 数组")
    if require_points and array.shape[1] == 0:
        raise ValueError("blocked 必须包含至少一个目标点")
    return array


def point_coverage(blocked: Any) -> BoolArray:
    """返回每个目标点是否被至少一个烟雾云覆盖。"""

    array = _blocked_matrix(blocked)
    if array.shape[0] == 0:
        return np.zeros(array.shape[1], dtype=np.bool_)
    return np.asarray(np.any(array, axis=0), dtype=np.bool_)


def joint_blocked(blocked: Any) -> bool:
    """仅当所有烟雾云的并集覆盖了每一个目标点时返回 True。"""

    array = _blocked_matrix(blocked, require_points=False)
    if array.shape[0] == 0:
        return False
    if array.shape[1] == 0:
        raise ValueError("blocked 必须包含至少一个目标点")
    return bool(np.all(np.any(array, axis=0)))


def coverage_ratio(blocked: Any) -> float:
    """返回被烟雾云并集覆盖的目标点比例。"""

    covered = point_coverage(blocked)
    return float(np.mean(covered))


def standalone_blocked(blocked: Any) -> BoolArray:
    """返回每个单独的烟雾云是否覆盖了所有目标点。"""

    array = _blocked_matrix(blocked)
    return np.asarray(np.all(array, axis=1), dtype=np.bool_)


def _finite_nonnegative(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} 必须为有限非负数") from exc
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} 必须为有限非负数")
    return result


def _interval(value: Any) -> Interval:
    try:
        left, right = value
    except (TypeError, ValueError) as exc:
        raise ValueError("每个区间必须恰好包含两个端点") from exc
    left_value = _finite_nonnegative(left, "interval endpoint")
    right_value = _finite_nonnegative(right, "interval endpoint")
    if right_value < left_value:
        raise ValueError("区间右端点不能早于左端点")
    return left_value, right_value


def merge_intervals(intervals: Iterable[Sequence[float]], merge_tol: float = 0.0) -> list[Interval]:
    """合并相互重叠或间隔不超过 ``merge_tol`` 的闭区间。"""

    tolerance = _finite_nonnegative(merge_tol, "merge_tol")
    ordered = sorted((_interval(item) for item in intervals), key=lambda item: (item[0], item[1]))
    if not ordered:
        return []
    merged: list[Interval] = [ordered[0]]
    for left, right in ordered[1:]:
        previous_left, previous_right = merged[-1]
        if left <= previous_right + tolerance:
            merged[-1] = (previous_left, max(previous_right, right))
        else:
            merged.append((left, right))
    return merged


def interval_length(intervals: Iterable[Sequence[float]]) -> float:
    """返回经验证的闭区间的长度总和。"""

    return float(sum(right - left for left, right in map(_interval, intervals)))


def refine_boundary(
    left: float,
    right: float,
    margin_fn: Callable[[float], float],
    root_tol: float,
) -> float:
    """使用 Brent 方法精细求解有限标量边际函数在括号区间内的零点。"""

    left_value = _finite_nonnegative(left, "left")
    right_value = _finite_nonnegative(right, "right")
    if right_value < left_value:
        raise ValueError("右端点不能早于左端点")
    tolerance = _finite_nonnegative(root_tol, "root_tol")
    if tolerance == 0.0:
        raise ValueError("root_tol 必须为正数")
    if not callable(margin_fn):
        raise TypeError("margin_fn 必须为可调用对象")

    def checked_margin(time: float) -> float:
        value = float(margin_fn(float(time)))
        if not np.isfinite(value):
            raise ValueError("margin_fn 必须返回有限值")
        return value

    left_margin = checked_margin(left_value)
    right_margin = checked_margin(right_value)
    if left_margin == 0.0:
        return left_value
    if right_margin == 0.0:
        return right_value
    if left_margin * right_margin > 0.0:
        raise ValueError("margin_fn 的值未包围根")
    try:
        return float(
            brentq(
                checked_margin,
                left_value,
                right_value,
                xtol=tolerance,
                rtol=max(4.0 * np.finfo(float).eps, tolerance),
            )
        )
    except ValueError as exc:
        raise ValueError(f"精细求解括号边界失败: {exc}") from exc


def boolean_intervals(
    times: Sequence[float],
    states: Sequence[bool],
    margin_fn: Callable[[float], float],
    root_tol: float,
) -> list[Interval]:
    """构建真值状态闭区间，并对每个采样过渡点进行精细求解。"""

    time_array = np.asarray(times, dtype=np.float64)
    state_array = np.asarray(states, dtype=np.bool_)
    if time_array.ndim != 1 or state_array.ndim != 1 or len(time_array) != len(state_array):
        raise ValueError("times 和 states 必须为等长一维数组")
    if len(time_array) == 0:
        return []
    if not np.all(np.isfinite(time_array)) or np.any(time_array < 0.0) or np.any(np.diff(time_array) <= 0.0):
        raise ValueError("times 必须为有限、非负且严格递增")
    intervals: list[Interval] = []
    start = float(time_array[0]) if state_array[0] else None
    for index in range(len(time_array) - 1):
        if state_array[index] == state_array[index + 1]:
            continue
        boundary = refine_boundary(time_array[index], time_array[index + 1], margin_fn, root_tol)
        if state_array[index]:
            assert start is not None
            intervals.append((start, boundary))
            start = None
        else:
            start = boundary
    if state_array[-1]:
        assert start is not None
        intervals.append((start, float(time_array[-1])))
    return intervals


@dataclass(frozen=True, slots=True)
class FeasibilityReport:
    feasible: bool
    violations: Mapping[str, float]

    def __post_init__(self) -> None:
        frozen = {str(key): float(value) for key, value in self.violations.items() if float(value) > 0.0}
        object.__setattr__(self, "violations", MappingProxyType(frozen))
        object.__setattr__(self, "feasible", not frozen)


def _add_violation(violations: dict[str, float], name: str, magnitude: float = 1.0) -> None:
    if np.isfinite(magnitude) and magnitude > 0.0:
        violations[name] = violations.get(name, 0.0) + float(magnitude)
    elif not np.isfinite(magnitude):
        violations[name] = violations.get(name, 0.0) + 1.0


def _as_finite_float(value: Any) -> float | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if np.isfinite(result) else None


def check_uav_plan(plan: UAVPlan, data: ProblemData) -> FeasibilityReport:
    """返回单条 UAV 航线上结构化的物理约束和排序约束违规信息。"""

    if not isinstance(data, ProblemData):
        raise TypeError("data 必须为 ProblemData 实例")
    if not isinstance(plan, UAVPlan):
        return FeasibilityReport(False, {"invalid_uav_plan": 1.0})
    violations: dict[str, float] = {}
    valid_uav = isinstance(plan.uav_index, (int, np.integer)) and not isinstance(plan.uav_index, (bool, np.bool_)) and 0 <= int(plan.uav_index) < len(data.uav_init)
    if not valid_uav:
        _add_violation(violations, "invalid_uav_index")

    heading = _as_finite_float(plan.heading_rad)
    if heading is None:
        _add_violation(violations, "nonfinite_heading")
    speed = _as_finite_float(plan.speed)
    if speed is None:
        _add_violation(violations, "nonfinite_speed")
    else:
        lower, upper = map(float, data.uav_speed_bounds)
        _add_violation(violations, "speed_low", lower - speed)
        _add_violation(violations, "speed_high", speed - upper)

    if not isinstance(plan.bombs, tuple):
        _add_violation(violations, "invalid_bomb_container")
        bombs: tuple[Any, ...] = ()
    else:
        bombs = plan.bombs
    seen_indices: set[int] = set()
    release_times: list[float] = []
    original_release_times: list[float] = []
    for bomb in bombs:
        if not isinstance(bomb, BombPlan):
            _add_violation(violations, "invalid_bomb_plan")
            continue
        if isinstance(bomb.bomb_index, (bool, np.bool_)) or not isinstance(bomb.bomb_index, (int, np.integer)) or int(bomb.bomb_index) < 0:
            _add_violation(violations, "invalid_bomb_index")
        elif int(bomb.bomb_index) in seen_indices:
            _add_violation(violations, "duplicate_bomb_index")
        else:
            seen_indices.add(int(bomb.bomb_index))

        release = _as_finite_float(bomb.release_time)
        fuse = _as_finite_float(bomb.fuse_delay)
        if release is None:
            _add_violation(violations, "nonfinite_release_time")
        else:
            _add_violation(violations, "negative_release_time", -release)
            release_times.append(release)
            original_release_times.append(release)
        if fuse is None:
            _add_violation(violations, "nonfinite_fuse_delay")
            continue
        _add_violation(violations, "negative_fuse_delay", -fuse)
        if release is None:
            continue
        explosion_time = release + fuse
        if not np.isfinite(explosion_time):
            _add_violation(violations, "nonfinite_explosion_time")
            continue
        assigned = bomb.assigned_missile
        valid_assigned = assigned is None or (
            isinstance(assigned, (int, np.integer))
            and not isinstance(assigned, (bool, np.bool_))
            and 0 <= int(assigned) < len(data.missile_init)
        )
        if not valid_assigned:
            _add_violation(violations, "invalid_assigned_missile")
        else:
            deadline = (
                max(missile_hit_time(index, data) for index in range(len(data.missile_init)))
                if assigned is None
                else missile_hit_time(int(assigned), data)
            )
            _add_violation(
                violations,
                "late_explosion",
                explosion_time - deadline - 1e-9,
            )
        if valid_uav:
            height = float(data.uav_init[int(plan.uav_index), 2]) - 0.5 * float(data.gravity) * fuse**2
            _add_violation(
                violations,
                "negative_explosion_height",
                -height - 1e-9,
            )

    for first, second in zip(original_release_times, original_release_times[1:]):
        _add_violation(violations, "release_order", first - second)
    ordered = sorted(release_times)
    for first, second in zip(ordered, ordered[1:]):
        _add_violation(violations, "release_gap", float(data.min_release_interval) - (second - first))
    return FeasibilityReport(not violations, violations)


def _question_number(question_id: int | str) -> int:
    if isinstance(question_id, bool):
        raise ValueError("question_id 必须为 Q1 至 Q5 之一")
    text = str(question_id).strip().lower()
    if text.startswith("question"):
        text = text[8:]
    elif text.startswith("q"):
        text = text[1:]
    if not text.isdigit() or int(text) not in range(1, 6):
        raise ValueError("question_id 必须为 Q1 至 Q5 之一")
    return int(text)


def check_solution(plans: Iterable[UAVPlan], question_id: int | str, data: ProblemData) -> FeasibilityReport:
    """将航线违规与某个问题的炸弹数量规则进行合并。"""

    try:
        route_list = list(plans)
    except TypeError as exc:
        raise TypeError("plans 必须为可迭代对象") from exc
    question = _question_number(question_id)
    violations: dict[str, float] = {}
    counts = np.zeros(len(data.uav_init), dtype=np.int64)
    route_counts = np.zeros(len(data.uav_init), dtype=np.int64)
    invalid_route_bombs = 0
    for route in route_list:
        report = check_uav_plan(route, data)
        for name, magnitude in report.violations.items():
            _add_violation(violations, name, magnitude)
        if isinstance(route, UAVPlan) and isinstance(route.uav_index, (int, np.integer)) and not isinstance(route.uav_index, (bool, np.bool_)) and 0 <= int(route.uav_index) < len(data.uav_init):
            index = int(route.uav_index)
            route_counts[index] += 1
            if isinstance(route.bombs, tuple):
                counts[index] += len(route.bombs)
        elif isinstance(route, UAVPlan) and isinstance(route.bombs, tuple):
            invalid_route_bombs += len(route.bombs)
    _add_violation(violations, "multiple_routes_per_uav", float(np.maximum(route_counts - 1, 0).sum()))
    if question in (1, 2):
        wrong = abs(int(counts[0]) - 1) + int(counts[1:].sum()) + invalid_route_bombs
    elif question == 3:
        wrong = abs(int(counts[0]) - 3) + int(counts[1:].sum()) + invalid_route_bombs
    elif question == 4:
        wrong = int(np.abs(counts[:3] - 1).sum()) + int(counts[3:].sum()) + invalid_route_bombs
    else:
        wrong = int(np.maximum(counts - 3, 0).sum()) + invalid_route_bombs
    _add_violation(violations, "wrong_bomb_count", float(wrong))
    return FeasibilityReport(not violations, violations)


def decode_ordered_release_times(raw: Sequence[float], horizon: float, min_gap: float) -> FloatArray:
    """将原始投放时间值确定性投影到可行的排序锥中。"""

    values = np.asarray(raw, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("原始投放时间必须为有限一维数组")
    end = _finite_nonnegative(horizon, "horizon")
    gap = _finite_nonnegative(min_gap, "min_gap")
    count = len(values)
    if count == 0:
        return np.empty(0, dtype=np.float64)
    if (count - 1) * gap > end + 1e-12:
        raise ValueError("时间窗口无法满足所需的投放间隔")
    projected = np.sort(np.clip(values, 0.0, end))
    lower = np.arange(count, dtype=np.float64) * gap
    upper = end - np.arange(count - 1, -1, -1, dtype=np.float64) * gap
    projected = np.maximum(projected, lower)
    for index in range(count - 2, -1, -1):
        projected[index] = min(projected[index], projected[index + 1] - gap)
    projected = np.minimum(np.maximum(projected, lower), upper)
    return np.asarray(projected, dtype=np.float64)


def _strong_readonly_float64(values: Any, shape: tuple[int, ...] | None = None) -> FloatArray:
    source = np.asarray(values, dtype=np.float64)
    if shape is not None and source.shape != shape:
        raise ValueError(f"数组形状必须为 {shape}")
    if not np.all(np.isfinite(source)):
        raise ValueError("数组必须只包含有限值")
    return np.frombuffer(source.tobytes(order="C"), dtype=np.float64).reshape(source.shape)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, np.ndarray):
        return _strong_readonly_float64(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """单个解的不可变目标值和数值诊断信息。"""

    feasible: bool
    violations: Mapping[str, float]
    intervals_by_missile: Mapping[int, Sequence[Sequence[float]]]
    duration_by_missile: FloatArray
    sum_objective: float
    min_duration: float
    coverage_ratio_summary: Mapping[int, Mapping[str, float]]
    boundary_residuals: Mapping[int, Sequence[float]]
    evaluation_profile: Mapping[str, Any]
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        violation_copy = {
            str(key): float(value) for key, value in self.violations.items()
        }
        intervals = {
            int(index): tuple(_interval(interval) for interval in values)
            for index, values in self.intervals_by_missile.items()
        }
        residuals = {
            int(index): tuple(float(value) for value in values)
            for index, values in self.boundary_residuals.items()
        }
        object.__setattr__(self, "violations", MappingProxyType(violation_copy))
        object.__setattr__(self, "intervals_by_missile", MappingProxyType(intervals))
        object.__setattr__(
            self,
            "duration_by_missile",
            _strong_readonly_float64(self.duration_by_missile, (3,)),
        )
        object.__setattr__(self, "sum_objective", float(self.sum_objective))
        object.__setattr__(self, "min_duration", float(self.min_duration))
        object.__setattr__(self, "coverage_ratio_summary", _freeze(self.coverage_ratio_summary))
        object.__setattr__(self, "boundary_residuals", MappingProxyType(residuals))
        object.__setattr__(self, "evaluation_profile", _freeze(self.evaluation_profile))
        object.__setattr__(self, "diagnostics", _freeze(self.diagnostics))


def _positive_profile_number(profile: Mapping[str, Any], name: str) -> float:
    if not isinstance(profile, Mapping):
        raise TypeError("评估配置必须为映射类型")
    value = _as_finite_float(profile.get(name))
    if value is None or value <= 0.0:
        raise ValueError(f"{name} 必须为有限正数")
    return value


def _missile_selection(indices: Iterable[int], data: ProblemData) -> tuple[int, ...]:
    try:
        selected = tuple(indices)
    except TypeError as exc:
        raise TypeError("missile_indices 必须为可迭代对象") from exc
    if not selected:
        raise ValueError("missile_indices 不能为空")
    normalized: list[int] = []
    for value in selected:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError("导弹索引必须为整数")
        index = int(value)
        if index < 0 or index >= len(data.missile_init) or index >= 3:
            raise IndexError(f"导弹索引 {index} 超出范围")
        if index in normalized:
            raise ValueError("missile_indices 不能包含重复项")
        normalized.append(index)
    return tuple(normalized)


def _empty_evaluation(
    feasibility: FeasibilityReport,
    selected: tuple[int, ...],
    sampling_profile: Mapping[str, Any],
    numerical: Mapping[str, Any],
) -> EvaluationResult:
    return EvaluationResult(
        feasible=False,
        violations=feasibility.violations,
        intervals_by_missile={index: () for index in range(3)},
        duration_by_missile=np.zeros(3, dtype=np.float64),
        sum_objective=0.0,
        min_duration=0.0,
        coverage_ratio_summary={},
        boundary_residuals={index: () for index in selected},
        evaluation_profile={"sampling": dict(sampling_profile), "numerical": dict(numerical)},
        diagnostics={"reason": "infeasible", "evaluated_missiles": selected},
    )


def _event_times(derived_bombs: Sequence[DerivedBomb], missile_index: int, data: ProblemData) -> tuple[float, ...]:
    hit_time = missile_hit_time(missile_index, data)
    events = {0.0, hit_time}
    for bomb in derived_bombs:
        explosion = float(bomb.explosion_time)
        candidates = [explosion, explosion + float(data.smoke_lifetime)]
        if float(data.smoke_sink_speed) > 0.0:
            candidates.append(
                explosion
                + (float(bomb.explosion_point[2]) + float(data.smoke_radius))
                / float(data.smoke_sink_speed)
            )
        for candidate in candidates:
            if np.isfinite(candidate):
                events.add(min(max(float(candidate), 0.0), hit_time))
    return tuple(sorted(events))


def _base_times(events: Sequence[float], time_step: float) -> list[float]:
    samples: set[float] = set(float(value) for value in events)
    for left, right in zip(events, events[1:]):
        if right <= left:
            continue
        count = int(np.floor((right - left) / time_step))
        for offset in range(1, count + 1):
            value = left + offset * time_step
            if value < right:
                samples.add(float(value))
    return sorted(samples)


def evaluate_solution(
    plans: Iterable[UAVPlan],
    missile_indices: Iterable[int],
    data: ProblemData,
    sampling_profile: Mapping[str, Any],
    numerical: Mapping[str, Any],
    question_id: int | str = 5,
) -> EvaluationResult:
    """评估选定导弹的烟雾云并集覆盖区间。

    不可行的方案返回零目标值结果，并携带结构化的违规信息，
    以便优化循环可以在无需异常处理路径的情况下对其进行评分。
    """

    if not isinstance(data, ProblemData):
        raise TypeError("data 必须为 ProblemData 实例")
    selected = _missile_selection(missile_indices, data)
    time_step = _positive_profile_number(sampling_profile, "time_step")
    root_tolerance = _positive_profile_number(
        {"root_tolerance": numerical.get("root_tolerance", 1e-6)},
        "root_tolerance",
    )
    merge_tolerance = _finite_nonnegative(
        numerical.get("interval_merge_tolerance", 1e-8),
        "interval_merge_tolerance",
    )
    raw_depth = numerical.get("max_refinement_depth", 8)
    if isinstance(raw_depth, (bool, np.bool_)) or not isinstance(raw_depth, (int, np.integer)) or int(raw_depth) < 0:
        raise ValueError("max_refinement_depth 必须为非负整数")
    max_depth = int(raw_depth)
    route_list = list(plans)
    feasibility = check_solution(route_list, question_id, data)
    if not feasibility.feasible:
        return _empty_evaluation(feasibility, selected, sampling_profile, numerical)

    surface_points = sample_cylinder_surface(sampling_profile, data)
    derived_bombs = [
        derive_bomb(route, bomb, data)
        for route in route_list
        for bomb in route.bombs
    ]
    duration = np.zeros(3, dtype=np.float64)
    intervals_by_missile: dict[int, tuple[Interval, ...]] = {
        index: () for index in range(3)
    }
    coverage_summary: dict[int, Mapping[str, float]] = {}
    boundary_residuals: dict[int, tuple[float, ...]] = {}
    diagnostic_events: dict[int, tuple[float, ...]] = {}
    diagnostic_sample_counts: dict[int, int] = {}

    for missile_index in selected:
        events = _event_times(derived_bombs, missile_index, data)
        diagnostic_events[missile_index] = events
        cache: dict[float, tuple[bool, float, float]] = {}

        def state_margin_ratio(time: float) -> tuple[bool, float, float]:
            key = float(time)
            if key in cache:
                return cache[key]
            missile = missile_position(key, missile_index, data)
            visible = visible_target_points(missile, surface_points, data)
            centers = [
                center
                for bomb in derived_bombs
                if (center := smoke_center(key, bomb, missile_index, data)) is not None
            ]
            if not centers:
                result = (False, float(data.smoke_radius) + 1.0, 0.0)
            else:
                distances, _ = point_to_segments_distance(
                    np.asarray(centers, dtype=np.float64), missile, visible
                )
                blocked = np.asarray(distances <= float(data.smoke_radius), dtype=np.bool_)
                margin = float(np.max(np.min(distances, axis=0)) - float(data.smoke_radius))
                result = (joint_blocked(blocked), margin, coverage_ratio(blocked))
            cache[key] = result
            return result

        times = _base_times(events, time_step)
        for time in times:
            state_margin_ratio(time)

        def inspect_narrow_window(left: float, right: float, depth: int) -> None:
            if depth >= max_depth or right - left <= root_tolerance:
                return
            left_state, left_margin, _ = state_margin_ratio(left)
            right_state, right_margin, _ = state_margin_ratio(right)
            if left_state or right_state:
                return
            midpoint = 0.5 * (left + right)
            midpoint_state, midpoint_margin, _ = state_margin_ratio(midpoint)
            if midpoint_state or midpoint_margin < min(left_margin, right_margin):
                inspect_narrow_window(left, midpoint, depth + 1)
                inspect_narrow_window(midpoint, right, depth + 1)

        for left, right in zip(times, times[1:]):
            inspect_narrow_window(left, right, 0)
        sampled_times = sorted(cache)
        states = [cache[time][0] for time in sampled_times]

        def margin_fn(time: float) -> float:
            return state_margin_ratio(time)[1]

        raw_intervals = boolean_intervals(
            sampled_times, states, margin_fn, root_tolerance
        )
        merged = merge_intervals(raw_intervals, merge_tolerance)
        intervals_by_missile[missile_index] = tuple(merged)
        duration[missile_index] = interval_length(merged)
        ratios = np.asarray([cache[time][2] for time in sampled_times], dtype=np.float64)
        coverage_summary[missile_index] = {
            "minimum": float(np.min(ratios)),
            "maximum": float(np.max(ratios)),
            "mean": float(np.mean(ratios)),
            "sample_count": float(len(ratios)),
        }
        residuals: list[float] = []
        for left, right in merged:
            residuals.extend((abs(margin_fn(left)), abs(margin_fn(right))))
        boundary_residuals[missile_index] = tuple(residuals)
        diagnostic_sample_counts[missile_index] = len(sampled_times)

    selected_duration = duration[np.asarray(selected, dtype=np.int64)]
    return EvaluationResult(
        feasible=True,
        violations={},
        intervals_by_missile=intervals_by_missile,
        duration_by_missile=duration,
        sum_objective=float(np.sum(selected_duration)),
        min_duration=float(np.min(selected_duration)),
        coverage_ratio_summary=coverage_summary,
        boundary_residuals=boundary_residuals,
        evaluation_profile={"sampling": dict(sampling_profile), "numerical": dict(numerical)},
        diagnostics={
            "events_by_missile": diagnostic_events,
            "sample_count_by_missile": diagnostic_sample_counts,
            "derived_bomb_count": len(derived_bombs),
            "evaluated_missiles": selected,
        },
    )
