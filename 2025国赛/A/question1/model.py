"""导弹、UAV、投放炸弹和烟雾云的运动学模型。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from math import cos, sin, sqrt
from typing import Any

import numpy as np
from numpy.typing import NDArray

from question1.data_processing import ProblemData

logger = logging.getLogger(__name__)


FloatArray = NDArray[np.float64]

# 距离导弹飞行区间的端点不超过该容差的时刻会被钳制到端点上。
# 其他物理窗口使用精确的包含边界。
MISSILE_TIME_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class BombPlan:
    """UAV 携带的单个烟雾弹的决策变量。"""

    bomb_index: int
    release_time: float
    fuse_delay: float
    assigned_missile: int | None


@dataclass(frozen=True, slots=True)
class UAVPlan:
    """单架 UAV 的定高、定向飞行计划。"""

    uav_index: int
    heading_rad: float
    speed: float
    bombs: tuple[BombPlan, ...]


def _readonly_float64_vector(values: Any, name: str) -> FloatArray:
    source = np.asarray(values, dtype=np.float64)
    if source.shape != (3,):
        raise ValueError(f"{name} 形状必须为 (3,)")
    if not np.all(np.isfinite(source)):
        raise ValueError(f"{name} 必须只包含有限值")
    return np.frombuffer(source.tobytes(order="C"), dtype=np.float64).reshape(3)


@dataclass(frozen=True, slots=True)
class DerivedBomb:
    """由投弹决策推导出的经验证的时间和坐标信息。"""

    uav_index: int
    bomb_index: int
    assigned_missile: int | None
    release_time: float
    fuse_delay: float
    explosion_time: float
    release_point: FloatArray
    explosion_point: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "release_point",
            _readonly_float64_vector(self.release_point, "release_point"),
        )
        object.__setattr__(
            self,
            "explosion_point",
            _readonly_float64_vector(self.explosion_point, "explosion_point"),
        )


def _require_problem_data(data: ProblemData) -> None:
    if not isinstance(data, ProblemData):
        raise TypeError("data 必须为 ProblemData 实例")


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} 必须为有限数")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} 必须为有限数") from exc
    if not np.isfinite(numeric):
        raise ValueError(f"{name} 必须为有限数")
    return numeric


def _index(value: Any, size: int, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{label} 索引必须为整数")
    index = int(value)
    if index < 0 or index >= size:
        raise IndexError(f"{label} 索引 {index} 超出范围")
    return index


def _nonnegative_time(value: Any, name: str) -> float:
    numeric = _finite_float(value, name)
    if numeric < 0.0:
        raise ValueError(f"{name} 必须为非负数")
    return numeric


def direction_from_heading(theta: float) -> FloatArray:
    """返回给定航向角的有穷水平单位方向向量。"""

    heading = _finite_float(theta, "heading")
    return np.array([cos(heading), sin(heading), 0.0], dtype=np.float64)


def missile_hit_time(missile_index: int, data: ProblemData) -> float:
    """返回导弹飞向原点的到达时间。"""

    _require_problem_data(data)
    logger.debug("计算导弹 %d 的命中时间", missile_index)
    index = _index(missile_index, len(data.missile_init), "missile")
    speed = _finite_float(data.missile_speed, "missile speed")
    if speed <= 0.0:
        raise ValueError("导弹速度必须为正数")
    initial = np.asarray(data.missile_init[index], dtype=np.float64)
    if initial.shape != (3,) or not np.all(np.isfinite(initial)):
        raise ValueError("导弹初始位置必须为有限三维向量")
    return float(np.linalg.norm(initial) / speed)


def missile_position(t: float, missile_index: int, data: ProblemData) -> FloatArray:
    """返回导弹在区间 ``[0, hit_time]`` 上的位置。

    与 :data:`MISSILE_TIME_TOLERANCE` 端点的时间差在容差内的值会被钳制到端点；
    远超区间外的值会被拒绝。
    """

    _require_problem_data(data)
    index = _index(missile_index, len(data.missile_init), "missile")
    time = _finite_float(t, "time")
    hit_time = missile_hit_time(index, data)
    if time < -MISSILE_TIME_TOLERANCE or time > hit_time + MISSILE_TIME_TOLERANCE:
        raise ValueError(f"time 必须在 [0, {hit_time}] 范围内")
    clamped_time = min(max(time, 0.0), hit_time)
    fraction_remaining = 1.0 - clamped_time / hit_time
    if clamped_time == hit_time:
        fraction_remaining = 0.0
    return np.asarray(data.missile_init[index], dtype=np.float64) * fraction_remaining


def _validate_uav_plan(uav: UAVPlan, data: ProblemData) -> int:
    if not isinstance(uav, UAVPlan):
        raise TypeError("uav 必须为 UAVPlan 类型")
    index = _index(uav.uav_index, len(data.uav_init), "UAV")
    direction_from_heading(uav.heading_rad)
    speed = _finite_float(uav.speed, "UAV speed")
    lower, upper = data.uav_speed_bounds
    lower_value = _finite_float(lower, "UAV speed lower bound")
    upper_value = _finite_float(upper, "UAV speed upper bound")
    if speed < lower_value or speed > upper_value:
        raise ValueError(f"UAV 速度必须在 [{lower_value}, {upper_value}] 范围内")
    return index


def uav_position(t: float, uav_plan: UAVPlan, data: ProblemData) -> FloatArray:
    """返回 UAV 的定高直线飞行位置。"""

    _require_problem_data(data)
    logger.debug("计算 UAV %d 在时刻 t=%.3f 的位置", uav_plan.uav_index, t)
    index = _validate_uav_plan(uav_plan, data)
    time = _nonnegative_time(t, "time")
    direction = direction_from_heading(uav_plan.heading_rad)
    return (
        np.asarray(data.uav_init[index], dtype=np.float64)
        + float(uav_plan.speed) * time * direction
    )


def _validate_bomb_plan(bomb: BombPlan, data: ProblemData) -> tuple[float, float]:
    if not isinstance(bomb, BombPlan):
        raise TypeError("bomb 必须为 BombPlan 类型")
    if isinstance(bomb.bomb_index, (bool, np.bool_)) or not isinstance(
        bomb.bomb_index, (int, np.integer)
    ):
        raise ValueError("炸弹索引必须为非负整数")
    if int(bomb.bomb_index) < 0:
        raise ValueError("炸弹索引必须为非负整数")
    release_time = _nonnegative_time(bomb.release_time, "release time")
    fuse_delay = _nonnegative_time(bomb.fuse_delay, "fuse delay")
    if bomb.assigned_missile is not None:
        _index(bomb.assigned_missile, len(data.missile_init), "assigned missile")
    return release_time, fuse_delay


def _validate_plan_bomb_container(uav: UAVPlan) -> None:
    if not isinstance(uav.bombs, tuple):
        raise TypeError("UAV 方案的 bombs 必须为元组")
    if not all(isinstance(item, BombPlan) for item in uav.bombs):
        raise TypeError("UAV 方案的 bombs 必须只包含 BombPlan 实例")


def derive_bomb(uav: UAVPlan, bomb: BombPlan, data: ProblemData) -> DerivedBomb:
    """根据 UAV 和投弹计划推算投放点和爆炸点坐标。"""

    _require_problem_data(data)
    logger.debug("推算炸弹: UAV=%d, bomb_index=%d, release_time=%.3f, fuse_delay=%.3f",
                 uav.uav_index, bomb.bomb_index, bomb.release_time, bomb.fuse_delay)
    uav_index = _validate_uav_plan(uav, data)
    _validate_plan_bomb_container(uav)
    release_time, fuse_delay = _validate_bomb_plan(bomb, data)
    explosion_time = release_time + fuse_delay
    if not np.isfinite(explosion_time):
        raise ValueError("爆炸时间必须为有限值")

    fuse_limit = max_fuse_delay(uav_index, data)
    if fuse_delay > fuse_limit:
        raise ValueError("引信延迟将使爆炸点位于地面以下")

    direction = direction_from_heading(uav.heading_rad)
    initial = np.asarray(data.uav_init[uav_index], dtype=np.float64)
    release_point = initial + float(uav.speed) * release_time * direction
    explosion_point = initial + float(uav.speed) * explosion_time * direction
    explosion_point[2] -= 0.5 * float(data.gravity) * fuse_delay**2
    if explosion_point[2] < 0.0:
        # 在解析精确的引信极限处，浮点舍入误差可能会产生微小的负高度值。
        explosion_point[2] = 0.0

    return DerivedBomb(
        uav_index=uav_index,
        bomb_index=int(bomb.bomb_index),
        assigned_missile=(
            None if bomb.assigned_missile is None else int(bomb.assigned_missile)
        ),
        release_time=release_time,
        fuse_delay=fuse_delay,
        explosion_time=explosion_time,
        release_point=release_point,
        explosion_point=explosion_point,
    )


def _validate_derived_bomb(derived: DerivedBomb, data: ProblemData) -> None:
    if not isinstance(derived, DerivedBomb):
        raise TypeError("derived 必须为 DerivedBomb 类型")
    _index(derived.uav_index, len(data.uav_init), "UAV")
    if derived.assigned_missile is not None:
        _index(derived.assigned_missile, len(data.missile_init), "assigned missile")
    _nonnegative_time(derived.release_time, "release time")
    _nonnegative_time(derived.fuse_delay, "fuse delay")
    _nonnegative_time(derived.explosion_time, "explosion time")
    if derived.release_point.shape != (3,) or not np.all(np.isfinite(derived.release_point)):
        raise ValueError("投放点必须为有限三维向量")
    if derived.explosion_point.shape != (3,) or not np.all(
        np.isfinite(derived.explosion_point)
    ):
        raise ValueError("爆炸点必须为有限三维向量")


def bomb_position(
    s: float,
    uav: UAVPlan,
    derived: DerivedBomb,
    data: ProblemData,
) -> FloatArray:
    """返回炸弹投放后经过 ``s`` 秒时的位置。"""

    _require_problem_data(data)
    logger.debug("计算炸弹 %d (UAV=%d) 在投放后 s=%.3f 秒的位置", derived.bomb_index, derived.uav_index, s)
    _validate_uav_plan(uav, data)
    _validate_derived_bomb(derived, data)
    if uav.uav_index != derived.uav_index:
        raise ValueError("UAV 方案与衍生炸弹必须具有相同的 UAV 索引")
    elapsed = _finite_float(s, "time since release")
    if elapsed < 0.0 or elapsed > derived.fuse_delay:
        raise ValueError(f"投放后时间必须在 [0, {derived.fuse_delay}] 范围内")
    direction = direction_from_heading(uav.heading_rad)
    displacement = float(uav.speed) * elapsed * direction
    displacement[2] -= 0.5 * float(data.gravity) * elapsed**2
    return np.asarray(derived.release_point, dtype=np.float64) + displacement


def smoke_center(
    t: float,
    derived: DerivedBomb,
    missile_index: int,
    data: ProblemData,
) -> FloatArray | None:
    """返回正在下沉的烟雾云中心位置，仅在其有效窗口内返回。

    参数:
        t: 当前时刻。
        derived: 已推导的炸弹数据。
        missile_index: 关联的导弹索引。
        data: 题目数据。

    返回:
        若烟雾云有效则返回其中心位置 (FloatArray)，否则返回 None。
    """

    _require_problem_data(data)
    logger.debug("计算烟雾中心: missile=%d, t=%.3f, explosion_time=%.3f", missile_index, t, derived.explosion_time)
    _validate_derived_bomb(derived, data)
    index = _index(missile_index, len(data.missile_init), "missile")
    time = _finite_float(t, "time")
    explosion_time = float(derived.explosion_time)
    lifetime = _finite_float(data.smoke_lifetime, "smoke lifetime")
    if lifetime < 0.0:
        raise ValueError("烟雾寿命必须为非负数")
    radius = _finite_float(data.smoke_radius, "smoke radius")
    if radius < 0.0:
        raise ValueError("烟雾半径必须为非负数")
    sink_speed = _finite_float(data.smoke_sink_speed, "smoke sink speed")
    if sink_speed < 0.0:
        raise ValueError("烟雾下沉速度必须为非负数")

    lifetime_end = explosion_time + lifetime
    missile_end = missile_hit_time(index, data)
    if sink_speed == 0.0:
        ground_end = float("inf")
    else:
        ground_end = explosion_time + (
            float(derived.explosion_point[2]) + radius
        ) / sink_speed
    effective_end = min(lifetime_end, missile_end, ground_end)
    if time < explosion_time or time > effective_end:
        return None

    center = np.array(derived.explosion_point, dtype=np.float64, copy=True)
    center[2] -= sink_speed * (time - explosion_time)
    return center


def max_fuse_delay(uav_index: int, data: ProblemData) -> float:
    """返回 UAV 从当前高度自由落体到地面的最长引信延迟时间。"""

    _require_problem_data(data)
    logger.debug("计算 UAV %d 的最大引信延迟", uav_index)
    index = _index(uav_index, len(data.uav_init), "UAV")
    altitude = _finite_float(data.uav_init[index, 2], "UAV initial altitude")
    gravity = _finite_float(data.gravity, "gravity")
    if altitude < 0.0:
        raise ValueError("UAV 初始高度必须为非负数")
    if gravity <= 0.0:
        raise ValueError("重力加速度必须为正数")
    return sqrt(2.0 * altitude / gravity)


def _point_matrix(values: Any, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        if array.shape != (3,):
            raise ValueError(f"{name} 最后一维必须为 3")
        array = array.reshape(1, 3)
    elif array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} 形状必须为 (3,) 或 (N, 3)")
    if len(array) == 0:
        raise ValueError(f"{name} 不能为空")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} 必须只包含有限值")
    return array


def point_to_segments_distance(
    smoke_centers: Any, missile_pos: Any, target_points: Any
) -> tuple[FloatArray, FloatArray]:
    """计算烟雾中心到导弹-目标有限线段之间的投影距离。

    即使输入仅包含单个点，结果仍保持二维 ``(centers, targets)`` 形状。

    参数:
        smoke_centers: 烟雾中心坐标，形状为 (N, 3) 或 (3,)。
        missile_pos: 导弹位置，形状为 (3,)。
        target_points: 目标点坐标，形状为 (M, 3) 或 (3,)。

    返回:
        (distances, lambda_star) 元组:
        - distances: 形状 (N, M)，每对烟雾中心到目标点的距离。
        - lambda_star: 形状 (N, M)，在线段上的投影参数。
    """

    centers = _point_matrix(smoke_centers, "smoke_centers")
    targets = _point_matrix(target_points, "target_points")
    missile_array = np.asarray(missile_pos, dtype=np.float64)
    if missile_array.shape == (1, 3):
        missile_array = missile_array[0]
    if missile_array.shape != (3,):
        raise ValueError("missile_pos 形状必须为 (3,)")
    if not np.all(np.isfinite(missile_array)):
        raise ValueError("missile_pos 必须只包含有限值")

    segment_vectors = targets - missile_array
    denominator = np.einsum("nj,nj->n", segment_vectors, segment_vectors)
    if np.any(denominator <= 0.0):
        raise ValueError("目标点不能与 missile_pos 重合")

    center_offsets = centers[:, None, :] - missile_array
    projection = np.einsum(
        "bnj,nj->bn", center_offsets, segment_vectors
    ) / denominator[None, :]
    lambda_star = np.clip(projection, 0.0, 1.0)
    closest_points = (
        missile_array
        + lambda_star[:, :, None] * segment_vectors[None, :, :]
    )
    differences = centers[:, None, :] - closest_points
    distance = np.linalg.norm(differences, axis=2)
    if not np.all(np.isfinite(distance)) or not np.all(np.isfinite(lambda_star)):
        raise ValueError("视线几何计算产生了非有限值")
    return (
        np.asarray(distance, dtype=np.float64),
        np.asarray(lambda_star, dtype=np.float64),
    )


def line_of_sight_blocked(distance: Any, smoke_radius: Any) -> NDArray[np.bool_]:
    """判断有限线段到各点的距离是否与烟雾球体相交。"""

    if isinstance(smoke_radius, (bool, np.bool_)) or not np.isscalar(smoke_radius):
        raise ValueError("烟雾半径必须为有限正数")
    try:
        radius = float(smoke_radius)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("烟雾半径必须为有限正数") from exc
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("烟雾半径必须为有限正数")

    distances = np.asarray(distance, dtype=np.float64)
    if not np.all(np.isfinite(distances)):
        raise ValueError("距离必须只包含有限值")
    if np.any(distances < 0.0):
        raise ValueError("距离必须为非负数")
    return np.asarray(distances <= radius, dtype=np.bool_)
