"""Kinematics for missiles, UAVs, released bombs, and smoke clouds."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin, sqrt
from typing import Any

import numpy as np
from numpy.typing import NDArray

from question1.data_processing import ProblemData


FloatArray = NDArray[np.float64]

# Times no farther than this outside a missile flight interval are clamped to
# its endpoint. Other physical windows use exact inclusive boundaries.
MISSILE_TIME_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class BombPlan:
    """Decision variables for one smoke bomb carried by a UAV."""

    bomb_index: int
    release_time: float
    fuse_delay: float
    assigned_missile: int | None


@dataclass(frozen=True, slots=True)
class UAVPlan:
    """Constant-altitude, constant-heading flight plan for one UAV."""

    uav_index: int
    heading_rad: float
    speed: float
    bombs: tuple[BombPlan, ...]


def _readonly_float64_vector(values: Any, name: str) -> FloatArray:
    source = np.asarray(values, dtype=np.float64)
    if source.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    if not np.all(np.isfinite(source)):
        raise ValueError(f"{name} must contain only finite values")
    return np.frombuffer(source.tobytes(order="C"), dtype=np.float64).reshape(3)


@dataclass(frozen=True, slots=True)
class DerivedBomb:
    """Validated times and coordinates derived from a bomb decision."""

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
        raise TypeError("data must be a ProblemData instance")


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite number")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be a finite number")
    return numeric


def _index(value: Any, size: int, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{label} index must be an integer")
    index = int(value)
    if index < 0 or index >= size:
        raise IndexError(f"{label} index {index} is out of range")
    return index


def _nonnegative_time(value: Any, name: str) -> float:
    numeric = _finite_float(value, name)
    if numeric < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return numeric


def direction_from_heading(theta: float) -> FloatArray:
    """Return the horizontal unit vector for a finite heading angle."""

    heading = _finite_float(theta, "heading")
    return np.array([cos(heading), sin(heading), 0.0], dtype=np.float64)


def missile_hit_time(missile_index: int, data: ProblemData) -> float:
    """Return the time at which a missile flying toward the origin arrives."""

    _require_problem_data(data)
    index = _index(missile_index, len(data.missile_init), "missile")
    speed = _finite_float(data.missile_speed, "missile speed")
    if speed <= 0.0:
        raise ValueError("missile speed must be positive")
    initial = np.asarray(data.missile_init[index], dtype=np.float64)
    if initial.shape != (3,) or not np.all(np.isfinite(initial)):
        raise ValueError("missile initial position must be a finite 3-vector")
    return float(np.linalg.norm(initial) / speed)


def missile_position(t: float, missile_index: int, data: ProblemData) -> FloatArray:
    """Return a missile position on ``[0, hit_time]``.

    Values within :data:`MISSILE_TIME_TOLERANCE` of either endpoint are
    clamped to that endpoint; values farther outside the interval are rejected.
    """

    _require_problem_data(data)
    index = _index(missile_index, len(data.missile_init), "missile")
    time = _finite_float(t, "time")
    hit_time = missile_hit_time(index, data)
    if time < -MISSILE_TIME_TOLERANCE or time > hit_time + MISSILE_TIME_TOLERANCE:
        raise ValueError(f"time must lie in [0, {hit_time}]")
    clamped_time = min(max(time, 0.0), hit_time)
    fraction_remaining = 1.0 - clamped_time / hit_time
    if clamped_time == hit_time:
        fraction_remaining = 0.0
    return np.asarray(data.missile_init[index], dtype=np.float64) * fraction_remaining


def _validate_uav_plan(uav: UAVPlan, data: ProblemData) -> int:
    if not isinstance(uav, UAVPlan):
        raise TypeError("uav must be a UAVPlan")
    index = _index(uav.uav_index, len(data.uav_init), "UAV")
    direction_from_heading(uav.heading_rad)
    speed = _finite_float(uav.speed, "UAV speed")
    lower, upper = data.uav_speed_bounds
    lower_value = _finite_float(lower, "UAV speed lower bound")
    upper_value = _finite_float(upper, "UAV speed upper bound")
    if speed < lower_value or speed > upper_value:
        raise ValueError(f"UAV speed must lie in [{lower_value}, {upper_value}]")
    return index


def uav_position(t: float, uav_plan: UAVPlan, data: ProblemData) -> FloatArray:
    """Return the constant-altitude straight-line position of a UAV."""

    _require_problem_data(data)
    index = _validate_uav_plan(uav_plan, data)
    time = _nonnegative_time(t, "time")
    direction = direction_from_heading(uav_plan.heading_rad)
    return (
        np.asarray(data.uav_init[index], dtype=np.float64)
        + float(uav_plan.speed) * time * direction
    )


def _validate_bomb_plan(bomb: BombPlan, data: ProblemData) -> tuple[float, float]:
    if not isinstance(bomb, BombPlan):
        raise TypeError("bomb must be a BombPlan")
    if isinstance(bomb.bomb_index, (bool, np.bool_)) or not isinstance(
        bomb.bomb_index, (int, np.integer)
    ):
        raise ValueError("bomb index must be a non-negative integer")
    if int(bomb.bomb_index) < 0:
        raise ValueError("bomb index must be a non-negative integer")
    release_time = _nonnegative_time(bomb.release_time, "release time")
    fuse_delay = _nonnegative_time(bomb.fuse_delay, "fuse delay")
    if bomb.assigned_missile is not None:
        _index(bomb.assigned_missile, len(data.missile_init), "assigned missile")
    return release_time, fuse_delay


def _validate_plan_bomb_container(uav: UAVPlan) -> None:
    if not isinstance(uav.bombs, tuple):
        raise TypeError("UAV plan bombs must be a tuple")
    if not all(isinstance(item, BombPlan) for item in uav.bombs):
        raise TypeError("UAV plan bombs must contain only BombPlan instances")


def derive_bomb(uav: UAVPlan, bomb: BombPlan, data: ProblemData) -> DerivedBomb:
    """Derive release and explosion coordinates from a UAV and bomb plan."""

    _require_problem_data(data)
    uav_index = _validate_uav_plan(uav, data)
    _validate_plan_bomb_container(uav)
    release_time, fuse_delay = _validate_bomb_plan(bomb, data)
    explosion_time = release_time + fuse_delay
    if not np.isfinite(explosion_time):
        raise ValueError("explosion time must be finite")

    fuse_limit = max_fuse_delay(uav_index, data)
    if fuse_delay > fuse_limit:
        raise ValueError("fuse delay would place the explosion below ground")

    direction = direction_from_heading(uav.heading_rad)
    initial = np.asarray(data.uav_init[uav_index], dtype=np.float64)
    release_point = initial + float(uav.speed) * release_time * direction
    explosion_point = initial + float(uav.speed) * explosion_time * direction
    explosion_point[2] -= 0.5 * float(data.gravity) * fuse_delay**2
    if explosion_point[2] < 0.0:
        # At the analytically exact fuse limit, floating-point roundoff can
        # produce a tiny negative altitude.
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
        raise TypeError("derived must be a DerivedBomb")
    _index(derived.uav_index, len(data.uav_init), "UAV")
    if derived.assigned_missile is not None:
        _index(derived.assigned_missile, len(data.missile_init), "assigned missile")
    _nonnegative_time(derived.release_time, "release time")
    _nonnegative_time(derived.fuse_delay, "fuse delay")
    _nonnegative_time(derived.explosion_time, "explosion time")
    if derived.release_point.shape != (3,) or not np.all(np.isfinite(derived.release_point)):
        raise ValueError("release point must be a finite 3-vector")
    if derived.explosion_point.shape != (3,) or not np.all(
        np.isfinite(derived.explosion_point)
    ):
        raise ValueError("explosion point must be a finite 3-vector")


def bomb_position(
    s: float,
    uav: UAVPlan,
    derived: DerivedBomb,
    data: ProblemData,
) -> FloatArray:
    """Return a bomb position ``s`` seconds after release."""

    _require_problem_data(data)
    _validate_uav_plan(uav, data)
    _validate_derived_bomb(derived, data)
    if uav.uav_index != derived.uav_index:
        raise ValueError("UAV plan and derived bomb must have the same UAV index")
    elapsed = _finite_float(s, "time since release")
    if elapsed < 0.0 or elapsed > derived.fuse_delay:
        raise ValueError(f"time since release must lie in [0, {derived.fuse_delay}]")
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
    """Return the sinking smoke-cloud center while its effective window is open."""

    _require_problem_data(data)
    _validate_derived_bomb(derived, data)
    index = _index(missile_index, len(data.missile_init), "missile")
    time = _finite_float(t, "time")
    explosion_time = float(derived.explosion_time)
    if time < explosion_time:
        return None
    if time == explosion_time:
        return np.array(derived.explosion_point, dtype=np.float64, copy=True)

    lifetime = _finite_float(data.smoke_lifetime, "smoke lifetime")
    if lifetime < 0.0:
        raise ValueError("smoke lifetime must be non-negative")
    radius = _finite_float(data.smoke_radius, "smoke radius")
    if radius < 0.0:
        raise ValueError("smoke radius must be non-negative")
    sink_speed = _finite_float(data.smoke_sink_speed, "smoke sink speed")
    if sink_speed < 0.0:
        raise ValueError("smoke sink speed must be non-negative")

    lifetime_end = explosion_time + lifetime
    missile_end = missile_hit_time(index, data)
    if sink_speed == 0.0:
        ground_end = float("inf")
    else:
        ground_end = explosion_time + (
            float(derived.explosion_point[2]) + radius
        ) / sink_speed
    effective_end = min(lifetime_end, missile_end, ground_end)
    if time > effective_end:
        return None

    center = np.array(derived.explosion_point, dtype=np.float64, copy=True)
    center[2] -= sink_speed * (time - explosion_time)
    return center


def max_fuse_delay(uav_index: int, data: ProblemData) -> float:
    """Return the longest free-fall delay before a UAV-height bomb hits ground."""

    _require_problem_data(data)
    index = _index(uav_index, len(data.uav_init), "UAV")
    altitude = _finite_float(data.uav_init[index, 2], "UAV initial altitude")
    gravity = _finite_float(data.gravity, "gravity")
    if altitude < 0.0:
        raise ValueError("UAV initial altitude must be non-negative")
    if gravity <= 0.0:
        raise ValueError("gravity must be positive")
    return sqrt(2.0 * altitude / gravity)
