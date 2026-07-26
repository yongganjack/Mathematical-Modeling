"""Tests for missile, UAV, bomb, and smoke-cloud kinematics."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from math import pi

import numpy as np
import pytest

from question1.model import (
    BombPlan,
    DerivedBomb,
    UAVPlan,
    bomb_position,
    derive_bomb,
    direction_from_heading,
    max_fuse_delay,
    missile_hit_time,
    missile_position,
    smoke_center,
    uav_position,
)


def _q1_plan(*, speed: float = 120.0, delay: float = 3.6) -> UAVPlan:
    bomb = BombPlan(
        bomb_index=0,
        release_time=1.5,
        fuse_delay=delay,
        assigned_missile=0,
    )
    return UAVPlan(uav_index=0, heading_rad=pi, speed=speed, bombs=(bomb,))


def _derived_at(
    explosion_time: float,
    explosion_point: list[float],
    *,
    assigned_missile: int | None = 0,
) -> DerivedBomb:
    point = np.asarray(explosion_point, dtype=np.float64)
    return DerivedBomb(
        uav_index=0,
        bomb_index=0,
        assigned_missile=assigned_missile,
        release_time=explosion_time,
        fuse_delay=0.0,
        explosion_time=explosion_time,
        release_point=point,
        explosion_point=point,
    )


def test_direction_from_heading_returns_float64_unit_xy_vector() -> None:
    direction = direction_from_heading(pi / 2.0)

    assert direction.dtype == np.float64
    np.testing.assert_allclose(direction, [0.0, 1.0, 0.0], atol=1e-15)


@pytest.mark.parametrize("theta", [np.nan, np.inf, -np.inf])
def test_direction_from_heading_rejects_nonfinite_values(theta: float) -> None:
    with pytest.raises(ValueError, match="heading"):
        direction_from_heading(theta)


def test_missile_positions_match_initial_and_origin_endpoints(problem_data) -> None:
    hit_time = missile_hit_time(0, problem_data)

    np.testing.assert_array_equal(
        missile_position(0.0, 0, problem_data), problem_data.missile_init[0]
    )
    np.testing.assert_allclose(
        missile_position(hit_time, 0, problem_data), np.zeros(3), atol=1e-12
    )


def test_missile_position_has_documented_endpoint_tolerance(problem_data) -> None:
    hit_time = missile_hit_time(0, problem_data)

    np.testing.assert_allclose(
        missile_position(hit_time + 5e-10, 0, problem_data),
        np.zeros(3),
        atol=1e-12,
    )
    with pytest.raises(ValueError, match="time"):
        missile_position(hit_time + 2e-9, 0, problem_data)


@pytest.mark.parametrize("missile_index", [-1, 3, True])
def test_missile_functions_reject_invalid_indices(problem_data, missile_index) -> None:
    with pytest.raises((IndexError, TypeError), match="missile"):
        missile_hit_time(missile_index, problem_data)
    with pytest.raises((IndexError, TypeError), match="missile"):
        missile_position(0.0, missile_index, problem_data)


@pytest.mark.parametrize("time", [-1.0, np.nan, np.inf])
def test_missile_position_rejects_invalid_times(problem_data, time: float) -> None:
    with pytest.raises(ValueError, match="time"):
        missile_position(time, 0, problem_data)


@pytest.mark.parametrize("speed", [70.0, 140.0])
def test_uav_position_preserves_altitude_at_speed_boundaries(
    problem_data, speed: float
) -> None:
    plan = UAVPlan(uav_index=1, heading_rad=0.3, speed=speed, bombs=())

    position = uav_position(12.5, plan, problem_data)

    assert position.dtype == np.float64
    assert position[2] == problem_data.uav_init[1, 2]
    np.testing.assert_allclose(
        position,
        problem_data.uav_init[1]
        + speed * 12.5 * direction_from_heading(plan.heading_rad),
    )


@pytest.mark.parametrize("speed", [69.999, 140.001, np.nan, np.inf])
def test_uav_position_rejects_invalid_speed(problem_data, speed: float) -> None:
    plan = UAVPlan(uav_index=0, heading_rad=0.0, speed=speed, bombs=())

    with pytest.raises(ValueError, match="speed"):
        uav_position(1.0, plan, problem_data)


def test_uav_position_rejects_invalid_index_heading_and_time(problem_data) -> None:
    with pytest.raises(IndexError, match="UAV"):
        uav_position(
            0.0,
            UAVPlan(uav_index=5, heading_rad=0.0, speed=100.0, bombs=()),
            problem_data,
        )
    with pytest.raises(ValueError, match="heading"):
        uav_position(
            0.0,
            UAVPlan(uav_index=0, heading_rad=np.nan, speed=100.0, bombs=()),
            problem_data,
        )
    with pytest.raises(ValueError, match="time"):
        uav_position(
            -0.1,
            UAVPlan(uav_index=0, heading_rad=0.0, speed=100.0, bombs=()),
            problem_data,
        )


def test_q1_bomb_release_and_explosion_points(problem_data) -> None:
    plan = _q1_plan()

    derived = derive_bomb(plan, plan.bombs[0], problem_data)

    assert derived.explosion_time == pytest.approx(5.1)
    np.testing.assert_allclose(
        derived.release_point, [17620.0, 0.0, 1800.0], atol=1e-12
    )
    np.testing.assert_allclose(
        derived.explosion_point, [17188.0, 0.0, 1736.496], atol=1e-12
    )


def test_bomb_position_matches_release_and_explosion_endpoints(problem_data) -> None:
    plan = _q1_plan()
    derived = derive_bomb(plan, plan.bombs[0], problem_data)

    np.testing.assert_allclose(bomb_position(0.0, plan, derived, problem_data), derived.release_point)
    np.testing.assert_allclose(
        bomb_position(derived.fuse_delay, plan, derived, problem_data),
        derived.explosion_point,
    )


def test_explosion_height_matches_scalar_and_vector_formulas(problem_data) -> None:
    plan = _q1_plan(delay=2.75)
    bomb = plan.bombs[0]
    derived = derive_bomb(plan, bomb, problem_data)
    expected_height = (
        problem_data.uav_init[plan.uav_index, 2]
        - 0.5 * problem_data.gravity * bomb.fuse_delay**2
    )
    expected_vector = (
        problem_data.uav_init[plan.uav_index]
        + plan.speed
        * derived.explosion_time
        * direction_from_heading(plan.heading_rad)
        - np.array([0.0, 0.0, 0.5 * problem_data.gravity * bomb.fuse_delay**2])
    )

    assert derived.explosion_point[2] == pytest.approx(expected_height)
    np.testing.assert_allclose(derived.explosion_point, expected_vector)


def test_derive_bomb_rejects_fuse_beyond_ground_contact(problem_data) -> None:
    limit = max_fuse_delay(0, problem_data)
    plan = _q1_plan(delay=limit + 1e-6)

    with pytest.raises(ValueError, match="height|ground|fuse"):
        derive_bomb(plan, plan.bombs[0], problem_data)


@pytest.mark.parametrize(
    "bomb",
    [
        BombPlan(0, -0.1, 1.0, 0),
        BombPlan(0, 0.0, -0.1, 0),
        BombPlan(0, np.nan, 1.0, 0),
        BombPlan(-1, 0.0, 1.0, 0),
        BombPlan(0, 0.0, 1.0, 3),
    ],
)
def test_derive_bomb_validates_bomb_fields(problem_data, bomb: BombPlan) -> None:
    plan = UAVPlan(uav_index=0, heading_rad=0.0, speed=100.0, bombs=(bomb,))

    with pytest.raises((ValueError, IndexError), match="bomb|release|fuse|missile"):
        derive_bomb(plan, bomb, problem_data)


@pytest.mark.parametrize("elapsed", [-0.1, 3.6001, np.nan])
def test_bomb_position_rejects_elapsed_time_outside_fuse(
    problem_data, elapsed: float
) -> None:
    plan = _q1_plan()
    derived = derive_bomb(plan, plan.bombs[0], problem_data)

    with pytest.raises(ValueError, match="time"):
        bomb_position(elapsed, plan, derived, problem_data)


def test_smoke_center_is_none_before_explosion_and_starts_at_explosion(problem_data) -> None:
    plan = _q1_plan()
    derived = derive_bomb(plan, plan.bombs[0], problem_data)

    assert smoke_center(derived.explosion_time - 1e-6, derived, 0, problem_data) is None
    np.testing.assert_array_equal(
        smoke_center(derived.explosion_time, derived, 0, problem_data),
        derived.explosion_point,
    )


def test_smoke_center_expires_after_lifetime_inclusively(problem_data) -> None:
    derived = _derived_at(1.0, [1000.0, 0.0, 1000.0])

    assert smoke_center(21.0, derived, 0, problem_data) is not None
    assert smoke_center(21.0 + 2e-9, derived, 0, problem_data) is None


def test_smoke_center_expires_after_missile_hit_inclusively(problem_data) -> None:
    hit_time = missile_hit_time(0, problem_data)
    derived = _derived_at(hit_time - 1.0, [100.0, 0.0, 1000.0])

    assert smoke_center(hit_time, derived, 0, problem_data) is not None
    assert smoke_center(hit_time + 2e-9, derived, 0, problem_data) is None


def test_smoke_center_is_none_when_explosion_occurs_after_missile_hit(problem_data) -> None:
    hit_time = missile_hit_time(0, problem_data)
    derived = _derived_at(hit_time + 1.0, [100.0, 0.0, 1000.0])

    assert smoke_center(derived.explosion_time, derived, 0, problem_data) is None


def test_smoke_center_ground_boundary_is_valid_then_expires(problem_data) -> None:
    derived = _derived_at(1.0, [100.0, 0.0, 20.0])
    ground_time = 1.0 + (20.0 + problem_data.smoke_radius) / problem_data.smoke_sink_speed

    center = smoke_center(ground_time, derived, 0, problem_data)
    assert center is not None
    assert center[2] + problem_data.smoke_radius == pytest.approx(0.0)
    assert smoke_center(ground_time + 2e-9, derived, 0, problem_data) is None


def test_zero_sink_speed_disables_ground_expiry(problem_data) -> None:
    zero_sink_data = replace(problem_data, smoke_sink_speed=0.0)
    derived = _derived_at(1.0, [100.0, 0.0, -20.0])

    np.testing.assert_array_equal(
        smoke_center(1.0, derived, 0, zero_sink_data), derived.explosion_point
    )
    assert smoke_center(21.0, derived, 0, zero_sink_data) is not None
    assert smoke_center(21.0 + 2e-9, derived, 0, zero_sink_data) is None


def test_derived_bomb_is_frozen_and_owns_strongly_readonly_arrays() -> None:
    source = np.array([1.0, 2.0, 3.0])
    derived = DerivedBomb(0, 0, None, 1.0, 2.0, 3.0, source, source)

    source[0] = 99.0
    np.testing.assert_array_equal(derived.release_point, [1.0, 2.0, 3.0])
    with pytest.raises(FrozenInstanceError):
        derived.explosion_time = 4.0
    for point in (derived.release_point, derived.explosion_point):
        with pytest.raises(ValueError):
            point.setflags(write=True)


def test_plan_dataclasses_are_frozen() -> None:
    bomb = BombPlan(0, 1.0, 2.0, None)
    plan = UAVPlan(0, 0.0, 100.0, (bomb,))

    with pytest.raises(FrozenInstanceError):
        bomb.release_time = 3.0
    with pytest.raises(FrozenInstanceError):
        plan.speed = 120.0
