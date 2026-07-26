"""Tests for finite line-of-sight segment geometry."""

from __future__ import annotations

from math import sqrt

import numpy as np
import pytest

from question1.model import line_of_sight_blocked, point_to_segments_distance


def test_center_on_segment_has_zero_distance_and_interior_projection() -> None:
    distance, lambda_star = point_to_segments_distance(
        [0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]
    )

    assert distance.shape == (1, 1)
    assert lambda_star.shape == (1, 1)
    assert distance[0, 0] == pytest.approx(0.0)
    assert 0.0 < lambda_star[0, 0] < 1.0


def test_projection_before_missile_clamps_to_missile_endpoint() -> None:
    distance, lambda_star = point_to_segments_distance(
        [-1.0, 2.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]
    )

    assert lambda_star[0, 0] == 0.0
    assert distance[0, 0] == pytest.approx(sqrt(5.0))


def test_projection_beyond_target_uses_finite_target_endpoint() -> None:
    distance, lambda_star = point_to_segments_distance(
        [2.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]
    )

    assert lambda_star[0, 0] == 1.0
    assert distance[0, 0] == pytest.approx(sqrt(2.0))


def test_multiple_centers_and_targets_broadcast_to_matrix() -> None:
    centers = np.array([[0.5, 0.0, 0.0], [0.0, 2.0, 0.0]])
    targets = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]])

    distance, lambda_star = point_to_segments_distance(
        centers, [0.0, 0.0, 0.0], targets
    )

    assert distance.shape == (2, 3)
    assert lambda_star.shape == (2, 3)
    assert np.all(np.isfinite(distance))
    assert np.all((lambda_star >= 0.0) & (lambda_star <= 1.0))


def test_target_coincident_with_missile_is_rejected() -> None:
    with pytest.raises(ValueError, match="coincide|zero-length"):
        point_to_segments_distance(
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        )


@pytest.mark.parametrize(
    ("distance", "radius", "expected"),
    [
        (np.array([[10.0]]), 10.0, True),
        (np.array([[10.0 + 1e-9]]), 10.0, False),
    ],
)
def test_line_of_sight_blocking_includes_tangency(distance, radius, expected) -> None:
    blocked = line_of_sight_blocked(distance, radius)

    assert blocked.shape == (1, 1)
    assert bool(blocked[0, 0]) is expected


@pytest.mark.parametrize("radius", [0.0, -1.0, np.nan, np.inf, True])
def test_line_of_sight_blocking_rejects_invalid_radius(radius) -> None:
    with pytest.raises(ValueError, match="radius"):
        line_of_sight_blocked(np.array([[1.0]]), radius)


def test_geometry_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        point_to_segments_distance(
            [np.nan, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]
        )
    with pytest.raises(ValueError, match="finite"):
        line_of_sight_blocked(np.array([[np.nan]]), 10.0)
