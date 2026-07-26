"""测试确定性目标表面采样和可见性判断。"""

from __future__ import annotations

import numpy as np
import pytest

from question1.data_processing import (
    sample_cylinder_surface,
    visible_mask,
    visible_target_points,
)


def test_surface_sampling_is_legal_unique_and_strongly_readonly(
    quick_config, problem_data
) -> None:
    profile = quick_config["sampling"]["fast"]

    points = sample_cylinder_surface(profile, problem_data)

    assert points.ndim == 2 and points.shape[1] == 3
    assert len(points) > 0
    assert points.dtype == np.float64
    assert np.all(np.isfinite(points))
    assert len(np.unique(points, axis=0)) == len(points)

    center = problem_data.target_center_xy
    radius = problem_data.target_radius
    height = problem_data.target_height
    radial_distance = np.linalg.norm(points[:, :2] - center, axis=1)
    on_side = np.isclose(radial_distance, radius)
    on_top = np.isclose(points[:, 2], height) & (radial_distance <= radius + 1e-12)
    assert np.all(on_side | on_top)
    assert np.all((points[:, 2] >= 0.0) & (points[:, 2] <= height))
    assert np.any(np.all(np.isclose(points, [center[0], center[1], height]), axis=1))
    assert np.any(on_side & np.isclose(points[:, 2], 0.0))
    assert np.any(on_side & np.isclose(points[:, 2], height))

    with pytest.raises(ValueError):
        points.setflags(write=True)
    with pytest.raises(ValueError):
        points[0, 0] = 123.0


def test_surface_sampling_is_deterministic(quick_config, problem_data) -> None:
    profile = quick_config["sampling"]["verify"]

    first = sample_cylinder_surface(profile, problem_data)
    second = sample_cylinder_surface(profile, problem_data)

    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("invalid", [0, -1, 1.5, "72", True])
def test_surface_sampling_rejects_invalid_target_count(problem_data, invalid) -> None:
    with pytest.raises(ValueError, match="target_surface_points"):
        sample_cylinder_surface({"target_surface_points": invalid}, problem_data)


def test_visibility_filters_back_side_and_keeps_front_and_silhouette(
    problem_data,
) -> None:
    center_x, center_y = problem_data.target_center_xy
    radius = problem_data.target_radius
    missile = problem_data.missile_init[0]
    viewer_offset = missile[:2] - problem_data.target_center_xy
    offset_squared = float(np.dot(viewer_offset, viewer_offset))
    perpendicular = np.array([-viewer_offset[1], viewer_offset[0]])
    tangent_offset = (
        radius**2 * viewer_offset
        + radius * np.sqrt(offset_squared - radius**2) * perpendicular
    ) / offset_squared
    silhouette_xy = problem_data.target_center_xy + tangent_offset
    points = np.array(
        [
            [center_x + radius, center_y, 5.0],  # 正面，朝向导弹 M1
            [center_x - radius, center_y, 5.0],  # 背面
            [silhouette_xy[0], silhouette_xy[1], 5.0],
            [center_x, center_y, problem_data.target_height],  # 顶面
        ],
        dtype=np.float64,
    )

    mask = visible_mask(missile, points, problem_data)

    np.testing.assert_array_equal(mask, [True, False, True, True])
    np.testing.assert_array_equal(
        visible_target_points(missile, points, problem_data), points[mask]
    )


def test_sampled_visible_points_are_nonempty_subset(quick_config, problem_data) -> None:
    points = sample_cylinder_surface(
        quick_config["sampling"]["fast"], problem_data
    )

    visible = visible_target_points(problem_data.missile_init[0], points, problem_data)

    assert 0 < len(visible) <= len(points)
    assert len(visible) < len(points)


def test_visibility_rejects_empty_nonfinite_and_no_visible_points(problem_data) -> None:
    with pytest.raises(ValueError, match="empty"):
        visible_mask(problem_data.missile_init[0], np.empty((0, 3)), problem_data)

    invalid_points = np.array([[7.0, 200.0, np.nan]])
    with pytest.raises(ValueError, match="finite"):
        visible_target_points(problem_data.missile_init[0], invalid_points, problem_data)

    with pytest.raises(ValueError, match="finite"):
        visible_mask([np.inf, 0.0, 0.0], np.array([[7.0, 200.0, 5.0]]), problem_data)

    back_only = np.array([[-7.0, 200.0, 5.0]])
    with pytest.raises(ValueError, match="visible"):
        visible_target_points(problem_data.missile_init[0], back_only, problem_data)
