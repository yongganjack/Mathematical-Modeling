import numpy as np
import pytest

from question1.evaluation import coverage_ratio, joint_blocked, point_coverage, standalone_blocked


def test_joint_quantifier_requires_same_cloud_to_cover_each_point():
    blocked = np.array([[True, False], [False, True]])
    np.testing.assert_array_equal(point_coverage(blocked), [True, True])
    assert joint_blocked(blocked)
    np.testing.assert_array_equal(standalone_blocked(blocked), [False, False])
    assert coverage_ratio(blocked) == 1.0
    assert not joint_blocked(blocked[[0]])


def test_empty_clouds_and_points_have_explicit_semantics():
    assert not np.any(point_coverage(np.empty((0, 3), dtype=bool)))
    assert joint_blocked(np.empty((0, 0), dtype=bool)) is False
    assert coverage_ratio(np.empty((0, 3), dtype=bool)) == 0.0
    with pytest.raises(ValueError):
        point_coverage(np.empty((2, 0), dtype=bool))


@pytest.mark.parametrize("bad", [np.array([True, False]), np.ones((2, 2, 1), bool)])
def test_coverage_rejects_non_two_dimensional_input(bad):
    with pytest.raises(ValueError):
        point_coverage(bad)
