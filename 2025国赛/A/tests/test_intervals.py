import numpy as np
import pytest

from question1.evaluation import (
    boolean_intervals,
    interval_length,
    merge_intervals,
    refine_boundary,
)


def test_merge_intervals_handles_empty_overlap_adjacency_and_zero_length():
    assert merge_intervals([]) == []
    assert merge_intervals([(0, 1), (1, 2), (3, 3), (3.000000001, 4)], merge_tol=1e-6) == [(0.0, 2.0), (3.0, 4.0)]
    assert interval_length((2, 2)) == 0.0
    assert interval_length((1, 4)) == 3.0
    with pytest.raises(ValueError):
        merge_intervals([(-1, 2)])
    with pytest.raises(ValueError):
        merge_intervals([(2, 1)])


def test_refine_boundary_uses_brent_root_and_reports_missing_bracket():
    root = refine_boundary(0.0, 2.0, lambda t: t - 1.25, 1e-10)
    assert root == pytest.approx(1.25)
    with pytest.raises(ValueError, match="bracket"):
        refine_boundary(0.0, 1.0, lambda t: t + 1.0, 1e-8)
    with pytest.raises(ValueError, match="finite"):
        refine_boundary(0.0, 1.0, lambda t: np.nan, 1e-8)


def test_boolean_intervals_refines_state_changes_and_keeps_zero_length_events():
    out = boolean_intervals([0.0, 1.0, 2.0], [False, True, False], lambda t: (t - 0.5) * (1.5 - t), 1e-8)
    assert out[0][0] == pytest.approx(0.5)
    assert out[0][1] == pytest.approx(1.5)
