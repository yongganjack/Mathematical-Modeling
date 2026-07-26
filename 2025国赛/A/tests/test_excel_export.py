from __future__ import annotations

from pathlib import Path

import numpy as np
from openpyxl import load_workbook

from question1.data_processing import load_config, load_problem_data
from question3.data_processing import export_result1
from question3.model import decode_q3_candidate


def test_export_result1_preserves_template_and_round_trips(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "quick.json")
    data = load_problem_data(config)
    candidate = np.array([0.4, 100.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    plan = decode_q3_candidate(candidate, data)
    derived = [None]
    out, validation = export_result1(
        plan,
        derived,
        data,
        root / "00_赛题资料" / "附件" / "result1.xlsx",
        tmp_path / "result1.xlsx",
        contributions=[1.25, 2.5, 3.75],
    )
    assert Path(out).resolve() == (tmp_path / "result1.xlsx").resolve()
    assert validation["valid"] is True
    ws = load_workbook(out, data_only=False)["Sheet1"]
    assert [ws.cell(r, 3).value for r in (2, 3, 4)] == [1, 2, 3]
    assert ws["A6"].value and "x轴" in ws["A6"].value
    assert ws["A5"].value is None
    assert isinstance(ws["A2"].value, float)


def test_decode_q3_enforces_release_spacing(problem_data) -> None:
    plan = decode_q3_candidate([0.0, 90.0, 2.0, -10.0, 0.1, 1.0, 2.0, 0.0], problem_data)
    releases = [bomb.release_time for bomb in plan.bombs]
    assert all(second - first >= 1.0 - 1e-9 for first, second in zip(releases, releases[1:]))
