from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest


PYTHON = Path(r"E:\Anaconda\envs\newShuMo\python.exe")


def _run_question1(project_root: Path, output_root: Path, run_id: str):
    return subprocess.run(
        [
            str(PYTHON),
            "question1/main.py",
            "--config",
            "configs/quick.json",
            "--output-root",
            str(output_root),
            "--run-id",
            run_id,
            "--no-plots",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_question1_quick_run_creates_structured_outputs(
    project_root: Path, tmp_path: Path
) -> None:
    completed = _run_question1(project_root, tmp_path, "smoke")
    assert completed.returncode == 0, completed.stdout + completed.stderr

    run_dir = tmp_path / "question1" / "smoke"
    expected = {
        "manifest.json",
        "raw_solution.json",
        "intervals.csv",
        "convergence.csv",
        "optimization_history.csv",
        "config.json",
        "input_snapshot.json",
    }
    assert expected.issubset(path.name for path in run_dir.iterdir())

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    raw_solution = json.loads(
        (run_dir / "raw_solution.json").read_text(encoding="utf-8")
    )
    assert set(manifest) == {
        "profile",
        "master_seed",
        "versions",
        "input_file_hashes",
        "started_at",
        "finished_at",
        "command",
        "status",
    }
    assert manifest["status"] == "succeeded"
    assert raw_solution["question_id"] == 1
    assert raw_solution["feasible"] is True
    assert math.isfinite(raw_solution["verified_objective"])


def test_question1_reproducible(project_root: Path, tmp_path: Path) -> None:
    first = _run_question1(project_root, tmp_path / "first", "repeat")
    second = _run_question1(project_root, tmp_path / "second", "repeat")
    assert first.returncode == 0, first.stdout + first.stderr
    assert second.returncode == 0, second.stdout + second.stderr

    first_result = json.loads(
        (tmp_path / "first/question1/repeat/raw_solution.json").read_text(
            encoding="utf-8"
        )
    )
    second_result = json.loads(
        (tmp_path / "second/question1/repeat/raw_solution.json").read_text(
            encoding="utf-8"
        )
    )
    assert first_result["decision_variables"] == second_result["decision_variables"]
    assert first_result["verified_objective"] == pytest.approx(
        second_result["verified_objective"], rel=0.0, abs=0.0
    )
    assert first_result["intervals"] == second_result["intervals"]


def test_question1_english_plotting_api_creates_png_svg_and_csv(
    problem_data, tmp_path: Path
) -> None:
    from question1.model import BombPlan, UAVPlan, derive_bomb
    from question1.visualization import (
        plot_convergence,
        plot_intervals,
        plot_trajectory,
    )

    bomb = BombPlan(
        bomb_index=1,
        release_time=1.5,
        fuse_delay=3.6,
        assigned_missile=0,
    )
    plan = UAVPlan(
        uav_index=0,
        heading_rad=np.pi,
        speed=120.0,
        bombs=(bomb,),
    )
    derived = derive_bomb(plan, bomb, problem_data)

    trajectory_paths = plot_trajectory(problem_data, plan, derived, tmp_path)
    interval_paths = plot_intervals({0: [(8.0, 9.5)]}, tmp_path)
    convergence_paths = plot_convergence(
        [
            {
                "profile": "fast",
                "duration": 1.4,
                "time_step": 0.05,
                "target_surface_points": 72,
            },
            {
                "profile": "verify",
                "duration": 1.39,
                "time_step": 0.01,
                "target_surface_points": 288,
            },
        ],
        tmp_path,
    )

    for paths in (trajectory_paths, interval_paths, convergence_paths):
        assert paths["png"].is_file()
        assert paths["svg"].is_file()
        assert paths["csv"].is_file()
