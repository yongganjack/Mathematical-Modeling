"""Tests for configuration, fixed problem data, and output utilities."""

from __future__ import annotations

import json
import random
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from question1.data_processing import (
    create_run_directory,
    load_config,
    save_json,
    sha256_file,
    validate_problem_data,
)


def _write_config(path: Path, config: dict[str, Any]) -> Path:
    path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    return path


def test_quick_config_loads_fixed_problem_data(quick_config, problem_data) -> None:
    assert quick_config["profile"] == "quick"
    assert quick_config["master_seed"] == 2025
    assert problem_data.missile_init.shape == (3, 3)
    assert problem_data.uav_init.shape == (5, 3)
    assert problem_data.missile_init.dtype == np.float64
    assert problem_data.uav_init.dtype == np.float64
    np.testing.assert_array_equal(problem_data.missile_init[0], [20000.0, 0.0, 2000.0])
    np.testing.assert_array_equal(problem_data.uav_init[0], [17800.0, 0.0, 1800.0])
    np.testing.assert_array_equal(problem_data.target_center_xy, [0.0, 200.0])
    assert problem_data.gravity == 9.8


def test_load_config_rejects_wrong_gravity(
    tmp_path: Path, quick_config: dict[str, Any]
) -> None:
    quick_config["physics"]["gravity"] = 9.81
    path = _write_config(tmp_path / "wrong-gravity.json", quick_config)

    with pytest.raises(ValueError, match="gravity"):
        load_config(path)


def test_load_config_rejects_wrong_master_seed(
    tmp_path: Path, quick_config: dict[str, Any]
) -> None:
    quick_config["master_seed"] = 7
    path = _write_config(tmp_path / "wrong-seed.json", quick_config)

    with pytest.raises(ValueError, match="master_seed"):
        load_config(path)


@pytest.mark.parametrize(
    ("missing", "message"),
    [
        ("profile", "profile"),
        ("numerical", "numerical"),
        ("optimization", "optimization"),
        ("output", "output"),
    ],
)
def test_load_config_rejects_missing_required_top_level_keys(
    tmp_path: Path,
    quick_config: dict[str, Any],
    missing: str,
    message: str,
) -> None:
    del quick_config[missing]
    path = _write_config(tmp_path / f"missing-{missing}.json", quick_config)

    with pytest.raises(ValueError, match=message):
        load_config(path)


@pytest.mark.parametrize("missing", ["fast", "verify"])
def test_load_config_requires_both_sampling_profiles(
    tmp_path: Path, quick_config: dict[str, Any], missing: str
) -> None:
    del quick_config["sampling"][missing]
    path = _write_config(tmp_path / f"missing-sampling-{missing}.json", quick_config)

    with pytest.raises(ValueError, match=rf"sampling\.{missing}"):
        load_config(path)


def test_competition_config_has_higher_q2_to_q5_budgets(
    project_root: Path, quick_config: dict[str, Any]
) -> None:
    competition = load_config(project_root / "configs" / "competition.json")

    assert competition["profile"] == "competition"
    assert competition["master_seed"] == 2025
    assert competition["physics"]["gravity"] == 9.8
    for question in ("q2", "q3", "q4", "q5"):
        quick_budget = quick_config["optimization"]["budgets"][question]
        competition_budget = competition["optimization"]["budgets"][question]
        assert competition_budget["max_evaluations"] > quick_budget["max_evaluations"]


def test_problem_data_is_frozen(problem_data) -> None:
    with pytest.raises(FrozenInstanceError):
        problem_data.gravity = 1.0

    assert not problem_data.missile_init.flags.writeable
    assert not problem_data.uav_init.flags.writeable


@pytest.mark.parametrize(
    ("field", "invalid_value", "message"),
    [
        ("missile_speed", np.inf, "missile_speed"),
        ("missile_speed", 0.0, "missile_speed"),
        ("uav_speed_bounds", (140.0, 70.0), "uav_speed_bounds"),
        ("target_radius", 0.0, "target_radius"),
        ("target_height", -1.0, "target_height"),
        ("smoke_radius", 0.0, "smoke_radius"),
        ("smoke_sink_speed", -3.0, "smoke_sink_speed"),
        ("smoke_lifetime", 0.0, "smoke_lifetime"),
        ("min_release_interval", 0.0, "min_release_interval"),
        ("gravity", np.nan, "gravity"),
    ],
)
def test_validate_problem_data_rejects_invalid_scalar_values(
    problem_data, field: str, invalid_value: Any, message: str
) -> None:
    invalid = replace(problem_data, **{field: invalid_value})

    with pytest.raises(ValueError, match=message):
        validate_problem_data(invalid)


def test_validate_problem_data_rejects_nonfinite_coordinates(problem_data) -> None:
    missile_init = problem_data.missile_init.copy()
    missile_init[1, 2] = np.nan

    with pytest.raises(ValueError, match="missile_init"):
        validate_problem_data(replace(problem_data, missile_init=missile_init))


@pytest.mark.parametrize(
    ("field", "shape", "message"),
    [
        ("missile_init", (2, 3), "missile_init"),
        ("uav_init", (5, 2), "uav_init"),
        ("target_center_xy", (3,), "target_center_xy"),
    ],
)
def test_validate_problem_data_rejects_wrong_array_shapes(
    problem_data, field: str, shape: tuple[int, ...], message: str
) -> None:
    invalid = replace(problem_data, **{field: np.zeros(shape, dtype=np.float64)})

    with pytest.raises(ValueError, match=message):
        validate_problem_data(invalid)


def test_create_run_directory_refuses_to_overwrite(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    run_dir = create_run_directory(2, output_root, run_id="fixed-run")

    assert run_dir == output_root / "question2" / "fixed-run"
    assert run_dir.is_dir()
    with pytest.raises(FileExistsError):
        create_run_directory(2, output_root, run_id="fixed-run")


def test_automatic_run_id_does_not_consume_global_random_state(tmp_path: Path) -> None:
    random.seed(2025)
    state_before = random.getstate()

    run_dir = create_run_directory("question1", tmp_path / "outputs")

    assert run_dir.is_dir()
    assert random.getstate() == state_before


def test_save_json_serializes_numpy_values_with_stable_keys(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "result.json"
    payload = {
        "z_array": np.array([1.0, 2.5], dtype=np.float64),
        "a_integer": np.int64(3),
        "m_float": np.float32(1.25),
    }

    save_json(path, payload)

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "a_integer": 3,
        "m_float": 1.25,
        "z_array": [1.0, 2.5],
    }
    text = path.read_text(encoding="utf-8")
    assert text.index('"a_integer"') < text.index('"m_float"') < text.index('"z_array"')


def test_sha256_file_is_repeatable(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"smoke-interference\n" * 1000)

    first = sha256_file(path)
    second = sha256_file(path)

    assert first == second
    assert len(first) == 64
