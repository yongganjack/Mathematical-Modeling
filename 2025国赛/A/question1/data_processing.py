"""Configuration, fixed problem data, and reproducible output helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ProblemData:
    """Immutable container for the physical constants supplied by the problem."""

    missile_init: FloatArray
    uav_init: FloatArray
    missile_speed: float
    uav_speed_bounds: tuple[float, float]
    target_center_xy: FloatArray
    target_radius: float
    target_height: float
    smoke_radius: float
    smoke_sink_speed: float
    smoke_lifetime: float
    min_release_interval: float
    gravity: float

    def __post_init__(self) -> None:
        for name in ("missile_init", "uav_init", "target_center_xy"):
            value = getattr(self, name)
            if isinstance(value, np.ndarray) and value.flags.writeable:
                raise ValueError(f"{name} must be read-only")


_REQUIRED_TOP_LEVEL = {
    "profile",
    "master_seed",
    "physics",
    "sampling",
    "numerical",
    "optimization",
    "output",
}


def load_config(path: str | Path) -> dict[str, Any]:
    """Read and validate a solver configuration from a UTF-8 JSON file."""

    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON configuration {config_path}: {exc.msg}") from exc

    if not isinstance(config, dict):
        raise ValueError("configuration top level must be a JSON object")

    missing = sorted(_REQUIRED_TOP_LEVEL.difference(config))
    if missing:
        raise ValueError(f"configuration is missing required keys: {', '.join(missing)}")

    if not isinstance(config["profile"], str) or not config["profile"].strip():
        raise ValueError("profile must be a non-empty string")
    if type(config["master_seed"]) is not int or config["master_seed"] != 2025:
        raise ValueError("master_seed must be the integer 2025")

    for section_name in ("physics", "sampling", "numerical", "optimization", "output"):
        if not isinstance(config[section_name], dict):
            raise ValueError(f"{section_name} must be a JSON object")

    gravity = config["physics"].get("gravity")
    if type(gravity) not in (int, float) or gravity != 9.8:
        raise ValueError("physics.gravity must be exactly 9.8")

    for sampling_name in ("fast", "verify"):
        if sampling_name not in config["sampling"]:
            raise ValueError(f"configuration is missing sampling.{sampling_name}")
        if not isinstance(config["sampling"][sampling_name], dict):
            raise ValueError(f"sampling.{sampling_name} must be a JSON object")

    optimization = config["optimization"]
    budgets = optimization.get("budgets")
    if not isinstance(budgets, dict):
        raise ValueError("optimization.budgets must be a JSON object")
    for question in ("q2", "q3", "q4", "q5"):
        budget_path = f"optimization.budgets.{question}"
        budget = budgets.get(question)
        if not isinstance(budget, dict):
            raise ValueError(f"{budget_path} must be a JSON object")
        _validate_positive_config_integer(
            f"{budget_path}.max_evaluations", budget.get("max_evaluations")
        )
    _validate_positive_config_integer(
        "optimization.workers", optimization.get("workers")
    )

    return config


def _validate_positive_config_integer(name: str, value: Any) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _readonly_float64(values: Any) -> FloatArray:
    """Create a float64 array that cannot be mutated through this data object."""

    source = np.asarray(values, dtype=np.float64)
    immutable_buffer = source.tobytes(order="C")
    return np.frombuffer(immutable_buffer, dtype=np.float64).reshape(source.shape)


def load_problem_data(config: Mapping[str, Any]) -> ProblemData:
    """Build the fixed numerical data supplied in the competition statement."""

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    physics = config.get("physics")
    if not isinstance(physics, Mapping) or "gravity" not in physics:
        raise ValueError("config must contain physics.gravity")

    data = ProblemData(
        missile_init=_readonly_float64(
            [
                [20000.0, 0.0, 2000.0],
                [19000.0, 600.0, 2100.0],
                [18000.0, -600.0, 1900.0],
            ]
        ),
        uav_init=_readonly_float64(
            [
                [17800.0, 0.0, 1800.0],
                [12000.0, 1400.0, 1400.0],
                [6000.0, -3000.0, 700.0],
                [11000.0, 2000.0, 1800.0],
                [13000.0, -2000.0, 1300.0],
            ]
        ),
        missile_speed=np.float64(300.0),
        uav_speed_bounds=(np.float64(70.0), np.float64(140.0)),
        target_center_xy=_readonly_float64([0.0, 200.0]),
        target_radius=np.float64(7.0),
        target_height=np.float64(10.0),
        smoke_radius=np.float64(10.0),
        smoke_sink_speed=np.float64(3.0),
        smoke_lifetime=np.float64(20.0),
        min_release_interval=np.float64(1.0),
        gravity=np.float64(physics["gravity"]),
    )
    validate_problem_data(data)
    return data


def _validate_array(name: str, value: Any, shape: tuple[int, ...]) -> None:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be a NumPy array")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if value.dtype != np.float64:
        raise ValueError(f"{name} must use float64 values")
    if value.flags.writeable:
        raise ValueError(f"{name} must be read-only")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")


def _validate_positive(name: str, value: Any) -> None:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite positive number")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")


def validate_problem_data(data: ProblemData) -> None:
    """Reject malformed or physically invalid problem data."""

    if not isinstance(data, ProblemData):
        raise TypeError("data must be a ProblemData instance")

    _validate_array("missile_init", data.missile_init, (3, 3))
    _validate_array("uav_init", data.uav_init, (5, 3))
    _validate_array("target_center_xy", data.target_center_xy, (2,))

    _validate_positive("missile_speed", data.missile_speed)
    try:
        lower, upper = data.uav_speed_bounds
    except (TypeError, ValueError) as exc:
        raise ValueError("uav_speed_bounds must contain two finite speeds") from exc
    _validate_positive("uav_speed_bounds lower", lower)
    _validate_positive("uav_speed_bounds upper", upper)
    if float(lower) >= float(upper):
        raise ValueError("uav_speed_bounds lower limit must be less than upper limit")

    for name in (
        "target_radius",
        "target_height",
        "smoke_radius",
        "smoke_sink_speed",
        "smoke_lifetime",
        "min_release_interval",
        "gravity",
    ):
        _validate_positive(name, getattr(data, name))
    if float(data.gravity) != 9.8:
        raise ValueError("gravity must be exactly 9.8")


def _normalise_question_id(question_id: int | str) -> str:
    if isinstance(question_id, bool):
        raise ValueError("question_id must identify a positive question number")
    text = str(question_id).strip().lower()
    if text.startswith("question"):
        text = text[len("question") :]
    elif text.startswith("q"):
        text = text[1:]
    if not text.isdigit() or int(text) <= 0:
        raise ValueError("question_id must identify a positive question number")
    return str(int(text))


def create_run_directory(
    question_id: int | str,
    output_root: str | Path,
    run_id: str | None = None,
) -> Path:
    """Create a fresh ``questionN/run_id`` directory without overwriting data."""

    question_number = _normalise_question_id(question_id)
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be a non-empty single path component")

    run_directory = Path(output_root) / f"question{question_number}" / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_directory


def _numpy_json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def save_json(path: str | Path, payload: Any) -> None:
    """Write JSON in a stable, readable form, including NumPy values."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
                default=_numpy_json_default,
            )
            handle.write("\n")
    except ValueError as exc:
        if "Out of range float values" in str(exc):
            raise ValueError("payload contains a non-finite JSON number") from exc
        raise


def sha256_file(path: str | Path) -> str:
    """Return a SHA-256 digest while reading the file in bounded chunks."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
