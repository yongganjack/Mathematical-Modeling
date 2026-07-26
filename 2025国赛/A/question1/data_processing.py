"""配置加载、固定问题数据和可复现输出辅助工具。"""

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
    """题目所提供物理常量的不可变容器。"""

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
            array = np.asarray(getattr(self, name), dtype=np.float64)
            frozen = np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)
            object.__setattr__(self, name, frozen)


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
    """读取并验证来自 UTF-8 JSON 文件的求解器配置。"""

    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"无效的 JSON 配置 {config_path}: {exc.msg}") from exc

    if not isinstance(config, dict):
        raise ValueError("配置顶层必须为 JSON 对象")

    missing = sorted(_REQUIRED_TOP_LEVEL.difference(config))
    if missing:
        raise ValueError(f"配置缺少必需键: {', '.join(missing)}")

    if not isinstance(config["profile"], str) or not config["profile"].strip():
        raise ValueError("profile 必须为非空字符串")
    if type(config["master_seed"]) is not int or config["master_seed"] != 2025:
        raise ValueError("master_seed 必须为整数 2025")

    for section_name in ("physics", "sampling", "numerical", "optimization", "output"):
        if not isinstance(config[section_name], dict):
            raise ValueError(f"{section_name} 必须为 JSON 对象")

    gravity = config["physics"].get("gravity")
    if type(gravity) not in (int, float) or gravity != 9.8:
        raise ValueError("physics.gravity 必须恰好为 9.8")

    for sampling_name in ("fast", "verify"):
        if sampling_name not in config["sampling"]:
            raise ValueError(f"配置缺少 sampling.{sampling_name}")
        if not isinstance(config["sampling"][sampling_name], dict):
            raise ValueError(f"sampling.{sampling_name} 必须为 JSON 对象")

    optimization = config["optimization"]
    budgets = optimization.get("budgets")
    if not isinstance(budgets, dict):
        raise ValueError("optimization.budgets 必须为 JSON 对象")
    for question in ("q2", "q3", "q4", "q5"):
        budget_path = f"optimization.budgets.{question}"
        budget = budgets.get(question)
        if not isinstance(budget, dict):
            raise ValueError(f"{budget_path} 必须为 JSON 对象")
        _validate_positive_config_integer(
            f"{budget_path}.max_evaluations", budget.get("max_evaluations")
        )
    _validate_positive_config_integer(
        "optimization.workers", optimization.get("workers")
    )

    return config


def _validate_positive_config_integer(name: str, value: Any) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} 必须为正整数")


def _readonly_float64(values: Any) -> FloatArray:
    """创建一个 float64 数组，确保无法通过此数据对象对其进行修改。"""

    source = np.asarray(values, dtype=np.float64)
    immutable_buffer = source.tobytes(order="C")
    return np.frombuffer(immutable_buffer, dtype=np.float64).reshape(source.shape)


def load_problem_data(config: Mapping[str, Any]) -> ProblemData:
    """根据赛题说明构建固定的数值数据。"""

    if not isinstance(config, Mapping):
        raise TypeError("config 必须为映射类型")
    physics = config.get("physics")
    if not isinstance(physics, Mapping) or "gravity" not in physics:
        raise ValueError("config 必须包含 physics.gravity")

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
        raise ValueError(f"{name} 必须为 NumPy 数组")
    if value.shape != shape:
        raise ValueError(f"{name} 形状必须为 {shape}，实际为 {value.shape}")
    if value.dtype != np.float64:
        raise ValueError(f"{name} 必须使用 float64 类型")
    if value.flags.writeable:
        raise ValueError(f"{name} 必须为只读")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} 必须只包含有限值")


def _validate_positive(name: str, value: Any) -> None:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} 必须为有限正数")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} 必须为有限正数") from exc
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} 必须为有限正数")


def validate_problem_data(data: ProblemData) -> None:
    """拒绝格式错误或物理上无效的问题数据。"""

    if not isinstance(data, ProblemData):
        raise TypeError("data 必须为 ProblemData 实例")

    _validate_array("missile_init", data.missile_init, (3, 3))
    _validate_array("uav_init", data.uav_init, (5, 3))
    _validate_array("target_center_xy", data.target_center_xy, (2,))

    _validate_positive("missile_speed", data.missile_speed)
    try:
        lower, upper = data.uav_speed_bounds
    except (TypeError, ValueError) as exc:
        raise ValueError("uav_speed_bounds 必须包含两个有限速度值") from exc
    _validate_positive("uav_speed_bounds lower", lower)
    _validate_positive("uav_speed_bounds upper", upper)
    if float(lower) >= float(upper):
        raise ValueError("uav_speed_bounds 下限必须小于上限")

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
        raise ValueError("gravity 必须恰好为 9.8")


def sample_cylinder_surface(
    profile: Mapping[str, Any], data: ProblemData
) -> FloatArray:
    """确定性地在圆柱体的侧面、顶面和轮廓边缘上进行采样。

    请求的点数预算控制近似均匀的轴向和极向间距。
    故意包含边界环；在返回一个强只读 ``float64`` 数组之前，会移除重复的边缘点。
    """

    if not isinstance(profile, Mapping):
        raise TypeError("采样配置必须为映射类型")
    target_count = profile.get("target_surface_points")
    if (
        isinstance(target_count, (bool, np.bool_))
        or not isinstance(target_count, (int, np.integer))
        or int(target_count) <= 0
    ):
        raise ValueError("target_surface_points 必须为正整数")
    if not isinstance(data, ProblemData):
        raise TypeError("data 必须为 ProblemData 实例")
    validate_problem_data(data)

    budget = int(target_count)
    radius = float(data.target_radius)
    height = float(data.target_height)
    center_x, center_y = np.asarray(data.target_center_xy, dtype=np.float64)

    side_fraction = (2.0 * radius * height) / (
        2.0 * radius * height + radius**2
    )
    side_budget = max(8, int(round(budget * side_fraction)))
    circumference_to_height = 2.0 * np.pi * radius / height
    side_azimuths = max(
        4, int(round(np.sqrt(side_budget * circumference_to_height)))
    )
    height_layers = max(2, int(round(side_budget / side_azimuths)))

    points: list[tuple[float, float, float]] = []
    side_angles = np.linspace(0.0, 2.0 * np.pi, side_azimuths, endpoint=False)
    for z in np.linspace(0.0, height, height_layers):
        for angle in side_angles:
            points.append(
                (
                    center_x + radius * np.cos(angle),
                    center_y + radius * np.sin(angle),
                    z,
                )
            )

    top_budget = max(1, budget - side_budget)
    radial_layers = max(1, int(round(np.sqrt(top_budget / np.pi))))
    points.append((center_x, center_y, height))
    for radial_index in range(1, radial_layers + 1):
        radial_distance = radius * radial_index / radial_layers
        ring_azimuths = max(4, int(round(2.0 * np.pi * radial_index)))
        for angle in np.linspace(
            0.0, 2.0 * np.pi, ring_azimuths, endpoint=False
        ):
            points.append(
                (
                    center_x + radial_distance * np.cos(angle),
                    center_y + radial_distance * np.sin(angle),
                    height,
                )
            )

    sampled = np.asarray(points, dtype=np.float64)
    sampled = np.unique(sampled, axis=0)
    if sampled.ndim != 2 or sampled.shape[1] != 3 or len(sampled) == 0:
        raise RuntimeError("圆柱表面采样未生成任何点")
    if not np.all(np.isfinite(sampled)):
        raise RuntimeError("圆柱表面采样生成了非有限值点")
    return _readonly_float64(sampled)


def _validate_view_geometry(
    missile_pos: Any, surface_points: Any, data: ProblemData
) -> tuple[FloatArray, FloatArray]:
    if not isinstance(data, ProblemData):
        raise TypeError("data 必须为 ProblemData 实例")
    validate_problem_data(data)

    missile = np.asarray(missile_pos, dtype=np.float64)
    if missile.shape != (3,):
        raise ValueError("missile_pos 形状必须为 (3,)")
    if not np.all(np.isfinite(missile)):
        raise ValueError("missile_pos 必须只包含有限值")

    points = np.asarray(surface_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError("surface_points 形状必须为 (N, 3)")
    if len(points) == 0:
        raise ValueError("surface_points 不能为空")
    if not np.all(np.isfinite(points)):
        raise ValueError("surface_points 必须只包含有限值")
    return missile, points


def visible_mask(
    missile_pos: Any, surface_points: Any, data: ProblemData
) -> NDArray[np.bool_]:
    """返回从 ``missile_pos`` 处可见的采样圆柱体表面点。

    对于凸体，当至少一个向外的支撑法线朝向观察者时，边界点即为可见。
    在顶面/侧面/底面边缘处，相邻面测试的并集使切线轮廓点可见。
    """

    missile, points = _validate_view_geometry(missile_pos, surface_points, data)
    center = np.asarray(data.target_center_xy, dtype=np.float64)
    radius = float(data.target_radius)
    height = float(data.target_height)
    radial = points[:, :2] - center
    radial_squared = np.einsum("ij,ij->i", radial, radial)
    radial_distance = np.sqrt(radial_squared)

    surface_tolerance = 1e-10 * max(1.0, radius, height)
    on_side = (
        np.abs(radial_distance - radius) <= surface_tolerance
    ) & (points[:, 2] >= -surface_tolerance) & (
        points[:, 2] <= height + surface_tolerance
    )
    within_disk = radial_distance <= radius + surface_tolerance
    on_top = within_disk & (np.abs(points[:, 2] - height) <= surface_tolerance)
    on_bottom = within_disk & (np.abs(points[:, 2]) <= surface_tolerance)
    if not np.all(on_side | on_top | on_bottom):
        raise ValueError("surface_points 必须位于圆柱表面上")

    side_dot = np.einsum("ij,ij->i", radial, missile[:2] - points[:, :2])
    viewer_distance = np.linalg.norm(missile[:2] - points[:, :2], axis=1)
    side_scale = np.maximum(1.0, radius * viewer_distance)
    side_visible = on_side & (side_dot >= -1e-10 * side_scale)
    top_visible = on_top & (missile[2] - height >= -surface_tolerance)
    bottom_visible = on_bottom & (missile[2] <= surface_tolerance)
    mask = np.asarray(side_visible | top_visible | bottom_visible, dtype=np.bool_)
    if not np.any(mask):
        raise ValueError("从 missile_pos 看不到任何目标表面点")
    return mask


def visible_target_points(
    missile_pos: Any, surface_points: Any, data: ProblemData
) -> FloatArray:
    """返回从 ``missile_pos`` 处可见的目标点的副本。"""

    points = np.asarray(surface_points, dtype=np.float64)
    mask = visible_mask(missile_pos, points, data)
    visible = np.array(points[mask], dtype=np.float64, copy=True)
    if len(visible) == 0:
        raise ValueError("从 missile_pos 看不到任何目标表面点")
    return visible


def _normalise_question_id(question_id: int | str) -> str:
    if isinstance(question_id, bool):
        raise ValueError("question_id 必须标识为有效题号")
    text = str(question_id).strip().lower()
    if text.startswith("question"):
        text = text[len("question") :]
    elif text.startswith("q"):
        text = text[1:]
    if not text.isdigit() or int(text) <= 0:
        raise ValueError("question_id 必须标识为有效题号")
    return str(int(text))


def create_run_directory(
    question_id: int | str,
    output_root: str | Path,
    run_id: str | None = None,
) -> Path:
    """创建一个新的 ``questionN/run_id`` 目录，不会覆盖已有数据。"""

    question_number = _normalise_question_id(question_id)
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id 必须为非空单路径组件")

    run_directory = Path(output_root) / f"question{question_number}" / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_directory


def _numpy_json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"类型 {type(value).__name__} 的对象不可 JSON 序列化")


def save_json(path: str | Path, payload: Any) -> None:
    """以稳定、可读的形式写入 JSON，支持 NumPy 值。"""

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
            raise ValueError("payload 包含非有限 JSON 数值") from exc
        raise


def sha256_file(path: str | Path) -> str:
    """以分块读取方式返回文件的 SHA-256 摘要值。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
