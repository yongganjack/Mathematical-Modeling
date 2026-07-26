"""问题4边界约束与受保护的 ``result2.xlsx`` 导出。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from question1.data_processing import ProblemData, sha256_file
from question1.model import DerivedBomb, UAVPlan, derive_bomb, max_fuse_delay, missile_hit_time
from question3.data_processing import (
    TEMPLATE_SPECS,
    copy_excel_template,
    read_excel_rows,
    validate_excel_template,
    write_excel_rows,
)


def q4_bounds(data: ProblemData) -> list[tuple[float, float]]:
    """返回 FY1/FY2/FY3 对应的 ``(航向角, 速度, 投放时间, 引信延时)`` 块。"""

    hit = missile_hit_time(0, data)
    bounds: list[tuple[float, float]] = []
    for uav_index in range(3):
        bounds.extend([
            (-float(np.pi), float(np.pi)),
            tuple(map(float, data.uav_speed_bounds)),
            (0.0, hit),
            (0.0, min(max_fuse_delay(uav_index, data), hit)),
        ])
    return bounds


def q4_config(config: Mapping[str, Any] | None) -> dict[str, int]:
    """解析紧凑的快速预算和按比例缩放的竞赛预算。"""

    cfg = config or {}
    runtime = cfg.get("optimization", {}).get("q4_runtime", {})
    if runtime:
        return {name: int(runtime[name]) for name in (
            "pso_particles", "pso_iterations", "de_particles", "de_iterations"
        )}
    requested = int(cfg.get("optimization", {}).get("budgets", {}).get("q4", {}).get("max_evaluations", 6000))
    if str(cfg.get("profile", "quick")).lower() == "quick":
        # 快速模式下：90次PSO + 72次DE评估，另加少量种子和验证评估。
        return {"pso_particles": 10, "pso_iterations": 8, "de_particles": 3, "de_iterations": 1}
    pso_particles = max(16, min(48, requested // 300))
    pso_iterations = max(10, min(60, requested // max(2 * pso_particles, 1) - 1))
    remaining = max(24, requested - pso_particles * (pso_iterations + 1))
    de_particles = max(3, min(10, remaining // 12 // 8))
    de_iterations = max(2, min(30, remaining // max(12 * de_particles, 1) - 1))
    return {
        "pso_particles": pso_particles, "pso_iterations": pso_iterations,
        "de_particles": de_particles, "de_iterations": de_iterations,
    }


def export_result2(
    plans: Sequence[UAVPlan],
    derived_bombs: Sequence[DerivedBomb | None] | None,
    data: ProblemData,
    template_path: str | Path,
    destination: str | Path,
    contributions: Sequence[float],
) -> tuple[Path, dict[str, Any]]:
    """写入Q4的第B至J行，同时保护固定标识符和模板注释。"""

    ordered = sorted(plans, key=lambda plan: plan.uav_index)
    if [plan.uav_index for plan in ordered] != [0, 1, 2] or any(len(plan.bombs) != 1 for plan in ordered):
        raise ValueError("result2 导出需要 FY1、FY2、FY3 各一枚炸弹")
    contribution_values = np.asarray(contributions, dtype=float)
    if contribution_values.shape != (3,) or not np.all(np.isfinite(contribution_values)):
        raise ValueError("贡献值必须包含三个有限值")
    if derived_bombs is None or len(derived_bombs) != 3 or not all(isinstance(item, DerivedBomb) for item in derived_bombs):
        derived = [derive_bomb(plan, plan.bombs[0], data) for plan in ordered]
    else:
        derived = list(derived_bombs)
        derived.sort(key=lambda item: item.uav_index)

    output, source_hash = copy_excel_template(template_path, destination, "result2.xlsx")
    rows: list[dict[int, float]] = []
    for plan, bomb, contribution in zip(ordered, derived, contribution_values):
        rows.append({
            2: float(np.degrees(plan.heading_rad) % 360.0), 3: float(plan.speed),
            4: float(bomb.release_point[0]), 5: float(bomb.release_point[1]), 6: float(bomb.release_point[2]),
            7: float(bomb.explosion_point[0]), 8: float(bomb.explosion_point[1]), 9: float(bomb.explosion_point[2]),
            10: float(contribution),
        })
    write_excel_rows(output, "result2.xlsx", rows, number_format="0.000")
    validation = validate_excel_template(output, "result2.xlsx", require_official_hash=False)
    readback = read_excel_rows(output, "result2.xlsx")
    numeric_ok = all(
        np.isclose(float(readback[index][column]), float(rows[index][column]), rtol=0.0, atol=1e-9)
        for index in range(3) for column in TEMPLATE_SPECS["result2.xlsx"]["safe_columns"]
    )
    source_after = sha256_file(template_path)
    validation.update({
        "source_hash_before": source_hash, "source_hash_after": source_after,
        "destination": str(output), "numeric_readback_valid": bool(numeric_ok), "readback": readback,
    })
    validation["valid"] = bool(validation["valid"] and numeric_ok and source_hash == source_after)
    if not validation["valid"]:
        raise RuntimeError("result2.xlsx 导出验证失败")
    return output, validation
