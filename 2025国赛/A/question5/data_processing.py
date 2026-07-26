"""问题5的配置、序列化以及受保护的result3导出。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from question1.data_processing import ProblemData, sha256_file
from question1.model import UAVPlan, derive_bomb
from question3.data_processing import (
    TEMPLATE_SPECS,
    copy_excel_template,
    read_excel_rows,
    validate_excel_template,
    write_excel_rows,
)


def q5_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """解析紧凑的快速配置和更大的比赛配置。"""

    cfg = config or {}
    override = cfg.get("optimization", {}).get("q5_runtime", {})
    quick = str(cfg.get("profile", "quick")).lower() == "quick"
    defaults: dict[str, Any] = {
        "dt_cov": 0.5 if quick else 0.35,
        "target_surface_points": 36 if quick else 64,
        "max_routes_per_uav": 12 if quick else 24,
        "pso_particles": 30 if quick else 50,
        "stage1_iterations": 50,
        "stage2_iterations": 50,
        "epsilon_J": 0.25 if quick else 0.1,
        "refine": not quick,
        "refine_maxiter": 1 if quick else 6,
        "refine_popsize": 2 if quick else 4,
    }
    defaults.update(override)
    for key in ("target_surface_points", "max_routes_per_uav", "pso_particles", "stage1_iterations", "stage2_iterations", "refine_maxiter", "refine_popsize"):
        defaults[key] = int(defaults[key])
    for key in ("dt_cov", "epsilon_J"):
        defaults[key] = float(defaults[key])
    defaults["refine"] = bool(defaults["refine"])
    return defaults


def route_to_dict(route: Any) -> dict[str, Any]:
    return {
        "route_index": int(route.route_index),
        "uav_index": int(route.uav_index),
        "heading_rad": float(route.heading_rad),
        "heading_deg": float(np.degrees(route.heading_rad) % 360.0),
        "speed": float(route.speed),
        "assigned": [None if value is None else int(value) for value in route.assigned],
        "coverage": dict(route.coverage),
        "metadata": dict(route.metadata),
        "source": str(route.source),
        "bombs": [
            {
                "bomb_index": int(bomb.bomb_index),
                "release_time": float(bomb.release_time),
                "fuse_delay": float(bomb.fuse_delay),
                "assigned_missile": None if bomb.assigned_missile is None else int(bomb.assigned_missile),
            }
            for bomb in route.bombs
        ],
    }


def export_result3(
    plans: Sequence[UAVPlan],
    data: ProblemData,
    template_path: str | Path,
    destination: str | Path,
    sequential_contributions: Mapping[tuple[int, int], float],
) -> tuple[Path, dict[str, Any]]:
    """仅写入B:C、E:L列，保留官方的result3标识符和备注。"""

    ordered = sorted(plans, key=lambda plan: plan.uav_index)
    if [plan.uav_index for plan in ordered] != list(range(5)):
        raise ValueError("result3 导出需要恰好 FY1 至 FY5")
    if any(len(plan.bombs) > 3 for plan in ordered):
        raise ValueError("每个 UAV 最多导出三枚炸弹")
    output, source_hash = copy_excel_template(template_path, destination, "result3.xlsx")
    rows: list[dict[int, Any]] = []
    expected: list[dict[int, Any]] = []
    safe_columns = TEMPLATE_SPECS["result3.xlsx"]["safe_columns"]
    for plan in ordered:
        bombs = sorted(plan.bombs, key=lambda bomb: (bomb.release_time, bomb.bomb_index))
        for slot in range(3):
            row = {column: None for column in safe_columns}
            if slot < len(bombs):
                bomb = bombs[slot]
                derived = derive_bomb(plan, bomb, data)
                label = int(bomb.assigned_missile if bomb.assigned_missile is not None else 0)
                row.update({
                    2: float(np.degrees(plan.heading_rad) % 360.0),
                    3: float(plan.speed),
                    5: float(derived.release_point[0]), 6: float(derived.release_point[1]), 7: float(derived.release_point[2]),
                    8: float(derived.explosion_point[0]), 9: float(derived.explosion_point[1]), 10: float(derived.explosion_point[2]),
                    11: float(sequential_contributions.get((plan.uav_index, bomb.bomb_index), 0.0)),
                    12: f"M{label + 1}",
                })
            rows.append(row)
            expected.append(row)
    write_excel_rows(output, "result3.xlsx", rows, number_format="0.000")
    validation = validate_excel_template(output, "result3.xlsx", require_official_hash=False)
    readback = read_excel_rows(output, "result3.xlsx")
    readback_ok = True
    for actual, wanted in zip(readback, expected):
        for column in safe_columns:
            if wanted[column] is None:
                readback_ok &= actual[column] is None
            elif isinstance(wanted[column], (int, float, np.number)):
                readback_ok &= actual[column] is not None and np.isclose(float(actual[column]), float(wanted[column]), rtol=0.0, atol=1e-9)
            else:
                readback_ok &= actual[column] == wanted[column]
    source_after = sha256_file(template_path)
    validation.update({
        "source_hash_before": source_hash,
        "source_hash_after": source_after,
        "destination": str(output),
        "readback_valid": bool(readback_ok),
        "readback": readback,
    })
    validation["valid"] = bool(validation["valid"] and readback_ok and source_hash == source_after)
    if not validation["valid"]:
        raise RuntimeError("result3.xlsx 导出验证失败")
    return output, validation
