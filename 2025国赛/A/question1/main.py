"""问题一的端到端求解器，执行固定的投弹策略。"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import logging
import math
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from question1.data_processing import (  # noqa: E402
    ProblemData,
    create_run_directory,
    load_config,
    load_problem_data,
    save_json,
    sha256_file,
)
from question1.evaluation import EvaluationResult, evaluate_solution  # noqa: E402
from question1.model import BombPlan, DerivedBomb, UAVPlan, derive_bomb, missile_hit_time  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_input(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate.resolve()
    project_candidate = PROJECT_DIR / candidate
    if project_candidate.exists():
        return project_candidate.resolve()
    return candidate.resolve()


def _positive_float(text: str) -> float:
    value = float(text)
    if not np.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("必须为大于 0 的有限数")
    return value


def _scale_budgets(config: dict[str, Any], scale: float | None) -> None:
    if scale is None:
        return
    for question in ("q2", "q3", "q4", "q5"):
        for name, value in config["optimization"]["budgets"][question].items():
            if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
                config["optimization"]["budgets"][question][name] = max(1, int(float(value) * scale))


def _print_config_summary(config: Mapping[str, Any]) -> None:
    print(f"配置方案: {config['profile']}")
    print(f"主随机种子: {config['master_seed']}")
    print(f"优化预算: {config['optimization']['budgets']}")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in ("numpy", "scipy", "pandas", "openpyxl", "matplotlib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _input_hashes(config_path: Path) -> dict[str, str]:
    candidates = [
        config_path,
        PROJECT_DIR / "00_赛题资料" / "赛题原文.md",
        PROJECT_DIR / "00_赛题资料" / "数据说明.md",
        PROJECT_DIR / "00_赛题资料" / "附件" / "result1.xlsx",
        PROJECT_DIR / "00_赛题资料" / "附件" / "result2.xlsx",
        PROJECT_DIR / "00_赛题资料" / "附件" / "result3.xlsx",
    ]
    return {
        str(path.relative_to(PROJECT_DIR) if path.is_relative_to(PROJECT_DIR) else path): sha256_file(path)
        for path in candidates
        if path.is_file()
    }


def _problem_snapshot(data: ProblemData) -> dict[str, Any]:
    return {
        "missile_init": data.missile_init,
        "uav_init": data.uav_init,
        "missile_speed": data.missile_speed,
        "uav_speed_bounds": data.uav_speed_bounds,
        "target_center_xy": data.target_center_xy,
        "target_radius": data.target_radius,
        "target_height": data.target_height,
        "smoke_radius": data.smoke_radius,
        "smoke_sink_speed": data.smoke_sink_speed,
        "smoke_lifetime": data.smoke_lifetime,
        "min_release_interval": data.min_release_interval,
        "gravity": data.gravity,
    }


def _fixed_strategy() -> UAVPlan:
    bomb = BombPlan(
        bomb_index=1,
        release_time=1.5,
        fuse_delay=3.6,
        assigned_missile=0,
    )
    return UAVPlan(
        uav_index=0,
        heading_rad=math.pi,
        speed=120.0,
        bombs=(bomb,),
    )


def _effective_window(derived: DerivedBomb, data: ProblemData) -> dict[str, float]:
    lifetime_end = derived.explosion_time + float(data.smoke_lifetime)
    missile_end = missile_hit_time(0, data)
    if float(data.smoke_sink_speed) == 0.0:
        ground_end = math.inf
    else:
        ground_end = derived.explosion_time + (
            float(derived.explosion_point[2]) + float(data.smoke_radius)
        ) / float(data.smoke_sink_speed)
    return {
        "missile_index": 0,
        "start_time": derived.explosion_time,
        "end_time": min(lifetime_end, missile_end, ground_end),
        "lifetime_end": lifetime_end,
        "missile_hit_time": missile_end,
        "ground_clearance_end": ground_end,
    }


def _convergence_rows(
    fast: EvaluationResult,
    verify: EvaluationResult,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    difference = abs(float(fast.sum_objective) - float(verify.sum_objective))
    rows = []
    for name, result in (("fast", fast), ("verify", verify)):
        sampling = config["sampling"][name]
        rows.append(
            {
                "profile": name,
                "duration": result.sum_objective,
                "time_step": sampling["time_step"],
                "target_surface_points": sampling["target_surface_points"],
                "absolute_difference": difference,
            }
        )
    return rows


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _interval_rows(result: EvaluationResult) -> list[dict[str, Any]]:
    rows = []
    for missile_index in sorted(result.intervals_by_missile):
        for interval_index, (start, end) in enumerate(
            result.intervals_by_missile[missile_index], 1
        ):
            rows.append(
                {
                    "missile_index": missile_index,
                    "interval_index": interval_index,
                    "start_time": start,
                    "end_time": end,
                    "duration": end - start,
                }
            )
    return rows


def _max_residual(result: EvaluationResult) -> float:
    residuals = [
        abs(float(value))
        for values in result.boundary_residuals.values()
        for value in values
    ]
    return max(residuals, default=0.0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/quick.json")
    parser.add_argument("--output-root")
    parser.add_argument("--run-id")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--validate-config-only", action="store_true")
    parser.add_argument("--budget-scale", type=_positive_float)
    return parser


def run(args: argparse.Namespace) -> tuple[int, Path]:
    started_at = _utc_now()
    config_path = _resolve_input(args.config)
    logger.info("加载配置文件: %s", config_path)
    logger.info("配置 profile: %s", args.config)
    config = load_config(config_path)
    data = load_problem_data(config)
    logger.info("问题数据已加载: 导弹数量=%d, UAV数量=%d, 目标半径=%.2f",
                len(data.missile_init), len(data.uav_init), data.target_radius)
    _scale_budgets(config, getattr(args, "budget_scale", None))
    if getattr(args, "validate_config_only", False):
        _print_config_summary(config)
        return 0, Path()
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else (PROJECT_DIR / config["output"]["root"]).resolve()
    )
    run_dir = create_run_directory(1, output_root, args.run_id)
    manifest_path = run_dir / "manifest.json"
    manifest = {
        "profile": config["profile"],
        "master_seed": config["master_seed"],
        "versions": _package_versions(),
        "input_file_hashes": _input_hashes(config_path),
        "started_at": started_at,
        "finished_at": None,
        "command": subprocess.list2cmdline([sys.executable, *sys.argv]),
        "status": "running",
    }
    save_json(manifest_path, manifest)

    try:
        save_json(run_dir / "config.json", config)
        save_json(run_dir / "input_snapshot.json", _problem_snapshot(data))

        plan = _fixed_strategy()
        derived = derive_bomb(plan, plan.bombs[0], data)
        logger.info("开始快速评估...")
        fast = evaluate_solution(
            [plan], [0], data, config["sampling"]["fast"], config["numerical"], question_id=1
        )
        logger.info("快速评估完成: 目标值=%s, 可行=%s", fast.sum_objective, fast.feasible)
        logger.info("开始验证评估...")
        verify = evaluate_solution(
            [plan], [0], data, config["sampling"]["verify"], config["numerical"], question_id=1
        )
        logger.info("验证评估完成: 目标值=%s, 可行=%s", verify.sum_objective, verify.feasible)
        convergence_rows = _convergence_rows(fast, verify, config)
        difference = float(convergence_rows[0]["absolute_difference"])
        convergence_tolerance = max(
            float(config["sampling"]["fast"]["time_step"]),
            float(config["sampling"]["verify"]["time_step"]),
        )
        converged = difference <= convergence_tolerance
        feasible = bool(fast.feasible and verify.feasible)
        status = "succeeded" if feasible and converged else "failed"

        decision_variables = {
            "uav_index": plan.uav_index,
            "heading_rad": plan.heading_rad,
            "speed": plan.speed,
            "bombs": [
                {
                    "bomb_index": bomb.bomb_index,
                    "release_time": bomb.release_time,
                    "fuse_delay": bomb.fuse_delay,
                    "assigned_missile": bomb.assigned_missile,
                }
                for bomb in plan.bombs
            ],
        }
        raw_solution = {
            "question_id": 1,
            "profile": config["profile"],
            "decision_variables": decision_variables,
            "release_points": [derived.release_point],
            "explosion_points": [derived.explosion_point],
            "effective_windows": [_effective_window(derived, data)],
            "intervals": _plain(verify.intervals_by_missile),
            "duration_by_missile": verify.duration_by_missile,
            "fast_objective": fast.sum_objective,
            "verified_objective": verify.sum_objective,
            "feasible": feasible,
            "violations": _plain(verify.violations),
            "diagnostics": {
                "fast": _plain(fast.diagnostics),
                "verify": _plain(verify.diagnostics),
                "coverage_ratio_summary": _plain(verify.coverage_ratio_summary),
                "boundary_residuals": _plain(verify.boundary_residuals),
                "max_boundary_residual": _max_residual(verify),
                "converged": converged,
                "convergence_tolerance": convergence_tolerance,
                "absolute_difference": difference,
            },
        }
        save_json(run_dir / "raw_solution.json", raw_solution)
        interval_rows = _interval_rows(verify)
        _write_csv(
            run_dir / "intervals.csv",
            ["missile_index", "interval_index", "start_time", "end_time", "duration"],
            interval_rows,
        )
        _write_csv(
            run_dir / "convergence.csv",
            ["profile", "duration", "time_step", "target_surface_points", "absolute_difference"],
            convergence_rows,
        )
        _write_csv(
            run_dir / "optimization_history.csv",
            ["iteration", "evaluation", "objective", "best_objective"],
            [],
        )

        if not args.no_plots:
            logger.info("开始生成图表...")
            from question1.visualization import plot_convergence, plot_intervals, plot_trajectory

            plot_trajectory(data, plan, derived, run_dir)
            logger.info("轨迹图已保存到 %s", run_dir)
            plot_intervals(verify.intervals_by_missile, run_dir)
            logger.info("区间图已保存到 %s", run_dir)
            plot_convergence(convergence_rows, run_dir)
            logger.info("收敛图已保存到 %s", run_dir)
            logger.info("图表生成完成")

        manifest.update(
            {
                "finished_at": _utc_now(),
                "status": status,
            }
        )
        save_json(manifest_path, manifest)

        print(f"策略参数: {decision_variables}")
        print(f"投放点: {derived.release_point.tolist()}")
        print(f"爆炸点: {derived.explosion_point.tolist()}")
        print(f"快速评估时长: {fast.sum_objective:.15g}")
        print(f"验证评估时长: {verify.sum_objective:.15g}")
        print(f"遮挡区间: {_plain(verify.intervals_by_missile)}")
        print(f"最大残差: {_max_residual(verify):.15g}")
        print(f"状态: {status}")
        print(f"输出目录: {run_dir}")
        return (0 if status == "succeeded" else 1), run_dir
    except Exception as exc:
        manifest.update(
            {
                "finished_at": _utc_now(),
                "status": "failed",
            }
        )
        save_json(manifest_path, manifest)
        print(f"状态: 失败 ({type(exc).__name__}: {exc})", file=sys.stderr)
        return 1, run_dir


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    logger.info("问题一求解器启动")
    try:
        exit_code, _ = run(_parser().parse_args())
        logger.info("问题一求解器完成, 退出码=%d", exit_code)
        return exit_code
    except Exception as exc:
        logger.exception("求解器运行过程中发生致命错误")
        print(f"错误: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
