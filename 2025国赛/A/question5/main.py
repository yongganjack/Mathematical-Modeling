"""运行问题5：针对M1-M3的字典序五无人机路径优化。"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import logging
import math
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# True: 启动时在控制台选择配置；False: 使用 PyCharm/命令行中的 --config 参数。
USE_CONSOLE_CONFIG_SELECTION = True

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from question1.data_processing import create_run_directory, load_config, load_problem_data, save_json, sha256_file  # noqa: E402
from question1.model import derive_bomb, direction_from_heading  # noqa: E402
from question5.data_processing import export_result3, q5_config, route_to_dict  # noqa: E402
from question5.evaluation import coarse_selected_summary, contribution_analysis, evaluate_final_routes, routes_to_plans  # noqa: E402
from question5.model import build_route_library, refine_selected_routes, solve_integer_routes  # noqa: E402


def _utc_now() -> str: return datetime.now(timezone.utc).isoformat()


def _resolve_input(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists(): return candidate.resolve()
    project_candidate = PROJECT_DIR / candidate
    return (project_candidate if project_candidate.exists() else candidate).resolve()


def _select_config_path(config_arg: str) -> str:
    if not USE_CONSOLE_CONFIG_SELECTION:
        return config_arg
    choices = {"1": "configs/quick.json", "2": "configs/competition.json"}
    while True:
        print("\n请选择运行配置：")
        print("1. Quick（快速可复现）")
        print("2. Competition（竞赛级高预算）")
        try:
            choice = input("请输入选项 [1/2]：").strip()
        except EOFError as exc:
            raise RuntimeError("无法读取控制台输入；如需使用 PyCharm 参数，请将 USE_CONSOLE_CONFIG_SELECTION 改为 False") from exc
        if choice in choices:
            selected = choices[choice]
            print(f"已选择配置：{selected}")
            return selected
        print("输入无效，请输入 1 或 2。")


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
    if is_dataclass(value): return _plain(asdict(value))
    if isinstance(value, Mapping): return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, (tuple, list)): return [_plain(item) for item in value]
    return value


def _versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for name in ("numpy", "scipy", "matplotlib", "openpyxl"):
        try: result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: result[name] = "not-installed"
    return result


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/quick.json")
    parser.add_argument("--output-root"); parser.add_argument("--run-id"); parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--validate-config-only", action="store_true")
    parser.add_argument("--budget-scale", type=_positive_float)
    return parser


def run(args: argparse.Namespace) -> tuple[int, Path]:
    wall_start = time.perf_counter(); started = _utc_now()
    config_path = _resolve_input(args.config); config = load_config(config_path); data = load_problem_data(config)
    logger.info("加载配置: %s", config_path)
    logger.info("配置概要: profile=%s, master_seed=%s, budget_scale=%s",
                config['profile'], config['master_seed'], getattr(args, 'budget_scale', None))
    _scale_budgets(config, getattr(args, "budget_scale", None))
    if getattr(args, "validate_config_only", False):
        _print_config_summary(config)
        return 0, Path()
    runtime = q5_config(config)
    output_root = Path(args.output_root).resolve() if args.output_root else (PROJECT_DIR / config["output"]["root"]).resolve()
    run_dir = create_run_directory(5, output_root, args.run_id); manifest_path = run_dir / "manifest.json"
    template = PROJECT_DIR / "00_赛题资料" / "附件" / "result3.xlsx"
    input_files = [config_path, template, PROJECT_DIR / "00_赛题资料" / "赛题原文.md", PROJECT_DIR / "00_赛题资料" / "数据说明.md"]
    manifest = {
        "profile": config["profile"], "master_seed": 2025, "versions": _versions(),
        "input_file_hashes": {str(path): sha256_file(path) for path in input_files if path.is_file()},
        "started_at": started, "finished_at": None,
        "command": subprocess.list2cmdline([sys.executable, *sys.argv]), "status": "running",
    }
    save_json(manifest_path, manifest)
    try:
        save_json(run_dir / "config.json", config)
        save_json(run_dir / "input_snapshot.json", {name: getattr(data, name) for name in data.__dataclass_fields__})
        library_started = time.perf_counter()
        logger.info("开始构建路径库 (dt_cov=%.2f, target_surface_points=%d, max_routes_per_uav=%d)...",
                    runtime['dt_cov'], runtime['target_surface_points'], runtime['max_routes_per_uav'])
        library = build_route_library(data, runtime, 2025); library_seconds = time.perf_counter() - library_started
        route_counts = [len(routes) for routes in library.routes]
        logger.info("路径库构建完成，耗时 %.3f 秒, 路径数量: %s", library_seconds, route_counts)
        save_json(run_dir / "route_library.json", {
            "dt_cov": library.grid.dt, "target_surface_points": len(library.grid.surface_points),
            "route_counts": route_counts,
            "routes": [[route_to_dict(route) for route in routes] for routes in library.routes],
        })

        pso_started = time.perf_counter()
        logger.info("开始整数PSO优化 (particles=%d, stage1_iters=%d, stage2_iters=%d, epsilon_J=%.3f)...",
                    runtime['pso_particles'], runtime['stage1_iterations'], runtime['stage2_iterations'], runtime['epsilon_J'])
        result = solve_integer_routes(library, runtime, np.random.default_rng(2025)); pso_seconds = time.perf_counter() - pso_started
        logger.info("整数PSO完成，耗时 %.3f 秒, 总评估次数=%d, 选定路径IDs=%s",
                    pso_seconds, result.evaluations, list(result.selected_ids))
        logger.info("PSO结果: Stage1(Jsum=%.3f, Jmin=%.3f), Stage2(Jsum=%.3f, Jmin=%.3f)",
                    result.stage1_best[0], result.stage1_best[1], result.stage2_best[0], result.stage2_best[1])
        refined_routes, refine = refine_selected_routes(result.selected_routes, library, data, runtime)
        coarse = coarse_selected_summary(refined_routes, library, data)
        logger.info("路径粗粒度评估: Jsum=%.3f, Jmin=%.3f", coarse['Jsum'], coarse['Jmin'])
        fast, verify = evaluate_final_routes(refined_routes, data, config)
        if not verify.feasible:
            raise RuntimeError(f"选定的 Q5 解不可行: {dict(verify.violations)}")
        contributions = contribution_analysis(refined_routes, data, config)
        plans = routes_to_plans(refined_routes)
        logger.info("导出Excel模板并验证...")
        excel_path, excel_validation = export_result3(
            plans, data, template, run_dir / "excel" / "result3.xlsx", contributions["sequential_lookup"]
        )
        save_json(run_dir / "excel" / "export_validation.json", _plain(excel_validation))
        logger.info("Excel导出完成: %s (验证通过=%s)", excel_path, excel_validation['valid'])

        bomb_rows: list[dict[str, Any]] = []
        for plan in plans:
            for bomb in plan.bombs:
                derived = derive_bomb(plan, bomb, data)
                bomb_rows.append({
                    "uav_index": plan.uav_index, "bomb_index": bomb.bomb_index,
                    "heading_rad": plan.heading_rad, "heading_deg": float(np.degrees(plan.heading_rad) % 360.0), "speed": plan.speed,
                    "release_time": bomb.release_time, "fuse_delay": bomb.fuse_delay, "explosion_time": derived.explosion_time,
                    "assigned_missile": bomb.assigned_missile,
                    "release_point": derived.release_point, "explosion_point": derived.explosion_point,
                    "sequential_marginal": contributions["sequential_lookup"].get((plan.uav_index, bomb.bomb_index), 0.0),
                })
        fast_durations = np.asarray(fast.duration_by_missile, dtype=float); verify_durations = np.asarray(verify.duration_by_missile, dtype=float)
        actual_evaluations = {
            "coarse_integer_pso": result.evaluations,
            "continuous_refinement": int(refine.get("evaluations", 0)),
            "complete_final_fast": 1, "complete_final_verify": 1,
            "fast_contribution_evaluations": contributions["evaluation_count"],
            "complete_total": 2 + contributions["evaluation_count"],
        }
        raw = {
            "question_id": 5, "profile": config["profile"],
            "profile_result_type": "quick_search_result" if str(config["profile"]).lower() == "quick" else "competition_search_result",
            "quick_search_result": str(config["profile"]).lower() == "quick",
            "route_counts": route_counts, "selected_route_ids": result.selected_ids,
            "selected_routes": [route_to_dict(route) for route in refined_routes], "bombs": bomb_rows,
            "approx_stage1": {"Jsum": result.stage1_best[0], "Jmin": result.stage1_best[1]},
            "approx_stage2": {"Jsum": result.stage2_best[0], "Jmin": result.stage2_best[1]},
            "approx_refined": coarse, "epsilon_J": result.epsilon_J,
            "fast": {"T1": fast_durations[0], "T2": fast_durations[1], "T3": fast_durations[2], "Jsum": float(fast.sum_objective), "Jmin": float(fast.min_duration)},
            "verify": {"T1": verify_durations[0], "T2": verify_durations[1], "T3": verify_durations[2], "Jsum": float(verify.sum_objective), "Jmin": float(verify.min_duration), "intervals": verify.intervals_by_missile},
            "refine": refine,
            "contributions": {key: value for key, value in contributions.items() if key != "sequential_lookup"},
            "actual_evaluations": actual_evaluations, "termination_reasons": result.termination,
            "timing_seconds": {"route_library": library_seconds, "integer_pso": pso_seconds, "total": time.perf_counter() - wall_start},
            "feasible": bool(verify.feasible), "violations": verify.violations,
            "excel_path": str(excel_path), "excel_validation": bool(excel_validation["valid"]),
            "optimization_note": "Heuristic route-library and integer-PSO result; no claim of global optimality.",
        }
        save_json(run_dir / "raw_solution.json", _plain(raw))

        fields = ["stage", "iteration", "Jsum", "Jmin", "diversity", "evaluations", "route_ids", "sum_residual", "termination"]
        _write_csv(run_dir / "optimization_history.csv", fields, result.history)
        _write_csv(run_dir / "route_selection_history.csv", fields, result.history)
        interval_rows = [
            {"missile_index": missile, "interval_index": index, "start_time": start, "end_time": end, "duration": end - start}
            for missile in range(3) for index, (start, end) in enumerate(verify.intervals_by_missile[missile], 1)
        ]
        _write_csv(run_dir / "intervals.csv", ["missile_index", "interval_index", "start_time", "end_time", "duration"], interval_rows)
        convergence_rows = []
        for profile, evaluation in (("fast", fast), ("verify", verify)):
            for missile in range(3):
                convergence_rows.append({
                    "profile": profile, "missile_index": missile,
                    "duration": float(evaluation.duration_by_missile[missile]),
                    "Jsum": float(evaluation.sum_objective), "Jmin": float(evaluation.min_duration),
                    **config["sampling"][profile],
                })
        _write_csv(run_dir / "convergence.csv", ["profile", "missile_index", "duration", "Jsum", "Jmin", "time_step", "target_surface_points"], convergence_rows)
        if not args.no_plots:
            logger.info("生成诊断图表...")
            from question5.visualization import plot_duration_bars, plot_intervals, plot_stage_history, plot_trajectories
            plot_trajectories(data, plans, run_dir); plot_duration_bars(verify_durations, run_dir)
            plot_stage_history(result.history, run_dir); plot_intervals(verify.intervals_by_missile, run_dir)
            logger.info("图表生成完成")

        status = "succeeded" if verify.feasible and np.all(np.isfinite(verify_durations)) and excel_validation["valid"] else "failed"
        manifest.update({"finished_at": _utc_now(), "status": status}); save_json(manifest_path, manifest)
        print("\n最优投放策略：")
        for row in bomb_rows:
            print(f"无人机 FY{row['uav_index'] + 1}，烟幕干扰弹 {row['bomb_index']}：")
            print(f"  飞行方向: 航向角={row['heading_deg']:.6f} deg, 方向向量={[float(value) for value in direction_from_heading(row['heading_rad'])]}")
            print(f"  飞行速度: {row['speed']:.6f} m/s")
            print(f"  烟幕干扰弹投放点: {[float(value) for value in row['release_point']]} m")
            print(f"  烟幕干扰弹起爆点: {[float(value) for value in row['explosion_point']]} m")
        elapsed = time.perf_counter() - wall_start
        print(f"路径数量: {route_counts}")
        print(f"整数PSO评估次数: {result.evaluations}")
        print(f"选定路径ID: {list(result.selected_ids)}")
        print(f"细化: {'已应用' if refine['applied'] else '已跳过'} ({refine['reason']})")
        print(f"验证: T1={verify_durations[0]:.6f}, T2={verify_durations[1]:.6f}, T3={verify_durations[2]:.6f}, Jsum={float(verify.sum_objective):.6f}, Jmin={float(verify.min_duration):.6f}")
        print(f"Excel: {excel_path} (验证通过={excel_validation['valid']})")
        print(f"耗时: {elapsed:.3f}s"); print(f"状态: {status}"); print(f"输出目录: {run_dir}")
        return (0 if status == "succeeded" else 1), run_dir
    except Exception as exc:
        manifest.update({"finished_at": _utc_now(), "status": "failed"}); save_json(manifest_path, manifest)
        print(f"状态: 失败 ({type(exc).__name__}: {exc})", file=sys.stderr); print(f"输出目录: {run_dir}", file=sys.stderr)
        return 1, run_dir


def main() -> int:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    try:
        args = _parser().parse_args()
        args.config = _select_config_path(args.config)
        return run(args)[0]
    except Exception as exc:
        print(f"错误: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__": raise SystemExit(main())
