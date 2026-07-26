"""运行问题三：FY1 对 M1 投放三枚烟雾弹。"""

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

# True: 启动时在控制台选择配置；False: 使用 PyCharm/命令行中的 --config 参数。
USE_CONSOLE_CONFIG_SELECTION = True

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from question1.data_processing import create_run_directory, load_config, load_problem_data, save_json, sha256_file  # noqa: E402
from question1.model import derive_bomb, direction_from_heading  # noqa: E402
from question3.data_processing import export_result1  # noqa: E402
from question3.evaluation import bomb_contributions  # noqa: E402
from question3.model import decode_q3_candidate, solve_question3  # noqa: E402


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
    if isinstance(value, Mapping): return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, (tuple, list)): return [_plain(item) for item in value]
    if hasattr(value, "__dict__"): return _plain(vars(value))
    return value


def _versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for name in ("numpy", "scipy", "matplotlib", "openpyxl"):
        try: result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: result[name] = "not-installed"
    return result


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
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
    started = _utc_now(); config_path = _resolve_input(args.config)
    logger.info("加载配置文件: %s", config_path)
    config = load_config(config_path); data = load_problem_data(config)
    _scale_budgets(config, getattr(args, "budget_scale", None))
    if getattr(args, "validate_config_only", False):
        _print_config_summary(config)
        return 0, Path()
    output_root = Path(args.output_root).resolve() if args.output_root else (PROJECT_DIR / config["output"]["root"]).resolve()
    run_dir = create_run_directory(3, output_root, args.run_id); manifest_path = run_dir / "manifest.json"
    template = PROJECT_DIR / "00_赛题资料" / "附件" / "result1.xlsx"
    input_files = [config_path, template, PROJECT_DIR / "00_赛题资料" / "赛题原文.md", PROJECT_DIR / "00_赛题资料" / "数据说明.md"]
    manifest = {"profile": config["profile"], "master_seed": 2025, "versions": _versions(), "input_file_hashes": {str(path): sha256_file(path) for path in input_files if path.is_file()}, "started_at": started, "finished_at": None, "command": subprocess.list2cmdline([sys.executable, *sys.argv]), "status": "running"}
    save_json(manifest_path, manifest)
    try:
        save_json(run_dir / "config.json", config)
        save_json(run_dir / "input_snapshot.json", {name: getattr(data, name) for name in data.__dataclass_fields__})
        pso_seed, de_seed = np.random.SeedSequence(2025).spawn(2)
        t_opt_start = _utc_now()
        logger.info("开始优化求解问题三（PSO + DE）...")
        result = solve_question3(data, config, np.random.default_rng(pso_seed), np.random.default_rng(de_seed))
        t_opt_end = _utc_now()
        logger.info("优化完成，耗时: %s 开始 ~ %s 结束，PSO 最佳得分: %s，DE 最佳得分: %s",
                     t_opt_start, t_opt_end, result.metadata["pso"].best_score, result.metadata["de"].best_score)
        plan = decode_q3_candidate(result.best_position, data); derived = [derive_bomb(plan, bomb, data) for bomb in plan.bombs]
        verified = result.metadata["verified_evaluation"]
        if verified is None: raise RuntimeError("最佳 Q3 候选解未产生评估结果")
        contributions = bomb_contributions(plan, data, config, "verify")
        sequential = [contributions["sequential_marginal"][bomb.bomb_index] for bomb in plan.bombs]
        logger.info("开始导出 Excel 结果并验证模板...")
        excel_path, excel_validation = export_result1(plan, derived, data, template, run_dir / "excel" / "result1.xlsx", sequential)
        save_json(run_dir / "excel" / "export_validation.json", excel_validation)
        logger.info("Excel 导出完成，路径: %s，验证: %s", excel_path, excel_validation["valid"])
        pso, de = result.metadata["pso"], result.metadata["de"]
        bombs = [{
            "bomb_index": bomb.bomb_index, "assigned_missile": bomb.assigned_missile,
            "release_time": bomb.release_time, "fuse_delay": bomb.fuse_delay,
            "explosion_time": item.explosion_time, "release_point": item.release_point, "explosion_point": item.explosion_point,
        } for bomb, item in zip(plan.bombs, derived)]
        raw = {
            "question_id": 3, "profile": config["profile"],
            "profile_result_type": "quick_search_result" if str(config["profile"]).lower() == "quick" else "competition_search_result",
            "decision_variables": result.best_position,
            "shared_route": {"uav_index": 0, "heading_rad": plan.heading_rad, "heading_deg": float(np.degrees(plan.heading_rad) % 360.0), "direction": direction_from_heading(plan.heading_rad), "speed": plan.speed},
            "bombs": bombs,
            "joint_intervals": _plain(verified.intervals_by_missile[0]), "total_duration": float(verified.duration_by_missile[0]),
            "contributions": contributions,
            "pso_fast_best": pso.best_score, "de_fast_best": de.best_score, "selected_source": result.metadata["selected_source"],
            "verified_objective": float(verified.duration_by_missile[0]), "feasible": bool(verified.feasible), "violations": _plain(verified.violations),
            "actual_evaluations": result.metadata["evaluation_counts"],
            "termination_reasons": {"pso": pso.termination_reason, "de": de.termination_reason, "combined": result.termination_reason},
            "excel_path": str(excel_path), "excel_validation": bool(excel_validation["valid"]),
            "optimization_note": "Heuristic search result; no claim of global optimality.",
        }
        save_json(run_dir / "raw_solution.json", raw)
        interval_rows = [{"missile_index": 0, "interval_index": index, "start_time": start, "end_time": end, "duration": end - start} for index, (start, end) in enumerate(verified.intervals_by_missile[0], 1)]
        _write_csv(run_dir / "intervals.csv", ["missile_index", "interval_index", "start_time", "end_time", "duration"], interval_rows)
        history = [{**row, "source": "pso"} for row in pso.history] + [{**row, "source": "de"} for row in de.history]
        _write_csv(run_dir / "optimization_history.csv", ["source", "iteration", "best", "mean", "std", "diversity", "evaluations", "convergence"], history)
        fast_best = max(float(pso.best_score), float(de.best_score), *map(float, result.metadata["seed_fast_scores"].values()))
        convergence = [
            {"profile": "fast", "duration": fast_best, **config["sampling"]["fast"]},
            {"profile": "verify", "duration": float(verified.duration_by_missile[0]), **config["sampling"]["verify"]},
        ]
        _write_csv(run_dir / "convergence.csv", ["profile", "duration", "time_step", "target_surface_points"], convergence)
        if not args.no_plots:
            logger.info("开始生成诊断图表...")
            from question3.visualization import plot_contributions, plot_intervals, plot_optimizer_history, plot_trajectory
            plot_trajectory(data, plan, derived, run_dir); plot_intervals(verified.intervals_by_missile, run_dir); plot_contributions(contributions, run_dir); plot_optimizer_history(history, run_dir)
            logger.info("图表生成完成，输出目录: %s", run_dir)
        status = "succeeded" if verified.feasible and math.isfinite(float(verified.duration_by_missile[0])) and excel_validation["valid"] else "failed"
        manifest.update({"finished_at": _utc_now(), "status": status}); save_json(manifest_path, manifest)
        direction = direction_from_heading(plan.heading_rad)
        print("\n最优投放策略：")
        print(f"无人机: FY{plan.uav_index + 1}")
        print(f"飞行方向: 航向角={np.degrees(plan.heading_rad) % 360.0:.6f} deg, 方向向量={[float(value) for value in direction]}")
        print(f"飞行速度: {plan.speed:.6f} m/s")
        for bomb, item in zip(plan.bombs, derived):
            print(f"烟幕干扰弹 {bomb.bomb_index} 投放点: {[float(value) for value in item.release_point]} m")
            print(f"烟幕干扰弹 {bomb.bomb_index} 起爆点: {[float(value) for value in item.explosion_point]} m")
        print(f"验证时长: {float(verified.duration_by_missile[0]):.15g}")
        print(f"实际评估次数: PSO={pso.evaluations}, DE={de.evaluations}, 合计={result.metadata['evaluation_counts']['total']}")
        print(f"Excel: {excel_path} (验证通过={excel_validation['valid']})"); print(f"状态: {status}"); print(f"输出目录: {run_dir}")
        return (0 if status == "succeeded" else 1), run_dir
    except Exception as exc:
        manifest.update({"finished_at": _utc_now(), "status": "failed"}); save_json(manifest_path, manifest)
        print(f"状态: 失败 ({type(exc).__name__}: {exc})", file=sys.stderr); return 1, run_dir


def main() -> int:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    try:
        args = _parser().parse_args()
        args.config = _select_config_path(args.config)
        return run(args)[0]
    except Exception as exc:
        logger.exception("主程序异常退出")
        print(f"错误: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__": raise SystemExit(main())
