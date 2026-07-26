"""Run Question 5: lexicographic five-UAV route optimization for M1--M3."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
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


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from question1.data_processing import create_run_directory, load_config, load_problem_data, save_json, sha256_file  # noqa: E402
from question1.model import derive_bomb  # noqa: E402
from question5.data_processing import export_result3, q5_config, route_to_dict  # noqa: E402
from question5.evaluation import coarse_selected_summary, contribution_analysis, evaluate_final_routes, routes_to_plans  # noqa: E402
from question5.model import build_route_library, refine_selected_routes, solve_integer_routes  # noqa: E402


def _utc_now() -> str: return datetime.now(timezone.utc).isoformat()


def _resolve_input(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists(): return candidate.resolve()
    project_candidate = PROJECT_DIR / candidate
    return (project_candidate if project_candidate.exists() else candidate).resolve()


def _positive_float(text: str) -> float:
    value = float(text)
    if not np.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite number greater than 0")
    return value


def _scale_budgets(config: dict[str, Any], scale: float | None) -> None:
    if scale is None:
        return
    for question in ("q2", "q3", "q4", "q5"):
        for name, value in config["optimization"]["budgets"][question].items():
            if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
                config["optimization"]["budgets"][question][name] = max(1, int(float(value) * scale))


def _print_config_summary(config: Mapping[str, Any]) -> None:
    print(f"profile: {config['profile']}")
    print(f"master_seed: {config['master_seed']}")
    print(f"budgets: {config['optimization']['budgets']}")


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
        library_started = time.perf_counter(); library = build_route_library(data, runtime, 2025); library_seconds = time.perf_counter() - library_started
        route_counts = [len(routes) for routes in library.routes]
        save_json(run_dir / "route_library.json", {
            "dt_cov": library.grid.dt, "target_surface_points": len(library.grid.surface_points),
            "route_counts": route_counts,
            "routes": [[route_to_dict(route) for route in routes] for routes in library.routes],
        })

        pso_started = time.perf_counter(); result = solve_integer_routes(library, runtime, np.random.default_rng(2025)); pso_seconds = time.perf_counter() - pso_started
        refined_routes, refine = refine_selected_routes(result.selected_routes, library, data, runtime)
        coarse = coarse_selected_summary(refined_routes, library, data)
        fast, verify = evaluate_final_routes(refined_routes, data, config)
        if not verify.feasible:
            raise RuntimeError(f"selected Q5 solution is infeasible: {dict(verify.violations)}")
        contributions = contribution_analysis(refined_routes, data, config)
        plans = routes_to_plans(refined_routes)
        excel_path, excel_validation = export_result3(
            plans, data, template, run_dir / "excel" / "result3.xlsx", contributions["sequential_lookup"]
        )
        save_json(run_dir / "excel" / "export_validation.json", _plain(excel_validation))

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
            from question5.visualization import plot_duration_bars, plot_intervals, plot_stage_history, plot_trajectories
            plot_trajectories(data, plans, run_dir); plot_duration_bars(verify_durations, run_dir)
            plot_stage_history(result.history, run_dir); plot_intervals(verify.intervals_by_missile, run_dir)

        status = "succeeded" if verify.feasible and np.all(np.isfinite(verify_durations)) and excel_validation["valid"] else "failed"
        manifest.update({"finished_at": _utc_now(), "status": status}); save_json(manifest_path, manifest)
        elapsed = time.perf_counter() - wall_start
        print(f"route counts: {route_counts}")
        print(f"integer PSO evaluations: {result.evaluations}")
        print(f"selected route ids: {list(result.selected_ids)}")
        print(f"refine: {'applied' if refine['applied'] else 'skipped'} ({refine['reason']})")
        print(f"verify: T1={verify_durations[0]:.6f}, T2={verify_durations[1]:.6f}, T3={verify_durations[2]:.6f}, Jsum={float(verify.sum_objective):.6f}, Jmin={float(verify.min_duration):.6f}")
        print(f"excel: {excel_path} (validated={excel_validation['valid']})")
        print(f"elapsed: {elapsed:.3f}s"); print(f"status: {status}"); print(f"output: {run_dir}")
        return (0 if status == "succeeded" else 1), run_dir
    except Exception as exc:
        manifest.update({"finished_at": _utc_now(), "status": "failed"}); save_json(manifest_path, manifest)
        print(f"status: failed ({type(exc).__name__}: {exc})", file=sys.stderr); print(f"output: {run_dir}", file=sys.stderr)
        return 1, run_dir


def main() -> int:
    try: return run(_parser().parse_args())[0]
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__": raise SystemExit(main())
