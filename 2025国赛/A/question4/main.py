"""Run Question 4: FY1, FY2, and FY3 each deploy one smoke bomb."""

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path: sys.path.insert(0, str(PROJECT_DIR))

from question1.data_processing import create_run_directory, load_config, load_problem_data, save_json, sha256_file  # noqa: E402
from question1.model import derive_bomb, direction_from_heading  # noqa: E402
from question4.data_processing import export_result2  # noqa: E402
from question4.evaluation import uav_contributions  # noqa: E402
from question4.model import decode_q4_candidate, solve_question4  # noqa: E402


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
    if scale is None: return
    for question in ("q2", "q3", "q4", "q5"):
        for name, value in config["optimization"]["budgets"][question].items():
            if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
                config["optimization"]["budgets"][question][name] = max(1, int(float(value) * scale))
def _print_config_summary(config: Mapping[str, Any]) -> None:
    print(f"profile: {config['profile']}")
    print(f"master_seed: {config['master_seed']}")
    print(f"budgets: {config['optimization']['budgets']}")
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
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", default="configs/quick.json"); parser.add_argument("--output-root"); parser.add_argument("--run-id"); parser.add_argument("--no-plots", action="store_true"); parser.add_argument("--validate-config-only", action="store_true"); parser.add_argument("--budget-scale", type=_positive_float); return parser


def run(args: argparse.Namespace) -> tuple[int, Path]:
    started = _utc_now(); wall_started = time.perf_counter(); config_path = _resolve_input(args.config); config = load_config(config_path); data = load_problem_data(config)
    _scale_budgets(config, getattr(args, "budget_scale", None))
    if getattr(args, "validate_config_only", False):
        _print_config_summary(config)
        return 0, Path()
    output_root = Path(args.output_root).resolve() if args.output_root else (PROJECT_DIR / config["output"]["root"]).resolve()
    run_dir = create_run_directory(4, output_root, args.run_id); manifest_path = run_dir / "manifest.json"
    template = PROJECT_DIR / "00_赛题资料" / "附件" / "result2.xlsx"
    input_files = [config_path, template, PROJECT_DIR / "00_赛题资料" / "赛题原文.md", PROJECT_DIR / "00_赛题资料" / "数据说明.md"]
    manifest = {"profile": config["profile"], "master_seed": 2025, "versions": _versions(), "input_file_hashes": {str(path): sha256_file(path) for path in input_files if path.is_file()}, "started_at": started, "finished_at": None, "command": subprocess.list2cmdline([sys.executable, *sys.argv]), "status": "running"}
    save_json(manifest_path, manifest)
    try:
        save_json(run_dir / "config.json", config); save_json(run_dir / "input_snapshot.json", {name: getattr(data, name) for name in data.__dataclass_fields__})
        pso_seed, de_seed = np.random.SeedSequence(2025).spawn(2)
        result = solve_question4(data, config, np.random.default_rng(pso_seed), np.random.default_rng(de_seed))
        plans = decode_q4_candidate(result.best_position, data); derived = [derive_bomb(plan, plan.bombs[0], data) for plan in plans]
        verified = result.metadata["verified_evaluation"]
        if verified is None: raise RuntimeError("best Q4 candidate did not produce an evaluation")
        contributions = uav_contributions(plans, data, config, "verify")
        sequential = [contributions["sequential_marginal"][index] for index in range(3)]
        excel_path, excel_validation = export_result2(plans, derived, data, template, run_dir / "excel" / "result2.xlsx", sequential)
        save_json(run_dir / "excel" / "export_validation.json", excel_validation)
        pso, de = result.metadata["pso"], result.metadata["de"]
        rows = []
        for plan, bomb in zip(plans, derived):
            rows.append({"uav_index": plan.uav_index, "heading_rad": plan.heading_rad, "heading_deg": float(np.degrees(plan.heading_rad) % 360.0), "direction": direction_from_heading(plan.heading_rad), "speed": plan.speed, "bomb_index": 1, "assigned_missile": 0, "release_time": bomb.release_time, "fuse_delay": bomb.fuse_delay, "explosion_time": bomb.explosion_time, "release_point": bomb.release_point, "explosion_point": bomb.explosion_point, "sequential_marginal": contributions["sequential_marginal"][plan.uav_index]})
        optimizer_summaries = {
            "pso": {"fast_best": pso.best_score, "evaluations": pso.evaluations, "termination": pso.termination_reason},
            "de": {"fast_best": de.best_score, "evaluations": de.evaluations, "termination": de.termination_reason},
        }
        raw = {"question_id": 4, "profile": config["profile"], "profile_result_type": "quick_search_result" if str(config["profile"]).lower() == "quick" else "competition_search_result", "decision_variables": result.best_position, "plans": rows, "joint_intervals": _plain(verified.intervals_by_missile[0]), "total_duration": float(verified.duration_by_missile[0]), "contributions": contributions, "pso_fast_best": pso.best_score, "de_fast_best": de.best_score, "selected_source": result.metadata["selected_source"], "optimizer_summaries": optimizer_summaries, "quick_search_result": {"selected_source": result.metadata["selected_source"], "optimizer_fast_best": result.metadata["fast_best"], "heuristic_fast_scores": result.metadata["seed_fast_scores"]}, "verified_objective": float(verified.duration_by_missile[0]), "feasible": bool(verified.feasible), "violations": _plain(verified.violations), "actual_evaluations": result.metadata["evaluation_counts"], "termination_reasons": {"pso": pso.termination_reason, "de": de.termination_reason, "combined": result.termination_reason}, "excel_path": str(excel_path), "excel_validation": bool(excel_validation["valid"]), "optimization_note": "Heuristic search result; no claim of global optimality."}
        save_json(run_dir / "raw_solution.json", raw)
        interval_rows = [{"missile_index": 0, "interval_index": index, "start_time": start, "end_time": end, "duration": end - start} for index, (start, end) in enumerate(verified.intervals_by_missile[0], 1)]
        _write_csv(run_dir / "intervals.csv", ["missile_index", "interval_index", "start_time", "end_time", "duration"], interval_rows)
        history = result.history; _write_csv(run_dir / "optimization_history.csv", ["source", "iteration", "best", "mean", "std", "diversity", "evaluations", "convergence"], history)
        fast_best = max(float(pso.best_score), float(de.best_score), *map(float, result.metadata["seed_fast_scores"].values()))
        _write_csv(run_dir / "convergence.csv", ["profile", "duration", "time_step", "target_surface_points"], [{"profile": "fast", "duration": fast_best, **config["sampling"]["fast"]}, {"profile": "verify", "duration": float(verified.duration_by_missile[0]), **config["sampling"]["verify"]}])
        if not args.no_plots:
            from question4.visualization import plot_contributions, plot_intervals, plot_optimizer_history, plot_trajectory
            plot_trajectory(data, plans, derived, run_dir); plot_intervals(verified.intervals_by_missile, run_dir); plot_contributions(contributions, run_dir); plot_optimizer_history(history, run_dir)
        status = "succeeded" if verified.feasible and math.isfinite(float(verified.duration_by_missile[0])) and excel_validation["valid"] else "failed"
        manifest.update({"finished_at": _utc_now(), "status": status, "elapsed_seconds": time.perf_counter() - wall_started}); save_json(manifest_path, manifest)
        print(f"verified objective: {float(verified.duration_by_missile[0]):.15g}"); print(f"actual evaluations: PSO={pso.evaluations}, DE={de.evaluations}, total={result.metadata['evaluation_counts']['total']}"); print(f"excel: {excel_path} (validated={excel_validation['valid']})"); print(f"elapsed seconds: {time.perf_counter() - wall_started:.3f}"); print(f"status: {status}"); print(f"output: {run_dir}")
        return (0 if status == "succeeded" else 1), run_dir
    except Exception as exc:
        manifest.update({"finished_at": _utc_now(), "status": "failed", "elapsed_seconds": time.perf_counter() - wall_started}); save_json(manifest_path, manifest); print(f"status: failed ({type(exc).__name__}: {exc})", file=sys.stderr); return 1, run_dir

def main() -> int:
    try: return run(_parser().parse_args())[0]
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
if __name__ == "__main__": raise SystemExit(main())
