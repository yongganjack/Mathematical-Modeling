"""Run the Question 2 single-UAV, single-bomb optimization."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import math
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


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
from question1.model import derive_bomb, direction_from_heading  # noqa: E402
from question2.evaluation import candidate_summary  # noqa: E402
from question2.model import solve_question2  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_input(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate.resolve()
    project_candidate = PROJECT_DIR / candidate
    return (project_candidate if project_candidate.exists() else candidate).resolve()


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if hasattr(value, "__dict__"):
        return _plain(vars(value))
    return value


def _versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for name in ("numpy", "scipy", "matplotlib"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def _snapshot(data: ProblemData) -> dict[str, Any]:
    return {name: getattr(data, name) for name in data.__dataclass_fields__}


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def _interval_rows(intervals: Mapping[int, Sequence[Sequence[float]]]) -> list[dict[str, Any]]:
    return [{"missile_index": missile, "interval_index": index, "start_time": start, "end_time": end, "duration": end - start} for missile, values in intervals.items() for index, (start, end) in enumerate(values, 1)]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/competition.json")
    parser.add_argument("--output-root")
    parser.add_argument("--run-id")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def run(args: argparse.Namespace) -> tuple[int, Path]:
    started = _utc_now()
    config_path = _resolve_input(args.config)
    config = load_config(config_path)
    data = load_problem_data(config)
    output_root = Path(args.output_root).resolve() if args.output_root else (PROJECT_DIR / config["output"]["root"]).resolve()
    run_dir = create_run_directory(2, output_root, args.run_id)
    manifest_path = run_dir / "manifest.json"
    input_files = [config_path, PROJECT_DIR / "00_赛题资料" / "赛题原文.md", PROJECT_DIR / "00_赛题资料" / "数据说明.md"]
    manifest = {"profile": config["profile"], "master_seed": 2025, "versions": _versions(), "input_file_hashes": {str(path): sha256_file(path) for path in input_files if path.is_file()}, "started_at": started, "finished_at": None, "command": subprocess.list2cmdline([sys.executable, *sys.argv]), "status": "running"}
    save_json(manifest_path, manifest)
    try:
        save_json(run_dir / "config.json", config)
        save_json(run_dir / "input_snapshot.json", _snapshot(data))
        pso_seed, de_seed = np.random.SeedSequence(2025).spawn(2)
        result = solve_question2(data, config, np.random.default_rng(pso_seed), np.random.default_rng(de_seed))
        plan = __import__("question2.model", fromlist=["decode_q2_candidate"]).decode_q2_candidate(result.best_position, data)
        derived = derive_bomb(plan, plan.bombs[0], data)
        verified = result.metadata["verified_evaluation"]
        pso, de = result.metadata["pso"], result.metadata["de"]
        fast_winner = pso if pso.best_score >= de.best_score else de
        summary = candidate_summary(result.best_position, data)
        direction = direction_from_heading(plan.heading_rad)
        history = [{**row, "source": "pso"} for row in pso.history] + [{**row, "source": "de"} for row in de.history]
        raw = {
            "question_id": 2,
            "profile": config["profile"],
            "profile_result_type": "quick_search_result" if str(config["profile"]).lower() == "quick" else "competition_search_result",
            "decision_variables": {"candidate": result.best_position, **summary},
            "heading_deg": summary["heading_deg"],
            "direction": direction,
            "speed": plan.speed,
            "release_time": plan.bombs[0].release_time,
            "fuse_delay": plan.bombs[0].fuse_delay,
            "explosion_time": derived.explosion_time,
            "release_point": derived.release_point,
            "explosion_point": derived.explosion_point,
            "pso_fast_best": pso.best_score,
            "de_fast_best": de.best_score,
            "selected_fast_best": fast_winner.best_score,
            "verified_objective": float(verified.duration_by_missile[0]),
            "intervals": _plain(verified.intervals_by_missile),
            "feasible": bool(verified.feasible),
            "violations": _plain(verified.violations),
            "termination_reasons": {"pso": pso.termination_reason, "de": de.termination_reason, "combined": result.termination_reason},
            "actual_evaluations": {"pso": pso.evaluations, "de": de.evaluations, "total": pso.evaluations + de.evaluations},
            "optimization_note": "Heuristic search result; no claim of global optimality.",
        }
        save_json(run_dir / "raw_solution.json", raw)
        _write_csv(run_dir / "intervals.csv", ["missile_index", "interval_index", "start_time", "end_time", "duration"], _interval_rows(verified.intervals_by_missile))
        fields = ["source", "iteration", "best", "mean", "std", "diversity", "evaluations", "convergence"]
        _write_csv(run_dir / "optimization_history.csv", fields, history)
        convergence = [{"profile": "fast", "duration": float(fast_winner.best_score), "time_step": config["sampling"]["fast"]["time_step"], "target_surface_points": config["sampling"]["fast"]["target_surface_points"]}, {"profile": "verify", "duration": float(verified.duration_by_missile[0]), "time_step": config["sampling"]["verify"]["time_step"], "target_surface_points": config["sampling"]["verify"]["target_surface_points"]}]
        _write_csv(run_dir / "convergence.csv", ["profile", "duration", "time_step", "target_surface_points"], convergence)
        if not args.no_plots:
            from question2.visualization import plot_intervals, plot_optimizer_history, plot_trajectory
            plot_trajectory(data, plan, derived, run_dir); plot_intervals(verified.intervals_by_missile, run_dir); plot_optimizer_history(history, run_dir)
        status = "succeeded" if verified.feasible and math.isfinite(float(verified.duration_by_missile[0])) else "failed"
        manifest.update({"finished_at": _utc_now(), "status": status}); save_json(manifest_path, manifest)
        print(f"verified duration: {float(verified.duration_by_missile[0]):.15g}")
        print(f"actual evaluations: PSO={pso.evaluations}, DE={de.evaluations}")
        print(f"status: {status}"); print(f"output: {run_dir}")
        return (0 if status == "succeeded" else 1), run_dir
    except Exception as exc:
        manifest.update({"finished_at": _utc_now(), "status": "failed"}); save_json(manifest_path, manifest)
        print(f"status: failed ({type(exc).__name__}: {exc})", file=sys.stderr)
        return 1, run_dir


def main() -> int:
    return run(_parser().parse_args())[0]


if __name__ == "__main__":
    raise SystemExit(main())
