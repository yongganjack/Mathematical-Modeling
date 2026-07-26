"""Route library, coarse joint-coverage model, and lexicographic integer PSO."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import product
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from question1.data_processing import ProblemData, sample_cylinder_surface, visible_mask
from question1.evaluation import check_solution
from question1.model import (
    BombPlan,
    UAVPlan,
    derive_bomb,
    max_fuse_delay,
    missile_hit_time,
    missile_position,
    point_to_segments_distance,
    smoke_center,
)


Score = tuple[float, float]


@dataclass(frozen=True, slots=True)
class Route:
    route_index: int
    uav_index: int
    heading_rad: float
    speed: float
    bombs: tuple[BombPlan, ...]
    assigned: tuple[int | None, ...] = ()
    coverage: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source: str = "deterministic"

    def to_plan(self) -> UAVPlan:
        return UAVPlan(self.uav_index, self.heading_rad, self.speed, self.bombs)


@dataclass(slots=True)
class CoarseGrid:
    dt: float
    surface_points: np.ndarray
    times: dict[int, np.ndarray]
    widths: dict[int, np.ndarray]
    missile_positions: dict[int, np.ndarray]
    visible: dict[int, np.ndarray]


@dataclass(slots=True)
class RouteLibrary:
    routes: list[list[Route]]
    coverage: list[list[dict[int, np.ndarray]]]
    grid: CoarseGrid


@dataclass(slots=True)
class RouteOptimizationResult:
    selected_ids: tuple[int, ...]
    selected_routes: tuple[Route, ...]
    stage1_best: Score
    stage2_best: Score
    epsilon_J: float
    history: list[dict[str, Any]]
    evaluations: int
    termination: Mapping[str, str]


def decode_route_particle(position: Sequence[float], route_counts: Sequence[int]) -> tuple[int, ...]:
    """Round a real particle to legal route ids after clipping each dimension."""

    values = np.asarray(position, dtype=float)
    counts = np.asarray(route_counts, dtype=int)
    if values.shape != counts.shape or values.ndim != 1 or np.any(counts <= 0):
        raise ValueError("position and positive route_counts must be equal-length vectors")
    if not np.all(np.isfinite(values)):
        raise ValueError("particle position must be finite")
    clipped = np.clip(values, 0.0, counts.astype(float) - 1.0)
    return tuple(np.floor(clipped + 0.5).astype(int).tolist())


def enumerate_lexicographic(
    route_counts: Sequence[int], evaluator: Callable[[tuple[int, ...]], Score], epsilon_J: float
) -> tuple[tuple[int, ...], Score]:
    """Reference enumerator: preserve near-best Jsum, then maximize Jmin."""

    candidates = [(tuple(ids), tuple(map(float, evaluator(tuple(ids))))) for ids in product(*(range(int(n)) for n in route_counts))]
    best_sum = max(score[0] for _, score in candidates)
    eligible = [(ids, score) for ids, score in candidates if score[0] >= best_sum - float(epsilon_J) - 1e-12]
    return max(eligible, key=lambda item: (item[1][1], item[1][0], tuple(-value for value in item[0])))


def build_coarse_grid(data: ProblemData, dt: float = 0.5, target_surface_points: int = 36) -> CoarseGrid:
    surface = np.asarray(sample_cylinder_surface({"target_surface_points": int(target_surface_points)}, data), dtype=float)
    times: dict[int, np.ndarray] = {}
    widths: dict[int, np.ndarray] = {}
    positions: dict[int, np.ndarray] = {}
    visible: dict[int, np.ndarray] = {}
    for missile in range(3):
        hit = missile_hit_time(missile, data)
        edges = np.arange(0.0, hit, float(dt))
        cell_widths = np.minimum(float(dt), hit - edges)
        midpoints = edges + 0.5 * cell_widths
        pos = np.asarray([missile_position(float(time), missile, data) for time in midpoints])
        vis = np.asarray([visible_mask(point, surface, data) for point in pos], dtype=bool)
        times[missile], widths[missile], positions[missile], visible[missile] = midpoints, cell_widths, pos, vis
    return CoarseGrid(float(dt), surface, times, widths, positions, visible)


def route_coverage(route: Route, grid: CoarseGrid, data: ProblemData) -> dict[int, np.ndarray]:
    """Point-level blocked masks; clouds are ORed but points remain independent."""

    plan = route.to_plan()
    derived = [derive_bomb(plan, bomb, data) for bomb in plan.bombs]
    result: dict[int, np.ndarray] = {}
    for missile in range(3):
        mask = np.zeros((len(grid.times[missile]), len(grid.surface_points)), dtype=bool)
        for index, time in enumerate(grid.times[missile]):
            centers = [center for bomb in derived if (center := smoke_center(float(time), bomb, missile, data)) is not None]
            if centers:
                distances, _ = point_to_segments_distance(np.asarray(centers), grid.missile_positions[missile][index], grid.surface_points)
                mask[index] = np.any(distances <= float(data.smoke_radius), axis=0)
        result[missile] = mask
    return result


def approximate_score(coverages: Sequence[Mapping[int, np.ndarray]], grid: CoarseGrid) -> Score:
    durations: list[float] = []
    for missile in range(3):
        union = np.zeros_like(grid.visible[missile], dtype=bool)
        for coverage in coverages:
            union |= coverage[missile]
        fully_blocked = np.all(union | ~grid.visible[missile], axis=1)
        durations.append(float(np.sum(grid.widths[missile][fully_blocked])))
    return float(sum(durations)), float(min(durations))


def approximate_intervals(coverages: Sequence[Mapping[int, np.ndarray]], grid: CoarseGrid) -> dict[int, tuple[tuple[float, float], ...]]:
    result: dict[int, tuple[tuple[float, float], ...]] = {}
    for missile in range(3):
        union = np.zeros_like(grid.visible[missile], dtype=bool)
        for coverage in coverages:
            union |= coverage[missile]
        states = np.all(union | ~grid.visible[missile], axis=1)
        intervals: list[tuple[float, float]] = []
        start: float | None = None
        for time, width, state in zip(grid.times[missile], grid.widths[missile], states):
            left, right = float(time - width / 2.0), float(time + width / 2.0)
            if state and start is None:
                start = left
            if not state and start is not None:
                intervals.append((start, left)); start = None
        if start is not None:
            intervals.append((start, float(grid.times[missile][-1] + grid.widths[missile][-1] / 2.0)))
        result[missile] = tuple(intervals)
    return result


def _crossing_time(uav_index: int, missile_index: int, data: ProblemData) -> float:
    initial = np.asarray(data.missile_init[missile_index], dtype=float)
    uav = np.asarray(data.uav_init[uav_index], dtype=float)
    fraction = float(np.clip(np.dot(uav, initial) / np.dot(initial, initial), 0.0, 1.0))
    return missile_hit_time(missile_index, data) * (1.0 - fraction)


def _heading(origin: np.ndarray, destination: np.ndarray) -> float:
    delta = np.asarray(destination, dtype=float)[:2] - np.asarray(origin, dtype=float)[:2]
    return float(np.arctan2(delta[1], delta[0]))


def _bomb_pattern(uav_index: int, labels: Sequence[int], offset: float, data: ProblemData) -> tuple[BombPlan, ...]:
    fuse_limit = max_fuse_delay(uav_index, data)
    records: list[tuple[float, float, int]] = []
    for index, label in enumerate(labels):
        desired_explosion = max(0.4, _crossing_time(uav_index, label, data) + offset + 0.8 * index)
        fuse = min(fuse_limit, 4.0 + 0.6 * index, desired_explosion)
        release = max(0.0, desired_explosion - fuse)
        records.append((release, fuse, int(label)))
    records.sort(key=lambda item: item[0])
    repaired: list[tuple[float, float, int]] = []
    previous = -float(data.min_release_interval)
    for release, fuse, label in records:
        release = max(release, previous + float(data.min_release_interval))
        fuse = min(fuse, max_fuse_delay(uav_index, data), missile_hit_time(label, data) - release)
        if fuse >= 0.0:
            repaired.append((release, fuse, label)); previous = release
    return tuple(BombPlan(index + 1, release, fuse, label) for index, (release, fuse, label) in enumerate(repaired))


def _intercept_routes(uav_index: int, data: ProblemData) -> list[Route]:
    """Place smoke centers directly on sampled missile-to-target sight lines."""

    origin = np.asarray(data.uav_init[uav_index], dtype=float)
    target_xy = np.asarray(data.target_center_xy, dtype=float)
    candidates: list[tuple[float, int, float, float, BombPlan, dict[str, float]]] = []
    fuse_limit = max_fuse_delay(uav_index, data)
    for missile in range(3):
        hit = missile_hit_time(missile, data)
        for cover_time in np.linspace(4.0, hit - 0.5, 48):
            missile_pos = missile_position(float(cover_time), missile, data)
            if missile_pos[2] <= 0.0:
                continue
            for lag in (0.0, 2.0, 4.0, 7.0, 10.0, 14.0, 18.0):
                explosion_time = float(cover_time - lag)
                if explosion_time <= 0.2:
                    continue
                for fuse in (2.0, 3.5, 5.0, 7.0, 9.0, 11.0, 13.0, 16.0):
                    fuse = min(fuse, fuse_limit)
                    release = explosion_time - fuse
                    smoke_z = origin[2] - 0.5 * float(data.gravity) * fuse**2 - float(data.smoke_sink_speed) * lag
                    if release < 0.0 or smoke_z <= 0.0 or smoke_z >= missile_pos[2]:
                        continue
                    lam = smoke_z / missile_pos[2]
                    desired_xy = target_xy + lam * (missile_pos[:2] - target_xy)
                    delta = desired_xy - origin[:2]
                    speed = float(np.linalg.norm(delta) / explosion_time)
                    lower, upper = map(float, data.uav_speed_bounds)
                    if speed < lower or speed > upper:
                        continue
                    heading = float(np.arctan2(delta[1], delta[0]))
                    bomb = BombPlan(1, float(release), float(fuse), missile)
                    centrality = abs(speed - 105.0) + 0.05 * cover_time + 0.1 * lag
                    candidates.append((centrality, missile, heading, speed, bomb, {"cover_time": float(cover_time), "lag": lag}))
    routes: list[Route] = []
    for missile in range(3):
        pool = sorted((item for item in candidates if item[1] == missile), key=lambda item: item[0])
        selected: list[tuple[float, int, float, float, BombPlan, dict[str, float]]] = []
        for item in pool:
            if all(abs(item[4].release_time - old[4].release_time) >= 1.0 or abs(item[2] - old[2]) >= 0.03 for old in selected):
                selected.append(item)
            if len(selected) == 4:
                break
        for _, _, heading, speed, bomb, metadata in selected:
            plan = UAVPlan(uav_index, heading, speed, (bomb,))
            if check_solution([plan], 5, data).feasible:
                routes.append(Route(-1, uav_index, heading, speed, (bomb,), (missile,), {}, {"intercept_missile": missile, **metadata}, "sightline_intercept"))
    return routes


def generate_route_candidates(uav_index: int, data: ProblemData, rng: np.random.Generator) -> list[Route]:
    """Small deterministic geometry library plus seeded Latin-hypercube-like variants."""

    origin = np.asarray(data.uav_init[uav_index], dtype=float)
    target = np.array([data.target_center_xy[0], data.target_center_xy[1], 0.0])
    false_target = np.zeros(3)
    base = _heading(origin, target)
    headings = [
        base,
        _heading(origin, false_target),
        base - 0.08,
        base + 0.08,
        base - 0.16,
        base + 0.16,
        base - 0.28,
        base + 0.28,
    ]
    patterns = ((0,), (1,), (2,), (0, 1), (1, 2), (0, 2), (0, 1, 2), (2, 1, 0))
    speeds = (80.0, 110.0, 140.0)
    routes = [Route(0, uav_index, base, 110.0, (), (), {"Jsum": 0.0, "Jmin": 0.0}, {"template": "empty"}, "empty")]
    routes.extend(_intercept_routes(uav_index, data))
    for index, (heading, labels) in enumerate(zip(headings, patterns), 1):
        bombs = _bomb_pattern(uav_index, labels, (-1.5, 0.0, 1.5)[index % 3], data)
        plan = UAVPlan(uav_index, heading, speeds[index % 3], bombs)
        if check_solution([plan], 5, data).feasible:
            routes.append(Route(len(routes), uav_index, heading, speeds[index % 3], bombs, tuple(b.assigned_missile for b in bombs), {}, {"template": index}, "deterministic"))
    # Four reproducible stratified variants widen heading, speed, and timing coverage.
    for sample in range(4):
        stratum = (sample + rng.random(3)) / 4.0
        heading = base + (stratum[0] - 0.5) * 0.7
        speed = float(data.uav_speed_bounds[0] + stratum[1] * (data.uav_speed_bounds[1] - data.uav_speed_bounds[0]))
        labels = patterns[4 + sample]
        bombs = _bomb_pattern(uav_index, labels, (stratum[2] - 0.5) * 4.0, data)
        plan = UAVPlan(uav_index, heading, speed, bombs)
        if check_solution([plan], 5, data).feasible:
            routes.append(Route(len(routes), uav_index, heading, speed, bombs, tuple(b.assigned_missile for b in bombs), {}, {"lhs_sample": sample}, "lhs_seed_2025"))
    return routes


def _coverage_signature(coverage: Mapping[int, np.ndarray]) -> bytes:
    return b"".join(np.packbits(coverage[index], axis=None).tobytes() for index in range(3))


def _label_bombs(route: Route, grid: CoarseGrid, data: ProblemData) -> Route:
    if not route.bombs:
        return route
    labelled: list[BombPlan] = []
    for bomb in route.bombs:
        scores = np.full(3, -np.inf)
        for missile in range(3):
            if bomb.release_time + bomb.fuse_delay <= missile_hit_time(missile, data) + 1e-9:
                single = Route(-1, route.uav_index, route.heading_rad, route.speed, (replace(bomb, assigned_missile=missile),))
                coverage = route_coverage(single, grid, data)
                scores[missile] = approximate_score([coverage], grid)[0]
        label = int(np.argmax(scores)) if np.any(np.isfinite(scores)) else int(bomb.assigned_missile or 0)
        labelled.append(replace(bomb, assigned_missile=label))
    return replace(route, bombs=tuple(labelled), assigned=tuple(b.assigned_missile for b in labelled))


def build_route_library(data: ProblemData, runtime: Mapping[str, Any], seed: int = 2025) -> RouteLibrary:
    grid = build_coarse_grid(data, float(runtime["dt_cov"]), int(runtime["target_surface_points"]))
    seeds = np.random.SeedSequence(seed).spawn(5)
    all_routes: list[list[Route]] = []
    all_coverage: list[list[dict[int, np.ndarray]]] = []
    for uav in range(5):
        candidates = generate_route_candidates(uav, data, np.random.default_rng(seeds[uav]))
        evaluated: list[tuple[Route, dict[int, np.ndarray], Score, bytes]] = []
        for route in candidates:
            route = _label_bombs(route, grid, data)
            coverage = route_coverage(route, grid, data)
            score = approximate_score([coverage], grid)
            route = replace(route, coverage={"Jsum": score[0], "Jmin": score[1]})
            evaluated.append((route, coverage, score, _coverage_signature(coverage)))
        by_signature: dict[bytes, tuple[Route, dict[int, np.ndarray], Score, bytes]] = {}
        for item in evaluated:
            route = item[0]
            key = item[3]
            incumbent = by_signature.get(key)
            route_key = (len(route.bombs), max((b.release_time for b in route.bombs), default=0.0))
            incumbent_key = (len(incumbent[0].bombs), max((b.release_time for b in incumbent[0].bombs), default=0.0)) if incumbent else (999, float("inf"))
            if incumbent is None or route_key < incumbent_key:
                by_signature[key] = item
        unique = list(by_signature.values())
        empty = next(item for item in unique if not item[0].bombs)
        nonempty = [item for item in unique if item[0].bombs]
        nonempty.sort(key=lambda item: (item[2][0], item[2][1], -len(item[0].bombs)), reverse=True)
        chosen = [empty, *nonempty[: max(0, int(runtime["max_routes_per_uav"]) - 1)]]
        routes = [replace(item[0], route_index=index) for index, item in enumerate(chosen)]
        all_routes.append(routes); all_coverage.append([item[1] for item in chosen])
    return RouteLibrary(all_routes, all_coverage, grid)


def evaluate_route_ids(route_ids: Sequence[int], library: RouteLibrary) -> Score:
    if len(route_ids) != 5:
        raise ValueError("five route ids are required")
    coverages = [library.coverage[uav][int(route_id)] for uav, route_id in enumerate(route_ids)]
    return approximate_score(coverages, library.grid)


def _integer_pso_stage(
    library: RouteLibrary,
    rng: np.random.Generator,
    particles: int,
    iterations: int,
    stage: int,
    threshold: float | None,
    initial_ids: tuple[int, ...] | None,
    evaluation_offset: int,
) -> tuple[tuple[int, ...], Score, list[dict[str, Any]], int]:
    counts = np.asarray([len(routes) for routes in library.routes], dtype=int)
    lower = np.zeros(5); upper = counts.astype(float) - 1.0
    positions = rng.uniform(lower, upper, size=(particles, 5)); velocities = np.zeros_like(positions)
    if initial_ids is not None:
        positions[0] = np.asarray(initial_ids, dtype=float)
    personal_ids = [decode_route_particle(row, counts) for row in positions]
    personal_scores = [evaluate_route_ids(ids, library) for ids in personal_ids]
    evaluations = particles

    def key(score: Score) -> tuple[float, ...]:
        if stage == 1:
            return (score[0], score[1])
        assert threshold is not None
        feasible = score[0] >= threshold - 1e-12
        return (1.0 if feasible else 0.0, score[1] if feasible else score[0], score[0])

    global_index = max(range(particles), key=lambda index: key(personal_scores[index]))
    global_ids, global_score = personal_ids[global_index], personal_scores[global_index]
    history: list[dict[str, Any]] = []
    for iteration in range(iterations):
        r1, r2 = rng.random((particles, 5)), rng.random((particles, 5))
        personal_positions = np.asarray(personal_ids, dtype=float)
        velocities = 0.65 * velocities + 1.45 * r1 * (personal_positions - positions) + 1.45 * r2 * (np.asarray(global_ids) - positions)
        positions = np.clip(positions + velocities, lower, upper)
        decoded = [decode_route_particle(row, counts) for row in positions]
        scores = [evaluate_route_ids(ids, library) for ids in decoded]
        evaluations += particles
        for index in range(particles):
            if key(scores[index]) > key(personal_scores[index]):
                personal_ids[index], personal_scores[index] = decoded[index], scores[index]
            if key(scores[index]) > key(global_score):
                global_ids, global_score = decoded[index], scores[index]
        diversity = float(np.mean(np.std(np.asarray(decoded, dtype=float), axis=0)))
        history.append({
            "stage": stage, "iteration": iteration + 1,
            "Jsum": global_score[0], "Jmin": global_score[1],
            "diversity": diversity, "evaluations": evaluation_offset + evaluations,
            "route_ids": "-".join(map(str, global_ids)),
            "sum_residual": None if threshold is None else global_score[0] - threshold,
            "termination": "iteration_budget" if iteration + 1 == iterations else "running",
        })
    return global_ids, global_score, history, evaluations


def solve_integer_routes(library: RouteLibrary, runtime: Mapping[str, Any], rng: np.random.Generator | None = None) -> RouteOptimizationResult:
    rng = rng or np.random.default_rng(2025)
    particles = int(runtime["pso_particles"])
    ids1, score1, history1, eval1 = _integer_pso_stage(
        library, rng, particles, int(runtime["stage1_iterations"]), 1, None, None, 0
    )
    epsilon = float(runtime["epsilon_J"]); threshold = score1[0] - epsilon
    ids2, score2, history2, eval2 = _integer_pso_stage(
        library, rng, particles, int(runtime["stage2_iterations"]), 2, threshold, ids1, eval1
    )
    if score2[0] < threshold - 1e-12:
        ids2, score2 = ids1, score1
    selected = tuple(library.routes[uav][route_id] for uav, route_id in enumerate(ids2))
    return RouteOptimizationResult(ids2, selected, score1, score2, epsilon, history1 + history2, eval1 + eval2, {"stage1": "iteration_budget", "stage2": "iteration_budget"})


def refine_selected_routes(
    routes: Sequence[Route], library: RouteLibrary, data: ProblemData, runtime: Mapping[str, Any]
) -> tuple[tuple[Route, ...], dict[str, Any]]:
    """Keep quick runs honest; competition profiles may enable a tiny bounded DE."""

    before = approximate_score([route_coverage(route, library.grid, data) for route in routes], library.grid)
    if not bool(runtime.get("refine", False)):
        return tuple(routes), {"applied": False, "skipped": True, "reason": "skipped_quick_budget", "before": before, "after": before, "gain": [0.0, 0.0]}
    try:
        from scipy.optimize import differential_evolution
    except ImportError:
        return tuple(routes), {"applied": False, "skipped": True, "reason": "scipy_unavailable", "before": before, "after": before, "gain": [0.0, 0.0]}

    dimensions: list[tuple[int, int, str]] = []
    x0: list[float] = []; bounds: list[tuple[float, float]] = []
    for uav, route in enumerate(routes):
        dimensions.extend([(uav, -1, "heading"), (uav, -1, "speed")])
        x0.extend([route.heading_rad, route.speed]); bounds.extend([(-np.pi, np.pi), tuple(map(float, data.uav_speed_bounds))])
        for bomb_index, bomb in enumerate(route.bombs):
            dimensions.extend([(uav, bomb_index, "release"), (uav, bomb_index, "fuse")])
            x0.extend([bomb.release_time, bomb.fuse_delay])
            bounds.extend([(0.0, max(missile_hit_time(i, data) for i in range(3))), (0.0, max_fuse_delay(uav, data))])

    def decode(values: Sequence[float]) -> tuple[Route, ...]:
        headings = [route.heading_rad for route in routes]; speeds = [route.speed for route in routes]
        releases = [[bomb.release_time for bomb in route.bombs] for route in routes]
        fuses = [[bomb.fuse_delay for bomb in route.bombs] for route in routes]
        for value, (uav, bomb_index, kind) in zip(values, dimensions):
            if kind == "heading": headings[uav] = float((value + np.pi) % (2 * np.pi) - np.pi)
            elif kind == "speed": speeds[uav] = float(np.clip(value, *map(float, data.uav_speed_bounds)))
            elif kind == "release": releases[uav][bomb_index] = float(value)
            else: fuses[uav][bomb_index] = float(value)
        decoded: list[Route] = []
        for uav, route in enumerate(routes):
            order = np.argsort(releases[uav]); repaired: list[BombPlan] = []; previous = -float(data.min_release_interval)
            for new_index, old_index in enumerate(order):
                release = max(releases[uav][old_index], previous + float(data.min_release_interval))
                old = route.bombs[int(old_index)]; deadline = missile_hit_time(int(old.assigned_missile or 0), data)
                fuse = float(np.clip(fuses[uav][old_index], 0.0, min(max_fuse_delay(uav, data), max(0.0, deadline - release))))
                repaired.append(BombPlan(new_index + 1, release, fuse, old.assigned_missile)); previous = release
            decoded.append(replace(route, heading_rad=headings[uav], speed=speeds[uav], bombs=tuple(repaired), assigned=tuple(b.assigned_missile for b in repaired), source=route.source + "+de_refined"))
        return tuple(decoded)

    def objective(values: Sequence[float]) -> float:
        decoded = decode(values)
        if not check_solution([route.to_plan() for route in decoded], 5, data).feasible:
            return 1e6
        score = approximate_score([route_coverage(route, library.grid, data) for route in decoded], library.grid)
        return -(score[1] + 1e-4 * score[0])

    result = differential_evolution(objective, bounds, x0=np.asarray(x0), seed=2025, popsize=int(runtime["refine_popsize"]), maxiter=int(runtime["refine_maxiter"]), polish=False, workers=1)
    refined = decode(result.x)
    after = approximate_score([route_coverage(route, library.grid, data) for route in refined], library.grid)
    if after[1] < before[1] - 1e-12:
        refined, after = tuple(routes), before
    return tuple(refined), {"applied": True, "skipped": False, "reason": "differential_evolution_completed", "before": before, "after": after, "gain": [after[0] - before[0], after[1] - before[1]], "evaluations": int(result.nfev)}
