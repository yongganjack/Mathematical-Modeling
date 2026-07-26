# 烟幕干扰弹投放策略 Python 求解系统 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 完成 2025 国赛 A 题问题 1 至问题 5 的模块化 Python 求解代码，提供快速可复现配置和竞赛级配置，并生成可验证的结构化结果、英文图表与 Excel 提交副本。

**Architecture:** 问题 1 提供稳定的题面数据、运动学、几何遮蔽和连续区间评价接口，问题 2 至问题 5 通过该接口逐级增加 PSO、DE、多弹协同、多机协同和路线选择。每个小问保留 `main.py`、`data_processing.py`、`model.py`、`evaluation.py`、`visualization.py` 五文件结构，测试集中在 `tests/`，运行参数集中在 `configs/`。

**Tech Stack:** Python 3.10、NumPy、SciPy、pandas、openpyxl、matplotlib、pytest、JSON/CSV/NPZ。

---

## 1. 文档目的

本文将已批准的系统设计拆解为可直接执行的测试驱动实施步骤。每项任务先编写能够因功能缺失而失败的测试，再编写最小实现使其通过，最后运行相关测试和全量回归。

计划只规定代码和验证工作，不预设任何尚未实际计算的遮蔽时长、最优策略、收敛次数或运行时间。

## 2. 输入信息来源

- `docs/superpowers/specs/2026-07-26-smoke-interference-solver-design.md`；
- `2025国赛/A/00_赛题资料/赛题原文.md`；
- `2025国赛/A/00_赛题资料/数据说明.md`；
- `2025国赛/A/03_模型设计/符号说明.md`；
- `2025国赛/A/04_模型求解/模型推导.md`；
- `2025国赛/A/04_模型求解/算法流程.md`；
- `2025国赛/A/04_模型求解/代码实现说明.md`；
- 三个 Excel 模板文件。

可选文件 `2025国赛/A/02_数据处理/数据预处理方案.md` 不存在。

## 3. 核心内容

### 3.1 文件映射

**新增配置：**

- `2025国赛/A/configs/quick.json`：默认快速配置；
- `2025国赛/A/configs/competition.json`：竞赛级配置。

**新增问题目录：**

- `2025国赛/A/question1/` 至 `question5/`：每个目录包含五个规定文件；
- 每个问题目录增加空的 `__init__.py` 以支持稳定导入；该文件不承担业务逻辑。

**新增测试：**

- `2025国赛/A/tests/conftest.py`；
- `test_config_and_data.py`；
- `test_kinematics.py`；
- `test_geometry.py`；
- `test_sampling.py`；
- `test_coverage.py`；
- `test_intervals.py`；
- `test_feasibility.py`；
- `test_optimizers.py`；
- `test_excel_export.py`；
- `test_cli_smoke.py`。

**修改文档：**

- `2025国赛/A/04_模型求解/代码实现说明.md`。

### Task 1: 配置、题面数据与运行目录

**Files:**

- Create: `2025国赛/A/configs/quick.json`
- Create: `2025国赛/A/configs/competition.json`
- Create: `2025国赛/A/question1/__init__.py`
- Create: `2025国赛/A/question1/data_processing.py`
- Create: `2025国赛/A/tests/conftest.py`
- Create: `2025国赛/A/tests/test_config_and_data.py`

- [ ] **Step 1: 编写配置和题面数据失败测试**

```python
from pathlib import Path

import numpy as np
import pytest

from question1.data_processing import load_config, load_problem_data


def test_problem_data_matches_statement(project_root: Path) -> None:
    config = load_config(project_root / "configs" / "quick.json")
    data = load_problem_data(config)
    assert data.missile_init.shape == (3, 3)
    assert data.uav_init.shape == (5, 3)
    np.testing.assert_allclose(data.missile_init[0], [20000.0, 0.0, 2000.0])
    np.testing.assert_allclose(data.uav_init[0], [17800.0, 0.0, 1800.0])
    assert data.gravity == pytest.approx(9.8)


def test_config_rejects_wrong_gravity(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"profile":"quick","physics":{"gravity":9.81}}', encoding="utf-8")
    with pytest.raises(ValueError, match="gravity"):
        load_config(path)
```

- [ ] **Step 2: 运行测试并确认因模块缺失而失败**

Run:

```powershell
E:\Anaconda\envs\newShuMo\python.exe -m pytest tests/test_config_and_data.py -v
```

Expected: collection fails with `ModuleNotFoundError: No module named 'question1'` or missing API error。

- [ ] **Step 3: 实现不可变数据对象与配置读取**

在 `question1/data_processing.py` 中实现：

```python
@dataclass(frozen=True)
class ProblemData:
    missile_init: np.ndarray
    uav_init: np.ndarray
    missile_speed: float
    uav_speed_bounds: tuple[float, float]
    target_center_xy: np.ndarray
    target_radius: float
    target_height: float
    smoke_radius: float
    smoke_sink_speed: float
    smoke_lifetime: float
    min_release_interval: float
    gravity: float


def load_config(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload["physics"]["gravity"] != 9.8:
        raise ValueError("formal gravity must equal 9.8 m/s^2")
    if payload["master_seed"] != 2025:
        raise ValueError("master_seed must equal 2025")
    return payload


def load_problem_data(config: Mapping[str, Any]) -> ProblemData:
    data = ProblemData(
        missile_init=np.asarray([[20000, 0, 2000], [19000, 600, 2100], [18000, -600, 1900]], dtype=np.float64),
        uav_init=np.asarray([[17800, 0, 1800], [12000, 1400, 1400], [6000, -3000, 700], [11000, 2000, 1800], [13000, -2000, 1300]], dtype=np.float64),
        missile_speed=300.0,
        uav_speed_bounds=(70.0, 140.0),
        target_center_xy=np.asarray([0.0, 200.0]),
        target_radius=7.0,
        target_height=10.0,
        smoke_radius=10.0,
        smoke_sink_speed=3.0,
        smoke_lifetime=20.0,
        min_release_interval=1.0,
        gravity=9.8,
    )
    validate_problem_data(data)
    return data
```

配置文件必须包含 `profile`、`master_seed`、`physics`、`sampling.fast`、`sampling.verify`、`numerical`、`optimization`、`output`，并为 Q2—Q5 分别提供预算。

- [ ] **Step 4: 实现运行目录和 JSON 序列化助手**

实现 `create_run_directory(question_id, output_root, run_id=None)`、`save_json(path, payload)`、`sha256_file(path)`，禁止覆盖已有运行目录。

- [ ] **Step 5: 运行配置测试**

Run:

```powershell
E:\Anaconda\envs\newShuMo\python.exe -m pytest tests/test_config_and_data.py -v
```

Expected: all tests pass。

- [ ] **Step 6: 提交任务 1**

```powershell
git add -- 2025国赛/A/configs 2025国赛/A/question1 2025国赛/A/tests
git commit -m "feat: add solver configuration and problem data"
```

### Task 2: 运动学、弹道和云团有效窗

**Files:**

- Create: `2025国赛/A/question1/model.py`
- Create: `2025国赛/A/tests/test_kinematics.py`

- [ ] **Step 1: 编写运动学失败测试**

```python
import numpy as np
import pytest

from question1.model import BombPlan, UAVPlan, derive_bomb, missile_hit_time, missile_position, smoke_center


def test_missile_reaches_origin(problem_data) -> None:
    hit = missile_hit_time(0, problem_data)
    np.testing.assert_allclose(missile_position(hit, 0, problem_data), np.zeros(3), atol=1e-9)


def test_question1_bomb_points(problem_data) -> None:
    plan = UAVPlan(uav_index=0, heading_rad=np.pi, speed=120.0, bombs=(BombPlan(0, 1.5, 3.6, 0),))
    derived = derive_bomb(plan, plan.bombs[0], problem_data)
    np.testing.assert_allclose(derived.release_point, [17620.0, 0.0, 1800.0], atol=1e-9)
    np.testing.assert_allclose(derived.explosion_point, [17188.0, 0.0, 1736.496], atol=1e-9)


def test_smoke_top_ground_boundary_is_active(problem_data) -> None:
    bomb = DerivedBomb(
        uav_index=0,
        bomb_index=0,
        assigned_missile=0,
        release_time=0.0,
        fuse_delay=0.0,
        explosion_time=0.0,
        release_point=np.asarray([0.0, 0.0, 20.0]),
        explosion_point=np.asarray([0.0, 0.0, 20.0]),
    )
    boundary_time = (20.0 + problem_data.smoke_radius) / problem_data.smoke_sink_speed
    center = smoke_center(boundary_time, bomb, 0, problem_data)
    assert center is not None
    assert center[2] + problem_data.smoke_radius == pytest.approx(0.0)
    assert smoke_center(boundary_time + 1e-6, bomb, 0, problem_data) is None
```

- [ ] **Step 2: 运行测试并确认失败原因是 API 尚未实现**

Run: `E:\Anaconda\envs\newShuMo\python.exe -m pytest tests/test_kinematics.py -v`

Expected: import or attribute failure。

- [ ] **Step 3: 实现数据类和运动方程**

在 `question1/model.py` 中实现 `BombPlan`、`UAVPlan`、`DerivedBomb`，以及：

```python
def direction_from_heading(heading_rad: float) -> np.ndarray:
    return np.asarray([np.cos(heading_rad), np.sin(heading_rad), 0.0], dtype=np.float64)


def missile_hit_time(missile_index: int, data: ProblemData) -> float:
    return float(np.linalg.norm(data.missile_init[missile_index]) / data.missile_speed)


def missile_position(t: float, missile_index: int, data: ProblemData) -> np.ndarray:
    initial = data.missile_init[missile_index]
    if not 0.0 <= t <= missile_hit_time(missile_index, data) + 1e-12:
        raise ValueError("missile time outside valid interval")
    return initial * (1.0 - data.missile_speed * t / np.linalg.norm(initial))


def derive_bomb(uav: UAVPlan, bomb: BombPlan, data: ProblemData) -> DerivedBomb:
    direction = direction_from_heading(uav.heading_rad)
    release_point = data.uav_init[uav.uav_index] + uav.speed * bomb.release_time * direction
    explosion_time = bomb.release_time + bomb.fuse_delay
    explosion_point = data.uav_init[uav.uav_index] + uav.speed * explosion_time * direction
    explosion_point = explosion_point - np.asarray([0.0, 0.0, 0.5 * data.gravity * bomb.fuse_delay**2])
    return DerivedBomb(
        uav_index=uav.uav_index,
        bomb_index=bomb.bomb_index,
        assigned_missile=bomb.assigned_missile,
        release_time=bomb.release_time,
        fuse_delay=bomb.fuse_delay,
        explosion_time=explosion_time,
        release_point=release_point,
        explosion_point=explosion_point,
    )
```

`smoke_center` 必须同时检查起爆前、20 秒寿命、导弹终止和 `center_z + smoke_radius < 0`。

- [ ] **Step 4: 运行运动学测试并确认通过**

Run: `E:\Anaconda\envs\newShuMo\python.exe -m pytest tests/test_kinematics.py -v`

Expected: all tests pass。

- [ ] **Step 5: 提交任务 2**

```powershell
git add -- 2025国赛/A/question1/model.py 2025国赛/A/tests/test_kinematics.py
git commit -m "feat: implement projectile and smoke kinematics"
```

### Task 3: 目标采样、可见性与有限线段距离

**Files:**

- Modify: `2025国赛/A/question1/data_processing.py`
- Modify: `2025国赛/A/question1/model.py`
- Create: `2025国赛/A/tests/test_sampling.py`
- Create: `2025国赛/A/tests/test_geometry.py`

- [ ] **Step 1: 编写几何与采样失败测试**

```python
def test_segment_distance_clips_past_target() -> None:
    missile = np.asarray([0.0, 0.0, 0.0])
    targets = np.asarray([[1.0, 0.0, 0.0]])
    centers = np.asarray([[2.0, 1.0, 0.0]])
    distance, lam = point_to_segments_distance(centers, missile, targets)
    assert lam[0, 0] == pytest.approx(1.0)
    assert distance[0, 0] == pytest.approx(np.sqrt(2.0))


def test_tangent_is_blocked() -> None:
    distance = np.asarray([[10.0]])
    assert line_of_sight_blocked(distance, 10.0)[0, 0]


def test_surface_sampling_is_finite_and_nonempty(problem_data, quick_config) -> None:
    points = sample_cylinder_surface(quick_config["sampling"]["fast"], problem_data)
    assert points.ndim == 2 and points.shape[1] == 3
    assert len(points) > 0
    assert np.isfinite(points).all()
```

- [ ] **Step 2: 运行测试并观察失败**

Run: `E:\Anaconda\envs\newShuMo\python.exe -m pytest tests/test_sampling.py tests/test_geometry.py -v`

Expected: missing functions。

- [ ] **Step 3: 实现圆柱采样与可见点过滤**

实现 `sample_cylinder_surface(profile, data)`，包括侧面方位角×高度、顶面半径×方位角、顶面边界圆和上下轮廓点，并使用坐标舍入键去重。

实现 `visible_target_points(missile_pos, surface_points, data)`：用射线—有限圆柱首次相交参数判断候选表面点是否在导弹可见侧；若结果为空则抛出 `RuntimeError("visible target sample is empty")`。

- [ ] **Step 4: 实现广播线段距离**

```python
def point_to_segments_distance(
    smoke_centers: np.ndarray,
    missile_pos: np.ndarray,
    target_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    centers = np.atleast_2d(np.asarray(smoke_centers, dtype=np.float64))
    points = np.atleast_2d(np.asarray(target_points, dtype=np.float64))
    line = points - missile_pos
    denom = np.einsum("nd,nd->n", line, line)
    if np.any(denom <= 0.0):
        raise ValueError("target point coincides with missile position")
    relative = centers - missile_pos
    lam = np.einsum("bd,nd->bn", relative, line) / denom[None, :]
    lam = np.clip(lam, 0.0, 1.0)
    closest = missile_pos + lam[..., None] * line[None, :, :]
    distance = np.linalg.norm(centers[:, None, :] - closest, axis=-1)
    return distance, lam
```

- [ ] **Step 5: 运行相关测试和回归**

Run: `E:\Anaconda\envs\newShuMo\python.exe -m pytest tests/test_sampling.py tests/test_geometry.py tests/test_kinematics.py -v`

Expected: all tests pass。

- [ ] **Step 6: 提交任务 3**

```powershell
git add -- 2025国赛/A/question1 2025国赛/A/tests/test_sampling.py 2025国赛/A/tests/test_geometry.py
git commit -m "feat: add target visibility and line-of-sight geometry"
```

### Task 4: 多云团联合覆盖、区间评价与可行性

**Files:**

- Create: `2025国赛/A/question1/evaluation.py`
- Create: `2025国赛/A/tests/test_coverage.py`
- Create: `2025国赛/A/tests/test_intervals.py`
- Create: `2025国赛/A/tests/test_feasibility.py`

- [ ] **Step 1: 编写联合覆盖量词回归测试**

```python
def test_two_clouds_can_jointly_cover_two_points() -> None:
    blocked = np.asarray([[True, False], [False, True]])
    np.testing.assert_array_equal(point_coverage(blocked), [True, True])
    assert joint_blocked(blocked)
    np.testing.assert_array_equal(standalone_blocked(blocked), [False, False])


def test_no_active_cloud_is_not_blocked() -> None:
    blocked = np.zeros((0, 2), dtype=bool)
    assert not joint_blocked(blocked)
```

- [ ] **Step 2: 编写区间与约束失败测试**

```python
def test_merge_and_measure_discontinuous_intervals() -> None:
    merged = merge_intervals([(1.0, 2.0), (1.5, 3.0), (5.0, 5.0), (6.0, 7.0)], 1e-9)
    assert merged == [(1.0, 3.0), (5.0, 5.0), (6.0, 7.0)]
    assert interval_length(merged) == pytest.approx(3.0)


def test_question3_requires_exactly_three_bombs(problem_data) -> None:
    report = check_solution([make_plan_with_bomb_count(2)], question_id=3, data=problem_data)
    assert not report.feasible
    assert report.violations["wrong_bomb_count"] == pytest.approx(1.0)
```

- [ ] **Step 3: 运行测试并确认失败**

Run: `E:\Anaconda\envs\newShuMo\python.exe -m pytest tests/test_coverage.py tests/test_intervals.py tests/test_feasibility.py -v`

Expected: missing APIs。

- [ ] **Step 4: 实现覆盖函数和结构化违规报告**

实现 `point_coverage`、`joint_blocked`、`coverage_ratio`、`standalone_blocked`。对形状不是 `[B,N]`、`N=0` 的输入抛出明确异常；`B=0` 返回未遮蔽。

实现 `FeasibilityReport(feasible, violations)`、`check_uav_plan`、`check_solution`，违规键严格包含：`speed_low`、`speed_high`、`negative_release_time`、`late_explosion`、`negative_explosion_height`、`release_gap`、`wrong_bomb_count`、`multiple_routes_per_uav`。

- [ ] **Step 5: 实现区间扫描、细化和总时长评价**

实现：

```python
def merge_intervals(intervals: Iterable[tuple[float, float]], merge_tol: float) -> list[tuple[float, float]]:
    ordered = sorted((float(left), float(right)) for left, right in intervals)
    merged: list[tuple[float, float]] = []
    for left, right in ordered:
        if right < left:
            raise ValueError("interval right endpoint is smaller than left endpoint")
        if not merged or left > merged[-1][1] + merge_tol:
            merged.append((left, right))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
    return merged


def interval_length(intervals: Iterable[tuple[float, float]]) -> float:
    return float(sum(max(0.0, right - left) for left, right in intervals))


def refine_boundary(left: float, right: float, margin_fn: Callable[[float], float], root_tol: float) -> float:
    return float(brentq(margin_fn, left, right, xtol=root_tol, rtol=4.0 * np.finfo(float).eps))
```

`evaluate_solution` 的实现顺序固定为：调用 `check_solution`；不可行时返回带违规量的 `EvaluationResult`；对每枚导弹收集起爆、寿命、入地和导弹终止事件；逐事件段生成时间网格；逐时刻计算活动云团、可见目标点、`distance[B,N]`、逐点最小距离和联合裕量；对符号变化调用 `refine_boundary`；对端点同号但中点更低的区段递归二分到 `max_refinement_depth`；最后调用 `merge_intervals` 和 `interval_length`。返回对象必须包含每枚导弹区间、时长、覆盖率摘要、边界残差、评价配置和诊断数据。

- [ ] **Step 6: 运行全套内核测试**

Run: `E:\Anaconda\envs\newShuMo\python.exe -m pytest tests/test_coverage.py tests/test_intervals.py tests/test_feasibility.py tests/test_geometry.py tests/test_kinematics.py -v`

Expected: all tests pass。

- [ ] **Step 7: 提交任务 4**

```powershell
git add -- 2025国赛/A/question1/evaluation.py 2025国赛/A/tests
git commit -m "feat: implement joint coverage and interval evaluation"
```

### Task 5: 问题 1 端到端求解与英文图表

**Files:**

- Create: `2025国赛/A/question1/main.py`
- Create: `2025国赛/A/question1/visualization.py`
- Modify: `2025国赛/A/tests/test_cli_smoke.py`

- [ ] **Step 1: 编写 Q1 CLI 失败测试**

```python
def test_question1_quick_run_creates_structured_outputs(project_root: Path, tmp_path: Path) -> None:
    result = subprocess.run(
        [PYTHON, str(project_root / "question1" / "main.py"), "--config", str(project_root / "configs" / "quick.json"), "--output-root", str(tmp_path)],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    run_dirs = list((tmp_path / "question1").iterdir())
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "manifest.json").exists()
    assert (run_dirs[0] / "raw_solution.json").exists()
    assert (run_dirs[0] / "intervals.csv").exists()
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `E:\Anaconda\envs\newShuMo\python.exe -m pytest tests/test_cli_smoke.py::test_question1_quick_run_creates_structured_outputs -v`

Expected: missing `main.py`。

- [ ] **Step 3: 实现 Q1 入口和结果保存**

`main.py` 使用 `argparse` 接受 `--config`、`--output-root`、`--run-id`、`--no-plots`。固定策略为 FY1、航向 $\pi$、速度 120、投放 1.5、延时 3.6。依次运行快速和验证评价，输出关键中间结果到控制台，并将投放点、起爆点、区间、时长、误差和边界残差写入结构化文件。

- [ ] **Step 4: 实现 Q1 图表**

`visualization.py` 实现 `plot_trajectory`、`plot_margin_history`、`plot_intervals`、`plot_convergence`。标题、图例和坐标轴使用英文，保存 PNG 与 SVG，并把绘图原始数据保存为 CSV。

- [ ] **Step 5: 运行 Q1 测试与真实快速配置**

Run:

```powershell
E:\Anaconda\envs\newShuMo\python.exe -m pytest tests/test_cli_smoke.py::test_question1_quick_run_creates_structured_outputs -v
E:\Anaconda\envs\newShuMo\python.exe question1/main.py --config configs/quick.json
```

Expected: pytest passes；命令退出码 0；实际数值只从生成文件读取和报告。

- [ ] **Step 6: 提交任务 5**

```powershell
git add -- 2025国赛/A/question1 2025国赛/A/tests/test_cli_smoke.py
git commit -m "feat: add question one end-to-end solver"
```

### Task 6: 通用连续 PSO、DE 包装与可复现性

**Files:**

- Create: `2025国赛/A/question2/__init__.py`
- Create: `2025国赛/A/question2/model.py`
- Create: `2025国赛/A/tests/test_optimizers.py`

- [ ] **Step 1: 编写人工目标与固定种子失败测试**

```python
def sphere_objective(x: np.ndarray) -> float:
    return -float(np.dot(x, x))


def test_pso_is_reproducible() -> None:
    bounds = np.asarray([[-5.0, 5.0], [-5.0, 5.0]])
    first = solve_pso(sphere_objective, bounds, np.random.default_rng(2025), particles=12, iterations=15)
    second = solve_pso(sphere_objective, bounds, np.random.default_rng(2025), particles=12, iterations=15)
    np.testing.assert_allclose(first.best_position, second.best_position)
    assert first.best_score == pytest.approx(second.best_score)


def test_de_wrapper_returns_feasible_candidate() -> None:
    result = solve_de(sphere_objective, [(-5.0, 5.0), (-5.0, 5.0)], seed=2025, maxiter=5, popsize=5)
    assert np.all(result.best_position >= -5.0)
    assert np.all(result.best_position <= 5.0)
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `E:\Anaconda\envs\newShuMo\python.exe -m pytest tests/test_optimizers.py -v`

Expected: missing optimizer functions。

- [ ] **Step 3: 实现自定义 PSO 与 DE 包装**

`solve_pso` 使用惯性权重线性下降、个体最优、群体最优、速度截断和位置裁剪；每代保存最佳值、均值、标准差和粒子离散度。目标接口统一为“越大越好”。

`solve_de` 调用 `scipy.optimize.differential_evolution`，内部对目标取负，显式传入种子、边界、`polish=False`，并通过 callback 保存历史。两者均返回 `OptimizerResult(best_position, best_score, history, evaluations, termination_reason)`。

- [ ] **Step 4: 运行优化器测试**

Run: `E:\Anaconda\envs\newShuMo\python.exe -m pytest tests/test_optimizers.py -v`

Expected: all tests pass。

- [ ] **Step 5: 提交任务 6**

```powershell
git add -- 2025国赛/A/question2 2025国赛/A/tests/test_optimizers.py
git commit -m "feat: add reproducible pso and differential evolution"
```

### Task 7: 问题 2 单机单弹优化

**Files:**

- Create: `2025国赛/A/question2/data_processing.py`
- Modify: `2025国赛/A/question2/model.py`
- Create: `2025国赛/A/question2/evaluation.py`
- Create: `2025国赛/A/question2/visualization.py`
- Create: `2025国赛/A/question2/main.py`
- Modify: `2025国赛/A/tests/test_cli_smoke.py`

- [ ] **Step 1: 编写 Q2 解码和 CLI 失败测试**

```python
def test_decode_q2_candidate_derives_points(problem_data) -> None:
    plan = decode_q2_candidate(np.asarray([np.pi, 120.0, 1.5, 3.6]), problem_data)
    assert plan.uav_index == 0
    assert len(plan.bombs) == 1


def test_question2_quick_run_is_reproducible(project_root: Path, tmp_path: Path) -> None:
    first = run_question(project_root, 2, tmp_path, run_id="first")
    second = run_question(project_root, 2, tmp_path, run_id="second")
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    first_payload = json.loads((tmp_path / "question2" / "first" / "raw_solution.json").read_text(encoding="utf-8"))
    second_payload = json.loads((tmp_path / "question2" / "second" / "raw_solution.json").read_text(encoding="utf-8"))
    assert first_payload["decision_variables"] == second_payload["decision_variables"]
    assert first_payload["verified_objective"] == second_payload["verified_objective"]
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `E:\Anaconda\envs\newShuMo\python.exe -m pytest tests/test_cli_smoke.py -k question2 -v`

Expected: missing Q2 APIs/entry point。

- [ ] **Step 3: 实现候选生成、解码和目标函数**

变量范围：`heading_rad ∈ [-pi, pi]`、`speed ∈ [70,140]`、`release_time ∈ [0,t_hit]`、`fuse_delay ∈ [0,sqrt(2*z0/g)]`。超出 `release_time + fuse_delay <= t_hit` 的候选返回结构化不可行评分。

`generate_initial_candidates` 合并 Latin Hypercube 与几何反演初值；`candidate_objective` 返回完整遮蔽时长和小权重覆盖率积分组成的搜索分数，同时单独保留正式时长。

- [ ] **Step 4: 实现 PSO/DE 双算法求解和验证回代**

运行配置中相同的函数评价预算；合并两算法候选，按周期角差、速度、时序和区间签名去重；前若干候选使用 Q1 验证评价器重新排序。保存两算法历史、函数评价次数、最优差异和最终候选来源。

- [ ] **Step 5: 实现 Q2 图表和 CLI**

图表包括三维轨迹、遮蔽区间、PSO/DE 历史和候选散点；CLI 参数与 Q1 一致。

- [ ] **Step 6: 运行 Q2 测试与快速配置**

Run:

```powershell
E:\Anaconda\envs\newShuMo\python.exe -m pytest tests/test_cli_smoke.py -k question2 -v
E:\Anaconda\envs\newShuMo\python.exe question2/main.py --config configs/quick.json
```

Expected: deterministic outputs for identical seeds；实际最优值来自运行文件。

- [ ] **Step 7: 提交任务 7**

```powershell
git add -- 2025国赛/A/question2 2025国赛/A/tests/test_cli_smoke.py
git commit -m "feat: implement question two single bomb optimization"
```

### Task 8: Excel 模板校验与通用导出

**Files:**

- Create: `2025国赛/A/question3/__init__.py`
- Create: `2025国赛/A/question3/data_processing.py`
- Create: `2025国赛/A/tests/test_excel_export.py`

- [ ] **Step 1: 编写模板保护和格式失败测试**

```python
def test_export_does_not_overwrite_template(project_root: Path, tmp_path: Path) -> None:
    template = project_root / "00_赛题资料" / "附件" / "result1.xlsx"
    before = template.read_bytes()
    output = export_result1(template, tmp_path / "result1.xlsx", example_rows())
    assert output.exists()
    assert template.read_bytes() == before


def test_modified_header_is_rejected(tmp_path: Path) -> None:
    bad = make_modified_template(tmp_path)
    with pytest.raises(ValueError, match="header"):
        inspect_excel_template(bad, question_id=3)
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `E:\Anaconda\envs\newShuMo\python.exe -m pytest tests/test_excel_export.py -v`

Expected: missing export functions。

- [ ] **Step 3: 实现模板检查和复制填写**

实现 `inspect_excel_template`、`validate_template`、`copy_template`、`write_rows`、`read_exported_rows`。程序必须定位实际表头行，保持其余单元格格式，写入值使用数值类型并设置 `0.000` 格式。航向转换使用 `(degrees(theta) % 360.0)`。

- [ ] **Step 4: 实现顺序新增时长接口**

`sequential_marginal_durations(plans, assigned_missiles, evaluator)` 按投放时刻、无人机编号、烟幕编号稳定排序，逐枚重新评价主要目标导弹，计算非负新增值；小于数值容差的负值归零并记录，大于容差则抛出异常。

- [ ] **Step 5: 运行 Excel 测试**

Run: `E:\Anaconda\envs\newShuMo\python.exe -m pytest tests/test_excel_export.py -v`

Expected: all tests pass。

- [ ] **Step 6: 提交任务 8**

```powershell
git add -- 2025国赛/A/question3 2025国赛/A/tests/test_excel_export.py
git commit -m "feat: add protected excel template export"
```

### Task 9: 问题 3 单机三弹协同优化

**Files:**

- Create: `2025国赛/A/question3/model.py`
- Create: `2025国赛/A/question3/evaluation.py`
- Create: `2025国赛/A/question3/visualization.py`
- Create: `2025国赛/A/question3/main.py`
- Modify: `2025国赛/A/tests/test_feasibility.py`
- Modify: `2025国赛/A/tests/test_cli_smoke.py`

- [ ] **Step 1: 编写有序投放解码失败测试**

```python
def test_decode_ordered_release_times_enforces_one_second_gap() -> None:
    decoded = decode_q3_candidate(np.asarray([0.0, 100.0, 2.0, 0.0, 0.5, 0.0, 1.0, 1.0]))
    releases = [bomb.release_time for bomb in decoded.bombs]
    assert releases[1] - releases[0] >= 1.0
    assert releases[2] - releases[1] >= 1.0
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `E:\Anaconda\envs\newShuMo\python.exe -m pytest tests/test_feasibility.py -k ordered -v`

Expected: missing decoder。

- [ ] **Step 3: 实现八维参数化与协同目标**

使用 `(theta, speed, base_release, gap_slack_2, gap_slack_3, tau1, tau2, tau3)`，解码为 `t1=base`、`t2=t1+1+slack2`、`t3=t2+1+slack3`。所有三枚弹共享同一航向速度。目标使用联合逐点覆盖时长，不使用单弹区间简单并集。

- [ ] **Step 4: 实现 PSO/DE、分块精修和贡献评价**

先联合搜索，再交替固定航线优化时序、固定时序优化航线。实现独立诊断时长、顺序新增时长、移除边际贡献。快速配置使用有限轮次，竞赛配置增加种子和预算。

- [ ] **Step 5: 实现 Q3 CLI、图表和 result1.xlsx**

输出三弹轨迹、联合区间、覆盖热图、算法历史和贡献图；复制模板并正好写入 3 条有效记录，回读后重建方案并比较舍入前后时长。

- [ ] **Step 6: 运行 Q3 测试与快速配置**

Run:

```powershell
E:\Anaconda\envs\newShuMo\python.exe -m pytest tests/test_feasibility.py tests/test_excel_export.py tests/test_cli_smoke.py -k "question3 or ordered or result1" -v
E:\Anaconda\envs\newShuMo\python.exe question3/main.py --config configs/quick.json
```

Expected: tests pass；生成结构化结果和复制后的 `result1.xlsx`。

- [ ] **Step 7: 提交任务 9**

```powershell
git add -- 2025国赛/A/question3 2025国赛/A/tests
git commit -m "feat: implement question three cooperative optimization"
```

### Task 10: 问题 4 三机单弹协同优化

**Files:**

- Create: `2025国赛/A/question4/__init__.py`
- Create: `2025国赛/A/question4/data_processing.py`
- Create: `2025国赛/A/question4/model.py`
- Create: `2025国赛/A/question4/evaluation.py`
- Create: `2025国赛/A/question4/visualization.py`
- Create: `2025国赛/A/question4/main.py`
- Modify: `2025国赛/A/tests/test_cli_smoke.py`
- Modify: `2025国赛/A/tests/test_excel_export.py`

- [ ] **Step 1: 编写三机解码和 result2 失败测试**

```python
def test_q4_candidate_contains_fy1_to_fy3(problem_data) -> None:
    plans = decode_q4_candidate(np.zeros(12), problem_data, repair=True)
    assert [plan.uav_index for plan in plans] == [0, 1, 2]
    assert all(len(plan.bombs) == 1 for plan in plans)


def test_result2_contains_each_required_uav_once(project_root: Path, tmp_path: Path) -> None:
    template = project_root / "00_赛题资料" / "附件" / "result2.xlsx"
    output = export_result2(template, tmp_path / "result2.xlsx", example_result2_rows())
    rows = read_exported_rows(output, question_id=4)
    assert [row["无人机编号"] for row in rows] == ["FY1", "FY2", "FY3"]
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `E:\Anaconda\envs\newShuMo\python.exe -m pytest tests/test_cli_smoke.py tests/test_excel_export.py -k question4 -v`

Expected: missing Q4 modules。

- [ ] **Step 3: 实现候选库、组合筛选和十二维目标**

为 FY1—FY3 分别生成单弹候选，按变量距离、区间签名和逐点覆盖签名去重；使用有限束搜索选择初始组合。12 维联合变量按每架 `(theta,speed,release,tau)` 排列，每架独立约束。

- [ ] **Step 4: 实现分块 PSO/DE 与联合高精度回代**

轮换优化 FY1、FY2、FY3，使用多个更新顺序；对最佳组合执行联合 PSO 与 DE。输出每机移除边际和全局顺序新增时长。

- [ ] **Step 5: 实现 Q4 CLI、图表和 result2.xlsx**

Excel 必须保留 FY1、FY2、FY3 且各一条。图表包括三机轨迹、联合区间、覆盖热图和算法比较。

- [ ] **Step 6: 运行 Q4 测试与快速配置**

Run:

```powershell
E:\Anaconda\envs\newShuMo\python.exe -m pytest tests/test_cli_smoke.py tests/test_excel_export.py -k question4 -v
E:\Anaconda\envs\newShuMo\python.exe question4/main.py --config configs/quick.json
```

Expected: tests pass；生成 `result2.xlsx` 副本并通过回读。

- [ ] **Step 7: 提交任务 10**

```powershell
git add -- 2025国赛/A/question4 2025国赛/A/tests
git commit -m "feat: implement question four multi-uav optimization"
```

### Task 11: 问题 5 路线库、整数 PSO 与字典序优化

**Files:**

- Create: `2025国赛/A/question5/__init__.py`
- Create: `2025国赛/A/question5/data_processing.py`
- Create: `2025国赛/A/question5/model.py`
- Create: `2025国赛/A/question5/evaluation.py`
- Create: `2025国赛/A/question5/visualization.py`
- Create: `2025国赛/A/question5/main.py`
- Modify: `2025国赛/A/tests/test_optimizers.py`
- Modify: `2025国赛/A/tests/test_excel_export.py`
- Modify: `2025国赛/A/tests/test_cli_smoke.py`

- [ ] **Step 1: 编写人工路线组合和字典序失败测试**

```python
def test_integer_route_selection_matches_enumeration() -> None:
    coverage = artificial_route_coverage()
    expected = enumerate_lexicographic_optimum(coverage, epsilon_sum=0.0)
    result = solve_route_pso(coverage, np.random.default_rng(2025), particles=30, iterations=50, epsilon_sum=0.0)
    assert result.sum_objective == pytest.approx(expected.sum_objective)
    assert result.min_duration == pytest.approx(expected.min_duration)


def test_route_particle_decoding_is_always_legal() -> None:
    position = np.asarray([-10.2, 0.49, 2.51, 99.0, 1.5])
    decoded = decode_route_particle(position, np.asarray([3, 2, 4, 1, 5]))
    assert np.array_equal(decoded, [0, 0, 3, 0, 2])
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `E:\Anaconda\envs\newShuMo\python.exe -m pytest tests/test_optimizers.py -k route -v`

Expected: missing route optimizer APIs。

- [ ] **Step 3: 实现完整路线对象和候选生成**

`Route` 固定一架无人机的一组航向速度及 0—3 枚弹。构造空路线、按主要目标生成的单弹候选和共享航线多弹组合；检查投放间隔、起爆高度和资源上限。覆盖签名按导弹、时间块和目标点位打包，删除重复路线及被同机另一条路线完全支配且资源不更少的路线。

- [ ] **Step 4: 实现整数编码 PSO 两阶段选择**

粒子长度为 5，每维为对应无人机路线编号。连续更新后使用稳定取整和合法范围截断。第一阶段比较 `(J_sum, J_min)` 时只以 `J_sum` 为主；第二阶段将 `J_sum < J_sum_star - epsilon_J` 视为不可行，再最大化 `J_min`。保存群体最佳、路线多样性、函数评价次数和总和容差残差。

- [ ] **Step 5: 实现 DE 连续精修与所有实际遮蔽计分**

固定选中路线的弹数和主要任务标签，优化共享航向速度及各弹时序。评价时每个云团都进入 M1—M3 的几何判定，不用任务标签门控。最终使用连续验证评价器得到 `T1,T2,T3,J_sum,J_min`。

- [ ] **Step 6: 实现 Q5 CLI、图表和 result3.xlsx**

输出路线库摘要、选择日志、三导弹时长、均衡性、覆盖热图、轨迹和敏感性数据。Excel 每机不超过 3 行，未使用预留行保持空白，主要目标标签来自路线定义；回读后重建所有路线。

- [ ] **Step 7: 运行人工最优测试、Q5 冒烟测试和快速配置**

Run:

```powershell
E:\Anaconda\envs\newShuMo\python.exe -m pytest tests/test_optimizers.py tests/test_excel_export.py tests/test_cli_smoke.py -k "route or question5 or result3" -v
E:\Anaconda\envs\newShuMo\python.exe question5/main.py --config configs/quick.json
```

Expected: 人工问题与枚举一致；快速运行生成当前最好可行解和真实终止原因。

- [ ] **Step 8: 提交任务 11**

```powershell
git add -- 2025国赛/A/question5 2025国赛/A/tests
git commit -m "feat: implement question five lexicographic route optimization"
```

### Task 12: 竞赛配置、全量验证与代码实现说明更新

**Files:**

- Modify: `2025国赛/A/configs/quick.json`
- Modify: `2025国赛/A/configs/competition.json`
- Modify: `2025国赛/A/04_模型求解/代码实现说明.md`
- Modify: `2025国赛/A/tests/test_cli_smoke.py`

- [ ] **Step 1: 根据 Q1 实际收敛结果锁定配置**

运行至少三级时间步长和三级目标采样，比较相邻级时长差、区间端点和最大边界残差。将快速配置设为可在短时间内完成五问冒烟运行的参数，将竞赛配置设为更严格的时间/空间容差和更大优化预算。不得把未运行的竞赛配置性能写入文档。

- [ ] **Step 2: 运行全量测试**

Run:

```powershell
E:\Anaconda\envs\newShuMo\python.exe -m pytest tests -v
```

Expected: zero failures, zero errors。

- [ ] **Step 3: 运行五问快速配置**

Run:

```powershell
E:\Anaconda\envs\newShuMo\python.exe question1/main.py --config configs/quick.json
E:\Anaconda\envs\newShuMo\python.exe question2/main.py --config configs/quick.json
E:\Anaconda\envs\newShuMo\python.exe question3/main.py --config configs/quick.json
E:\Anaconda\envs\newShuMo\python.exe question4/main.py --config configs/quick.json
E:\Anaconda\envs\newShuMo\python.exe question5/main.py --config configs/quick.json
```

Expected: five commands exit 0；每个运行目录包含 manifest、config、input snapshot、raw solution、intervals、history、convergence 和 figures；Q3—Q5 含 Excel 副本和回读报告。

- [ ] **Step 4: 受控验证竞赛配置入口**

为 CLI 增加 `--validate-config-only` 和 `--budget-scale`。先运行：

```powershell
E:\Anaconda\envs\newShuMo\python.exe question5/main.py --config configs/competition.json --validate-config-only
E:\Anaconda\envs\newShuMo\python.exe question5/main.py --config configs/competition.json --budget-scale 0.01 --no-plots
```

Expected: 配置验证退出 0；受控启动完成小预算流程，但输出标记 `budget_scale=0.01`，不得视为竞赛级最终结果。

- [ ] **Step 5: 检查结构化结果和 Excel 回读**

编写/运行测试遍历最新快速运行目录，校验 JSON 可解析、CSV 非空、数值有限、区间长度非负、轨迹满足约束、图表文件存在、模板原件哈希未改变、输出 Excel 可回读。

- [ ] **Step 6: 更新代码实现说明**

保持用户要求的六个 Markdown 章节，更新：实际代码文件树、公共接口复用方式、解释器与依赖、快速/竞赛配置区别、完整运行命令、输入文件、输出目录、已实际完成的测试和运行、未完成的高成本竞赛级全量运行、可选预处理文件缺失说明。

- [ ] **Step 7: 最终要求核对**

逐项核对用户代码要求：配置、数据读取、预处理、建模、求解、评价、可视化、结果保存、注释、随机种子、异常处理、中间结果、空值/维度/边界/收敛检查、英文图表、结构化结果和不虚构结果。

- [ ] **Step 8: 提交最终实现**

```powershell
git add -- 2025国赛/A/configs 2025国赛/A/question1 2025国赛/A/question2 2025国赛/A/question3 2025国赛/A/question4 2025国赛/A/question5 2025国赛/A/tests 2025国赛/A/04_模型求解/代码实现说明.md
git commit -m "feat: complete smoke interference optimization suite"
```

## 4. 关键结论

1. 实施顺序必须先完成并验证统一评价内核，再扩展 Q2—Q5 优化器。
2. 每项新行为先通过失败测试证明测试有效，再写最小生产实现。
3. Q2—Q4 使用 PSO 与 DE 独立搜索，Q5 使用路线库、整数编码 PSO 两阶段选择与 DE 连续精修。
4. 快速配置用于完整流程复现；竞赛配置必须经过实际全量运行后才能产生竞赛级结论。
5. Excel 导出始终复制模板、写入三位小数、回读重建，并保留完整精度结构化原始解。

## 5. 待解决问题

1. 快速和竞赛配置的最终数值参数需由 Q1 收敛试验及实际运行时间确定。
2. Q5 路线库规模、覆盖位集尺寸和整数 PSO 预算需根据实际性能逐级扩充。
3. 当前仓库已有用户暂存及未提交变更；执行期间只能提交本任务文件，不能改写或混入无关变更。

## 6. 与后续步骤的衔接

计划批准后，使用 `superpowers:subagent-driven-development` 按任务逐项执行，或使用 `superpowers:executing-plans` 在当前会话分批执行。每批完成后运行相应测试并核对实际输出，最终执行 Task 12 的完整验证与文档更新。
