"""烟幕干扰求解器的共享 pytest 夹具。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def project_root() -> Path:
    """返回包含各题目包和配置文件的目录。"""

    return PROJECT_ROOT


@pytest.fixture
def quick_config(project_root: Path) -> dict[str, Any]:
    """加载大多数单元测试使用的可复现快速配置文件。"""

    from question1.data_processing import load_config

    return load_config(project_root / "configs" / "quick.json")


@pytest.fixture
def problem_data(quick_config: dict[str, Any]):
    """返回从快速配置文件中加载并验证的问题常量。"""

    from question1.data_processing import load_problem_data

    return load_problem_data(quick_config)
