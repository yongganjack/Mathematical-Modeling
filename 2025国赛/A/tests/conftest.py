"""Shared pytest fixtures for the smoke-interference solver."""

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
    """Return the directory containing the question packages and configs."""

    return PROJECT_ROOT


@pytest.fixture
def quick_config(project_root: Path) -> dict[str, Any]:
    """Load the reproducible quick profile used by most unit tests."""

    from question1.data_processing import load_config

    return load_config(project_root / "configs" / "quick.json")


@pytest.fixture
def problem_data(quick_config: dict[str, Any]):
    """Return validated problem constants loaded from the quick profile."""

    from question1.data_processing import load_problem_data

    return load_problem_data(quick_config)
