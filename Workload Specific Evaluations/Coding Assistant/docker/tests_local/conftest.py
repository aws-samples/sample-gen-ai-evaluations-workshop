"""Local checks of the module code. They need no Docker, no Harbor, and no AWS credentials.

Run from the module directory: python -m pytest tests_local
"""

import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_DIR = Path(__file__).resolve().parent.parent
DATASETS = MODULE_DIR / "datasets"
PR_TASKS = sorted((DATASETS / "awslabs-mcp-pr-review" / "tasks").iterdir())
CODING_TASKS = sorted((DATASETS / "strands-evals-coding" / "tasks").iterdir())

sys.path.insert(0, str(MODULE_DIR))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses with slots look the module up by name
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def score():
    return load_module("pr_review_score", DATASETS / "_shared" / "pr-review" / "tests" / "score.py")


@pytest.fixture(scope="session")
def metric():
    return load_module("pr_review_metric", DATASETS / "awslabs-mcp-pr-review" / "metric.py")


@pytest.fixture(scope="session")
def sync():
    return load_module("shared_sync", DATASETS / "_shared" / "sync.py")
