"""Hygiene rules for every notebook in the module (and naming rules for every .py file).

* Notebooks are saved with outputs cleared and the upstream python3 kernelspec.
* Markdown avoids a list of banned words and uses at most one em-dash per paragraph.
* Names never describe history or implementation (naive, improved, enhanced, _v2, new_).
* The setup cell imports model IDs from Foundational Evaluations with an ImportError fallback.
* The setup cell sets Nova 2 Lite as the agent under test and checks judge independence.
* One code cell is tagged ``separation-summary``.

Set FSI_EVAL_NOTEBOOK_DIR to check notebooks in another folder.
"""

import json
import os
import re
from pathlib import Path

import pytest

MODULE_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = Path(os.environ.get("FSI_EVAL_NOTEBOOK_DIR", MODULE_ROOT))
NOTEBOOKS = sorted(NOTEBOOK_DIR.glob("*.ipynb"))

BANNED_WORDS = re.compile(
    r"\b(load-bearing|surfaces|genuinely|honest|honestly|plainly|mint|plumbing|seams?)\b", re.IGNORECASE)
BANNED_NAMES = re.compile(r"(naive|improved|enhanced|_v2\b|\bnew_)", re.IGNORECASE)


def cells(path, kind):
    """Sources of one kind of cell ("markdown" or "code") in a notebook."""
    notebook = json.loads(path.read_text())
    return [("".join(c["source"]), c) for c in notebook["cells"] if c["cell_type"] == kind]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_outputs_cleared_and_kernelspec(path):
    notebook = json.loads(path.read_text())
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    for _, cell in cells(path, "code"):
        assert cell["outputs"] == []
        assert cell["execution_count"] is None


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_markdown_register(path):
    for text, _ in cells(path, "markdown"):
        assert not BANNED_WORDS.search(text), BANNED_WORDS.search(text).group(0)
        for paragraph in re.split(r"\n\s*\n", text):
            assert paragraph.count("—") <= 1, paragraph[:80]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_code_names(path):
    for text, _ in cells(path, "code"):
        assert not BANNED_NAMES.search(text), BANNED_NAMES.search(text).group(0)


# Folders that hold installed or generated code, not module code. The README creates
# .venv inside the module folder, so the naming rule must not walk into it.
NON_MODULE_DIRS = {".venv", "venv", "env", ".env", "site-packages", "node_modules", ".tox", "__pycache__",
                   "build", "dist", ".git", ".ipynb_checkpoints"}


def module_python_files(root):
    """Every .py file under ``root`` that belongs to the module, skipping virtual environments and build output."""
    return sorted(path for path in root.rglob("*.py")
                  if not NON_MODULE_DIRS & set(path.relative_to(root).parts[:-1]))


def test_module_python_files_skip_virtual_environments(tmp_path):
    for folder in (".venv/lib/python3.14/site-packages", "venv/lib", "env/lib", "fsi_eval"):
        (tmp_path / folder).mkdir(parents=True)
    for path in (".venv/lib/python3.14/site-packages/threadpoolctl.py", "venv/lib/a.py", "env/lib/b.py",
                 "fsi_eval/module.py"):
        (tmp_path / path).write_text("new_ = 1\n")
    assert module_python_files(tmp_path) == [tmp_path / "fsi_eval" / "module.py"]


def test_python_names():
    for path in module_python_files(MODULE_ROOT):
        if path.name == "test_notebook_hygiene.py":
            continue  # this file names the banned patterns
        match = BANNED_NAMES.search(path.read_text())
        assert not match, (path, match and match.group(0))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_setup_cell_imports_model_config_with_fallback(path):
    setup = next(text for text, _ in cells(path, "code") if "model_config" in text)
    assert "Foundational Evaluations" in setup
    assert re.search(r"try:.*except ImportError", setup, re.DOTALL)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_agent_under_test_is_nova_2_lite(path):
    # The module standardizes on Nova 2 Lite as the agent in every notebook; the judge comes from model_config.
    setup = next(text for text, _ in cells(path, "code") if "model_config" in text)
    assert re.search(r'^AGENT_MODEL_ID = "us\.amazon\.nova-2-lite-v1:0"', setup, re.MULTILINE)
    assert "assert_judge_independent(AGENT_MODEL_ID, JUDGE_MODEL_ID)" in setup


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_separation_summary_cell_is_tagged(path):
    tagged = [c for _, c in cells(path, "code") if "separation-summary" in c.get("metadata", {}).get("tags", [])]
    assert len(tagged) == 1
    assert "SEPARATION_SUMMARY" in "".join(tagged[0]["source"])
