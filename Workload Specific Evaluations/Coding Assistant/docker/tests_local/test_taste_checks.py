"""The taste judge of the coding tasks (notebook 03) answers yes/no checks, not 1-5 scores."""

import importlib.util
import json
import sys
import tomllib
import types
from pathlib import Path

import pytest

LIB = Path(__file__).resolve().parents[1] / "datasets" / "_shared" / "coding" / "tests"


@pytest.fixture
def judges(tmp_path, monkeypatch):
    """judges.py with a fake bedrock module: the 'model' answers from a dict, and files go to tmp_path."""
    written, answers = {}, {}

    class FakeConversation:
        def __init__(self, role, system, tools, submit_name, schema):
            self.schema, self.usage = schema, {"input_tokens": 0, "output_tokens": 0}

        def ask(self, prompt):
            return answers

    fake = types.SimpleNamespace(LOGS=tmp_path, REPO=tmp_path, TESTS=tmp_path, Conversation=FakeConversation,
                                 read_only_tools=lambda paths: [], write_json=lambda name, data: written.__setitem__(name, data))
    monkeypatch.setitem(sys.modules, "bedrock", fake)
    (tmp_path / "judge").mkdir()
    (tmp_path / "judge" / "instruction.md").write_text("Fix the bug.\n## General instructions\nx")
    (tmp_path / "judge" / "reference.patch").write_text("diff --git a/x.py b/x.py\n+y = 1\n")
    (tmp_path / "agent.patch").write_text("diff --git a/x.py b/x.py\n+y = 2\n")
    spec = importlib.util.spec_from_file_location("judges_under_test", LIB / "lib" / "judges.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.REFERENCE = tmp_path / "judge" / "reference.patch"
    return module, answers, written


def test_taste_is_a_pass_rate_of_yes_no_checks(judges):
    module, answers, written = judges
    yes = set(module.RELATIVE[:3]) | set(module.PRACTICE[:2])  # 3 of 5, and 2 of 5
    answers.update({k: {"passed": k in yes, "reason": "r"} for k in module.RELATIVE + module.PRACTICE})
    module.taste()
    taste = written["taste.json"]
    assert taste["relative_taste"] == 0.6 and taste["practice_alignment"] == 0.4


def test_taste_schema_has_no_numeric_scale(judges):
    module, _, _ = judges
    assert "1-5" not in module.TASTE_SYSTEM and "integer" not in module.TASTE_SYSTEM


@pytest.mark.parametrize("passed, gate", [(3, True), (2, False), (5, True)])
def test_taste_gate_is_at_least_3_of_5(passed, gate):
    gates = tomllib.loads((LIB / "judge" / "models.toml").read_text())["gates"]
    assert "taste_min" not in gates
    assert (passed / 5 >= gates["taste_min_pass_rate"]) is gate
    source = (LIB / "lib" / "aggregate.py").read_text()
    assert "GATES['taste_min_pass_rate']" in source and "GATES['taste_min']" not in source
