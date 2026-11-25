"""The PR-review scorer writes the verdict and pass/fail into its reward."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCORE = Path(__file__).resolve().parents[2] / "datasets/_shared/pr-review/tests/score.py"
spec = importlib.util.spec_from_file_location("review_score", SCORE)
score = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = score
spec.loader.exec_module(score)


def oracle(*severities):
    return [score.OracleFinding(id=i, statement=f"o{i}", severity=s, reference_only=False, locations=(score.Location("a.py", i, i),))
            for i, s in enumerate(severities)]


def agent(*severities):
    return [score.AgentFinding(index=i, statement=f"o{i}", severity=s, path="a.py", line=i) for i, s in enumerate(severities)]


@pytest.mark.parametrize("oracle_sev,agent_sev,matches,blocks,expected,passed", [
    ((), (), True, 0, 0, 1),                            # approve task, agent reports nothing
    ((), ("medium",), True, 0, 0, 1),                   # approve task, a non-blocking finding is still an approve
    ((), ("high",), True, 1, 0, 0),                     # approve task, agent blocks a good PR
    (("high", "low"), ("high", "low"), True, 1, 1, 1),  # block task, agent blocks and finds the required finding
    (("high",), ("high",), False, 1, 1, 0),             # block task, agent blocks for a reason the oracle does not have
    (("high",), (), True, 0, 1, 0),                     # block task, agent approves
])
def test_verdict_and_pass(oracle_sev, agent_sev, matches, blocks, expected, passed):
    judge = (lambda o, a: o.id == a.index) if matches else (lambda o, a: False)
    r = score.Score.compute_reward(oracle(*oracle_sev), agent(*agent_sev), judge)
    assert (r["agent_blocks"], r["expected_blocks"], r["passed"]) == (blocks, expected, passed)


def test_verdict_file_overrides_the_oracle(tmp_path):
    f = tmp_path / "verdict.json"
    f.write_text(json.dumps({"expected_verdict": "block"}))
    r = score.Score.compute_reward(oracle("medium"), agent("high"), lambda o, a: False, score.load_expected_block(f))
    assert r["expected_blocks"] == 1 and r["passed"] == 1
    assert score.load_expected_block(tmp_path / "missing.json") is None
    f.write_text(json.dumps({"expected_verdict": "maybe"}))
    with pytest.raises(ValueError):
        score.load_expected_block(f)


def test_tasks_ship_a_verdict_that_matches_task_toml():
    import tomllib
    tasks = Path(__file__).resolve().parents[2] / "datasets/awslabs-mcp-pr-review/tasks"
    for t in tasks.iterdir():
        want = tomllib.loads((t / "task.toml").read_text())["metadata"]["harvest"]["expected_verdict"]
        assert json.loads((t / "tests/verdict.json").read_text())["expected_verdict"] == want, t.name


def test_verdict_json_is_baked_into_each_verifier_image():
    tasks = Path(__file__).resolve().parents[2] / "datasets/awslabs-mcp-pr-review/tasks"
    for t in tasks.iterdir():
        assert "COPY --chmod=644 verdict.json /tests/verdict.json" in (t / "tests/Dockerfile").read_text(), t.name


def test_severity_case_does_not_matter():
    r = score.Score.compute_reward(oracle(), agent("High"), lambda o, a: False)
    assert r["agent_blocks"] == 1


@pytest.mark.parametrize("findings,passed", [(None, 0.0), ([], 1.0)])
def test_no_findings_file_is_no_review(tmp_path, monkeypatch, findings, passed):
    (tmp_path / "oracle.json").write_text(json.dumps({"findings": []}))
    (tmp_path / "rec.json").write_text(json.dumps({"decisions": []}))
    f = tmp_path / "findings.json"
    if findings is not None:
        f.write_text(json.dumps(findings))
    for name, p in (("_ORACLE_PATH", "oracle.json"), ("_VERDICT_PATH", "verdict.json"), ("_REWARD_PATH", "reward.json"),
                    ("_REWARD_DETAILS_PATH", "details.json"), ("_JUDGE_RECORD_PATH", "judge.json")):
        monkeypatch.setattr(score, name, tmp_path / p)
    monkeypatch.setenv("EVAL_HARVEST_AGENT_FINDINGS", str(f))
    monkeypatch.setenv("EVAL_HARVEST_JUDGE_RECORDING", str(tmp_path / "rec.json"))
    score.main()
    r = json.loads((tmp_path / "reward.json").read_text())
    assert r["submitted"] == float(findings is not None) and r["passed"] == passed
