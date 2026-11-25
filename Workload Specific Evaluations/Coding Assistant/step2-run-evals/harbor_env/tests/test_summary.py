"""Offline tests of harbor_env/summary.py on small job folders written in tmp_path (no AWS, no Harbor run)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harbor_env import summary  # noqa: E402

APPROVE = "awslabs/mcp__pr4121-approve"
REJECT = "awslabs/mcp__pr4615-reject"
CODING = "strands-agents/evals__pr371-fix"


def datasets(tmp_path):
    d = tmp_path / "datasets"
    for name, verdict in ((APPROVE, "approve"), (REJECT, "block")):
        t = d / "pr-review" / "tasks" / name.replace("/", "__")
        t.mkdir(parents=True)
        (t / "task.toml").write_text(f'[task]\nname = "{name}"\n\n[metadata.harvest]\nexpected_verdict = "{verdict}"\n'
                                     'blocking_severity = "high"\n')
    t = d / "coding" / "tasks" / CODING.replace("/", "__")
    t.mkdir(parents=True)
    (t / "task.toml").write_text(f'[task]\nname = "{CODING}"\n')
    return d


def trial(job, name, task, reward, findings=None, mode=None):
    t = job / name
    (t / "verifier").mkdir(parents=True)
    (t / "config.json").write_text("{}")
    (t / "result.json").write_text(json.dumps({"task_name": task, "agent_info": {"name": "codex", "model_info": {"name": "m"}},
                                               "verifier_environment_mode": mode}))
    if reward is not None:
        (t / "verifier" / "reward.json").write_text(json.dumps(reward))
    if findings is not None:
        (t / "artifacts" / "logs" / "artifacts").mkdir(parents=True)
        (t / "artifacts" / "logs" / "artifacts" / "findings.json").write_text(json.dumps(findings))
    return t


def finding(severity):
    return {"path": "a.py", "line": 1, "statement": "x", "severity": severity}


APPROVE_REWARD = {"coverage_required": 1.0, "coverage_all": 1.0, "oracle_high": 0.0, "oracle_low": 0.0, "oracle_medium": 0.0}


def test_approve_task_with_medium_findings_approves_and_passes(tmp_path):
    t = trial(tmp_path / "job", "a", APPROVE, APPROVE_REWARD, [finding("medium"), finding("low")], "separate")
    row = summary.trial_row(t, datasets(tmp_path))
    assert row["predicted_verdict"] == "approve" and row["expected_verdict"] == "approve" and row["passed"] is True
    assert row["n_findings"] == 2 and row["severities"] == {"medium": 1, "low": 1} and row["oracle_all"] == 0.0
    assert row["grading"] == "separate"


def test_approve_task_with_a_high_finding_blocks_and_fails_despite_full_coverage(tmp_path):
    t = trial(tmp_path / "job", "a", APPROVE, APPROVE_REWARD, [finding("high")])
    row = summary.trial_row(t, datasets(tmp_path))
    assert row["predicted_verdict"] == "block" and row["passed"] is False


def test_reject_task_with_a_high_finding_and_full_coverage_passes(tmp_path):
    reward = {"coverage_required": 1.0, "coverage_all": 0.5, "oracle_high": 1.0, "oracle_medium": 2.0, "oracle_low": 3.0}
    t = trial(tmp_path / "job", "r", REJECT, reward, [finding("high")])
    row = summary.trial_row(t, datasets(tmp_path))
    assert row["predicted_verdict"] == "block" and row["passed"] is True and row["oracle_all"] == 6.0


def test_reject_task_blocking_without_the_required_finding_fails(tmp_path):
    t = trial(tmp_path / "job", "r", REJECT, {"coverage_required": 0.0}, [finding("high")])
    assert summary.trial_row(t, datasets(tmp_path))["passed"] is False


def test_missing_findings_file_is_no_review(tmp_path):
    t = trial(tmp_path / "job", "a", APPROVE, APPROVE_REWARD)
    row = summary.trial_row(t, datasets(tmp_path))
    assert row["predicted_verdict"] == "not submitted" and row["n_findings"] == 0 and row["passed"] is False


def test_no_reward_gives_no_pass(tmp_path):
    t = trial(tmp_path / "job", "a", APPROVE, None, [])
    assert summary.trial_row(t, datasets(tmp_path))["passed"] is None


def test_coding_task_has_no_verdict_and_grader_mode_event_wins(tmp_path):
    t = trial(tmp_path / "job", "c", CODING, {"reward": 1.0, "correct": 1.0}, mode="shared")
    (t / "agentcore.jsonl").write_text(json.dumps({"event": "grader_mode", "mode": "staged"}) + "\n")
    row = summary.trial_row(t, datasets(tmp_path))
    assert "predicted_verdict" not in row and row["grading"] == "staged"


def test_job_rows_and_markdown(tmp_path):
    ds = datasets(tmp_path)
    job = tmp_path / "job"
    trial(job, "a", APPROVE, APPROVE_REWARD, [finding("high")], "separate")
    trial(job, "c", CODING, {"reward": 0.0, "correct": 0.0}, mode="shared")
    (job / "result.json").write_text("{}")  # the job's own result.json is not a trial
    rows = summary.job_rows(job, ds)
    assert [r["trial"] for r in rows] == ["a", "c"]
    md = summary.markdown(rows)
    assert "block / approve" in md and "| no |" in md and "reward 0.0" in md and "oracle_all 0.0" in md


def test_the_graders_passed_key_wins(tmp_path):
    t = trial(tmp_path / "job", "a", APPROVE, {**APPROVE_REWARD, "passed": 0.0}, [finding("medium")])
    assert summary.trial_row(t, datasets(tmp_path))["passed"] is False
