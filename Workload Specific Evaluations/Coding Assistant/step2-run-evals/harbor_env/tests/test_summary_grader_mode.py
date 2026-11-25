"""summary.py flags a coding (shared-verifier) trial that has no grader_mode event: how it was graded is unknown."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import summary  # noqa: E402


def coding_trial(job, name, events=None):
    t = job / name
    (t / "verifier").mkdir(parents=True)
    (t / "config.json").write_text("{}")
    (t / "result.json").write_text(json.dumps({"task_name": "strands-agents/evals__pr371-fix", "agent_info": {"name": "oracle"},
                                               "verifier_environment_mode": "shared"}))
    (t / "verifier" / "reward.json").write_text(json.dumps({"reward": 1.0, "correct": 1.0}))
    if events is not None:
        (t / "agentcore.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    return t


def test_coding_trial_without_a_grader_mode_event_is_flagged(tmp_path):
    rows = [summary.trial_row(coding_trial(tmp_path, "a", [{"event": "session_start"}]), tmp_path),
            summary.trial_row(coding_trial(tmp_path, "b"), tmp_path),
            summary.trial_row(coding_trial(tmp_path, "c", [{"event": "grader_mode", "mode": "staged"}]), tmp_path)]
    assert [r["grading"] for r in rows] == ["MISSING grader_mode (shared)", "MISSING grader_mode (shared)", "staged"]
    assert "| MISSING grader_mode (shared) |" in summary.markdown(rows)


def test_main_warns_about_trials_without_grader_mode(tmp_path, monkeypatch, capsys):
    coding_trial(tmp_path / "job", "a", [{"event": "session_start"}])
    monkeypatch.setattr(sys, "argv", ["summary.py", str(tmp_path / "job"), "--datasets", str(tmp_path)])
    summary.main()
    assert "WARNING: no grader_mode event in 1 trial(s)" in capsys.readouterr().out
