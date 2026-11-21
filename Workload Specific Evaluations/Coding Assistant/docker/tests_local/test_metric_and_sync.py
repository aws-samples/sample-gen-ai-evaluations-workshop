"""The dataset metric of notebook 02, and the copies of the shared verifier files."""

import json
import re
import tomllib

from conftest import CODING_TASKS, PR_TASKS


def run_metric(metric, tmp_path, rows):
    src, dst = tmp_path / "rewards.jsonl", tmp_path / "metric.json"
    src.write_text("".join(("null" if r is None else json.dumps(r)) + "\n" for r in rows))
    metric.main(src, dst)
    return json.loads(dst.read_text())


def test_metric_skips_undefined_precision(metric, tmp_path):
    reject = {"coverage_all": 1.0, "coverage_required": 1.0, "precision_adjudicated": 0.5, "precision_strict": 1.0}
    approve_missing = {"coverage_all": 1.0, "coverage_required": 1.0}  # what reward.json has with no findings
    approve_null = {"coverage_all": 1.0, "coverage_required": 1.0, "precision_adjudicated": None, "precision_strict": None}
    out = run_metric(metric, tmp_path, [reject, approve_missing, approve_null])
    assert out["mean_precision_adjudicated"] == 0.5
    assert out["mean_precision_strict"] == 1.0
    assert out["n_precision_adjudicated"] == 1
    assert out["mean_coverage_all"] == 1.0
    assert out["trials"] == 3


def test_metric_all_undefined_and_failed_trials(metric, tmp_path):
    out = run_metric(metric, tmp_path, [{"coverage_all": 1.0}])
    assert out["mean_precision_adjudicated"] is None
    # A failed trial (null row) still counts as 0, so it does not raise the mean.
    out = run_metric(metric, tmp_path, [None, {"coverage_all": 1.0, "precision_strict": 1.0}])
    assert out["mean_coverage_all"] == 0.5
    assert out["mean_precision_strict"] == 0.5


def test_metric_keys_match_score_keys(metric, score):
    assert set(metric.KEYS) == set(score.REWARD_KEYS)
    assert set(metric.UNDEFINED_WHEN_EMPTY) == set(score.UNDEFINED_WHEN_EMPTY)


def test_shared_copies_in_sync(sync):
    assert sync.stale_copies() == []
    copied = {dst for _, dst in sync.pairs()}
    for task in PR_TASKS:
        assert {task / "tests" / f for f in ("score.py", "judge.toml", "test.sh", "Dockerfile", "rubric.md")} <= copied
    for task in CODING_TASKS:
        assert task / "tests" / "judge" / "models.toml" in copied
        assert task / "tests" / "lib" / "bedrock.py" in copied


def test_sync_reports_a_changed_copy(sync, tmp_path, monkeypatch):
    """A copy that differs from its source is reported, and sync() restores it."""
    src, dst = sync.pairs()[0]
    original = dst.read_bytes()
    try:
        dst.write_bytes(original + b"\n# local edit\n")
        assert dst in sync.stale_copies()
        assert sync.main(["--check"]) == 1
        assert dst in sync.sync()
        assert dst.read_bytes() == src.read_bytes()
    finally:
        dst.write_bytes(original)
    assert sync.stale_copies() == []


def test_task_dirs_have_every_referenced_file():
    """Each file that a Dockerfile copies, or that test.sh or task.toml names, is in the task directory."""
    for task in PR_TASKS + CODING_TASKS:
        for dockerfile in (task / "environment" / "Dockerfile", task / "tests" / "Dockerfile"):
            if not dockerfile.exists():
                continue
            for line in dockerfile.read_text().splitlines():
                if line.startswith("COPY") and "--from=" not in line:
                    source = [p for p in line.split()[1:-1] if not p.startswith("--")]
                    for name in source:
                        assert (dockerfile.parent / name).exists(), f"{dockerfile}: {name} is missing"
        test_sh = (task / "tests" / "test.sh").read_text()
        for rel in re.findall(r"/tests/([\w./-]+\.\w+)", test_sh):
            assert (task / "tests" / rel).exists(), f"{task.name}: test.sh names /tests/{rel}"
        if hidden := re.search(r"^HIDDEN=(\S+)", test_sh, re.M):
            assert (task / "tests" / "verify" / hidden.group(1)).exists()
        meta = tomllib.loads((task / "task.toml").read_text()).get("metadata", {}).get("verifier", {})
        for key in ("hidden_tests", "validation_spec", "rubric"):
            if key in meta:
                assert (task / meta[key]).exists(), f"{task.name}: {key} = {meta[key]} is missing"
        if task in CODING_TASKS:
            for rel in ("tests/lib/aggregate.py", "tests/lib/agent_tests.py", "tests/lib/validate.py",
                        "tests/lib/judges.py", "tests/lib/bedrock.py", "tests/judge/models.toml",
                        "tests/judge/instruction.md", "tests/judge/reference.patch", "solution/oracle.patch"):
                assert (task / rel).exists(), f"{task.name}: {rel} is missing"


def test_recorded_runs_are_complete_and_clean():
    """Each recorded trial has the files that notebook 04 reads, and nothing that looks like a secret."""
    import coding_eval as ce
    trials = sorted(ce.RECORDED_DIR.glob("*/*/*/result.json"))
    assert {p.parts[-4] for p in trials} == {"codex", "claude-code"} and len(trials) == 8
    for result in trials:
        trial = result.parent
        assert (trial / "verifier" / "reward.json").is_file() and (trial / "agent" / "trajectory.json").is_file()
        if result.parts[-3] == "coding":
            details = json.loads((trial / "verifier" / "reward_details.json").read_text())
            assert all(isinstance(v.get("passed"), bool) for v in details["taste"]["dimensions"].values()), trial
        assert "config" not in json.loads(result.read_text())
    for path in ce.RECORDED_DIR.rglob("*"):
        if path.is_file():
            assert not ce._SECRET.search(path.read_text(errors="replace")), path
