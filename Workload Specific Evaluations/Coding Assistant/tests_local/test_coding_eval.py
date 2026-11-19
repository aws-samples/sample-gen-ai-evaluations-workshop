"""coding_eval.py, the helpers of notebooks 02 and 03. No Docker: the Docker calls are replaced."""

import json
import subprocess

import pytest

import coding_eval as ce
from conftest import CODING_TASKS, PR_TASKS


def snapshot(task_dir):
    return sorted((str(p.relative_to(task_dir)), p.stat().st_mtime_ns) for p in task_dir.rglob("*"))


@pytest.mark.parametrize("task_dir", PR_TASKS + CODING_TASKS, ids=lambda p: p.name)
def test_leakage_scan_plants_canary_in_a_temporary_copy(task_dir):
    before = snapshot(task_dir)
    assert ce.leakage_scan(task_dir, ["a needle that is not in any file 0x5eed"]) == []
    assert snapshot(task_dir) == before
    assert not (task_dir / "environment" / "canary.txt").exists()


def test_leakage_scan_finds_a_needle_and_a_broken_scan_fails(tmp_path, monkeypatch):
    task = tmp_path / "task"
    (task / "environment").mkdir(parents=True)
    (task / "instruction.md").write_text("Review the change.")
    (task / "environment" / "Dockerfile").write_text("RUN echo see pull/4615")
    assert ce.leakage_scan(task, ["pull/4615", "absent"]) == [("environment/Dockerfile", "pull/4615")]
    monkeypatch.setattr(ce, "agent_visible_files", lambda d: iter([d / "instruction.md"]))  # misses environment/
    with pytest.raises(AssertionError, match="canary"):
        ce.leakage_scan(task, ["x"])


def test_check_region():
    tasks = PR_TASKS + CODING_TASKS
    assert all(r == {"us-east-1"} for r in ce.task_regions(tasks).values())
    assert ce.check_region("us-east-1", tasks) == "us-east-1"
    with pytest.raises(RuntimeError, match="AWS_REGION=us-east-1 in .env"):
        ce.check_region("us-west-2", tasks)
    with pytest.raises(RuntimeError):
        ce.check_region(None, tasks)


class FakeDocker:
    def __init__(self, existing=(), fail=()):
        self.existing, self.fail, self.calls = set(existing), set(fail), []

    def __call__(self, *args, timeout=None):
        self.calls.append(args)
        if args[:2] == ("image", "inspect"):
            code = 0 if args[2] in self.existing else 1
        elif args[0] == "build":
            tag = args[args.index("-t") + 1]
            code = 1 if tag in self.fail else 0
            if not code:
                self.existing.add(tag)
        else:
            code = 0
        return subprocess.CompletedProcess(args, code, stdout="sha256:abc" if not code else "", stderr="boom" if code else "")


def test_build_images_skips_existing_unless_forced(monkeypatch, capsys):
    docker = FakeDocker(existing={ce.PROXY_IMAGE, ce.image_tag(PR_TASKS[0])})
    monkeypatch.setattr(ce, "_docker", docker)
    ce.build_images(PR_TASKS)
    builds = [c for c in docker.calls if c[0] == "build"]
    assert [c[c.index("-t") + 1] for c in builds] == [ce.image_tag(PR_TASKS[1])]
    out = capsys.readouterr().out
    assert out.count("SKIP") == 2 and " s " in out  # elapsed time of each build
    docker.calls.clear()
    ce.build_images(PR_TASKS, force=True)
    assert len([c for c in docker.calls if c[0] == "build"]) == 3


def test_build_images_raises_on_failure(monkeypatch):
    monkeypatch.setattr(ce, "_docker", FakeDocker(fail={ce.image_tag(CODING_TASKS[0])}))
    with pytest.raises(RuntimeError, match=CODING_TASKS[0].name):
        ce.build_images(CODING_TASKS)


@pytest.mark.parametrize("output,kwargs,error", [
    ("1 0 3", {"expect_changes": True}, None),
    ("2 0 3", {"expect_changes": True}, "More than one commit"),
    ("1 0 0", {"expect_changes": True}, "change.patch"),
    ("1 0 0 0", {"expect_changes": False, "check_hidden_tests": True}, None),
    ("1 0 0 1", {"expect_changes": False, "check_hidden_tests": True}, "hidden tests"),
    ("1 1 0 0", {"expect_changes": False, "check_hidden_tests": True}, "remote"),
])
def test_seal_check(monkeypatch, output, kwargs, error):
    seen = []

    def fake(*args, timeout=None):
        seen.append(args)
        return subprocess.CompletedProcess(args, 0, stdout=output + "\n", stderr="")

    monkeypatch.setattr(ce, "_docker", fake)
    if error:
        with pytest.raises(AssertionError, match=error):
            ce.seal_check(PR_TASKS[:1], **kwargs)
    else:
        ce.seal_check(PR_TASKS[:1], **kwargs)
    assert ("_verify.py" in seen[0][-1]) == kwargs.get("check_hidden_tests", False)


def test_trial_rewards(tmp_path):
    job = tmp_path / "job"
    for trial, task, reward, details in [
        ("t1", "awslabs/mcp__pr4121-approve", {"coverage_all": 1.0}, None),
        ("t2", "strands-agents/evals__pr371-fix", None, {"invalid": ["validation error: throttled"]}),
        ("t3", "strands-agents/evals__pr338-fix", None, None),
    ]:
        (job / trial / "verifier").mkdir(parents=True)
        (job / trial / "result.json").write_text(json.dumps({"task_name": task}))
        if reward:
            (job / trial / "verifier" / "reward.json").write_text(json.dumps(reward))
        if details:
            (job / trial / "verifier" / "reward_details.json").write_text(json.dumps(details))
    rewards = ce.trial_rewards(job)
    assert rewards["awslabs__mcp__pr4121-approve"].get("precision_adjudicated") is None
    assert rewards["strands-agents__evals__pr371-fix"] == {"reward": None, "invalid": ["validation error: throttled"]}
    assert rewards["strands-agents__evals__pr338-fix"]["invalid"] == ["no verifier output"]


def test_harbor_run_command(monkeypatch, tmp_path):
    seen = {}

    def fake_run(cmd, stdout=None, stderr=None, stdin=None, env=None):
        seen["cmd"], seen["env"] = cmd, env
        stdout.write("done\n")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(ce, "refresh_credentials", lambda region=None: (None, None))
    monkeypatch.setattr(ce.subprocess, "run", fake_run)
    monkeypatch.setenv("AWS_PROFILE", "my-profile")
    ce.harbor_run("job", "-a", "oracle", tasks_dir=tmp_path / "tasks", tasks=CODING_TASKS[:1],
                  n_concurrent=2, jobs_dir=tmp_path / "jobs")
    cmd = seen["cmd"]
    assert cmd[:4] == ["harbor", "run", "-p", str(CODING_TASKS[0])]
    assert cmd[cmd.index("-n") + 1] == "2" and cmd[-2:] == ["-a", "oracle"]
    assert str(ce.MODULE_DIR) in seen["env"]["PYTHONPATH"].split(":")
    assert "AWS_PROFILE" not in seen["env"]  # the task containers get the exported keys, not a profile name
    ce.harbor_run("job2", tasks_dir=tmp_path / "tasks", jobs_dir=tmp_path / "jobs")
    assert "-n" not in seen["cmd"] and seen["cmd"][2:4] == ["-p", str(tmp_path / "tasks")]


def test_egress_check_logic(monkeypatch):
    """The egress check with a fake Docker: the allowed hosts answer, the blocked hosts do not."""
    allowed = ("bedrock-mantle.us-east-1.api.aws", "bedrock-runtime.us-east-1.amazonaws.com")  # Codex, Claude Code
    calls = []

    def fake(*args, timeout=None):
        calls.append(args)
        out = ""
        if args[0] == "run" and "--rm" in args:
            script = args[-1]
            out = "404" if any(h in script for h in allowed) and "--noproxy" not in script else "000"
        return subprocess.CompletedProcess(args, 0, stdout=out, stderr="")

    monkeypatch.setattr(ce, "_docker", fake)
    results = ce.egress_check(PR_TASKS[0])
    assert all(results.values()) and len(results) == 6  # 2 allowed + 2 blocked hosts x 2 paths
    assert any(c[:2] == ("rm", "-f") for c in calls) and sum(c[:2] == ("network", "rm") for c in calls) == 2

    def leaky(*args, timeout=None):
        return subprocess.CompletedProcess(args, 0, stdout="200" if args[0] == "run" and "--rm" in args else "", stderr="")

    monkeypatch.setattr(ce, "_docker", leaky)
    with pytest.raises(AssertionError, match="github.com"):
        ce.egress_check(PR_TASKS[0])


def _fake_trial(job, name, extra=""):
    trial = job / name
    (trial / "verifier").mkdir(parents=True)
    (trial / "agent").mkdir()
    (trial / "agent" / "sessions").mkdir()
    (trial / "result.json").write_text(json.dumps({"task_name": "t", "config": {"env": {"X": "${X}"}}, "agent_result": {"cost_usd": 0.1}}))
    (trial / "verifier" / "reward.json").write_text('{"reward": 1.0}')
    (trial / "agent" / "trajectory.json").write_text('{"steps": []}' + extra)
    (trial / "agent" / "sessions" / "raw.jsonl").write_text("big")
    return trial


def test_record_job_keeps_needed_files_only(tmp_path):
    job = tmp_path / "job"
    _fake_trial(job, "trial1")
    written = ce.record_job(job, tmp_path / "out")
    names = {p.relative_to(tmp_path / "out").as_posix() for p in written}
    assert names == {"trial1/result.json", "trial1/verifier/reward.json", "trial1/agent/trajectory.json"}
    result = json.loads((tmp_path / "out" / "trial1" / "result.json").read_text())
    assert "config" not in result and result["agent_result"]["cost_usd"] == 0.1


@pytest.mark.parametrize("leak", ["AKIAABCDEFGHIJKLMNOP", "arn:aws:iam::123456789012:role/x", "/Users/someone/repo"])
def test_record_job_refuses_secrets(tmp_path, leak):
    job = tmp_path / "job"
    _fake_trial(job, "trial1", extra=leak)
    with pytest.raises(ValueError, match="Not recorded"):
        ce.record_job(job, tmp_path / "out")
    assert not (tmp_path / "out").exists()
