"""Offline tests of Step 1 (no AWS, no Harbor): coding_tasks.py on a synthetic git repository, and the tasks it
writes run in harbor_env's grading stages with local processes.

The staged grader runs `test.sh --stage untrusted` as the agent user, refuses the trial if the repository
changed, copies the stage's regular files (never a `reward*` name) into a root-owned IN, then runs
`test.sh --stage trusted` with IN and a fresh OUT; only OUT/reward.json becomes the reward. These tests take
the same steps. The repository and its test runner (stdlib Python that writes JUnit) are made up here.
"""
import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tomllib
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import coding_tasks  # noqa: E402

RUNNER = '''import sys
junit, files = sys.argv[1], sys.argv[2:]
open(".runner-cache", "w").write("cache")
cases = []
for path in files:
    ns = {"__name__": "t"}
    exec(compile(open(path).read(), path, "exec"), ns)
    for name in sorted(n for n in ns if n.startswith("test_")):
        try:
            ns[name]()
            cases.append('<testcase classname="t" name="%s" file="%s"/>' % (name, path))
        except Exception:
            cases.append('<testcase classname="t" name="%s" file="%s"><failure message="x"/></testcase>' % (name, path))
open(junit, "w").write("<testsuites>%s</testsuites>" % "".join(cases))
'''
BASE_CALC = "def add(a, b):\n    return a + b\n"
GOLD_CALC = BASE_CALC + "\n\ndef mul(a, b):\n    return a * b\n"
BASE_TEST = "import sys\nsys.path.insert(0, '.')\nimport calc\n\n\ndef test_add():\n    assert calc.add(2, 3) == 5\n"
HIDDEN_TEST = BASE_TEST + "\n\ndef test_mul():\n    assert calc.mul(2, 3) == 6\n"
PROFILE = """[coding]
image = "python:3.12-bookworm"
command = "python3 run_tests.py {junit} {files}"
display_command = "python3 run_tests.py report.xml {files}"
run = ["tests/test_*.py"]
"""
F2P, P2P = "tests/test_calc.py::test_mul", "tests/test_calc.py::test_add"
ENV = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull, "GIT_AUTHOR_NAME": "t",
       "GIT_AUTHOR_EMAIL": "t@example.com", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com", "TZ": "UTC"}
TRUSTED_PYTHON = next((p for p in ("/usr/bin/python3", "/usr/local/bin/python3")
                       if os.path.isfile(p) and os.stat(os.path.realpath(p)).st_uid == 0), None)
staged = pytest.mark.skipif(sys.platform == "win32" or not shutil.which("bash") or TRUSTED_PYTHON is None,
                            reason="the stages need bash and a root-owned /usr/bin/python3 or /usr/local/bin/python3")


def git(repo, *args, date="2026-01-01T00:00:00"):
    env = {**ENV, "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}
    return subprocess.run(["git", "-C", str(repo), *args], env=env, capture_output=True, text=True, check=True).stdout


def write(repo, files):
    for rel, text in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)


def merge_pr(repo, number, title, files, day):
    git(repo, "checkout", "-q", "-b", f"pr{number}")
    write(repo, files)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", title, date=f"2026-01-{day:02d}T00:00:00")
    git(repo, "checkout", "-q", "main")
    git(repo, "merge", "-q", "--no-ff", "-m", f"Merge pull request #{number} from acme/pr{number}", "-m", title, f"pr{number}",
        date=f"2026-01-{day:02d}T12:00:00")


@pytest.fixture
def repo(tmp_path):
    """main: the base; PR #7 adds mul and its test; PR #8 docs only; PR #9 tests only; PR #10 edits .gitignore too."""
    clone = tmp_path / "clone"
    clone.mkdir()
    git(clone, "init", "-q", "-b", "main")
    git(clone, "remote", "add", "origin", "https://github.com/acme/calc.git")
    write(clone, {"calc.py": BASE_CALC, "run_tests.py": RUNNER, "tests/test_calc.py": BASE_TEST, "README.md": "calc\n",
                  ".gitignore": "__pycache__/\n"})
    git(clone, "add", "-A")
    git(clone, "commit", "-q", "-m", "base")
    merge_pr(clone, 7, "Add multiplication", {"calc.py": GOLD_CALC, "tests/test_calc.py": HIDDEN_TEST}, 2)
    merge_pr(clone, 8, "Document mul", {"README.md": "calc, now with mul\n"}, 3)
    merge_pr(clone, 9, "Test add again", {"tests/test_more.py": BASE_TEST}, 4)
    sub_test = HIDDEN_TEST + "\n\ndef test_sub():\n    assert calc.sub(3, 2) == 1\n"
    merge_pr(clone, 10, "Add subtraction", {"calc.py": GOLD_CALC + "\n\ndef sub(a, b):\n    return a - b\n",
                                            ".gitignore": "__pycache__/\n.venv/\n", "tests/test_calc.py": sub_test}, 5)
    (tmp_path / "coding-profile.toml").write_text(PROFILE)
    return clone


def build(repo, pr=7):
    profile = coding_tasks.read_profile(repo.parent / "coding-profile.toml")
    return coding_tasks.build(repo, pr, profile, repo.parent / "dataset")


def snapshot(task):
    data = gzip.decompress((task / "environment" / "repo.tar.gz").read_bytes())
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        return {m.name: archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}


# ---------------------------------------------------------------- build
def test_a_task_holds_the_base_the_hidden_tests_and_the_change_apart(repo):
    task = build(repo)
    assert task.name == "acme__calc__pr7-code"
    files = snapshot(task)
    assert files["calc.py"] == BASE_CALC.encode() and b"test_mul" not in files["tests/test_calc.py"]
    assert "def test_mul" in (task / "tests/files/tests/test_calc.py").read_text()
    gold = (task / "solution/gold.patch").read_text()
    assert "+def mul(a, b):" in gold and "test_calc.py" not in gold
    config = tomllib.loads((task / "task.toml").read_text())
    assert config["task"]["name"] == "acme/calc__pr7-code"
    assert "environment_mode" not in config["verifier"] and config["verifier"]["network_mode"] == "no-network"
    assert config["environment"]["network_mode"] == "allowlist"
    instruction = (task / "instruction.md").read_text()
    assert instruction.startswith("# Add multiplication\n") and "#7" not in instruction and "mul(a, b)" not in instruction
    dockerfile = (task / "environment/Dockerfile").read_text()
    assert "WORKDIR /work/repo" in dockerfile and "> /var/lib/task_base_ref" in dockerfile


def test_the_same_pull_request_gives_the_same_bytes(repo):
    def files(task):
        return {p.relative_to(task).as_posix(): p.read_bytes() for p in task.rglob("*") if p.is_file()}
    assert files(build(repo)) == files(build(repo))


@pytest.mark.parametrize(("pr", "code"), [(8, "no-test-change"), (9, "no-source-change"), (99, "pr-not-found")])
def test_a_pull_request_that_is_not_a_coding_task_is_refused(repo, pr, code):
    with pytest.raises(coding_tasks.Refused) as refusal:
        build(repo, pr)
    assert refusal.value.code == code


def test_base_text_in_the_change_is_not_taken_for_a_leak(repo):
    # PR #10 edits .gitignore, so "__pycache__/" (base text) is a context line of its patch.
    assert " __pycache__/" in (build(repo, 10) / "solution/gold.patch").read_text()
    # A context or removed line is the base's own text, which the agent has anyway: not an answer.
    patch = b" __pycache__/ in the base\n-removed line of the base\n+def sub(a, b):\n"
    files = {"solution/gold.patch": patch, "instruction.md": b"__pycache__/ in the base; removed line of the base\n",
             "environment/Dockerfile": b""}
    assert coding_tasks.answer_leaks(files, 10) == []


def test_a_line_the_change_adds_is_a_leak_where_the_agent_can_read_it():
    files = {"solution/gold.patch": b"+def mul(a, b):\n", "instruction.md": b"Hint: def mul(a, b):\n",
             "environment/Dockerfile": b""}
    assert coding_tasks.answer_leaks(files, 7) == ["'def mul(a, b):' in instruction.md"]


def test_every_profile_problem_is_reported(tmp_path):
    (tmp_path / "p.toml").write_text('[coding]\nimage = ""\ncommand = "pytest"\nrun = "tests/*.py"\ntimeout = 5\n')
    with pytest.raises(SystemExit) as error:
        coding_tasks.read_profile(tmp_path / "p.toml")
    problems = ("coding.timeout: not a profile key", "coding.image", "coding.run", "must contain {files}", "must contain {junit}")
    for key in problems:
        assert key in str(error.value)


# ---------------------------------------------------------------- the stages
def tree_digest(root):
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if rel.parts[0] != ".git":
            digest.update(str(rel).encode() + oct(path.lstat().st_mode).encode())
            if path.is_file():
                digest.update(path.read_bytes())
    return digest.hexdigest()


class Trial:
    """One trial of a task, the way harbor_env grades it, with local processes."""

    def __init__(self, root, task, gold=False):
        self.root, self.task, self.app = root, task, root / "app"
        self.app.mkdir(parents=True)
        with tarfile.open(fileobj=io.BytesIO(gzip.decompress((task / "environment/repo.tar.gz").read_bytes()))) as archive:
            archive.extractall(self.app, filter="data")
        for args in (("init", "-q"), ("add", "-A"), ("commit", "-q", "-m", "base"), ("tag", "task-base")):
            git(self.app, *args)
        if gold:
            (self.app / "calc.py").write_text(GOLD_CALC)

    def sh(self, *args, env=None):
        env = {**os.environ, "TASK_APP": str(self.app), **(env or {})}
        script = str(self.task / "tests/test.sh")
        return subprocess.run(["bash", script, *args], cwd=self.app, capture_output=True, text=True, env=env, timeout=120)

    def untrusted(self):
        out = self.root / "untrusted-out"
        out.mkdir()
        before = tree_digest(self.app)
        assert self.sh("--stage", "untrusted", env={"OUT": str(out)}).returncode == 0
        assert tree_digest(self.app) == before, "the untrusted stage changed the repository"
        assert not list(out.glob("reward*"))
        return out

    def root_copy(self, out, plant=None):
        evidence = self.root / "in"
        evidence.mkdir()
        for path in out.rglob("*"):
            if path.is_file() and not path.name.startswith("reward"):
                shutil.copy2(path, evidence / path.name)
        for name, text in (plant or {}).items():
            (evidence / name).write_text(text)
        return evidence

    def trusted(self, evidence, env=None):
        out = self.root / "grade-out"
        out.mkdir(exist_ok=True)
        stage_env = {"IN": str(evidence), "OUT": str(out), **(env or {})}
        assert self.sh("--stage", "trusted", env=stage_env).returncode == 0
        reward = out / "reward.json"
        return (json.loads(reward.read_text()) if reward.exists() else None), out

    def run(self, plant=None):
        return self.trusted(self.root_copy(self.untrusted(), plant))


def as_harbor_trial(results, jobs, task, trial_id):
    """Where Harbor keeps a trial's files: the folder name cut to 32 characters, the task id in result.json."""
    folder = jobs / f"{task.name[:32]}__{trial_id}"
    (folder / "verifier").mkdir(parents=True)
    task_id = tomllib.loads((task / "task.toml").read_text())["task"]["name"]
    (folder / "result.json").write_text(json.dumps({"task_name": task_id}))
    shutil.copy2(results, folder / "verifier/results.json")


@pytest.fixture
def calibrated(repo, tmp_path):
    task = build(repo)
    for kind, gold in (("nop", False), ("oracle", True)):
        reward, out = Trial(tmp_path / f"cal-{kind}", task, gold).run()
        assert reward is None, "an uncalibrated task gives no reward"
        as_harbor_trial(out / "results.json", tmp_path / "jobs" / kind, task, "a1")
    assert coding_tasks.calibrate(task, tmp_path / "jobs/nop", tmp_path / "jobs/oracle") == ([F2P], [P2P], [])
    return task


@staged
def test_calibration_then_oracle_scores_1_and_nop_0(calibrated, tmp_path):
    assert "- test_mul (`tests/test_calc.py`)" in (calibrated / "instruction.md").read_text()
    assert Trial(tmp_path / "nop", calibrated).run()[0] == {"reward": 0.0}
    reward, out = Trial(tmp_path / "oracle", calibrated, gold=True).run()
    assert reward == {"reward": 1.0} and not (out / "reward.txt").exists()


@staged
def test_nothing_the_agent_could_write_becomes_the_score(calibrated, tmp_path):
    planted = {"reward.json": '{"reward": 1.0}', "grade.json": '{"reward": 1}',
               "results.json": json.dumps({"statuses": {F2P: "passed", P2P: "passed"}})}
    assert Trial(tmp_path / "planted", calibrated).run(planted)[0] == {"reward": 0.0}


@staged
def test_a_forged_passing_copy_of_a_failing_test_scores_0(calibrated, tmp_path):
    trial = Trial(tmp_path / "forged", calibrated)
    evidence = trial.root_copy(trial.untrusted())
    junit = (evidence / "junit.xml").read_text()
    forged = '<testsuites><testcase classname="t" name="test_mul" file="tests/test_calc.py"/>'
    (evidence / "junit.xml").write_text(junit.replace("<testsuites>", forged, 1))  # the forged copy first
    assert trial.trusted(evidence)[0] == {"reward": 0.0}


@staged
def test_the_trusted_stage_reads_neither_the_repository_nor_path(calibrated, tmp_path):
    trial = Trial(tmp_path / "isolated", calibrated, gold=True)
    evidence = trial.root_copy(trial.untrusted())
    shutil.rmtree(trial.app)
    trial.app.mkdir()
    hostile = tmp_path / "bin"
    hostile.mkdir()
    (hostile / "python3").write_text('#!/bin/sh\necho \'{"reward": 1.0}\' > "$OUT/reward.json"; touch "$OUT/planted"\n')
    (hostile / "python3").chmod(0o755)
    reward, out = trial.trusted(evidence, env={"PATH": f"{hostile}:{os.environ['PATH']}"})
    assert reward == {"reward": 1.0} and not (out / "planted").exists()


@staged
def test_no_fail_to_pass_test_is_refused(repo, tmp_path):
    task = build(repo)
    for kind in ("nop", "oracle"):  # both on the base: nothing tells the change apart
        _, out = Trial(tmp_path / kind, task).run()
        as_harbor_trial(out / "results.json", tmp_path / "jobs" / kind, task, "b1")
    with pytest.raises(coding_tasks.Refused) as refusal:
        coding_tasks.calibrate(task, tmp_path / "jobs/nop", tmp_path / "jobs/oracle")
    assert refusal.value.code == "no-fail-to-pass"


def test_calibration_counts_only_this_tasks_trials(tmp_path):
    task = tmp_path / "pallets__itsdangerous__pr378-code"  # 33 characters: Harbor cuts its trial folders to 32
    (task / "tests").mkdir(parents=True)
    (task / "tests/grader.py").write_text("")
    (task / "task.toml").write_text('[task]\nname = "pallets/itsdangerous__pr378-code"\n')
    (task / "instruction.md").write_text(f"{coding_tasks.CALIBRATION_BEGIN}\n{coding_tasks.CALIBRATION_END}\n")
    for jobs, trial_id, task_id, status in (("nop", "a", "pallets/itsdangerous__pr378-code", "failed"),
                                            ("oracle", "b", "pallets/itsdangerous__pr378-code", "passed"),
                                            ("oracle", "c", "pallets/itsdangerous__pr378-cods", "failed")):
        folder = tmp_path / jobs / f"{task.name[:32]}__{trial_id}"
        (folder / "verifier").mkdir(parents=True)
        (folder / "result.json").write_text(json.dumps({"task_name": task_id}))
        (folder / "verifier/results.json").write_text(json.dumps({"statuses": {"t.py::f": status}}))
    assert coding_tasks.calibrate(task, tmp_path / "nop", tmp_path / "oracle") == (["t.py::f"], [], [])
