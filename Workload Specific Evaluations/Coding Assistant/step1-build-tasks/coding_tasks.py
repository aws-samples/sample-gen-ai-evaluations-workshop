"""Coding tasks from a repository's merged pull requests, and their calibration. Standard library only.

    python coding_tasks.py build --clone <repo clone> --pr <n> [--pr <n> ...] --profile <toml> --out <dataset dir>
    python coding_tasks.py calibrate <task dir> --nop <Harbor job dir> --oracle <Harbor job dir>

build writes one Harbor task per merged pull request into <dataset dir>/tasks/<owner>__<name>__pr<n>-code/:

- environment/: the repository at the merge's first parent (`git archive`: no history, no file of the
  change), installed, as one fresh commit at /work/repo, where harbor_env's staged grader looks for it;
- tests/: the after-version of every changed test file (hidden from the agent until grading), the grading
  spec, and the staged grader (test.sh --stage untrusted|trusted, grader.py, from task_template/);
- solution/: the source side of the change, for Harbor's oracle agent;
- instruction.md: the pull request's title, the hidden test files and how they run. Never the PR's
  description, which usually says how the fix works.

Which hidden tests the change makes pass is found by running them. Before calibration the task's grader
records every test's status and gives no reward; calibrate reads Harbor's nop and oracle runs of the task
and writes tests/expected.json: fail-to-pass tests passed in no nop run and every oracle run,
pass-to-pass tests in every run, and tests whose result changed between runs of one agent are left out.

It reads only the local clone (no GitHub API), and finds a pull request by the commit subjects GitHub
writes: "Merge pull request #<n> ..." or a squash ending in "(#<n>)".
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import subprocess  # nosec B404 - git only, argv lists, never a shell string
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TEMPLATES = Path(__file__).resolve().parent / "task_template"
CALIBRATION_BEGIN = "<!-- coding_tasks.py calibrate: begin -->"
CALIBRATION_END = "<!-- coding_tasks.py calibrate: end -->"
NO_TESTS_REPORTED = "(no tests reported)"
#: The test side when the profile names none: the usual test folders and file names.
DEFAULT_TEST_GLOBS = ("**/test/**", "**/tests/**", "**/spec/**", "**/__tests__/**", "**/test_*", "**/*_test.*",
                      "**/*.test.*", "**/*.spec.*", "**/*Test.java", "**/conftest.py")
#: What the agent may reach: the Bedrock endpoints that Codex and Claude Code use, and nothing else.
DEFAULT_ALLOWED_HOSTS = ("bedrock-mantle.us-east-1.api.aws", "bedrock-runtime.us-east-1.amazonaws.com")
PROFILE_KEYS = {"image", "install", "command", "display_command", "run", "tests", "workdir", "timeout_sec", "env",
                "agent_setup", "allowed_hosts"}
#: The shortest line of the change worth treating as part of the answer (a brace or `return x` is not).
MIN_ANSWER_LINE = 12
AGENT_TIMEOUT_SEC = 3600.0


class Refused(Exception):
    """Why a pull request is not a coding task: a short code and a sentence to act on."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


@dataclass(frozen=True)
class Profile:
    """How the repository's tests run inside the task image (coding-profile.toml, [coding])."""

    image: str
    command: str
    display_command: str
    run_globs: tuple[str, ...]
    test_globs: tuple[str, ...]
    install: str
    workdir: str
    timeout_sec: float
    env: tuple[tuple[str, str], ...]
    agent_setup: tuple[str, ...]
    allowed_hosts: tuple[str, ...]


@dataclass(frozen=True)
class Change:
    """One merged pull request, from the clone."""

    repository: str
    pr: int
    merge: str
    base: str
    title: str
    base_date: str
    merged_at: str


@dataclass(frozen=True)
class Split:
    """The change's paths: what the agent writes, what grades it."""

    source: tuple[str, ...]
    hidden_tests: tuple[str, ...]
    deleted_tests: tuple[str, ...]
    run_files: tuple[str, ...]


# ---------------------------------------------------------------- the profile
def read_profile(path: Path) -> Profile:
    """The profile, or SystemExit with every problem in it."""
    try:
        section = tomllib.loads(path.read_text()).get("coding")
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise SystemExit(f"{path}: {error}") from error
    if not isinstance(section, dict):
        raise SystemExit(f"{path}: needs a [coding] table")
    problems = [f"coding.{key}: not a profile key" for key in sorted(set(section) - PROFILE_KEYS)]

    def text(key: str, required: bool = False) -> str:
        value = section.get(key, "")
        if not isinstance(value, str) or (required and not value.strip()):
            problems.append(f"coding.{key}: {'required, ' if required else ''}a string")
            return ""
        return value

    def strings(key: str, required: bool = False) -> tuple[str, ...]:
        value = section.get(key)
        if value is None and not required:
            return ()
        if isinstance(value, list) and value and all(isinstance(item, str) and item for item in value):
            return tuple(value)
        problems.append(f"coding.{key}: a non-empty list of strings (a lone string would be read as its characters)")
        return ()

    command = text("command", required=True)
    problems += [f"coding.command: must contain {p}" for p in ("{files}", "{junit}") if command and p not in command]
    env = section.get("env", {})
    if not (isinstance(env, dict) and all(isinstance(v, str) for v in env.values())):
        problems.append("coding.env: a table of strings")
        env = {}
    timeout = section.get("timeout_sec", 900)
    if isinstance(timeout, bool) or not isinstance(timeout, int | float) or timeout <= 0:
        problems.append("coding.timeout_sec: a positive number of seconds")
        timeout = 900
    profile = Profile(
        image=text("image", required=True),
        command=command,
        display_command=text("display_command") or command.replace(" --junitxml={junit}", ""),
        run_globs=strings("run", required=True),
        test_globs=strings("tests") or DEFAULT_TEST_GLOBS,
        install=text("install"),
        workdir=text("workdir") or ".",
        timeout_sec=float(timeout),
        env=tuple(sorted((str(k), str(v)) for k, v in env.items())),
        agent_setup=strings("agent_setup"),
        allowed_hosts=strings("allowed_hosts") or DEFAULT_ALLOWED_HOSTS,
    )
    if problems:
        raise SystemExit(f"{path}:\n  " + "\n  ".join(problems))
    return profile


def compile_glob(pattern: str) -> re.Pattern[str]:
    """A glob as a regular expression: `**` crosses folders, `*` and `?` do not (the grader reads globs the same way)."""
    out, i = [], 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("/**", i) and i + 3 == len(pattern):
            out.append("(?:/.*)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        else:
            out.append({"*": "[^/]*", "?": "[^/]"}.get(pattern[i], re.escape(pattern[i])))
            i += 1
    return re.compile("^" + "".join(out) + "$")


# ---------------------------------------------------------------- git
def git(clone: Path, *args: str) -> str:
    return git_bytes(clone, *args).decode("utf-8", "replace").strip()


def git_bytes(clone: Path, *args: str, check: bool = True) -> bytes:
    done = subprocess.run(["git", "-C", str(clone), "--no-pager", *args], capture_output=True, check=False)  # nosec B603 B607
    if check and done.returncode != 0:
        raise Refused("git-failed", f"git {' '.join(args[:3])}: {done.stderr.decode(errors='replace').strip()[:200]}")
    return done.stdout


def repository_name(clone: Path) -> str | None:
    """`owner/name` from the clone's origin URL (https or ssh), or None."""
    url = git_bytes(clone, "config", "--get", "remote.origin.url", check=False).decode().strip()
    match = re.search(r"[:/]([^/:]+/[^/]+?)(?:\.git)?/?$", url)
    return match.group(1) if match else None


def find_merge(clone: Path, pr: int) -> str | None:
    """The newest first-parent commit of HEAD that merged or squashed PR `pr`."""
    merged, squashed = re.compile(rf"^Merge pull request #{pr}\b"), re.compile(rf"\(#{pr}\)\s*$")
    for line in git(clone, "log", "--first-parent", "--format=%H%x1f%s", "HEAD").splitlines():
        sha, _, subject = line.partition("\x1f")
        if merged.search(subject) or squashed.search(subject):
            return sha
    return None


def read_change(clone: Path, pr: int, merge: str | None, repository: str | None) -> Change:
    name = repository or repository_name(clone)
    if not name:
        raise Refused("repository-unknown", "the clone has no origin remote; pass --repository <owner/name>")
    commit = merge or find_merge(clone, pr)
    if not commit:
        raise Refused("pr-not-found", f"no first-parent commit of HEAD is 'Merge pull request #{pr}' or ends in '(#{pr})'")
    base = git_bytes(clone, "rev-parse", "--verify", "-q", f"{commit}^1", check=False).decode().strip()
    if not base:
        raise Refused("no-base-commit", f"{commit[:12]} has no parent to start the task from")
    subject = git(clone, "show", "-s", "--format=%s", commit)
    if subject.startswith(f"Merge pull request #{pr}"):
        body = git(clone, "show", "-s", "--format=%b", commit)
        title = next((line.strip() for line in body.splitlines() if line.strip()), subject)
    else:
        title = re.sub(rf"\s*\(#{pr}\)\s*$", "", subject)
    return Change(name, pr, git(clone, "rev-parse", commit), base, title,
                  git(clone, "show", "-s", "--format=%cs", base), git(clone, "show", "-s", "--format=%cs", commit))


def split_change(clone: Path, change: Change, profile: Profile) -> Split:
    raw = git_bytes(clone, "diff", "--name-status", "-z", "--no-renames", change.base, change.merge)
    fields = [field.decode("utf-8", "surrogateescape") for field in raw.split(b"\0") if field]
    changes = list(zip(fields[0::2], fields[1::2], strict=True))
    tests = [compile_glob(g) for g in profile.test_globs]
    runs = [compile_glob(g) for g in profile.run_globs]
    test_side = {path for _, path in changes if any(t.match(path) for t in tests)}
    source = tuple(path for _, path in changes if path not in test_side)
    hidden = tuple(path for status, path in changes if path in test_side and status != "D")
    deleted = tuple(path for status, path in changes if path in test_side and status == "D")
    run_files = tuple(path for path in hidden if any(r.match(path) for r in runs))
    if not test_side:
        raise Refused("no-test-change", "the change touches no test file; pick a PR that adds or changes tests")
    if not source:
        raise Refused("no-source-change", "the change touches only tests: there is nothing for an agent to write")
    if not run_files:
        raise Refused("no-run-files", f"no changed test file matches the profile's run globs ({', '.join(hidden) or 'none'})")
    return Split(source, hidden, deleted, run_files)


# ---------------------------------------------------------------- the task files
def toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(float(value)) if isinstance(value, float) else str(value)
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    return json.dumps(str(value))  # a JSON string is a valid TOML basic string


def task_toml(task_id: str, change: Change, split: Split, profile: Profile) -> str:
    """task.toml: the agent reaches only the profile's hosts, the grader nothing; graded in the agent's container."""
    sections = [
        ("task", {"name": task_id, "version": "1.0.0",
                  "description": "Make the change a merged pull request made; the tests it added or changed grade it.",
                  "keywords": ["coding-agent", "from-pull-request"]}),
        ("agent", {"timeout_sec": AGENT_TIMEOUT_SEC}),
        # No environment_mode: the verifier runs in the agent's container (Harbor's shared mode), in stages.
        ("verifier", {"timeout_sec": profile.timeout_sec + 300, "network_mode": "no-network"}),
        ("environment", {"network_mode": "allowlist", "os": "linux", "allowed_hosts": list(profile.allowed_hosts)}),
        ("metadata.origin", {"repo": change.repository, "pr_numbers": [change.pr], "base_commit": change.base,
                             "merge_commit": change.merge, "base_commit_date": change.base_date, "merged_at": change.merged_at}),
        ("metadata.task", {"kind": "coding", "built_by": "step1-build-tasks/coding_tasks.py",
                           "grading": "staged: tests/test.sh --stage untrusted, then --stage trusted (reward.json)",
                           "run_files": list(split.run_files)}),
    ]
    lines = ['schema_version = "1.4"']
    for header, table in sections:
        lines += ["", f"[{header}]", *(f"{key} = {toml_value(value)}" for key, value in table.items())]
    return "\n".join(lines) + "\n"


def instruction(change: Change, split: Split, profile: Profile) -> str:
    where = "the repository root" if profile.workdir in (".", "") else f"`{profile.workdir}`"
    files = "\n".join(f"- `{path}`" for path in split.hidden_tests)
    return (f"# {change.title}\n\n"
            f"The repository at `/work/repo` is {change.repository} at the commit just before this change was merged. "
            "Make the change described here.\n\n"
            "## How the change is checked\n\n"
            f"Hidden tests are added to the repository and run from {where} with:\n\n"
            f"    {profile.display_command.replace('{files}', ' '.join(split.run_files))}\n\n"
            f"{CALIBRATION_BEGIN}\n{CALIBRATION_END}\n\n"
            "These test files are replaced with their own versions before grading, so edits to them do not count:\n\n"
            f"{files}\n")


def dockerfile(task_id: str, change: Change, profile: Profile) -> str:
    env = "".join(f" \\\n    {key}={json.dumps(value)}" for key, value in profile.env)
    return (TEMPLATES / "Dockerfile.tmpl").read_text().format(
        task_name=task_id, repository=change.repository, base_short=change.base[:12], image=profile.image,
        environment_lines=env,
        install_lines=f"RUN cd /work/repo && {profile.install}\n" if profile.install else "",
        agent_setup_lines="".join(f"RUN {command}\n" for command in profile.agent_setup))


def task_files(clone: Path, change: Change, split: Split, profile: Profile, task_id: str) -> dict[str, bytes]:
    diff = ("diff", "--binary", "--no-renames", change.base, change.merge, "--")
    gold = b"".join(git_bytes(clone, *diff, *split.source[i:i + 200]) for i in range(0, len(split.source), 200))
    spec = {"workdir": profile.workdir, "command": profile.command, "run_files": list(split.run_files),
            "test_globs": list(profile.test_globs), "deleted": list(split.deleted_tests),
            "timeout_sec": profile.timeout_sec, "env": dict(profile.env)}
    files = {
        "task.toml": task_toml(task_id, change, split, profile).encode(),
        "instruction.md": instruction(change, split, profile).encode(),
        "environment/Dockerfile": dockerfile(task_id, change, profile).encode(),
        # mtime 0: the same base commit gives the same bytes
        "environment/repo.tar.gz": gzip.compress(git_bytes(clone, "archive", "--format=tar", change.base), mtime=0),
        "tests/test.sh": (TEMPLATES / "test.sh").read_bytes(),
        "tests/grader.py": (TEMPLATES / "grader.py").read_bytes(),
        "tests/spec.json": (json.dumps(spec, indent=2, sort_keys=True) + "\n").encode(),
        "solution/solve.sh": (TEMPLATES / "solve.sh").read_bytes(),
        "solution/gold.patch": gold,
    }
    for path in split.hidden_tests:
        files[f"tests/files/{path}"] = git_bytes(clone, "show", f"{change.merge}:{path}")
    return files


def answer_leaks(files: dict[str, bytes], pr: int) -> list[str]:
    """Where the agent could read the answer: a line the change adds, or the PR number, in instruction.md or the Dockerfile.

    Only the patch's added lines count: its context and removed lines are the base's own text, which the agent has anyway.
    """
    added = {raw[1:].strip() for raw in files["solution/gold.patch"].decode("utf-8", "replace").splitlines()
             if raw.startswith("+") and not raw.startswith("+++")}
    answers = sorted(line for line in added if len(line) >= MIN_ANSWER_LINE) + [f"#{pr}"]
    found = []
    for visible in ("instruction.md", "environment/Dockerfile"):
        text = files[visible].decode("utf-8", "replace")
        found += [f"{answer!r} in {visible}" for answer in answers if answer in text]
    return found


def build(clone: Path, pr: int, profile: Profile, out: Path, merge: str | None = None, repository: str | None = None) -> Path:
    """Write one coding task and return its folder, or raise Refused."""
    change = read_change(clone, pr, merge, repository)
    split = split_change(clone, change, profile)
    owner, _, name = change.repository.partition("/")
    task_id = f"{owner}/{name.replace('/', '__')}__pr{pr}-code"  # Harbor's `org/name` id
    files = task_files(clone, change, split, profile, task_id)
    leaks = answer_leaks(files, pr)
    if leaks:
        raise Refused("answer-visible", "; ".join(leaks[:3]))
    tasks = out / "tasks"
    tasks.mkdir(parents=True, exist_ok=True)
    final = tasks / task_id.replace("/", "__")
    staging = Path(tempfile.mkdtemp(prefix=f".{final.name}-", dir=tasks))
    try:
        for rel, data in files.items():
            (staging / rel).parent.mkdir(parents=True, exist_ok=True)
            (staging / rel).write_bytes(data)
            if rel.endswith(".sh") or rel == "tests/grader.py":
                (staging / rel).chmod(0o755)
        if final.exists():
            shutil.rmtree(final)
        staging.rename(final)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    print(f"wrote {final}: {len(split.source)} source file(s) in the reference change, "
          f"{len(split.hidden_tests)} hidden test file(s), {len(split.run_files)} run")
    return final


# ---------------------------------------------------------------- calibration
def json_object(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


def trial_statuses(jobs: Path, task_dir: Path, task_id: str) -> list[dict[str, str]]:
    """The test statuses of every trial of the task under a Harbor job folder.

    A trial is matched by Harbor's own result.json (task_name) or config.json (task path): Harbor cuts a
    trial folder's name to 32 characters, so a long task name does not identify it.
    """
    tables = []
    for results in sorted(jobs.rglob("verifier/results.json")):
        trial = results.parent.parent
        task = json_object(trial / "config.json").get("task")
        ours = (json_object(trial / "result.json").get("task_name") == task_id
                or (isinstance(task, dict) and Path(str(task.get("path") or "")).name == task_dir.name)
                or trial.name.startswith(f"{task_dir.name}__"))
        statuses = json_object(results).get("statuses")
        if ours and isinstance(statuses, dict) and statuses:
            tables.append({str(k): str(v) for k, v in statuses.items()})
    return tables


def classify(nop: list[dict[str, str]], oracle: list[dict[str, str]]) -> tuple[list[str], list[str], list[str]]:
    """(fail-to-pass, pass-to-pass, left out because their result changed between runs of one agent)."""
    tests = sorted({t for table in nop + oracle for t in table if not t.endswith(NO_TESTS_REPORTED)})
    f2p, p2p, flaky = [], [], []
    for test in tests:
        base = [table.get(test) == "passed" for table in nop]
        gold = [table.get(test) == "passed" for table in oracle]
        if all(gold) and not any(base):
            f2p.append(test)
        elif all(gold) and all(base):
            p2p.append(test)
        elif any(gold) != all(gold) or any(base) != all(base):
            flaky.append(test)
    return f2p, p2p, flaky


def calibrate(task_dir: Path, nop_jobs: Path, oracle_jobs: Path) -> tuple[list[str], list[str], list[str]]:
    """Write tests/expected.json and list the fail-to-pass tests in instruction.md, or raise Refused."""
    if not (task_dir / "tests" / "grader.py").is_file():
        raise Refused("not-a-coding-task", f"{task_dir} was not written by coding_tasks.py build")
    task_id = str((tomllib.loads((task_dir / "task.toml").read_text()).get("task") or {}).get("name") or "")
    nop, oracle = trial_statuses(nop_jobs, task_dir, task_id), trial_statuses(oracle_jobs, task_dir, task_id)
    if not nop or not oracle:
        missing = "nop" if not nop else "oracle"
        raise Refused("no-calibration-runs", f"no {missing} trial of {task_dir.name} with verifier/results.json")
    f2p, p2p, flaky = classify(nop, oracle)
    if not f2p:
        raise Refused("no-fail-to-pass", f"{len(p2p)} tests pass with and without the change: they cannot tell a fix from no fix")
    (task_dir / "tests" / "expected.json").write_text(json.dumps({"fail_to_pass": f2p, "pass_to_pass": p2p}, indent=1) + "\n")
    text = (task_dir / "instruction.md").read_text()
    begin, end = text.find(CALIBRATION_BEGIN), text.find(CALIBRATION_END)
    if 0 <= begin < end:
        shown = [f"- {t.partition('::')[2].replace('::', ' > ')} (`{t.partition('::')[0]}`)" for t in f2p[:40]]
        more = [f"- and {len(f2p) - 40} more in the same files"] if len(f2p) > 40 else []
        block = "\n".join([CALIBRATION_BEGIN, "These tests must pass. Their names, as the test runner reports them:", "",
                           *shown, *more, "", "Tests in those files that pass today must keep passing."])
        (task_dir / "instruction.md").write_text(text[:begin] + block + "\n" + text[end:])
    print(f"calibrated {task_dir.name}: {len(f2p)} fail-to-pass, {len(p2p)} pass-to-pass, {len(flaky)} left out as flaky "
          f"(from {len(nop)} nop and {len(oracle)} oracle trial(s))")
    return f2p, p2p, flaky


# ---------------------------------------------------------------- command line
def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="coding_tasks.py", description=__doc__.split("\n\n")[0])
    verbs = parser.add_subparsers(dest="verb", required=True)
    b = verbs.add_parser("build", help="one coding task per merged pull request")
    b.add_argument("--clone", type=Path, required=True, help="a local clone of the repository (with its history)")
    b.add_argument("--pr", type=int, action="append", required=True, help="a merged pull request number (repeatable)")
    b.add_argument("--profile", type=Path, required=True, help="coding-profile.toml")
    b.add_argument("--out", type=Path, required=True, help="the dataset folder; tasks go to <out>/tasks/")
    b.add_argument("--merge-commit", help="the merge or squash commit, if its subject does not name the PR (one --pr)")
    b.add_argument("--repository", help="owner/name, when the clone has no origin remote")
    c = verbs.add_parser("calibrate", help="fail-to-pass and pass-to-pass tests from Harbor's nop and oracle runs")
    c.add_argument("task_dir", type=Path)
    c.add_argument("--nop", type=Path, required=True, help="Harbor's job folder of the nop run(s)")
    c.add_argument("--oracle", type=Path, required=True, help="Harbor's job folder of the oracle run(s)")
    args = parser.parse_args(argv)
    if args.verb == "calibrate":
        try:
            calibrate(args.task_dir, args.nop, args.oracle)
        except Refused as refusal:
            print(f"refused: {refusal.code}: {refusal.detail}", file=sys.stderr)
            return 3
        return 0
    profile = read_profile(args.profile)
    refused = 0
    for pr in args.pr:
        try:
            build(args.clone, pr, profile, args.out, merge=args.merge_commit, repository=args.repository)
        except Refused as refusal:
            refused += 1
            print(f"PR #{pr} refused: {refusal.code}: {refusal.detail}", file=sys.stderr)
    return 3 if refused == len(args.pr) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
