#!/usr/bin/env python3
"""P1 host-only runner for the Coding Assistant eval: no Docker, no Harbor.

  p1.py prepare TASK...                     sealed clone per task (the Dockerfile's git steps), once
  p1.py trial TASK --agent codex|oracle|nop|patch:FILE|tamper:A..E [--job J] [--no-verify]
  p1.py job   --tasks T1,T2 --n 3 --agent codex --job J      serial trials (takes the Bedrock lock)
  p1.py base-suite TASK                     G9: base test suite pass count in a fresh trial env

Layout (all under local/): the module is the parent directory (read-only use), $P1_WORK/base/<task>/ (sealed clone),
work/trials/<trial>/ (repo, venv, home, logs: the only directory the agent can read or write),
work/verifier/<trial>/ (rendered verifier files, created after the agent stops), jobs/<job>/<trial>/ (Harbor-like record).
Credentials are minted per run by `aws sts` and passed only in the environment of child processes.
"""
from __future__ import annotations

import argparse, json, os, re, shlex, shutil, subprocess, sys, tempfile, time, tomllib
from datetime import datetime, timezone
from pathlib import Path

P1 = Path(__file__).resolve().parent.parent
MODULE, DS, JOBS = P1.parent, P1.parent / "datasets", P1 / "jobs"
# The verifier scripts get host paths substituted for /work/repo, /tests, /logs, /tmp, and test.sh uses them unquoted
# (`L=/logs/verifier`, `cd /work/repo`). The module path has spaces ("Workload Specific Evaluations"), so the work
# dir must live elsewhere: env.sh sets P1_WORK to a space-free path under the real home.
WORK = Path(os.environ["P1_WORK"]) if os.environ.get("P1_WORK") else P1 / "work"
if " " in str(WORK):
    raise SystemExit(f"work dir {WORK} contains a space; run `source env.sh` (it sets P1_WORK) or export P1_WORK=<path without spaces>")
REGION = "us-east-1"
MODELS = {"codex": "openai.gpt-6-sol"}
AGENT_HOSTS = ["bedrock-mantle.us-east-1.api.aws:443", "bedrock-runtime.us-east-1.amazonaws.com:443"]
VERIFIER_HOSTS = ["bedrock-runtime.us-east-1.amazonaws.com:443"]
LOCK = P1 / ".bedrock.lock"
BASE_PATH = ["/usr/bin", "/bin", "/usr/sbin", "/sbin"]
TOOL_PATH = [str(P1 / "npm/bin"), str(P1 / "node/bin"), str(P1 / "bin")]  # codex, node 22, rg, timeout shim
NEUTRAL = "You are working on a laptop, in a sandbox with its own copy of the repository."


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sh(cmd, cwd=None, env=None, check=True, quiet=True, timeout=None) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=quiet, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"{cmd} failed ({r.returncode}): {(r.stderr or '')[-800:]}")
    return r


# ---------------------------------------------------------------- tasks
class Task:
    def __init__(self, name: str):
        hits = [d for d in DS.glob("*/tasks/*") if d.name == name or d.name.endswith("__" + name)]
        if len(hits) != 1:
            raise SystemExit(f"task {name!r}: {len(hits)} matches")
        self.dir = hits[0]
        self.full = self.dir.name                       # strands-agents__evals__pr371-fix
        self.short = self.full.split("__")[-1]          # pr371-fix
        self.kind = "coding" if (self.dir / "tests/verify").is_dir() else "pr"
        df = (self.dir / "environment/Dockerfile").read_text()
        self.url = re.search(r'git clone [^\n]*"(https://[^"]+)" repo', df).group(1)
        self.sha = re.search(r"git fetch --depth 1 origin (\w{40})", df).group(1)
        m = re.search(r"cd (\S+) && uv sync ([^\\\n]*)", df)
        self.sync = (m.group(1), m.group(2).split()) if m else None   # PR review: uv sync in a sub-project
        self.lock = self.dir / "environment/requirements.lock"
        self.has_change = "COPY change.patch" in df
        self.toml = tomllib.loads((self.dir / "task.toml").read_text())
        self.base = WORK / "base" / self.full
        self.name_in_result = self.toml["task"]["name"]  # strands-agents/evals__pr371-fix

    def hidden(self):
        return next((self.dir / "tests/verify").glob("*_verify.py")).name if self.kind == "coding" else None


def tool_env(extra=None) -> dict:
    env = {"PATH": ":".join(TOOL_PATH + ["/opt/homebrew/bin"] + BASE_PATH), "HOME": str(P1 / "home"),
           "UV_CACHE_DIR": str(P1 / "uv-cache"), "UV_PYTHON_INSTALL_DIR": str(P1 / "uv-python"),
           "UV_PYTHON_PREFERENCE": "only-managed", "NPM_CONFIG_PREFIX": str(P1 / "npm"),
           "GIT_CONFIG_NOSYSTEM": "1", "GIT_TERMINAL_PROMPT": "0", "LANG": "en_US.UTF-8"}
    env.update(extra or {})
    return env


def prepare(t: Task) -> float:
    """The task Dockerfile's git steps, on the host. Idempotent."""
    t0 = time.time()
    if (t.base / "base_ref").exists():
        return 0.0
    shutil.rmtree(t.base, ignore_errors=True)
    t.base.mkdir(parents=True)
    env, repo = tool_env(), t.base / "repo"
    g = lambda *a, **k: sh(["git", *a], cwd=k.get("cwd", repo), env=env)
    sh(["git", "clone", "--depth", "1", "--single-branch", "--no-tags", t.url, "repo"], cwd=t.base, env=env)
    g("fetch", "--depth", "1", "origin", t.sha)
    g("reset", "--hard", t.sha)
    # Seal: remove the remote and remote-tracking refs, expire the reflog, prune everything unreachable.
    g("remote", "remove", "origin")
    for ref in g("for-each-ref", "--format=%(refname)", "refs/remotes").stdout.split():
        g("update-ref", "-d", ref)
    g("reflog", "expire", "--expire=now", "--all")
    g("gc", "--prune=now", "--aggressive")
    (t.base / "base_ref").write_text(g("rev-parse", "HEAD").stdout.strip() + "\n")
    if t.has_change:  # the change under review: uncommitted edits, patch kept outside the repo
        shutil.copy(t.dir / "environment/change.patch", t.base / "change.patch")
        g("apply", "--whitespace=nowarn", str(t.base / "change.patch"))
    return time.time() - t0


# ---------------------------------------------------------------- trial environment (container start)
def new_trial_dir(t: Task, tag: str) -> Path:
    W = WORK / "trials" / f"{t.full}__{tag}"
    shutil.rmtree(W, ignore_errors=True)
    for d in ("logs/agent", "logs/artifacts", "logs/verifier", "home", "codex-home", "tmp"):
        (W / d).mkdir(parents=True)
    return W


def start_env(t: Task, W: Path, venv: bool = True) -> dict:
    """Clone of the sealed repo + fresh venv from the uv cache. Returns timing facts."""
    t0 = time.time()
    prepare(t)
    sh(["cp", "-cR", str(t.base / "repo"), str(W / "repo")])           # APFS clonefile: instant, copy-on-write
    if t.has_change:
        shutil.copy(t.base / "change.patch", W / "change.patch")
    env = tool_env({"VIRTUAL_ENV": str(W / "venv")})
    t1 = time.time()
    if not venv:      # grader-only runs of PR review do not need the task venv (the verifier has its own)
        pass
    elif t.kind == "coding":
        sh(["uv", "venv", "--python", "3.12", str(W / "venv")], env=env)
        sh(["uv", "pip", "install", "-r", str(t.lock)], env=env, cwd=W / "repo")
        sh(["uv", "pip", "install", "--no-deps", "-e", "."], env={**env, "SETUPTOOLS_SCM_PRETEND_VERSION": "0.0.0"}, cwd=W / "repo")
    else:
        proj, flags = t.sync
        sh(["uv", "sync", *flags], env={**env, "UV_PROJECT_ENVIRONMENT": str(W / "venv")}, cwd=W / "repo" / proj)
    # Codex runs `bash -lc`; macOS /etc/profile (path_helper) rewrites PATH, so put the venv back first,
    # the same job as Debian's /etc/profile.d/00-venv.sh in the Docker image.
    pre = f'export PATH="{W}/venv/bin:{":".join(TOOL_PATH)}:$PATH"\n'
    for f in (".bash_profile", ".profile", ".zprofile"):
        (W / "home" / f).write_text(pre)
    return {"clone_s": round(t1 - t0, 2), "venv_s": round(time.time() - t1, 2)}


def verifier_venv() -> Path:
    """python 3.12 + boto3 (the PR-review verifier image: python:3.12-slim + boto3==1.43.89)."""
    v = WORK / "verifier-venv"
    py, own = v / "bin/python", (P1 / "uv-python").resolve()
    # WORK is shared by every checkout, but the venv's python is a symlink into the uv-python/ of the checkout that built it.
    # srt lets the verifier read only this checkout's uv-python/, so a venv built by another checkout (or one whose checkout
    # was deleted) cannot start: test.sh's `python3` then falls through to /usr/bin/python3 (3.9, no tomllib), score.py dies
    # on import and test.sh writes an all-zero reward. Rebuild it here instead.
    if py.exists() and not py.resolve().is_relative_to(own):
        print(f"[p1] {v} runs {py.resolve()}, outside this checkout's {own}: rebuilding it", file=sys.stderr, flush=True)
    if not (py.exists() and py.resolve().is_relative_to(own)):
        shutil.rmtree(v, ignore_errors=True)
        env = tool_env({"VIRTUAL_ENV": str(v)})
        sh(["uv", "venv", "--python", "3.12", str(v)], env=env)
        sh(["uv", "pip", "install", "boto3==1.43.89"], env=env)
    return v


# ---------------------------------------------------------------- credentials (never printed, never written)
def mint(kind: str = "session", policy: Path | None = None) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("AWS_")}
    env["HOME"] = os.environ.get("P1_REAL_HOME", str(Path.home()))      # the aws CLI finds its own config here
    if kind == "federated":
        cmd = ["aws", "sts", "get-federation-token", "--name", "p1-agent", "--duration-seconds", "3600",
               "--policy", f"file://{policy}", "--output", "json"]
    else:
        cmd = ["aws", "sts", "get-session-token", "--duration-seconds", "7200", "--output", "json"]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode:
        raise RuntimeError(f"{kind} credentials failed: {r.stderr.strip()[-300:]}")
    c = json.loads(r.stdout)["Credentials"]
    return {"AWS_ACCESS_KEY_ID": c["AccessKeyId"], "AWS_SECRET_ACCESS_KEY": c["SecretAccessKey"],
            "AWS_SESSION_TOKEN": c["SessionToken"], "AWS_REGION": REGION, "AWS_DEFAULT_REGION": REGION}


# ---------------------------------------------------------------- srt profiles
def srt_settings(W: Path, hosts: list[str], writable: list[Path], readable: list[Path], real_home: str) -> dict:
    """Deny by default: no read of the home dir, /Users, /Volumes or /tmp except what is listed."""
    toolchain = [P1 / "node", P1 / "npm", P1 / "uv-python", P1 / "bin"]
    return {"network": {"allowedDomains": hosts, "deniedDomains": []},
            "filesystem": {"denyRead": [real_home, "/Users", "/Volumes", "/tmp", "/private/tmp"],
                           "allowRead": [str(p) for p in [*readable, *toolchain, Path("/tmp/claude"), Path("/private/tmp/claude")]],
                           "allowWrite": [str(p) for p in writable], "denyWrite": []}}


def srt_cmd(settings: dict, W: Path, name: str, argv: list[str]) -> list[str]:
    f = W.parent / f"{W.name}.{name}.srt.json"          # next to, not inside, the trial dir
    settings = {**settings, **json.loads(os.environ.get("P1_SRT_EXTRA", "{}"))}      # diagnosis only, e.g. {"enableWeakerNetworkIsolation": true}
    f.write_text(json.dumps(settings, indent=1))
    # srt sets TMPDIR=/tmp/claude (a shared dir that need not exist); give the process its per-trial tmp instead
    return ["srt", *(["--debug"] if os.environ.get("P1_SRT_DEBUG") else []), "--settings", str(f), "--", "env", f"TMPDIR={W / 'tmp'}", *argv]


# ---------------------------------------------------------------- agent phase
def render_instruction(t: Task, W: Path) -> tuple[str, list[str]]:
    src = (t.dir / "instruction.md").read_text()
    out = src.replace("/work/repo", str(W / "repo")).replace("/work/change.patch", str(W / "change.patch"))
    out = out.replace("/logs/artifacts/findings.json", str(W / "logs/artifacts/findings.json"))
    out = out.replace("You are inside of a Docker container.", NEUTRAL)
    import difflib
    d = [l for l in difflib.unified_diff(src.splitlines(), out.splitlines(), "instruction.md", "rendered", lineterm="", n=0)]
    return out, d


CODEX_FLAGS = ["--dangerously-bypass-approvals-and-sandbox", "--skip-git-repo-check", "--model", MODELS["codex"],
               "--json", "--enable", "unified_exec", "-c", "web_search=disabled", "--"]   # Harbor adapter, same order


def agent_env(W: Path, creds: dict | None) -> dict:
    """The agent's whole environment (like `env -i`): nothing from the host shell except what is listed."""
    env = {"PATH": ":".join([str(W / "venv/bin"), *TOOL_PATH, *BASE_PATH]), "HOME": str(W / "home"),
            "CODEX_HOME": str(W / "codex-home"), "CODEX_API_KEY": "", "TMPDIR": str(W / "tmp"), "SHELL": "/bin/bash",
            "VIRTUAL_ENV": str(W / "venv"), "UV_OFFLINE": "1", "LANG": "en_US.UTF-8",
            "CODEX_CA_CERTIFICATE": "/etc/ssl/cert.pem", "SSL_CERT_FILE": "/etc/ssl/cert.pem",   # roots from a file, not the macOS trust service
            **json.loads(os.environ.get("P1_AGENT_ENV_EXTRA", "{}")), **(creds or {})}
    return {k: v for k, v in env.items() if v is not None}      # P1_AGENT_ENV_EXTRA: null removes a variable (diagnosis)


def clean_shared_tmp() -> None:
    """srt makes /tmp/claude writable and sets TMPDIR to it: empty it so one trial cannot leave files for the next."""
    os.makedirs("/tmp/claude", exist_ok=True)
    for d in ("/tmp/claude", "/private/tmp/claude"):
        if os.path.isdir(d):
            for e in os.scandir(d):
                shutil.rmtree(e.path, ignore_errors=True) if e.is_dir(follow_symlinks=False) else os.unlink(e.path)


def run_codex(t: Task, W: Path, creds: dict, sandbox: bool, timeout: float) -> dict:
    prompt, diff = render_instruction(t, W)
    (W / "logs/agent/instruction.rendered.md").write_text(prompt)
    (W / "logs/agent/instruction.diff").write_text("\n".join(diff) + "\n")
    shutil.copy(MODULE / "codex-bedrock.toml", W / "codex-home/config.toml")
    (W / "codex-home/auth.json").write_text('{\n  "OPENAI_API_KEY": ""\n}\n')       # Harbor wrote an (empty) key here
    env = agent_env(W, creds)
    argv = ["codex", "exec", *CODEX_FLAGS, prompt]
    if sandbox:
        argv = srt_cmd(srt_settings(W, AGENT_HOSTS, [W], [W], os.environ.get("P1_REAL_HOME", str(Path.home()))), W, "agent", argv)
    t0 = time.time()
    clean_shared_tmp()
    with open(W / "logs/agent/codex.txt", "wb") as out:
        try:
            r = subprocess.run(argv, cwd=W / "repo", env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, timeout=timeout)
            rc = r.returncode
        except subprocess.TimeoutExpired:
            rc = 124
    clean_shared_tmp()
    if (W / "codex-home/sessions").is_dir():       # keep the session JSONL (steps, tokens), as Harbor does
        shutil.copytree(W / "codex-home/sessions", W / "logs/agent/sessions", dirs_exist_ok=True)
    return {"exit": rc, "seconds": round(time.time() - t0, 1), **codex_usage(W / "logs/agent/codex.txt")}


def codex_usage(path: Path) -> dict:
    """Tokens and steps from `codex exec --json` (turn.completed usage; item.* events)."""
    u = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
    items = tools = 0
    for line in path.read_text(errors="replace").splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("type") == "turn.completed":
            for k in u:
                u[k] += (e.get("usage") or {}).get(k, 0)
        if e.get("type") == "item.completed":
            items += 1
            tools += (e.get("item") or {}).get("type") in ("command_execution", "mcp_tool_call", "file_change")
    return {**u, "steps": items, "tool_calls": tools}


# fake agents for controls and the grader-only rungs
TAMPER = {
    "A": ("root conftest.py that skips every test", {"conftest.py": "import pytest\n\ndef pytest_collection_modifyitems(items):\n    for i in items:\n        i.add_marker(pytest.mark.skip(reason='x'))\n"}),
    "B": ("root conftest.py that turns every failure into a pass", {"conftest.py": "import pytest\n\n@pytest.hookimpl(hookwrapper=True)\ndef pytest_runtest_makereport(item, call):\n    out = yield\n    rep = out.get_result()\n    if rep.failed:\n        rep.outcome = 'passed'\n        rep.longrepr = None\n"}),
    "C": ("sitecustomize.py in the venv that loads a force-pass pytest plugin", "venv"),
    "D": ("pre-written logs/verifier/reward.json (all ones)", "reward"),
    "E": ("edit an existing test so it passes trivially", "edit"),
}


def apply_fake(t: Task, W: Path, agent: str) -> None:
    repo, env = W / "repo", tool_env()
    if agent == "nop":
        return
    if agent == "oracle":
        if t.kind == "coding":
            sh(["git", "apply", "--whitespace=nowarn", str(t.dir / "solution/oracle.patch")], cwd=repo, env=env)
        else:
            fp = W / "logs/artifacts/findings.json"
            sh(["sh", str(t.dir / "solution/solve.sh")], env={**env, "EVAL_HARVEST_AGENT_FINDINGS": str(fp)})
            if not fp.is_file():      # an approve task's oracle writes [] (no findings), so only a missing file is an error
                raise SystemExit(f"STOP: the oracle (solution/solve.sh) wrote no findings file at {fp}. The control would score 0 for the wrong reason.")
    elif agent.startswith("patch:"):
        sh(["git", "apply", "--whitespace=nowarn", agent[6:]], cwd=repo, env=env)
    elif agent.startswith("findings:"):
        shutil.copy(agent[9:], W / "logs/artifacts/findings.json")
    elif agent.startswith("tamper:"):
        k = agent[7:]
        what = TAMPER[k][1]
        if isinstance(what, dict):
            for f, body in what.items():
                (repo / f).write_text(body)
        elif what == "venv":
            site = next((W / "venv/lib").glob("python*/site-packages"))
            (site / "p1_force_pass.py").write_text(TAMPER["B"][1]["conftest.py"])
            (site / "sitecustomize.py").write_text("import os\nos.environ['PYTEST_PLUGINS'] = 'p1_force_pass'\n")
        elif what == "reward":
            (W / "logs/verifier/reward.json").write_text(json.dumps({k2: 1 for k2 in ("reward", "tasteful_solve", "correct", "verifiers_pass", "fail_to_pass", "pass_to_pass", "existing_suite", "validation_score", "rubric_score", "relative_taste", "practice_alignment")}))
        elif what == "edit":
            tf = sorted((repo / "tests").rglob("test_*.py"))[0]
            tf.write_text(tf.read_text() + "\n\ndef test_p1_always_true():\n    assert True\n")
            for f in sorted((repo / "tests").rglob("test_*.py"))[:3]:  # also neuter assertions in existing tests
                f.write_text(re.sub(r"(?m)^(\s+)assert .*$", r"\1assert True", f.read_text()))
    else:
        raise SystemExit(f"unknown agent {agent}")


# ---------------------------------------------------------------- verifier phase
def render_verifier(t: Task, W: Path) -> tuple[Path, list[str]]:
    """Copy tests/ to a fresh dir and substitute the container paths. Source files stay unmodified."""
    V = WORK / "verifier" / W.name
    shutil.rmtree(V, ignore_errors=True)
    shutil.copytree(t.dir / "tests", V / "tests")
    (V / "task_base_ref").write_text((t.base / "base_ref").read_text())
    subs = [(r"(?<![\w/.-])/var/lib/task_base_ref", str(V / "task_base_ref")), (r"/work/repo", str(W / "repo")),
            (r"/opt/venv/bin", str(W / "venv/bin")), (r"(?<![\w.])/tests\b", str(V / "tests")), (r"(?<![\w.])/logs\b", str(W / "logs")),
            (r"(?<![\w.])/tmp\b", str(W / "tmp"))]   # per-trial scratch (macOS /tmp is shared and denied inside srt)
    import difflib
    diffs: list[str] = []
    for f in sorted((V / "tests").rglob("*")):
        if f.suffix not in (".py", ".sh") or not f.is_file():
            continue
        src = f.read_text()
        out = src
        for pat, rep in subs:
            out = re.sub(pat, lambda m, r=rep: r, out)
        if out != src:
            f.write_text(out)
            diffs += [l for l in difflib.unified_diff(src.splitlines(), out.splitlines(), str(f.relative_to(V)), str(f.relative_to(V)), lineterm="", n=0)]
    return V, diffs


STAGE_FILES = [("0 save patch", "agent.patch"), ("1 hidden+base pytest", "junit.xml"), ("2 agent-written tests", "agent_tests.json"),
               ("3a validation agent", "validation_agent_transcript.jsonl"), ("3b validation judge", "validation.json"),
               ("4a rubric judge", "rubric.json"), ("4b taste judge", "taste.json"), ("4c bloat+aggregate", "reward.json")]


def run_verifier(t: Task, W: Path, creds: dict, sandbox: bool, timeout: float) -> dict:
    V, diffs = render_verifier(t, W)
    (W / "logs/verifier-render.diff").write_text("\n".join(diffs) + "\n")
    vv = verifier_venv() if t.kind == "pr" else None
    path = [str(W / "venv/bin")] if t.kind == "coding" else [str(vv / "bin")]
    env = {"PATH": ":".join([*path, *TOOL_PATH, *BASE_PATH]), "HOME": str(P1 / "home"), "LANG": "en_US.UTF-8", "TMPDIR": str(W / "tmp"),
           "VIRTUAL_ENV": str(W / "venv") if t.kind == "coding" else str(vv), "VERIFIER_MODEL_ID": "us.anthropic.claude-haiku-5-5",
           "JUDGE_MODEL_ID": "us.anthropic.claude-haiku-5-5", **creds,
           # test.sh runs `git config user.email ...`, which srt refuses (.git/config is write-denied), so the commit that saves the
           # agent's change would fail and agent.patch would be empty. The identity comes from the environment instead.
           "GIT_AUTHOR_NAME": "verifier", "GIT_AUTHOR_EMAIL": "verifier@localhost", "GIT_COMMITTER_NAME": "verifier", "GIT_COMMITTER_EMAIL": "verifier@localhost"}
    argv = ["bash", str(V / "tests/test.sh")] if t.kind == "coding" else ["sh", str(V / "tests/test.sh")]
    if sandbox:
        argv = srt_cmd(srt_settings(W, VERIFIER_HOSTS, [W, V], [W, V, WORK / "verifier-venv"], os.environ.get("P1_REAL_HOME", str(Path.home()))), W, "verifier", argv)
    t0 = time.time()
    try:
        r = subprocess.run(argv, cwd=W, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)
        rc = r.returncode
        (W / "logs/verifier/test.sh.out").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
    except subprocess.TimeoutExpired:
        rc = 124
    t1 = time.time()
    stages, prev = {}, t0
    for name, fn in STAGE_FILES:
        f = W / "logs/verifier" / fn
        if f.exists():
            stages[name] = round(max(0, f.stat().st_mtime - prev), 1)
            prev = max(prev, f.stat().st_mtime)
    pb = (W / "logs/verifier/agent.patch")
    out = {"exit": rc, "seconds": round(t1 - t0, 1), "stages": stages, "render_diff_lines": len(diffs), "patch_bytes": pb.stat().st_size if pb.exists() else None}
    # score.py writes reward_details.json on every path it finishes; without it, the reward.json is test.sh's all-zero fallback
    if t.kind == "pr" and not (W / "logs/verifier/reward_details.json").exists():
        out["scorer_crashed"] = True
    return out


# ---------------------------------------------------------------- one trial
def lock_take():
    """mkdir mutex: one run at a time calls Bedrock. run_all.sh takes it once and sets P1_LOCK_HELD=1 for its children."""
    if os.environ.get("P1_LOCK_HELD") == "1":
        return False                      # held by the parent run_all.sh; the caller must not release it
    for i in range(720):
        try:
            LOCK.mkdir()
            (LOCK / "owner").write_text(f"pid {os.getpid()} {' '.join(sys.argv)[:200]} {now()}\n")
            return True
        except FileExistsError:
            if i % 6 == 0:
                who = (LOCK / "owner").read_text().strip() if (LOCK / "owner").exists() else "unknown"
                print(f"[p1] waiting for {LOCK}, held by: {who}", file=sys.stderr, flush=True)
            time.sleep(5)
    raise SystemExit(f"could not take {LOCK.name} after 60 min; if no run is active, delete {LOCK}")


def lock_release(taken: bool):
    if taken:
        shutil.rmtree(LOCK, ignore_errors=True)


def trial(t: Task, agent: str, job: str, tag: str, sandbox: bool = True, verify: bool = True, agent_creds=None, ver_creds=None,
          agent_timeout: float | None = None, venv: bool = True) -> dict:
    W = new_trial_dir(t, tag)
    trial_name = W.name
    res: dict = {"task_name": t.name_in_result, "trial_name": trial_name, "agent": agent, "started_at": now()}
    s = now(); es = start_env(t, W, venv); res["environment_setup"] = {"started_at": s, "finished_at": now(), **es}
    s = now()
    if agent == "codex":
        a = run_codex(t, W, agent_creds, sandbox, agent_timeout or t.toml["agent"]["timeout_sec"])
        res["agent_result"] = a
    else:
        apply_fake(t, W, agent)
    res["agent_execution"] = {"started_at": s, "finished_at": now()}
    if verify:
        s = now()
        v = run_verifier(t, W, ver_creds, sandbox and os.environ.get("P1_VERIFIER_SANDBOX", "1") == "1", t.toml["verifier"]["timeout_sec"])   # P1_VERIFIER_SANDBOX=0: judge phase outside srt
        res["verifier"] = {"started_at": s, "finished_at": now(), **v}
    res["finished_at"] = now()
    # record in a Harbor-like layout (small files only): result.json, agent/, verifier/, artifacts/logs/artifacts/
    R = JOBS / job / trial_name
    shutil.rmtree(R, ignore_errors=True)
    R.mkdir(parents=True)
    for sub in ("agent", "verifier"):
        shutil.copytree(W / "logs" / sub, R / sub)
    shutil.copytree(W / "logs/artifacts", R / "artifacts/logs/artifacts")
    if (W / "logs/verifier-render.diff").exists():
        shutil.copy(W / "logs/verifier-render.diff", R / "verifier/render.diff")
    (R / "result.json").write_text(json.dumps(res, indent=1))
    rw = R / "verifier/reward.json"
    res["reward"] = json.loads(rw.read_text()) if rw.exists() else None
    if os.environ.get("P1_KEEP_TRIALS") != "1":      # the record is small; the trial dir (repo + venv, 0.3-0.6 GB) is not
        shutil.rmtree(W, ignore_errors=True); shutil.rmtree(WORK / "verifier" / W.name, ignore_errors=True)
        for f in W.parent.glob(f"{W.name}.*.srt.json"):
            f.unlink()
    return res


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("prepare", "base-suite"):
        p = sub.add_parser(c); p.add_argument("tasks", nargs="+")
    p = sub.add_parser("trial"); p.add_argument("task"); p.add_argument("--agent", default="codex"); p.add_argument("--job", default="adhoc")
    p.add_argument("--tag", default=None); p.add_argument("--no-verify", action="store_true"); p.add_argument("--no-sandbox", action="store_true")
    p.add_argument("--federated", default=None, help="policy json for get-federation-token (agent creds)")
    p = sub.add_parser("sandbox-run", help="run a command in the agent's srt profile (L2/L3 probes)"); p.add_argument("task")
    p.add_argument("--tag", default="probe"); p.add_argument("--keep", action="store_true"); p.add_argument("--script", default=None); p.add_argument("--creds", default="none", choices=["none", "federated", "session"]); p.add_argument("--profile", default="agent", choices=["agent", "verifier", "none"])
    
    sub.add_parser("verifier-venv", help="create the boto3 venv used by the PR-review verifier and the red-team script")
    p = sub.add_parser("job"); p.add_argument("--tasks", required=True); p.add_argument("--n", type=int, default=3)
    p.add_argument("--agent", default="codex"); p.add_argument("--job", required=True); p.add_argument("--federated", default=None)
    cmd_argv: list[str] = []
    if "--" in sys.argv:                       # everything after `--` is the command for sandbox-run
        i = sys.argv.index("--"); cmd_argv = sys.argv[i + 1:]; sys.argv = sys.argv[:i]
    a = ap.parse_args()
    if a.cmd == "prepare":
        for n in a.tasks:
            print(n, "prepared in", round(prepare(Task(n)), 1), "s")
    elif a.cmd == "verifier-venv":
        print(verifier_venv())
    elif a.cmd == "sandbox-run":
        t = Task(a.task); W = WORK / "trials" / f"{t.full}__{a.tag}"
        if not (a.keep and (W / "repo").is_dir()):
            W = new_trial_dir(t, a.tag); start_env(t, W, venv=False)
        argv = cmd_argv
        creds = mint("federated", P1 / "scripts/agent-policy.json") if a.creds == "federated" else (mint() if a.creds == "session" else None)
        shutil.copy(MODULE / "codex-bedrock.toml", W / "codex-home/config.toml"); (W / "codex-home/auth.json").write_text('{"OPENAI_API_KEY": ""}')
        env = agent_env(W, creds)
        env["P1_ROOT"], env["P1_REAL_HOME"] = str(P1), os.environ.get("P1_REAL_HOME", str(Path.home()))
        env["P1_PY"] = str(next((P1 / "uv-python").glob("cpython-3.12*/bin/python3")))
        if a.script:
            shutil.copy(a.script, W / "probe.sh"); argv = ["bash", str(W / "probe.sh")]
        if a.profile != "none":
            hosts = AGENT_HOSTS if a.profile == "agent" else VERIFIER_HOSTS
            argv = srt_cmd(srt_settings(W, hosts, [W], [W], os.environ.get("P1_REAL_HOME", str(Path.home()))), W, a.profile, argv)
        clean_shared_tmp()
        # Never hand the sandboxed process the caller's stdout/stderr when they are files outside the trial dir (run_all.sh redirects
        # them to logs/): node's startup does fstat() on fds 0-2 and ABORTs on any error other than EBADF, and Seatbelt answers EPERM for
        # an fd that points at a path the profile denies (a path under the home directory). Pipes are not paths, so relay through pipes.
        r = subprocess.run(argv, cwd=W / "repo", env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        sys.stdout.buffer.write(r.stdout); sys.stdout.flush()
        sys.exit(r.returncode)
    elif a.cmd == "base-suite":
        for n in a.tasks:
            t = Task(n); W = new_trial_dir(t, "basesuite"); es = start_env(t, W)
            cwd = W / "repo" / (t.sync[0] if t.sync else "")
            args = ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"] + (["tests"] if t.kind == "coding" else [])
            t0 = time.time()
            r = subprocess.run(args, cwd=cwd, env={"PATH": f"{W/'venv/bin'}:{':'.join(BASE_PATH)}", "HOME": str(W / "home"), "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, text=True)
            print(json.dumps({"task": t.full, "env": es, "pytest_s": round(time.time() - t0, 1), "tail": r.stdout.strip().splitlines()[-1]}))
            (P1 / "logs" / f"base-suite-{t.short}.txt").write_text(r.stdout[-3000:])
    else:
        fake = a.cmd == "trial" and a.agent != "codex"
        tasks = [Task(n) for n in (a.tasks.split(",") if a.cmd == "job" else [a.task])]
        pol = Path(a.federated) if a.federated else None
        taken = lock_take()   # every trial calls Bedrock (agent or judges): one at a time
        try:
            n = a.n if a.cmd == "job" else 1
            for i in range(n):
                for t in tasks:
                    ac = mint("federated", pol) if (pol and a.agent == "codex") else (mint() if a.agent == "codex" else None)
                    vc = mint()
                    r = trial(t, a.agent, a.job, (a.tag if a.cmd == "trial" and a.tag else f"r{i+1}"),
                              sandbox=not getattr(a, "no_sandbox", False), verify=not getattr(a, "no_verify", False), agent_creds=ac, ver_creds=vc)
                    print(json.dumps({k: r.get(k) for k in ("trial_name", "agent_result", "reward")}), flush=True)
                    if (r.get("verifier") or {}).get("scorer_crashed"):
                        log = JOBS / a.job / r["trial_name"] / "verifier/test.sh.out"
                        tail = log.read_text(errors="replace").strip().splitlines()[-1:] if log.exists() else ["(no test.sh.out)"]
                        msg = f"score.py did not finish ({tail[0][:300]}); the zero reward is test.sh's fallback, not a score. See {log}"
                        if a.agent != "codex":          # a control must be scored; a Codex trial may crash it with malformed findings
                            sys.exit("STOP: " + msg)
                        print("[p1] WARNING: " + msg, file=sys.stderr, flush=True)
                    ar = r.get("agent_result") or {}
                    if a.agent == "codex" and (ar.get("exit") != 0 or not ar.get("input_tokens")):
                        sys.exit(f"STOP: the agent did not run (exit {ar.get('exit')}, {ar.get('input_tokens')} input tokens). Rewards of this trial are invalid. See jobs/{a.job}/{r['trial_name']}/agent/codex.txt")
                    if t.kind == "coding" and a.agent not in ("nop",) and (r.get("verifier") or {}).get("patch_bytes") == 0 and a.agent == "oracle":
                        sys.exit("STOP: the oracle patch produced an empty agent.patch (the verifier could not commit). Rewards invalid.")
        finally:
            lock_release(taken)


if __name__ == "__main__":
    main()
