"""Offline tests of AgentCoreEnvironment: no AWS call is made (the agentcore_aws functions that would call AWS are replaced)."""
import asyncio
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harbor.models.task.config import EnvironmentConfig, NetworkMode, NetworkPolicy  # noqa: E402
from harbor.models.trial.paths import TrialPaths  # noqa: E402

from harbor_env import agentcore_env as ace  # noqa: E402

MODULE = Path(__file__).resolve().parents[2]
PR371 = MODULE / "datasets/strands-evals-coding/tasks/strands-agents__evals__pr371-fix"
PR4121 = MODULE / "datasets/awslabs-mcp-pr-review/tasks/awslabs__mcp__pr4121-approve"
MANTLE, RUNTIME = "bedrock-mantle.us-east-1.api.aws", "bedrock-runtime.us-east-1.amazonaws.com"
BASE = NetworkPolicy(network_mode=NetworkMode.ALLOWLIST, allowed_hosts=[MANTLE, RUNTIME])
VERIFY = NetworkPolicy(network_mode=NetworkMode.ALLOWLIST, allowed_hosts=[RUNTIME])
PUBLIC = NetworkPolicy(network_mode=NetworkMode.PUBLIC)


class FakeSession:
    def __init__(self, protected="1 1", replies=()):
        self.commands, self.files, self.sid = [], {}, "fake-session"
        self.calls = self.throttles = 0
        self.start_attempts = 1
        self.protected = protected
        self.replies = list(replies)  # (substring of a command, output); the first match wins, and is used once

    def start(self, command, attempts=3):
        return "OK\n0\n1.0\n", 0, 0.1

    def cmd(self, command, timeout=300, retries=12):
        self.commands.append(command)
        for i, (key, out) in enumerate(self.replies):
            if key in command:
                del self.replies[i]
                return out, 0, 0.1
        if "seq 1 20" in command:
            return "alive=0 tries=1\n", 0, 0.1
        if "pkill -KILL -u agent" in command:
            return "alive=0 tries=1\n", 0, 0.1
        if "protected_hardlinks" in command:
            return f"protected={self.protected}\n", 0, 0.1
        return "ok", 0, 0.1

    def put(self, path, data, mode="0644", owner="root"):
        self.files[path] = (data, mode, owner)

    def stop(self):
        return None


@pytest.fixture
def fake_aws(monkeypatch):
    monkeypatch.setattr(ace.aws, "agent_creds", lambda seconds=3600: {"AWS_ACCESS_KEY_ID": "AGENTKEY", "AWS_SECRET_ACCESS_KEY": "s",
                                                                       "AWS_SESSION_TOKEN": "t"})
    monkeypatch.setattr(ace.aws, "runtime_role_creds", lambda s: {"AWS_ACCESS_KEY_ID": "GRADERKEY", "AWS_SECRET_ACCESS_KEY": "s",
                                                                  "AWS_SESSION_TOKEN": "t"})


def make_env(tmp_path, env_dir, session_id, baseline, phases):
    env = ace.AgentCoreEnvironment(environment_dir=env_dir, environment_name="t", session_id=session_id,
                                   trial_paths=TrialPaths(trial_dir=tmp_path), task_env_config=EnvironmentConfig(),
                                   network_policy=baseline, phase_network_policies=phases)
    env._session = FakeSession()
    return env


def extra_env(env):
    return env.session.files["/run/fw/extra.env"][0].decode()


def allow(env):
    return env.session.files["/run/fw/allow.txt"][0].decode().split()


def test_capabilities_are_exact_hostnames_only(tmp_path):
    env = make_env(tmp_path, PR371 / "environment", "x__env", BASE, [BASE, VERIFY])
    c = env.capabilities
    assert c.network_allowlist and c.network_allowlist_hostnames and c.disable_internet and c.dynamic_network_policy
    assert not c.network_allowlist_wildcard_hostnames and not c.network_allowlist_ipv4_cidrs and not c.mounted


def test_public_agent_environment_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="public"):
        make_env(tmp_path, PR371 / "environment", "x__env", PUBLIC, [PUBLIC])


def test_public_verifier_environment_is_accepted(tmp_path):
    env = make_env(tmp_path, PR4121 / "tests", "x__verifier__trial", PUBLIC, [PUBLIC])
    assert env._is_verifier_env


def test_allow_hosts():
    assert ace.allow_hosts(BASE) == [MANTLE, RUNTIME]
    assert ace.allow_hosts(NetworkPolicy(network_mode=NetworkMode.NO_NETWORK)) == []
    assert ace.allow_hosts(PUBLIC) == []


def test_credentials_from_harbor_never_reach_the_session(tmp_path):
    env = make_env(tmp_path, PR371 / "environment", "x__env", BASE, [BASE, VERIFY])
    s = env._script("echo hi", "/work/repo", {"AWS_ACCESS_KEY_ID": "ADMIN", "AWS_SECRET_ACCESS_KEY": "x", "AWS_SESSION_TOKEN": "y",
                                              "AWS_PROFILE": "my-admin", "AWS_BEARER_TOKEN_BEDROCK": "z", "AWS_REGION": "us-east-1",
                                              "CLAUDE_CODE_USE_BEDROCK": "1", "bad name": "q"})
    assert "ADMIN" not in s and "my-admin" not in s and "AWS_BEARER" not in s and "bad name" not in s
    assert "export AWS_REGION=us-east-1" in s and "export CLAUDE_CODE_USE_BEDROCK=1" in s and "cd /work/repo" in s


def test_exec_user_mapping(tmp_path):
    env = make_env(tmp_path, PR371 / "environment", "x__env", BASE, [BASE, VERIFY])
    asyncio.run(env.exec("id", user="root"))
    asyncio.run(env.exec("id"))
    asyncio.run(env.exec("id", user="agent"))
    root, agent, agent2 = env.session.commands
    assert not root.startswith("/opt/firewall/") and agent.startswith("/opt/firewall/as-agent ") and agent2.startswith("/opt/firewall/as-agent ")


def test_shared_mode_phase_switch(tmp_path, fake_aws):
    env = make_env(tmp_path, PR371 / "environment", "x__env", BASE, [BASE, VERIFY])
    asyncio.run(env._enter_phase("agent", BASE))
    assert "AGENTKEY" in extra_env(env) and allow(env) == [MANTLE, RUNTIME]
    asyncio.run(env.set_network_policy(VERIFY))  # Harbor before the shared verifier
    assert env._phase == "verifier" and "GRADERKEY" in extra_env(env) and "AGENTKEY" not in extra_env(env)
    assert allow(env) == [RUNTIME]
    assert any("pkill -KILL -u agent" in c for c in env.session.commands)
    asyncio.run(env.set_network_policy(BASE))  # Harbor after the verifier: the grader credentials stay, never the agent's again
    assert env._phase == "verifier" and "AGENTKEY" not in extra_env(env)


def test_tests_upload_ends_the_agent_phase_when_the_policies_are_equal(tmp_path, fake_aws):
    # same policy for both phases: the /tests upload ends the agent phase
    env = make_env(tmp_path, PR371 / "environment", "x__env", BASE, [BASE, BASE])
    asyncio.run(env._enter_phase("agent", BASE))
    asyncio.run(env.upload_dir(PR371 / "tests", "/tests"))
    assert env._phase == "verifier" and "GRADERKEY" in extra_env(env) and "AGENTKEY" not in extra_env(env)
    assert any("pkill -KILL -u agent" in c for c in env.session.commands)


def test_agent_phase_switch_keeps_agent_credentials(tmp_path, fake_aws):
    narrow = NetworkPolicy(network_mode=NetworkMode.ALLOWLIST, allowed_hosts=[MANTLE])
    env = make_env(tmp_path, PR371 / "environment", "x__env", BASE, [narrow, VERIFY])
    asyncio.run(env._enter_phase("agent", BASE))
    asyncio.run(env.set_network_policy(narrow))
    assert env._phase == "agent" and allow(env) == [MANTLE] and "AGENTKEY" in extra_env(env)


def test_separate_verifier_env_gets_grader_credentials(tmp_path, fake_aws):
    env = make_env(tmp_path, PR4121 / "tests", "x__verifier__trial", PUBLIC, [PUBLIC])
    asyncio.run(env._enter_phase("verifier", PUBLIC))
    assert "GRADERKEY" in extra_env(env) and allow(env) == []
    asyncio.run(env.exec("sh /tests/test.sh"))
    run, kill = env.session.commands[-2:]
    assert run.startswith("/opt/firewall/as-agent-open ") and "for i in $(seq 1 20)" in kill


def test_upload_owner(tmp_path):
    env = make_env(tmp_path, PR371 / "environment", "x__env", BASE, [BASE, VERIFY])
    assert env._owner("/tests") == "root" and env._owner("/tests/lib/x.py") == "root"
    assert env._owner("/solution/solve.sh") == "agent" and env._owner("/logs/agent/x") == "agent" and env._owner("/testsx") == "agent"


def test_image_key_depends_on_the_environment_dir(tmp_path):
    a = tmp_path / "a"
    a.mkdir()
    (a / "Dockerfile").write_text("FROM x\n")
    k1 = ace.image_key(a, "evals__pr371-fix", False)
    assert k1 == ace.image_key(a, "evals__pr371-fix", False) and k1.startswith("evalspr371fix_a")
    assert ace.image_key(a, "evals__pr371-fix", True).startswith("evalspr371fix_v")
    (a / "Dockerfile").write_text("FROM y\n")
    assert ace.image_key(a, "evals__pr371-fix", False) != k1
    assert len(ace.aws.Names.runtime(k1)) <= 48


def test_build_context_appends_the_layer(tmp_path):
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    ace.env_dir_context(PR4121 / "tests")("tag", None, ctx)
    df = (ctx / "Dockerfile").read_text()
    assert df.startswith((PR4121 / "tests" / "Dockerfile").read_text()) and "Harbor-on-AgentCore layer" in df
    for n in ("main.py", "start.sh", "proxy.py", "as-agent", "as-agent-open", "as-grader", "record-baseline.sh", "buildspec.yml", "score.py", "test.sh"):
        assert (ctx / n).is_file(), n


def test_agent_owned_uploads_are_written_as_agent(tmp_path):
    env = make_env(tmp_path, PR371 / "environment", "x__env", BASE, [BASE, VERIFY])
    f = tmp_path / "run.sh"
    f.write_text("echo hi\n")
    asyncio.run(env.upload_file(f, "/installed-agent/run.sh"))
    asyncio.run(env.upload_dir(PR371 / "environment", "/solution"))
    writes = [c for c in env.session.commands if "/installed-agent/run.sh" in c or "-C /solution" in c]
    assert len(writes) == 2
    for c in writes:
        assert "setpriv --reuid=agent" in c and "chown -R" not in c and "/tmp/" not in c
        assert "refusing symlink" in c and "chown -h agent:agent" in c
    # root only stages the bytes; it never writes the target
    assert all(p.startswith("/run/hb/") for p in env.session.files)
    assert any("chmod 0700 /run/hb" in c for c in env.session.commands)


def test_tests_upload_stays_root(tmp_path, fake_aws):
    env = make_env(tmp_path, PR371 / "environment", "x__env", BASE, [BASE, VERIFY])
    asyncio.run(env._enter_phase("agent", BASE))
    asyncio.run(env.upload_dir(PR371 / "tests", "/tests"))
    c = next(c for c in env.session.commands if "-C /tests" in c)
    assert "setpriv" not in c and "chown -R root:root /tests" in c and "/run/hb/" in c
    f = tmp_path / "x.py"
    f.write_text("x = 1\n")
    asyncio.run(env.upload_file(f, "/tests/lib/x.py"))
    assert env.session.files["/tests/lib/x.py"][2] == "root"


def test_multi_step_task_is_refused(tmp_path):
    task = tmp_path / "task"
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text("FROM x\n")
    (task / "task.toml").write_text('[[steps]]\nname = "one"\n\n[[steps]]\nname = "two"\n')
    with pytest.raises(ValueError, match="multi-step"):
        make_env(tmp_path, task / "environment", "x__env", BASE, [BASE, VERIFY])


@pytest.mark.parametrize("section", ["agent", "verifier"])
def test_root_task_user_is_refused(tmp_path, section):
    task = tmp_path / "task"
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text("FROM x\n")
    (task / "task.toml").write_text(f'[{section}]\nuser = "root"\n')
    with pytest.raises(ValueError, match="user = root"):
        make_env(tmp_path, task / "environment", "x__env", BASE, [BASE, VERIFY])


def test_root_only_through_the_call_argument(tmp_path):
    env = make_env(tmp_path, PR371 / "environment", "x__env", BASE, [BASE, VERIFY])
    env.default_user = "root"
    with pytest.raises(ValueError, match="default_user"):
        asyncio.run(env.exec("id"))
    assert env.session.commands == []
    asyncio.run(env.exec("id", user="root"))
    assert env.session.commands[-1] == "id"


def test_verifier_command_is_followed_by_the_kill_loop(tmp_path, fake_aws):
    env = make_env(tmp_path, PR371 / "environment", "x__env", BASE, [BASE, VERIFY])
    asyncio.run(env._enter_phase("agent", BASE))
    asyncio.run(env.set_network_policy(VERIFY))
    n = len(env.session.commands)
    asyncio.run(env.exec("python3 /tests/lib/aggregate.py x"))  # any verifier-phase command (test.sh itself: the tests below)
    run, kill = env.session.commands[n:]
    assert run.startswith("/opt/firewall/as-agent ") and "for i in $(seq 1 20)" in kill


def start_env(tmp_path, monkeypatch, fake_aws, protected):
    monkeypatch.setattr(ace, "ensure_runtime", lambda d, tag, log=print: {"tag": tag})
    monkeypatch.setattr(ace.aws, "Session", lambda tag: FakeSession(protected))
    env = make_env(tmp_path, PR371 / "environment", "x__env", BASE, [BASE, VERIFY])
    env._session = None
    asyncio.run(env.start(force_build=False))
    return env


def test_start_checks_the_kernel_link_protection(tmp_path, monkeypatch, fake_aws):
    env = start_env(tmp_path, monkeypatch, fake_aws, "1 1")
    assert any("sysctl -qw fs.protected_hardlinks=1 fs.protected_symlinks=1" in c for c in env.session.commands)
    text = (tmp_path / "agentcore.jsonl").read_text()
    assert '"protected_hardlinks": "1", "protected_symlinks": "1"' in text


@pytest.mark.parametrize("protected", ["0 1", "1 0", ""])
def test_start_refuses_without_kernel_link_protection(tmp_path, monkeypatch, fake_aws, protected):
    with pytest.raises(RuntimeError, match="protected_hardlinks"):
        start_env(tmp_path, monkeypatch, fake_aws, protected)


# Harbor 0.26's verifier command
HARBOR_TEST_CMD = "(/tests/test.sh) > /logs/verifier/test-stdout.txt 2>&1"
STAGED = [("grep -q -- '--stage'", "yes\n"), ("prepare", ""), ("sha256sum", "sha=aaa\n"), ("sha256sum", "sha=aaa\n"),
          ("find -P . -type f", "copied=7\n"),
          ("stat -c", "reward_files=reward.json \nmode /logs/verifier root 755\n")]


def grading_env(tmp_path, fake_aws, replies, **kwargs):
    env = ace.AgentCoreEnvironment(environment_dir=PR371 / "environment", environment_name="t", session_id="x__env",
                                   trial_paths=TrialPaths(trial_dir=tmp_path), task_env_config=EnvironmentConfig(),
                                   network_policy=BASE, phase_network_policies=[BASE, VERIFY], **kwargs)
    env._session = FakeSession(replies=replies)
    asyncio.run(env._enter_phase("agent", BASE))
    asyncio.run(env.set_network_policy(VERIFY))
    env.session.commands.clear()
    return env


def events(tmp_path, name):
    return [e for e in map(json.loads, (tmp_path / "agentcore.jsonl").read_text().splitlines()) if e["event"] == name]


def index_of(commands, *needles):
    """The index of the first command (from the previous match on) that contains each needle, in order."""
    out, start = [], 0
    for n in needles:
        i = next(i for i in range(start, len(commands)) if n in commands[i])
        out.append(i)
        start = i + 1
    return out


def test_staged_grading_command_sequence(tmp_path, fake_aws):
    env = grading_env(tmp_path, fake_aws, STAGED)
    r = asyncio.run(env.exec(HARBOR_TEST_CMD, env={"VERIFIER_MODEL_ID": "m", "AWS_SECRET_ACCESS_KEY": "ADMIN"}))
    c = env.session.commands
    steps = index_of(c, "grep -q -- '--stage'", "--git-dir=/opt/hidden/base.git", "/opt/firewall/as-agent ", "pkill -KILL -u agent;",
                     "/run/hb/grade/after/agent.patch", "find -P . -type f", "/opt/firewall/as-grader ", "pkill -KILL -u grader;",
                     "cp -rP /run/hb/grade/out/. /logs/verifier/")
    assert steps == sorted(steps) and r.return_code == 0
    prep, untrusted, trusted, final = c[steps[1]], c[steps[2]], c[steps[6]], c[steps[8]]
    # 1. root patch first, from the root-only git dir, with no config or hooks of the agent's repository
    assert "GIT_CONFIG_NOSYSTEM=1" in prep and "core.hooksPath=/dev/null" in prep and "core.fsmonitor=false" in prep
    assert "install -d -m 0700 -o grader -g grader /run/hb/grade/out" in prep and "rm -rf /run/hb/grade /run/hb/untrusted" in prep
    # 2. untrusted as agent into a fresh agent dir; 6. trusted as grader on root's copies
    assert "--stage untrusted" in untrusted and "OUT=/run/hb/untrusted/out" in untrusted and "ADMIN" not in untrusted
    assert "--stage trusted" in trusted and "IN=/run/hb/grade/in" in trusted and "OUT=/run/hb/grade/out" in trusted
    assert not any(x.startswith("/opt/firewall/as-agent ") and "--stage trusted" in x for x in c)
    # 5. root's patch replaces the untrusted stage's copy; 7. /logs/verifier ends root-owned and not agent-writable
    assert "cp /run/hb/grade/base/agent.patch /run/hb/grade/base/agent.stat /run/hb/grade/in/" in c[steps[5]]
    assert "chown -R root:root /logs/verifier" in final and "go=rX" in final
    mode = events(tmp_path, "grader_mode")[-1]
    assert mode["mode"] == "staged" and mode["result"] == "graded" and mode["reward_json"] is True


def test_repo_changed_during_the_untrusted_stage_is_not_graded(tmp_path, fake_aws):
    replies = list(STAGED)
    replies[3] = ("sha256sum", "sha=bbb\n")  # the second root patch differs
    env = grading_env(tmp_path, fake_aws, replies)
    r = asyncio.run(env.exec(HARBOR_TEST_CMD))
    assert r.return_code == 1 and not any(x.startswith("/opt/firewall/as-grader ") for x in env.session.commands)
    assert any("grading_invalid.txt" in x and "chown -R root:root /logs/verifier" in x for x in env.session.commands)
    mode = events(tmp_path, "grader_mode")[-1]
    assert mode["result"].startswith("invalid") and mode["patch_sha"] == "aaa" and mode["patch_sha_after"] == "bbb"


def test_test_sh_without_stage_is_graded_unsealed(tmp_path, fake_aws):
    env = grading_env(tmp_path, fake_aws, [("grep -q -- '--stage'", "no=test.sh has no --stage\n")])
    asyncio.run(env.exec(HARBOR_TEST_CMD))
    assert any(x.startswith("/opt/firewall/as-agent ") and HARBOR_TEST_CMD in x for x in env.session.commands)
    assert not any("--stage untrusted" in x or x.startswith("/opt/firewall/as-grader ") for x in env.session.commands)
    assert events(tmp_path, "grader_mode")[-1]["mode"] == "unsealed" and events(tmp_path, "unsealed_grader")


def test_staged_grader_switch_off(tmp_path, fake_aws):
    env = grading_env(tmp_path, fake_aws, [], staged_grader="false")
    asyncio.run(env.exec(HARBOR_TEST_CMD))
    assert events(tmp_path, "staged_grader_off")
    assert events(tmp_path, "grader_mode")[-1] == {**events(tmp_path, "grader_mode")[-1], "mode": "unsealed",
                                                    "reason": "staged_grader=false"}
    assert not any("grep -q -- '--stage'" in x for x in env.session.commands)


def test_other_verifier_commands_are_not_staged(tmp_path, fake_aws):
    env = grading_env(tmp_path, fake_aws, [])
    asyncio.run(env.exec("cat /logs/verifier/reward.json"))
    assert len(env.session.commands) == 2 and not events(tmp_path, "grader_mode")


def test_copy_step_refuses_symlinks(tmp_path):
    src, dest, outside = tmp_path / "stage", tmp_path / "grade" / "in", tmp_path / "secret.txt"
    (src / "sub").mkdir(parents=True)
    dest.mkdir(parents=True)
    outside.write_text("not for the grader\n")
    (src / "junit.xml").write_text("<x/>\n")
    (src / "sub" / "a.json").write_text("{}\n")
    (src / "big.log").write_bytes(b"x" * 2000)
    (src / "link.txt").symlink_to(outside)
    (src / "linkdir").symlink_to(tmp_path)
    out = subprocess.run(["bash", "-c", ace.copy_regular_files_cmd(str(src), str(dest), max_bytes=1000)],
                         capture_output=True, text=True, check=True).stdout
    got = sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file() or p.is_symlink())
    assert got == ["junit.xml", "sub/a.json"] and "copied=2" in out
    log = (dest.parent / "copy.log").read_text()
    assert "skipped symlink: ./link.txt" in log and "skipped symlink: ./linkdir" in log and "skipped size: ./big.log" in log


def test_root_patch_does_not_honor_an_agent_gitignore(tmp_path):
    repo, out = tmp_path / "repo", tmp_path / "out"
    (repo / "src").mkdir(parents=True)
    out.mkdir()
    (repo / "src" / "pkg.py").write_text("def add(a, b):\n    return a - b\n")
    (repo / ".gitignore").write_text("*.log\n")
    git = ["git", "-c", "user.email=a@b", "-c", "user.name=a", "-C", str(repo)]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(git + ["add", "-A"], check=True)
    subprocess.run(git + ["commit", "-qm", "base"], check=True)
    (tmp_path / "base_ref").write_text(subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True).stdout)
    import shutil
    record_baseline(repo, tmp_path / "base.git")  # as the image layer does, before the agent runs
    # a fix, files hidden by .gitignore and info/exclude, and caches
    (repo / "src" / "pkg.py").write_text("def add(a, b):\n    return a + b\n")
    (repo / "src" / ".gitignore").write_text("extra.py\n")
    (repo / "src" / "extra.py").write_text("X = 1\n")
    (repo / ".git" / "info").mkdir(exist_ok=True)
    (repo / ".git" / "info" / "exclude").write_text("other.py\n")
    (repo / "other.py").write_text("Y = 2\n")
    (repo / "run.log").write_text("ignored by the base .gitignore, still in the patch\n")
    for cache in ("src/__pycache__/pkg.cpython-312.pyc", "top.pyc", ".pytest_cache/v/cache/lastfailed"):
        (repo / cache).parent.mkdir(parents=True, exist_ok=True)
        (repo / cache).write_bytes(b"cache")
    r = root_patch(repo, tmp_path / "base.git", out)
    patch = (out / "agent.patch").read_text()
    for path in ("src/pkg.py", "src/.gitignore", "src/extra.py", "other.py", "run.log"):
        assert f"diff --git a/{path} b/{path}" in patch, path
    assert "__pycache__" not in patch and ".pyc" not in patch and ".pytest_cache" not in patch
    assert "+    return a + b" in patch and "src/extra.py" in (out / "agent.stat").read_text()
    assert len(r.stdout.strip().removeprefix("sha=")) == 64



def record_baseline(repo, base_git):
    """What image/layer.Dockerfile does: copy the git dir, then record-baseline.sh."""
    import shutil
    shutil.copytree(repo / ".git", base_git)
    subprocess.run(["bash", str(ace.LAYER_DIR / "record-baseline.sh"), str(repo), str(base_git)], check=True, capture_output=True)


def root_patch(repo, base_git, out):
    cmd = ace.root_patch_cmd(str(out), repo=str(repo), base_git=str(base_git), index=str(out.parent / "index"))
    return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=True)


def test_build_time_ignored_files_are_not_in_the_patch(tmp_path):
    repo, out = tmp_path / "repo", tmp_path / "out"
    (repo / "src" / "pkg").mkdir(parents=True)
    out.mkdir()
    (repo / "src" / "pkg" / "__init__.py").write_text("X = 1\n")
    (repo / ".gitignore").write_text("*.egg-info/\nsrc/pkg/_version.py\n")
    git = ["git", "-c", "user.email=a@b", "-c", "user.name=a", "-C", str(repo)]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(git + ["add", "-A"], check=True)
    subprocess.run(git + ["commit", "-qm", "base"], check=True)
    # files the image build writes, hidden by .gitignore
    (repo / "src" / "pkg.egg-info").mkdir()
    (repo / "src" / "pkg.egg-info" / "PKG-INFO").write_text("Name: pkg\n")
    (repo / "src" / "pkg" / "_version.py").write_text("version = '0.0.0'\n")
    (repo / "src" / "pkg" / "__pycache__").mkdir()
    (repo / "src" / "pkg" / "__pycache__" / "__init__.cpython-312.pyc").write_bytes(b"cache")
    record_baseline(repo, tmp_path / "base.git")
    # a nop agent; grading writes more caches
    (repo / ".pytest_cache").mkdir()
    (repo / ".pytest_cache" / "README.md").write_text("cache\n")
    r = root_patch(repo, tmp_path / "base.git", out)
    assert (out / "agent.patch").read_bytes() == b"" and (out / "agent.stat").read_bytes() == b""
    assert r.stdout.strip() == "sha=" + hashlib.sha256(b"").hexdigest()
    # an agent that changes a build-time file shows up
    (repo / "src" / "pkg" / "_version.py").write_text("version = '9.9.9'\n")
    root_patch(repo, tmp_path / "base.git", out)
    assert "diff --git a/src/pkg/_version.py b/src/pkg/_version.py" in (out / "agent.patch").read_text()


def test_baseline_script_and_root_patch_leave_out_the_same_caches():
    script = (ace.LAYER_DIR / "record-baseline.sh").read_text()
    assert all(f"'{p}'" in script for p in ace.CACHE_PATHSPECS) and ace.BASELINE_REF in script
    assert "record-baseline.sh /work/repo /opt/hidden/base.git" in (ace.LAYER_DIR / "layer.Dockerfile").read_text()


def test_harbor_verifier_command_is_this_string():
    from harbor.models.task.config import TaskOS
    from harbor.utils.scripts import build_execution_command
    assert build_execution_command("/tests/test.sh", stdout_path="/logs/verifier/test-stdout.txt",
                                   task_os=TaskOS.LINUX) == HARBOR_TEST_CMD
    assert ace.harbor_test_stdout(HARBOR_TEST_CMD) == "/logs/verifier/test-stdout.txt"
    for other in ("bash /tests/test.sh", "(/tests/test.sh; true) > /logs/verifier/x 2>&1", "(/tests/test.sh) > /tmp/x 2>&1",
                  "(/tests/test.sh) > /logs/verifier/../x 2>&1", "/tests/test.sh --stage trusted"):
        assert ace.harbor_test_stdout(other) is None, other


def test_staged_grade_writes_harbors_stdout_file(tmp_path, fake_aws):
    env = grading_env(tmp_path, fake_aws, STAGED)
    asyncio.run(env.exec(HARBOR_TEST_CMD))
    final = next(c for c in env.session.commands if "cp -rP /run/hb/grade/out/. /logs/verifier/" in c)
    assert "/run/hb/grade/out/test-stdout-trusted.txt > /logs/verifier/test-stdout.txt" in final
    assert events(tmp_path, "grader_mode")[-1]["stdout_path"] == "/logs/verifier/test-stdout.txt"
    assert not any(c.startswith("/opt/firewall/as-agent ") and "(/tests/test.sh)" in c for c in env.session.commands)


def test_other_test_sh_commands_are_refused_when_stages_are_supported(tmp_path, fake_aws):
    env = grading_env(tmp_path, fake_aws, [("grep -q -- '--stage'", "yes\n")])
    with pytest.raises(RuntimeError, match="not grading"):
        asyncio.run(env.exec("bash /tests/test.sh > /logs/verifier/test-stdout.txt 2>&1"))
    assert not any(c.startswith("/opt/firewall/as-agent ") for c in env.session.commands)
    assert events(tmp_path, "grader_refused") and events(tmp_path, "grader_mode")[-1]["mode"] == "refused"


def test_switch_off_runs_any_test_sh_command_unsealed(tmp_path, fake_aws):
    env = grading_env(tmp_path, fake_aws, [], staged_grader="false")
    asyncio.run(env.exec("bash /tests/test.sh"))
    assert any(c.startswith("/opt/firewall/as-agent ") and "bash /tests/test.sh" in c for c in env.session.commands)
    assert events(tmp_path, "grader_mode")[-1]["mode"] == "unsealed"


def test_separate_verifier_logs_grader_mode(tmp_path, fake_aws):
    env = make_env(tmp_path, PR4121 / "tests", "x__verifier__trial", PUBLIC, [PUBLIC])
    asyncio.run(env._enter_phase("verifier", PUBLIC))
    asyncio.run(env.exec(HARBOR_TEST_CMD))
    assert events(tmp_path, "grader_mode")[-1]["mode"] == "separate"


def test_a_failed_staged_step_logs_grader_mode(tmp_path, fake_aws):
    class Failing(FakeSession):
        def cmd(self, command, timeout=300, retries=12):
            if "read-tree" in command:
                self.commands.append(command)
                return "fatal: bad baseline", 128, 0.1
            return super().cmd(command, timeout, retries)
    env = grading_env(tmp_path, fake_aws, [])
    env._session = Failing(replies=[("grep -q -- '--stage'", "yes\n")])
    with pytest.raises(RuntimeError, match="prepare and patch"):
        asyncio.run(env.exec(HARBOR_TEST_CMD))
    assert events(tmp_path, "grader_mode")[-1]["result"].startswith("error: RuntimeError")


# ----- a reward file of the untrusted stage never becomes Harbor's reward
AGENT_REWARD = '{"reward": 1.0, "tampered": 1.0}'


def untrusted_stage_out(stage):
    """What a tampering untrusted stage leaves in STAGE_OUT: its real outputs plus planted reward and judge files."""
    (stage / "nested").mkdir(parents=True)
    for name, text in (("junit.xml", "<x/>\n"), ("validation.json", '{"status": "error"}\n'), ("test-stdout.txt", "untrusted\n"),
                       ("reward.json", AGENT_REWARD), ("reward.txt", "1.0"), ("reward_details.json", "{}"),
                       ("rubric.json", '{"rubric_score": 1.0}'), ("taste_judge_transcript.jsonl", "{}\n"),
                       ("validation_judge_transcript.jsonl", "{}\n"),
                       ("nested/reward.json", AGENT_REWARD)):
        (stage / name).write_text(text)


def run_sh(cmd):
    return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=True).stdout


def trusted_out(out, outcome):
    """The trusted stage's GRADE_DIR/out: it copies in/ (test.sh --stage trusted), then aggregate.py writes the reward."""
    out.mkdir()
    if outcome == "crashed":
        return
    for f in (out.parent / "in").iterdir():
        if f.is_file():
            (out / f.name).write_bytes(f.read_bytes())
    (out / "reward_details.json").write_text('{"invalid": []}')
    if outcome == "graded":
        (out / "reward.json").write_text('{"reward": 0.0}')
    else:  # aggregate.py on an invalid trial: no reward.json, an empty reward.txt
        (out / "reward.txt").write_text("")


@pytest.mark.parametrize("outcome", ["graded", "invalid", "crashed", "repository changed"])
def test_untrusted_reward_files_never_reach_harbor(tmp_path, outcome):
    stage, grade, logs = tmp_path / "stage", tmp_path / "grade", tmp_path / "logs" / "verifier"
    untrusted_stage_out(stage)
    (grade / "in").mkdir(parents=True)
    copied = run_sh(ace.copy_regular_files_cmd(str(stage), str(grade / "in")))
    copied_names = sorted(str(p.relative_to(grade / "in")) for p in (grade / "in").rglob("*") if p.is_file())
    # the untrusted stage writes the validation judge's transcript on purpose, so it is kept
    assert copied_names == ["junit.xml", "test-stdout.txt", "validation.json", "validation_judge_transcript.jsonl"] and "copied=4" in copied
    log = (grade / "copy.log").read_text()
    for name in ("reward.json", "reward.txt", "reward_details.json", "rubric.json", "taste_judge_transcript.jsonl", "nested/reward.json"):
        assert f"skipped name: ./{name}" in log, name
    graded = outcome != "repository changed"
    if graded:
        trusted_out(grade / "out", outcome)
    out = run_sh(ace.final_copy_cmd(str(grade), str(logs), graded=graded))
    top = sorted(p.name for p in logs.iterdir())
    rewards = [n for n in top if n in ("reward.json", "reward.txt")]
    assert ace.AgentCoreEnvironment._marker(out, "reward_files").split() == rewards
    assert rewards == (["reward.json"] if outcome == "graded" else [])
    if outcome == "graded":
        assert json.loads((logs / "reward.json").read_text()) == {"reward": 0.0}
    assert (logs / "untrusted" / "junit.xml").is_file() and not list(logs.rglob("*tampered*"))
    assert not any(AGENT_REWARD in p.read_text() for p in logs.rglob("*") if p.is_file())


def test_final_copy_keeps_a_reward_file_in_in_out_of_the_top(tmp_path):
    # defense in depth: even a reward file that got into in/ stays under untrusted/
    grade, logs = tmp_path / "grade", tmp_path / "verifier"
    (grade / "in").mkdir(parents=True)
    (grade / "in" / "reward.json").write_text(AGENT_REWARD)
    (grade / "in" / "reward.txt").write_text("1.0")
    (grade / "copy.log").write_text("")
    run_sh(ace.final_copy_cmd(str(grade), str(logs), graded=False))
    assert not (logs / "reward.json").exists() and not (logs / "reward.txt").exists()
    assert (logs / "untrusted" / "reward.json").is_file()


def test_harbor_stdout_path_is_never_a_reward_file():
    for path in ("/logs/verifier/reward.json", "/logs/verifier/reward.txt", "/logs/verifier/sub/reward.json"):
        assert ace.harbor_test_stdout(f"(/tests/test.sh) > {path} 2>&1") is None, path


@pytest.mark.parametrize("reward_files, result", [("reward.json ", "graded"), ("", "invalid: no reward.json from the trusted stage")])
def test_staged_grade_reports_the_trusted_outcome(tmp_path, fake_aws, reward_files, result):
    replies = STAGED[:-1] + [("stat -c", f"reward_files={reward_files}\nmode /logs/verifier root 755\n")]
    env = grading_env(tmp_path, fake_aws, replies)
    asyncio.run(env.exec(HARBOR_TEST_CMD))
    final = next(c for c in env.session.commands if "cp -rP /run/hb/grade/out/. /logs/verifier/" in c)
    assert "cp -rP /run/hb/grade/in/. /logs/verifier/untrusted/" in final
    assert final.index("rm -f /logs/verifier/reward.json /logs/verifier/reward.txt") < final.index("/run/hb/grade/out/.")
    mode = events(tmp_path, "grader_mode")[-1]
    assert mode["result"] == result and mode["reward_json"] is bool(reward_files)


def test_staged_grade_fails_on_an_unexpected_reward_file(tmp_path, fake_aws):
    replies = STAGED[:-1] + [("stat -c", "reward_files=reward.json reward.txt \n")]
    env = grading_env(tmp_path, fake_aws, replies)
    with pytest.raises(RuntimeError, match="unexpected reward files"):
        asyncio.run(env.exec(HARBOR_TEST_CMD))


def test_repo_changed_branch_leaves_no_reward_file(tmp_path, fake_aws):
    replies = list(STAGED)
    replies[3] = ("sha256sum", "sha=bbb\n")
    env = grading_env(tmp_path, fake_aws, replies)
    asyncio.run(env.exec(HARBOR_TEST_CMD))
    final = next(c for c in env.session.commands if "> /logs/verifier/grading_invalid.txt" in c)
    assert "cp -rP /run/hb/grade/in/. /logs/verifier/untrusted/" in final and "/run/hb/grade/out/." not in final
    assert "rm -f /logs/verifier/reward.json /logs/verifier/reward.txt" in final
    assert events(tmp_path, "grader_mode")[-1]["reward_json"] is False
    env2 = grading_env(tmp_path / "2", fake_aws, list(replies[:5]) + [("> /logs/verifier/grading_invalid.txt", "reward_files=reward.json \n")])
    with pytest.raises(RuntimeError, match="reward file is left"):
        asyncio.run(env2.exec(HARBOR_TEST_CMD))


# ----- root works only in directories reached through no symlink
def guard(path, owner=None):
    return subprocess.run(["bash", "-c", ace.real_dir_guard(str(path), owner) + " && echo passed"], capture_output=True, text=True)


def test_real_dir_guard(tmp_path):
    import getpass
    real, other = tmp_path / "real", tmp_path / "other"
    real.mkdir()
    other.mkdir()
    (tmp_path / "link").symlink_to(real)
    (tmp_path / "parent-link").symlink_to(tmp_path)
    assert guard(real).stdout.strip() == "passed" and guard(real, getpass.getuser()).stdout.strip() == "passed"
    for path, owner in ((tmp_path / "link", None), (tmp_path / "parent-link" / "real", None), (tmp_path / "missing", None),
                        (real, "root"), (real, "no-such-user-hb")):
        r = guard(path, owner)
        assert r.returncode == 1 and "refusing" in r.stdout and "passed" not in r.stdout, (path, owner)


def test_root_patch_refuses_a_symlinked_repo(tmp_path):
    repo, out = tmp_path / "repo", tmp_path / "out"
    repo.mkdir()
    out.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "a.py").write_text("A = 1\n")
    git = ["git", "-c", "user.email=a@b", "-c", "user.name=a", "-C", str(repo)]
    subprocess.run(git + ["add", "-A"], check=True)
    subprocess.run(git + ["commit", "-qm", "base"], check=True)
    record_baseline(repo, tmp_path / "base.git")
    (tmp_path / "work-link").symlink_to(repo)
    cmd = ace.root_patch_cmd(str(out), repo=str(tmp_path / "work-link"), base_git=str(tmp_path / "base.git"),
                             index=str(tmp_path / "index"))
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    assert r.returncode == 1 and "refusing" in r.stdout and not (out / "agent.patch").exists()


def test_staged_driver_guards_every_root_step(tmp_path, fake_aws):
    env = grading_env(tmp_path, fake_aws, STAGED)
    asyncio.run(env.exec(HARBOR_TEST_CMD))
    c = env.session.commands
    prep = next(x for x in c if "install -d -m 0711 /run/hb/untrusted" in x)
    assert "install -d -m 0700 -o agent -g agent /run/hb/untrusted/out" in prep
    assert prep.index(ace.real_dir_guard("/work/repo")) < prep.index("cd /work/repo")
    patches = [x for x in c if "read-tree refs/hb/baseline" in x]
    assert len(patches) == 2 and all(ace.real_dir_guard("/work/repo") + " && cd /work/repo && rm -f" in x for x in patches)
    copy = next(x for x in c if "find -P . -type f" in x)
    assert copy.startswith(ace.real_dir_guard("/run/hb/untrusted/out", "agent")) and "cd -P /run/hb/untrusted/out" in copy
    assert not any("/tmp/hb-stage" in x for x in c)


def test_layer_keeps_work_root_owned():
    df = (ace.LAYER_DIR / "layer.Dockerfile").read_text()
    assert "chown root:root /work && chmod 0755 /work" in df and "find /work -mindepth 1 -maxdepth 1 -exec chown -R agent:agent" in df
    assert "chown -R agent:agent /work;" not in df


# ----- downloads are checked against their sha256
class ShellSession(FakeSession):
    """Runs each command in a local bash; the first `cut` replies lose the second half of their output."""

    def __init__(self, cut=0):
        super().__init__()
        self.cut = cut

    def cmd(self, command, timeout=300, retries=12):
        self.commands.append(command)
        p = subprocess.run(["bash", "-c", command], capture_output=True, text=True)
        out = p.stdout + p.stderr
        if self.cut:
            self.cut -= 1
            out = out[: len(out) // 2]
        return out, p.returncode, 0.1


def download_env(tmp_path, monkeypatch, session):
    monkeypatch.setattr(ace, "STAGE_DIR", str(tmp_path / "stage"))
    env = ace.AgentCoreEnvironment.__new__(ace.AgentCoreEnvironment)
    env._session, env._phase, env._is_verifier_env = session, "verifier", False
    env.session_id, env._events_path = "t__env", tmp_path / "agentcore.jsonl"
    env.logger = type("L", (), {"warning": lambda self, m: None, "info": lambda self, m: None})()
    return env


def remote_dir(tmp_path):
    src = tmp_path / "remote"
    (src / "sub").mkdir(parents=True)
    (src / "reward.json").write_text('{"reward": 0.0}')
    (src / "sub" / "big.txt").write_bytes(os.urandom(200_000))
    return src


@pytest.mark.parametrize("cut", [0, 1, 2])
def test_download_dir_retries_a_cut_short_archive(tmp_path, monkeypatch, cut):
    src, dest = remote_dir(tmp_path), tmp_path / "local"
    env = download_env(tmp_path, monkeypatch, ShellSession(cut=cut))
    asyncio.run(env.download_dir(str(src), dest))
    assert (dest / "reward.json").read_text() == '{"reward": 0.0}'
    assert (dest / "sub" / "big.txt").read_bytes() == (src / "sub" / "big.txt").read_bytes()
    retries = events(tmp_path, "download_retry") if (tmp_path / "agentcore.jsonl").exists() else []
    assert len(env.session.commands) == cut + 1 and len(retries) == cut
    assert not list((tmp_path / "stage").iterdir())


def test_download_dir_fails_loud_after_three_cut_short_archives(tmp_path, monkeypatch):
    env = download_env(tmp_path, monkeypatch, ShellSession(cut=ace.DOWNLOAD_ATTEMPTS))
    with pytest.raises(RuntimeError, match="cut short 3 times"):
        asyncio.run(env.download_dir(str(remote_dir(tmp_path)), tmp_path / "local"))


def test_download_file_retries_and_reports_missing(tmp_path, monkeypatch):
    src = tmp_path / "f.bin"
    src.write_bytes(os.urandom(50_000))
    env = download_env(tmp_path, monkeypatch, ShellSession(cut=1))
    asyncio.run(env.download_file(str(src), tmp_path / "out" / "f.bin"))
    assert (tmp_path / "out" / "f.bin").read_bytes() == src.read_bytes()
    with pytest.raises(FileNotFoundError):
        asyncio.run(env.download_file(str(tmp_path / "missing"), tmp_path / "out" / "m"))
    with pytest.raises(FileNotFoundError):
        asyncio.run(env.download_dir(str(tmp_path / "missing"), tmp_path / "out2"))


# ----- platform version and the memory for large images
@pytest.mark.parametrize("gb,want", [(1.4, None), (2.0, None), (3, 4096), (9.9, 8192)])
def test_image_memory_mib(gb, want):
    assert ace.image_memory_mib(int(gb * ace.GIB)) == want


def test_image_memory_mib_refuses_an_image_over_the_limit():
    with pytest.raises(ValueError, match="over AgentCore"):
        ace.image_memory_mib(int(10.5 * ace.GIB))


@pytest.mark.parametrize("current,want_deploy", [(("V2", 4096), False), (("V1", None), True), (("V2", None), True)])
def test_ensure_runtime_redeploys_on_other_settings(monkeypatch, tmp_path, current, want_deploy):
    deployed = []
    monkeypatch.setattr(ace.aws, "image_exists", lambda tag: True)
    monkeypatch.setattr(ace.aws.ecr, "describe_images", lambda **k: {"imageDetails": [{"imageSizeInBytes": 3 * ace.GIB}]})
    monkeypatch.setattr(ace.aws, "find_runtime", lambda tag: {"agentRuntimeId": "rid", "status": "READY"})
    monkeypatch.setattr(ace, "runtime_settings", lambda rid: current)
    monkeypatch.setattr(ace.aws, "deploy", lambda tag, platform, memory: deployed.append((platform, memory)) or
                        {"secs": 1, "status": "READY", "reason": None, "imdsv2_required": True})
    out = ace.ensure_runtime(tmp_path, "t_a0", log=lambda m: None)
    assert deployed == ([("V2", 4096)] if want_deploy else []) and out["memory_mib"] == 4096



# ----- code that the patch would not show: compiled caches and nested git repositories
def test_prepare_drops_bytecode_and_a_nested_git_repo_is_not_graded(tmp_path, fake_aws):
    replies = [("grep -q -- '--stage'", "yes\n"), ("-name .git -print -quit", "nested=/work/repo/vendor/x/.git\nsha=aaa\n")]
    env = grading_env(tmp_path, fake_aws, replies)
    r = asyncio.run(env.exec(HARBOR_TEST_CMD))
    prep = next(c for c in env.session.commands if "-name .git -print -quit" in c)
    assert ace.DROP_BYTECODE in prep
    assert r.return_code == 1 and not any("--stage untrusted" in c for c in env.session.commands)
    assert any("grading_invalid.txt" in c and "touch /run/hb/grade/copy.log" in c for c in env.session.commands)
    mode = events(tmp_path, "grader_mode")[-1]
    assert "nested git" in mode["result"] and mode["reward_json"] is False


def test_drop_bytecode_keeps_source(tmp_path):
    repo = tmp_path / "repo"
    (repo / "pkg/__pycache__").mkdir(parents=True)
    (repo / "pkg/mod.py").write_text("x = 1\n")
    (repo / "pkg/__pycache__/mod.cpython-312.pyc").write_bytes(b"planted")
    (repo / "pkg/stray.pyc").write_bytes(b"planted")
    subprocess.run(["bash", "-c", ace.DROP_BYTECODE.replace("/work/repo", str(repo))], check=True)
    assert sorted(p.relative_to(repo).as_posix() for p in repo.rglob("*")) == ["pkg", "pkg/mod.py"]


def test_grading_stages_print_while_they_run(tmp_path, fake_aws):
    env = grading_env(tmp_path, fake_aws, list(STAGED))
    asyncio.run(env.exec(HARBOR_TEST_CMD))
    stages = [c for c in env.session.commands if "--stage untrusted" in c or "--stage trusted" in c]
    assert len(stages) == 2 and all(c.startswith(ace.KEEPALIVE) and "kill $ka" in c for c in stages)


def test_only_the_agent_session_name_counts_as_the_agent_environment():
    assert not ace.is_verifier_session("strands-agents__evals__pr371__abc1234__env")
    assert ace.is_verifier_session("strands-agents__evals__pr371__abc1234__verifier__main")
    assert ace.is_verifier_session("x__verifier__y__a1b2c3d4")  # Harbor shortens long verifier names with a hash


def test_events_keep_the_session_id_and_mask_account_ids(tmp_path, fake_aws):
    env = make_env(tmp_path, PR371 / "environment", "t__env", BASE, [BASE])
    env._event(event="session_start", session_id="hb-123456789012abcdef", note="arn:aws:iam::123456789012:role/x")
    e = events(tmp_path, "session_start")[-1]
    assert e["session_id"] == "hb-123456789012abcdef" and "123456789012" not in e["note"]


def test_local_junk_is_not_part_of_the_image(tmp_path):
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__/x.pyc").write_bytes(b"x")
    (tmp_path / ".DS_Store").write_bytes(b"x")
    (tmp_path / "Dockerfile").write_text("FROM debian\n")
    assert [p.name for p in ace.build_files(tmp_path)] == ["Dockerfile"]
