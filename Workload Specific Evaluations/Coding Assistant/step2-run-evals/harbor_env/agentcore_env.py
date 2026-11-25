"""A Harbor environment that runs each trial in an AgentCore Runtime session behind an in-session firewall.

    harbor run -p TASK_DIR -e harbor_env.agentcore_env:AgentCoreEnvironment -a codex ...

See README.md ("What the environment does", "Known limits").
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import io
import json
import os
import re
import shlex
import sys
import tarfile
import threading
import time
from pathlib import Path, PurePosixPath
from typing import Any, override

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.models.task.config import NetworkMode, NetworkPolicy

AGENTCORE_DIR = Path(__file__).resolve().parents[1]
if str(AGENTCORE_DIR) not in sys.path:
    sys.path.insert(0, str(AGENTCORE_DIR))
import agentcore_aws as aws  # noqa: E402
import boto3  # noqa: E402
from botocore.config import Config  # noqa: E402

# AgentCore ends a silent command after 15 min, so 16 min without a byte means the session is dead.
aws.dp = boto3.client("bedrock-agentcore", region_name=aws.R, config=aws.CFG.merge(Config(read_timeout=960)))

LAYER_DIR = Path(__file__).resolve().parent / "image"
FIREWALL_FILES = ("main.py", "start.sh", "proxy.py", "as-agent", "buildspec.yml")
DEFAULT_PREFIX = "evals-hb-"
MAX_COMMAND_SEC = 3600
DOWNLOAD_ATTEMPTS = 3
PLATFORM = "V2"  # AgentCore platform version: images up to 10 GB compressed, image prepared at deploy time
MAX_IMAGE_GB = 10
GIB = 1024 ** 3
CREDENTIAL_VARS = frozenset({
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_SECURITY_TOKEN", "AWS_PROFILE", "AWS_DEFAULT_PROFILE",
    "AWS_SHARED_CREDENTIALS_FILE", "AWS_CONFIG_FILE", "AWS_CONTAINER_CREDENTIALS_FULL_URI", "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_CONTAINER_AUTHORIZATION_TOKEN", "AWS_WEB_IDENTITY_TOKEN_FILE", "AWS_ROLE_ARN", "AWS_ROLE_SESSION_NAME"})
SAFE_PATH = re.compile(r"^/[A-Za-z0-9_./@+=:,-]*$")
STAGE_DIR = "/run/hb/stage"  # root-only
GRADE_DIR = "/run/hb/grade"
UNTRUSTED_DIR = "/run/hb/untrusted"
STAGE_OUT = f"{UNTRUSTED_DIR}/out"
BASE_GIT = "/opt/hidden/base.git"  # root-only
COPY_MAX_BYTES = 20_000_000
# Only the trusted stage writes these; an untrusted-stage file with such a name is never copied.
TRUSTED_NAMES = ("reward*", "rubric.json", "taste.json", "bloat.json", "rubric.stderr", "taste.stderr",
                 "rubric_judge_transcript.jsonl", "taste_judge_transcript.jsonl", "failed_tests.txt", "grading_invalid.txt",
                 "grader_copy.log", "test-stdout-trusted.txt")
# Harbor 0.26 runs the verifier as `(/tests/test.sh) > /logs/verifier/test-stdout.txt 2>&1`.
_HARBOR_TEST_CMD = re.compile(r"^\s*(?P<p>\()?\s*(?:/tests/test\.sh|'/tests/test\.sh')\s*(?(p)\))"
                              r"(?:\s*>\s*(?P<out>'[^']*'|[^\s']+)\s+2>&1)?\s*$")
_RUNS_TEST_SH = re.compile(r"/tests/test\.sh\b")
BASELINE_REF = "refs/hb/baseline"
# Compiled caches could run code that the patch does not show; root drops them before grading (Python recompiles from source).
DROP_BYTECODE = ("find -P /work/repo -type d -name __pycache__ -prune -exec rm -rf {} + && "
                 "find -P /work/repo -type f -name '*.pyc' -exec rm -f {} +")
# AgentCore ends a command that prints nothing for 15 min; a grading stage can run longer.
KEEPALIVE = "( while sleep 60; do echo '[agentcore_env] grading'; done ) & ka=$!; "
CACHE_PATHSPECS = ":(exclude,glob)**/__pycache__/**", ":(exclude,glob)**/*.pyc", ":(exclude,glob)**/.pytest_cache/**"


def root_git(repo: str = "/work/repo", base_git: str = BASE_GIT, index: str = f"{GRADE_DIR}/index") -> str:
    """git as root over the agent's tree, using none of the agent's git config, hooks, or index."""
    return (f"GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null HOME=/nonexistent GIT_INDEX_FILE={index} "
            f"git --no-replace-objects --git-dir={base_git} --work-tree={repo} -c core.hooksPath=/dev/null -c core.fsmonitor=false")


ROOT_GIT = root_git()
AS_AGENT_FILES = "setpriv --reuid=agent --regid=agent --init-groups --no-new-privs"
KERNEL_PROTECTION_CMD = ("sysctl -qw fs.protected_hardlinks=1 fs.protected_symlinks=1 2>&1; "
                         "echo \"protected=$(cat /proc/sys/fs/protected_hardlinks) $(cat /proc/sys/fs/protected_symlinks)\"")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_build_locks: dict[str, threading.Lock] = {}
_build_locks_guard = threading.Lock()


def is_credential_var(name: str) -> bool:
    return name in CREDENTIAL_VARS or name.startswith("AWS_BEARER_TOKEN")


def allow_hosts(policy: NetworkPolicy) -> list[str]:
    if policy.network_mode == NetworkMode.ALLOWLIST:
        return [h.lower().rstrip(".") for h in policy.allowed_hosts]
    return []


LOCAL_JUNK = ("__pycache__", ".DS_Store")


def build_files(root: Path):
    return sorted(p for p in Path(root).rglob("*") if p.is_file() and not set(p.relative_to(root).parts) & set(LOCAL_JUNK))


def is_verifier_session(session_id: str) -> bool:
    """Harbor names the agent environment <trial>__env; anything else is treated as a verifier (no agent runs there)."""
    return not session_id.endswith("__env")


def image_key(environment_dir: Path, name: str, verifier: bool) -> str:
    h = hashlib.sha256()
    for root in (Path(environment_dir), LAYER_DIR, aws.IMG):
        for f in build_files(root):
            if root == aws.IMG and f.name not in FIREWALL_FILES:
                continue
            h.update(str(f.relative_to(root)).encode() + b"\0" + f.read_bytes() + b"\0")
    short = re.sub(r"[^A-Za-z0-9]", "", name)[:20] or "env"
    return f"{short}_{'v' if verifier else 'a'}{h.hexdigest()[:10]}"


def task_toml(environment_dir: Path) -> dict:
    import tomllib
    try:
        return tomllib.loads((Path(environment_dir).parent / "task.toml").read_text())
    except (OSError, ValueError):
        return {}


def task_command_cap(cfg: dict) -> int:
    try:
        secs = max(float(cfg.get("agent", {}).get("timeout_sec") or 0), float(cfg.get("verifier", {}).get("timeout_sec") or 0))
    except (TypeError, ValueError):
        secs = 0
    return int(min(secs + 60, MAX_COMMAND_SEC)) if secs else MAX_COMMAND_SEC


def check_task_supported(cfg: dict) -> None:
    if cfg.get("steps"):
        raise ValueError("multi-step tasks ([[steps]]) are not supported by AgentCoreEnvironment: after a shared verifier, the next "
                         "step's agent would run with the grader credentials and an agent-writable /logs/verifier.")
    for section in ("agent", "verifier"):
        if str((cfg.get(section) or {}).get("user")) in ("root", "0"):
            raise ValueError(f"[{section}] user = root is not supported by AgentCoreEnvironment: root runs outside the firewall "
                             "(no proxy, the metadata service, open internet). Leave the user unset; commands then run as uid agent.")


def falsy(value: Any) -> bool:
    return str(value).strip().lower() in ("0", "false", "no", "off")


def harbor_test_stdout(command: str) -> str | None:
    """The stdout file of Harbor's verifier command, or None if the command is anything else."""
    m = _HARBOR_TEST_CMD.match(command)
    if not m:
        return None
    out = shlex.split(m.group("out"))[0] if m.group("out") else "/logs/verifier/test-stdout.txt"
    p = PurePosixPath(out)
    ok = SAFE_PATH.match(out) and ".." not in p.parts and PurePosixPath("/logs/verifier") in p.parents \
        and not p.name.startswith("reward")
    return str(p) if ok else None


def real_dir_guard(path: str, owner: str | None = None) -> str:
    """Shell: exit 1 unless path is a real directory (no symlink anywhere on it), owned by owner if given."""
    q = shlex.quote(path)
    own = f" && [ -n \"$(find {q} -maxdepth 0 -user {shlex.quote(owner)} 2>/dev/null)\" ]" if owner else ""
    return (f"{{ [ ! -L {q} ] && [ -d {q} ] && [ \"$(cd -P {q} && pwd -P)\" = {q} ]{own} || "
            f"{{ echo 'refusing {path}: a symlink, not a directory, not its own real path, or not owned by {owner or 'anyone'}'; "
            f"exit 1; }}; }}")


def root_patch_cmd(dest: str, repo: str = "/work/repo", base_git: str = BASE_GIT, index: str = f"{GRADE_DIR}/index",
                   baseline: str = BASELINE_REF) -> str:
    """Shell (root): agent.patch and agent.stat against the image's baseline, ignoring .gitignore; prints sha=<sha256>."""
    g = root_git(repo, base_git, index)
    excludes = " ".join(shlex.quote(p) for p in CACHE_PATHSPECS)
    return (f"{real_dir_guard(repo)} && cd {repo} && rm -f {index} && {g} read-tree {baseline} && {g} add -A --force -- . {excludes} && "
            f"{g} diff --cached {baseline} > {dest}/agent.patch && {g} diff --cached --stat {baseline} > {dest}/agent.stat && "
            f"echo \"sha=$(sha256sum < {dest}/agent.patch | cut -c1-64)\"")


def copy_regular_files_cmd(src: str, dest: str, max_bytes: int = COPY_MAX_BYTES, skip_names: tuple[str, ...] = TRUSTED_NAMES) -> str:
    """Shell: copy regular files only (no symlinks, no files over max_bytes, none named in skip_names); skips go to copy.log."""
    s, d = shlex.quote(src), shlex.quote(dest)
    log = shlex.quote(str(PurePosixPath(dest).parent / "copy.log"))
    named = " -o ".join(f"-name {shlex.quote(n)}" for n in skip_names) or "-false"
    return (f"cd -P {s} && {{ find -P . -type l | sed 's/^/skipped symlink: /'; find -P . -type f -size +{max_bytes - 1}c | "
            f"sed 's/^/skipped size: /'; find -P . -type f \\( {named} \\) | sed 's/^/skipped name: /'; }} > {log} && "
            f"find -P . -type f -size -{max_bytes}c ! \\( {named} \\) -print0 | while IFS= read -r -d '' f; do "
            f"mkdir -p {d}/\"$(dirname \"$f\")\" && cp -- \"$f\" {d}/\"$f\" || exit 1; done && echo \"copied=$(find {d} -type f | wc -l | tr -d ' ')\"")


def final_copy_cmd(grade: str = GRADE_DIR, dest: str = "/logs/verifier", graded: bool = True) -> str:
    """Shell (root): rebuild dest so that only the trusted stage's reward.json can become Harbor's reward."""
    g, d = shlex.quote(grade), shlex.quote(dest)
    cmd = (f"rm -rf {d} && install -d -m 0755 {d} && install -d -m 0755 {d}/untrusted && cp -rP {g}/in/. {d}/untrusted/ && "
           f"cp {g}/copy.log {d}/grader_copy.log && rm -f {d}/reward.json {d}/reward.txt")
    if graded:
        cmd += f" && cp -rP {g}/out/. {d}/ && rm -f {d}/reward.txt"
    return cmd + f" && echo \"reward_files=$(cd {d} && ls -A | grep -E '^reward\\.(json|txt)$' | tr '\\n' ' ')\""


def env_dir_context(environment_dir: Path):
    def ctx(_tag, _module, dest: Path):
        import shutil
        for p in Path(environment_dir).iterdir():
            if p.name == "Dockerfile" or p.name in LOCAL_JUNK:
                continue
            if p.is_dir():
                shutil.copytree(p, dest / p.name, ignore=shutil.ignore_patterns(*LOCAL_JUNK))
            else:
                shutil.copy2(p, dest / p.name)
        for n in FIREWALL_FILES:
            shutil.copy2(aws.IMG / n, dest / n)
        for n in ("as-agent-open", "as-grader", "record-baseline.sh"):
            shutil.copy2(LAYER_DIR / n, dest / n)
        (dest / "Dockerfile").write_text((Path(environment_dir) / "Dockerfile").read_text() + (LAYER_DIR / "layer.Dockerfile").read_text())
    return ctx


def image_memory_mib(size_bytes: int) -> int | None:
    """minimumMemoryInMiB for a compressed image size (AgentCore's build disk holds the image twice), None if not needed."""
    if size_bytes > MAX_IMAGE_GB * GIB:
        raise ValueError(f"image is {size_bytes / GIB:.1f} GB compressed, over AgentCore's {MAX_IMAGE_GB} GB limit")
    if size_bytes <= 2 * GIB:
        return None
    return 4096 if size_bytes <= 4 * GIB else 8192


def runtime_settings(runtime_id: str) -> tuple[str | None, int | None]:
    g = aws.ctl.get_agent_runtime(agentRuntimeId=runtime_id)
    mem = [c.get("minimumMemoryInMiB") for c in g.get("resourceConfigurations") or []]
    return g.get("platformVersion"), (mem[0] if mem else None)


def ensure_runtime(environment_dir: Path, tag: str, log=print) -> dict:
    """Build the image if missing, and deploy the runtime if missing, not READY, or on other settings."""
    with _build_locks_guard:
        lock = _build_locks.setdefault(tag, threading.Lock())
    with lock:
        out = {"tag": tag, "runtime": aws.runtime_name(tag), "built": False, "deployed": False}
        t0 = time.time()
        if not aws.image_exists(tag):
            log(f"agentcore: building image {tag} on CodeBuild")
            b = aws.build(tag, env_dir_context(environment_dir))
            out.update(built=True, build_secs=b["secs"], build_status=b["status"])
            if b["status"] != "SUCCEEDED":
                raise RuntimeError(f"image build failed for {tag}: {b['reason']}\n{b['log_tail'][-3000:]}")
        size = aws.ecr.describe_images(repositoryName=aws.Names.repo(), imageIds=[{"imageTag": tag}])["imageDetails"][0]["imageSizeInBytes"]
        memory = image_memory_mib(size)
        out.update(image_mb=round(size / 2 ** 20), memory_mib=memory)
        r = aws.find_runtime(tag)
        if not r or r["status"] != "READY" or runtime_settings(r["agentRuntimeId"]) != (PLATFORM, memory):
            log(f"agentcore: deploying runtime {aws.runtime_name(tag)} (takes several minutes)")
            d = aws.deploy(tag, PLATFORM, memory)
            out.update(deployed=True, deploy_secs=d["secs"], imdsv2_required=d.get("imdsv2_required"))
            if d["status"] != "READY":
                raise RuntimeError(f"runtime {aws.runtime_name(tag)} not READY: {d['status']} {d['reason']}")
            if not d.get("imdsv2_required"):
                raise RuntimeError(f"runtime {aws.runtime_name(tag)} does not require IMDSv2")
        out["secs"] = round(time.time() - t0, 1)
        return out


class AgentCoreEnvironment(BaseEnvironment):

    def __init__(self, *args: Any, prefix: str | None = None, staged_grader: bool | str = True, **kwargs: Any) -> None:
        aws.Names.prefix = prefix or os.environ.get("P2_PREFIX") or DEFAULT_PREFIX
        # operator switch only (--ek), never read from the task
        self._staged_grader = not falsy(staged_grader)
        self._session: aws.Session | None = None
        self._phase = "none"
        self._tag: str | None = None
        super().__init__(*args, **kwargs)
        self._is_verifier_env = is_verifier_session(self.session_id)
        phases = list(self._phase_network_policies)
        self._agent_phase_policy = None if self._is_verifier_env else (phases[0] if phases else self._network_policy)
        self._verifier_phase_policy = phases[1] if not self._is_verifier_env and len(phases) > 1 else None
        self._events_path = self.trial_paths.trial_dir / "agentcore.jsonl"
        cfg = task_toml(Path(self.environment_dir))
        check_task_supported(cfg)
        self._max_command_sec = task_command_cap(cfg)
        if not self._staged_grader:
            self._event(event="staged_grader_off", note="every shared-verifier grading of this trial is unsealed")

    @staticmethod
    @override
    def type() -> str:
        return "agentcore"

    @property
    @override
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(disable_internet=True, network_allowlist=True, network_allowlist_hostnames=True,
                                       dynamic_network_policy=True)

    @override
    def _validate_definition(self):
        if not (Path(self.environment_dir) / "Dockerfile").is_file():
            raise FileNotFoundError(f"{self.environment_dir}/Dockerfile not found (AgentCoreEnvironment builds images from a Dockerfile)")
        if self.task_env_config.docker_image:
            raise ValueError("AgentCoreEnvironment builds from environment/Dockerfile; a prebuilt docker_image is not supported")

    @override
    def validate_network_policy_support(self, network_policy: NetworkPolicy | None = None) -> None:
        network_policy = network_policy or self._network_policy
        if network_policy.network_mode == NetworkMode.PUBLIC and not getattr(self, "_is_verifier_env", is_verifier_session(self.session_id)):
            raise ValueError("network_mode='public' is not supported for an agent environment by AgentCoreEnvironment: outside the "
                             "firewall the agent could read the runtime role from the metadata service.")
        super().validate_network_policy_support(network_policy)

    # ------------------------------------------------------------------ bookkeeping
    def _event(self, **e: Any) -> None:
        e = {"t": round(time.time(), 3), "env": self.session_id, "role": "verifier" if self._is_verifier_env else "agent", **e}
        try:
            self._events_path.parent.mkdir(parents=True, exist_ok=True)
            with self._events_path.open("a") as f:
                line = json.loads(aws.mask(json.dumps(e)))
                if "session_id" in e:  # a session ID can hold 12 digits in a row; stop-sessions needs it unmasked
                    line["session_id"] = e["session_id"]
                f.write(json.dumps(line) + "\n")
        except OSError:
            pass

    @property
    def session(self) -> aws.Session:
        if self._session is None:
            raise RuntimeError("AgentCoreEnvironment is not started")
        return self._session

    async def _cmd(self, command: str, timeout: int = 300) -> tuple[str, int]:
        """One command, retried only on throttling, so it never runs twice."""
        t0, calls = time.time(), self.session.calls
        for i in range(6):
            try:
                out, code, secs = await asyncio.to_thread(self.session.cmd, command, timeout, 1)
                break
            except Exception as e:
                if i == 5 or not any(k in type(e).__name__ + str(e) for k in ("Throttl", "TooManyRequests", "ServiceQuota")):
                    raise
                self.session.throttles += 1
                await asyncio.sleep(2 + 3 * i)
        if time.time() - t0 > 60:
            self._event(event="slow_call", phase=self._phase, wall_secs=round(time.time() - t0, 1), last_call_secs=secs,
                        api_calls=self.session.calls - calls, code=code, command=command[-160:])
        return out, (124 if code is None else code)

    # ------------------------------------------------------------------ lifecycle
    @override
    async def start(self, force_build: bool) -> None:
        t0 = time.time()
        self._tag = image_key(Path(self.environment_dir), self.environment_name, self._is_verifier_env)
        info = await asyncio.to_thread(ensure_runtime, Path(self.environment_dir), self._tag, self.logger.info)
        self._event(event="runtime", **info)
        t1 = time.time()
        self._session = aws.Session(self._tag)
        out, code, secs = await asyncio.to_thread(self._session.start, "cat /run/fw.status; id -u; date +%s.%N")
        if not out.startswith("OK"):
            raise RuntimeError(f"AgentCore session {self._session.sid}: firewall not ready: {out[:300]}")
        hardlinks, symlinks = await self._kernel_protection()
        self._event(event="session_start", session_id=self._session.sid, runtime=aws.runtime_name(self._tag),
                    start_attempts=self._session.start_attempts, first_call_secs=secs, ensure_runtime_secs=round(t1 - t0, 1),
                    protected_hardlinks=hardlinks, protected_symlinks=symlinks)
        await self._enter_phase("verifier" if self._is_verifier_env else "agent", self._network_policy)
        self._event(event="started", secs=round(time.time() - t0, 1))

    async def _kernel_protection(self) -> tuple[str, str]:
        """Require fs.protected_hardlinks and fs.protected_symlinks = 1, or refuse the trial."""
        out, _ = await self._cmd(KERNEL_PROTECTION_CMD)
        line = next((ln for ln in out.splitlines() if ln.startswith("protected=")), "protected=")
        values = line.removeprefix("protected=").split()
        if values != ["1", "1"]:
            self._event(event="kernel_protection_failed", out=out[-600:])
            raise RuntimeError(f"AgentCore session {self.session.sid}: fs.protected_hardlinks/protected_symlinks are not 1 1 "
                               f"({line!r}), not running the trial: {out[-300:]}")
        return values[0], values[1]

    @override
    async def stop(self, delete: bool):
        if self._session is None:
            return
        err = await asyncio.to_thread(self._session.stop)
        self._event(event="stop", session_id=self._session.sid, calls=self._session.calls, throttles=self._session.throttles,
                    error=err)
        self._session = None

    # ------------------------------------------------------------------ network policy and phases
    async def _kill_agent_processes(self, user: str = "agent") -> str:
        """SIGKILL every process of user until none is alive; fail closed (no grade) if one survives."""
        out, _ = await self._cmd(
            f"for i in $(seq 1 20); do pkill -KILL -u {user}; sleep 0.5; "
            f"n=$(ps -u {user} -o stat= | grep -vc '^Z'); [ \"$n\" = 0 ] && break; done; "
            f"echo \"alive=$n tries=$i\"; ps -u {user} -o pid=,stat=,comm= | head -5")
        first = out.strip().splitlines()[0] if out.strip() else "no output"
        if not first.startswith("alive=0 "):
            self._event(event="agent_processes_survived", user=user, out=out[-600:])
            raise RuntimeError(f"{user} processes still alive after SIGKILL, not grading: {out[-300:]}")
        return first

    async def _enter_phase(self, phase: str, policy: NetworkPolicy) -> None:
        t0 = time.time()
        killed = None
        extra = {"AWS_REGION": aws.R, "AWS_DEFAULT_REGION": aws.R}
        if phase == "agent":
            extra.update(await asyncio.to_thread(aws.agent_creds))
            who = aws.Names.agent_role()
        else:
            if self._phase == "agent":
                killed = await self._kill_agent_processes()
                await self._cmd("rm -rf /logs/verifier && mkdir -p /logs/verifier && chown agent:agent /logs/verifier")
            extra.update(await asyncio.to_thread(aws.runtime_role_creds, self.session))
            who = aws.Names.runtime_role()
        hosts = allow_hosts(policy)
        await asyncio.to_thread(aws.set_phase, self.session, "\n".join(hosts), extra)
        self._phase = phase
        self._event(event="phase", phase=phase, network_mode=policy.network_mode.value, allow=hosts, credentials=who,
                    agent_procs_left_after_kill=killed, secs=round(time.time() - t0, 2))

    @override
    async def _apply_network_policy(self, network_policy: NetworkPolicy) -> None:
        if self._is_verifier_env:
            await self._enter_phase("verifier", network_policy)
        elif self._verifier_phase_policy is not None and network_policy == self._verifier_phase_policy \
                and network_policy != self._agent_phase_policy:
            await self._enter_phase("verifier", network_policy)
        elif self._phase == "verifier":
            await self._enter_phase("verifier", network_policy)
        else:
            await self._enter_phase("agent", network_policy)

    # ------------------------------------------------------------------ exec
    def _script(self, command: str, cwd: str | None, env: dict[str, str] | None) -> str:
        lines = [f"export {k}={shlex.quote(str(v))}" for k, v in (env or {}).items() if _ENV_NAME.match(k) and not is_credential_var(k)]
        if cwd:
            lines.append(f"cd {shlex.quote(cwd)} || exit 1")
        return "\n".join(lines + [command])

    def _runs_as_root(self, user: str | int | None) -> bool:
        return user in ("root", 0, "0")

    @override
    async def exec(self, command: str, cwd: str | None = None, env: dict[str, str] | None = None, timeout_sec: int | None = None,
                   user: str | int | None = None) -> ExecResult:
        resolved = self._resolve_user(user)
        if self._runs_as_root(resolved) and not self._runs_as_root(user):
            raise ValueError(f"exec as root through default_user={self.default_user!r} is refused by AgentCoreEnvironment: "
                             "the task's agent or verifier commands would run outside the firewall")
        user = resolved
        if self._phase == "verifier" and not self._runs_as_root(user) and _RUNS_TEST_SH.search(command):
            if self._is_verifier_env:
                self._event(event="grader_mode", mode="separate", command=command[:200])
            else:
                staged = await self._grade_shared(command, cwd, env)
                if staged is not None:
                    return staged
        script = self._script(command, cwd, self._merge_env(env))
        timeout = int(min(timeout_sec or self._max_command_sec, self._max_command_sec))
        if self._runs_as_root(user):
            out, code = await self._cmd(script, timeout)
        else:
            runner = "/opt/firewall/as-agent-open" if self._is_verifier_env and self._network_policy.network_mode == NetworkMode.PUBLIC \
                else "/opt/firewall/as-agent"
            run = f"{runner} {shlex.quote(script)}"
            if self._phase == "verifier":
                run = f"{run} </dev/null & p=$!; wait $p; rc=$?; pkill -KILL -u agent; exit $rc"
            out, code = await self._cmd(run, timeout)
            if self._phase == "verifier":
                await self._kill_agent_processes()
        if code != 0:
            self._event(event="exec_failed", user=str(user), phase=self._phase, code=code, command=command[:200], out_tail=out[-600:])
        return ExecResult(stdout=out, stderr="", return_code=code)

    # ------------------------------------------------------------------ staged grading (shared verifier, test.sh --stage)
    async def _grade_shared(self, command: str, cwd: str | None, env: dict[str, str] | None) -> ExecResult | None:
        """The staged grade, or None to run the command unsealed (logged)."""
        if not self._staged_grader:
            self._event(event="grader_mode", mode="unsealed", reason="staged_grader=false", command=command[:200])
            return None
        reason = await self._unstaged_reason()
        if reason is not None:
            self._event(event="grader_mode", mode="unsealed", reason=reason, command=command[:200])
            self._event(event="unsealed_grader", reason=reason, note="graded in one command as uid agent (reward can be hacked)")
            return None
        stdout_path = harbor_test_stdout(command)
        if stdout_path is None:
            why = "this test.sh supports --stage, but the command is not Harbor's verifier command, so it would bypass staged grading"
            self._event(event="grader_mode", mode="refused", reason=why, command=command[:300])
            self._event(event="grader_refused", reason=why, command=command[:300])
            raise RuntimeError(f"not grading: {why}: {command[:200]!r}")
        try:
            return await self._staged_grade(cwd, self._merge_env(env), stdout_path)
        except Exception as e:
            self._event(event="grader_mode", mode="staged", result=f"error: {type(e).__name__}: {str(e)[:300]}")
            raise

    async def _unstaged_reason(self) -> str | None:
        out, _ = await self._cmd(f"grep -q -- '--stage' /tests/test.sh || {{ echo 'no=test.sh has no --stage'; exit 0; }}; "
                                 f"test -d {BASE_GIT} && id -u grader >/dev/null 2>&1 && test -x /opt/firewall/as-grader "
                                 f"&& {ROOT_GIT} rev-parse -q --verify {BASELINE_REF} >/dev/null "
                                 f"&& echo yes || echo 'no=image without grader, as-grader, {BASE_GIT} or {BASELINE_REF}'")
        last = out.strip().splitlines()[-1] if out.strip() else "no=no output"
        return None if last == "yes" else last.removeprefix("no=")

    async def _root(self, what: str, command: str, timeout: int = 600) -> str:
        out, code = await self._cmd(command, timeout)
        if code != 0:
            self._event(event="grader_step_failed", step=what, code=code, out=out[-600:])
            raise RuntimeError(f"staged grading: {what} failed ({code}), not grading: {out[-300:]}")
        return out

    @staticmethod
    def _marker(out: str, key: str) -> str:
        return next((ln.removeprefix(key + "=") for ln in reversed(out.splitlines()) if ln.startswith(key + "=")), "")

    async def _invalid(self, reason: str, out: str, info: dict, stdout_path: str, lock: str) -> ExecResult:
        """No reward: /logs/verifier gets the untrusted stage's files and grading_invalid.txt."""
        msg = shlex.quote(f"{reason}; not graded")
        done = await self._root("final copy", f"touch {GRADE_DIR}/copy.log && {final_copy_cmd(GRADE_DIR, graded=False)} && echo {msg} > /logs/verifier/grading_invalid.txt "
                                              f"&& {lock}")
        if self._marker(done, "reward_files").strip():
            raise RuntimeError(f"staged grading: a reward file is left in /logs/verifier of an invalid trial: {done[-300:]}")
        self._event(event="grader_mode", mode="staged", result=f"invalid: {reason}", stdout_path=stdout_path, reward_json=False, **info)
        return ExecResult(stdout=out, stderr="", return_code=1)

    async def _staged_grade(self, cwd: str | None, env: dict[str, str] | None,
                            stdout_path: str = "/logs/verifier/test-stdout.txt") -> ExecResult:
        """Grade a coding task in stages. Untrusted stage as agent, kill, root copies, trusted stage as grader.
        Not sealed: junit.xml and validation.json come from the untrusted stage, so planted code can forge them."""
        t0 = time.time()
        timeout = self._max_command_sec
        g, base = GRADE_DIR, f"{GRADE_DIR}/base"
        prep = await self._root("prepare and patch", (
            f"rm -rf {g} {UNTRUSTED_DIR} && mkdir -p /run/hb && chown root:root /run/hb && chmod 0711 /run/hb && "
            f"install -d -m 0755 {g} {g}/in {base} && install -d -m 0700 {g}/after && "
            f"install -d -m 0700 -o grader -g grader {g}/out && install -d -m 0711 {UNTRUSTED_DIR} && "
            f"install -d -m 0700 -o agent -g agent {STAGE_OUT} && "
            f"rm -rf /logs/verifier && install -d -m 0755 /logs/verifier && {real_dir_guard('/work/repo')} && "
            f"{DROP_BYTECODE} && echo \"nested=$(find -P /work/repo -mindepth 2 -name .git -print -quit)\" && "
            f"{{ (cd /work/repo && {ROOT_GIT} show \"$(tr -d '[:space:]' < /var/lib/task_base_ref):pyproject.toml\") "
            f"> {base}/pyproject.toml 2>/dev/null && chmod 0644 {base}/pyproject.toml || rm -f {base}/pyproject.toml; }} && "
            f"{root_patch_cmd(base)} && chmod 0644 {base}/agent.patch {base}/agent.stat"))
        before, nested = self._marker(prep, "sha"), self._marker(prep, "nested")
        if not before:
            raise RuntimeError("staged grading: root's patch has no sha, not grading")
        lock = "chown -R root:root /logs/verifier && chmod -R u=rwX,go=rX /logs/verifier"
        if nested:
            return await self._invalid(f"a nested git repository ({nested}) would hide code from the patch", "", {"nested_git": nested},
                                       stdout_path, lock)

        script = self._script(f"/tests/test.sh --stage untrusted > {STAGE_OUT}/test-stdout.txt 2>&1", cwd,
                              {**(env or {}), "OUT": STAGE_OUT, "BASE_PYTEST_CONFIG": f"{base}/pyproject.toml"})
        out1, code1 = await self._cmd(f"{KEEPALIVE}/opt/firewall/as-agent {shlex.quote(script)} </dev/null & p=$!; wait $p; rc=$?; "
                                      f"kill $ka 2>/dev/null; pkill -KILL -u agent; exit $rc", timeout)
        killed = await self._kill_agent_processes()

        after = self._marker(await self._root("patch after the untrusted stage", root_patch_cmd(f"{g}/after")), "sha")
        copied = await self._root("copy", f"{real_dir_guard(STAGE_OUT, 'agent')} && {copy_regular_files_cmd(STAGE_OUT, g + '/in')} && "
                                          f"cp {base}/agent.patch {base}/agent.stat {g}/in/ && "
                                          f"chown -R root:root {g}/in && chmod -R u=rwX,go=rX {g}/in")
        info = {"untrusted_code": code1, "kill": killed, "patch_sha": before, "patch_sha_after": after,
                "copied": self._marker(copied, "copied")}
        if after != before:
            return await self._invalid(f"the repository changed during the untrusted stage ({before} -> {after})", out1, info,
                                       stdout_path, lock)

        script = self._script(f"/tests/test.sh --stage trusted > {g}/out/test-stdout-trusted.txt 2>&1", cwd,
                              {**(env or {}), "IN": f"{g}/in", "OUT": f"{g}/out"})
        out2, code2 = await self._cmd(f"{KEEPALIVE}/opt/firewall/as-grader {shlex.quote(script)} </dev/null & p=$!; wait $p; rc=$?; "
                                      f"kill $ka 2>/dev/null; pkill -KILL -u grader; exit $rc", timeout)
        killed_grader = await self._kill_agent_processes("grader")
        modes = await self._root("final copy", (
            f"{final_copy_cmd(g)} && "
            f"mkdir -p {shlex.quote(str(PurePosixPath(stdout_path).parent))} && "
            f"cat {g}/in/test-stdout.txt {g}/out/test-stdout-trusted.txt > {shlex.quote(stdout_path)} 2>/dev/null; "
            f"{lock} && stat -c 'mode %n %U %a' /logs/verifier {g}/out {g}/in"))
        reward_files = self._marker(modes, "reward_files").split()
        if reward_files not in ([], ["reward.json"]):
            raise RuntimeError(f"staged grading: unexpected reward files {reward_files} in /logs/verifier, not grading")
        self._event(event="grader_mode", mode="staged",
                    result="graded" if reward_files else "invalid: no reward.json from the trusted stage", trusted_code=code2,
                    kill_grader=killed_grader, modes=[ln.removeprefix("mode ") for ln in modes.splitlines() if ln.startswith("mode ")],
                    reward_json=bool(reward_files), stdout_path=stdout_path, secs=round(time.time() - t0, 1), **info)
        return ExecResult(stdout=out1 + out2, stderr="", return_code=0)

    # ------------------------------------------------------------------ files
    def _owner(self, target: str) -> str:
        return "root" if self._under_tests(target) else "agent"

    @staticmethod
    def _under_tests(target: str) -> bool:
        p = PurePosixPath(target)
        return p == PurePosixPath("/tests") or PurePosixPath("/tests") in p.parents

    async def _before_tests_upload(self, target: str) -> None:
        """The first /tests upload ends the agent phase, even if Harbor sends no policy change."""
        if self._under_tests(target) and not self._is_verifier_env and self._phase == "agent":
            await self._enter_phase("verifier", self._verifier_phase_policy or self._network_policy)

    @staticmethod
    def _safe_target(path: str) -> str:
        if not SAFE_PATH.match(path):
            raise ValueError(f"unsupported path for upload: {path!r}")
        return str(PurePosixPath(path))

    async def _stage(self, data: bytes, suffix: str = "") -> str:
        path = f"{STAGE_DIR}/up-{os.urandom(6).hex()}{suffix}"
        await self._cmd(f"mkdir -p {STAGE_DIR} && chown root:root /run/hb {STAGE_DIR} && chmod 0711 /run/hb && "
                        f"chmod 0700 {STAGE_DIR}")
        await asyncio.to_thread(self.session.put, path, data, "0600", "root")
        return path

    @staticmethod
    def _agent_dir(d: str) -> str:
        """Shell: make d writable by agent without root following a symlink."""
        q = shlex.quote(d)
        return (f"if [ -L {q} ]; then echo 'refusing symlink {d}'; exit 1; fi; "
                f"if [ ! -e {q} ]; then mkdir -p {q} || exit 1; "
                f"if [ -L {q} ] || [ ! -d {q} ] || [ \"$(realpath -e {q})\" != {q} ]; then echo 'refusing {d}'; exit 1; fi; "
                f"chown -h agent:agent {q} || exit 1; fi; [ -d {q} ] || {{ echo 'not a directory: {d}'; exit 1; }}")

    async def _run_upload(self, what: str, command: str) -> None:
        out, code = await self._cmd(command)
        if code != 0 or not out.strip().endswith("ok"):
            raise RuntimeError(f"{what} failed ({code}): {out[-500:]}")

    @override
    async def upload_file(self, source_path: Path | str, target_path: str):
        src = Path(source_path)
        target = self._safe_target(target_path)
        await self._before_tests_upload(target)
        mode = "0755" if os.access(src, os.X_OK) else "0644"
        q = shlex.quote(target)
        if self._owner(target) == "root":
            await self._cmd(f"mkdir -p {shlex.quote(str(PurePosixPath(target).parent))}")
            await asyncio.to_thread(self.session.put, target, src.read_bytes(), mode, "root")
            return
        staged = await self._stage(src.read_bytes())
        await self._run_upload(f"upload_file {src} -> {target}", (
            f"trap 'rm -f {staged}' EXIT; {self._agent_dir(str(PurePosixPath(target).parent))}; "
            f"{AS_AGENT_FILES} sh -c 'cat > \"$1\" && chmod {mode} \"$1\"' sh {q} < {staged} && echo ok"))

    @override
    async def upload_dir(self, source_dir: Path | str, target_dir: str):
        target = self._safe_target(target_dir)
        await self._before_tests_upload(target)
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            tf.add(str(source_dir), arcname=".")
        tgz = await self._stage(buf.getvalue(), ".tgz")
        q = shlex.quote(target)
        if self._owner(target) == "root":
            command = (f"trap 'rm -f {tgz}' EXIT; mkdir -p {q} && tar xzf {tgz} -C {q} --no-same-owner && "
                       f"chown -R root:root {q} && chmod -R a+rX {q} && echo ok")
        else:  # extracted as agent, never root over agent-controlled paths
            command = (f"trap 'rm -f {tgz}' EXIT; {self._agent_dir(target)}; "
                       f"{AS_AGENT_FILES} tar xzf - -C {q} --no-same-owner < {tgz} && echo ok")
        await self._run_upload(f"upload_dir {source_dir} -> {target_dir}", command)

    async def _fetch(self, what: str, produce: str) -> bytes:
        """The stdout of `produce`, checked against its sha256: a command's output can arrive cut short."""
        tmp = f"{STAGE_DIR}/dl-{os.urandom(6).hex()}"
        command = (f"mkdir -p {STAGE_DIR} && chmod 0700 {STAGE_DIR} && ( {produce} ) > {tmp} && "
                   f"echo \"sha=$(sha256sum < {tmp} | cut -c1-64)\" && base64 -w0 < {tmp}; rc=$?; rm -f {tmp}; exit $rc")
        for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
            out, code = await self._cmd(command, 600)
            if code != 0:
                raise FileNotFoundError(f"{what} not found in session {self.session.sid}")
            head, _, body = out.partition("\n")
            try:
                data = base64.b64decode(body.strip())
            except binascii.Error:
                data = None
            if data is not None and head.strip() == f"sha={hashlib.sha256(data).hexdigest()}":
                return data
            self._event(event="download_retry", what=what, attempt=attempt, got_chars=len(body))
        raise RuntimeError(f"{what}: the download from session {self.session.sid} arrived cut short {DOWNLOAD_ATTEMPTS} times")

    @override
    async def download_file(self, source_path: str, target_path: Path | str):
        q = shlex.quote(source_path)
        data = await self._fetch(source_path, f"test -f {q} && cat {q}")
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        Path(target_path).write_bytes(data)

    @override
    async def download_dir(self, source_dir: str, target_dir: Path | str):
        q = shlex.quote(source_dir)
        data = await self._fetch(source_dir, f"cd {q} || exit 1; tar czf - . 2>/dev/null; exit 0")
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        skipped, note = aws.extract_session_tar(data, Path(target_dir))
        if skipped or note:
            self.logger.warning(f"download_dir {source_dir}: skipped {skipped[:5]} {note or ''}")
