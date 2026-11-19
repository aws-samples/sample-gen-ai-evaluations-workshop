"""Helpers that notebooks 02 and 03 share: task files, leakage scan, Docker images, seal check,
AWS credentials, Harbor runs, and Harbor view.

The notebooks explain each step. This file holds the code, so that the two notebooks do not each keep
a copy. Where the notebooks differ, the helper has a parameter: the seal check of notebook 03 also
looks for the hidden tests, notebook 03 runs fewer trials at the same time, and it can run a subset
of the tasks.

    import coding_eval as ce
    ce.build_images(TASK_DIRS)
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import tomllib
import uuid
from collections.abc import Iterator
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
JOBS_DIR = Path("jobs")
HARBOR_ENV = "harvest_env.docker_allowlist:ProxyAllowlistDockerEnvironment"
PROXY_IMAGE = "harvest-env-egress-proxy:1"  # the same tag as harvest_env/docker_allowlist.py
PROXY_CONTEXT = MODULE_DIR / "harvest_env" / "egress_proxy"
CODEX_CONFIG = MODULE_DIR / "codex-bedrock.toml"
CANARY = "CANARY-7f3a91"


# ── Files ────────────────────────────────────────────────────────────────────

def read_toml(path: Path) -> dict:
    return tomllib.loads(Path(path).read_text())


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def show_tree(root: Path) -> None:
    """Print the files of a task directory as a tree."""
    root = Path(root)
    print(root.name + "/")
    for path in sorted(root.rglob("*")):
        if "__pycache__" in path.parts:
            continue
        depth = len(path.relative_to(root).parts)
        print("    " * depth + path.name + ("/" if path.is_dir() else ""))


def check_shared_files() -> bool:
    """Report if a task has a copy of a shared verifier file that differs from datasets/_shared/."""
    spec = importlib.util.spec_from_file_location("_shared_sync", MODULE_DIR / "datasets" / "_shared" / "sync.py")
    sync = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sync)
    stale = sync.stale_copies()
    if stale:
        print("WARNING: these copies differ from datasets/_shared/. Run `python datasets/_shared/sync.py`:")
        for path in stale:
            print(f"  {path.relative_to(MODULE_DIR)}")
        return False
    print(f"Shared verifier files: {len(sync.pairs())} copies in sync with datasets/_shared/.")
    return True


# ── Region ───────────────────────────────────────────────────────────────────

_BEDROCK_HOST = re.compile(r"^bedrock(?:-runtime|-mantle)?\.([a-z0-9-]+)\.(?:api\.aws|amazonaws\.com)$")


def task_regions(task_dirs: list[Path]) -> dict[str, set[str]]:
    """The Regions of the Bedrock hosts in each task.toml (agent and verifier allowlists)."""
    regions: dict[str, set[str]] = {}
    for task_dir in task_dirs:
        cfg = read_toml(Path(task_dir) / "task.toml")
        hosts = list(cfg.get("environment", {}).get("allowed_hosts", []))
        hosts += cfg.get("verifier", {}).get("allowed_hosts", [])
        regions[Path(task_dir).name] = {m.group(1) for h in hosts if (m := _BEDROCK_HOST.match(h))}
    return regions


def check_region(region: str | None, task_dirs: list[Path], codex_config: Path = CODEX_CONFIG) -> str:
    """Make sure that AWS_REGION is the Region that the task allowlists and codex-bedrock.toml use.

    Each task.toml allows only the Bedrock endpoints of one Region. If AWS_REGION is a different
    Region (for example us-west-2 from your shell), the allowlist blocks Codex and the judges, and
    every trial fails. Return the region.
    """
    allowed = set().union(*task_regions(task_dirs).values())
    codex_region = read_toml(codex_config).get("model_providers", {}).get("amazon-bedrock", {}).get("aws", {}).get("region")
    expected = allowed | ({codex_region} if codex_region else set())
    if len(expected) != 1:
        raise RuntimeError(
            f"The task files do not agree on one Region: allowed_hosts use {sorted(allowed)}, "
            f"{codex_config.name} uses {codex_region}. Make them the same."
        )
    (want,) = expected
    if region != want:
        raise RuntimeError(
            f"AWS_REGION is {region!r}, but the tasks allow only the Bedrock endpoints in {want!r} "
            f"(allowed_hosts in each task.toml, and region in {codex_config.name}). The network allowlist "
            f"would block Codex and the judges, and every trial would fail.\n"
            f"Repair: set AWS_REGION={want} in .env, then run the first cells of this notebook again. "
            f"To use a different Region, change allowed_hosts in every task.toml and the region in "
            f"{codex_config.name}."
        )
    print(f"AWS_REGION {region} matches the task allowlists.")
    return region


# ── Leakage scan ─────────────────────────────────────────────────────────────

def agent_visible_files(task_dir: Path) -> Iterator[Path]:
    """Files that the agent sees: the instruction and the files that build its container."""
    task_dir = Path(task_dir)
    yield task_dir / "instruction.md"
    yield from (p for p in sorted((task_dir / "environment").rglob("*")) if p.is_file())


def _scan(task_dir: Path, needles: list[str]) -> list[tuple[str, str]]:
    hits = []
    for path in agent_visible_files(task_dir):
        text = path.read_text(errors="ignore").lower()
        for needle in needles:
            if needle and needle.lower() in text:
                hits.append((str(path.relative_to(task_dir)), needle))
    return hits


def canary_selftest(task_dir: Path, canary: str = CANARY) -> None:
    """Plant a canary in a temporary copy of the task, and make sure that the scan finds it.

    The copy is in a temporary directory, so the task in the repository is never changed, also if the
    kernel stops during the test.
    """
    with tempfile.TemporaryDirectory(prefix="leakage-canary-") as tmp:
        copy = Path(tmp) / Path(task_dir).name
        shutil.copytree(task_dir, copy)
        (copy / "environment" / "canary.txt").write_text(canary)
        if not _scan(copy, [canary]):
            raise AssertionError("The scan did not find the canary. The scan has a defect.")


def leakage_scan(task_dir: Path, needles: list[str], canary: str | None = CANARY) -> list[tuple[str, str]]:
    """Search the agent-visible files for each needle. Return (file, needle) for each hit.

    First, if `canary` is set, test the scan itself with a canary (see canary_selftest).
    """
    if canary:
        canary_selftest(task_dir, canary)
    return _scan(Path(task_dir), needles)


# ── Docker ───────────────────────────────────────────────────────────────────

def _docker(*args: str, timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=timeout)


def require_docker() -> None:
    ok = shutil.which("docker") and _docker("info").returncode == 0
    assert ok, "Docker is not installed or does not run. Do the setup checks in notebook 01."
    print("Docker runs.")


def image_tag(task_dir: Path) -> str:
    return f"eval-{Path(task_dir).name}"


def image_exists(tag: str) -> bool:
    return _docker("image", "inspect", tag).returncode == 0


def build_images(task_dirs: list[Path], force: bool = False) -> None:
    """Build the egress proxy image and one image for each task.

    An image that already exists is not built again, so you can run the cell again after an
    interruption. After you change a Dockerfile, use force=True.
    """
    jobs = [("egress proxy", PROXY_IMAGE, PROXY_CONTEXT)]
    jobs += [(Path(t).name, image_tag(t), Path(t) / "environment") for t in task_dirs]
    failed = []
    for name, tag, context in jobs:
        if not force and image_exists(tag):
            print(f"SKIP {name}  {tag} exists (force=True builds it again)")
            continue
        start = time.monotonic()
        r = _docker("build", "-q", "-t", tag, str(context))
        elapsed = f"{time.monotonic() - start:.0f} s"
        ok = r.returncode == 0
        print(f"{'OK  ' if ok else 'ERR '} {name}  {elapsed}  {r.stdout.strip() or r.stderr.strip()[-300:]}")
        if not ok:
            failed.append(name)
    if failed:
        raise RuntimeError(f"These images did not build: {', '.join(failed)}. See the error above.")


def seal_check(task_dirs: list[Path], expect_changes: bool, check_hidden_tests: bool = False) -> None:
    """Check the git seal in each task image.

    `git log --all` must show one commit, and `git remote` must show no remotes. With
    expect_changes=True (PR review), `git status` must show the change under review. With
    expect_changes=False (coding), the working tree must be clean. With check_hidden_tests=True, no
    `*_verify.py` file may be in the image.
    """
    cmd = 'echo "$(git log --all --oneline | wc -l) $(git remote | wc -l) $(git status --porcelain | wc -l)'
    if check_hidden_tests:
        cmd += " $(find / -name '*_verify.py' -not -path '/proc/*' 2>/dev/null | wc -l)"
    cmd += '"'
    for task_dir in task_dirs:
        r = _docker("run", "--rm", image_tag(task_dir), "sh", "-c", cmd)
        if r.returncode != 0:
            raise RuntimeError(f"{Path(task_dir).name}: the seal check did not run: {r.stderr.strip()[-300:]}")
        counts = list(map(int, r.stdout.split()))
        commits, remotes, changed = counts[:3]
        line = f"{Path(task_dir).name}: commits={commits} remotes={remotes} changed_files={changed}"
        print(line + (f" hidden_tests={counts[3]}" if check_hidden_tests else ""))
        assert commits == 1, "More than one commit is reachable. The agent can find later commits, for example the fix."
        assert remotes == 0, "The repository has a remote."
        if expect_changes:
            assert changed > 0, "change.patch is not applied."
        else:
            assert changed == 0, "The working tree is not the base commit."
        if check_hidden_tests:
            assert counts[3] == 0, "The hidden tests are in the image."
    print("Seal checks passed.")


def egress_check(task_dir: Path, phase: str = "agent", blocked_hosts: tuple[str, ...] = ("github.com", "pypi.org"),
                 timeout_sec: int = 10) -> dict[str, bool]:
    """OPTIONAL. Check the network seal at run time, not only in task.toml.

    This makes the same topology as harvest_env/docker_allowlist.py, without Harbor: an internal Docker
    network with no route off the host, and the egress proxy on that network and on a normal network.
    The proxy gets the allowlist of the phase ("agent": [environment].allowed_hosts, "verifier":
    [verifier].allowed_hosts). Then a container of the task image, only on the internal network,
    tries to reach:

    - each allowed host through the proxy: it must get an HTTP response (any status code).
    - each blocked host through the proxy: the proxy must refuse it.
    - each blocked host directly, without the proxy: there must be no route.

    It needs the task image (build_images) and the proxy image. It removes its containers and
    networks at the end. It uses curl in the task image. Return {check: passed}. Raise AssertionError
    if a check fails.

    Tested on Colima (Docker 29) with all tasks, in the agent and the verifier phases. It rebuilds the
    topology of harvest_env; it does not run Harbor. Treat a failure as a reason to look closer, not as
    proof that the seal of a Harbor trial is broken.
    """
    cfg = read_toml(Path(task_dir) / "task.toml")
    section = cfg["environment"] if phase == "agent" else cfg["verifier"]
    allowed = list(section.get("allowed_hosts", []))
    concrete = [h for h in allowed if "*" not in h]  # a wildcard entry has no single host to test
    suffix = uuid.uuid4().hex[:8]
    internal, egress, proxy = f"egress-check-int-{suffix}", f"egress-check-ext-{suffix}", f"egress-check-proxy-{suffix}"
    proxy_url = "http://egress-proxy:3128"

    def probe(url: str, via_proxy: bool) -> str:
        """The HTTP status code that curl gets, or 000 if it gets no response."""
        flags = "" if via_proxy else "--noproxy '*'"
        script = f"curl -sS -o /dev/null -w '%{{http_code}}' --max-time {timeout_sec} {flags} '{url}' 2>/dev/null || true"
        r = _docker("run", "--rm", "--network", internal, "-e", f"HTTPS_PROXY={proxy_url}", "-e", f"HTTP_PROXY={proxy_url}",
                    "--entrypoint", "sh", image_tag(task_dir), "-c", script, timeout=timeout_sec + 60)
        return (r.stdout.strip() or "000")[-3:]

    results: dict[str, bool] = {}
    try:
        for args in (("network", "create", "--internal", internal), ("network", "create", egress),
                     ("run", "-d", "--name", proxy, "--network", internal, "--network-alias", "egress-proxy",
                      "-e", "POLICY_MODE=allow", "-e", f"ALLOWED_HOSTS={' '.join(allowed)}", PROXY_IMAGE),
                     ("network", "connect", egress, proxy)):
            r = _docker(*args)
            if r.returncode != 0:
                raise RuntimeError(f"docker {' '.join(args[:2])} failed: {r.stderr.strip()[-300:]}")
        for _ in range(30):  # wait until Squid listens
            if _docker("exec", proxy, "sh", "-c", "netstat -ltn | grep -q ':3128'").returncode == 0:
                break
            time.sleep(1)
        for host in concrete:
            code = probe(f"https://{host}/", via_proxy=True)
            results[f"allowed via proxy: {host} (HTTP {code})"] = code != "000"
        for host in blocked_hosts:
            code = probe(f"https://{host}/", via_proxy=True)
            results[f"blocked via proxy: {host} (HTTP {code})"] = code == "000"
            code = probe(f"https://{host}/", via_proxy=False)
            results[f"blocked without proxy: {host} (HTTP {code})"] = code == "000"
    finally:
        _docker("rm", "-f", proxy)
        _docker("network", "rm", internal)
        _docker("network", "rm", egress)
    for name, ok in results.items():
        print(f"{'pass' if ok else 'FAIL'}  {name}")
    failed = [name for name, ok in results.items() if not ok]
    assert not failed, f"The network seal does not behave as task.toml says: {failed}"
    return results


# ── AWS credentials ──────────────────────────────────────────────────────────

def refresh_credentials(region: str | None = None):
    """Put fresh AWS credentials from your profile or the default chain in the kernel environment.

    Harbor runs as a child process of the kernel, so it gets them too. Return (session, profile).
    """
    import boto3

    region = region or os.environ.get("AWS_REGION")
    # AWS_PROFILE comes from .env (see .env.example). If it is not set, boto3 uses the default credential chain.
    profile = os.environ.get("AWS_PROFILE") or None
    session = boto3.Session(profile_name=profile, region_name=region)
    creds = session.get_credentials()
    if creds is None:
        raise RuntimeError(
            f"boto3 found no AWS credentials. Profile used: {profile or 'none (default credential chain)'}.\n"
            f"Profiles that boto3 can see: {session.available_profiles}\n"
            "Set AWS_PROFILE in .env to one of these profiles. Then run the first cells of the notebook again."
        )
    frozen = creds.get_frozen_credentials()
    os.environ["AWS_ACCESS_KEY_ID"] = frozen.access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = frozen.secret_key
    os.environ.pop("AWS_SESSION_TOKEN", None)
    if frozen.token:
        os.environ["AWS_SESSION_TOKEN"] = frozen.token
    if region:
        os.environ["AWS_REGION"] = region
    return session, profile


# ── Harbor ───────────────────────────────────────────────────────────────────

def harbor_run(job_name: str, *args: str, tasks_dir: Path, tasks: list[Path] | None = None,
               n_concurrent: int | None = None, jobs_dir: Path = JOBS_DIR, region: str | None = None) -> Path:
    """Run the tasks as one Harbor job with the proxy environment. Return the job directory.

    tasks: a subset of task directories (default: all tasks in tasks_dir). n_concurrent: the number of
    trials at the same time (default: the Harbor default). The helper gets fresh AWS credentials first,
    writes the Harbor output to <jobs_dir>/<job_name>.log, and shows the end of the log.
    """
    jobs_dir = Path(jobs_dir)
    job_dir = jobs_dir / job_name
    if job_dir.exists():
        shutil.rmtree(job_dir)  # a new run replaces the old results
    jobs_dir.mkdir(exist_ok=True)
    refresh_credentials(region)
    paths = [arg for task in tasks for arg in ("-p", str(task))] if tasks else ["-p", str(tasks_dir)]
    cmd = ["harbor", "run", *paths, "-e", HARBOR_ENV, "-o", str(jobs_dir), "--job-name", job_name, "-y"]
    if n_concurrent:
        cmd += ["-n", str(n_concurrent)]
    cmd += list(args)
    # Harbor imports the proxy environment (harvest_env/) by name, so the module directory must be on PYTHONPATH.
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(MODULE_DIR), os.environ.get("PYTHONPATH")]))}
    # refresh_credentials() exported the frozen keys. A profile name means nothing in a task container, and
    # some agents (for example Claude Code) pass AWS_PROFILE through: the AWS SDK then prefers the missing
    # profile to the keys and finds no credentials. So Harbor gets the keys only.
    for name in ("AWS_PROFILE", "AWS_DEFAULT_PROFILE"):
        env.pop(name, None)
    log = jobs_dir / f"{job_name}.log"
    with log.open("w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env)
    print("\n".join(log.read_text().splitlines()[-8:]))
    assert r.returncode == 0, f"harbor run failed. See {log}."
    return job_dir


def trial_rewards(job_dir: Path) -> dict[str, dict]:
    """Return the reward of each trial, by task directory name.

    A trial with no reward.json is invalid (notebook 03): its reward is None, and `invalid` tells why
    (from reward_details.json). A key that is not in reward.json is undefined, for example the
    precision of a PR review with no findings (notebook 02): use reward.get(key).
    """
    rewards = {}
    for result in Path(job_dir).glob("*/result.json"):
        task = read_json(result)["task_name"].replace("/", "__")
        reward_file = result.parent / "verifier" / "reward.json"
        details_file = result.parent / "verifier" / "reward_details.json"
        if reward_file.exists():
            rewards[task] = read_json(reward_file)
        else:
            details = read_json(details_file) if details_file.exists() else {}
            rewards[task] = {"reward": None, "invalid": details.get("invalid") or ["no verifier output"]}
    return rewards


def start_harbor_view(jobs_dir: Path = JOBS_DIR, ports: str = "8080-8089", wait_sec: int = 120):
    """Start `harbor view` in the background. Return (process, url)."""
    jobs_dir = Path(jobs_dir)
    jobs_dir.mkdir(exist_ok=True)
    log = jobs_dir / "harbor-view.log"
    proc = subprocess.Popen(["harbor", "view", str(jobs_dir), "--port", ports],
                            stdout=log.open("w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    urls: list[str] = []
    for _ in range(wait_sec):  # wait until the server shows its URL
        urls = re.findall(r"https?://[\w.:-]+", log.read_text())
        if urls or proc.poll() is not None:
            break
        time.sleep(1)
    if not urls:
        print(log.read_text()[-2000:])
        raise RuntimeError(f"harbor view did not start. See {log}.")
    print(f"Harbor view runs at {urls[0]}")
    return proc, urls[0]


def stop_harbor_view(proc) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    print("Harbor view stopped.")


# ── Recorded runs (replay without Docker or AWS) ─────────────────────────────

RECORDED_DIR = MODULE_DIR / "datasets" / "recorded-runs"

# The files that a learner needs to grade a trial again. The raw agent sessions, junit.xml, and
# pytest.log are left out: they are large, and trajectory.json has the same steps.
RECORD_FILES = (
    "agent/trajectory.json",
    "artifacts/logs/artifacts/findings.json",
    "verifier/reward.json", "verifier/reward_details.json", "verifier/judge.json",
    "verifier/agent.patch", "verifier/failed_tests.txt",
    "verifier/rubric.json", "verifier/taste.json", "verifier/validation.json", "verifier/bloat.json",
)
RESULT_KEYS = ("task_name", "trial_name", "agent_info", "agent_result", "exception_info", "started_at", "finished_at")
_SECRET = re.compile(r"(AKIA|ASIA)[A-Z0-9]{16}|aws_secret_access_key|arn:aws:[a-z0-9-]+:[a-z0-9-]*:\d{12}|/Users/|/home/[a-z]")


def record_job(job_dir: Path, dest: Path) -> list[Path]:
    """Copy the files of each trial of a Harbor job to dest, for replay without Docker or AWS.

    result.json keeps only RESULT_KEYS (not the job config). Raise ValueError, and copy nothing, if a
    file looks like it has an AWS access key, an ARN with an account ID, or a local home path.
    """
    job_dir, dest = Path(job_dir), Path(dest)
    planned: dict[Path, str] = {}
    for result_path in sorted(job_dir.glob("*/result.json")):
        trial = result_path.parent
        result = read_json(result_path)
        planned[dest / trial.name / "result.json"] = json.dumps({k: result.get(k) for k in RESULT_KEYS}, indent=1)
        for rel in RECORD_FILES:
            if (trial / rel).is_file():
                planned[dest / trial.name / rel] = (trial / rel).read_text(errors="replace")
    hits = [f"{path.relative_to(dest)}: {m.group(0)[:20]}" for path, text in planned.items() if (m := _SECRET.search(text))]
    if hits:
        raise ValueError("Not recorded. These files look like they have credentials or local paths:\n" + "\n".join(hits))
    for path, text in planned.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return sorted(planned)
