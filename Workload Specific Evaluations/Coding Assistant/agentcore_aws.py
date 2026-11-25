#!/usr/bin/env python3
"""AWS helpers for harbor_env: resource names, clients, bootstrap, image builds on CodeBuild, AgentCore runtimes, and sessions.

Every resource is created with boto3 calls, tagged, and named from one prefix (Names.prefix, $P2_PREFIX).
"""
import base64, datetime, hashlib, importlib.util, io, json, os, pathlib, random, re, sys, tarfile, tempfile, time, urllib.parse, uuid, zipfile

import boto3
from botocore.config import Config
from botocore.exceptions import ParamValidationError

MIN_BOTO3 = (1, 44, 7)  # first release whose AgentCore create/update_agent_runtime accept platformVersion
if tuple(int(x) for x in re.findall(r"\d+", boto3.__version__)[:3]) < MIN_BOTO3:  # before any client or AWS call: an old boto3 would only fail at `deploy`, after the build
    sys.exit(f"agentcore_aws: boto3 {boto3.__version__} is too old, need >= {'.'.join(map(str, MIN_BOTO3))}. "
             f"In a venv: python3 -m pip install -U 'boto3>=1.44.7'  (this python: {sys.executable})")

R = "us-east-1"
# Expiry is only a tag (nothing deletes on it): a date two weeks out, or $P2_EXPIRY.
TAGS = {"Project": "coding-assistant-evals",
        "Expiry": os.environ.get("P2_EXPIRY") or (datetime.date.today() + datetime.timedelta(days=14)).isoformat()}
TAGL = [{"Key": k, "Value": v} for k, v in TAGS.items()]
HERE = pathlib.Path(__file__).resolve().parent
IMG = HERE / "image"
# Defaults; load_models() replaces them with the values in model_config.py.
JUDGE_MODEL = "us.anthropic.claude-haiku-5-5"      # PR review judge (JUDGE_MODEL_ID)
VERIFIER_MODEL = "us.anthropic.claude-haiku-5-5"   # coding verifier (VERIFIER_MODEL_ID)
class Names:
    """Every resource name comes from one prefix ($P2_PREFIX, default evals-hb-)."""
    prefix = os.environ.get("P2_PREFIX", "evals-hb-")

    @classmethod
    def repo(c): return c.prefix + "fw"
    @classmethod
    def project(c): return c.prefix + "build"
    @classmethod
    def loggroup(c): return "/aws/codebuild/" + c.prefix + "build"
    @classmethod
    def build_role(c): return c.prefix + "codebuild-role"
    @classmethod
    def runtime_role(c): return c.prefix + "grader-role"       # AgentCore execution role = the grader phase's credentials
    @classmethod
    def agent_role(c): return c.prefix + "agent-role"          # credentials for the agent phase
    @classmethod
    def runtime(c, task): return re.sub(r"[^A-Za-z0-9_]", "_", c.prefix) + task


CFG = Config(read_timeout=3700, connect_timeout=10, max_pool_connections=64, retries={"max_attempts": 6, "mode": "adaptive"})
sts = boto3.client("sts", region_name=R)
iam = boto3.client("iam", region_name=R)
ecr = boto3.client("ecr", region_name=R)
s3 = boto3.client("s3", region_name=R)
cb = boto3.client("codebuild", region_name=R)
logs = boto3.client("logs", region_name=R)
bedrock = boto3.client("bedrock", region_name=R)
ctl = boto3.client("bedrock-agentcore-control", region_name=R, config=CFG)
dp = boto3.client("bedrock-agentcore", region_name=R, config=CFG)

_acct = None


def acct():
    global _acct
    _acct = _acct or sts.get_caller_identity()["Account"]
    return _acct


def mask(x):
    return re.sub(r"\d{12}", "ACCT", x)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_models(module=HERE):
    """The judge model IDs from model_config.py, so that file stays the one place to change them."""
    global JUDGE_MODEL, VERIFIER_MODEL
    f = module / "model_config.py"
    if not f.exists():
        return
    spec = importlib.util.spec_from_file_location("model_config_of_module", f)
    mc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mc)
    JUDGE_MODEL = getattr(mc, "JUDGE_MODEL_ID", JUDGE_MODEL)
    VERIFIER_MODEL = getattr(mc, "VERIFIER_MODEL_ID", VERIFIER_MODEL)


def bucket():
    return f"{Names.prefix}src-{acct()}-{R}"


def runtime_name(task):
    return Names.runtime(task)


def retry(fn, errs, what, deadline=240, first=2.0):
    """Call fn until it stops failing with an error whose text contains one of `errs` (IAM changes take seconds to reach other services)."""
    t0, n = time.time(), 0
    while True:
        try:
            r = fn()
            if n:
                log(f"  {what}: succeeded after {n} propagation retr{'y' if n == 1 else 'ies'}, {time.time() - t0:.0f}s")
            PROP[what] = {"retries": n, "secs": round(time.time() - t0, 1)}
            return r
        except Exception as e:
            msg = type(e).__name__ + " " + str(e)
            # ParamValidationError text lists every valid parameter (roleArn...), so it would match "role" below: never retry it
            if isinstance(e, ParamValidationError) or not any(k.lower() in msg.lower() for k in errs) or time.time() - t0 > deadline:
                raise
            n += 1
            time.sleep(min(first * (1.5 ** min(n, 6)), 12))


PROP = {}


# ---------------------------------------------------------------- policies (what bootstrap installs)
def judge_arns():
    """The judge models' inference profiles and the foundation models they route to (what Converse is authorised against)."""
    out = []
    for model in dict.fromkeys([JUDGE_MODEL, VERIFIER_MODEL]):
        try:
            p = bedrock.get_inference_profile(inferenceProfileIdentifier=model)
            out += [p["inferenceProfileArn"]] + [m["modelArn"] for m in p["models"]]
        except Exception as e:  # no bedrock:GetInferenceProfile: assume the documented routing of the US profile
            log(f"WARNING: GetInferenceProfile {model} failed ({type(e).__name__}); the grader policy uses guessed ARNs")
            fm = model.split(".", 1)[1]
            out += [f"arn:aws:bedrock:{R}:{acct()}:inference-profile/{model}"] + \
                   [f"arn:aws:bedrock:{r}::foundation-model/{fm}" for r in ("us-east-1", "us-east-2", "us-west-2")]
    return list(dict.fromkeys(out))


def trust(service=None):
    if service:
        st = {"Effect": "Allow", "Principal": {"Service": service}, "Action": "sts:AssumeRole"}
        if service.startswith("bedrock-agentcore"):
            st["Condition"] = {"StringEquals": {"aws:SourceAccount": acct()}}
    else:  # any principal of this account that is allowed sts:AssumeRole (the environment)
        st = {"Effect": "Allow", "Principal": {"AWS": f"arn:aws:iam::{acct()}:root"}, "Action": "sts:AssumeRole"}
    return {"Version": "2012-10-17", "Statement": [st]}


def policy_build():
    lg = f"arn:aws:logs:{R}:{acct()}:log-group:{Names.loggroup()}"
    return {"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Action": ["logs:CreateLogStream", "logs:PutLogEvents"], "Resource": [lg, lg + ":*"]},
        {"Effect": "Allow", "Action": ["s3:GetObject", "s3:GetObjectVersion"], "Resource": f"arn:aws:s3:::{bucket()}/*"},
        {"Effect": "Allow", "Action": ["s3:GetBucketLocation", "s3:ListBucket"], "Resource": f"arn:aws:s3:::{bucket()}"},
        {"Effect": "Allow", "Action": "ecr:GetAuthorizationToken", "Resource": "*"},
        {"Effect": "Allow", "Action": ["ecr:BatchCheckLayerAvailability", "ecr:InitiateLayerUpload", "ecr:UploadLayerPart", "ecr:CompleteLayerUpload",
                                       "ecr:PutImage", "ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"],
         "Resource": f"arn:aws:ecr:{R}:{acct()}:repository/{Names.repo()}"}]}


def policy_runtime():
    """Grader phase: image pull, logs, and InvokeModel for ONLY the judge model (verifier and judges call Converse = bedrock:InvokeModel)."""
    return {"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Action": "ecr:GetAuthorizationToken", "Resource": "*"},
        {"Effect": "Allow", "Action": ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"], "Resource": f"arn:aws:ecr:{R}:{acct()}:repository/{Names.repo()}"},
        {"Effect": "Allow", "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"],
         "Resource": f"arn:aws:logs:{R}:{acct()}:log-group:/aws/bedrock-agentcore/runtimes/*"},
        {"Effect": "Allow", "Action": "logs:DescribeLogGroups", "Resource": "*"},
        {"Effect": "Allow", "Action": "bedrock:InvokeModel", "Resource": judge_arns()}]}


# ---------------------------------------------------------------- bootstrap (idempotent, direct API calls)
STEPS = []


def step(name, action, t0):
    STEPS.append({"step": name, "action": action, "secs": round(time.time() - t0, 1)})
    log(f"  {name}: {action} ({STEPS[-1]['secs']}s)")


def ensure_role(name, trust_doc, policy_name, policy, desc):
    t0 = time.time()
    try:
        iam.get_role(RoleName=name)
        iam.update_assume_role_policy(RoleName=name, PolicyDocument=json.dumps(trust_doc))
        act = "exists, policy refreshed"
    except iam.exceptions.NoSuchEntityException:
        iam.create_role(RoleName=name, AssumeRolePolicyDocument=json.dumps(trust_doc), Tags=TAGL, Description=desc)
        iam.get_waiter("role_exists").wait(RoleName=name)
        act = "created"
    iam.put_role_policy(RoleName=name, PolicyName=policy_name, PolicyDocument=json.dumps(policy))
    step(f"iam role {name}", act, t0)


def bootstrap(agent_policy_name, agent_policy):
    """Create everything a fresh account needs (idempotent); the agent role gets only agent_policy. Returns the report dict."""
    t_all = time.time()
    STEPS.clear(); PROP.clear()
    a = acct()
    log(f"bootstrap: prefix {Names.prefix}, region {R}")
    # 1. ECR repo
    t0 = time.time()
    try:
        ecr.describe_repositories(repositoryNames=[Names.repo()]); act = "exists"
    except ecr.exceptions.RepositoryNotFoundException:
        ecr.create_repository(repositoryName=Names.repo(), tags=TAGL, imageTagMutability="MUTABLE"); act = "created"
    step(f"ecr repository {Names.repo()}", act, t0)
    # 2. S3 source bucket (private, objects expire after 3 days)
    t0 = time.time()
    try:
        s3.head_bucket(Bucket=bucket()); act = "exists"
    except Exception:
        s3.create_bucket(Bucket=bucket()); act = "created"
        s3.get_waiter("bucket_exists").wait(Bucket=bucket())
    s3.put_public_access_block(Bucket=bucket(), PublicAccessBlockConfiguration={
        "BlockPublicAcls": True, "IgnorePublicAcls": True, "BlockPublicPolicy": True, "RestrictPublicBuckets": True})
    s3.put_bucket_tagging(Bucket=bucket(), Tagging={"TagSet": TAGL})
    s3.put_bucket_lifecycle_configuration(Bucket=bucket(), LifecycleConfiguration={"Rules": [
        {"ID": "expire-build-contexts", "Status": "Enabled", "Filter": {"Prefix": "build/"}, "Expiration": {"Days": 3}}]})
    step(f"s3 bucket {mask(bucket())}", act, t0)
    # 3. CodeBuild log group
    t0 = time.time()
    try:
        logs.create_log_group(logGroupName=Names.loggroup(), tags=TAGS); act = "created"
    except logs.exceptions.ResourceAlreadyExistsException:
        act = "exists"
    logs.put_retention_policy(logGroupName=Names.loggroup(), retentionInDays=3)
    step(f"log group {Names.loggroup()}", act, t0)
    # 4-6. IAM roles with scoped inline policies
    ensure_role(Names.build_role(), trust("codebuild.amazonaws.com"), "build", policy_build(), "coding evals: CodeBuild image builds")
    ensure_role(Names.runtime_role(), trust("bedrock-agentcore.amazonaws.com"), "grader", policy_runtime(),
                "coding evals: AgentCore execution role, grader phase (judge model only)")
    ensure_role(Names.agent_role(), trust(), agent_policy_name, agent_policy, "coding evals: credentials for the agent phase (agent models only)")
    # 7. CodeBuild project (the role may not be usable yet: retry until IAM has propagated)
    t0 = time.time()
    proj = dict(name=Names.project(), serviceRole=f"arn:aws:iam::{a}:role/{Names.build_role()}", artifacts={"type": "NO_ARTIFACTS"},
                source={"type": "S3", "location": f"{bucket()}/build/none.zip"},
                environment={"type": "ARM_CONTAINER", "computeType": "BUILD_GENERAL1_LARGE", "image": "aws/codebuild/amazonlinux-aarch64-standard:3.0",
                             "privilegedMode": True, "environmentVariables": [
                                 {"name": "REPO_URI", "value": f"{a}.dkr.ecr.{R}.amazonaws.com/{Names.repo()}", "type": "PLAINTEXT"}]},
                logsConfig={"cloudWatchLogs": {"status": "ENABLED", "groupName": Names.loggroup()}}, timeoutInMinutes=40)
    if cb.batch_get_projects(names=[Names.project()])["projects"]:
        cb.update_project(**proj); act = "exists, updated"
    else:
        retry(lambda: cb.create_project(tags=[{"key": k, "value": v} for k, v in TAGS.items()], **proj),
              ["not authorized to perform: sts:AssumeRole", "role"], "codebuild create_project")
        act = "created"
    step(f"codebuild project {Names.project()}", act, t0)
    # 8. prove the agent role is usable (this is also the IAM propagation wait for the roles above)
    t0 = time.time()
    retry(lambda: agent_creds(900), ["AccessDenied", "not authorized"], "assume agent role")
    step("iam propagation check (assume agent role)", "ok", t0)
    rep = {"prefix": Names.prefix, "secs": round(time.time() - t_all, 1), "steps": STEPS[:], "propagation": dict(PROP)}
    (HERE / "jobs").mkdir(exist_ok=True)
    (HERE / "jobs" / "bootstrap-report.json").write_text(mask(json.dumps(rep, indent=1)))
    log(f"bootstrap done in {rep['secs']}s, {len(STEPS)} steps")
    return rep


def agent_creds(seconds=3600):
    c = sts.assume_role(RoleArn=f"arn:aws:iam::{acct()}:role/{Names.agent_role()}", RoleSessionName="agent", DurationSeconds=seconds)["Credentials"]
    return {"AWS_ACCESS_KEY_ID": c["AccessKeyId"], "AWS_SECRET_ACCESS_KEY": c["SecretAccessKey"], "AWS_SESSION_TOKEN": c["SessionToken"]}


# ---------------------------------------------------------------- build (CodeBuild, arm64)
def image_exists(task):
    try:
        ecr.describe_images(repositoryName=Names.repo(), imageIds=[{"imageTag": task}])
        return True
    except Exception:
        return False


def build(task, context):
    """Build the image `task` on CodeBuild from the build context that context(task, None, dest) writes into dest."""
    t0 = time.time()
    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        context(task, None, d)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(d.rglob("*")):
                if p.is_file():
                    zi = zipfile.ZipInfo(str(p.relative_to(d)), (2020, 1, 1, 0, 0, 0))
                    zi.external_attr = (p.stat().st_mode & 0xFFFF) << 16
                    zi.compress_type = zipfile.ZIP_DEFLATED
                    z.writestr(zi, p.read_bytes())
    key = f"build/{task}.zip"
    s3.put_object(Bucket=bucket(), Key=key, Body=buf.getvalue(), Tagging=urllib.parse.urlencode(TAGS))
    b = cb.start_build(projectName=Names.project(), sourceLocationOverride=f"{bucket()}/{key}", computeTypeOverride="BUILD_GENERAL1_LARGE",
                       environmentVariablesOverride=[{"name": "TAG", "value": task, "type": "PLAINTEXT"}],
                       timeoutInMinutesOverride=40)["build"]["id"]
    log(task, "build started", b.split(":")[1][:8])
    while True:
        x = cb.batch_get_builds(ids=[b])["builds"][0]
        if x["buildStatus"] != "IN_PROGRESS":
            break
        time.sleep(10)
    phases = {p["phaseType"]: p.get("durationInSeconds") for p in x["phases"] if p.get("durationInSeconds") is not None}
    tail = ""
    try:
        ev = logs.get_log_events(logGroupName=x["logs"]["groupName"], logStreamName=x["logs"]["streamName"], startFromHead=False, limit=80)["events"]
        tail = "\n".join(mask(e["message"].rstrip()) for e in ev)
    except Exception as e:
        tail = repr(e)
    why = "; ".join(f'{p["phaseType"]}: {c.get("message") or c.get("statusCode")}'
                    for p in x["phases"] if p.get("phaseStatus") == "FAILED" for c in p.get("contexts", [])[:1])
    errs = [l for l in tail.splitlines() if l.startswith("ERROR:")]  # the docker/buildkit line that names the failing step
    if errs and x["buildStatus"] != "SUCCEEDED":
        why = errs[-1][:300] + " (" + why.split(":")[0] + " phase)"
    res = {"task": task, "status": x["buildStatus"], "secs": round(time.time() - t0), "phases": phases, "reason": mask(why), "log_tail": tail}
    log(task, "build", x["buildStatus"], res["secs"], "s")
    if x["buildStatus"] != "SUCCEEDED":
        print(tail)
        log(task, "BUILD FAILED:", res["reason"] or "see the log tail above")
    return res


# ---------------------------------------------------------------- runtimes
def find_runtime(task):
    kw = {}
    while True:
        page = ctl.list_agent_runtimes(maxResults=100, **kw)
        for r in page["agentRuntimes"]:
            if r["agentRuntimeName"] == runtime_name(task):
                return r
        if not page.get("nextToken"):
            return None
        kw = {"nextToken": page["nextToken"]}


def wait_ready(rid):
    while True:
        g = ctl.get_agent_runtime(agentRuntimeId=rid)
        if g["status"] not in ("CREATING", "UPDATING"):
            return g
        time.sleep(4)


def deploy(task, platform, memory_mib=None):
    """Create the runtime (PUBLIC network, no VPC) for the task image, or point an existing one at it.
    memory_mib: minimumMemoryInMiB (platform V2; images over 2 GB need it for the build disk)."""
    t0 = time.time()
    uri = f"{acct()}.dkr.ecr.{R}.amazonaws.com/{Names.repo()}:{task}"
    role = iam.get_role(RoleName=Names.runtime_role())["Role"]["Arn"]
    art = {"containerConfiguration": {"containerUri": uri}}
    net = {"networkMode": "PUBLIC"}
    res_cfg = {"resourceConfigurations": [{"minimumMemoryInMiB": memory_mib}]} if memory_mib else {}
    ex = find_runtime(task)
    if ex:
        rid = ex["agentRuntimeId"]
        ctl.update_agent_runtime(agentRuntimeId=rid, roleArn=role, agentRuntimeArtifact=art, networkConfiguration=net,
                                 metadataConfiguration={"requireMMDSV2": True}, platformVersion=platform, **res_cfg)
    else:
        # a role created seconds ago may not be assumable by AgentCore yet: retry on role errors
        rid = retry(lambda: ctl.create_agent_runtime(agentRuntimeName=runtime_name(task), roleArn=role, agentRuntimeArtifact=art,
                                                     networkConfiguration=net, platformVersion=platform, tags=TAGS,
                                                     **res_cfg)["agentRuntimeId"],
                    ["role", "assume", "not authorized"], f"agentcore create_runtime {task}")
        g = wait_ready(rid)
        # create takes no metadataConfiguration; V2 already requires IMDSv2, and every update costs another image preparation
        if g["status"] == "READY" and not (g.get("metadataConfiguration") or {}).get("requireMMDSV2"):
            ctl.update_agent_runtime(agentRuntimeId=rid, roleArn=role, agentRuntimeArtifact=art, networkConfiguration=net,
                                     metadataConfiguration={"requireMMDSV2": True}, platformVersion=platform, **res_cfg)
    time.sleep(2)
    g = wait_ready(rid)
    res = {"task": task, "id": rid, "status": g["status"], "secs": round(time.time() - t0, 1), "reason": g.get("failureReason"),
           "imdsv2_required": (g.get("metadataConfiguration") or {}).get("requireMMDSV2")}
    log(task, "runtime", res["status"], res["secs"], "s", res["reason"] or "")
    return res


_arn = {}


def runtime_arn(task):
    if task not in _arn:
        r = find_runtime(task)
        if not r:
            sys.exit(f"agentcore_aws: no AgentCore runtime {runtime_name(task)} (prefix {Names.prefix})")
        _arn[task] = ctl.get_agent_runtime(agentRuntimeId=r["agentRuntimeId"])["agentRuntimeArn"]
    return _arn[task]


def extract_session_tar(raw, out):
    """Extract the archive that came out of the session. It is untrusted: path traversal and links out of `out` are never written, but one
    bad member (Codex leaves an absolute symlink under agent/codex-home/tmp/arg0) must not cost the graded reward.json, so every member is
    extracted on its own. Returns (skipped member names, a note when the archive ended early or None).
    shortcut: needs tarfile.data_filter (Python 3.10.12+, 3.11.4+, 3.12+); older Pythons only get the plain-file check below."""
    skipped, note = [], None

    def flt(member, dest):  # a closure: 30 trials extract at the same time in threads
        try:
            if hasattr(tarfile, "data_filter"):
                return tarfile.data_filter(member, dest)
            if (member.isreg() or member.isdir()) and not os.path.isabs(member.name) and \
                    os.path.realpath(os.path.join(dest, member.name)).startswith(os.path.realpath(dest) + os.sep):
                return member
            raise tarfile.FilterError(member)
        except tarfile.FilterError as e:
            skipped.append(f"{member.name}: {type(e).__name__}")
            return None
    n = 0
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
        try:
            for m in tf:
                try:
                    m2 = flt(m, str(out))
                    if m2 is None:
                        continue
                    tf.extract(m2, out, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))  # the filter runs twice: the second is the stock one
                    n += 1
                except (OSError, tarfile.TarError) as e:  # one member that cannot be written: skip it, keep the rest
                    skipped.append(f"{m.name}: {type(e).__name__}")
        except (EOFError, tarfile.TarError, OSError) as e:  # truncated or corrupt tail: keep what came before
            note = f"archive ended early after {n} members: {e!r}"[:300]
    return skipped, note


# ---------------------------------------------------------------- one session
def shq(s):
    return "'" + s.replace("'", "'\"'\"'") + "'"


class Session:
    """One AgentCore session = one fresh microVM of the task's runtime. cmd() runs a command as root."""

    def __init__(self, task, sid=None):
        self.task, self.arn = task, runtime_arn(task)
        self.sid = sid or ("hb-" + uuid.uuid4().hex + uuid.uuid4().hex[:6])
        self.calls = 0
        self.throttles = 0
        self.start_attempts = 1

    def cmd(self, command, timeout=300, retries=12):
        last = None
        for i in range(retries):
            try:
                t0 = time.time()
                self.calls += 1
                resp = dp.invoke_agent_runtime_command(
                    agentRuntimeArn=self.arn, runtimeSessionId=self.sid, qualifier="DEFAULT", contentType="application/json",
                    accept="application/vnd.amazon.eventstream", body={"command": "/bin/bash -c " + shq(command), "timeout": timeout})
                out, code = [], None
                for ev in resp.get("stream", []):
                    c = ev.get("chunk", {})
                    if "contentDelta" in c:
                        d = c["contentDelta"]
                        out.append((d.get("stdout") or "") + (d.get("stderr") or ""))
                    if "contentStop" in c:
                        code = c["contentStop"].get("exitCode")
                return "".join(out), code, round(time.time() - t0, 2)
            except Exception as e:  # throttling or a transient service error: count it and retry
                last = e
                name = type(e).__name__
                if "Throttl" in name or "429" in str(e) or "ThrottlingException" in str(e):
                    self.throttles += 1
                if i < retries - 1 and any(k in (name + str(e)) for k in ("Throttl", "ServiceUnavailable", "InternalServer", "Timeout", "ConnectionClosed", "EndpointConnection", "ServiceQuota", "failed to become ready")):
                    time.sleep(2 + 3 * min(i, 3))
                    continue
                raise
        raise last

    def start(self, command, attempts=3):
        """The first call of a session (a read-only probe: no agent work has run yet). A 5xx or RuntimeClientError here means the microVM did
        not come up: stop that session and retry on a fresh one, with 2 to 10 s of backoff and jitter. Later calls are never retried this way,
        because they may already have run agent work."""
        for i in range(attempts):
            try:
                return self.cmd(command)
            except Exception as e:
                code = (getattr(e, "response", None) or {}).get("ResponseMetadata", {}).get("HTTPStatusCode") or 0
                if i == attempts - 1 or not ("RuntimeClientError" in type(e).__name__ + str(e) or code >= 500):
                    raise
                wait = min(10.0, 2.0 * 2 ** i) * random.uniform(0.75, 1.25)
                log(self.task, f"session start failed ({type(e).__name__}); retry {i + 2} of {attempts} in {wait:.0f} s on a new session")
                self.stop()
                self.sid = "hb-" + uuid.uuid4().hex + uuid.uuid4().hex[:6]
                self.start_attempts = i + 2
                time.sleep(wait)

    def put(self, path, data: bytes, mode="0644", owner="root"):
        """Each chunk goes to its own root-only file, so a retried command cannot write a chunk twice; the result is checked."""
        b = base64.b64encode(data).decode()
        parts = f"{path}.b64.d"
        self.cmd(f"rm -rf {parts} && (umask 077 && mkdir {parts})")
        for n, chunk in enumerate([b[i:i + 40000] for i in range(0, len(b), 40000)] or [""]):
            self.cmd(f"(umask 077 && printf %s '{chunk}' > {parts}/{n:06d})")
        sha = hashlib.sha256(data).hexdigest()
        same = f"[ \"$(sha256sum < {path} 2>/dev/null | cut -c1-64)\" = {sha} ]"
        o, c, _ = self.cmd(f"{{ {same} || {{ cat {parts}/* | base64 -d > {path}; }}; }} && rm -rf {parts} && {same} "
                           f"&& chmod {mode} {path} && chown {owner} {path}")
        assert c == 0, f"upload of {path} failed or arrived changed: {o[-200:]}"

    def stop(self):
        try:
            dp.stop_runtime_session(agentRuntimeArn=self.arn, runtimeSessionId=self.sid, qualifier="DEFAULT")
        except Exception as e:
            return repr(e)[:100]


def env_file(d):
    return "".join(f"{k}={v}\n" for k, v in d.items())


def set_phase(s, host, extra):
    """Switch the proxy allowlist and the agent-side environment for the next phase (root-owned files)."""
    s.put("/run/fw/allow.txt", (host + "\n").encode() if host else b"", "0644")
    s.put("/run/fw/extra.env", env_file(extra).encode(), "0600")


def runtime_role_creds(s):
    """Root, main netns: the runtime execution role's credentials from IMDSv2 (never given to the agent; unreachable from its netns)."""
    o, c, _ = s.cmd('T=$(curl -s -m 5 -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 300"); '
                    'R=$(curl -s -m 5 -H "X-aws-ec2-metadata-token: $T" http://169.254.169.254/latest/meta-data/iam/security-credentials/); '
                    'curl -s -m 5 -H "X-aws-ec2-metadata-token: $T" http://169.254.169.254/latest/meta-data/iam/security-credentials/$R')
    j = json.loads(o)
    return {"AWS_ACCESS_KEY_ID": j["AccessKeyId"], "AWS_SECRET_ACCESS_KEY": j["SecretAccessKey"], "AWS_SESSION_TOKEN": j["Token"]}
