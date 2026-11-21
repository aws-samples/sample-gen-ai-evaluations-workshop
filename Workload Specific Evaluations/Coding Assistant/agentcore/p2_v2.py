#!/usr/bin/env python3
"""P2 v2: the Coding Assistant eval on Amazon Bedrock AgentCore Runtime, from an empty AWS account with one command.

No CloudFormation, no local Docker, no Harbor. Everything is created with boto3 calls and tagged. Names start with --prefix.

  p2_v2.py all TASK[:N]... [--job NAME]   bootstrap if needed, build missing images, deploy, run all trials at once, print a summary
  p2_v2.py bootstrap                     idempotent: IAM roles, ECR repo, S3 bucket, log group, CodeBuild project (+ report with timings)
  p2_v2.py policies                      print the IAM policies this tool installs (account id masked)
  p2_v2.py build  TASK...                build the arm64 image of each task on CodeBuild (parallel), push to ECR
  p2_v2.py deploy TASK...                create or update one AgentCore runtime per task (parallel)
  p2_v2.py run    TASK[:N]... [--job NAME] [--agent codex|empty|patch:FILE] [--variant par|hard|par,hard]
  p2_v2.py cleanup                       delete runtimes, ECR images and repo, CodeBuild project, IAM roles made by this tool
                                         (S3 bucket and log groups: scripts/teardown.sh lists the commands)

TASK is one of pr371, pr338, pr4121, pr4615. The module is found from --module, $P2_MODULE, or by looking for coding_eval.py upwards
(this file lives in <module>/agentcore/, so the module is its parent). Model IDs come from the module's model_config.py.
--variant (coding tasks only, opt-in): par = base+hidden pytest runs at the same time as the LLM stages; hard = that pytest stage runs on a
fresh root-owned checkout of the sealed repo with the agent's patch applied (see image/variants/).
"""
import argparse, base64, concurrent.futures as cf, datetime, importlib.util, io, json, os, pathlib, re, shutil
import sys, tarfile, tempfile, time, urllib.parse, uuid, zipfile

import boto3
from botocore.config import Config

R = "us-east-1"
# Expiry is only a tag (nothing deletes on it): a date two weeks out, or $P2_EXPIRY. Run `cleanup` when you are done.
TAGS = {"Project": "evals-streamline-spike",
        "Expiry": os.environ.get("P2_EXPIRY") or (datetime.date.today() + datetime.timedelta(days=14)).isoformat()}
TAGL = [{"Key": k, "Value": v} for k, v in TAGS.items()]
HERE = pathlib.Path(__file__).resolve().parent
IMG = HERE / "image"
# Defaults; main() replaces them with the values in the module's model_config.py when it finds the module.
CODEX_MODEL = "openai.gpt-6-sol"
JUDGE_MODEL = "us.anthropic.claude-haiku-5-5"      # PR review judge (JUDGE_MODEL_ID)
VERIFIER_MODEL = "us.anthropic.claude-haiku-5-5"   # coding verifier (VERIFIER_MODEL_ID)
HOST_MANTLE = "bedrock-mantle.us-east-1.api.aws"
HOST_RUNTIME = "bedrock-runtime.us-east-1.amazonaws.com"


class Names:
    """Every resource name comes from one prefix (--prefix / $P2_PREFIX, default evals-p2b-)."""
    prefix = os.environ.get("P2_PREFIX", "evals-p2b-")

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
    def agent_role(c): return c.prefix + "agent-role"          # assumed by the runner: credentials for the agent phase
    @classmethod
    def runtime(c, task): return re.sub(r"[^A-Za-z0-9_]", "_", c.prefix) + task


TASKS = {  # short name -> (kind, dataset dir, task dir name)
    "pr371": ("coding", "strands-evals-coding", "strands-agents__evals__pr371-fix"),
    "pr338": ("coding", "strands-evals-coding", "strands-agents__evals__pr338-fix"),
    "pr4121": ("review", "awslabs-mcp-pr-review", "awslabs__mcp__pr4121-approve"),
    "pr4615": ("review", "awslabs-mcp-pr-review", "awslabs__mcp__pr4615-reject"),
}

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


def module_dir(arg=None, required=True):
    p = arg or os.environ.get("P2_MODULE")
    if p:
        return pathlib.Path(p)
    for base in (pathlib.Path.cwd(), HERE):
        for d in [base, *list(base.parents)[:6]]:
            for c in (d, d / "Coding Assistant", d / "Workload Specific Evaluations" / "Coding Assistant",
                      *d.glob("*/Workload Specific Evaluations/Coding Assistant")):
                if (c / "coding_eval.py").exists() and (c / "datasets").is_dir():
                    return c
    if not required:
        return None
    sys.exit("module not found: set --module or $P2_MODULE to the 'Coding Assistant' module directory (read only, never written)")


def load_models(module):
    """Read the model IDs from the module's own model_config.py, so the module stays the one place to change them."""
    global CODEX_MODEL, JUDGE_MODEL, VERIFIER_MODEL
    f = module / "model_config.py"
    if not f.exists():
        return
    spec = importlib.util.spec_from_file_location("model_config_of_module", f)
    mc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mc)
    CODEX_MODEL = getattr(mc, "CODEX_MODEL_ID", CODEX_MODEL)
    JUDGE_MODEL = getattr(mc, "JUDGE_MODEL_ID", JUDGE_MODEL)
    VERIFIER_MODEL = getattr(mc, "VERIFIER_MODEL_ID", VERIFIER_MODEL)


def task_dir(task, module):
    kind, ds, name = TASKS[task]
    return module / "datasets" / ds / "tasks" / name


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
            if not any(k.lower() in msg.lower() for k in errs) or time.time() - t0 > deadline:
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
        except Exception:  # no bedrock:GetInferenceProfile: assume the documented routing of the US profile
            fm = model.split(".", 1)[1]
            out += [f"arn:aws:bedrock:{R}:{acct()}:inference-profile/{model}"] + \
                   [f"arn:aws:bedrock:{r}::foundation-model/{fm}" for r in ("us-east-1", "us-east-2", "us-west-2")]
    return list(dict.fromkeys(out))


def trust(service=None):
    if service:
        st = {"Effect": "Allow", "Principal": {"Service": service}, "Action": "sts:AssumeRole"}
        if service.startswith("bedrock-agentcore"):
            st["Condition"] = {"StringEquals": {"aws:SourceAccount": acct()}}
    else:  # any principal of this account that is allowed sts:AssumeRole (the runner)
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


def policy_agent():
    """Agent phase: only the Codex model on bedrock-mantle."""
    return {"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Action": "bedrock-mantle:CreateInference", "Resource": "*",
         "Condition": {"StringEquals": {"bedrock-mantle:Model": CODEX_MODEL}}}]}


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


def missing():
    """What a fresh account lacks (read-only checks, about 2 s)."""
    out = []
    checks = [(Names.repo(), lambda: ecr.describe_repositories(repositoryNames=[Names.repo()])),
              (bucket(), lambda: s3.head_bucket(Bucket=bucket())),
              (Names.project(), lambda: cb.batch_get_projects(names=[Names.project()])["projects"][0]),
              (Names.build_role(), lambda: iam.get_role(RoleName=Names.build_role())),
              (Names.runtime_role(), lambda: iam.get_role(RoleName=Names.runtime_role())),
              (Names.agent_role(), lambda: iam.get_role(RoleName=Names.agent_role()))]
    for n, fn in checks:
        try:
            fn()
        except Exception:
            out.append(n)
    return out


def bootstrap(force=False):
    """Create everything a fresh account needs (idempotent). Returns the report dict (also written to bootstrap-report.json)."""
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
        {"ID": "expire-build-contexts", "Status": "Enabled", "Filter": {"Prefix": "p2/"}, "Expiration": {"Days": 3}}]})
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
    ensure_role(Names.build_role(), trust("codebuild.amazonaws.com"), "build", policy_build(), "evals P2: CodeBuild image builds")
    ensure_role(Names.runtime_role(), trust("bedrock-agentcore.amazonaws.com"), "grader", policy_runtime(),
                "evals P2: AgentCore execution role, grader phase (judge model only)")
    ensure_role(Names.agent_role(), trust(), "codex-model-only", policy_agent(), "evals P2: credentials for the agent phase (Codex model only)")
    # 7. CodeBuild project (the role may not be usable yet: retry until IAM has propagated)
    t0 = time.time()
    proj = dict(name=Names.project(), serviceRole=f"arn:aws:iam::{a}:role/{Names.build_role()}", artifacts={"type": "NO_ARTIFACTS"},
                source={"type": "S3", "location": f"{bucket()}/p2/none.zip"},
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
    c = sts.assume_role(RoleArn=f"arn:aws:iam::{acct()}:role/{Names.agent_role()}", RoleSessionName="p2-agent", DurationSeconds=seconds)["Credentials"]
    return {"AWS_ACCESS_KEY_ID": c["AccessKeyId"], "AWS_SECRET_ACCESS_KEY": c["SecretAccessKey"], "AWS_SESSION_TOKEN": c["SessionToken"]}


# ---------------------------------------------------------------- build (CodeBuild, arm64)
def build_context(task, module, dest):
    """environment/ of the task (Dockerfile used verbatim) + P2 layer + firewall files + hidden verifier files (+ variants for coding tasks)."""
    td = task_dir(task, module)
    env = td / "environment"
    for p in env.iterdir():
        if p.name != "Dockerfile":
            (shutil.copytree if p.is_dir() else shutil.copy2)(p, dest / p.name)
    shutil.copytree(td / "tests", dest / "hidden")
    for n in ("main.py", "start.sh", "proxy.py", "as-agent", "buildspec.yml"):
        shutil.copy2(IMG / n, dest / n)
    df = (env / "Dockerfile").read_text() + (IMG / "layer.Dockerfile").read_text()
    if TASKS[task][0] == "coding":
        shutil.copytree(IMG / "variants", dest / "variants")
        df += (IMG / "layer_variants.Dockerfile").read_text()
    (dest / "Dockerfile").write_text(df)


def image_exists(task):
    try:
        ecr.describe_images(repositoryName=Names.repo(), imageIds=[{"imageTag": task}])
        return True
    except Exception:
        return False


def build(task, module):
    t0 = time.time()
    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        build_context(task, module, d)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(d.rglob("*")):
                if p.is_file():
                    zi = zipfile.ZipInfo(str(p.relative_to(d)), (2020, 1, 1, 0, 0, 0))
                    zi.external_attr = (p.stat().st_mode & 0xFFFF) << 16
                    zi.compress_type = zipfile.ZIP_DEFLATED
                    z.writestr(zi, p.read_bytes())
    key = f"p2/{task}.zip"
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
    res = {"task": task, "status": x["buildStatus"], "secs": round(time.time() - t0), "phases": phases, "log_tail": tail}
    log(task, "build", x["buildStatus"], res["secs"], "s")
    if x["buildStatus"] != "SUCCEEDED":
        print(tail)
    return res


# ---------------------------------------------------------------- runtimes
def find_runtime(task):
    for r in ctl.list_agent_runtimes(maxResults=100)["agentRuntimes"]:
        if r["agentRuntimeName"] == runtime_name(task):
            return r
    return None


def wait_ready(rid):
    while True:
        g = ctl.get_agent_runtime(agentRuntimeId=rid)
        if g["status"] not in ("CREATING", "UPDATING"):
            return g
        time.sleep(4)


def deploy(task, platform="V1"):
    """Create the runtime (PUBLIC network, no VPC) for the task image, or point an existing one at it."""
    t0 = time.time()
    uri = f"{acct()}.dkr.ecr.{R}.amazonaws.com/{Names.repo()}:{task}"
    role = iam.get_role(RoleName=Names.runtime_role())["Role"]["Arn"]
    art = {"containerConfiguration": {"containerUri": uri}}
    net = {"networkMode": "PUBLIC"}
    ex = find_runtime(task)
    if ex:
        rid = ex["agentRuntimeId"]
        ctl.update_agent_runtime(agentRuntimeId=rid, roleArn=role, agentRuntimeArtifact=art, networkConfiguration=net,
                                 metadataConfiguration={"requireMMDSV2": True}, platformVersion=platform)
    else:
        # a role created seconds ago may not be assumable by AgentCore yet: retry on role errors
        rid = retry(lambda: ctl.create_agent_runtime(agentRuntimeName=runtime_name(task), roleArn=role, agentRuntimeArtifact=art,
                                                     networkConfiguration=net, platformVersion=platform, tags=TAGS)["agentRuntimeId"],
                    ["role", "assume", "not authorized"], f"agentcore create_runtime {task}")
        g = wait_ready(rid)
        if g["status"] == "READY":  # create has no metadataConfiguration in the API model: set IMDSv2-only right after
            ctl.update_agent_runtime(agentRuntimeId=rid, roleArn=role, agentRuntimeArtifact=art, networkConfiguration=net,
                                     metadataConfiguration={"requireMMDSV2": True}, platformVersion=platform)
    time.sleep(2)
    g = wait_ready(rid)
    res = {"task": task, "id": rid, "status": g["status"], "secs": round(time.time() - t0, 1), "reason": g.get("failureReason")}
    log(task, "runtime", res["status"], res["secs"], "s", res["reason"] or "")
    return res


_arn = {}


def runtime_arn(task):
    if task not in _arn:
        r = find_runtime(task)
        _arn[task] = ctl.get_agent_runtime(agentRuntimeId=r["agentRuntimeId"])["agentRuntimeArn"]
    return _arn[task]


# ---------------------------------------------------------------- one session
def shq(s):
    return "'" + s.replace("'", "'\"'\"'") + "'"


class Session:
    """One AgentCore session = one fresh microVM of the task's runtime. cmd() runs a command as root; agent() as `agent` in the firewall."""

    def __init__(self, task, sid=None):
        self.task, self.arn = task, runtime_arn(task)
        self.sid = sid or ("p2-" + uuid.uuid4().hex + uuid.uuid4().hex[:6])
        self.calls = 0
        self.throttles = 0

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

    def agent(self, command, timeout=300):
        return self.cmd("/opt/firewall/as-agent " + shq(command), timeout)

    def put(self, path, data: bytes, mode="0644", owner="root"):
        b = base64.b64encode(data).decode()
        self.cmd(f": > {path}.b64")
        for i in range(0, len(b), 40000):
            self.cmd(f"printf %s {b[i:i + 40000]} >> {path}.b64")
        o, c, _ = self.cmd(f"base64 -d {path}.b64 > {path} && rm -f {path}.b64 && chmod {mode} {path} && chown {owner} {path}")
        assert c == 0, o

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


CODEX_CMD = ('cd /work/repo && mkdir -p $CODEX_HOME && cp /logs/agent/config.toml $CODEX_HOME/config.toml && '
             'if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; CODEX_API_KEY= codex exec --dangerously-bypass-approvals-and-sandbox '
             '--skip-git-repo-check --model ' + CODEX_MODEL + ' --json --enable unified_exec -c web_search=disabled '
             '-c \'shell_environment_policy.exclude=["*PROXY*"]\' -- '
             '"$(cat /logs/agent/instruction.md)" 2>&1 </dev/null | tee /logs/agent/codex.txt')

TAR_CMD = "cd /logs && tar czf - agent verifier artifacts 2>/dev/null | base64 -w0"


def run_trial(task, module, job_dir, mode="codex", patch=None, script=None, label=None, files=None, creds_mode="scoped",
              agent_timeout=1500, verifier_timeout=1500, variant=()):
    """The whole trial in one session. mode: codex | none (nothing) | patch (apply `patch` bytes as the agent) | script (run shell `script` as the agent).
    variant: subset of {"par", "hard"} (coding tasks only): the verifier runs image/variants/orchestrate.sh instead of tests/test.sh.
    Returns the result dict; files are written to job_dir/<trial>/ in the layout of Harbor jobs (result.json, verifier/, agent/)."""
    kind, ds, tname = TASKS[task]
    td = task_dir(task, module)
    trial = f"{tname}__{uuid.uuid4().hex[:7]}"
    T = {"t0": time.time()}
    s = Session(task)
    variant = tuple(sorted(variant)) if kind == "coding" else ()
    res = {"task_name": None, "trial_name": trial, "label": label, "mode": mode, "task": task, "variant": list(variant), "timings": {}, "errors": []}
    tn = re.search(r'^name = "([^"]+)"', (td / "task.toml").read_text(), re.M).group(1)
    res["task_name"] = tn
    tm = res["timings"]
    out = pathlib.Path(job_dir) / trial
    try:
        o, c, secs = s.cmd("cat /run/fw.status; id -u; date +%s.%N; df -BM --output=avail / | tail -1")
        tm["session_start"] = secs
        res["start_probe"] = o.strip().replace("\n", " | ")
        if not o.startswith("OK"):
            raise RuntimeError("firewall not ready: " + o[:200])
        # --- agent phase: proxy allowlist = bedrock-mantle only; scoped credentials
        t = time.time()
        extra = {"CODEX_HOME": "/logs/agent/codex-home", "AWS_REGION": R, "AWS_DEFAULT_REGION": R}
        if mode == "codex":
            extra.update(agent_creds() if creds_mode == "scoped" else {})
        set_phase(s, HOST_MANTLE, extra)
        s.put("/logs/agent/instruction.md", (td / "instruction.md").read_bytes(), "0644", "agent")
        s.put("/logs/agent/config.toml", (module / "codex-bedrock.toml").read_bytes(), "0644", "agent")
        for path, data in (files or {}).items():
            s.put(path, data, "0644", "agent")
        tm["agent_prepare"] = round(time.time() - t, 2)
        t = time.time()
        agent_out = ""
        if mode == "codex":
            agent_out, code, _ = s.agent(CODEX_CMD, timeout=agent_timeout)
            res["agent_exit"] = code
        elif mode == "patch":
            s.put("/logs/agent/fix.patch", patch, "0644", "agent")
            agent_out, code, _ = s.agent("cd /work/repo && git apply --whitespace=nowarn /logs/agent/fix.patch && echo applied", 120)
            res["agent_exit"] = code
        elif mode == "script":
            agent_out, code, _ = s.agent(script, 300)
            res["agent_exit"] = code
        tm["agent_loop"] = round(time.time() - t, 2)
        res["agent_tail"] = agent_out[-600:] if mode != "codex" else ""
        # --- stop the agent, switch the firewall to the verifier phase
        t = time.time()
        s.cmd("pkill -KILL -u agent; sleep 0.3; pgrep -u agent | wc -l")
        vextra = {"AWS_REGION": R, "AWS_DEFAULT_REGION": R}
        vextra.update(runtime_role_creds(s))
        vextra["VERIFIER_MODEL_ID" if kind == "coding" else "JUDGE_MODEL_ID"] = VERIFIER_MODEL if kind == "coding" else JUDGE_MODEL
        if kind == "review":
            vextra["PATH"] = "/opt/verifier-venv/bin:/usr/local/bin:/usr/bin:/bin"
        set_phase(s, HOST_RUNTIME, vextra)
        o, c, _ = s.cmd("cp -a /opt/hidden/tests /tests && chmod -R a+rX /tests && mkdir -p /logs/verifier && chown agent:agent /logs/verifier && echo ok")
        assert o.strip() == "ok", o
        tm["phase_switch"] = round(time.time() - t, 2)
        # --- verifier phase (as agent uid in the firewall: the grader runs agent-influenced code, so it gets no root and no direct network)
        t = time.time()
        res["verifier_start_epoch"] = time.time()
        if variant:
            flags = f"P2_PAR={int('par' in variant)} P2_HARD={int('hard' in variant)}"
            vout, vcode, _ = s.cmd(f"{flags} bash /opt/hidden/variants/orchestrate.sh", timeout=verifier_timeout)
            res["verifier_out"] = vout[-600:]
        else:
            vcmd = "cd /work/repo && bash /tests/test.sh" if kind == "coding" else "cd /work/repo && sh /tests/test.sh"
            vout, vcode, _ = s.agent(vcmd, timeout=verifier_timeout)
        tm["verifier"] = round(time.time() - t, 2)
        res["verifier_exit"] = vcode
        # --- read back
        t = time.time()
        o, _, _ = s.cmd("find /logs/verifier -type f -printf '%T@ %s %p\\n' | sort -n")
        res["verifier_files_mtime"] = o
        o, _, _ = s.cmd(TAR_CMD, timeout=120)
        tm["readback"] = round(time.time() - t, 2)
        o2, _, _ = s.cmd("df -BM --output=used / | tail -1; free -m | sed -n 2p; ls /logs/agent/codex-home/sessions 2>/dev/null | head -1")
        res["end_probe"] = o2.strip().replace("\n", " | ")
        out.mkdir(parents=True, exist_ok=True)
        try:
            raw = base64.b64decode(o.strip())
            tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz").extractall(out)
        except Exception as e:
            res["errors"].append(f"readback: {e!r}")
    except Exception as e:
        res["errors"].append(repr(e)[:400])
        out.mkdir(parents=True, exist_ok=True)
    res["throttles"] = s.throttles
    res["calls"] = s.calls
    tm["total"] = round(time.time() - T["t0"], 2)
    res["stop"] = s.stop()
    (out / "result.json").write_text(json.dumps(res, indent=1))
    rf = out / "verifier" / "reward.json"
    res["reward"] = json.loads(rf.read_text()) if rf.exists() else None
    (out / "result.json").write_text(json.dumps(res, indent=1))
    return res


def parse_specs(specs):
    out = []
    for sp in specs:
        t, _, n = sp.partition(":")
        assert t in TASKS, t
        out += [t] * int(n or 1)
    return out


def summarize(rs, job, wall):
    """One table of the trials: reward fields, time. Written to <job>/summary.md and summary.json."""
    keys = ("reward", "correct", "fail_to_pass", "validation_score", "rubric_score", "relative_taste", "coverage_required", "coverage_all")
    lines = ["| task | trial | total s | agent s | verifier s | " + " | ".join(keys) + " | errors |", "|" + "---|" * (6 + len(keys))]
    for r in rs:
        w = r.get("reward") or {}
        t = r["timings"]
        lines.append(f"| {r['task']} | {r['trial_name'][-7:]} | {t.get('total')} | {t.get('agent_loop')} | {t.get('verifier')} | "
                     + " | ".join(str(w.get(k, "")) for k in keys) + f" | {'; '.join(r['errors'])[:80]} |")
    lines.append(f"\nbatch wall clock {wall:.0f} s, {len(rs)} trials, jobs dir {job}")
    txt = "\n".join(lines)
    pathlib.Path(job).mkdir(parents=True, exist_ok=True)
    (pathlib.Path(job) / "summary.md").write_text(txt + "\n")
    (pathlib.Path(job) / "summary.json").write_text(json.dumps([{k: r.get(k) for k in ("task", "trial_name", "variant", "timings", "reward", "errors")} for r in rs], indent=1))
    print(txt)


def run_batch(tasks, module, job, mode="codex", patch=None, variant=()):
    t0 = time.time()
    with cf.ThreadPoolExecutor(len(tasks)) as ex:
        rs = list(ex.map(lambda t: run_trial(t, module, job, mode=mode, patch=patch, variant=variant), tasks))
    summarize(rs, job, time.time() - t0)
    return rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["all", "bootstrap", "policies", "setup", "build", "deploy", "run", "cleanup", "status"])
    ap.add_argument("tasks", nargs="*")
    ap.add_argument("--module")
    ap.add_argument("--job", default=None)
    ap.add_argument("--agent", default="codex")
    ap.add_argument("--platform", default="V1")
    ap.add_argument("--prefix", default=None, help="name prefix of every resource (default evals-p2b-, or $P2_PREFIX)")
    ap.add_argument("--variant", default="", help="par, hard or par,hard (coding tasks; default none = the module's test.sh)")
    ap.add_argument("--rebuild", action="store_true", help="all: rebuild images even if present")
    a = ap.parse_args()
    if a.prefix:
        Names.prefix = a.prefix
    m = module_dir(a.module, required=False)
    if m:
        load_models(m)
    variant = tuple(v for v in a.variant.split(",") if v)
    assert set(variant) <= {"par", "hard"}, variant
    if a.cmd in ("bootstrap", "setup"):
        print(json.dumps(bootstrap(), indent=1))
    elif a.cmd == "policies":
        for n, p in (("codebuild role", policy_build()), ("grader (runtime) role", policy_runtime()), ("agent role", policy_agent())):
            print("###", n); print(mask(json.dumps(p, indent=1)))
    elif a.cmd in ("build", "deploy"):
        tasks = a.tasks or list(TASKS)
        with cf.ThreadPoolExecutor(len(tasks)) as ex:
            if a.cmd == "build":
                rs = list(ex.map(lambda t: build(t, module_dir(a.module)), tasks))
            else:
                rs = list(ex.map(lambda t: deploy(t, a.platform), tasks))
        print(json.dumps(rs, indent=1))
    elif a.cmd == "run":
        module = module_dir(a.module)
        tasks = parse_specs(a.tasks or ["pr4121"])
        job = pathlib.Path(a.job or (HERE / "jobs" / time.strftime("%Y%m%d-%H%M%S")))
        patch, mode = None, a.agent
        if mode.startswith("patch:"):
            patch, mode = pathlib.Path(mode[6:]).read_bytes(), "patch"
        elif mode == "empty":
            mode = "none"
        run_batch(tasks, module, job, mode, patch, variant)
    elif a.cmd == "all":
        t_all = time.time()
        module = module_dir(a.module)
        tasks = parse_specs(a.tasks or ["pr4121"])
        uniq = sorted(set(tasks))
        miss = missing()
        if miss:
            log("missing, running bootstrap:", ", ".join(mask(m) for m in miss))
            bootstrap()
        todo = [t for t in uniq if a.rebuild or not image_exists(t)]
        if todo:
            log("building", todo)
            with cf.ThreadPoolExecutor(len(todo)) as ex:
                bs = list(ex.map(lambda t: build(t, module), todo))
            bad = [b["task"] for b in bs if b["status"] != "SUCCEEDED"]
            if bad:
                sys.exit(f"build failed for {bad}")
        need = [t for t in uniq if t in todo or not find_runtime(t)]
        if need:
            log("deploying", need)
            with cf.ThreadPoolExecutor(len(need)) as ex:
                ds = list(ex.map(lambda t: deploy(t, a.platform), need))
            bad = [d["task"] for d in ds if d["status"] != "READY"]
            if bad:
                sys.exit(f"deploy failed for {bad}: {[d['reason'] for d in ds]}")
        job = pathlib.Path(a.job or (HERE / "jobs" / time.strftime("%Y%m%d-%H%M%S")))
        log("running", tasks)
        run_batch(tasks, module, job, "codex", None, variant)
        log(f"all done in {time.time() - t_all:.0f}s (including setup)")
    elif a.cmd == "status":
        for t in TASKS:
            r = find_runtime(t)
            print(t, r and r["status"])
    elif a.cmd == "cleanup":
        for t in TASKS:
            r = find_runtime(t)
            if r:
                ctl.delete_agent_runtime(agentRuntimeId=r["agentRuntimeId"]); print("deleting runtime", t)
        try:
            ecr.delete_repository(repositoryName=Names.repo(), force=True); print("deleted ecr repository", Names.repo())
        except ecr.exceptions.RepositoryNotFoundException:
            pass
        try:
            cb.delete_project(name=Names.project()); print("deleted codebuild project")
        except Exception as e:
            print("codebuild project:", repr(e)[:100])
        for rn, pn in ((Names.build_role(), "build"), (Names.runtime_role(), "grader"), (Names.agent_role(), "codex-model-only")):
            try:
                iam.delete_role_policy(RoleName=rn, PolicyName=pn)
                iam.delete_role(RoleName=rn); print("deleted role", rn)
            except iam.exceptions.NoSuchEntityException:
                pass
        print("not deleted here: S3 bucket", mask(bucket()), "and the log groups; run scripts/teardown.sh (prints the commands) or see agentcore/TESTER-GUIDE.md")


if __name__ == "__main__":
    main()
