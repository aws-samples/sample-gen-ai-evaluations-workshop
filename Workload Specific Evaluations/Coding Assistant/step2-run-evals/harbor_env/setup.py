#!/usr/bin/env python3
"""AWS setup for harbor_env: bootstrap, image prebuilds, and teardown.

  setup.py bootstrap --agent-model ID...     IAM roles, ECR, S3, CodeBuild; the agent role may call only these models
  setup.py policies  --agent-model ID...     print that policy
  setup.py prebuild  TASK_DIR...             build and deploy the images of the tasks
  setup.py resources                         list every AWS resource of the prefix
  setup.py teardown                          delete them
  setup.py stop-sessions TRIAL_DIR...        stop the sessions of trials that hung

Prefix: --prefix, else $P2_PREFIX, else evals-hb-.
"""
import argparse, json, os, pathlib, re, sys, time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import agentcore_aws as aws  # noqa: E402
from botocore.exceptions import ClientError  # noqa: E402

AGENT_POLICY_NAME = "harbor-agent-models"


def model_arns(model):
    """An inference profile and its models, or one on-demand model (for example moonshotai.kimi-k2.5)."""
    try:
        p = aws.bedrock.get_inference_profile(inferenceProfileIdentifier=model)
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise
        return [aws.bedrock.get_foundation_model(modelIdentifier=model)["modelDetails"]["modelArn"]]
    return [p["inferenceProfileArn"]] + [m["modelArn"] for m in p["models"]]


def is_mantle_model(model):
    return model.startswith("openai.")


def policy_agent_models(models):
    """Only the named models: OpenAI models on bedrock-mantle, every other model through bedrock-runtime."""
    if not models:
        raise SystemExit("name every model the agents may call: --agent-model ID (for example openai.gpt-6-sol)")
    mantle = [m for m in models if is_mantle_model(m)]
    arns = [a for m in models if not is_mantle_model(m) for a in model_arns(m)]
    statements = []
    if mantle:
        statements.append({"Effect": "Allow", "Action": "bedrock-mantle:CreateInference", "Resource": "*",
                           "Condition": {"StringEquals": {"bedrock-mantle:Model": mantle}}})
    if arns:
        statements.append({"Effect": "Allow", "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
                           "Resource": list(dict.fromkeys(arns))})
    return {"Version": "2012-10-17", "Statement": statements}


def task_env_dirs(task_dir):
    from harbor.models.task.task import Task
    from harbor.models.task.verifier_mode import VerifierEnvironmentMode, resolve_task_verifier_mode, resolve_verifier_environment_definition
    t = Task(task_dir)
    out = [(t.paths.environment_dir, False)]
    if resolve_task_verifier_mode(t.config) == VerifierEnvironmentMode.SEPARATE:
        out.append((resolve_verifier_environment_definition(t.config, t.paths).directory, True))
    return t.short_name, out


def runtime_pattern():
    """Runtime names of this prefix only: <prefix>_<task>_<a|v><hash> (image_key), so evals-hb-2- is not evals-hb-."""
    return re.escape(aws.Names.runtime("")) + r"[A-Za-z0-9]{1,20}_[av][0-9a-f]{10}"


def own_runtime(name):
    return re.fullmatch(runtime_pattern(), name) is not None


def own_log_group(name):
    return name == aws.Names.loggroup() or \
        re.fullmatch("/aws/bedrock-agentcore/runtimes/" + runtime_pattern() + r"-[A-Za-z0-9]+-[A-Za-z0-9_]+", name) is not None


def runtimes():
    out, kw = [], {}
    while True:
        r = aws.ctl.list_agent_runtimes(maxResults=100, **kw)
        out += [x for x in r["agentRuntimes"] if own_runtime(x["agentRuntimeName"])]
        if not r.get("nextToken"):
            return out
        kw = {"nextToken": r["nextToken"]}


def roles():
    names = (aws.Names.build_role(), aws.Names.runtime_role(), aws.Names.agent_role())
    return [r["RoleName"] for page in aws.iam.get_paginator("list_roles").paginate() for r in page["Roles"] if r["RoleName"] in names]


def log_groups():
    out = []
    for pre in (aws.Names.loggroup(), "/aws/bedrock-agentcore/runtimes/" + aws.Names.runtime("")):
        for page in aws.logs.get_paginator("describe_log_groups").paginate(logGroupNamePrefix=pre):
            out += [g["logGroupName"] for g in page["logGroups"] if own_log_group(g["logGroupName"])]
    return list(dict.fromkeys(out))


def bucket_exists():
    try:
        aws.s3.head_bucket(Bucket=aws.bucket())
        return True
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("404", "NoSuchBucket", "NotFound"):
            return False
        raise


def resources():
    rep = {"runtimes": [f'{r["agentRuntimeName"]} {r["status"]}' for r in runtimes()], "roles": roles(), "log_groups": log_groups()}
    try:
        imgs = [d for p in aws.ecr.get_paginator("describe_images").paginate(repositoryName=aws.Names.repo()) for d in p["imageDetails"]]
        rep["ecr"] = [f"{aws.Names.repo()} ({len(imgs)} images)"] + \
            [f'  {",".join(d.get("imageTags", ["-"]))} {d["imageSizeInBytes"] // 2**20} MB compressed' for d in imgs]
    except aws.ecr.exceptions.RepositoryNotFoundException:
        rep["ecr"] = []
    rep["codebuild"] = [p["name"] for p in aws.cb.batch_get_projects(names=[aws.Names.project()])["projects"]]
    rep["s3"] = [aws.mask(aws.bucket())] if bucket_exists() else []
    return rep


def teardown(wait_secs=900):
    assert aws.Names.prefix.startswith("evals-"), aws.Names.prefix
    failed = []

    def attempt(what, fn):
        try:
            fn()
            aws.log("deleted", aws.mask(what))
        except Exception as e:
            failed.append(what)
            aws.log(f"FAILED {aws.mask(what)}: {type(e).__name__}: {aws.mask(str(e))[:200]}")

    for r in runtimes():
        if r["status"] != "DELETING":
            attempt(f'runtime {r["agentRuntimeName"]}', lambda r=r: aws.ctl.delete_agent_runtime(agentRuntimeId=r["agentRuntimeId"]))

    def del_repo():
        try:
            aws.ecr.delete_repository(repositoryName=aws.Names.repo(), force=True)
        except aws.ecr.exceptions.RepositoryNotFoundException:
            pass
    attempt(f"ecr repository {aws.Names.repo()}", del_repo)

    def del_project():
        if aws.cb.batch_get_projects(names=[aws.Names.project()])["projects"]:
            aws.cb.delete_project(name=aws.Names.project())
    attempt(f"codebuild project {aws.Names.project()}", del_project)

    def del_role(name):
        for pn in aws.iam.list_role_policies(RoleName=name)["PolicyNames"]:
            aws.iam.delete_role_policy(RoleName=name, PolicyName=pn)
        for ap in aws.iam.list_attached_role_policies(RoleName=name)["AttachedPolicies"]:
            aws.iam.detach_role_policy(RoleName=name, PolicyArn=ap["PolicyArn"])
        aws.iam.delete_role(RoleName=name)
    for rn in roles():
        attempt(f"iam role {rn}", lambda rn=rn: del_role(rn))

    def del_bucket():
        b = aws.bucket()
        if not bucket_exists():
            return
        for page in aws.s3.get_paginator("list_object_versions").paginate(Bucket=b):
            objs = [{"Key": o["Key"], "VersionId": o["VersionId"]} for o in page.get("Versions", []) + page.get("DeleteMarkers", [])]
            if objs:
                aws.s3.delete_objects(Bucket=b, Delete={"Objects": objs})
        aws.s3.delete_bucket(Bucket=b)
    attempt(f"s3 bucket {aws.bucket()}", del_bucket)

    for g in log_groups():
        attempt(f"log group {g}", lambda g=g: aws.logs.delete_log_group(logGroupName=g))

    t0 = time.time()  # a runtime delete takes minutes, and its log group can be written until it is gone
    while any(r["status"] == "DELETING" for r in runtimes()) and time.time() - t0 < wait_secs:
        time.sleep(15)
    for g in log_groups():
        attempt(f"log group {g}", lambda g=g: aws.logs.delete_log_group(logGroupName=g))
    return failed, resources()


def stop_sessions(trial_dirs):
    out = []
    for d in trial_dirs:
        f = pathlib.Path(d) / "agentcore.jsonl"
        if not f.exists():
            continue
        ev = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        stopped = {e["session_id"] for e in ev if e["event"] == "stop"}
        for e in ev:
            if e["event"] == "session_start" and e["session_id"] not in stopped:
                name = e["runtime"]
                assert name.startswith(aws.Names.runtime("")), name
                rid = next((r["agentRuntimeId"] for r in runtimes() if r["agentRuntimeName"] == name), None)
                if rid is None:
                    out.append({"trial": pathlib.Path(d).name, "session_id": e["session_id"], "result": "runtime gone"})
                    continue
                arn = aws.ctl.get_agent_runtime(agentRuntimeId=rid)["agentRuntimeArn"]
                try:
                    aws.dp.stop_runtime_session(agentRuntimeArn=arn, runtimeSessionId=e["session_id"], qualifier="DEFAULT")
                    res = "stopped"
                except Exception as x:
                    res = f"{type(x).__name__}: {str(x)[:150]}"
                out.append({"trial": pathlib.Path(d).name, "session_id": e["session_id"], "result": aws.mask(res)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bootstrap", "policies", "prebuild", "resources", "teardown", "stop-sessions"])
    ap.add_argument("tasks", nargs="*")
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--agent-model", action="append", default=[], help="a model the agents may call: openai.* (bedrock-mantle) or an inference profile ID")
    a = ap.parse_args()
    aws.Names.prefix = a.prefix or os.environ.get("P2_PREFIX") or "evals-hb-"
    aws.load_models()
    if a.cmd == "policies":
        print("### agent role inline policy", AGENT_POLICY_NAME)
        print(aws.mask(json.dumps(policy_agent_models(a.agent_model), indent=1)))
    elif a.cmd == "bootstrap":
        pol = policy_agent_models(a.agent_model)
        rep = aws.bootstrap(AGENT_POLICY_NAME, pol)
        rep["agent_models_policy"] = pol
        print(aws.mask(json.dumps(rep, indent=1)))
    elif a.cmd == "prebuild":
        import concurrent.futures as cf
        from harbor_env.agentcore_env import ensure_runtime, image_key
        jobs = []
        for t in a.tasks:
            short, dirs = task_env_dirs(t)
            jobs += [(d, image_key(d, short, v)) for d, v in dirs]
        with cf.ThreadPoolExecutor(len(jobs)) as ex:
            rs = list(ex.map(lambda j: ensure_runtime(j[0], j[1], log=aws.log), jobs))
        print(json.dumps(rs, indent=1))
    elif a.cmd == "resources":
        print(aws.mask(json.dumps(resources(), indent=1)))
    elif a.cmd == "stop-sessions":
        print(json.dumps(stop_sessions(a.tasks), indent=1))
    elif a.cmd == "teardown":
        failed, left = teardown()
        print(aws.mask(json.dumps({"failed": failed, "left": left}, indent=1)))
        if failed or any(left.values()):
            sys.exit("teardown incomplete: re-run it in a few minutes")


if __name__ == "__main__":
    main()
