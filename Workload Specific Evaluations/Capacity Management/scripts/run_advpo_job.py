#!/usr/bin/env python3
"""
Run the Advanced Prompt Optimization job that produces this module's optimized prompts.

This script is an AUTHORING TOOL, not part of the workshop flow. An AdvPO job is
asynchronous and takes 15 minutes to several hours, which does not fit inside a
notebook cell, so the job is run once here and its real output is committed to
``data/advpo_results.jsonl``. The notebook then shows the same setup code and
loads the committed results.

Requires boto3 >= 1.43 (that is when ``create_advanced_prompt_optimization_job``
landed) and credentials with permission to manage S3, IAM, Lambda, and Bedrock.

Usage
-----
    python run_advpo_job.py build       # build the input JSONL locally, no AWS calls
    python run_advpo_job.py setup       # create S3 bucket, IAM role, Lambda evaluator
    python run_advpo_job.py submit      # upload input and start the job
    python run_advpo_job.py wait        # poll until the job finishes
    python run_advpo_job.py fetch       # download results into data/
    python run_advpo_job.py all         # setup -> submit -> wait -> fetch
    python run_advpo_job.py teardown    # remove the Lambda and IAM role
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
import zipfile

import boto3
from botocore.exceptions import ClientError

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE = os.path.dirname(HERE)
DATA = os.path.join(MODULE, "data")
sys.path.insert(0, MODULE)

import banking77 as b77                     # noqa: E402
import model_config as mc                   # noqa: E402

REGION = mc.REGION
STATE_PATH = os.path.join(HERE, ".advpo_state.json")

LAMBDA_NAME = "capacity-mgmt-banking77-evaluator"
ROLE_NAME = "capacity-mgmt-advpo-lambda-role"
METRIC_LABEL = "banking77_exact_and_terse"
TEMPLATE_ID_FREE = "banking77-free"
TEMPLATE_ID_SCOPED = "banking77-scoped"
JOB_PREFIX = "capacity-mgmt-banking77"

INPUT_KEY = "advpo/input/banking77_templates.jsonl"
OUTPUT_PREFIX = "advpo/output/"

# Only maxTokens is set. Claude Sonnet 5 and GPT-5.6 Luna reject `temperature`
# outright, so a portfolio-wide inference config has to leave it off.
MODEL_MAX_TOKENS = {
    mc.NOVA_2_LITE: 1024,
    mc.CLAUDE_HAIKU_45: 1024,
    mc.CLAUDE_SONNET_5: 1024,
    mc.GPT_OSS_120B: 2048,      # emits reasoning tokens before its answer
    mc.GPT_56_LUNA: 1024,
}


# ---------------------------------------------------------------------------
# tiny state file so stages can run independently
# ---------------------------------------------------------------------------

def load_state() -> dict:
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {}


def save_state(**kw) -> dict:
    state = load_state()
    state.update(kw)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)
    return state


def account_id() -> str:
    return boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]


def bucket_name() -> str:
    return f"capacity-mgmt-advpo-{account_id()}-{REGION}"


# ---------------------------------------------------------------------------
# build: the AdvPO input dataset
# ---------------------------------------------------------------------------

def build_input(path: str | None = None) -> str:
    """Write the JSONL input file: one line per prompt template.

    We submit a single template. The 77-label vocabulary is wrapped in
    <advpo:exclude> so the optimizer rewrites the instructions but cannot
    paraphrase the class names, which would silently break exact-match scoring.
    """
    path = path or os.path.join(DATA, "advpo_input.jsonl")
    samples = b77.load_optimizer_samples()

    evaluation_samples = [
        {
            # inputVariables must be a list of SINGLE-key objects. Putting
            # several keys in one object fails silently.
            "inputVariables": [{"customerMessage": s["text"]}],
            "referenceResponse": s["label"],
        }
        for s in samples
    ]

    # Two templates in one job, so we can compare unrestricted optimization
    # against optimization scoped with <advpo:optimize>. A job accepts up to 10
    # templates and up to 5 models, and every template is optimized for every
    # model.
    templates = [
        (TEMPLATE_ID_FREE, b77.ADVPO_TEMPLATE_FREE),
        (TEMPLATE_ID_SCOPED, b77.ADVPO_TEMPLATE_SCOPED),
    ]
    records = [
        {
            "version": mc.ADVPO_INPUT_VERSION,
            "templateId": template_id,
            "promptTemplate": template,
            "customEvaluationMetricLabel": METRIC_LABEL,
            "evaluationSamples": evaluation_samples,
        }
        for template_id, template in templates
    ]
    # evaluationMetricLambdaArn is added at submit time, once the ARN is known.

    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"built {path}")
    print(f"  templates         : {len(records)} (limit 10)")
    for template_id, template in templates:
        print(f"     {template_id:<34} {len(template):>5} chars  "
              f"optimize-tagged={'<advpo:optimize>' in template}")
    print(f"  evaluation samples: {len(samples)} per template (limit 100)")
    print(f"  placeholders      : {{{{customerMessage}}}}")
    return path


# ---------------------------------------------------------------------------
# setup: S3 bucket, IAM role, Lambda evaluator
# ---------------------------------------------------------------------------

def ensure_bucket() -> str:
    s3 = boto3.client("s3", region_name=REGION)
    name = bucket_name()
    try:
        s3.head_bucket(Bucket=name)
        print(f"bucket exists: s3://{name}")
        return name
    except ClientError:
        pass
    kwargs = {"Bucket": name}
    if REGION != "us-east-1":
        kwargs["CreateBucketConfiguration"] = {"LocationConstraint": REGION}
    s3.create_bucket(**kwargs)
    s3.put_public_access_block(
        Bucket=name,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True, "IgnorePublicAcls": True,
            "BlockPublicPolicy": True, "RestrictPublicBuckets": True},
    )
    s3.put_bucket_encryption(
        Bucket=name,
        ServerSideEncryptionConfiguration={"Rules": [
            {"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]},
    )
    print(f"created bucket: s3://{name} (private, SSE-S3)")
    return name


def ensure_role() -> str:
    iam = boto3.client("iam")
    trust = {"Version": "2012-10-17", "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "lambda.amazonaws.com"},
        "Action": "sts:AssumeRole"}]}
    try:
        arn = iam.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]
        print(f"role exists: {arn}")
        return arn
    except iam.exceptions.NoSuchEntityException:
        pass
    arn = iam.create_role(
        RoleName=ROLE_NAME,
        AssumeRolePolicyDocument=json.dumps(trust),
        Description="Execution role for the Capacity Management AdvPO evaluator Lambda",
    )["Role"]["Arn"]
    iam.attach_role_policy(
        RoleName=ROLE_NAME,
        PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole")
    print(f"created role: {arn}  (waiting for propagation)")
    time.sleep(12)
    return arn


def lambda_zip() -> bytes:
    """Package lambda_evaluator.py as lambda_function.py, which is the handler name."""
    src = os.path.join(HERE, "lambda_evaluator.py")
    with open(src) as f:
        code = f.read()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        info = zipfile.ZipInfo("lambda_function.py")
        info.external_attr = 0o644 << 16
        z.writestr(info, code)
    return buf.getvalue()


def ensure_lambda(role_arn: str) -> str:
    lam = boto3.client("lambda", region_name=REGION)
    payload = lambda_zip()
    try:
        lam.get_function(FunctionName=LAMBDA_NAME)
        lam.update_function_code(FunctionName=LAMBDA_NAME, ZipFile=payload)
        lam.get_waiter("function_updated_v2").wait(FunctionName=LAMBDA_NAME)
        lam.update_function_configuration(
            FunctionName=LAMBDA_NAME, Timeout=900, MemorySize=512)
        lam.get_waiter("function_updated_v2").wait(FunctionName=LAMBDA_NAME)
        arn = lam.get_function(FunctionName=LAMBDA_NAME)["Configuration"]["FunctionArn"]
        print(f"lambda updated: {arn}")
    except lam.exceptions.ResourceNotFoundException:
        for attempt in range(6):
            try:
                arn = lam.create_function(
                    FunctionName=LAMBDA_NAME,
                    Runtime="python3.12",
                    Role=role_arn,
                    Handler="lambda_function.lambda_handler",
                    Code={"ZipFile": payload},
                    Timeout=900,          # docs recommend the 15 min maximum
                    MemorySize=512,
                    Description="Exact-and-terse scorer for BANKING77 intent classification",
                )["FunctionArn"]
                break
            except ClientError as exc:
                # IAM role propagation is eventually consistent.
                if "cannot be assumed" in str(exc) and attempt < 5:
                    print("   role not yet assumable, retrying...")
                    time.sleep(10)
                    continue
                raise
        lam.get_waiter("function_active_v2").wait(FunctionName=LAMBDA_NAME)
        print(f"lambda created: {arn}")

    # Bedrock must be allowed to invoke the evaluator.
    try:
        lam.add_permission(
            FunctionName=LAMBDA_NAME,
            StatementId="AllowBedrockAdvPOInvoke",
            Action="lambda:InvokeFunction",
            Principal="bedrock.amazonaws.com",
            SourceAccount=account_id(),
        )
        print("added resource policy allowing bedrock.amazonaws.com to invoke")
    except lam.exceptions.ResourceConflictException:
        print("resource policy already present")
    return arn


def smoke_test_lambda(arn: str) -> None:
    """Invoke the deployed Lambda with a known payload before trusting it in a job."""
    lam = boto3.client("lambda", region_name=REGION)
    event = {"preds": ["card_arrival", "The intent is card_linking here", "gibberish"],
             "golds": ["card_arrival", "card_linking", "card_arrival"]}
    resp = lam.invoke(FunctionName=arn, Payload=json.dumps(event).encode())
    body = json.loads(resp["Payload"].read())
    print(f"lambda smoke test -> {body}")
    expected = [1.0, 0.6, 0.0]
    assert body.get("scores") == expected, f"expected {expected}, got {body.get('scores')}"
    print("lambda smoke test PASSED")


def stage_setup() -> dict:
    bucket = ensure_bucket()
    role = ensure_role()
    fn = ensure_lambda(role)
    smoke_test_lambda(fn)
    return save_state(bucket=bucket, role_arn=role, lambda_arn=fn)


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------

def stage_submit() -> dict:
    state = load_state()
    bucket = state.get("bucket") or bucket_name()
    lambda_arn = state["lambda_arn"]

    local = build_input()
    with open(local) as f:
        records = [json.loads(line) for line in f if line.strip()]
    for record in records:
        record["evaluationMetricLambdaArn"] = lambda_arn
    payload = "".join(json.dumps(r) + "\n" for r in records).encode()
    with open(local, "wb") as f:
        f.write(payload)

    boto3.client("s3", region_name=REGION).put_object(
        Bucket=bucket, Key=INPUT_KEY, Body=payload)
    print(f"uploaded s3://{bucket}/{INPUT_KEY} ({len(payload):,} bytes)")

    bedrock = boto3.client("bedrock", region_name=REGION)
    job_name = f"{JOB_PREFIX}-{int(time.time())}"
    # AdvPO and Converse do not accept the same model identifiers. See
    # ADVPO_MODEL_IDS in model_config.py for the mapping and why it is needed.
    model_configs = [
        {"modelId": mc.ADVPO_MODEL_IDS[m],
         "inferenceConfig": {"maxTokens": MODEL_MAX_TOKENS[m]}}
        for m in mc.CANDIDATE_MODELS
    ]
    print(f"submitting job {job_name} for {len(model_configs)} models:")
    for runtime_id, c in zip(mc.CANDIDATE_MODELS, model_configs):
        note = "" if c["modelId"] == runtime_id else f"   (runtime id: {runtime_id})"
        print(f"   {c['modelId']:<46} maxTokens={c['inferenceConfig']['maxTokens']}{note}")

    resp = bedrock.create_advanced_prompt_optimization_job(
        jobName=job_name,
        jobDescription="BANKING77 intent classification, optimized per model "
                       "for the Capacity Management workshop module",
        inputConfig={"s3Uri": f"s3://{bucket}/{INPUT_KEY}"},
        outputConfig={"s3Uri": f"s3://{bucket}/{OUTPUT_PREFIX}"},
        modelConfigurations=model_configs,
    )
    job_arn = resp["jobArn"]
    print(f"\njob submitted\n  arn: {job_arn}")
    return save_state(job_arn=job_arn, job_name=job_name,
                      submitted_at=time.time())


# ---------------------------------------------------------------------------
# wait / fetch
# ---------------------------------------------------------------------------

# The API returns PascalCase statuses ("InProgress", "Completed", "Failed"),
# not the SCREAMING_CASE the user guide implies, so compare case-insensitively.
TERMINAL = {"completed", "failed", "stopped", "partiallycompleted"}


def _is_terminal(status: str) -> bool:
    return status.replace("_", "").lower() in TERMINAL


def stage_wait(poll_s: int = 60, max_hours: float = 6.0) -> str:
    state = load_state()
    job_arn = state["job_arn"]
    bedrock = boto3.client("bedrock", region_name=REGION)
    t0 = time.time()
    last = None
    while True:
        resp = bedrock.get_advanced_prompt_optimization_job(jobIdentifier=job_arn)
        status = resp["jobStatus"]
        mins = (time.time() - t0) / 60
        if status != last:
            print(f"[{mins:6.1f} min] status: {status}")
            last = status
        else:
            print(f"[{mins:6.1f} min] still {status}", flush=True)
        if _is_terminal(status):
            if resp.get("failureMessage"):
                print(f"failureMessage: {resp['failureMessage']}")
            save_state(final_status=status,
                       failure_message=resp.get("failureMessage"))
            return status
        if mins / 60 > max_hours:
            print(f"giving up after {max_hours}h; job is still {status}")
            return status
        time.sleep(poll_s)


def stage_fetch() -> str:
    state = load_state()
    bucket = state.get("bucket") or bucket_name()
    job_id = state["job_arn"].split("/")[-1]
    key = f"{OUTPUT_PREFIX}{job_id}/advanced_prompt_optimization_results.jsonl"
    s3 = boto3.client("s3", region_name=REGION)

    print(f"reading s3://{bucket}/{key}")
    try:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode()
    except ClientError as exc:
        print(f"could not read expected key ({exc}); listing job output instead")
        listing = s3.list_objects_v2(Bucket=bucket,
                                     Prefix=f"{OUTPUT_PREFIX}{job_id}/")
        for obj in listing.get("Contents", []):
            print(f"   {obj['Key']}  {obj['Size']:,} bytes")
        raise

    out = os.path.join(DATA, "advpo_results.jsonl")
    with open(out, "w") as f:
        f.write(body)
    print(f"wrote {out} ({len(body):,} bytes)")

    for line in body.strip().split("\n"):
        result = json.loads(line)
        print(f"\ntemplate: {result.get('promptTemplateId')}")
        for opt in result.get("promptOptimizationResults", []):
            print(f"   {opt.get('modelId'):<48} status={opt.get('status')}")
    return out


def stage_teardown() -> None:
    """Remove the Lambda and IAM role. The S3 bucket is left in place."""
    lam = boto3.client("lambda", region_name=REGION)
    iam = boto3.client("iam")
    try:
        lam.delete_function(FunctionName=LAMBDA_NAME)
        print(f"deleted lambda {LAMBDA_NAME}")
    except lam.exceptions.ResourceNotFoundException:
        print("lambda already gone")
    try:
        iam.detach_role_policy(
            RoleName=ROLE_NAME,
            PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole")
        iam.delete_role(RoleName=ROLE_NAME)
        print(f"deleted role {ROLE_NAME}")
    except iam.exceptions.NoSuchEntityException:
        print("role already gone")
    print(f"NOTE: bucket s3://{bucket_name()} left in place (holds job input/output)")


# ---------------------------------------------------------------------------

STAGES = {
    "build": lambda: build_input(),
    "setup": stage_setup,
    "submit": stage_submit,
    "wait": stage_wait,
    "fetch": stage_fetch,
    "teardown": stage_teardown,
}


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage == "all":
        stage_setup()
        stage_submit()
        status = stage_wait()
        if status.replace("_", "").lower() in ("completed", "partiallycompleted"):
            stage_fetch()
        else:
            print(f"job ended as {status}; nothing to fetch")
        return
    if stage not in STAGES:
        print(f"unknown stage {stage!r}; choose from {list(STAGES)} or 'all'")
        sys.exit(2)
    STAGES[stage]()


if __name__ == "__main__":
    main()
