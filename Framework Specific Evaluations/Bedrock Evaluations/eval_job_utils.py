"""Helpers for the Bedrock Evaluations notebook.

They create and remove the AWS resources an evaluation job needs (an S3 bucket and
an IAM service role), wait for a job to finish, and read its results from S3.
Every resource name starts with RESOURCE_PREFIX, so cleanup() removes only what
the notebook created.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import boto3
from botocore.exceptions import ClientError

from evaluation_config import RESOURCE_PREFIX


@dataclass
class JobResources:
    region: str
    account_id: str
    bucket: str
    role_name: str
    role_arn: str


def resource_names(region: str) -> tuple[str, str, str]:
    """Bucket name, role name and account ID for this account and Region."""
    account_id = boto3.client("sts", region_name=region).get_caller_identity()["Account"]
    bucket = f"{RESOURCE_PREFIX}-{account_id}-{region}"
    role_name = f"{RESOURCE_PREFIX}-bedrock-eval-{region}"
    return bucket, role_name, account_id


def _trust_policy(account_id: str, region: str) -> dict:
    """Lets only Bedrock evaluation jobs in this account and Region assume the role."""
    return {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "bedrock.amazonaws.com"},
            "Action": "sts:AssumeRole",
            "Condition": {
                "StringEquals": {"aws:SourceAccount": account_id},
                "ArnLike": {"aws:SourceArn": f"arn:aws:bedrock:{region}:{account_id}:evaluation-job/*"},
            },
        }],
    }


def _permissions_policy(account_id: str, bucket: str) -> dict:
    """The S3 and Bedrock actions an LLM-as-a-judge job needs, scoped to the workshop bucket.

    A cross-Region inference profile can route a request to any Region in its
    geography, so the model resources use a wildcard Region.
    """
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "ReadDatasetWriteResults",
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject", "s3:PutObject", "s3:ListBucket", "s3:GetBucketLocation",
                    "s3:AbortMultipartUpload", "s3:ListBucketMultipartUploads",
                ],
                "Resource": [f"arn:aws:s3:::{bucket}", f"arn:aws:s3:::{bucket}/*"],
                "Condition": {"StringEquals": {"aws:ResourceAccount": account_id}},
            },
            {
                "Sid": "InvokeEvaluatorModel",
                "Effect": "Allow",
                "Action": ["bedrock:InvokeModel", "bedrock:CreateModelInvocationJob", "bedrock:StopModelInvocationJob"],
                "Resource": [
                    "arn:aws:bedrock:*::foundation-model/*",
                    f"arn:aws:bedrock:*:{account_id}:inference-profile/*",
                ],
            },
        ],
    }


def create_resources(region: str) -> JobResources:
    """Create (or reuse) the workshop's S3 bucket and evaluation service role."""
    bucket, role_name, account_id = resource_names(region)
    s3 = boto3.client("s3", region_name=region)
    try:
        if region == "us-east-1":
            s3.create_bucket(Bucket=bucket)
        else:
            s3.create_bucket(Bucket=bucket, CreateBucketConfiguration={"LocationConstraint": region})
        print(f"Created bucket s3://{bucket}")
    except ClientError as err:
        if err.response["Error"]["Code"] != "BucketAlreadyOwnedByYou":
            raise
        print(f"Reusing bucket s3://{bucket}")
    s3.put_public_access_block(Bucket=bucket, PublicAccessBlockConfiguration={
        "BlockPublicAcls": True, "IgnorePublicAcls": True, "BlockPublicPolicy": True, "RestrictPublicBuckets": True,
    })

    iam = boto3.client("iam")
    try:
        role_arn = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(_trust_policy(account_id, region)),
            Description="Service role for the GenAI evaluations workshop Bedrock evaluation jobs",
            Tags=[{"Key": "project", "Value": RESOURCE_PREFIX}],
        )["Role"]["Arn"]
        print(f"Created role {role_name}")
        created = True
    except ClientError as err:
        if err.response["Error"]["Code"] != "EntityAlreadyExists":
            raise
        role_arn = iam.get_role(RoleName=role_name)["Role"]["Arn"]
        print(f"Reusing role {role_name}")
        created = False
    iam.put_role_policy(
        RoleName=role_name,
        PolicyName="bedrock-evaluation-job",
        PolicyDocument=json.dumps(_permissions_policy(account_id, bucket)),
    )
    if created:
        print("Waiting 15 seconds for the new role to reach every Region...")
        time.sleep(15)  # IAM is eventually consistent; a job created at once can fail to assume the role
    return JobResources(region, account_id, bucket, role_name, role_arn)


def upload_jsonl(resources: JobResources, key: str, records: list[dict]) -> str:
    """Write records as JSON Lines to the workshop bucket and return the S3 URI."""
    body = "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n"
    boto3.client("s3", region_name=resources.region).put_object(
        Bucket=resources.bucket, Key=key, Body=body.encode("utf-8"), ContentType="application/jsonl"
    )
    return f"s3://{resources.bucket}/{key}"


def wait_for_job(bedrock, job_arn: str, poll_seconds: int = 30, timeout_seconds: int = 3600) -> dict:
    """Poll an evaluation job until it leaves InProgress, printing each status change."""
    start, last = time.time(), None
    while True:
        job = bedrock.get_evaluation_job(jobIdentifier=job_arn)
        status = job["status"]
        if status != last:
            print(f"{int(time.time() - start):>5}s  {status}")
            last = status
        if status not in ("InProgress", "Stopping"):
            return job
        if time.time() - start > timeout_seconds:
            raise TimeoutError(f"Job still {status} after {timeout_seconds} seconds: {job_arn}")
        time.sleep(poll_seconds)


def read_job_results(resources: JobResources, output_s3_uri: str, job_arn: str) -> list[dict]:
    """Return every record from the job's output .jsonl files.

    The job writes under <output prefix>/<job name>/<job id>/. This reads every
    .jsonl file below the output prefix whose key contains the job ID.
    """
    s3 = boto3.client("s3", region_name=resources.region)
    job_id = job_arn.rsplit("/", 1)[-1]
    prefix = output_s3_uri.split(f"s3://{resources.bucket}/", 1)[1]
    records = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=resources.bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if job_id in key and key.endswith(".jsonl"):
                text = s3.get_object(Bucket=resources.bucket, Key=key)["Body"].read().decode("utf-8")
                records.extend(json.loads(line) for line in text.splitlines() if line.strip())
    return records


def scores_by_metric(record: dict) -> dict[str, dict]:
    """Map metric name to {"result": score, "explanation": text} for one output record."""
    out = {}
    for score in record.get("automatedEvaluationResult", {}).get("scores", []):
        explanation = ""
        details = score.get("evaluatorDetails") or []
        if details:
            explanation = details[0].get("explanation", "")
        out[score["metricName"]] = {"result": score.get("result"), "explanation": explanation}
    return out


def cleanup(region: str, bedrock=None) -> None:
    """Delete the workshop's evaluation jobs, S3 bucket (and its objects), and IAM role."""
    bucket, role_name, _ = resource_names(region)
    bedrock = bedrock or boto3.client("bedrock", region_name=region)

    job_arns = []
    for page in bedrock.get_paginator("list_evaluation_jobs").paginate(nameContains=RESOURCE_PREFIX):
        job_arns += [j["jobArn"] for j in page["jobSummaries"] if j["status"] not in ("Deleting",)]
    for i in range(0, len(job_arns), 25):
        bedrock.batch_delete_evaluation_job(jobIdentifiers=job_arns[i:i + 25])
    print(f"Deleted {len(job_arns)} evaluation job(s)")

    s3 = boto3.resource("s3", region_name=region)
    try:
        s3.Bucket(bucket).objects.all().delete()
        s3.Bucket(bucket).delete()
        print(f"Deleted bucket s3://{bucket}")
    except ClientError as err:
        if err.response["Error"]["Code"] != "NoSuchBucket":
            raise
        print(f"Bucket s3://{bucket} does not exist")

    iam = boto3.client("iam")
    try:
        for name in iam.list_role_policies(RoleName=role_name)["PolicyNames"]:
            iam.delete_role_policy(RoleName=role_name, PolicyName=name)
        iam.delete_role(RoleName=role_name)
        print(f"Deleted role {role_name}")
    except ClientError as err:
        if err.response["Error"]["Code"] != "NoSuchEntity":
            raise
        print(f"Role {role_name} does not exist")
