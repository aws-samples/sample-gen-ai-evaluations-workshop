"""Provision the single AgentCore Gateway with two inference provider targets, via boto3.

One gateway, two inference targets:
  - `openai`  -> OpenAI first-party (SOURCE model, GPT-4.1), API key from an
                 AgentCore API-key credential provider.
  - `bedrock` -> Amazon Bedrock     (TARGET model, newest Sonnet), gateway IAM role.

The agent (and the notebook) then talk to ONE endpoint and select the backend purely
by the request `model` field — model-based routing, the single-seam design.

Why boto3 and not the AgentCore CLI: the GA CLI's `add gateway-target` provisions
MCP/HTTP/Runtime targets only, not inference targets. The control-plane API
(`bedrock-agentcore-control`) does expose inference targets, and boto3 >= 1.43.119
carries it. The CLI is still used elsewhere in the module (Runtime deploy, custom
evaluators, `eval ondemand simulate`).

Usage:
    python gateway/setup_gateway.py create     # role + credential provider + gateway + 2 targets
    python gateway/setup_gateway.py endpoint    # print the resolved /inference/v1 base_url
    python gateway/setup_gateway.py status      # show gateway + target status
    python gateway/setup_gateway.py teardown    # delete everything this script created

Everything created is recorded in gateway/.teardown.json so `teardown` removes
exactly what `create` made, in reverse order, and nothing leaks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

GATEWAY_DIR = Path(__file__).resolve().parent
RESOLVED_PATH = GATEWAY_DIR / ".resolved.json"
TEARDOWN_PATH = GATEWAY_DIR / ".teardown.json"
OPENAI_TARGET = GATEWAY_DIR / "target-openai.json"
BEDROCK_TARGET = GATEWAY_DIR / "target-bedrock.json"

REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
GATEWAY_NAME = "model-upgrade-gw"
OPENAI_KEY_NAME = "model-upgrade-openai-key"
ROLE_NAME = "model-upgrade-gateway-role"

acc = boto3.client("bedrock-agentcore-control", region_name=REGION)
iam = boto3.client("iam")


# --- teardown manifest ------------------------------------------------------
def _manifest() -> dict:
    return json.loads(TEARDOWN_PATH.read_text()) if TEARDOWN_PATH.exists() else {}


def _record(**kw) -> None:
    m = _manifest()
    m.update(kw)
    TEARDOWN_PATH.write_text(json.dumps(m, indent=2))


def _save_resolved(**kw) -> None:
    cur = json.loads(RESOLVED_PATH.read_text()) if RESOLVED_PATH.exists() else {}
    cur.update(kw)
    RESOLVED_PATH.write_text(json.dumps(cur, indent=2))


def inference_base_url(gateway_url: str) -> str:
    # gatewayUrl is the MCP/base URL; inference is served under /inference/v1.
    return gateway_url.rstrip("/") + "/inference/v1"


# --- IAM role ---------------------------------------------------------------
GATEWAY_TRUST = {
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
        "Action": "sts:AssumeRole",
    }],
}
GATEWAY_INLINE_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {
            # The bedrock-runtime CONNECTOR uses the native Amazon Bedrock
            # InvokeModel API (the deprecation-safe path; we do NOT use the older
            # bedrock-mantle endpoint). It needs InvokeModel + model discovery.
            # Scoped to * for this workshop role; scope to specific model/inference
            # -profile ARNs in production.
            "Sid": "BedrockInvoke",
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream",
                "bedrock:Converse",
                "bedrock:ConverseStream",
                # Read-only discovery: the connector enumerates models/profiles at
                # target activation (ListFoundationModels, ListInferenceProfiles,
                # ListProvisionedModelThroughputs, ...). List*/Get* covers them all
                # without granting any write/management action.
                "bedrock:List*",
                "bedrock:Get*",
            ],
            "Resource": "*",
        },
    ],
}


def _ensure_role() -> str:
    try:
        r = iam.create_role(
            RoleName=ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(GATEWAY_TRUST),
            Description="Model Upgrade module: AgentCore Gateway inference role (Bedrock target).",
        )
        iam.put_role_policy(RoleName=ROLE_NAME, PolicyName="bedrock-invoke",
                            PolicyDocument=json.dumps(GATEWAY_INLINE_POLICY))
        _record(role_name=ROLE_NAME)
        print(f"  created IAM role {ROLE_NAME}")
        time.sleep(10)  # let the role propagate before the gateway assumes it
        return r["Role"]["Arn"]
    except ClientError as e:
        if e.response["Error"]["Code"] == "EntityAlreadyExists":
            print(f"  reusing existing IAM role {ROLE_NAME}")
            return iam.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]
        raise


def _load_target(path: Path) -> dict:
    data = json.loads(path.read_text())
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _wait_ready(gateway_id: str, timeout: int = 180) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        g = acc.get_gateway(gatewayIdentifier=gateway_id)
        status = g["status"]
        if status == "READY":
            return status
        if status in ("FAILED", "UPDATE_UNSUCCESSFUL"):
            raise RuntimeError(f"gateway {gateway_id} status {status}: {g.get('statusReasons')}")
        time.sleep(5)
    raise TimeoutError(f"gateway {gateway_id} not READY within {timeout}s")


def create(with_openai: bool = False) -> None:
    """Provision the single Gateway. Default: Bedrock inference target only (no API
    key needed — runnable with just Bedrock access). Pass with_openai=True to also
    add the OpenAI first-party example target (requires OPENAI_API_KEY). The Azure
    example target is provisioned the same way from target-azure-openai.json; wire
    it in by analogy when you need it.
    """
    print("1/4 IAM role ...")
    role_arn = _ensure_role()

    print("2/4 gateway ...")
    g = acc.create_gateway(
        name=GATEWAY_NAME,
        roleArn=role_arn,
        authorizerType="AWS_IAM",
        protocolType="MCP",
        description="Model Upgrade module: single inference seam (Bedrock default; OpenAI/Azure optional).",
    )
    gateway_id = g["gatewayId"]
    _record(gateway_id=gateway_id)
    print(f"  created gateway {gateway_id}, waiting for READY ...")
    _wait_ready(gateway_id)

    print("3/4 Bedrock inference target (default live path) ...")
    bedrock_cfg = _load_target(BEDROCK_TARGET)
    t2 = acc.create_gateway_target(
        gatewayIdentifier=gateway_id,
        name=bedrock_cfg["name"],
        targetConfiguration=bedrock_cfg["targetConfiguration"],
        credentialProviderConfigurations=bedrock_cfg["credentialProviderConfigurations"],
    )
    _record(bedrock_target_id=t2["targetId"])
    print(f"  created target {t2['targetId']}")

    if with_openai:
        openai_key = os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            sys.exit("--with-openai needs OPENAI_API_KEY set (stored in the credential provider).")
        print("4/4 OpenAI first-party inference target (example) ...")
        try:
            cp = acc.create_api_key_credential_provider(name=OPENAI_KEY_NAME, apiKey=openai_key)
            _record(credential_provider_name=OPENAI_KEY_NAME)
            cp_arn = cp["credentialProviderArn"]
        except ClientError as e:
            if e.response["Error"]["Code"] in ("ConflictException", "ValidationException") and "exist" in str(e).lower():
                cp_arn = acc.get_api_key_credential_provider(name=OPENAI_KEY_NAME)["credentialProviderArn"]
            else:
                raise
        openai_cfg = _load_target(OPENAI_TARGET)
        openai_cfg["credentialProviderConfigurations"][0]["credentialProvider"][
            "apiKeyCredentialProvider"]["providerArn"] = cp_arn
        t1 = acc.create_gateway_target(
            gatewayIdentifier=gateway_id,
            name=openai_cfg["name"],
            targetConfiguration=openai_cfg["targetConfiguration"],
            credentialProviderConfigurations=openai_cfg["credentialProviderConfigurations"],
        )
        _record(openai_target_id=t1["targetId"])
        print(f"  created target {t1['targetId']}")
    else:
        print("4/4 skipping OpenAI/Azure example targets (Bedrock-only). Use --with-openai to add OpenAI.")

    base_url = inference_base_url(g["gatewayUrl"])
    _save_resolved(gateway_id=gateway_id, gateway_url=g["gatewayUrl"],
                   inference_base_url=base_url, region=REGION)
    print(f"\nGateway ready. Inference base_url:\n  {base_url}")


def status() -> None:
    m = _manifest()
    gid = m.get("gateway_id")
    if not gid:
        sys.exit("No gateway recorded. Run `create` first.")
    g = acc.get_gateway(gatewayIdentifier=gid)
    print("gateway:", gid, g["status"])
    for key in ("openai_target_id", "bedrock_target_id"):
        tid = m.get(key)
        if tid:
            t = acc.get_gateway_target(gatewayIdentifier=gid, targetId=tid)
            print(f"  {key}: {tid} {t['status']}")


def endpoint() -> None:
    if not RESOLVED_PATH.exists():
        sys.exit("No resolved gateway yet. Run `create` first.")
    print(json.loads(RESOLVED_PATH.read_text())["inference_base_url"])


def teardown() -> None:
    m = _manifest()
    if not m:
        print("Nothing recorded to tear down.")
        return
    gid = m.get("gateway_id")
    # targets -> gateway -> credential provider -> role (reverse of create)
    for key in ("openai_target_id", "bedrock_target_id"):
        tid = m.get(key)
        if tid and gid:
            try:
                acc.delete_gateway_target(gatewayIdentifier=gid, targetId=tid)
                print(f"deleted target {tid}")
            except ClientError as e:
                print(f"target {tid}: {e.response['Error']['Code']}")
    if gid:
        # Target deletion is async; the gateway refuses deletion while any target
        # is still attached. Wait for targets to drain, then retry a few times.
        for attempt in range(5):
            remaining = acc.list_gateway_targets(gatewayIdentifier=gid).get("items", [])
            if not remaining:
                try:
                    acc.delete_gateway(gatewayIdentifier=gid)
                    print(f"deleted gateway {gid}")
                    break
                except ClientError as e:
                    print(f"gateway {gid} (attempt {attempt}): {e.response['Error']['Code']}")
            time.sleep(8)
        else:
            print(f"gateway {gid}: could not delete (targets may still be draining — retry teardown)")
    if m.get("credential_provider_name"):
        try:
            acc.delete_api_key_credential_provider(name=m["credential_provider_name"])
            print(f"deleted credential provider {m['credential_provider_name']}")
        except ClientError as e:
            print(f"credential provider: {e.response['Error']['Code']}")
    if m.get("role_name"):
        try:
            for p in iam.list_role_policies(RoleName=m["role_name"])["PolicyNames"]:
                iam.delete_role_policy(RoleName=m["role_name"], PolicyName=p)
            iam.delete_role(RoleName=m["role_name"])
            print(f"deleted IAM role {m['role_name']}")
        except ClientError as e:
            print(f"role: {e.response['Error']['Code']}")
    for p in (TEARDOWN_PATH, RESOLVED_PATH):
        if p.exists():
            p.unlink()
    print("teardown complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["create", "endpoint", "status", "teardown"])
    parser.add_argument("--with-openai", action="store_true",
                        help="also add the OpenAI first-party example target (needs OPENAI_API_KEY)")
    args = parser.parse_args()
    if args.action == "create":
        create(with_openai=args.with_openai)
    else:
        {"endpoint": endpoint, "status": status, "teardown": teardown}[args.action]()


if __name__ == "__main__":
    main()
