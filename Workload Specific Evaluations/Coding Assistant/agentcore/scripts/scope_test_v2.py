#!/usr/bin/env python3
"""Negative test of the scoped roles, from inside a session: the grader phase's credentials (runtime role, via IMDS) must be able to call
the judge model and nothing else; the agent phase's credentials (agent role) must not be able to call bedrock-runtime at all.
Usage: python scripts/scope_test_v2.py [--prefix PREFIX]   (needs the pr4121 runtime of that prefix, deployed; default prefix
evals-p2b- or $P2_PREFIX; only one-token model calls)"""
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import p2_v2 as p2

PY = r'''
import boto3, json, sys
from botocore.config import Config
c = boto3.client("bedrock-runtime", region_name="us-east-1", config=Config(retries={"max_attempts": 1}, read_timeout=60))
out = {}
for m in sys.argv[1:]:
    try:
        r = c.converse(modelId=m, messages=[{"role": "user", "content": [{"text": "Reply with the word ok."}]}], inferenceConfig={"maxTokens": 8})
        out[m] = "OK " + r["output"]["message"]["content"][0]["text"][:10]
    except Exception as e:
        out[m] = type(e).__name__ + ": " + str(e)[:140].replace(chr(10), " ")
print(json.dumps(out, indent=1))
'''
MODELS = ["us.anthropic.claude-haiku-5-5", "us.anthropic.claude-sonnet-5", "us.amazon.nova-lite-v1:0"]

if "--prefix" in sys.argv:
    p2.Names.prefix = sys.argv[sys.argv.index("--prefix") + 1]
s = p2.Session("pr4121")
s.put("/tmp/neg.py", PY.encode(), "0644", "root")
o, c, _ = s.cmd("cat /run/fw.status")
print("firewall:", o.strip())
# grader phase: runtime role credentials
p2.set_phase(s, p2.HOST_RUNTIME, {"AWS_REGION": p2.R, "AWS_DEFAULT_REGION": p2.R, **p2.runtime_role_creds(s)})
print("--- grader phase (runtime role):")
print(s.agent("/opt/verifier-venv/bin/python /tmp/neg.py " + " ".join(MODELS), 120)[0])
# agent phase credentials against bedrock-runtime (the proxy allowlist is switched to runtime to test IAM, not the firewall)
p2.set_phase(s, p2.HOST_RUNTIME, {"AWS_REGION": p2.R, "AWS_DEFAULT_REGION": p2.R, **p2.agent_creds(900)})
print("--- agent role credentials against bedrock-runtime:")
print(s.agent("/opt/verifier-venv/bin/python /tmp/neg.py " + MODELS[0], 120)[0])
print("stop:", s.stop())
