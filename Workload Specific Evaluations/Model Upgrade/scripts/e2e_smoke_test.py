#!/usr/bin/env python3
"""Live end-to-end smoke test for the Model Upgrade module's AgentCore path.

Proves, against REAL AWS resources, the AgentCore elements the module relies on:

    AgentCore Runtime (deploy a Strands agent)  ->  live invoke (real agent turn)
    ->  AgentCore Observability (traces/spans)  ->  AgentCore Evaluations
    (LLM-as-a-judge, SESSION level)  ->  pass/fail scores

Flow (all via the GA `agentcore` CLI, so it exercises the real product surface):
  1. `agentcore create`  -- scaffold a Strands-Python project (the Runtime)
  2. overwrite the agent with the module's travel-booking agent (calls Bedrock Luna)
  3. `agentcore add evaluator llm-as-a-judge`  -- a real SESSION evaluator
  4. `agentcore deploy`  -- provision the Runtime for real
  5. `agentcore eval ondemand simulate`  -- replay a TINY 2-scenario dataset against
     the Runtime and score the resulting sessions with the evaluator
  6. assert scores came back, then ALWAYS tear down

The Gateway inference seam (bedrock-runtime connector) is proven separately by
gateway/setup_gateway.py + a live /inference call; this test keeps the agent's model
call direct-to-Bedrock so a smoke test is not also debugging gateway-auth-from-runtime.
Set USE_GATEWAY=1 to point the agent at a already-provisioned gateway instead.

Usage:
    python scripts/e2e_smoke_test.py          # provision, run, verify, teardown
    python scripts/e2e_smoke_test.py --keep   # leave the Runtime up for inspection

Requires: `agentcore` on PATH, AWS creds (Bedrock + AgentCore + CDK), target model
enabled. Takes several minutes (a real CDK deploy).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import secrets
from pathlib import Path

REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
TARGET_MODEL = os.environ.get("E2E_TARGET_MODEL", "us.openai.gpt-6-luna")  # direct Bedrock id
JUDGE_MODEL = os.environ.get("E2E_JUDGE_MODEL", "amazon.nova-micro-v1:0")
PROJECT = "mue2e" + secrets.token_hex(3)
EVALUATOR = "goalsuccess"

AGENT_MAIN = '''\
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands import Agent

app = BedrockAgentCoreApp()

SYSTEM = ("You are a travel booking assistant. Help with flights and hotels only. "
          "Always confirm details before booking. Politely decline anything off-topic.")

# Strands defaults to Bedrock; the target model id is baked in (smoke test keeps it
# simple — the module itself routes this through the Gateway inference seam instead).
TARGET_MODEL = "__TARGET_MODEL__"

@app.entrypoint
def invoke(payload):
    prompt = payload.get("prompt") or payload.get("input") or ""
    agent = Agent(model=TARGET_MODEL, system_prompt=SYSTEM)
    return {"result": str(agent(prompt))}

if __name__ == "__main__":
    app.run()
'''

DATASET = [
    {"example_id": "travel-flight", "input": "I need a flight from Seattle to Tokyo on 2025-11-10 for Sam Rivera."},
    {"example_id": "off-topic-decline", "input": "Can you write me a Python web scraper for flight prices?"},
]


def run(cmd, cwd=None, check=True, capture=True):
    print(f"  $ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=cwd, text=True, capture_output=capture,
                       env={**os.environ, "AWS_REGION": REGION})
    if capture and r.stdout:
        print("   ", r.stdout.strip()[:800])
    if r.returncode != 0 and check:
        if r.stderr:
            print(r.stderr, file=sys.stderr)
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(cmd)}")
    return r


def _runtime_id(proj: Path) -> str:
    r = run(["agentcore", "runtime", "list", "--json"], cwd=proj, check=False)
    try:
        data = json.loads(r.stdout)
        items = (data.get("agentRuntimes") or data.get("items")
                 or (data if isinstance(data, list) else []))
        if items:
            it = items[0]
            return (it.get("agentRuntimeId") or it.get("runtimeId")
                    or it.get("id") or it.get("agentRuntimeName") or it.get("name"))
    except Exception:
        pass
    raise RuntimeError(f"could not resolve runtime id from: {r.stdout[:300]}")


def _evaluator_id(proj: Path, name: str) -> str:
    r = run(["agentcore", "eval", "evaluator", "list", "--json"], cwd=proj, check=False)
    try:
        data = json.loads(r.stdout)
        items = (data.get("evaluators") or data.get("items")
                 or (data if isinstance(data, list) else []))
        # Our custom evaluator's id matches the pattern <name>-<10 chars> and is NOT
        # a Builtin.*; match on that, never fall back to a built-in evaluator.
        for it in items:
            eid = it.get("evaluatorId") or it.get("id") or ""
            enm = it.get("evaluatorName") or it.get("name") or ""
            if eid.startswith("Builtin.") or eid.startswith("ThirdParty."):
                continue
            if name in enm or name in eid:
                return eid or it.get("evaluatorArn")
    except Exception:
        pass
    raise RuntimeError(f"could not resolve CUSTOM evaluator id for '{name}' from: {r.stdout[:400]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keep", action="store_true", help="leave the Runtime up after the run")
    args = ap.parse_args()

    if not shutil.which("agentcore"):
        sys.exit("agentcore CLI not on PATH. Install: npm install -g @aws/agentcore@1.0.0")

    work = Path(tempfile.mkdtemp(prefix="mue2e-", dir=os.environ.get("KIROCREW_SCRATCH") or None))
    proj = work / PROJECT
    print(f"[0] region={REGION} target={TARGET_MODEL} judge={JUDGE_MODEL}")
    print(f"    scratch: {work}")
    created = False
    try:
        print("[1] create project + Strands runtime")
        run(["agentcore", "create", "--name", PROJECT, "--template",
             "agent-python-strands", "--skip-git"], cwd=work)
        created = True

        main_py = proj / "app" / "agent" / "main.py"
        main_py.write_text(AGENT_MAIN.replace("__TARGET_MODEL__", TARGET_MODEL))
        print(f"    wrote agent entrypoint (model baked in): {main_py}")

        print("[3] add an LLM-as-a-judge SESSION evaluator")
        run(["agentcore", "add", "evaluator", "llm-as-a-judge", "--name", EVALUATOR,
             "--level", "SESSION", "--model", JUDGE_MODEL, "--rating-scale", "pass-fail",
             "--instructions",
             "You are scoring a travel assistant. PASS if it helped with the travel "
             "request (flights/hotels) OR correctly declined an off-topic request; "
             "FAIL otherwise. {context}"], cwd=proj, check=False)

        print("[4] deploy (real CDK deploy — several minutes)")
        run(["agentcore", "deploy"], cwd=proj, capture=False)

        rid = _runtime_id(proj)
        print(f"    runtime id: {rid}")

        print("[5] live invoke both scenarios (real agent turns -> Observability traces)")
        import uuid
        session_ids = []
        for d in DATASET:
            sid = f"e2e-{d['example_id']}-{uuid.uuid4().hex[:8]}"
            session_ids.append(sid)
            run(["agentcore", "invoke", "--runtime", rid, "--session-id", sid,
                 "--payload", json.dumps({"prompt": d["input"]})], cwd=proj, check=False)

        print("[6] wait for span ingestion, then eval ondemand evaluate the real sessions")
        time.sleep(150)  # let traces land in Observability before grading
        eval_id = _evaluator_id(proj, EVALUATOR)
        print(f"    evaluator id: {eval_id}")
        # Select by the agent's recent sessions (lookback) AND by explicit ids — the
        # lookback window reliably matches ingested sessions for this agent.
        sim = run(["agentcore", "eval", "ondemand", "evaluate",
                   "--agent", rid,
                   "--lookback-days", "1",
                   "--evaluators", eval_id, "--json"], cwd=proj, check=False, capture=True)

        out = (sim.stdout or "") + "\n" + (sim.stderr or "")
        print("    --- full evaluate output ---")
        print(out[:4000])
        # A REAL pass requires at least one session actually evaluated with a verdict.
        import re as _re
        m = _re.search(r'"sessionsEvaluated"\s*:\s*(\d+)', out)
        evaluated = int(m.group(1)) if m else 0
        has_verdict = any(t in out.lower() for t in ('"score"', '"label"', '"pass"', '"fail"', '"result"'))
        ok = evaluated > 0 and (has_verdict or '"results": [' in out and '"results": []' not in out)
        print("\n" + "=" * 60)
        print(f"[RESULT] end-to-end {'PROVEN — evaluator returned scores' if ok else 'INCONCLUSIVE — see output above'}")
        print("=" * 60)
        return 0 if ok else 2

    finally:
        if created and not args.keep:
            print("[teardown] destroying the project stack (removes the Runtime)")
            # Remove declared resources, then deploy the now-empty project so CFN
            # tears the stack down; --yes confirms stack removal.
            run(["agentcore", "remove", "runtime", "--name", "agent"], cwd=proj, check=False)
            run(["agentcore", "deploy", "--yes"], cwd=proj, check=False, capture=False)
            # delete the evaluator (not part of the CFN stack)
            run(["agentcore", "eval", "evaluator", "delete", EVALUATOR], cwd=proj, check=False)
        shutil.rmtree(work, ignore_errors=True)
        kept = " (resources KEPT)" if args.keep else ""
        print(f"[teardown] removed scratch {work}{kept}")


if __name__ == "__main__":
    raise SystemExit(main())
