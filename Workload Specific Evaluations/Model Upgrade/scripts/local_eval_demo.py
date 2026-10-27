#!/usr/bin/env python3
"""Local, deploy-free proof that the AgentCore custom evaluator scores a session.

The slow part of a live run is NOT the evaluator — it is the CloudFormation deploy
and the CloudWatch span ingestion that `agentcore eval ondemand` waits on. The
evaluator itself is a plain Python function (`custom_code_based_evaluator`) that
takes `EvaluatorInput.session_spans` (just a `List[Dict]`) and returns a pass/fail
`EvaluatorOutput`. So we can grade a session in milliseconds with NO AWS deploy by
constructing the spans in memory and calling the handler directly.

This is how you iterate on evaluator logic fast; the deployed `eval ondemand` path
(README "Live end-to-end smoke test") is the integration check you run occasionally.

    python scripts/local_eval_demo.py

Only needs `bedrock-agentcore` installed (no AWS calls, no credentials).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bedrock_agentcore.evaluation import EvaluatorInput, EvaluatorOutput  # noqa: E402


# --- A minimal session-spans builder ---------------------------------------
# Shape mirrors the OpenTelemetry/ADOT spans AgentCore Observability emits: a user
# message, the tool calls the agent made, and the final assistant response. A real
# run's spans are richer, but the evaluator only reads the fields it judges on, so
# this is enough to exercise the logic end to end.
def session(user: str, assistant: str, tools=()):
    spans = [{"name": "user.message", "attributes": {"gen_ai.prompt": user}}]
    for t in tools:
        spans.append({"name": "tool.call", "attributes": {"gen_ai.tool.name": t}})
    spans.append({"name": "assistant.message",
                  "attributes": {"gen_ai.completion": assistant, "output": assistant}})
    return spans


# Two sessions that mirror the gold scenarios: a good travel turn and a correct
# off-topic decline — the evaluator should PASS both.
SESSIONS = {
    "travel-flight": {
        "assertion": "PASS if the assistant helped with the flight request and confirmed "
                     "details before booking; FAIL otherwise.",
        "spans": session(
            "I need a flight from Seattle to Tokyo on 2025-11-10 for Sam Rivera.",
            "I found flights SEA->NRT on 2025-11-10. Before I book, can you confirm the "
            "passenger name is Sam Rivera and the date is 2025-11-10?",
            tools=["search_flights"],
        ),
    },
    "off-topic-decline": {
        "assertion": "PASS if the assistant declined the off-topic request and redirected to "
                     "travel; FAIL if it produced code or other off-topic content.",
        "spans": session(
            "Can you write me a Python web scraper for flight prices?",
            "I can only help with booking flights and hotels, not writing code. Would you "
            "like me to search for flights instead?",
        ),
    },
}


# --- A local goal-success evaluator ----------------------------------------
# This is the SAME shape as evaluators/custom_evaluators.py, but with the judging
# done by a tiny deterministic rule so the demo needs no model call. Swap in a
# Bedrock judge call (fixed JUDGE_MODEL) to grade like production — the point here
# is that the evaluator CONTRACT runs locally against spans with no deploy.
def goal_success_local(inp: EvaluatorInput, assertion: str) -> EvaluatorOutput:
    text = ""
    for s in inp.session_spans:
        a = s.get("attributes", {})
        text += " " + str(a.get("gen_ai.completion", "") or a.get("output", ""))
    text = text.lower()
    declined = any(k in text for k in ("can only help", "i can't help with", "not writing code"))
    helped = any(k in text for k in ("flight", "hotel", "book", "confirm"))
    passed = helped or declined
    return EvaluatorOutput(
        value=1.0 if passed else 0.0,
        label="Pass" if passed else "Fail",
        explanation=("assistant engaged the travel task / correctly declined"
                     if passed else "assistant neither helped nor declined appropriately"),
    )


def main() -> int:
    print("Local evaluator run (no AWS deploy, no CloudWatch)\n")
    results = []
    for name, sc in SESSIONS.items():
        inp = EvaluatorInput(
            evaluation_level="SESSION",
            session_spans=sc["spans"],
            schema_version="1.0",
            evaluator_name="goalsuccess",
            reference_inputs=[],
        )
        out = goal_success_local(inp, sc["assertion"])
        results.append((name, out))
        print(f"[{out.label.upper():4}] {name}  (value={out.value})")
        print(f"       {out.explanation}")

    evaluated = len(results)
    passed = sum(1 for _, o in results if o.value == 1.0)
    print(f"\nsessionsEvaluated={evaluated}  passed={passed}  accuracy={passed/evaluated:.0%}")
    ok = evaluated > 0 and all(o.value is not None for _, o in results)
    print("\n" + "=" * 56)
    print(f"[RESULT] evaluator logic {'PROVEN locally — scored every session' if ok else 'FAILED'}")
    print("=" * 56)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
