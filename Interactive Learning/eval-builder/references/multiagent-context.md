# Multi-agent context — reference

## What it evaluates

**Shared memory and context propagation** in multi-agent systems — where failures happen _before_
the final answer: a sub-agent receives stale context, the coordinator updates a constraint but
doesn't re-dispatch, or two agents work from different versions of the same plan. These coordination
failures are invisible unless you instrument for them. This reference evaluates how context flows
through hub-and-spoke and peer-to-peer architectures, and how resilient the system is to
**mid-session constraint changes** (baseline vs. conflict).

## Metrics (definitions + how computed)

Six semantic context properties, each judged **binary** per agent call (LLM judge sees the agent's
handoff input, retrieved context, and output):

| Check                    | Failure it detects                                                     |
| ------------------------ | ---------------------------------------------------------------------- |
| **context_freshness**    | agent working from stale context that predates the latest user message |
| **handoff_completeness** | handoff omitted facts the agent needed                                 |
| **context_utilization**  | agent ignored context it read from memory                              |
| **state_consistency**    | agents disagree on key facts (cross-agent, per turn)                   |
| **write_accuracy**       | what the agent wrote to memory is factually wrong                      |
| **redundant_context**    | excessive repeated/irrelevant context transferred                      |

Deterministic support metric: **C2 alignment** = pairwise **cosine similarity** of peer output
embeddings (Titan Embeddings V2). Low alignment after a scope change signals divergence (some peers
updated, others inherited stale context). Also instrument **handoff query, memory read/write
latency, and token usage** per agent call for operational context.

## Binary judge template(s)

**Binary, one failure mode per judge**, returning `{"passed": true|false, "reason": "..."}`. Replaces
the workshop's 1–5 scores.

````python
import boto3, json
bedrock = boto3.client("bedrock-runtime")
JUDGE_MODEL = "us.anthropic.claude-sonnet-4-20250514-v1:0"  # from eval_config.yaml

def _judge(prompt: str) -> dict:
    body = json.dumps({"anthropic_version": "bedrock-2023-05-31", "max_tokens": 512,
                       "temperature": 0.0, "messages": [{"role": "user", "content": prompt}]})
    resp = bedrock.invoke_model(modelId=JUDGE_MODEL, body=body)
    text = json.loads(resp["body"].read())["content"][0]["text"]
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return {"passed": False, "reason": f"unparseable judge output: {text[:200]}"}

def judge_context_freshness(latest_user_msg: str, retrieved_context: str, agent_name: str) -> dict:
    """FAIL if the agent's retrieved context does not reflect the latest user requirements."""
    if not retrieved_context.strip():
        return {"passed": True, "reason": "no prior context (first call)"}
    return _judge(f"""Does the {agent_name} agent's retrieved context reflect the LATEST user
requirements (no stale/overridden fields)?
Latest user message: \"\"\"{latest_user_msg}\"\"\"
Retrieved context: \"\"\"{retrieved_context[:3000]}\"\"\"
PASS only if the context is current. Return ONLY JSON:
{{"passed": true|false, "reason": "<one sentence>"}}""")

def judge_handoff_completeness(handoff_query: str, latest_user_msg: str, agent_name: str) -> dict:
    """FAIL if the handoff to this agent omitted facts it needed."""
    return _judge(f"""Did the handoff to the {agent_name} agent include ALL facts it needs from the
user's request?
User request: \"\"\"{latest_user_msg}\"\"\"
Handoff query passed to agent: \"\"\"{handoff_query[:2000]}\"\"\"
PASS only if nothing important was dropped. Return ONLY JSON:
{{"passed": true|false, "reason": "<one sentence>"}}""")

def judge_context_utilization(retrieved_context: str, response: str, agent_name: str) -> dict:
    """FAIL if the agent ignored the context it read from memory."""
    if not retrieved_context.strip():
        return {"passed": True, "reason": "no context to use (first call)"}
    return _judge(f"""Did the {agent_name} agent actually USE the shared context it read (reference
specific details), rather than ignoring it?
Retrieved context: \"\"\"{retrieved_context[:2500]}\"\"\"
Agent response: \"\"\"{response[:2500]}\"\"\"
PASS only if the context was used. Return ONLY JSON:
{{"passed": true|false, "reason": "<one sentence>"}}""")

def judge_state_consistency(responses: dict) -> dict:
    """FAIL if agents contradict each other on key facts (numbers, dates, constraints)."""
    joined = "\n\n".join(f"[{n}]:\n{r[:2000]}" for n, r in responses.items())
    return _judge(f"""Are these agent responses factually CONSISTENT with each other (agree on
numbers, dates, constraints)?
{joined}
PASS only if there are no genuine contradictions. Return ONLY JSON:
{{"passed": true|false, "reason": "<one sentence>"}}""")

def judge_write_accuracy(response: str, memory_written: str, agent_name: str) -> dict:
    """FAIL if what the agent wrote to shared memory misrepresents its own output."""
    return _judge(f"""Is what the {agent_name} agent wrote to shared memory an ACCURATE
representation of its response (no fabricated or distorted facts)?
Agent response: \"\"\"{response[:2500]}\"\"\"
Written to memory: \"\"\"{memory_written[:2500]}\"\"\"
PASS only if the written memory is faithful. Return ONLY JSON:
{{"passed": true|false, "reason": "<one sentence>"}}""")

def judge_redundant_context(retrieved_context: str, agent_name: str) -> dict:
    """FAIL if the transferred context is bloated with repeated/irrelevant material."""
    return _judge(f"""Is the context transferred to the {agent_name} agent free of significant
repeated or irrelevant material?
Retrieved context: \"\"\"{retrieved_context[:3000]}\"\"\"
PASS only if it is concise and relevant. Return ONLY JSON:
{{"passed": true|false, "reason": "<one sentence>"}}""")
````

Optional deterministic C2-alignment gate (embedding divergence after a scope change):

```python
import math

def cosine(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

def check_alignment(peer_embeddings: dict, min_sim: float) -> dict:
    """FAIL if any pair of peer outputs diverges below the alignment floor."""
    names = list(peer_embeddings)
    worst = ("", "", 1.0)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            s = cosine(peer_embeddings[names[i]], peer_embeddings[names[j]])
            if s < worst[2]:
                worst = (names[i], names[j], s)
    return {"passed": worst[2] >= min_sim,
            "reason": f"min pairwise cosine {worst[2]:.3f} ({worst[0]}↔{worst[1]}) vs floor {min_sim}"}
```

## Aggregation

- **Per-check pass rate** for each of the six checks (`state_consistency` is per turn, cross-agent;
  the rest per agent call).
- **All-checks-pass rate** = share of agent calls where every applicable check passes.
- For **baseline vs. conflict**: run the same architecture with stable inputs, then with a
  mid-session constraint change; report the **delta in each per-check pass rate**. A resilient system
  shows little degradation; a fragile one shows freshness/consistency pass rates dropping sharply on
  the conflict turn.

```python
CHECKS = ["context_freshness", "handoff_completeness", "context_utilization",
          "state_consistency", "write_accuracy", "redundant_context"]

def aggregate(cases: list[dict]) -> dict:
    n = len(cases)
    per_check = {c: sum(case["verdicts"].get(c, True) for case in cases) / n for c in CHECKS}
    all_pass = sum(1 for case in cases if all(case["verdicts"].values())) / n
    return {"per_check_pass_rate": per_check, "all_checks_pass_rate": all_pass, "n": n}
```

## Code pattern

Instrument each agent call to capture the four judge inputs — handoff query, retrieved context,
response, and what was written to memory. A shared-memory hook (Strands) records them:

```python
import time

class ContextRecorder:
    """Captures per-call context I/O for later binary judging."""
    def __init__(self):
        self.calls = []            # list of dicts: agent, handoff, context, response, written
    def record(self, agent, handoff, context, response, written):
        self.calls.append({"agent": agent, "handoff": handoff, "context": context,
                           "response": response, "written": written})

def format_memory(memory: list) -> str:
    return "\n".join(f"[{e.get('agent','?')}] {e.get('role','')}: {e.get('content','')}"
                     for e in memory)

def run_peer(name, prompt, task, shared_memory, recorder, agent_model):
    from strands import Agent
    context = format_memory(shared_memory)                       # what this agent reads
    full_prompt = prompt + (f"\n\nShared memory:\n{context}\n\nUse this context; reference "
                            f"specific details." if context else "")
    agent = Agent(name=name, model=agent_model, system_prompt=full_prompt)
    resp = str(agent(task))
    shared_memory.append({"agent": name, "role": "assistant", "content": resp, "ts": time.time()})
    recorder.record(name, task, context, resp, resp)             # written == resp for a list store
    return resp
```

Then, after a session, judge each recorded call and aggregate. For baseline/conflict, run two
sessions (stable vs. mid-session change) and diff the per-check pass rates.

```python
def judge_call(call, latest_user_msg) -> dict:
    return {
        "context_freshness":   judge_context_freshness(latest_user_msg, call["context"], call["agent"])["passed"],
        "handoff_completeness":judge_handoff_completeness(call["handoff"], latest_user_msg, call["agent"])["passed"],
        "context_utilization": judge_context_utilization(call["context"], call["response"], call["agent"])["passed"],
        "write_accuracy":      judge_write_accuracy(call["response"], call["written"], call["agent"])["passed"],
        "redundant_context":   judge_redundant_context(call["context"], call["agent"])["passed"],
    }
```

## Adaptation notes

- Works for **hub-and-spoke** (coordinator → spokes; watch the coordinator's handoff compression)
  and **peer-to-peer** (sequential or dynamic swarm; no hub to re-dispatch). The instrumentation is
  identical — only who reads/writes memory changes.
- Swap the Python-list memory for **AgentCore Memory** (`bedrock_agentcore.memory.MemoryClient`) in
  production; the record/judge pattern is unchanged.
- Substitute the user's agent prompts, `AGENT_MODEL_ID`, `JUDGE_MODEL_ID`, and
  `EMBEDDING_MODEL_ID` (from `eval_config.yaml`).
- Design a **conflict conversation** that changes a real constraint mid-session (budget, dates,
  scope) so freshness/consistency checks can actually fail. Set the C2 `min_sim` floor from a stable
  baseline run.
- `state_consistency` needs ≥2 agent responses per turn; skip it for single-agent turns.
