# Tool-calling — reference

## What it evaluates

An agent's **tool-calling decisions** — without executing real tools (real calls are expensive,
slow, non-deterministic, and can have side effects). Every tool call decomposes into three
independent decisions: **whether** to call a tool, **which** tool to call, and **what** parameters
to pass. This reference evaluates those decisions with a pyramid of approaches, cheapest first:

| Layer | Approach                   | Cost | Catches                                                                 |
| ----- | -------------------------- | ---- | ----------------------------------------------------------------------- |
| 1     | Static trajectory analysis | $0   | wrong selection, wrong sequence, missing params (on recorded traces)    |
| 2     | Schema validation          | $0   | parameter hallucination, type/pattern errors                            |
| 3     | Mock-tool evaluation       | $    | live model decisions, safely (Converse `toolConfig` loop, mock results) |
| 4     | Binary LLM-as-judge        | $$   | subjective quality (appropriateness, efficiency, helpfulness)           |

Separate **what you evaluate** (the model's decisions) from **what you execute** (nothing real).

## Metrics (definitions + how computed)

Deterministic, computed on the trajectory (`tools_called`, `params_used`) vs. ground truth
(`expected_tools`, `should_call_tool`, `expected_params`):

- **Tool precision** = `|expected ∩ actual| / |actual|` (1.0 if no calls made).
- **Tool recall** = `|expected ∩ actual| / |expected|` (1.0 if none expected).
- **Tool F1** = harmonic mean of precision and recall — the single most important metric.
- **Call/no-call accuracy** = did the agent call a tool exactly when it should have?
- **Sequence match** = `tools_called == expected_tools` (order-sensitive).
- **Parameter accuracy** = fraction of expected `(key, value)` pairs present and correct.
- **Schema compliance** = params validate against the tool's JSON Schema (+ domain patterns).

These feed binary checks via thresholds (e.g. **tool_selection passes iff F1 == 1.0 AND call/no-call
correct**). The LLM judges below cover what metrics can't.

## Binary judge template(s)

**Every judge is binary and covers exactly one failure mode.** Return
`{"passed": true|false, "reason": "..."}`. Do **not** use 1–5 scales.

Shared judge caller + robust JSON parse:

````python
import boto3, json, re
bedrock = boto3.client("bedrock-runtime")
JUDGE_MODEL = "us.anthropic.claude-sonnet-4-20250514-v1:0"  # from eval_config.yaml

def parse_llm_json(text: str) -> dict:
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return {"passed": False, "reason": f"unparseable judge output: {text[:200]}"}

def run_binary_judge(judge_prompt: str) -> dict:
    resp = bedrock.converse(
        modelId=JUDGE_MODEL,
        messages=[{"role": "user", "content": [{"text": judge_prompt}]}],
        system=[{"text": 'You are a strict evaluator. Return ONLY JSON '
                         '{"passed": true|false, "reason": "<one sentence>"}.'}],
    )
    return parse_llm_json(resp["output"]["message"]["content"][0]["text"])
````

Judge context block reused by each judge:

```python
def _context(test_case, result, tool_defs):
    tools = "\n".join(f"- {t['name']}: {t['description']}" for t in tool_defs["tools"])
    traj = json.dumps(result["trajectory"], indent=2) if result["trajectory"] else "(no tools called)"
    return f"""<available_tools>
{tools}
</available_tools>
<user_request>
{test_case['input']}
</user_request>
<expected_tools>{json.dumps(test_case['expected_tools'])}</expected_tools>
<actual_trajectory>
{traj}
</actual_trajectory>
<final_response>
{result['final_output'][:500]}
</final_response>"""
```

One judge per failure mode:

```python
def judge_tool_selection(tc, result, tool_defs) -> dict:
    """FAIL if the agent chose the wrong tool(s) for the task."""
    return run_binary_judge(_context(tc, result, tool_defs) + """

Question: Did the agent select the RIGHT tool(s) for this request — no wrong tools, none missing?
PASS only if the tool selection is fully appropriate for the task.
Return ONLY JSON: {"passed": true|false, "reason": "<one sentence>"}""")

def judge_parameter_quality(tc, result, tool_defs) -> dict:
    """FAIL if any tool was called with incorrect, incomplete, or malformed parameters."""
    return run_binary_judge(_context(tc, result, tool_defs) + """

Question: Were ALL tool parameters correct, complete, and well-formed?
PASS only if every parameter is right; FAIL if any is wrong, missing, or fabricated.
Return ONLY JSON: {"passed": true|false, "reason": "<one sentence>"}""")

def judge_sequence_logic(tc, result, tool_defs) -> dict:
    """FAIL if tools were called in an illogical order (e.g. return before lookup)."""
    return run_binary_judge(_context(tc, result, tool_defs) + """

Question: Were the tools called in a sensible, logical order given dependencies between them?
PASS only if the ordering is defensible.
Return ONLY JSON: {"passed": true|false, "reason": "<one sentence>"}""")

def judge_efficiency(tc, result, tool_defs) -> dict:
    """FAIL if there were redundant or unnecessary tool calls."""
    return run_binary_judge(_context(tc, result, tool_defs) + """

Question: Did the agent use the MINIMUM necessary tool calls, with no duplicate or wasted calls?
PASS only if there is no redundancy or waste.
Return ONLY JSON: {"passed": true|false, "reason": "<one sentence>"}""")

def judge_response_quality(tc, result, tool_defs) -> dict:
    """FAIL if the final response does not correctly address the user's request."""
    return run_binary_judge(_context(tc, result, tool_defs) + """

Question: Does the final response correctly and helpfully address the user's original request?
PASS only if it does.
Return ONLY JSON: {"passed": true|false, "reason": "<one sentence>"}""")
```

Deterministic $0 binary checks (run these first, on every trace):

```python
def check_tool_selection_metric(trace, tc) -> dict:
    """Binary gate on F1 + call/no-call — $0, no API call."""
    exp, act = set(tc["expected_tools"]), set(trace["tools_called"])
    tp, fp, fn = len(exp & act), len(act - exp), len(exp - act)
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    call_ok = tc["should_call_tool"] == (len(trace["tools_called"]) > 0)
    passed = f1 == 1.0 and call_ok
    return {"passed": passed, "reason": f"F1={f1:.2f}, call/no-call {'ok' if call_ok else 'WRONG'}"}

def check_schema_compliance(trace, tool_defs) -> dict:
    """Binary gate on JSON-Schema validation of every param set — $0."""
    import jsonschema
    errors = []
    for name in trace["tools_called"]:
        spec = next((t for t in tool_defs["tools"] if t["name"] == name), None)
        if not spec:
            errors.append(f"unknown tool {name}"); continue
        v = jsonschema.Draft7Validator(spec["parameters"])
        errors += [e.message for e in v.iter_errors(trace["params_used"].get(name, {}))]
    return {"passed": len(errors) == 0, "reason": "; ".join(errors) or "all params valid"}
```

## Aggregation

- **Per-check pass rate** = passing cases / total, for each judge/check.
- **All-checks-pass rate** = share of cases where **every** check (tool_selection, parameter_quality,
  sequence_logic, efficiency, response_quality, schema) passes. A case passes overall only if all
  pass.
- Track **Tool F1** and **call/no-call accuracy** as headline deterministic numbers alongside the
  pass rates. Never average the judge verdicts into a single score.

```python
CHECKS = ["tool_selection", "parameter_quality", "sequence_logic", "efficiency", "response_quality"]

def aggregate(cases: list[dict]) -> dict:
    n = len(cases)
    per_check = {c: sum(case["verdicts"][c] for case in cases) / n for c in CHECKS}
    all_pass = sum(1 for case in cases if all(case["verdicts"].values())) / n
    return {"per_check_pass_rate": per_check, "all_checks_pass_rate": all_pass, "n": n}
```

## Code pattern

Mock-tool loop (Layer 3) — runs the live model but returns mock `toolResult`s, so no real tool
executes. Produces the `trajectory` the judges and metrics consume.

```python
def to_converse_tool_config(tool_defs):
    return {"tools": [{"toolSpec": {"name": t["name"], "description": t["description"],
                                    "inputSchema": {"json": t["parameters"]}}}
                      for t in tool_defs["tools"]]}

def run_with_mock_tools(user_input, tool_config, mock_responses, agent_model,
                        system_prompt, max_turns=10):
    messages = [{"role": "user", "content": [{"text": user_input}]}]
    trajectory, final_text = [], "[max turns reached]"
    for turn in range(max_turns):
        resp = bedrock.converse(modelId=agent_model, messages=messages,
                                system=[{"text": system_prompt}], toolConfig=tool_config)
        content = resp["output"]["message"]["content"]
        messages.append({"role": "assistant", "content": content})
        if resp["stopReason"] == "end_turn":
            final_text = "".join(b["text"] for b in content if "text" in b); break
        if resp["stopReason"] == "tool_use":
            results = []
            for b in content:
                if "toolUse" in b:
                    call = b["toolUse"]
                    trajectory.append({"tool": call["name"], "params": call["input"], "turn": turn})
                    fn = mock_responses.get(call["name"])
                    out = fn(call["input"]) if fn else {"error": f"unknown tool {call['name']}"}
                    results.append({"toolResult": {"toolUseId": call["toolUseId"],
                                                   "content": [{"json": out}], "status": "success"}})
            messages.append({"role": "user", "content": results})
    return {"trajectory": trajectory,
            "tools_called": [t["tool"] for t in trajectory],
            "params_used": {t["tool"]: t["params"] for t in trajectory},
            "final_output": final_text}
```

For fully synthetic, stateful runs the workshop uses Strands `ToolSimulator` + `ActorSimulator`;
prefer the mock loop for CI (deterministic, $0 tools) and reserve the simulator for exploratory
edge-case discovery.

## Adaptation notes

- Replace the tool set, `mock_responses`, and `SYSTEM_PROMPT` with the user's own agent's tools and
  prompt. Keep the JSON-Schema `parameters` accurate so schema validation is meaningful.
- Build test cases from production logs where possible: each case needs `input`, `expected_tools`,
  `should_call_tool`, and `expected_params`. Include at least one **no-tool** case and one
  **multi-tool** case.
- Substitute the user's `AGENT_MODEL` and `JUDGE_MODEL` (from `eval_config.yaml`).
- Add domain-specific pattern checks in schema validation (e.g. an ID must match `^ORD-\d{5}$`) to
  catch fabricated parameter values.
- For irreversible tools (booking, payments), lean harder on Layers 1–2 and mock evaluation — never
  execute the real tool during evals.
