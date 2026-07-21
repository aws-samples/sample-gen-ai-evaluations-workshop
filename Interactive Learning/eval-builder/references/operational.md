# Operational — reference

## What it evaluates

How a Bedrock workload **runs**, independent of answer quality: per-call **cost**, **latency**
(end-to-end and streaming), **throughput** (tokens/sec), and **error/throttle rate**. These are
deterministic measurements taken from the Converse / Converse-stream API, turned into binary
pass/fail checks against user-defined budgets and SLAs. Operational is a near-universal baseline —
apply it to almost any LLM workload.

## Metrics (definitions + how computed)

| Metric                    | Definition                                     | How computed                                                                                    |
| ------------------------- | ---------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| **Total cost (USD)**      | Dollar cost of one invocation                  | `(input_tokens/1000)*price_in + (output_tokens/1000)*price_out` using a per-model pricing table |
| **Server latency (ms)**   | End-to-end request→full-response time          | `response["metrics"]["latencyMs"]` from the Converse API (no manual timing)                     |
| **TTFT (ms)**             | Time to first token — perceived responsiveness | first `contentBlockDelta` timestamp − request start (streaming)                                 |
| **TTLT (ms)**             | Time to last token — total generation time     | last `contentBlockDelta` timestamp − request start (streaming)                                  |
| **Tokens/sec**            | Throughput                                     | `output_tokens / (latency_ms/1000)`                                                             |
| **Error / throttle rate** | Share of calls that fail or are throttled      | failed calls / total calls over the run                                                         |

Definitions are exact; the **pass/fail** decision comes from comparing each measurement to a
user-supplied threshold (budget or SLA), captured in `eval_config.yaml`.

## Binary judge template(s)

Operational checks are **threshold assertions**, not LLM judgments — but they return the same binary
contract as every other judge: `{"passed": bool, "reason": str}`. **One failure mode per check.**

```python
def check_latency(measurement: dict, max_latency_ms: float) -> dict:
    """FAIL if end-to-end latency exceeds the SLA."""
    latency = measurement["server_latency_ms"]
    passed = latency <= max_latency_ms
    return {"passed": passed,
            "reason": f"latency {latency:.0f}ms <= {max_latency_ms:.0f}ms SLA" if passed
                      else f"latency {latency:.0f}ms EXCEEDS {max_latency_ms:.0f}ms SLA"}

def check_cost(measurement: dict, max_cost_usd: float) -> dict:
    """FAIL if per-call cost exceeds the budget."""
    cost = measurement["total_cost_usd"]
    passed = cost <= max_cost_usd
    return {"passed": passed,
            "reason": f"cost ${cost:.6f} <= ${max_cost_usd:.6f} budget" if passed
                      else f"cost ${cost:.6f} EXCEEDS ${max_cost_usd:.6f} budget"}

def check_ttft(measurement: dict, max_ttft_ms: float) -> dict:
    """FAIL if time-to-first-token exceeds the responsiveness target."""
    ttft = measurement["ttft_ms"]
    passed = ttft <= max_ttft_ms
    return {"passed": passed,
            "reason": f"TTFT {ttft:.0f}ms <= {max_ttft_ms:.0f}ms target" if passed
                      else f"TTFT {ttft:.0f}ms EXCEEDS {max_ttft_ms:.0f}ms target"}
```

Add one function per operational failure mode (latency, cost, TTFT, throughput floor, error-rate
ceiling). Do not collapse them into a single multi-metric score.

## Aggregation

- **Per-check pass rate** = passing calls / total calls, for each threshold check.
- **All-checks-pass rate** = share of calls where **every** operational check passes (a call passes
  overall only if it is within budget AND within the latency/TTFT SLAs).
- Report both. Never average raw latencies/costs into a single "operational score".

```python
def aggregate(cases: list[dict], check_names: list[str]) -> dict:
    n = len(cases)
    per_check = {c: sum(case["verdicts"][c] for case in cases) / n for c in check_names}
    all_pass = sum(1 for case in cases if all(case["verdicts"].values())) / n
    return {"per_check_pass_rate": per_check, "all_checks_pass_rate": all_pass, "n": n}
```

## Code pattern

Measure a call, then apply the threshold checks. Uses the Converse / Converse-stream API only.

```python
import boto3, time

bedrock = boto3.client("bedrock-runtime")

# Pricing per 1K tokens — keep current; source from eval_config.yaml
MODEL_PRICING = {
    "us.amazon.nova-lite-v1:0": {"input": 0.00006, "output": 0.000015},
    "us.amazon.nova-pro-v1:0":  {"input": 0.0008,  "output": 0.0002},
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0": {"input": 0.003, "output": 0.015},
}

def measure_streaming(model_id: str, prompt: str, max_tokens: int = 200) -> dict:
    start = time.time()
    stream = bedrock.converse_stream(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": 0.1},
    )
    first = last = None
    out_tokens = in_tokens = 0
    for event in stream["stream"]:
        now = time.time()
        if "contentBlockDelta" in event:
            first = first or now
            last = now
        elif "metadata" in event:
            usage = event["metadata"].get("usage", {})
            out_tokens = usage.get("outputTokens", 0)
            in_tokens = usage.get("inputTokens", 0)
    ttft_ms = (first - start) * 1000
    ttlt_ms = (last - start) * 1000
    p = MODEL_PRICING[model_id]
    cost = (in_tokens / 1000) * p["input"] + (out_tokens / 1000) * p["output"]
    return {"model_id": model_id, "ttft_ms": ttft_ms, "ttlt_ms": ttlt_ms,
            "server_latency_ms": ttlt_ms, "total_cost_usd": cost,
            "tokens_per_second": out_tokens / (ttlt_ms / 1000) if ttlt_ms else 0.0}
```

Optionally publish the raw measurements to CloudWatch (`put_metric_data`, namespace
`llm_custom_operational_metrics`, `Model` dimension) for dashboards — but the **eval verdict** comes
from the binary threshold checks above, not the dashboard.

## Adaptation notes

- Replace `MODEL_PRICING` with the user's actual model IDs and current per-1K prices; store them in
  `eval_config.yaml`.
- Set thresholds (`max_latency_ms`, `max_cost_usd`, `max_ttft_ms`, throughput floor, error-rate
  ceiling) from the user's stated SLA/budget in discovery. If they don't have targets, ask — don't
  invent silent defaults.
- For non-streaming workloads, use `converse` and `response["metrics"]["latencyMs"]`; TTFT/TTLT
  require `converse_stream`.
- Run the same prompt set across candidate models to compare cost/latency tradeoffs; each model
  becomes its own set of measurements feeding the same threshold checks.
