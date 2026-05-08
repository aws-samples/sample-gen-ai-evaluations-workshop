---
name: Operational Metrics for LLMs
description: Learn to measure LLM operational metrics like cost, latency, TTFT, and throughput. Activate when asked to "measure bedrock performance", "track LLM costs", "set up CloudWatch for bedrock", "compare model latency", or "build an LLM metrics dashboard".
---

# Operational Metrics for LLMs on Amazon Bedrock

Measure what matters before optimizing. This module builds a complete observability pipeline for LLM applications — from per-call cost tracking through streaming latency to a CloudWatch dashboard that surfaces problems before users notice them.

## Prerequisites

- AWS account with Amazon Bedrock model access (Nova Lite, Nova Pro, Claude 3.7 Sonnet)
- Python 3.10+
- `boto3` installed and configured with appropriate IAM permissions
- CloudWatch PutMetricData permissions

## Learning Objectives

1. **Calculate** per-invocation cost from token usage and model pricing tables
2. **Measure** end-to-end latency and tokens-per-second using the Converse API
3. **Distinguish** TTFT from TTLT using streaming responses and explain when each matters
4. **Publish** custom operational metrics to CloudWatch with appropriate dimensions
5. **Compare** model performance across cost, speed, and throughput for a real workload

## Setup

```python
import boto3
import json
import time
import statistics
from datetime import datetime
from typing import Dict
from decimal import Decimal

bedrock_client = boto3.client("bedrock-runtime")
cloudwatch = boto3.client("cloudwatch")

# Pricing per 1K tokens — update as pricing changes
MODEL_PRICING = {
    "us.amazon.nova-lite-v1:0": {"input": 0.00006, "output": 0.000015},
    "us.amazon.nova-pro-v1:0": {"input": 0.0008, "output": 0.0002},
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0": {"input": 0.003, "output": 0.015},
}
```

## Section 1: Cost Tracking

**Concept:** Every LLM call has a measurable cost determined by input tokens × input price + output tokens × output price. Without tracking this per-call, costs become invisible until the bill arrives. Publishing cost as a CloudWatch metric lets you set alarms before budgets blow.

**Build:**

```python
def calculate_cost(model_id: str, input_tokens: int, output_tokens: int) -> Dict:
    """Calculate cost and publish to CloudWatch."""
    pricing = MODEL_PRICING[model_id]
    input_cost = (input_tokens / 1000) * pricing["input"]
    output_cost = (output_tokens / 1000) * pricing["output"]
    total_cost = input_cost + output_cost

    cloudwatch.put_metric_data(
        Namespace="llm_custom_operational_metrics",
        MetricData=[{
            "MetricName": "TotalCost",
            "Value": total_cost,
            "Dimensions": [{"Name": "Model", "Value": model_id}],
        }],
    )

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_cost_usd": f"${total_cost:.8f}",
    }
```

Run it: `calculate_cost("us.anthropic.claude-3-7-sonnet-20250219-v1:0", 20000, 1500)` → expect ~$0.0825.

## Section 2: Latency Measurement

**Concept:** Latency (time from request to complete response) determines user experience for synchronous calls. The Bedrock Converse API returns `metrics.latencyMs` directly — no manual timing needed. Combining latency with token count gives you tokens-per-second, the key throughput indicator.

**Build:**

```python
def measure_latency(model_id: str, prompt: str, max_tokens: int = 100) -> Dict:
    """Measure end-to-end latency using Converse API built-in metrics."""
    response = bedrock_client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": 0.1},
    )

    latency_ms = response["metrics"]["latencyMs"]
    usage = response["usage"]

    return {
        "model_id": model_id,
        "server_latency_ms": latency_ms,
        "input_tokens": usage["inputTokens"],
        "output_tokens": usage["outputTokens"],
        "tokens_per_second": round(usage["outputTokens"] / (latency_ms / 1000), 1),
    }
```

Compare: call with `"us.amazon.nova-pro-v1:0"` vs `"us.anthropic.claude-3-7-sonnet-20250219-v1:0"` on the same prompt. Notice the latency/cost tradeoff.

## Section 3: Streaming — TTFT vs TTLT

**Concept:** Time to First Token (TTFT) measures perceived responsiveness — how fast the user sees *something*. Time to Last Token (TTLT) measures total generation time. For interactive UIs, low TTFT matters most. For batch pipelines, TTLT determines throughput. The `converse_stream` API exposes both.

**Build:**

```python
def measure_streaming_metrics(model_id: str, prompt: str, max_tokens: int = 200) -> Dict:
    """Measure TTFT and TTLT from streaming response."""
    start_time = time.time()

    response_stream = bedrock_client.converse_stream(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": 0.1},
    )

    first_token_time = None
    last_token_time = None
    output_tokens = 0

    for event in response_stream["stream"]:
        current_time = time.time()
        if "contentBlockDelta" in event:
            if first_token_time is None:
                first_token_time = current_time
            last_token_time = current_time
        elif "metadata" in event:
            output_tokens = event["metadata"].get("usage", {}).get("outputTokens", 0)

    ttft_ms = round((first_token_time - start_time) * 1000, 2)
    ttlt_ms = round((last_token_time - start_time) * 1000, 2)

    return {
        "model_id": model_id,
        "ttft_ms": ttft_ms,
        "ttlt_ms": ttlt_ms,
        "generation_time_ms": round(ttlt_ms - ttft_ms, 2),
        "tokens_per_second": round(output_tokens / (ttlt_ms / 1000), 1),
    }
```

## Section 4: Publishing Custom Metrics to CloudWatch

**Concept:** Individual measurements are useful for debugging; aggregated metrics in CloudWatch are useful for operations. By publishing TTFT, TTLT, and cost with a `Model` dimension, you can build dashboards that compare models over time and set alarms on degradation.

**Build:**

```python
def put_custom_operational_cw_metrics(model_id: str, ttft_ms, ttlt_ms, total_cost_usd):
    """Publish TTFT, TTLT, and cost to CloudWatch with model dimension."""
    cloudwatch.put_metric_data(
        Namespace="llm_custom_operational_metrics",
        MetricData=[
            {
                "MetricName": "TimeToFirstToken",
                "Value": ttft_ms,
                "Unit": "Milliseconds",
                "Dimensions": [{"Name": "Model", "Value": model_id}],
            },
            {
                "MetricName": "TimeToLastToken",
                "Value": ttlt_ms,
                "Unit": "Milliseconds",
                "Dimensions": [{"Name": "Model", "Value": model_id}],
            },
            {
                "MetricName": "TotalCost",
                "Value": Decimal(total_cost_usd.replace("$", "")),
                "Dimensions": [{"Name": "Model", "Value": model_id}],
            },
        ],
    )
```

After running the email summarization comparison across models, these metrics appear in CloudWatch under the `llm_custom_operational_metrics` namespace, filterable by model.

## Section 5: Multi-Model Comparison on a Real Workload

**Concept:** Metrics only matter in context. Comparing models on your actual workload (not synthetic prompts) reveals the real cost/speed/quality tradeoffs. An email summarization task shows how model choice impacts every metric simultaneously.

**Build:**

```python
def run_model_comparison(emails: list, models: list, prompt_template: str):
    """Compare models on a real workload, publishing metrics for each call."""
    results = []
    for email in emails:
        prompt = prompt_template.format(email_content=email["content"])
        for model_id in models:
            result = measure_streaming_metrics(model_id, prompt, max_tokens=400)
            if not result.get("error"):
                cost = calculate_cost(model_id, result.get("input_tokens", 0), result.get("output_tokens", 0))
                put_custom_operational_cw_metrics(
                    model_id, result["ttft_ms"], result["ttlt_ms"], cost["total_cost_usd"]
                )
                results.append({**result, **cost, "email_subject": email.get("subject")})
            time.sleep(0.5)  # Avoid throttling
    return results
```

Run this against 2–3 sample emails with all three models. Observe: Nova Lite is fastest/cheapest, Claude is slowest/most expensive but may produce higher quality summaries. The next module evaluates that quality dimension.

## Challenges

### Challenge 1: Build a CloudWatch Dashboard with Alarms

Set up a complete CloudWatch dashboard displaying TTFT, TTLT, cost, and error rate for your models. Then configure alarms for:
- Latency exceeding a threshold you choose
- Throttling errors (5xx responses)
- Cost per hour exceeding a budget

Generate synthetic load by running the email summarization comparison in a loop (10+ iterations) to populate the dashboard and trigger at least one alarm.

**Assessment criteria:**
1. Runs without errors
2. Correct metric dimensions (Model dimension on all custom metrics)
3. Handles missing data (alarm treats missing data as "not breaching")
4. Learner explains threshold choices (why those values?)
5. *Stretch:* Identifies all 3 misconfigurations (see below)

### Challenge 2 (Stretch): Find the Misconfigurations

Using the dashboard you just built in Challenge 1, three intentional misconfigurations have been introduced:

1. One metric uses the wrong `Unit` (e.g., "Seconds" instead of "Milliseconds")
2. One alarm's comparison operator is inverted (triggers when metric is *below* threshold instead of above)
3. One dimension value has a typo causing metrics to split into two unrelated time series

Identify all three, explain why each is problematic, and fix them.

**Assessment criteria:**
1. Runs without errors after fixes
2. Correct metric dimensions
3. Handles missing data
4. Learner explains thresholds
5. Identifies all 3 misconfigurations with explanations

## Wrap-Up

**Key Takeaways:**
- Every LLM call has measurable cost, latency, and throughput — track all three
- TTFT drives user experience; TTLT drives system throughput — optimize for your use case
- CloudWatch custom metrics with dimensions enable per-model comparison over time
- Model choice is a tradeoff: cheaper/faster models may sacrifice quality (evaluated in Module 02)

**This module does NOT cover:**
- Quality/accuracy evaluation of LLM outputs (→ Module 02: Quality Metrics)
- Agentic workflow metrics like tool-use success rate (→ Module 03)
- Guardrails and safety metrics (→ Module 04)
- Automated evaluation frameworks like PromptFoo (→ Module 05)
- Bedrock batch inference pricing and invocation patterns
- Advanced CloudWatch features (anomaly detection, Contributor Insights)

**Next Steps:**
- Module 02 evaluates whether the cheapest/fastest model actually produces *good enough* summaries
- Set up the automatic Bedrock dashboard alongside your custom metrics for full visibility
