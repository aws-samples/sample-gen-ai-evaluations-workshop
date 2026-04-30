---
name: "Operational Metrics for LLMs"
description: "How to monitor LLM performance with CloudWatch, set up latency and cost dashboards, create alarms for Bedrock throttling and errors, publish custom metrics like time to first token"
---

In this module you build a complete CloudWatch observability stack for Amazon Bedrock LLM invocations. You will publish custom metrics (latency, cost, TTFT), create a dashboard that visualizes them, configure alarms for throttling and error conditions, and run synthetic load to verify the alarms fire. By the end you will have a reusable monitoring pattern for any Bedrock-powered application.

## Prerequisites

- AWS account with Amazon Bedrock model access (Nova Lite, Nova Pro, or Claude 3.7 Sonnet)
- Python 3.10+
- `boto3` library installed
- IAM permissions for CloudWatch (`PutMetricData`, `PutDashboard`, `PutMetricAlarm`) and Bedrock Runtime (`InvokeModel`, `Converse`, `ConverseStream`)

## Learning Objectives

- Publish custom CloudWatch metrics with appropriate dimensions using `PutMetricData`
- Create a CloudWatch dashboard that visualizes LLM latency, cost, and error rates
- Configure CloudWatch alarms that detect throttling, elevated latency, and invocation errors
- Generate synthetic Bedrock load to validate alarm thresholds
- Analyze tradeoffs between model speed, cost, and throughput using metric data

## Setup

Install dependencies and configure your Bedrock and CloudWatch clients:

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
```

Verify access by running a quick Bedrock call:

```python
response = bedrock_client.converse(
    modelId="us.amazon.nova-lite-v1:0",
    messages=[{"role": "user", "content": [{"text": "Hello"}]}],
    inferenceConfig={"maxTokens": 10, "temperature": 0.1}
)
print(f"Bedrock OK — latency: {response['metrics']['latencyMs']}ms")
```

---

### Publishing Custom Metrics to CloudWatch

When you run an LLM in production, the built-in Bedrock metrics (invocation count, latency) only tell part of the story. You also need application-level signals — cost per request, time to first token, tokens per second — to make informed decisions about model selection and scaling. CloudWatch custom metrics let you emit these signals alongside AWS-native metrics so everything lives in one place.

The key building block is `PutMetricData`. Each data point belongs to a **namespace** (a logical container) and can carry **dimensions** that let you slice the data — for example, by model ID. Dimensions are critical: without them you cannot compare Nova Lite vs Claude on the same graph.

**Build: Publish a cost metric per invocation**

Create a function that calculates cost from token counts and publishes it to CloudWatch:

```python
MODEL_PRICING = {
    "us.amazon.nova-lite-v1:0": {"input": 0.00006, "output": 0.000015},
    "us.amazon.nova-pro-v1:0": {"input": 0.0008, "output": 0.0002},
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0": {"input": 0.003, "output": 0.015}
}

def publish_cost_metric(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost and publish to CloudWatch."""
    pricing = MODEL_PRICING[model_id]
    total_cost = (input_tokens / 1000) * pricing["input"] + (output_tokens / 1000) * pricing["output"]

    cloudwatch.put_metric_data(
        Namespace='LLMOperationalMetrics',
        MetricData=[{
            'MetricName': 'InvocationCost',
            'Value': total_cost,
            'Dimensions': [{'Name': 'ModelId', 'Value': model_id}]
        }]
    )
    return total_cost
```

Test it by invoking a model and publishing the result:

```python
response = bedrock_client.converse(
    modelId="us.amazon.nova-pro-v1:0",
    messages=[{"role": "user", "content": [{"text": "Summarize cloud computing in one sentence."}]}],
    inferenceConfig={"maxTokens": 100, "temperature": 0.1}
)
cost = publish_cost_metric(
    "us.amazon.nova-pro-v1:0",
    response["usage"]["inputTokens"],
    response["usage"]["outputTokens"]
)
print(f"Published cost: ${cost:.8f}")
```

---

### Measuring Streaming Latency (TTFT and TTLT)

End-to-end latency tells you how long the user waits for a complete answer, but it hides an important detail: how quickly the first token arrives. Time to First Token (TTFT) drives perceived responsiveness — a user who sees text appearing in 300ms feels the system is fast even if the full response takes 4 seconds. Time to Last Token (TTLT) determines throughput capacity.

You measure these by using the `converse_stream` API and recording timestamps as tokens arrive. The difference between the two metrics also reveals generation speed (tokens per second), which varies dramatically across models.

**Build: Stream a response and publish TTFT/TTLT metrics**

```python
def measure_and_publish_streaming_metrics(model_id: str, prompt: str, max_tokens: int = 200) -> Dict:
    """Measure TTFT and TTLT via streaming, publish to CloudWatch."""
    start_time = time.time()

    response_stream = bedrock_client.converse_stream(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": 0.1}
    )

    first_token_time = None
    last_token_time = None
    output_tokens = 0

    for event in response_stream["stream"]:
        current_time = time.time()
        if 'contentBlockDelta' in event:
            if first_token_time is None:
                first_token_time = current_time
            last_token_time = current_time
        elif 'metadata' in event:
            output_tokens = event['metadata'].get('usage', {}).get('outputTokens', 0)

    ttft_ms = round((first_token_time - start_time) * 1000, 2) if first_token_time else 0
    ttlt_ms = round((last_token_time - start_time) * 1000, 2) if last_token_time else 0

    cloudwatch.put_metric_data(
        Namespace='LLMOperationalMetrics',
        MetricData=[
            {'MetricName': 'TimeToFirstToken', 'Value': ttft_ms, 'Unit': 'Milliseconds',
             'Dimensions': [{'Name': 'ModelId', 'Value': model_id}]},
            {'MetricName': 'TimeToLastToken', 'Value': ttlt_ms, 'Unit': 'Milliseconds',
             'Dimensions': [{'Name': 'ModelId', 'Value': model_id}]},
        ]
    )
    return {"model_id": model_id, "ttft_ms": ttft_ms, "ttlt_ms": ttlt_ms, "output_tokens": output_tokens}
```

Run it against two models and compare:

```python
for model in ["us.amazon.nova-pro-v1:0", "us.amazon.nova-lite-v1:0"]:
    result = measure_and_publish_streaming_metrics(model, "Write a haiku about monitoring.", 50)
    print(f"{model}: TTFT={result['ttft_ms']}ms, TTLT={result['ttlt_ms']}ms")
```

---

### Building a CloudWatch Dashboard

Metrics are only useful if you can see them at a glance. A CloudWatch dashboard assembles widgets — time-series graphs, numbers, text — into a single view. For LLM operations you want to see latency trends, cost accumulation, and error spikes side by side so you can correlate issues (e.g., latency spikes that coincide with throttling).

Dashboards are defined as JSON and created via `PutDashboard`. Each widget specifies which metrics to plot, the stat (Average, Sum, Maximum), and the period.

**Build: Create a dashboard with latency and cost widgets**

```python
dashboard_body = {
    "widgets": [
        {
            "type": "metric",
            "x": 0, "y": 0, "width": 12, "height": 6,
            "properties": {
                "title": "Time to First Token (ms)",
                "metrics": [
                    ["LLMOperationalMetrics", "TimeToFirstToken", "ModelId", "us.amazon.nova-pro-v1:0"],
                    ["LLMOperationalMetrics", "TimeToFirstToken", "ModelId", "us.amazon.nova-lite-v1:0"]
                ],
                "stat": "Average",
                "period": 60,
                "region": "us-east-1"
            }
        },
        {
            "type": "metric",
            "x": 12, "y": 0, "width": 12, "height": 6,
            "properties": {
                "title": "Time to Last Token (ms)",
                "metrics": [
                    ["LLMOperationalMetrics", "TimeToLastToken", "ModelId", "us.amazon.nova-pro-v1:0"],
                    ["LLMOperationalMetrics", "TimeToLastToken", "ModelId", "us.amazon.nova-lite-v1:0"]
                ],
                "stat": "Average",
                "period": 60,
                "region": "us-east-1"
            }
        },
        {
            "type": "metric",
            "x": 0, "y": 6, "width": 12, "height": 6,
            "properties": {
                "title": "Invocation Cost (USD)",
                "metrics": [
                    ["LLMOperationalMetrics", "InvocationCost", "ModelId", "us.amazon.nova-pro-v1:0"],
                    ["LLMOperationalMetrics", "InvocationCost", "ModelId", "us.amazon.nova-lite-v1:0"]
                ],
                "stat": "Sum",
                "period": 300,
                "region": "us-east-1"
            }
        },
        {
            "type": "metric",
            "x": 12, "y": 6, "width": 12, "height": 6,
            "properties": {
                "title": "Invocation Errors",
                "metrics": [
                    ["LLMOperationalMetrics", "InvocationError", "ModelId", "us.amazon.nova-pro-v1:0"],
                    ["LLMOperationalMetrics", "InvocationError", "ModelId", "us.amazon.nova-lite-v1:0"]
                ],
                "stat": "Sum",
                "period": 60,
                "region": "us-east-1"
            }
        }
    ]
}

cloudwatch.put_dashboard(
    DashboardName='LLM-Operational-Metrics',
    DashboardBody=json.dumps(dashboard_body)
)
print("Dashboard 'LLM-Operational-Metrics' created.")
```

Open the CloudWatch console → Dashboards → **LLM-Operational-Metrics** to verify it appears.

---

### Configuring Alarms for Throttling and Errors

A dashboard shows you what happened; an alarm tells you when something is wrong *right now*. For LLM workloads the most critical alarm conditions are: (1) latency exceeding your SLA, (2) throttling from Bedrock rate limits, and (3) elevated error rates. Each alarm evaluates a metric over a window and transitions to ALARM state when the threshold is breached.

The `TreatMissingData` setting matters: if your application has bursty traffic, periods with no data points should not trigger or suppress alarms incorrectly. Use `notBreaching` for latency alarms (no data = no problem) and `breaching` for error-count alarms where silence might mean the system is down.

**Build: Create latency and error alarms**

```python
# Alarm: TTLT exceeds 5000ms on average over 3 consecutive periods
cloudwatch.put_metric_alarm(
    AlarmName='LLM-HighLatency-TTLT',
    Namespace='LLMOperationalMetrics',
    MetricName='TimeToLastToken',
    Dimensions=[{'Name': 'ModelId', 'Value': 'us.amazon.nova-pro-v1:0'}],
    Statistic='Average',
    Period=60,
    EvaluationPeriods=3,
    Threshold=5000,
    ComparisonOperator='GreaterThanThreshold',
    TreatMissingData='notBreaching',
    ActionsEnabled=False
)

# Alarm: More than 5 errors in a 1-minute window
cloudwatch.put_metric_alarm(
    AlarmName='LLM-InvocationErrors',
    Namespace='LLMOperationalMetrics',
    MetricName='InvocationError',
    Dimensions=[{'Name': 'ModelId', 'Value': 'us.amazon.nova-pro-v1:0'}],
    Statistic='Sum',
    Period=60,
    EvaluationPeriods=1,
    Threshold=5,
    ComparisonOperator='GreaterThanThreshold',
    TreatMissingData='notBreaching',
    ActionsEnabled=False
)

# Alarm: Throttling — any throttle event in a 1-minute window
cloudwatch.put_metric_alarm(
    AlarmName='LLM-Throttling',
    Namespace='LLMOperationalMetrics',
    MetricName='ThrottleCount',
    Dimensions=[{'Name': 'ModelId', 'Value': 'us.amazon.nova-pro-v1:0'}],
    Statistic='Sum',
    Period=60,
    EvaluationPeriods=1,
    Threshold=0,
    ComparisonOperator='GreaterThanThreshold',
    TreatMissingData='notBreaching',
    ActionsEnabled=False
)

print("Alarms created: LLM-HighLatency-TTLT, LLM-InvocationErrors, LLM-Throttling")
```

---

### Synthetic Load and Alarm Verification

Alarms are only trustworthy if you have verified they fire under the conditions you expect. A synthetic load script sends a burst of requests that intentionally exceeds normal thresholds — triggering throttling, pushing latency up, or generating errors — so you can confirm the alarm transitions to ALARM state.

This is also how you validate your `TreatMissingData` and `EvaluationPeriods` settings before going to production. If an alarm does not fire during synthetic load, your thresholds or periods are too lenient.

**Build: Run synthetic load and publish error/throttle metrics**

```python
def run_synthetic_load(model_id: str, num_requests: int = 20, delay: float = 0.1):
    """Send rapid requests to trigger throttling and publish metrics."""
    errors = 0
    throttles = 0

    for i in range(num_requests):
        try:
            response = bedrock_client.converse(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": f"Request {i}: Summarize AI safety."}]}],
                inferenceConfig={"maxTokens": 50, "temperature": 0.1}
            )
            # Publish latency
            cloudwatch.put_metric_data(
                Namespace='LLMOperationalMetrics',
                MetricData=[{
                    'MetricName': 'TimeToLastToken',
                    'Value': response['metrics']['latencyMs'],
                    'Unit': 'Milliseconds',
                    'Dimensions': [{'Name': 'ModelId', 'Value': model_id}]
                }]
            )
        except bedrock_client.exceptions.ThrottlingException:
            throttles += 1
            cloudwatch.put_metric_data(
                Namespace='LLMOperationalMetrics',
                MetricData=[{
                    'MetricName': 'ThrottleCount',
                    'Value': 1,
                    'Dimensions': [{'Name': 'ModelId', 'Value': model_id}]
                }]
            )
        except Exception as e:
            errors += 1
            cloudwatch.put_metric_data(
                Namespace='LLMOperationalMetrics',
                MetricData=[{
                    'MetricName': 'InvocationError',
                    'Value': 1,
                    'Dimensions': [{'Name': 'ModelId', 'Value': model_id}]
                }]
            )
        time.sleep(delay)

    print(f"Load complete: {num_requests} requests, {throttles} throttled, {errors} errors")
    return {"total": num_requests, "throttles": throttles, "errors": errors}

# Run the load test
results = run_synthetic_load("us.amazon.nova-pro-v1:0", num_requests=20, delay=0.2)
```

After the load completes, check alarm state:

```python
for alarm_name in ['LLM-HighLatency-TTLT', 'LLM-InvocationErrors', 'LLM-Throttling']:
    resp = cloudwatch.describe_alarms(AlarmNames=[alarm_name])
    state = resp['MetricAlarms'][0]['StateValue']
    print(f"{alarm_name}: {state}")
```

---

## Challenges

### Challenge 1: Full Observability Stack

Build a complete CloudWatch observability setup for a Bedrock-powered email summarization workload:

1. Create a CloudWatch dashboard named `Email-Summarizer-Ops` with widgets for: TTFT, TTLT, invocation cost, and error count — each broken down by model dimension.
2. Configure three alarms:
   - Latency alarm: average TTLT > 4000ms over 2 evaluation periods
   - Error alarm: sum of errors > 3 in a single period
   - Throttle alarm: any throttle event in a 60-second window
3. Write a synthetic load script that sends 15 email-summarization requests with minimal delay to trigger at least one alarm.
4. Verify alarm states transition from OK to ALARM.

Use the email summarization prompt from the lesson:

```
You are an AI assistant that summarizes business emails for busy executives.
Analyze the following email and provide: Key Points, Action Items, Deadlines, People Involved, Impact.
```

**Assessment criteria:**

1. All code runs without errors
2. Dashboard uses correct metric dimensions (`ModelId`) and appropriate stats (Average for latency, Sum for cost/errors)
3. Alarms use `TreatMissingData='notBreaching'` and appropriate evaluation periods
4. Synthetic load script publishes metrics on both success and failure paths
5. Learner can explain why they chose their threshold values and evaluation periods

### Challenge 2 (Stretch): Find the Misconfigurations

This challenge reuses the dashboard you built in Challenge 1. Three intentional misconfigurations have been introduced to your working dashboard configuration below. Identify and fix all three.

```python
broken_dashboard = {
    "widgets": [
        {
            "type": "metric",
            "x": 0, "y": 0, "width": 12, "height": 6,
            "properties": {
                "title": "Time to First Token",
                "metrics": [
                    # Misconfig 1: Wrong namespace
                    ["AWS/Bedrock", "TimeToFirstToken", "ModelId", "us.amazon.nova-pro-v1:0"]
                ],
                "stat": "Average",
                "period": 60,
                "region": "us-east-1"
            }
        },
        {
            "type": "metric",
            "x": 12, "y": 0, "width": 12, "height": 6,
            "properties": {
                "title": "Invocation Cost",
                "metrics": [
                    ["LLMOperationalMetrics", "InvocationCost", "ModelId", "us.amazon.nova-pro-v1:0"]
                ],
                # Misconfig 2: Using Average for cost (should be Sum)
                "stat": "Average",
                "period": 300,
                "region": "us-east-1"
            }
        },
        {
            "type": "metric",
            "x": 0, "y": 6, "width": 12, "height": 6,
            "properties": {
                "title": "Error Count",
                "metrics": [
                    # Misconfig 3: Missing dimension — no ModelId filter
                    ["LLMOperationalMetrics", "InvocationError"]
                ],
                "stat": "Sum",
                "period": 60,
                "region": "us-east-1"
            }
        }
    ]
}
```

**Assessment criteria:**

1. Correctly identifies the wrong namespace (`AWS/Bedrock` should be `LLMOperationalMetrics`)
2. Correctly identifies the wrong stat for cost (`Average` should be `Sum`)
3. Correctly identifies the missing dimension on the error metric
4. Learner can explain the operational impact of each misconfiguration

---

## Wrap-Up

You built a full CloudWatch observability stack for Bedrock LLM workloads: custom metrics with model dimensions, a multi-widget dashboard, threshold-based alarms, and a synthetic load harness to validate everything works. This pattern transfers directly to any Bedrock application — swap in your namespace and model IDs and you have production-grade monitoring from day one.

**Feedback:** How did this module go? Was the pacing between concepts comfortable? Let your facilitator know or drop a note in the workshop feedback form.

**Profile update:** You can now add "CloudWatch custom metrics and dashboards for LLM observability" to your builder profile.

**Next module:** Module 02 takes the operational data you are now collecting and layers on *quality* evaluation — measuring whether the LLM outputs are actually good, not just fast and cheap.
