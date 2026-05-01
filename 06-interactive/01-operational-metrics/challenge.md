# Challenge: Operational Metrics with CloudWatch

Build a CloudWatch observability stack for Amazon Bedrock LLM invocations. You will publish custom metrics, create dashboards, configure alarms, and validate them with synthetic load.

## Prerequisites

- [SKILL: Operational Metrics for LLMs](./SKILL.md)
- Source notebook: [01-Operational-Metrics.ipynb](../../01-operational-metrics/01-Operational-Metrics.ipynb)

## Exercise 1: Publish Custom Cost and Latency Metrics

Write a function that invokes a Bedrock model (Nova Lite or Nova Pro), calculates the invocation cost from token counts using a pricing table, and publishes both the cost and end-to-end latency as custom CloudWatch metrics under the namespace `LLMOperationalMetrics`. Use `ModelId` as a dimension.

**Success criteria:**
- Function accepts a model ID and prompt, returns a dict with `cost`, `latency_ms`, `input_tokens`, and `output_tokens`
- Cost calculation uses per-1K-token pricing (e.g., Nova Lite input: $0.00006/1K, output: $0.000015/1K)
- Two `PutMetricData` calls publish `InvocationCost` (no unit) and `EndToEndLatency` (Milliseconds) with `ModelId` dimension
- Run the function against two different models and print the results

## Exercise 2: Measure Streaming Latency (TTFT and TTLT)

Write a function that uses `converse_stream` to measure Time to First Token (TTFT) and Time to Last Token (TTLT). Publish both metrics to CloudWatch.

**Success criteria:**
- Function uses `converse_stream` and records wall-clock timestamps for the first and last `contentBlockDelta` events
- TTFT and TTLT are computed in milliseconds and published to CloudWatch under `LLMOperationalMetrics` with `ModelId` dimension
- Run against at least two models with the same prompt and print a comparison table showing model, TTFT, TTLT, and tokens-per-second

## Exercise 3: Build a CloudWatch Dashboard

Create a CloudWatch dashboard named `LLM-Eval-Workshop` with at least four widgets: TTFT by model, TTLT by model, cumulative invocation cost, and error count.

**Success criteria:**
- Dashboard is created via `put_dashboard` with a valid JSON body
- Each widget references the `LLMOperationalMetrics` namespace with correct `ModelId` dimensions
- Latency widgets use `Average` stat; cost widget uses `Sum` stat; error widget uses `Sum` stat
- Dashboard is visible in the CloudWatch console (verify by calling `list_dashboards`)

## Exercise 4: Configure Alarms and Validate with Synthetic Load

Create three CloudWatch alarms (high latency, error spike, throttling) and write a synthetic load script that sends rapid Bedrock requests to trigger at least one alarm.

**Success criteria:**
- Three alarms created via `put_metric_alarm`: latency > 4000ms average over 2 periods, errors > 3 sum in 1 period, any throttle event in 60s
- All alarms use `TreatMissingData='notBreaching'`
- Synthetic load function sends 15+ requests with minimal delay, publishing metrics on both success and failure paths (catching `ThrottlingException` separately)
- After load completes, call `describe_alarms` and print the state of each alarm
- Explain why you chose each threshold value

## Tips

- Use the SKILL doc's `MODEL_PRICING` dict as your pricing reference
- The `response['metrics']['latencyMs']` field from `converse` gives end-to-end latency without streaming
- For streaming, `time.time()` around the stream iteration loop is more reliable than response metadata
- Set `ActionsEnabled=False` on alarms to avoid triggering SNS notifications during the workshop
