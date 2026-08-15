---
name: Amazon Bedrock AgentCore Evaluation Lifecycle
description: Run, observe, evaluate, and improve agents with Amazon Bedrock AgentCore Runtime, Observability, built-in and custom evaluators, curated datasets, batch and online monitoring, simulation, and controlled optimization. Activate when asked to "learn AgentCore evaluations", "evaluate an AgentCore agent", "use AgentCore built-in evaluators", "build an AgentCore dataset", "create a custom AgentCore evaluator", "set up online evaluation", or "simulate users with AgentCore".
---

# Amazon Bedrock AgentCore: Run, Observe, Evaluate, Improve

Treat AgentCore evaluation as one connected lifecycle:

```text
run -> observe -> evaluate -> find a failure -> improve -> validate again
```

You will reuse one deterministic `CityAnalyst` Runtime throughout the lesson. Its
fixed city dataset and three tools make responses, tool choices, parameters, and
multi-turn behavior stable enough for regression testing.

The source module contains six notebooks. Notebooks 01-04 are the core path;
notebooks 05-06 extend it into production monitoring and optional optimization.

## Prerequisites

- Recommended: Quality Metrics, Understanding Failures, and Agentic Metrics
- Source module: `../../Framework Specific Evaluations/AgentCore/`
- Node.js 20 or later, Python 3.10 or later, and `uv`
- AWS credentials for a workshop or development account
- Access to the configured Amazon Bedrock model
- Permissions for AgentCore, CloudFormation, IAM, CloudWatch, X-Ray, and Bedrock model invocation

AgentCore Runtime hosts an agent loop that you own. AgentCore Harness is a
managed, configuration-driven loop. This lesson uses Runtime so the boundaries
remain visible, but the evaluation concepts also apply to Harness traces.

## Learning Objectives

By the end of this module, you will:

1. Place Runtime, Observability, Evaluations, the CLI, and the Python SDK in one service map
2. Deploy one instrumented Runtime and connect a session to its traces and spans
3. Select built-in evaluators by failure mode and evaluation level
4. Add expected responses, assertions, and expected trajectories as ground truth
5. Turn curated single-turn and multi-turn scenarios into a regression dataset
6. Choose between built-in, focused LLM, and deterministic code evaluators
7. Calibrate a judge against human labels before using it in an automated gate
8. Choose among on-demand, dataset, batch, and sampled online evaluation
9. Use simulation and failure evidence to validate an improvement safely

## Setup

Run from the repository root:

```bash
cd "Framework Specific Evaluations/AgentCore"
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip uninstall -y bedrock-agentcore-starter-toolkit
npm install -g @aws/agentcore
hash -r
agentcore --help
aws sts get-caller-identity
agentcore validate
```

The current CLI is the Node.js package `@aws/agentcore`. Stop if
`agentcore --help` does not include `create`, `dev`, `deploy`, `traces`, and
`run eval`; another installation is still shadowing the current CLI.

Create the small Runtime adapter used by the dataset and simulation sections.
Defining it here keeps the data flow visible throughout the lesson:

```python
from pathlib import Path
from textwrap import dedent

Path("generated").mkdir(exist_ok=True)
Path("generated/interactive_agentcore_helpers.py").write_text(
    dedent(
        '''
        import json
        import os
        from dataclasses import dataclass
        from pathlib import Path
        from typing import Any

        import boto3
        from botocore.config import Config
        from bedrock_agentcore.evaluation import (
            AgentInvokerInput,
            AgentInvokerOutput,
        )

        MODULE_ROOT = Path(__file__).resolve().parents[1]

        @dataclass(frozen=True)
        class RuntimeInfo:
            runtime_id: str
            runtime_arn: str
            region: str
            endpoint_name: str

            @property
            def log_group_name(self):
                return (
                    "/aws/bedrock-agentcore/runtimes/"
                    f"{self.runtime_id}-{self.endpoint_name}"
                )

        def load_runtime_info(runtime_name="CityAnalyst"):
            state = json.loads(
                (
                    MODULE_ROOT
                    / "agentcore"
                    / ".cli"
                    / "deployed-state.json"
                ).read_text()
            )
            targets = state.get("targets", {})
            if not targets:
                raise RuntimeError(
                    "No deployed target found. Complete Section 1 first."
                )

            target_name, target = next(iter(targets.items()))
            runtime = (
                target.get("resources", {})
                .get("runtimes", {})
                .get(runtime_name)
            )
            if runtime is None:
                raise RuntimeError(
                    f"Runtime {runtime_name!r} is not deployed."
                )

            target_specs = json.loads(
                (
                    MODULE_ROOT / "agentcore" / "aws-targets.json"
                ).read_text()
            )
            region = next(
                (
                    item["region"]
                    for item in target_specs
                    if item.get("name") == target_name
                ),
                boto3.Session().region_name,
            )
            if not region:
                raise RuntimeError("Could not determine an AWS region.")
            return RuntimeInfo(
                runtime_id=runtime["runtimeId"],
                runtime_arn=runtime["runtimeArn"],
                region=region,
                endpoint_name=os.getenv(
                    "AGENTCORE_RUNTIME_ENDPOINT",
                    "DEFAULT",
                ),
            )

        class RuntimeInvoker:
            def __init__(self, runtime_info):
                self.runtime_info = runtime_info
                self.client = boto3.client(
                    "bedrock-agentcore",
                    region_name=runtime_info.region,
                    config=Config(
                        retries={
                            "total_max_attempts": 5,
                            "mode": "adaptive",
                        },
                        connect_timeout=5,
                        read_timeout=180,
                    ),
                )

            def invoke(self, prompt, session_id):
                response = self.client.invoke_agent_runtime(
                    agentRuntimeArn=self.runtime_info.runtime_arn,
                    qualifier=self.runtime_info.endpoint_name,
                    runtimeSessionId=session_id,
                    payload=json.dumps(
                        {"prompt": prompt}
                    ).encode("utf-8"),
                )
                return json.loads(response["response"].read())

            def __call__(self, invoker_input: AgentInvokerInput):
                payload = invoker_input.payload
                if hasattr(payload, "model_dump"):
                    payload = payload.model_dump(mode="json")
                prompt = (
                    payload
                    if isinstance(payload, str)
                    else json.dumps(payload)
                )
                if not invoker_input.session_id:
                    raise ValueError(
                        "Dataset runner did not provide a session ID."
                    )
                output = self.invoke(
                    prompt=prompt,
                    session_id=invoker_input.session_id,
                )
                return AgentInvokerOutput(agent_output=output)
        '''
    ).lstrip()
)
```

## Section 1: Run and Observe One Shared Agent

**Concept:** Runtime answers where the agent runs; Observability records what
happened; Evaluations score a specific session, trace, or tool call.

```text
session
  +-- trace: one user turn and agent response
        +-- model spans
        +-- tool-call spans
        +-- application spans
```

The evaluation levels answer different questions:

| Level | Unit | Example question |
|---|---|---|
| `SESSION` | Whole conversation or task | Did the user achieve the goal? |
| `TRACE` | One agent turn | Was this response correct? |
| `TOOL_CALL` | One instrumented tool execution | Was this the right tool? |

`CityAnalyst` uses `lookup_city`, `compare_cities`, and `calculate_density`
against a fixed fixture, removing changing web results from the baseline.

**Build:**

CloudWatch Transaction Search is a one-time account prerequisite for searchable
traces. Enable it in the CloudWatch console, or run this once with the account
owner's approval:

```bash
aws xray update-trace-segment-destination --destination CloudWatchLogs
aws xray get-trace-segment-destination

agentcore deploy
agentcore status

mkdir -p generated
export SESSION_ID=$(python -c 'import secrets; print("eval-" + secrets.token_hex(14))')
agentcore invoke \
  --runtime CityAnalyst \
  --session-id "$SESSION_ID" \
  --prompt "Compare Seattle, WA with Portland, OR. Which city is denser?" \
  --json

printf '{"session_id":"%s"}\n' "$SESSION_ID" \
  > generated/interactive_session.json

TRACE_ID=""
for attempt in $(seq 1 12); do
  trace_list=$(
    NO_COLOR=1 agentcore traces list \
      --runtime CityAnalyst \
      --since 30m \
      --limit 50
  )
  printf '%s\n' "$trace_list"
  TRACE_ID=$(
    printf '%s\n' "$trace_list" |
      awk -v session="$SESSION_ID" '$NF == session {print $1; exit}'
  )
  if [ -n "$TRACE_ID" ]; then
    break
  fi
  sleep 10
done

if [ -z "$TRACE_ID" ]; then
  echo "No trace appeared for session $SESSION_ID within 120 seconds." >&2
  exit 1
fi

agentcore traces get "$TRACE_ID" \
  --runtime CityAnalyst \
  --output generated/interactive_trace.json
```

**Inspect the span hierarchy:**

```python
import json
import os
from pathlib import Path

trace_document = json.loads(
    Path("generated/interactive_trace.json").read_text()
)

def collect_spans(node):
    spans = []
    if isinstance(node, dict):
        if "spanId" in node and ("traceId" in node or "name" in node):
            spans.append(node)
        for value in node.values():
            spans.extend(collect_spans(value))
    elif isinstance(node, list):
        for value in node:
            spans.extend(collect_spans(value))
    return spans

span_rows = []
for span in collect_spans(trace_document):
    attributes = span.get("attributes", {})
    span_rows.append(
        {
            "name": span.get("name", ""),
            "span_id": span.get("spanId", ""),
            "parent_span_id": span.get("parentSpanId", ""),
            "operation": attributes.get("gen_ai.operation.name", ""),
            "tool": attributes.get("gen_ai.tool.name", ""),
        }
    )

for row in span_rows:
    print(row)
```

On first enablement, wait until the destination reports `CloudWatchLogs` and
`ACTIVE`; initial search availability can take up to 10 minutes. After that
one-time setup, complete traces normally become searchable in under about
10 seconds. The bounded loop accommodates ordinary ingestion latency and
reports a clear timeout when telemetry does not arrive.

Before continuing, explain why the session ID joins invocation, telemetry, and evaluation.

## Section 2: Evaluate Recent Behavior with Built-Ins

**Concept:** Start with the smallest built-in set that tests a suspected failure.
Running every evaluator increases cost and creates noisy results. AgentCore
currently provides 13 managed LLM-based evaluators:

| Level | Evaluator | Failure it detects |
|---|---|---|
| `SESSION` | `Builtin.GoalSuccessRate` | The conversation did not complete the user's goal |
| `TRACE` | `Builtin.Helpfulness` | The response was not useful |
| `TRACE` | `Builtin.Correctness` | Claims conflict with facts or references |
| `TRACE` | `Builtin.Faithfulness` | Claims are unsupported by supplied context |
| `TRACE` | `Builtin.Harmfulness` | The response contains harmful content |
| `TRACE` | `Builtin.Stereotyping` | The response generalizes about people or groups |
| `TRACE` | `Builtin.Refusal` | The response refuses inappropriately |
| `TRACE` | `Builtin.Coherence` | The response is not logically coherent |
| `TRACE` | `Builtin.ResponseRelevance` | The response does not address the request |
| `TRACE` | `Builtin.Conciseness` | The response is needlessly verbose or incomplete |
| `TRACE` | `Builtin.InstructionFollowing` | The agent ignored its instructions or output contract |
| `TOOL_CALL` | `Builtin.ToolSelectionAccuracy` | The agent selected the wrong tool |
| `TOOL_CALL` | `Builtin.ToolParameterAccuracy` | The agent extracted incorrect tool parameters |

Ground-truth trajectories add three deterministic session-level evaluators:

| Evaluator | Match rule |
|---|---|
| `Builtin.TrajectoryExactOrderMatch` | Same tools, same order, no extras |
| `Builtin.TrajectoryInOrderMatch` | Expected tools occur in order; extras are allowed |
| `Builtin.TrajectoryAnyOrderMatch` | Expected tools occur in any order; extras are allowed |

This is the current service catalog. Node CLI `1.0.0-preview.22` has local
target-level metadata for ten evaluator IDs and defaults unrecognized
`Builtin.*` IDs to `SESSION`. Check `agentcore --version` and use the Python or
AWS SDK when the installed CLI does not recognize targeted `Harmfulness`,
`Stereotyping`, or `ToolParameterAccuracy` evaluation. The examples below use
CLI-compatible IDs.

Expected responses and assertions are semantic references rather than
exact-string rules. Expected trajectories are evaluated by the deterministic
matching rule selected above.

**Build with the CLI:**

```bash
export SESSION_ID=$(
  python -c 'import json; print(json.load(open("generated/interactive_session.json"))["session_id"])'
)

agentcore run eval \
  --runtime CityAnalyst \
  --session-id "$SESSION_ID" \
  --evaluator \
    Builtin.GoalSuccessRate \
    Builtin.Correctness \
    Builtin.TrajectoryExactOrderMatch \
  --expected-response \
    "Seattle is more populous and denser than Portland in the workshop dataset." \
  --assertion \
    "The response compares both requested cities and identifies Seattle as denser." \
  --expected-trajectory compare_cities \
  --json

agentcore evals history --limit 10
```

**Automate the same evaluation with Python:**

```python
import json
from datetime import timedelta
from pathlib import Path

from bedrock_agentcore.evaluation import EvaluationClient, ReferenceInputs

state = json.loads(Path("agentcore/.cli/deployed-state.json").read_text())
target_name, target_state = next(iter(state["targets"].items()))
runtime = target_state["resources"]["runtimes"]["CityAnalyst"]

target_specs = json.loads(Path("agentcore/aws-targets.json").read_text())
region = next(
    item["region"] for item in target_specs if item["name"] == target_name
)
session_id = json.loads(Path("generated/interactive_session.json").read_text())["session_id"]

results = EvaluationClient(region_name=region).run(
    evaluator_ids=[
        "Builtin.GoalSuccessRate",
        "Builtin.Correctness",
        "Builtin.TrajectoryExactOrderMatch",
    ],
    session_id=session_id,
    agent_id=runtime["runtimeId"],
    look_back_time=timedelta(hours=1),
    reference_inputs=ReferenceInputs(
        expected_response=(
            "Seattle is more populous and denser than Portland "
            "in the workshop dataset."
        ),
        assertions=[
            "The response compares Seattle, WA and Portland, OR.",
            "The response identifies Seattle as denser.",
        ],
        expected_trajectory=["compare_cities"],
    ),
)

for item in results:
    print(
        item.get("evaluatorId"),
        item.get("value"),
        item.get("label"),
        item.get("explanation"),
    )
```

Interpret `value`, `label`, and `explanation` separately. Do not turn an
arbitrary score such as `0.8` into a production gate before calibrating the
evaluator against representative human labels.

## Section 3: Add Ground Truth and Curated Datasets

**Concept:** One-session evaluation diagnoses a trace; a curated dataset creates
a repeatable benchmark that invokes the agent under controlled conditions.

Use each reference type for a distinct purpose:

| Reference | Best for |
|---|---|
| `expected_response` | Semantic answer comparison |
| `assertions` | Requirements with multiple valid answers |
| `expected_trajectory` | Intended tool choice or sequence |

A useful dataset includes common tasks, historical failures, no-tool requests,
missing data, invalid parameters, similar tools, and multi-turn follow-ups.

**Inspect the checked-in scenarios:**

```python
import json
import os
from pathlib import Path

from bedrock_agentcore.evaluation import (
    CloudWatchAgentSpanCollector,
    Dataset,
    EvaluationRunConfig,
    EvaluatorConfig,
    OnDemandEvaluationDatasetRunner,
    PredefinedScenario,
    Turn,
)
from generated.interactive_agentcore_helpers import (
    RuntimeInvoker,
    load_runtime_info,
)

records = [
    json.loads(line)
    for line in Path("data/city_scenarios.jsonl").read_text().splitlines()
    if line.strip()
]

scenarios = [
    PredefinedScenario(
        scenario_id=item["scenario_id"],
        turns=[
            Turn(
                input=turn["input"],
                expected_response=turn.get("expectedResponse"),
            )
            for turn in item["turns"]
        ],
        assertions=item.get("assertions"),
        expected_trajectory=item.get("expected_trajectory"),
    )
    for item in records
]

dataset = Dataset(scenarios=scenarios)
for scenario in dataset.scenarios:
    print(scenario.scenario_id, len(scenario.turns))

runtime_info = load_runtime_info()
invoker = RuntimeInvoker(runtime_info)
collector = CloudWatchAgentSpanCollector(
    log_group_name=runtime_info.log_group_name,
    region=runtime_info.region,
    max_wait_seconds=180,
    poll_interval_seconds=10,
)
run_config = EvaluationRunConfig(
    evaluator_config=EvaluatorConfig(
        evaluator_ids=[
            "Builtin.GoalSuccessRate",
            "Builtin.Correctness",
            "Builtin.TrajectoryExactOrderMatch",
        ]
    ),
    evaluation_delay_seconds=0,
    max_concurrent_scenarios=3,
)

small_dataset = Dataset(scenarios=scenarios[:3])
dataset_result = OnDemandEvaluationDatasetRunner(
    region=runtime_info.region
).run(
    config=run_config,
    dataset=small_dataset,
    agent_invoker=invoker,
    span_collector=collector,
)
result_label = os.getenv("AGENTCORE_RESULT_LABEL", "baseline")
result_path = Path(
    f"generated/city_dataset_{result_label}.json"
)
result_path.write_text(dataset_result.model_dump_json(indent=2) + "\n")
print(result_path)
print(dataset_result.model_dump_json(indent=2))
```

This single block loads the scenarios, invokes the first three, polls their
CloudWatch spans, evaluates them, and saves the result. The inline
`RuntimeInvoker` isolates the Runtime protocol details so the learner can focus
on dataset design. Set `AGENTCORE_RESULT_LABEL=candidate` when repeating the
same block against a candidate endpoint.

`CloudWatchAgentSpanCollector` polls until the telemetry appears, so the fixed
evaluation delay remains `0`. Use a managed dataset when scenarios must be
shared and versioned through the CLI:

```bash
agentcore add dataset \
  --name CityRegression \
  --schema-type AGENTCORE_EVALUATION_PREDEFINED_V1 \
  --description "Stable CityAnalyst regression scenarios"

mkdir -p agentcore/datasets
cp data/city_scenarios.jsonl agentcore/datasets/CityRegression.jsonl

agentcore deploy

agentcore run eval \
  --runtime CityAnalyst \
  --dataset CityRegression \
  --evaluator \
    Builtin.GoalSuccessRate \
    Builtin.Correctness \
    Builtin.TrajectoryExactOrderMatch
```

Start with three scenarios before paying to invoke and evaluate the full pack.

## Section 4: Build and Calibrate Custom Evaluators

**Concept:** Built-ins are the default. Create a custom evaluator only for a
product-, schema-, domain-, or policy-specific failure.

Choose the narrowest reliable mechanism:

| Need | Evaluator |
|---|---|
| General quality criterion already in the catalog | Built-in |
| Semantic product rule | Focused binary LLM judge |
| Exact schema, type, range, or policy check | Deterministic code |

Avoid a broad prompt that scores many qualities on a 1–5 scale. One failure mode
and a binary decision make changes easier to diagnose and calibrate.

**Register a focused binary LLM judge:**

```bash
agentcore add evaluator \
  --name FactualConsistency \
  --level TRACE \
  --model global.anthropic.claude-sonnet-4-6 \
  --instructions "Evaluate only factual consistency. Pass when every city fact and comparison agrees with the reference information. Fail wrong numbers, reversed comparisons, substituted cities, invented missing data, or unanswered factual questions. Ignore style and verbosity. Reference and conversation context: {context}. Assistant response: {assistant_turn}." \
  --rating-scale pass-fail

agentcore deploy
```

Verify that the evaluator model or inference profile is supported in your
region before deployment.

**Unit-test the typed code evaluator before registration:**

```python
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from bedrock_agentcore.evaluation.custom_code_based_evaluators import (
    EvaluatorInput,
    EvaluatorOutput,
    custom_code_based_evaluator,
)

REQUIRED_TOP_LEVEL_KEYS = {"answer", "cities", "tools_used"}
RESPONSE_ATTRIBUTE_KEYS = (
    "gen_ai.response.content",
    "gen_ai.completion",
    "gen_ai.output.messages",
    "output",
)

def candidate_text(value: Any):
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "content", "output"):
            candidate = candidate_text(value.get(key))
            if candidate:
                return candidate
    if isinstance(value, list):
        for item in reversed(value):
            candidate = candidate_text(item)
            if candidate:
                return candidate
    return None

def extract_response_text(session_spans):
    for span in reversed(session_spans):
        attributes = span.get("attributes", {})
        for key in RESPONSE_ATTRIBUTE_KEYS:
            if key in attributes:
                candidate = candidate_text(attributes[key])
                if candidate:
                    return candidate
    return None

def validate_response_schema(response_text):
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError:
        return False, "The final response is not valid JSON."
    if not isinstance(payload, dict):
        return False, "The final response must be a JSON object."

    missing = REQUIRED_TOP_LEVEL_KEYS - payload.keys()
    if missing:
        return False, f"Missing required keys: {', '.join(sorted(missing))}."
    if not isinstance(payload["answer"], str):
        return False, "answer must be a string."
    if not isinstance(payload["cities"], list):
        return False, "cities must be a list."
    if not isinstance(payload["tools_used"], list) or not all(
        isinstance(item, str) for item in payload["tools_used"]
    ):
        return False, "tools_used must be a list of strings."
    return True, "The final response matches the required JSON schema."

@custom_code_based_evaluator()
def handler(evaluator_input: EvaluatorInput, context):
    response_text = extract_response_text(evaluator_input.session_spans)
    if response_text is None:
        return EvaluatorOutput(
            value=0.0,
            label="Fail",
            explanation="No final response text was found in the spans.",
        )

    passed, explanation = validate_response_schema(response_text)
    return EvaluatorOutput(
        value=1.0 if passed else 0.0,
        label="Pass" if passed else "Fail",
        explanation=explanation,
    )

fixture_spans = json.loads(Path("data/sample_spans.json").read_text())
passing_input = EvaluatorInput(
    evaluation_level="TRACE",
    session_spans=fixture_spans,
    target_trace_id=fixture_spans[0]["traceId"],
    evaluator_id="local-response-format",
    evaluator_name="ResponseFormat",
)
passing_result = handler.unwrapped(passing_input, context=None)
assert passing_result.value == 1.0

failing_spans = deepcopy(fixture_spans)
failing_spans[-1]["attributes"]["gen_ai.response.content"] = (
    "Seattle has 780995 residents."
)
failing_input = EvaluatorInput(
    evaluation_level="TRACE",
    session_spans=failing_spans,
    target_trace_id=failing_spans[0]["traceId"],
    evaluator_id="local-response-format",
    evaluator_name="ResponseFormat",
)
failing_result = handler.unwrapped(failing_input, context=None)
assert failing_result.value == 0.0

print(passing_result.model_dump())
print(failing_result.model_dump())
```

The decorator exposes the original typed function as `handler.unwrapped`, so
these fixtures test evaluator logic without Lambda event parsing or deployment.
Validate either judge before automating a decision:

1. Split human-labeled examples by question family
2. Run the evaluator on development and held-out records
3. Inspect every judge-human disagreement
4. Measure accuracy, true-positive rate, true-negative rate, and repeatability
5. Revise definitions and examples rather than changing inconvenient labels

Run the deployed judge on every checked-in fixture through the `Evaluate` API.
This path constructs trace-level Strands spans directly, so calibration does
not require a Runtime invocation or a CloudWatch ingestion wait:

```python
import hashlib
import json
import time
from pathlib import Path

import boto3
from botocore.config import Config

validation_examples = [
    json.loads(line)
    for line in Path("data/judge_validation.jsonl").read_text().splitlines()
    if line.strip()
]

target_specs = json.loads(Path("agentcore/aws-targets.json").read_text())
region = next(
    (
        item["region"]
        for item in target_specs
        if item.get("region")
    ),
    boto3.Session().region_name,
)
if not region:
    raise RuntimeError("Configure an AWS region before calibration.")

client_config = Config(
    retries={"total_max_attempts": 5, "mode": "adaptive"},
    connect_timeout=5,
    read_timeout=180,
)
control_client = boto3.client(
    "bedrock-agentcore-control",
    region_name=region,
    config=client_config,
)
evaluation_client = boto3.client(
    "bedrock-agentcore",
    region_name=region,
    config=client_config,
)

def resolve_active_evaluator_id(evaluator_name):
    matches = []
    request = {}
    while True:
        page = control_client.list_evaluators(**request)
        matches.extend(
            item
            for item in page["evaluators"]
            if item["evaluatorName"] == evaluator_name
        )
        next_token = page.get("nextToken")
        if not next_token:
            break
        request = {"nextToken": next_token}

    active = [item for item in matches if item["status"] == "ACTIVE"]
    if len(active) != 1:
        statuses = [
            f"{item['evaluatorId']}={item['status']}" for item in matches
        ]
        raise RuntimeError(
            f"Expected one ACTIVE {evaluator_name} evaluator; found {statuses}."
        )
    return active[0]["evaluatorId"]

def fixture_span(record):
    digest = hashlib.sha256(record["id"].encode("utf-8")).hexdigest()
    trace_id = digest[:32]
    span_id = digest[32:48]
    session_id = f"calibration-{digest}"
    start_time = time.time_ns()
    duration = 1_000_000
    span = {
        "traceId": trace_id,
        "spanId": span_id,
        "name": "invoke_agent CityAnalystCalibration",
        "kind": "INTERNAL",
        "scope": {"name": "strands.telemetry.tracer"},
        "startTimeUnixNano": start_time,
        "endTimeUnixNano": start_time + duration,
        "durationNano": duration,
        "attributes": {
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.agent.name": "CityAnalystCalibration",
            "session.id": session_id,
        },
        "events": [
            {
                "name": "gen_ai.user.message",
                "timeUnixNano": start_time + 100_000,
                "attributes": {
                    "content": json.dumps(
                        [{"text": record["question"]}]
                    )
                },
            },
            {
                "name": "gen_ai.choice",
                "timeUnixNano": start_time + 900_000,
                "attributes": {
                    "message": record["response"],
                    "finish_reason": "end_turn",
                },
            },
        ],
        "status": {"code": "OK"},
    }
    return session_id, trace_id, [span]

def evaluate_fixture(evaluator_id, record):
    session_id, trace_id, spans = fixture_span(record)
    response = evaluation_client.evaluate(
        evaluatorId=evaluator_id,
        evaluationInput={"sessionSpans": spans},
        evaluationTarget={"traceIds": [trace_id]},
        evaluationReferenceInputs=[
            {
                "context": {
                    "spanContext": {
                        "sessionId": session_id,
                        "traceId": trace_id,
                    }
                },
                "expectedResponse": {"text": record["reference"]},
            }
        ],
    )
    results = response["evaluationResults"]
    if len(results) != 1:
        raise RuntimeError(
            f"{record['id']} returned {len(results)} evaluation results."
        )
    result = results[0]
    if result.get("errorCode") or result.get("errorMessage"):
        raise RuntimeError(
            f"{record['id']} failed: "
            f"{result.get('errorCode')} {result.get('errorMessage')}"
        )

    label = str(result.get("label", "")).strip().lower()
    if label not in {"pass", "fail"}:
        raise RuntimeError(
            f"{record['id']} returned unexpected label {result.get('label')!r}."
        )
    return {
        "judge_label": label,
        "judge_value": result.get("value"),
        "judge_explanation": result.get("explanation"),
    }

def judge_scorecard(records):
    if len(records) != 8:
        raise ValueError("Calibration requires all 8 checked-in fixtures.")
    true_pass = [item for item in records if item["human_label"] == "pass"]
    true_fail = [item for item in records if item["human_label"] == "fail"]
    return {
        "examples": len(records),
        "accuracy": sum(
            item["human_label"] == item["judge_label"]
            for item in records
        ) / len(records),
        "true_positive_rate": sum(
            item["judge_label"] == "pass" for item in true_pass
        ) / len(true_pass),
        "true_negative_rate": sum(
            item["judge_label"] == "fail" for item in true_fail
        ) / len(true_fail),
    }

factual_consistency_id = resolve_active_evaluator_id(
    "FactualConsistency"
)
for item in validation_examples:
    item.update(evaluate_fixture(factual_consistency_id, item))
    print(
        item["id"],
        item["human_label"],
        item["judge_label"],
        item["judge_explanation"],
    )

scorecard = judge_scorecard(validation_examples)
Path("generated").mkdir(exist_ok=True)
Path("generated/factual_consistency_calibration.json").write_text(
    json.dumps(
        {
            "evaluator_id": factual_consistency_id,
            "scorecard": scorecard,
            "records": validation_examples,
        },
        indent=2,
    )
    + "\n"
)
print(scorecard)
```

Automate a decision only after the error tradeoff matches the operational use.

## Section 5: Operate the Evaluate-Improve Loop

**Concept:** Choose an execution mode from the decision you need to make.

Current service status as of August 14, 2026:

- AgentCore Evaluations is generally available
- Dataset evaluation runners remain in public preview
- Batch evaluations, recommendations, and A/B tests are generally available
- AgentCore Insights and the Python SDK simulation surface remain in preview

Availability varies by region, and optional CLI surfaces evolve. Verify feature
availability in the target region and installed CLI before running the optional
examples.

| Need | Mechanism |
|---|---|
| Investigate one recent session | On-demand evaluation |
| Run scenarios that invoke the agent | Curated dataset |
| Re-score existing historical sessions | Batch evaluation |
| Continuously sample live traces | Online evaluation |

Batch and online modes reuse the same evaluator design. Online evaluation is
asynchronous; it does not block the user's response.

**Run a historical evaluation:**

```bash
agentcore run batch-evaluation \
  --runtime CityAnalyst \
  --evaluator Builtin.Helpfulness Builtin.GoalSuccessRate \
  --lookback-days 1 \
  --name CityAnalystRegression \
  --wait
```

**Make the release gate fail closed:**

```python
import json
import subprocess

def run_quality_gate(
    runtime_name: str,
    evaluator_id: str,
    threshold: float,
) -> dict:
    completed = subprocess.run(
        [
            "agentcore", "run", "eval",
            "--runtime", runtime_name,
            "--evaluator", evaluator_id,
            "--days", "1",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    if payload.get("success") is False:
        raise RuntimeError(payload.get("error", "Evaluation command failed."))

    run = payload.get("run", payload)
    rows = run.get("results", [])
    if not rows:
        raise RuntimeError("Quality gate found no evaluation results.")

    row = next(
        (
            item for item in rows
            if item.get("evaluator") == evaluator_id
            or item.get("evaluatorId") == evaluator_id
        ),
        None,
    )
    if row is None:
        raise RuntimeError(f"No result found for {evaluator_id}.")

    session_errors = [
        item.get("errorMessage")
        for item in row.get("sessionScores", [])
        if item.get("errorMessage")
    ]
    if session_errors:
        raise RuntimeError(f"Evaluator errors: {session_errors}")

    score = row.get("aggregateScore")
    if not isinstance(score, (int, float)):
        raise RuntimeError("Evaluation result did not include aggregateScore.")
    if score < threshold:
        raise RuntimeError(
            f"Quality gate failed: {score:.3f} < {threshold:.3f}"
        )
    return {"score": score, "threshold": threshold, "passed": True}

WORKSHOP_THRESHOLD = 0.70
print(
    run_quality_gate(
        runtime_name="CityAnalyst",
        evaluator_id="Builtin.Helpfulness",
        threshold=WORKSHOP_THRESHOLD,
    )
)
```

Replace the lesson's `WORKSHOP_THRESHOLD` with the value justified by judge
calibration before using the gate for a real release.

**Add sampled online monitoring:**

```bash

agentcore add online-eval \
  --name CityQualityMonitor \
  --runtime CityAnalyst \
  --evaluator \
    Builtin.Helpfulness \
    Builtin.GoalSuccessRate \
    Builtin.ToolSelectionAccuracy \
  --sampling-rate 100 \
  --enable-on-create

agentcore deploy --diff
agentcore deploy

agentcore logs evals --runtime CityAnalyst --since 1h

agentcore pause online-eval CityQualityMonitor
agentcore resume online-eval CityQualityMonitor
```

Use `100%` sampling only for controlled workshop traffic. Start production near
`1-5%`, then adjust using risk, volume, cost, and human-review evidence.

Before production, set CloudWatch retention, encrypt sensitive result paths,
restrict trace access, scrub PII before export, monitor evaluator errors
separately from low scores, version evaluator definitions and thresholds with
the application, and preserve a small human-review sample to detect judge
drift.

**Discover metrics before creating alarms:**

```python
import json
from pathlib import Path

import boto3
from botocore.config import Config

target_specs = json.loads(Path("agentcore/aws-targets.json").read_text())
region = next(
    (
        item["region"]
        for item in target_specs
        if item.get("region")
    ),
    boto3.Session().region_name,
)
if not region:
    raise RuntimeError("Configure an AWS region before querying CloudWatch.")

cloudwatch = boto3.client(
    "cloudwatch",
    region_name=region,
    config=Config(
        retries={"total_max_attempts": 5, "mode": "adaptive"},
        connect_timeout=5,
        read_timeout=30,
    ),
)

namespace = "Bedrock-AgentCore/Evaluations"
discovered = []
paginator = cloudwatch.get_paginator("list_metrics")
for page in paginator.paginate(Namespace=namespace):
    for metric in page.get("Metrics", []):
        discovered.append(
            {
                "namespace": namespace,
                "metric_name": metric["MetricName"],
                "dimensions": metric.get("Dimensions", []),
            }
        )

if not discovered:
    raise RuntimeError(
        "No AgentCore metrics found. Generate traffic and wait for ingestion."
    )

for metric in sorted(
    discovered,
    key=lambda item: (item["namespace"], item["metric_name"]),
):
    print(json.dumps(metric, sort_keys=True))
```

CloudWatch namespace, metric, and dimension names are case-sensitive and may
evolve. Discover the emitted metrics and create alarms only after confirming
which deployed metric represents evaluator errors or a calibrated aggregate
score.

Simulation targets interaction states that are rare or awkward to script. Give
the actor a goal, context, traits, turn limit, and assertions. Treat low scores
and insight clusters as hypotheses until trace and human review support them.

**Run one bounded simulation:**

```python
import os

from bedrock_agentcore.evaluation import (
    ActorProfile,
    CloudWatchAgentSpanCollector,
    Dataset,
    EvaluationRunConfig,
    EvaluatorConfig,
    OnDemandEvaluationDatasetRunner,
    SimulatedScenario,
    SimulationConfig,
)
from generated.interactive_agentcore_helpers import (
    RuntimeInvoker,
    load_runtime_info,
)

runtime_info = load_runtime_info()
simulated_dataset = Dataset(
    scenarios=[
        SimulatedScenario(
            scenario_id="ambiguous-portland-follow-up",
            scenario_description=(
                "A beginner asks about Portland without a state, then "
                "clarifies and requests a Seattle comparison."
            ),
            actor_profile=ActorProfile(
                traits={"communication_style": "brief follow-up questions"},
                context="The user means Portland, Oregon.",
                goal=(
                    "Obtain Portland facts and compare them with Seattle."
                ),
            ),
            input="Tell me about Portland.",
            max_turns=6,
            assertions=[
                "The agent clarifies the state before lookup.",
                "The final comparison uses Portland, OR and Seattle, WA.",
            ],
        )
    ]
)
simulation_config = EvaluationRunConfig(
    evaluator_config=EvaluatorConfig(
        evaluator_ids=[
            "Builtin.GoalSuccessRate",
            "Builtin.InstructionFollowing",
            "Builtin.Helpfulness",
        ]
    ),
    evaluation_delay_seconds=0,
    max_concurrent_scenarios=1,
    simulation_config=SimulationConfig(
        model_id=os.getenv(
            "AGENTCORE_SIMULATOR_MODEL_ID",
            "global.anthropic.claude-sonnet-4-6",
        )
    ),
)
collector = CloudWatchAgentSpanCollector(
    log_group_name=runtime_info.log_group_name,
    region=runtime_info.region,
    max_wait_seconds=180,
    poll_interval_seconds=10,
)
simulation_result = OnDemandEvaluationDatasetRunner(
    region=runtime_info.region
).run(
    config=simulation_config,
    dataset=simulated_dataset,
    agent_invoker=RuntimeInvoker(runtime_info),
    span_collector=collector,
)
print(simulation_result.model_dump_json(indent=2))
```

Simulation invokes both the actor model and the agent. Keep concurrency and turn
count low until the scenario proves useful.

**Turn failure evidence into a candidate improvement:**

```bash
set -euo pipefail
mkdir -p generated
cp \
  app/CityAnalyst/system_prompt.txt \
  generated/system_prompt.baseline.txt

agentcore run insights \
  --runtime CityAnalyst \
  --insights \
    Builtin.Insight.FailureAnalysis \
    Builtin.Insight.UserIntent \
  --evaluator Builtin.GoalSuccessRate \
  --lookback-days 7 \
  --name CityAnalystFailures \
  --wait \
  --json \
  > generated/city_insights.json

INSIGHTS_ID=$(
  python -c \
    'import json; print(json.load(open("generated/city_insights.json"))["id"])'
)
agentcore view insights "$INSIGHTS_ID" --json

agentcore run recommendation \
  --type system-prompt \
  --from-insights "$INSIGHTS_ID" \
  --prompt-file app/CityAnalyst/system_prompt.txt \
  --run CityPromptRecommendation \
  --wait \
  --json \
  > generated/city_recommendation.json

RECOMMENDATION_ID=$(
  python -c \
    'import json; print(json.load(open("generated/city_recommendation.json"))["id"])'
)
agentcore view recommendation "$RECOMMENDATION_ID" --json \
  > generated/city_recommendation_detail.json

python - <<'PY'
import json
from pathlib import Path

payload = json.loads(
    Path("generated/city_recommendation_detail.json").read_text()
)

def find_value(node, keys):
    if isinstance(node, dict):
        for key in keys:
            value = node.get(key)
            if isinstance(value, str) and value.strip():
                return value
        for value in node.values():
            found = find_value(value, keys)
            if found:
                return found
    elif isinstance(node, list):
        for value in node:
            found = find_value(value, keys)
            if found:
                return found
    return None

recommended_prompt = find_value(
    payload,
    {"recommendedSystemPrompt", "recommended_system_prompt"},
)
if recommended_prompt is None:
    raise RuntimeError(
        "Completed recommendation did not contain a system prompt."
    )

Path("generated/system_prompt.candidate.txt").write_text(
    recommended_prompt.rstrip() + "\n"
)
PY

diff -u \
  generated/system_prompt.baseline.txt \
  generated/system_prompt.candidate.txt \
  || true

agentcore add config-bundle --help
agentcore add runtime-endpoint --help
agentcore run ab-test --help
```

The Insights command includes `GoalSuccessRate`, so `--from-insights` resolves
the corresponding batch evaluation and preserves the failure evidence used by
the recommendation job. Insights clusters are hypotheses and recommendations
are candidates. Inspect the contributing traces and prompt diff before
deploying the candidate as a separate Runtime version and named endpoint. Then
set `AGENTCORE_RUNTIME_ENDPOINT` to that endpoint,
`AGENTCORE_RESULT_LABEL=candidate`, and repeat the Section 3 block without
changing its scenarios or evaluator IDs.

**Compare baseline and candidate evidence before promotion:**

```python
import json
from pathlib import Path

def load_scores(path):
    payload = json.loads(Path(path).read_text())
    scores = {}
    for scenario in payload.get("scenario_results", []):
        scenario_id = scenario["scenario_id"]
        if scenario.get("status") != "COMPLETED":
            raise RuntimeError(
                f"{scenario_id} did not complete: {scenario.get('error')}"
            )
        for evaluator in scenario.get("evaluator_results", []):
            values = [
                result["value"]
                for result in evaluator.get("results", [])
                if isinstance(result.get("value"), (int, float))
            ]
            if not values:
                raise RuntimeError(
                    f"No numeric result for {scenario_id} / "
                    f"{evaluator['evaluator_id']}."
                )
            scores[(scenario_id, evaluator["evaluator_id"])] = (
                sum(values) / len(values)
            )
    if not scores:
        raise RuntimeError(f"No completed scores found in {path}.")
    return scores

baseline = load_scores("generated/city_dataset_baseline.json")
candidate = load_scores("generated/city_dataset_candidate.json")
if baseline.keys() != candidate.keys():
    missing = sorted(set(baseline) - set(candidate))
    added = sorted(set(candidate) - set(baseline))
    raise RuntimeError(
        f"Runs are not comparable. Missing={missing}; added={added}"
    )

regressions = []
for key in sorted(baseline):
    delta = candidate[key] - baseline[key]
    print(
        key[0],
        key[1],
        f"baseline={baseline[key]:.3f}",
        f"candidate={candidate[key]:.3f}",
        f"delta={delta:+.3f}",
    )
    if delta < 0:
        regressions.append((key, delta))

if regressions:
    raise RuntimeError(f"Candidate regressed: {regressions}")
print("Candidate passed the unchanged regression comparison.")
```

Configuration bundles version accepted candidates. Gateway A/B tests
additionally require a deployed Gateway, representative traffic, online
evaluation, effect-size review, and rollback readiness. Treat exact zero-drop
gating as the workshop default; use calibrated repeatability bounds for a
stochastic production judge.

Clean up after the module:

```bash
python src/cleanup.py --yes
```

## Challenges

### Challenge: Diagnose and Improve an Ambiguous Multi-Turn Failure

The user asks, "Tell me about Portland," then clarifies the state and requests a
Seattle comparison. Decide among a prompt, tool-description, parameter-validation,
or session-state change; do not assume the prompt is the cause.

Build a release decision that:

1. Adds the ambiguous conversation to a curated regression dataset
2. Selects the minimum built-in evaluator set needed to diagnose it
3. Names one product-specific gap and adds one focused custom evaluator for it
4. Calibrates that evaluator against human-labeled fixtures
5. Implements a fail-closed pre-release quality gate
6. Validates the candidate change against the unchanged baseline dataset
7. Defines a sampled online or controlled A/B rollout with a rollback trigger

**Assessment criteria:**

- Dataset has at least 6 scenarios: 1 multi-turn, 1 no-tool, and 1 missing-data
- Ground truth uses expected responses, assertions, or expected trajectories
  according to what each scenario actually constrains
- Evaluator plan has at least 2 built-ins and no evaluator without a named failure
- Custom evaluator makes one binary decision at the correct `SESSION`, `TRACE`,
  or `TOOL_CALL` level
- Calibration uses all 8 labeled fixtures and reports accuracy, TPR, and TNR
- Quality gate fails when the evaluation command errors, returns no results,
  contains evaluator errors, or crosses the calibrated threshold
- Baseline and candidate use the same dataset, evaluator versions, and score
  breakdown, reporting both improvements and regressions
- Rollout plan states sampling percentage, promotion evidence, and a concrete
  rollback condition

## Wrap-Up

You built one cumulative AgentCore evaluation lifecycle:

| Stage | Evidence |
|---|---|
| Run | Versioned Runtime and deterministic agent |
| Observe | Session, trace, model spans, and tool-call spans |
| Evaluate | Built-ins plus stable reference inputs |
| Regress | Curated single-turn and multi-turn dataset |
| Extend | Focused, calibrated custom evaluator |
| Operate | Batch gates and sampled online monitoring |
| Improve | Simulation, failure analysis, controlled validation |

**Key takeaway:** Define the failure, collect the evidence at the right level,
use the narrowest evaluator that can detect it, and validate the evaluator
before automating a decision.

See [CHALLENGE-deep-dive.md](./CHALLENGE-deep-dive.md) to extend one AgentCore
workflow beyond the guided module.
