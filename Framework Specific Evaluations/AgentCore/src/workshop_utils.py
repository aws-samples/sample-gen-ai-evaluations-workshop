import json
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import boto3
from botocore.config import Config
from bedrock_agentcore.evaluation import AgentInvokerInput, AgentInvokerOutput


MODULE_ROOT = Path(__file__).resolve().parents[1]
AGENTCORE_DIR = MODULE_ROOT / "agentcore"
DEPLOYED_STATE_PATH = AGENTCORE_DIR / ".cli" / "deployed-state.json"
AWS_TARGETS_PATH = AGENTCORE_DIR / "aws-targets.json"
GENERATED_DIR = MODULE_ROOT / "generated"
RUNTIME_NAME = "CityAnalyst"


@dataclass(frozen=True)
class RuntimeInfo:
    runtime_name: str
    runtime_id: str
    runtime_arn: str
    region: str
    target_name: str

    @property
    def log_group_name(self) -> str:
        return f"/aws/bedrock-agentcore/runtimes/{self.runtime_id}-DEFAULT"


def run_cli_json(*args: str) -> dict[str, Any]:
    command = ["agentcore", *args, "--json"]
    completed = subprocess.run(
        command,
        cwd=MODULE_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def load_runtime_info(
    runtime_name: str = RUNTIME_NAME,
    target_name: str | None = None,
) -> RuntimeInfo:
    state = json.loads(DEPLOYED_STATE_PATH.read_text(encoding="utf-8"))
    targets = state.get("targets", {})
    if not targets:
        raise RuntimeError("No deployed target found. Complete notebook 01 first.")

    selected_target = target_name or next(iter(targets))
    target_state = targets.get(selected_target, {})
    runtimes = target_state.get("resources", {}).get("runtimes", {})
    runtime = runtimes.get(runtime_name)
    if runtime is None:
        raise RuntimeError(
            f"Runtime {runtime_name!r} is not deployed in target {selected_target!r}."
        )

    target_specs = json.loads(AWS_TARGETS_PATH.read_text(encoding="utf-8"))
    region = next(
        (
            item["region"]
            for item in target_specs
            if item.get("name") == selected_target
        ),
        boto3.Session().region_name,
    )
    if not region:
        raise RuntimeError("Could not determine an AWS region.")

    return RuntimeInfo(
        runtime_name=runtime_name,
        runtime_id=runtime["runtimeId"],
        runtime_arn=runtime["runtimeArn"],
        region=region,
        target_name=selected_target,
    )


def make_session_id(prefix: str = "agentcore-evals") -> str:
    session_id = f"{prefix}-{uuid.uuid4()}"
    if len(session_id) < 33:
        session_id = f"{session_id}-{uuid.uuid4()}"
    return session_id


class RuntimeInvoker:
    """Reusable AgentCore Runtime client compatible with dataset runners."""

    def __init__(self, runtime_info: RuntimeInfo):
        self.runtime_info = runtime_info
        client_config = Config(
            retries={"total_max_attempts": 5, "mode": "adaptive"},
            connect_timeout=5,
            read_timeout=180,
        )
        self.client = boto3.client(
            "bedrock-agentcore",
            region_name=runtime_info.region,
            config=client_config,
        )

    def invoke(self, prompt: str, session_id: str) -> Any:
        response = self.client.invoke_agent_runtime(
            agentRuntimeArn=self.runtime_info.runtime_arn,
            qualifier="DEFAULT",
            runtimeSessionId=session_id,
            payload=json.dumps({"prompt": prompt}).encode("utf-8"),
        )
        body = response["response"].read()
        return json.loads(body)

    def __call__(self, invoker_input: AgentInvokerInput) -> AgentInvokerOutput:
        payload = invoker_input.payload
        if hasattr(payload, "model_dump"):
            payload = payload.model_dump(mode="json")
        prompt = payload if isinstance(payload, str) else json.dumps(payload)
        output = self.invoke(prompt=prompt, session_id=invoker_input.session_id or make_session_id())
        return AgentInvokerOutput(agent_output=output)


def wait_for_session_trace(
    session_id: str,
    runtime_name: str = RUNTIME_NAME,
    timeout_seconds: int = 120,
    poll_seconds: int = 10,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        payload = run_cli_json(
            "traces",
            "list",
            "--runtime",
            runtime_name,
            "--since",
            "30m",
            "--limit",
            "50",
        )
        for trace in payload.get("traces", []):
            if trace.get("sessionId") == session_id:
                return trace
        time.sleep(poll_seconds)
    raise TimeoutError(f"No trace appeared for session {session_id} within {timeout_seconds}s")


def find_span_records(value: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            if "spanId" in node and ("traceId" in node or "name" in node):
                records.append(node)
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return records


def summarize_spans(spans: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for span in spans:
        attributes = span.get("attributes", {})
        rows.append(
            {
                "name": span.get("name", ""),
                "trace_id": span.get("traceId", ""),
                "span_id": span.get("spanId", ""),
                "parent_span_id": span.get("parentSpanId", ""),
                "operation": attributes.get("gen_ai.operation.name", ""),
                "tool": attributes.get("gen_ai.tool.name", ""),
            }
        )
    return rows

