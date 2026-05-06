"""Model + agent factory — YOU FILL THIS IN (or replace).

The default skeleton uses Strands' `BedrockModel` with Sonnet 4.5. If
you'd rather use a different framework (LangGraph, raw boto3, etc.),
replace this whole file — the only thing the eval cares about is what
ends up at `--out` and `--trace-out` (see __main__.py and trace.py).

The signature `build_strands_agent(...)` is a suggestion, not a
requirement.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .trace import Trace


# The Bedrock model the rest of the workshop uses. You can swap to a
# faster/cheaper model (e.g. Haiku) for iteration, but report the model
# you actually scored against.
DEFAULT_MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"


def build_strands_agent(repo_path: Path, trace: Trace, model_id: str = DEFAULT_MODEL_ID) -> Any:
    """Return a Strands `Agent` configured with Bedrock + your tools.

    YOU IMPLEMENT THIS. Suggested skeleton:

        from strands import Agent
        from strands.models.bedrock import BedrockModel
        from .tools import build_toolset

        model = BedrockModel(model_id=model_id)
        tools = build_toolset(repo_path=repo_path, trace=trace)
        return Agent(model=model, tools=tools)

    Notes:
      - Strands handles the LLM <-> tool loop for you. Each tool you
        register MUST call `trace.tool(name, input)` so the eval can
        score tool-call quality.
      - To capture token usage, wrap the agent's invocation and pull
        usage from the model's response metadata, then call
        `trace.usage(input_tokens, output_tokens)` once at the end of
        agent.run(). See `agent.py` for where to call it.
      - For workshop reproducibility, set `temperature=0` on the model
        if your framework exposes it.
    """
    raise NotImplementedError(
        "build_strands_agent is a stub. Open my_agent/model.py and follow "
        "the TODO. Reference: https://docs.aws.amazon.com/bedrock/ for "
        "model IDs and Strands docs for the Agent API."
    )
