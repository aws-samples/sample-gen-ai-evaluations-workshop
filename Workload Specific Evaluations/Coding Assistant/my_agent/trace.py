"""Trace recorder.

The eval pipeline expects a JSON list at `--trace-out`, where each entry
is `{"tool": <str>, "input": <dict>}`. The runners parse this list to
score tool-call quality and (for the custom agent only) extract token
usage from a synthetic `_usage` entry.

Use `trace.tool(name, input)` for every tool call. Use `trace.usage(in_tok, out_tok)`
once at the end of `agent.run()` to record the cumulative token counts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Trace:
    entries: List[Dict[str, Any]] = field(default_factory=list)

    def tool(self, name: str, input: Dict[str, Any] | None = None) -> None:
        """Record a tool invocation. Call this from every tool implementation."""
        self.entries.append({"tool": name, "input": input or {}})

    def usage(self, input_tokens: int, output_tokens: int) -> None:
        """Record cumulative Bedrock token usage as the synthetic `_usage` entry.

        The eval looks for an entry of shape
        `{"tool": "_usage", "input": {"input_tokens": ..., "output_tokens": ...}}`.
        Without this, the custom agent's tokens_per_task column will be NaN.
        """
        self.entries.append({
            "tool": "_usage",
            "input": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        })

    def to_list(self) -> List[Dict[str, Any]]:
        return list(self.entries)
