"""
MetricsCollector — instruments memory/handoffs/time/tokens in  Multi-agent shared-context evaluation.

Structure Terminology:
  - TurnRecord: one user message → all agent work → final output
  - AgentRecord: one agent call within a turn (handoff, memory read, response)

Metric categories:
  1. Memory Context Metrics (LLM-as-judge)
  2. Memory Latency Metrics (timers + token counts)


What MetricsCollector does
------------
1. Records the context flow — for every agent call it captures what the
   agent was asked (handoff), what it read from memory, and what it produced.
   This gives you a full trace of how information moved through the system.

2. Scores context quality with an LLM judge — an LLM evaluates each
   agent call on six dimensions: Was the context fresh? Was the handoff
   complete? Did the agent use what it read? Is what it wrote accurate?
   How much context was wasted? Do agents agree on key facts?

3. Measures coordination overhead — timers and token counts show how
   much of each agent’s work is spent on memory I/O vs actual reasoning,
   so you can spot when shared memory becomes a bottleneck.

Memory-backend agnostic — works the same whether agents share a Python list,
AgentCore, or any other store. The notebooks wire up the memory layer; this
module just evaluates what flowed through it.
"""

import json
import math
import time
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

import boto3

from model_config import JUDGE_MODEL_ID

logger = logging.getLogger("metrics_collector")


# ---------------------------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------------------------

class LLMJudge:
    """Bedrock Claude Opus for semantic evaluation."""

    def __init__(self, region: str = "us-west-2", model_id: str = JUDGE_MODEL_ID):
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id

    def _call(self, prompt: str, max_retries: int = 5) -> dict:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": prompt}],
        })
        for attempt in range(max_retries):
            try:
                resp = self.client.invoke_model(
                    modelId=self.model_id, body=body,
                    contentType="application/json", accept="application/json")
                text = json.loads(resp["body"].read())["content"][0]["text"]
                if "```" in text:
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                return json.loads(text.strip())
            except self.client.exceptions.ThrottlingException as e:
                wait = 2 ** attempt + 1
                logger.warning(f"Throttled (attempt {attempt+1}/{max_retries}), waiting {wait}s...")
                time.sleep(wait)
            except Exception as e:
                if "ServiceUnavailable" in str(type(e).__name__) or "Too many connections" in str(e):
                    wait = 2 ** attempt + 1
                    logger.warning(f"Rate limited (attempt {attempt+1}/{max_retries}), waiting {wait}s...")
                    time.sleep(wait)
                else:
                    logger.error(f"LLM judge error: {e}")
                    import traceback
                    traceback.print_exc()
                    return {}
        logger.error(f"LLM judge failed after {max_retries} retries")
        return {}

    # -- Individual judge prompts ------------------------------------------

    def judge_context_freshness(self, latest_user_msg: str, retrieved_context: str,
                                agent_name: str) -> dict:
        """Is the agent working with the most current information?"""
        if not retrieved_context.strip():
            return {"score": 5, "reasoning": "No prior context (first call)"}
        prompt = f"""You are evaluating whether an agent in a multi-agent system is working with
current information. The user's latest message may have changed key facts (budget,
dates, preferences). Check if the retrieved context reflects those changes.

Latest user message:
\"\"\"{latest_user_msg}\"\"\"

Context the {agent_name} agent retrieved from shared memory:
\"\"\"{retrieved_context[:3000]}\"\"\"

Rate context freshness:
- Score 5: Context fully reflects the latest user requirements
- Score 4: Context is mostly current with minor gaps
- Score 3: Context is partially outdated — some fields are stale
- Score 2: Context is mostly outdated
- Score 1: Context is completely stale / contradicts latest requirements

Respond with JSON only: {{"score": <1-5>, "reasoning": "<one sentence>", "stale_fields": ["list any outdated fields, empty if none"]}}"""
        return self._call(prompt)

    def judge_handoff_completeness(self, original: str, handoff: str,
                                   agent_name: str) -> dict:
        """Did the handoff include all required context?"""
        prompt = f"""You are evaluating a multi-agent system. A coordinator forwarded a query to
the {agent_name} agent.

User's original request:
\"\"\"{original}\"\"\"

Query forwarded to {agent_name}:
\"\"\"{handoff}\"\"\"

Rate how complete the handoff is — does it include all facts the {agent_name} agent
needs (dates, locations, budget, preferences, constraints)?
- Score 5: All relevant facts included
- Score 4: Most facts included, minor omissions
- Score 3: Some important facts missing
- Score 2: Many facts missing
- Score 1: Most relevant facts missing

Respond with JSON only: {{"score": <1-5>, "reasoning": "<one sentence>", "missing_fields": ["list missing facts, empty if none"]}}"""
        return self._call(prompt)

    def judge_context_utilization(self, retrieved_context: str, response: str,
                                  agent_name: str) -> dict:
        """Did the agent use the context it read from memory?"""
        if not retrieved_context.strip():
            return {"score": 5, "reasoning": "No context to utilize (first call)"}
        prompt = f"""You are evaluating whether the {agent_name} agent incorporated retrieved
context into its response.

Retrieved context:
\"\"\"{retrieved_context[:3000]}\"\"\"

Agent's response:
\"\"\"{response[:3000]}\"\"\"

Rate how well the agent used the retrieved context:
- Score 5: Fully utilized all relevant context
- Score 4: Used most context, minor omissions
- Score 3: Used some context but ignored important parts
- Score 2: Mostly ignored the context
- Score 1: Completely ignored the context

Respond with JSON only: {{"score": <1-5>, "reasoning": "<one sentence>"}}"""
        return self._call(prompt)

    def judge_state_consistency(self, responses: Dict[str, str]) -> dict:
        """Do agents agree on key facts?"""
        agent_texts = "\n\n".join(
            f"[{name}]:\n{resp[:2000]}" for name, resp in responses.items())
        prompt = f"""You are evaluating factual consistency across multiple agent responses produced
as part of the same task. Check whether agents agree on key facts — numbers, dates,
names, constraints, and their interpretation.

Do NOT assume there are contradictions. Only flag genuine disagreements where agents
state different values or interpretations for the same fact.

Agent responses:
{agent_texts}

Rate factual consistency:
- Score 5: All agents agree on all key facts and their interpretation
- Score 4: Minor phrasing differences but no factual disagreements
- Score 3: Some disagreements on secondary details
- Score 2: Disagreements on important facts
- Score 1: Major contradictions on critical facts

Respond with JSON only: {{"score": <1-5>, "reasoning": "<one sentence>", "contradictions": ["list ONLY genuine contradictions, empty if none"]}}"""
        return self._call(prompt)

    def judge_memory_write_accuracy(self, agent_input: str, response: str,
                                    agent_name: str) -> dict:
        """Is what the agent wrote to memory factually correct?"""
        prompt = f"""You are evaluating whether the {agent_name} agent's response (which gets
written to shared memory for other agents to read) is factually accurate given
its input.

Input the agent received:
\"\"\"{agent_input[:2000]}\"\"\"

Agent's response (written to memory):
\"\"\"{response[:3000]}\"\"\"

Rate the factual accuracy of the response:
- Score 5: All facts are accurate and consistent with input
- Score 4: Mostly accurate, minor imprecisions
- Score 3: Some inaccurate or fabricated details
- Score 2: Significant inaccuracies
- Score 1: Mostly fabricated or wrong

Respond with JSON only: {{"score": <1-5>, "reasoning": "<one sentence>"}}"""
        return self._call(prompt)

    def judge_redundant_context(self, retrieved_context: str, agent_name: str) -> dict:
        """How much of the retrieved context is redundant or irrelevant?"""
        if not retrieved_context.strip():
            return {"score": 5, "reasoning": "No context retrieved", "redundancy_pct": 0}
        prompt = f"""You are evaluating the efficiency of context passed to the {agent_name} agent.

Retrieved context:
\"\"\"{retrieved_context[:3000]}\"\"\"

Estimate what percentage of this context is redundant (repeated information) or
irrelevant (not useful for this agent's task).

Rate context efficiency:
- Score 5: All context is relevant and non-redundant (0-10% waste)
- Score 4: Mostly efficient (10-25% waste)
- Score 3: Moderate waste (25-50% redundant/irrelevant)
- Score 2: Mostly wasteful (50-75% redundant/irrelevant)
- Score 1: Almost entirely redundant or irrelevant (75%+ waste)

Respond with JSON only: {{"score": <1-5>, "reasoning": "<one sentence>", "redundancy_pct": <estimated percentage>}}"""
        return self._call(prompt)


# ---------------------------------------------------------------------------
# AgentRecord — one agent call within a turn
# ---------------------------------------------------------------------------

@dataclass
class AgentRecord:
    """One agent invocation within a turn."""
    agent_name: str
    handoff_query: str = ""
    retrieved_context: str = ""
    response: str = ""

    # Latency
    memory_read_latency: float = 0.0
    memory_write_latency: float = 0.0
    total_agent_latency: float = 0.0

    # Tokens
    coordination_tokens: int = 0
    reasoning_input_tokens: int = 0
    reasoning_output_tokens: int = 0

    # LLM judge scores (filled by evaluate)
    judge_scores: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# TurnRecord — one user message → all agent work → final output
# ---------------------------------------------------------------------------

@dataclass
class TurnRecord:
    """One turn: user input → N agent calls → final output."""
    turn_number: int
    original_query: str
    agent_calls: List[AgentRecord] = field(default_factory=list)

    # Cross-agent scores for this turn
    state_consistency: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# MetricsCollector
# ---------------------------------------------------------------------------

class MetricsCollector:
    """Accumulates turns and computes multi-agent metrics.
    1. Records the context flow — for every agent call it captures what the
   agent was asked (handoff), what it read from memory, and what it produced.
   This gives you a full trace of how information moved through the system.

   2.Measures coordination overhead — timers and token counts show how
   much of each agent’s work is spent on memory I/O vs actual reasoning,
   so you can spot when shared memory becomes a bottleneck
"""

    def __init__(self, region: str = "us-west-2"):
        self.turns: List[TurnRecord] = []
        self._current_turn: Optional[TurnRecord] = None
        self._current_agents: Dict[str, AgentRecord] = {}
        self.judge = LLMJudge(region=region)

    # -- Turn lifecycle ------------------------------------------------------

    def begin_turn(self, turn_number: int, original_query: str):
        """Call when a new user message is sent to the system."""
        self._current_turn = TurnRecord(turn_number=turn_number,
                                        original_query=original_query)

    def end_turn(self):
        """Call when the system has finished responding to the user."""
        if self._current_turn:
            self.turns.append(self._current_turn)
            self._current_turn = None
            self._current_agents = {}

    # -- Agent call lifecycle ------------------------------------------------

    def record_handoff(self, agent_name: str, handoff_query: str):
        """Called when the hub/orchestrator dispatches to an agent."""
        rec = AgentRecord(agent_name=agent_name, handoff_query=handoff_query)
        self._current_agents[agent_name] = rec

    def record_retrieved_context(self, agent_name: str, context: str):
        """Called in on_agent_initialized after memory read."""
        if agent_name in self._current_agents:
            self._current_agents[agent_name].retrieved_context = context

    def record_response(self, agent_name: str, response: str):
        """Called in on_message_added when agent writes its response."""
        if agent_name in self._current_agents:
            rec = self._current_agents[agent_name]
            rec.response = response
            if self._current_turn:
                self._current_turn.agent_calls.append(rec)
            del self._current_agents[agent_name]

    def record_memory_read_latency(self, agent_name: str, latency: float):
        if agent_name in self._current_agents:
            self._current_agents[agent_name].memory_read_latency = latency

    def record_memory_write_latency(self, agent_name: str, latency: float):
        if agent_name in self._current_agents:
            self._current_agents[agent_name].memory_write_latency += latency
        elif self._current_turn:
            # Agent already moved to turn — find it there
            for rec in reversed(self._current_turn.agent_calls):
                if rec.agent_name == agent_name:
                    rec.memory_write_latency += latency
                    break

    def record_agent_latency(self, agent_name: str, latency: float):
        if self._current_turn:
            for rec in reversed(self._current_turn.agent_calls):
                if rec.agent_name == agent_name:
                    rec.total_agent_latency = latency
                    break

    def record_token_usage(self, agent_name: str, input_tokens: int, output_tokens: int):
        if self._current_turn:
            for rec in reversed(self._current_turn.agent_calls):
                if rec.agent_name == agent_name:
                    rec.reasoning_input_tokens = input_tokens
                    rec.reasoning_output_tokens = output_tokens
                    ctx = rec.retrieved_context
                    rec.coordination_tokens = int(len(ctx.split()) * 1.3) if ctx else 0
                    break

    # -- Evaluate all turns with LLM judge -----------------------------------

    def evaluate_all(self):
        """Run LLM-as-judge on every agent call in every turn."""
        logger.info("Running LLM-as-judge evaluations...")
        for turn in self.turns:
            for rec in turn.agent_calls:
                # Context Freshness
                rec.judge_scores["context_freshness"] = self.judge.judge_context_freshness(
                    turn.original_query, rec.retrieved_context, rec.agent_name)
                time.sleep(0.5)  # pace requests to avoid throttling

                # Handoff Completeness
                rec.judge_scores["handoff_completeness"] = self.judge.judge_handoff_completeness(
                    turn.original_query, rec.handoff_query, rec.agent_name)
                time.sleep(0.5)

                # Context Utilization
                rec.judge_scores["context_utilization"] = self.judge.judge_context_utilization(
                    rec.retrieved_context, rec.response, rec.agent_name)
                time.sleep(0.5)

                # Memory Write Accuracy
                rec.judge_scores["write_accuracy"] = self.judge.judge_memory_write_accuracy(
                    rec.handoff_query, rec.response, rec.agent_name)
                time.sleep(0.5)

                # Redundant Context
                rec.judge_scores["redundant_context"] = self.judge.judge_redundant_context(
                    rec.retrieved_context, rec.agent_name)
                time.sleep(0.5)

            # State Consistency (cross-agent per turn)
            if len(turn.agent_calls) >= 2:
                responses = {r.agent_name: r.response for r in turn.agent_calls}
                turn.state_consistency = self.judge.judge_state_consistency(responses)
            else:
                turn.state_consistency = {"score": 5, "reasoning": "Single agent, no cross-check needed",
                                          "contradictions": []}

        logger.info("LLM-as-judge evaluations complete.")

    # -- Context Compression Ratio (pure math) -------------------------------

    @staticmethod
    def ccr(original: str, handoff: str) -> float:
        if not original:
            return 0.0
        return len(handoff) / len(original)

    # -- Report: Context Flow Trace ------------------------------------------

    def trace_report(self) -> str:
        """Human-readable trace of context flow through the system."""
        lines = ["### Context Flow Trace", ""]
        mx = 300

        for turn in self.turns:
            lines.append(f"{'═'*20}")
            lines.append(f" TURN {turn.turn_number}")
            lines.append(f"{'═'*20}")
            lines.append(f"\n **User message:** {turn.original_query[:mx]}")
            lines.append("")

            for rec in turn.agent_calls:
                lines.append(f"---")
                lines.append(f"**{rec.agent_name}**")
                lines.append(f"")
                lines.append(f"HANDOFF: {rec.handoff_query[:mx]}{'...' if len(rec.handoff_query) > mx else ''}")
                lines.append(f"")
                if rec.retrieved_context:
                    ctx_preview = rec.retrieved_context[:mx]
                    lines.append(f"READ FROM MEMORY: {ctx_preview}{'...' if len(rec.retrieved_context) > mx else ''}")
                else:
                    lines.append(f"READ FROM MEMORY: _(empty)_")
                lines.append(f"")
                lines.append(f"RESPONSE: {rec.response[:mx]}{'...' if len(rec.response) > mx else ''}")
                lines.append(f"")
                lines.append(f"WRITTEN TO MEMORY: handoff + response above")
                lines.append("")

        return "\n".join(lines)

    # -- Report: Memory Context Metrics --------------------------------------

    def context_metrics_report(self) -> str:
        lines = ["### Memory Context Metrics", ""]

        header = ("| Turn | Agent | CCR | Freshness | Handoff Complete | "
                  "Context Util. | Write Accuracy | Redundancy |")
        sep = ("|------|-------|-----|-----------|------------------|"
               "---------------|----------------|------------|")
        lines.extend([header, sep])

        for turn in self.turns:
            for rec in turn.agent_calls:
                cf = rec.judge_scores.get("context_freshness", {})
                hc = rec.judge_scores.get("handoff_completeness", {})
                cu = rec.judge_scores.get("context_utilization", {})
                wa = rec.judge_scores.get("write_accuracy", {})
                rc = rec.judge_scores.get("redundant_context", {})
                ccr_val = self.ccr(turn.original_query, rec.handoff_query)

                lines.append(
                    f"| {turn.turn_number} | {rec.agent_name} "
                    f"| {ccr_val:.2f} "
                    f"| {cf.get('score', '-')}/5 "
                    f"| {hc.get('score', '-')}/5 "
                    f"| {cu.get('score', '-')}/5 "
                    f"| {wa.get('score', '-')}/5 "
                    f"| {rc.get('score', '-')}/5 |"
                )

        # State consistency per turn
        lines.append("")
        lines.append("**State Consistency (per turn):**")
        for turn in self.turns:
            sc = turn.state_consistency
            lines.append(f"- Turn {turn.turn_number}: {sc.get('score', '-')}/5 — {sc.get('reasoning', '')}")
            for c in sc.get("contradictions", []):
                lines.append(f"  - ⚠️ {c}")

        # Detailed reasoning
        lines.append("")
        lines.append("<details><summary>LLM Judge Reasoning</summary>")
        lines.append("")
        for turn in self.turns:
            for rec in turn.agent_calls:
                lines.append(f"**Turn {turn.turn_number} — {rec.agent_name}:**")
                for key in ["context_freshness", "handoff_completeness", "context_utilization",
                            "write_accuracy", "redundant_context"]:
                    s = rec.judge_scores.get(key, {})
                    if s.get("reasoning"):
                        lines.append(f"- {key}: {s['reasoning']}")
                    if s.get("stale_fields"):
                        lines.append(f"  - Stale fields: {s['stale_fields']}")
                    if s.get("missing_fields"):
                        lines.append(f"  - Missing: {s['missing_fields']}")
                lines.append("")
        lines.append("</details>")

        return "\n".join(lines)

    # -- Report: Memory Latency Metrics --------------------------------------

    def latency_metrics_report(self) -> str:
        lines = ["### Memory Latency Metrics", ""]
        header = "| Turn | Agent | Read (s) | Write (s) | Reasoning (s) | Coord Lat % | Coord Tok % |"
        sep    = "|------|-------|----------|-----------|---------------|-------------|-------------|"
        lines.extend([header, sep])

        total_coord = 0.0
        total_lat = 0.0
        total_ctok = 0
        total_tok = 0

        for turn in self.turns:
            for rec in turn.agent_calls:
                cl = rec.memory_read_latency + rec.memory_write_latency
                rl = max(rec.total_agent_latency - cl, 0.0)
                ta = rec.total_agent_latency or 1.0
                rt = rec.reasoning_input_tokens + rec.reasoning_output_tokens
                at = rec.coordination_tokens + rt

                lines.append(
                    f"| {turn.turn_number} | {rec.agent_name} "
                    f"| {rec.memory_read_latency:.2f} | {rec.memory_write_latency:.2f} "
                    f"| {rl:.2f} | {cl/ta:.0%} | {rec.coordination_tokens/at:.0%} |"
                    if at > 0 else
                    f"| {turn.turn_number} | {rec.agent_name} "
                    f"| {rec.memory_read_latency:.2f} | {rec.memory_write_latency:.2f} "
                    f"| {rl:.2f} | {cl/ta:.0%} | 0% |"
                )
                total_coord += cl
                total_lat += ta
                total_ctok += rec.coordination_tokens
                total_tok += at

        lines.append("")
        lines.append(f"**Avg Coordination Latency %:** {total_coord/total_lat:.0%}" if total_lat else "")
        lines.append(f"**Avg Coordination Token %:** {total_ctok/total_tok:.0%}" if total_tok else "")
        return "\n".join(lines)

    # -- Comparison report ---------------------------------------------------

    @staticmethod
    def comparison_report(a: "MetricsCollector", b: "MetricsCollector",
                          label_a: str = "Session 1", label_b: str = "Session 2") -> str:
        """Side-by-side comparison of two sessions."""
        lines = [f"### {label_a} vs {label_b}", ""]

        header = f"| Metric | {label_a} | {label_b} | Delta |"
        sep    = "|--------|----------|----------|-------|"
        lines.extend([header, sep])

        def avg_score(collector, key):
            scores = []
            for t in collector.turns:
                for r in t.agent_calls:
                    s = r.judge_scores.get(key, {}).get("score")
                    if s is not None:
                        scores.append(s)
            return sum(scores) / len(scores) if scores else 0

        for key, label in [
            ("context_freshness", "Avg Context Freshness"),
            ("handoff_completeness", "Avg Handoff Completeness"),
            ("context_utilization", "Avg Context Utilization"),
            ("write_accuracy", "Avg Write Accuracy"),
            ("redundant_context", "Avg Context Efficiency"),
        ]:
            sa = avg_score(a, key)
            sb = avg_score(b, key)
            d = sb - sa
            sign = "+" if d >= 0 else ""
            lines.append(f"| {label} | {sa:.1f}/5 | {sb:.1f}/5 | {sign}{d:.1f} |")

        # State consistency averages
        def avg_consistency(collector):
            scores = [t.state_consistency.get("score", 0) for t in collector.turns
                      if t.state_consistency.get("score")]
            return sum(scores) / len(scores) if scores else 0

        ca = avg_consistency(a)
        cb = avg_consistency(b)
        lines.append(f"| Avg State Consistency | {ca:.1f}/5 | {cb:.1f}/5 | {'+' if cb-ca>=0 else ''}{cb-ca:.1f} |")

        return "\n".join(lines)
