---
name: Multiagent Shared Context Evaluation
description: Help me evaluate shared memory and context propagation in multi-agent systems, instrument coordination metrics, detect stale context and state inconsistencies, and compare hub-spoke vs peer-to-peer architectures
---

# Evaluating Shared Context in Multi-Agent Systems

Multi-agent failures happen *before* the final answer — a sub-agent receives stale context, the coordinator updates a constraint but doesn't re-dispatch, or two agents work from different versions of the same plan. These are memory and coordination failures, and they're invisible unless you instrument for them. This module builds instrumented multi-agent systems and evaluates how context flows through them.

## Prerequisites

- AWS account with Bedrock access (us-west-2)
- Python 3.10+
- Familiarity with the Strands agent framework (`strands` SDK)
- Completed SKILL-operational (understands token/latency metrics) and SKILL-agentic (understands tool-calling patterns)

## Learning Objectives

By the end of this module, you will be able to:

1. Build a hub-and-spoke multi-agent system with instrumented shared memory
2. Build a peer-to-peer sequential swarm where agents coordinate without a central hub
3. Implement dynamic handoff routing where agents decide execution order via tool calls
4. Evaluate context quality using 6 LLM-as-judge metrics (freshness, completeness, utilization, consistency, write accuracy, redundancy)
5. Detect coordination failures by comparing baseline sessions against sessions with mid-conversation constraint changes

## Setup

```bash
pip install strands-agents strands-agents-builder boto3 matplotlib numpy
```

```python
import os
import time
import json
import logging

import boto3
from strands import Agent, tool
from strands.hooks import AgentInitializedEvent, HookProvider, HookRegistry, MessageAddedEvent

region = os.getenv("AWS_REGION", "us-west-2")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("multiagent-eval")
```

Model and prompt configuration (shared across all sections):

```python
# Model IDs
AGENT_MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"
JUDGE_MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"

# Hub-and-Spoke prompts (travel planning)
FLIGHT_PROMPT = (
    "You are a flight booking assistant. Help find flights, make reservations, "
    "and answer questions about airlines, routes, and pricing. Be specific with "
    "prices and schedules."
)
HOTEL_PROMPT = (
    "You are a hotel booking assistant. Help find hotels, make reservations, "
    "and answer questions about accommodations, amenities, and pricing. Be specific with prices."
)
ITINERARY_PROMPT = (
    "You are an itinerary planner. Read the flight and hotel information from "
    "conversation history and create a cohesive day-by-day travel itinerary. "
    "Reference specific flight times, hotel names, and prices from the prior agents' outputs."
)
HUB_PROMPT = """You are a travel planning coordinator. You delegate to specialized agents:
- flight_booking_assistant: for flight queries
- hotel_booking_assistant: for hotel queries
- itinerary_assistant: for building the final itinerary (call LAST, after flight + hotel)

For complete trip requests, call flight first, then hotel, then itinerary.
Keep messages short. Ask max 2 questions per turn."""

# Peer-to-Peer prompts (market research)
MARKET_TRENDS_PROMPT = (
    "You are a Market Trends Analyst specialising in the restaurant and food service industry. "
    "Analyse market trends, competitive landscape, market size, growth rates, and emerging "
    "opportunities. Be specific with numbers, brand names, and market segments. "
    "Keep your analysis concise (2-3 paragraphs)."
)
CUSTOMER_INSIGHTS_PROMPT = (
    "You are a Customer Insights Analyst specialising in the restaurant and food service industry. "
    "Analyse customer segments, dining behaviour, pain points, and demand patterns. "
    "Build on any prior market analysis available in context. Keep concise (2-3 paragraphs)."
)
STRATEGY_SYNTH_PROMPT = (
    "You are a Strategy Synthesizer specialising in the restaurant and food service industry. "
    "Read the market trends and customer insights from your colleagues, then produce a unified "
    "strategic recommendation. Reference specific findings from both prior analyses. "
    "Keep concise (2-3 paragraphs)."
)
```

Shared memory formatting helper:

```python
def format_memory(memory: list) -> str:
    """Format shared memory list as readable string for agent prompts."""
    if not memory:
        return ""
    parts = []
    for entry in memory:
        agent = entry.get("agent", "unknown")
        content = entry.get("content", "")
        role = entry.get("role", "")
        prefix = f"[{agent}]"
        if role:
            prefix += f" {role.title()}:"
        parts.append(f"{prefix} {content}")
    return "\n".join(parts)


def print_memory(memory: list, preview_chars: int = 200):
    """Print shared memory contents for inspection."""
    print(f"Memory has {len(memory)} entries:\n")
    for i, entry in enumerate(memory):
        print(f"  [{i}] agent={entry.get('agent', '?')}, role={entry.get('role', '-')}")
        print(f"      {entry.get('content', '')[:preview_chars]}...\n")
```

## Section 1: Hub-and-Spoke Architecture with Shared Memory

**Concept:** In a hub-and-spoke system, a coordinator agent receives user requests and delegates to specialized spoke agents via tool calls. Each spoke reads shared memory (prior agents' outputs), processes its query, and writes its response back. The coordinator controls execution order — flight first, hotel second, itinerary last. This makes context flow predictable but creates a single point of failure: if the coordinator compresses the user's message poorly, every spoke inherits that information loss.

The key instrumentation points are: (1) what the coordinator passes to each spoke (handoff query), (2) what each spoke reads from memory, (3) what each spoke writes back, and (4) timing for each operation.

**Build:** The `MetricsCollector` records every handoff, memory read, response, and latency. The `ListMemoryHook` attaches to each spoke agent and handles memory I/O:

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


@dataclass
class AgentRecord:
    """One agent invocation within a turn."""
    agent_name: str
    handoff_query: str = ""
    retrieved_context: str = ""
    response: str = ""
    memory_read_latency: float = 0.0
    memory_write_latency: float = 0.0
    total_agent_latency: float = 0.0
    coordination_tokens: int = 0
    reasoning_input_tokens: int = 0
    reasoning_output_tokens: int = 0
    judge_scores: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnRecord:
    """One turn: user input → N agent calls → final output."""
    turn_number: int
    original_query: str
    agent_calls: List[AgentRecord] = field(default_factory=list)
    state_consistency: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Accumulates turns and computes multi-agent context metrics."""

    def __init__(self, region: str = "us-west-2"):
        self.turns: List[TurnRecord] = []
        self._current_turn: Optional[TurnRecord] = None
        self._current_agents: Dict[str, AgentRecord] = {}
        self.region = region

    def begin_turn(self, turn_number: int, original_query: str):
        self._current_turn = TurnRecord(turn_number=turn_number, original_query=original_query)

    def end_turn(self):
        if self._current_turn:
            self.turns.append(self._current_turn)
            self._current_turn = None
            self._current_agents = {}

    def record_handoff(self, agent_name: str, handoff_query: str):
        rec = AgentRecord(agent_name=agent_name, handoff_query=handoff_query)
        self._current_agents[agent_name] = rec

    def record_retrieved_context(self, agent_name: str, context: str):
        if agent_name in self._current_agents:
            self._current_agents[agent_name].retrieved_context = context

    def record_response(self, agent_name: str, response: str):
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
```

Now the memory hook that attaches to each spoke:

```python
class ListMemoryHook(HookProvider):
    """Reads/writes a shared Python list for each spoke agent."""

    def __init__(self, memory: list, collector: MetricsCollector, agent_name: str):
        self.memory = memory
        self.collector = collector
        self.agent_name = agent_name

    def on_agent_initialized(self, event: AgentInitializedEvent):
        t0 = time.perf_counter()
        context = format_memory(self.memory)
        self.collector.record_memory_read_latency(self.agent_name, time.perf_counter() - t0)
        if context:
            event.agent.system_prompt += (
                f"\n\nShared memory from other agents:\n{context}"
                "\n\nUse this context. Reference specific details from other agents.")
            logger.info(f"[{self.agent_name}] Read {len(self.memory)} entries from memory")
        self.collector.record_retrieved_context(self.agent_name, context)

    def on_message_added(self, event: MessageAddedEvent):
        last = event.agent.messages[-1]
        if last["role"] != "assistant":
            return
        text_parts = [b["text"] for b in last.get("content", [])
                      if isinstance(b, dict) and b.get("text")]
        if not text_parts:
            return
        response_text = "\n".join(text_parts)
        self.collector.record_response(self.agent_name, response_text)
        t0 = time.perf_counter()
        self.memory.append({
            "agent": self.agent_name, "role": "assistant",
            "content": response_text, "ts": time.time()})
        self.collector.record_memory_write_latency(self.agent_name, time.perf_counter() - t0)

    def register_hooks(self, registry: HookRegistry):
        registry.add_callback(AgentInitializedEvent, self.on_agent_initialized)
        registry.add_callback(MessageAddedEvent, self.on_message_added)
```

Wire it together into a session runner:

```python
def run_hub_spoke_session(conversation: list, session_label: str) -> tuple:
    """Run a hub-spoke session. Returns (collector, shared_memory)."""
    shared_memory = []
    collector = MetricsCollector(region=region)
    turn_counter = 0

    @tool
    def flight_booking_assistant(query: str) -> str:
        """Process flight booking queries.
        Args:
            query: A flight-related question
        Returns:
            Flight information and booking options
        """
        collector.record_handoff("flight", query)
        hook = ListMemoryHook(shared_memory, collector, "flight")
        agent = Agent(hooks=[hook], model=AGENT_MODEL_ID, system_prompt=FLIGHT_PROMPT)
        t0 = time.perf_counter()
        resp = agent(query)
        collector.record_agent_latency("flight", time.perf_counter() - t0)
        usage = getattr(resp, "usage", None) or {}
        collector.record_token_usage("flight",
            usage.get("inputTokens", 0), usage.get("outputTokens", 0))
        return str(resp)

    @tool
    def hotel_booking_assistant(query: str) -> str:
        """Process hotel booking queries.
        Args:
            query: A hotel-related question
        Returns:
            Hotel information and booking options
        """
        collector.record_handoff("hotel", query)
        hook = ListMemoryHook(shared_memory, collector, "hotel")
        agent = Agent(hooks=[hook], model=AGENT_MODEL_ID, system_prompt=HOTEL_PROMPT)
        t0 = time.perf_counter()
        resp = agent(query)
        collector.record_agent_latency("hotel", time.perf_counter() - t0)
        usage = getattr(resp, "usage", None) or {}
        collector.record_token_usage("hotel",
            usage.get("inputTokens", 0), usage.get("outputTokens", 0))
        return str(resp)

    @tool
    def itinerary_assistant(query: str) -> str:
        """Build a travel itinerary from flight and hotel results.
        Args:
            query: Request to build an itinerary
        Returns:
            A cohesive travel itinerary
        """
        collector.record_handoff("itinerary", query)
        hook = ListMemoryHook(shared_memory, collector, "itinerary")
        agent = Agent(hooks=[hook], model=AGENT_MODEL_ID, system_prompt=ITINERARY_PROMPT)
        t0 = time.perf_counter()
        resp = agent(query)
        collector.record_agent_latency("itinerary", time.perf_counter() - t0)
        usage = getattr(resp, "usage", None) or {}
        collector.record_token_usage("itinerary",
            usage.get("inputTokens", 0), usage.get("outputTokens", 0))
        return str(resp)

    hub = Agent(system_prompt=HUB_PROMPT, model=AGENT_MODEL_ID,
               tools=[flight_booking_assistant, hotel_booking_assistant, itinerary_assistant])

    for msg in conversation:
        turn_counter += 1
        collector.begin_turn(turn_counter, msg)
        print(f"\n{'='*60}")
        print(f"[{session_label}] Turn {turn_counter}: {msg}")
        print(f"{'='*60}")
        resp = hub(msg)
        collector.end_turn()
        print(f"\nHub: {str(resp)[:300]}...")

    return collector, shared_memory
```

Run a simple session:

```python
simple_conversation = [
    "Book a trip from LA to NYC, July 10 to July 17, 1 traveler. "
    "Budget $1800 for flights. I want morning flights and a hotel in Midtown with a pool.",
    "Now build me a day-by-day itinerary for the trip based on the flight and hotel you found.",
]

simple_metrics, simple_memory = run_hub_spoke_session(simple_conversation, "simple")
print_memory(simple_memory)
```

> **Production alternative:** Replace the Python list with AgentCore Memory (`bedrock_agentcore.memory.MemoryClient`) for persistent, session-scoped shared memory. The instrumentation pattern is identical — only the read/write calls change.

## Section 2: Peer-to-Peer Sequential Swarm

**Concept:** In a sequential peer-to-peer system there is no coordinator — agents run in a fixed order and share context through common working memory. Each peer reads what prior peers wrote, does its work, and appends its output. Context propagation failures are harder to detect: if the first peer writes stale or incomplete analysis, every downstream peer inherits that error. There's no hub to re-dispatch.

The key difference from hub-spoke: no handoff compression. Each peer receives the raw task directly. The "handoff" is implicit — peer 2 sees peer 1's output in shared memory.

**Build:** A simple loop replaces the coordinator. For each peer we explicitly read memory, run the agent, record metrics, and write back:

```python
PEER_CONFIGS = [
    ("market_trends",      MARKET_TRENDS_PROMPT),
    ("customer_insights",  CUSTOMER_INSIGHTS_PROMPT),
    ("strategy_synth",     STRATEGY_SYNTH_PROMPT),
]


def run_peers(task: str, shared_memory: list, collector: MetricsCollector,
              turn_number: int, session_label: str) -> dict:
    """Run all peers sequentially. Returns {agent_name: response}."""
    collector.begin_turn(turn_number, task)
    responses = {}

    for name, prompt in PEER_CONFIGS:
        # 1. Record handoff (task passed directly — no compression)
        collector.record_handoff(name, task)

        # 2. Read memory
        t0 = time.perf_counter()
        context = format_memory(shared_memory)
        collector.record_memory_read_latency(name, time.perf_counter() - t0)
        collector.record_retrieved_context(name, context)

        # 3. Run agent with memory in system prompt
        full_prompt = prompt
        if context:
            full_prompt += (
                f"\n\nShared memory from other agents:\n{context}"
                "\n\nUse this context. Reference specific details from other agents.")
        agent = Agent(name=name, model=AGENT_MODEL_ID, system_prompt=full_prompt)

        t0 = time.perf_counter()
        resp = agent(task)
        latency = time.perf_counter() - t0
        response_text = str(resp)

        # 4. Record response + latency + tokens
        collector.record_response(name, response_text)
        collector.record_agent_latency(name, latency)
        usage = getattr(resp, "usage", None) or {}
        collector.record_token_usage(name,
            usage.get("inputTokens", 0), usage.get("outputTokens", 0))

        # 5. Write to shared memory
        t0 = time.perf_counter()
        shared_memory.append({
            "agent": name, "role": "assistant",
            "content": response_text, "ts": time.time()})
        collector.record_memory_write_latency(name, time.perf_counter() - t0)

        responses[name] = response_text
        print(f"\n{name}: {response_text[:200]}...")

    collector.end_turn()
    return responses


def run_peer_session(conversation: list, session_label: str) -> tuple:
    """Run a full peer-to-peer session. Returns (collector, shared_memory)."""
    shared_memory = []
    collector = MetricsCollector(region=region)
    for i, task in enumerate(conversation, start=1):
        run_peers(task, shared_memory, collector, turn_number=i, session_label=session_label)
    return collector, shared_memory
```

Run a two-turn session where the scope expands mid-conversation:

```python
feedback_conversation = [
    "Analyse the fast-casual restaurant market in the United States. "
    "Key players include Chipotle, Panera Bread, and Sweetgreen. "
    "Target segment: health-conscious urban diners aged 25-45. "
    "Market size: $60 billion, growing at 8% annually.",

    "Actually, expand the analysis to cover all of North America, not just the US. "
    "Include Canadian and Mexican fast-casual chains. "
    "The North American market size is $85 billion.",
]

feedback_metrics, feedback_memory = run_peer_session(feedback_conversation, "feedback")
print_memory(feedback_memory)
```

**C2 Alignment** — measure how much peers converge or diverge using embedding similarity:

```python
import math


def get_titan_embedding(text: str, region: str = "us-west-2") -> list:
    """Get embedding vector from Bedrock Titan Embeddings V2."""
    client = boto3.client("bedrock-runtime", region_name=region)
    body = json.dumps({"inputText": text[:8000]})
    resp = client.invoke_model(
        modelId=EMBEDDING_MODEL_ID, body=body,
        contentType="application/json", accept="application/json")
    return json.loads(resp["body"].read())["embedding"]


def cosine_similarity(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


# Compute pairwise similarity between peer outputs
final_responses = {}
for entry in feedback_memory:
    final_responses[entry["agent"]] = entry["content"]  # keeps last per agent

embeddings = {}
for name, text in final_responses.items():
    embeddings[name] = get_titan_embedding(text, region=region)

names = list(embeddings.keys())
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        sim = cosine_similarity(embeddings[names[i]], embeddings[names[j]])
        print(f"{names[i]} ↔ {names[j]}: {sim:.3f}")
```

Low C2 alignment after a scope change means peers diverged — some updated their analysis while others inherited stale context from run 1.

## Section 3: Dynamic Peer-to-Peer Swarm

**Concept:** In a dynamic swarm, no predetermined sequence exists. Each peer decides whether to hand off — and to whom — based on its own reasoning via tool calls. This makes context propagation failures even harder to detect: not only can a peer inherit stale memory, but the entire execution path can diverge depending on which handoffs get called. A peer could even hand back to a previous peer if it needs more analysis.

The dispatcher pattern: each peer has `handoff_to_<agent>` tools. Calling one signals the next peer. If a peer responds without calling any handoff tool, the swarm ends. Order is emergent, not prescribed.

**Build:** Define handoff tools and a dispatcher loop:

```python
# Shared dispatcher state
_handoff_target = {"next": None}


@tool
def handoff_to_customer_insights(reason: str) -> str:
    """Hand off to the customer insights analyst.
    Args:
        reason: Why you're handing off (1 sentence)
    Returns:
        Confirmation that the handoff was queued
    """
    _handoff_target["next"] = "customer_insights"
    return f"Handoff queued: {reason}"


@tool
def handoff_to_strategy_synth(reason: str) -> str:
    """Hand off to the strategy synthesizer.
    Args:
        reason: Why you're handing off (1 sentence)
    Returns:
        Confirmation that the handoff was queued
    """
    _handoff_target["next"] = "strategy_synth"
    return f"Handoff queued: {reason}"


@tool
def handoff_to_market_trends(reason: str) -> str:
    """Hand off back to market trends analyst if more analysis is needed.
    Args:
        reason: Why more market analysis is needed
    Returns:
        Confirmation that the handoff was queued
    """
    _handoff_target["next"] = "market_trends"
    return f"Handoff queued: {reason}"


PEER_SPECS = {
    "market_trends": (MARKET_TRENDS_PROMPT, ["customer_insights"]),
    "customer_insights": (CUSTOMER_INSIGHTS_PROMPT, ["strategy_synth", "market_trends"]),
    "strategy_synth": (STRATEGY_SYNTH_PROMPT, []),  # terminal peer
}

HANDOFF_TOOL_MAP = {
    "market_trends": handoff_to_market_trends,
    "customer_insights": handoff_to_customer_insights,
    "strategy_synth": handoff_to_strategy_synth,
}


def build_peer(name: str, memory: list, collector: MetricsCollector) -> Agent:
    """Create a peer agent with handoff tools and memory hook."""
    base_prompt, allowed = PEER_SPECS[name]
    prompt = base_prompt + (
        f"\n\nYou are the '{name}' peer in a multi-agent swarm. Complete your analysis, "
        f"then decide: if another peer should continue, call a handoff tool. "
        f"If analysis is complete, respond WITHOUT calling any handoff tool — that ends the swarm.")
    tools = [HANDOFF_TOOL_MAP[target] for target in allowed]
    hook = ListMemoryHook(memory, collector, name)
    return Agent(name=name, model=AGENT_MODEL_ID, system_prompt=prompt,
                 tools=tools, hooks=[hook])
```

The dispatcher loop follows handoffs until a peer produces a final answer:

```python
def run_dynamic_swarm(task: str, shared_memory: list, collector: MetricsCollector,
                      turn_number: int, session_label: str, max_iterations: int = 6) -> dict:
    """Run the swarm starting with market_trends. Follows dynamic handoffs."""
    collector.begin_turn(turn_number, task)
    responses = {}
    current = "market_trends"
    iterations = 0

    while current and iterations < max_iterations:
        iterations += 1
        _handoff_target["next"] = None

        print(f"\n-- Iteration {iterations}: running {current} --")
        collector.record_handoff(current, task)

        agent = build_peer(current, shared_memory, collector)
        t0 = time.perf_counter()
        resp = agent(task)
        latency = time.perf_counter() - t0

        collector.record_agent_latency(current, latency)
        usage = getattr(resp, "usage", None) or {}
        collector.record_token_usage(current,
            usage.get("inputTokens", 0), usage.get("outputTokens", 0))

        responses[current] = str(resp)

        next_peer = _handoff_target["next"]
        if next_peer and next_peer in PEER_SPECS:
            print(f"  → {current} handed off to {next_peer}")
            current = next_peer
        else:
            print(f"  → {current} produced final answer, swarm ends")
            current = None

    collector.end_turn()
    return responses


# Run a dynamic swarm session
swarm_memory = []
swarm_collector = MetricsCollector(region=region)
run_dynamic_swarm(
    "Analyse the fast-casual restaurant market in the United States. "
    "Key players: Chipotle, Panera Bread, Sweetgreen. "
    "Target: health-conscious urban diners aged 25-45. Market size: $60B.",
    swarm_memory, swarm_collector, turn_number=1, session_label="dynamic")
```

The dynamic swarm reveals a new evaluation dimension: **path divergence**. Run the same task twice — the execution order may differ because handoff decisions are non-deterministic.

## Section 4: LLM-as-Judge Context Quality Metrics

**Concept:** Timing and token counts tell you *how much* coordination costs, but not *whether it worked*. Six semantic metrics — evaluated by an LLM judge — tell you whether context actually propagated correctly:

| Metric | What it measures |
|--------|-----------------|
| Context Freshness | Is the agent working with the latest information? |
| Handoff Completeness | Did the handoff include all facts the agent needs? |
| Context Utilization | Did the agent use the context it read from memory? |
| State Consistency | Do all agents agree on key facts? |
| Memory Write Accuracy | Is what the agent wrote to memory factually correct? |
| Redundant Context | How much repeated/irrelevant context was transferred? |

Each metric is a **binary pass/fail** judged by an LLM that sees the agent's input, retrieved context, and output. Binary verdicts avoid the boundary drift and middle-value hedging of 1–5 scales and give you a clear pass rate per metric; granularity comes from the six specific checks below, not from a graded score.

**Build:** The `LLMJudge` class wraps Bedrock Converse calls with structured JSON output:

```python
class LLMJudge:
    """Bedrock Claude for semantic evaluation of context quality."""

    def __init__(self, region: str = "us-west-2", model_id: str = JUDGE_MODEL_ID):
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id

    def _call(self, prompt: str) -> dict:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024, "temperature": 0.0,
            "messages": [{"role": "user", "content": prompt}],
        })
        resp = self.client.invoke_model(
            modelId=self.model_id, body=body,
            contentType="application/json", accept="application/json")
        text = json.loads(resp["body"].read())["content"][0]["text"]
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())

    def judge_context_freshness(self, latest_user_msg: str, retrieved_context: str,
                                agent_name: str) -> dict:
        if not retrieved_context.strip():
            return {"passed": True, "reasoning": "No prior context (first call)"}
        prompt = f"""Evaluate whether the {agent_name} agent's retrieved context reflects
the latest user requirements. Return a BINARY pass/fail — PASS only if the context is
fully current; when in doubt, fail it.

Latest user message: \"\"\"{latest_user_msg}\"\"\"
Retrieved context: \"\"\"{retrieved_context[:3000]}\"\"\"

Respond JSON only: {{"passed": <true|false>, "reasoning": "<one sentence>",
"stale_fields": ["list outdated fields"]}}"""
        return self._call(prompt)

    def judge_state_consistency(self, responses: Dict[str, str]) -> dict:
        agent_texts = "\n\n".join(
            f"[{name}]:\n{resp[:2000]}" for name, resp in responses.items())
        prompt = f"""Evaluate factual consistency across these agent responses.
Do agents agree on key facts (numbers, dates, constraints)?

{agent_texts}

Return a BINARY pass/fail — PASS only if agents agree on all key facts; FAIL if there is
any genuine contradiction.
Respond JSON only: {{"passed": <true|false>, "reasoning": "<one sentence>",
"contradictions": ["list genuine contradictions"]}}"""
        return self._call(prompt)
```

Add `evaluate_all()` to MetricsCollector to run all judges:

```python
def evaluate_all(self):
    """Run LLM-as-judge on every agent call in every turn."""
    judge = LLMJudge(region=self.region)
    for turn in self.turns:
        for rec in turn.agent_calls:
            rec.judge_scores["context_freshness"] = judge.judge_context_freshness(
                turn.original_query, rec.retrieved_context, rec.agent_name)
            # ... (handoff_completeness, context_utilization,
            #      write_accuracy, redundant_context follow same pattern)
            time.sleep(0.5)  # pace to avoid throttling

        # State consistency is cross-agent per turn
        if len(turn.agent_calls) >= 2:
            responses = {r.agent_name: r.response for r in turn.agent_calls}
            turn.state_consistency = judge.judge_state_consistency(responses)

# Attach to MetricsCollector
MetricsCollector.evaluate_all = evaluate_all
```

Run evaluation on a completed session:

```python
# Using the feedback session from Section 2
feedback_metrics.evaluate_all()

# Print results
for turn in feedback_metrics.turns:
    print(f"\nTurn {turn.turn_number}:")
    for rec in turn.agent_calls:
        fresh = rec.judge_scores.get("context_freshness", {}).get("passed", "?")
        print(f"  {rec.agent_name}: freshness={'PASS' if fresh is True else 'FAIL' if fresh is False else '?'}")
    sc = turn.state_consistency.get("passed", "?")
    print(f"  State Consistency: {'PASS' if sc is True else 'FAIL' if sc is False else '?'}")
    for c in turn.state_consistency.get("contradictions", []):
        print(f"    ⚠️ {c}")
```

**Interpreting results:** Context Freshness fails when an agent reads memory that doesn't reflect the latest user message. State Consistency fails when agents disagree on key facts. In the feedback session, expect failures on turn 2 — that's where the scope change tests whether peers reconcile or inherit stale context. Track the **pass rate per metric across turns** rather than an average score; "State Consistency fails on 2 of 5 turns" points straight at the turns to debug.

## Section 5: Conflict Detection and Comparative Analysis

**Concept:** The real test of a multi-agent system is what happens when constraints change mid-session. A user changes their budget, dates, or scope — does the system propagate the update to all agents, or do some work with stale information? Conflict detection compares a baseline session (no changes) against a conflict session (mid-session update) and measures the delta in context quality metrics.

The pattern: run the same architecture twice — once with stable inputs, once with a mid-session change. Compare Context Freshness and State Consistency scores. A well-coordinated system shows minimal degradation; a poorly-coordinated one shows freshness dropping to 2-3 and consistency contradictions appearing.

**Build:** Run baseline and conflict sessions, then compare:

```python
# Baseline: stable constraints
baseline_conversation = [
    "Book a trip from LA to NYC, July 10 to July 17, 1 traveler. "
    "Budget $1800 for flights. Morning flights, hotel in Midtown with a pool.",
]

# Conflict: budget and dates change mid-session
conflict_conversation = [
    "Book a trip from LA to NYC, July 10 to July 17, 1 traveler. "
    "Budget $1800 for flights. Morning flights, hotel in Midtown with a pool.",

    "Actually, my budget just dropped to $900 total for flights AND hotel combined. "
    "No pool requirement anymore. Also change dates to July 15-20.",

    "Now build the final itinerary with the updated budget and dates."
]

baseline_metrics, baseline_memory = run_hub_spoke_session(baseline_conversation, "Baseline")
conflict_metrics, conflict_memory = run_hub_spoke_session(conflict_conversation, "Conflict")

# Evaluate both
baseline_metrics.evaluate_all()
conflict_metrics.evaluate_all()
```

Generate a side-by-side comparison report:

```python
def comparison_report(a: MetricsCollector, b: MetricsCollector,
                      label_a: str, label_b: str) -> str:
    """Side-by-side comparison of two sessions."""
    lines = [f"### {label_a} vs {label_b}", ""]
    header = f"| Metric | {label_a} | {label_b} | Delta |"
    lines.extend([header, "|--------|----------|----------|-------|"])

    def pass_rate(collector, key):
        flags = []
        for t in collector.turns:
            for r in t.agent_calls:
                v = r.judge_scores.get(key, {}).get("passed")
                if v is not None:
                    flags.append(1 if v else 0)
        return sum(flags) / len(flags) if flags else 0

    for key, label in [
        ("context_freshness", "Context Freshness"),
        ("handoff_completeness", "Handoff Completeness"),
        ("context_utilization", "Context Utilization"),
        ("write_accuracy", "Write Accuracy"),
        ("redundant_context", "Context Efficiency"),
    ]:
        sa, sb = pass_rate(a, key), pass_rate(b, key)
        d = sb - sa
        lines.append(f"| {label} | {sa:.0%} | {sb:.0%} | {'+' if d>=0 else ''}{d:.0%} |")

    return "\n".join(lines)

print(comparison_report(baseline_metrics, conflict_metrics, "Baseline", "Conflict"))
```

Visualize with radar charts:

```python
import matplotlib.pyplot as plt
import numpy as np


def plot_context_metrics_radar(collector, session_label: str):
    """Radar chart of per-metric LLM-judge pass rates."""
    metrics = ["context_freshness", "handoff_completeness", "context_utilization",
               "write_accuracy", "redundant_context"]
    labels = ["Freshness", "Handoff", "Ctx Util", "Write Acc", "Low Redund"]

    scores = []
    for key in metrics:
        vals = []
        for t in collector.turns:
            for r in t.agent_calls:
                v = r.judge_scores.get(key, {}).get("passed")
                if v is not None:
                    vals.append(1 if v else 0)
        scores.append(sum(vals) / len(vals) if vals else 0)

    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    scores_plot = scores + [scores[0]]
    angles += [angles[0]]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, scores_plot, alpha=0.25, color="#2196F3")
    ax.plot(angles, scores_plot, "o-", linewidth=2, color="#2196F3")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title(f"Context Quality (pass rate) — {session_label}", pad=20)
    plt.tight_layout()
    return fig

plot_context_metrics_radar(baseline_metrics, "Baseline")
plt.show()
plot_context_metrics_radar(conflict_metrics, "Conflict")
plt.show()
```

**Key takeaways from conflict detection:**
- A low Context Freshness pass rate means agents are working with stale information
- A low State Consistency pass rate means agents disagree on key facts
- The delta in pass rates between baseline and conflict sessions quantifies your system's resilience to mid-session changes
- Production fix: version memory entries and re-dispatch affected agents after constraint updates

## Challenges

### Challenge: Evaluate a Customer Support Escalation System

Design and instrument a multi-agent evaluation for a **customer support escalation** domain. You must:

1. **Choose an architecture** — hub-spoke (support coordinator → specialist agents) or peer-to-peer (triage → investigation → resolution). Justify your choice.
2. **Build ≥3 agents** with distinct roles (e.g., triage agent, billing specialist, technical support, resolution writer).
3. **Instrument with MetricsCollector** — record handoffs, memory reads/writes, latency, and tokens for every agent call.
4. **Introduce a mid-session constraint change** — e.g., the customer escalates from billing to technical, or changes their account type mid-conversation. This must test whether agents propagate the update.
5. **Run baseline vs conflict sessions** and produce a comparison report showing metric deltas.
6. **Interpret the results** — identify ≥2 specific coordination failures from the metrics (e.g., "the resolution agent used the old account type because Context Freshness failed on turn 3").

**Assessment criteria:**
- Implements ≥1 architecture with ≥3 agents that have distinct system prompts
- Includes memory instrumentation recording handoffs + retrieved context + responses
- Introduces a constraint change that produces measurable freshness/consistency degradation
- Produces a comparison report with metric deltas between baseline and conflict sessions
- Identifies ≥2 specific coordination failures with evidence from metric scores

For the full capstone challenge integrating all workload evaluation concepts, see `CHALLENGE-capstone.md`.

## Wrap-Up

**Key takeaways:**
- Multi-agent failures are coordination failures — stale context, incomplete handoffs, and state inconsistencies between agents
- Hub-spoke gives predictable flow but creates a compression bottleneck at the coordinator
- Peer-to-peer (sequential or dynamic) distributes coordination but makes failures harder to trace
- Six LLM-as-judge metrics quantify context quality without manual labeling
- Comparing baseline vs conflict sessions reveals how resilient your architecture is to mid-session changes
- C2 alignment (embedding similarity) measures whether agents converge on a shared understanding

**This module does NOT cover:**
- Long-term memory strategies (RAG, vector stores, knowledge graphs)
- Agent-to-agent authentication or trust boundaries
- Production deployment of multi-agent systems (orchestration frameworks, error recovery)
- Cost optimization for multi-agent token usage

**Next steps:**
- Explore `CHALLENGE-capstone.md` for an integrative challenge combining all workload evaluation concepts
- Try swapping `AGENT_MODEL_ID` in the config to compare how different foundation models handle coordination
- Add a reconciliation step before the final agent to detect and resolve contradictions before they reach the user
