# Multi-Agent Shared-Context Evaluation Metrics

## 1. What Is Being Measured and Why

In a multi-agent system, the most important failures often occur *before* the final answer is produced. A sub-agent may receive stale context, a planner may update the budget but not the downstream hotel search, or two agents may work from different versions of the same itinerary. These are **memory and coordination failures**.

This sample focuses on those system behaviors. The objective is not only to know whether the final itinerary is good, but to know whether the agents collaborated with a **consistent shared reality**.

**The core evaluation question:**
> Did the agents operate on the same current facts and constraints, and did updates move through the system quickly and correctly?

## 2. Scope

Memory and coordination quality in multi-agent workflows, **independent of final answer accuracy**.

## 3. Metrics Overview

The metrics below are architecture-neutral. They apply regardless of how you wire your agents together.

| Metric | What it measures | Why it matters |
|--------|-----------------|----------------|
| Context Freshness | How often an agent used the latest available memory | High freshness = updates propagating before work starts |
| Handoff Completeness | How much required context was included in a delegation | Incomplete handoffs force agents to guess or use defaults |
| Context Utilization | Did the agent use the context it read from memory? | Low utilization means memory is being written but not read — a silent failure |
| State Consistency | Whether active agents agree on critical fields at the same stage | Disagreement means agents are working from different realities |
| Memory Write Accuracy | Whether what the agent wrote to memory is factually correct | Prevents spreading wrong facts as shared truth |
| Redundant Context Transfer | How much repeated/irrelevant context is sent between agents | Captures efficiency cost of over-sharing instead of curating state |
| Context Compression Ratio | Length ratio of handoff vs original (pure math) | Measures how much the hub compresses before delegating |
| Memory Read / Write Latency | Time spent on memory operations | Coordination overhead that directly impacts user-perceived latency |
| Coordination Latency % | `(read + write) / total agent time` | Shows what fraction of time is spent on coordination vs reasoning |
| Coordination Token % | `context tokens / total tokens` | Shows what fraction of tokens are spent on coordination vs generation |
| C2 Alignment | Cosine similarity of agent responses (Notebook 2 only) | Measures semantic agreement between peer agents |

## 4. Two Scenarios Evaluated

We evaluate these metrics across two orchestration patterns:

### Notebook 1 — Hub-and-Spoke with Local Memory (Travel Planning)

A coordinator delegates to three spokes (Flight, Hotel, Itinerary) via a shared **in-process Python list** that agents read from and append to. No cloud memory service required — good for fast iteration and for understanding the pattern before adding infrastructure.

We will run two sessions for metrics comparison:

- **Session 1:** Fixed budget, no mid-session changes.
- **Session 2:** User changes budget mid-session.

### Notebook 2 — Hub-and-Spoke with AgentCore Memory (Travel Planning)

Same hub-and-spoke setup and scenarios as Notebook 1, but the shared memory is backed by **AgentCore Short-Term Memory**. Agents read with `get_last_k_turns` and write with `create_event` against a managed memory resource. Use this when you need persistence, session isolation, or multi-process access.

Same two sessions as above, so the metrics can be compared head-to-head against the local-memory version.

### Notebook 3 — Peer-to-Peer with Local Memory (Market Research)

Three peer agents (Market Trends → Customer Insights → Strategy Synthesizer) collaborate through a **shared in-process Python list**. Each peer reads what prior peers wrote and appends its own contribution.

We will run two sessions for metrics comparison:

- **Session 1:** A research brief.
- **Session 2:** A research brief where the research scope changes after the first swarm runs.

---

## 5. How This Sample Works

### 5.1 Shared Memory — Two Backends

The hub-and-spoke scenario is shown with two interchangeable memory backends so you can see the tradeoffs:

- **Local memory (Notebook 1 and Notebook 3):** A plain Python list is passed into the agent hooks. Each agent reads the list on initialization (injecting prior entries into its system prompt) and appends its response on completion. Zero setup, zero cloud calls — the list lives for the duration of the session and is discarded at the end.

- **AgentCore Short-Term Memory (Notebook 2):** A managed memory resource stores conversation turns keyed by `session_id` and `actor_id`. Agents read with `get_last_k_turns` and write with `create_event`. This adds real network latency but gives you persistence, isolation, and cross-process access.

The `MetricsCollector` doesn't care which backend is in use. Notebooks wire the memory layer into the agent hooks, and the hooks call `collector.record_retrieved_context()`, `record_response()`, and the latency recorders — the collector evaluates whatever flowed through.

### 5.2 TurnRecord and AgentRecord — the observation layer

Memory alone isn't enough for evaluation. AgentCore Memory stores *what was written* but not *what was read*, not the original user query, and not timing or token data.

A **Turn** = one user message → all agent work → final output delivered to the user. Inside each Turn, there are one or more **AgentRecords** — one per agent invocation.

```
TurnRecord (1 per user message)
├── turn_number: 1
├── original_query: "Book trip LA→NYC, $1800..."
├── agent_calls: [
│     AgentRecord("flight", handoff=..., context=..., response=...),
│     AgentRecord("hotel",  handoff=..., context=..., response=...),
│     AgentRecord("itinerary", handoff=..., context=..., response=...),
│   ]
└── state_consistency: {score: 5, contradictions: []}
```

Each AgentRecord captures what memory doesn't store:

| Field | Source | Why memory doesn't have it |
|-------|--------|---------------------------|
| `original_query` | The user's actual message to the hub | Memory only stores the compressed handoff, not the original |
| `handoff_query` | What the hub sent to the spoke | Same as what's written to memory, but paired with the original for comparison |
| `retrieved_context` | What the agent read FROM memory | Memory doesn't record "what was read", only "what was written" |
| `response` | The agent's output | Also in memory, but here it's paired with the above for metric computation |
| Latency timers | `time.perf_counter()` around memory calls | Not stored anywhere else |
| Token counts | From Strands Agent response | Not stored anywhere else |
| LLM judge scores | Computed after the conversation | Not stored anywhere else |

**TurnRecord = memory contents + the metadata around it that you need for evaluation.**

### 5.3 Granularity

This works the same for both patterns:
- **Hub-and-spoke:** Turn 1 might have 3 agent calls (flight, hotel, itinerary). Turn 2 might have 2. Depends on what the hub dispatches.
- **Swarm:** Turn 1 (topic 1) might have 3 agent calls. Turn 2 (topic 2) might have 1. Depends on how many peers contribute.

Metric computation:
- **Per agent call:** Context Freshness, Handoff Completeness, Context Utilization, Write Accuracy, Redundant Context — one score per AgentRecord.
- **Per turn (cross-agent):** State Consistency — compares all AgentRecords within a turn for factual agreement.
- **Per session:** Each `run_session()` creates its own `MetricsCollector`. Metrics are self-contained — they surface problems without needing a baseline.

### Context Flow Trace

Each notebook includes a trace report showing the exact data flow per agent call:

```
═══════════════════════════════════════════════════════
 TURN 1 — flight
═══════════════════════════════════════════════════════

📨 Original user message:
   "Book a trip from LA to NYC, July 10-17, budget $1800..."

📤 Handoff query (what the hub sent):
   "Find morning flights LA→NYC, July 10-17, budget $1800"

📥 Memory read (what the agent retrieved):
   (empty — first turn)

💬 Agent response:
   "Delta DL123, 8:15am LAX→JFK, $650 round trip..."

📝 Written to memory: user msg + assistant response
```

This lets you follow context flowing through the system and spot where things break.

## 6. Metric Details

### Memory Context Metrics (LLM-as-judge, 1-5 scale)

| Metric | What it measures | Why it matters |
|--------|-----------------|----------------|
| Context Freshness | How often an agent used the latest available memory and plan versions | High freshness shows updates are propagating before work starts. Stale fields are listed in the judge output. |
| Handoff Completeness | How much of the required context was actually included in a delegation | Incomplete handoffs force sub-agents to guess, re-query, or use defaults |
| Context Utilization | Whether the agent actually incorporated the context it read from memory | Low utilization means memory is being written but not read — a silent failure |
| State Consistency | Whether active agents agree on critical fields at the same stage of a run | Measures whether the system is operating on one shared reality |
| Memory Write Accuracy | Whether what the agent wrote to memory is factually correct and consistent with its input | Prevents the system from spreading wrong facts as shared truth |
| Redundant Context Transfer | How much repeated or irrelevant context is sent between agents | Captures the efficiency cost of over-sharing instead of curating state |
| Context Compression Ratio | `len(handoff) / len(original)` — pure math, no LLM | Detects over-compression (losing facts) or under-compression (passing noise) |
| C2 Alignment | Cosine similarity of agent responses via Bedrock Titan embeddings (Notebook 2 only) | Measures semantic divergence between peer agents |

All semantic judgments use Claude Opus as LLM-as-judge. This catches paraphrasing — if the hub says "Los Angeles" and the spoke says "LA", the LLM understands they're the same.

### Memory Latency Metrics (timers + token counts)

| Metric | What it measures |
|--------|-----------------|
| Memory Read Latency | Time to read shared memory (`list` access for local; `get_last_k_turns` for AgentCore) |
| Memory Write Latency | Time to write shared memory (`list.append` for local; `create_event` for AgentCore) |
| Coordination Latency % | `(read + write) / total agent time` |
| Coordination Token % | `context tokens / total tokens` |

## 7. How Data is Collected

Three instrumentation points per agent call, plus turn boundaries:

| # | Where | What | How |
|---|-------|------|-----|
| — | Before dispatching agents | Turn start | `collector.begin_turn(turn_number, user_message)` |
| 1 | `@tool` function / before peer call | Handoff query | `collector.record_handoff(agent_name, query)` |
| 2 | `on_agent_initialized` hook | Retrieved memory context | `collector.record_retrieved_context(agent_name, context)` |
| 3 | `on_message_added` hook | Agent response | `collector.record_response(agent_name, response)` |
| — | After all agents finish | Turn end | `collector.end_turn()` |

Latency timers wrap the memory read/write calls inside the hooks (list access for local-memory notebooks, `get_last_k_turns` / `create_event` for the AgentCore notebook). Token usage comes from the Strands Agent response object. The AgentCore hook uses `finally` blocks for metrics recording so it's never skipped even if memory operations fail.

## File Structure

```
metrics/
├── README.md
├── requirements.txt
├── model_config.py                                ← Centralised model IDs and prompts
├── metrics_collector.py                           ← LLM judge + latency tracking + reports
├── 01-hub-spoke-local-memory.ipynb                ← Hub-and-spoke, Python list memory
├── 02-hub-spoke-agentcore-memory.ipynb            ← Hub-and-spoke, AgentCore Memory
└── 02-peer-to-peer-market-research-metrics.ipynb  ← Peer-to-peer, Python list memory
```

## Prerequisites

- Python 3.10+
- AWS credentials with access to Bedrock models (required for all notebooks) and AgentCore Memory (required for Notebook 2 only)
- Dependencies: `pip install -r requirements.txt`
