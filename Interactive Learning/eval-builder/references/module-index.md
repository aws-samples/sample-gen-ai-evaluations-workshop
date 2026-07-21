# Module index — workload type → evaluation modules

Use this index during **Phase 1 (Discover)** to detect workload type(s) from repository signals,
and during **Phase 2 (Map & Report)** to choose which module references to build from. Detection is
heuristic — **always confirm the type with the user** rather than assuming (Req 4.4).

## Detection signals → workload type → modules

| Workload type                | Detection signals (examples)                                                                                           | Modules to apply                                    |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| **RAG**                      | vector-store / retriever libs, `retrieve(`, embeddings, `knowledge base`, chunking, reranking                          | `rag`, `quality`, `operational`                     |
| **Agent (tool-calling)**     | tool/function schemas, `toolConfig`, `tools=[...]`, `toolUse` blocks, agent frameworks (Strands), mock/live tool loops | `tool-calling`, `agentcore`, `operational`          |
| **Multi-agent**              | multiple agents, coordinator/hub + spokes, handoffs, shared memory, swarm                                              | `multiagent-context`, `tool-calling`, `operational` |
| **Chatbot (multi-turn)**     | conversation/session state, message history, turn loops, user simulation                                               | `chatbot`, `quality`, `operational`                 |
| **IDP / extraction**         | document parsing, field extraction, output schemas, PDF/OCR pipelines                                                  | `structured-data`, `quality`, `operational`         |
| **Any LLM output**           | direct prompt→response calls (`converse`, `invoke_model`) with no other signal                                         | `quality`, `operational`                            |
| **AgentCore-deployed agent** | `bedrock-agentcore`, `@app.entrypoint`, `invoke_agent_runtime`, CloudWatch runtime log groups                          | `agentcore`, `tool-calling`, `operational`          |

## How to use the mapping

- **`operational` and `quality` are near-universal baselines.** Almost every workload benefits from
  operational metrics (cost, latency, error rate) and quality checks. Include them unless the user
  opts out.
- **Layer workload-specific modules on top.** A tool-calling agent gets `tool-calling` +
  `operational`; if it is deployed on AgentCore, add `agentcore`; if it coordinates other agents,
  add `multiagent-context`.
- **Multiple types can match one repo.** A repo can be both a RAG pipeline and a tool-calling agent.
  Each matched workload/eval type becomes **its own tab** in the Phase 4 HTML report.
- **Confirm before building.** Present detected types and proposed modules in the Phase 2 report and
  wait for approval.

## v1 reference availability

Distilled, build-ready references shipped in this skill:

| Module                | Reference file                     | Status                                           |
| --------------------- | ---------------------------------- | ------------------------------------------------ |
| Operational           | `references/operational.md`        | ✅ available                                     |
| Tool-calling          | `references/tool-calling.md`       | ✅ available                                     |
| AgentCore             | `references/agentcore.md`          | ✅ available                                     |
| Multi-agent context   | `references/multiagent-context.md` | ✅ available                                     |
| Quality               | `references/quality.md`            | ⏳ deferred (pending workshop notebook decision) |
| RAG                   | `references/rag.md`                | ⏳ not yet distilled                             |
| Chatbot               | `references/chatbot.md`            | ⏳ not yet distilled                             |
| Structured data / IDP | `references/structured-data.md`    | ⏳ not yet distilled                             |

If a mapped module has no reference yet, tell the user it is not available in this version and offer
to build the evals for the modules that are, plus the universal `operational` baseline. The index is
extensible — add one `references/<module>.md` per module as they are distilled.
