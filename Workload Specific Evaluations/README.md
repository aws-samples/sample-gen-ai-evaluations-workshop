# Workload Specific Evaluations

This section demonstrates evaluation techniques tailored to specific generative AI workload types. Each module is self-contained and can be completed independently in any order. Choose the workloads most relevant to what you're building.

These modules assume you have completed the [Foundational Evaluations](../Foundational%20Evaluations/) and are familiar with core evaluation concepts like LLM-as-Judge, metrics design, and test case construction.

---

## Modules

### [Intelligent Document Processing](Intelligent%20Document%20Processing/)

**Evaluate structured data extraction accuracy**

Focuses on evaluating the accuracy of structured data extraction from documents — the kind used in IDP pipelines that pull invoice numbers, vendor names, and totals from business documents. You classify extraction results into five categories (true positive, true negative, false alarm, false discovery, false negative), compute precision/recall/F1/accuracy at both field and document levels, and visualize performance to identify weak spots.

Key topics: field-level classification, five-category extraction evaluation, precision/recall/F1 metrics, ground truth comparison, visual performance analysis.

---

### [Guardrails](Guardrails/)

**Configure, test, and evaluate Amazon Bedrock Guardrails**

Covers five types of guardrails for generative AI applications: content filters for trust and safety, contextual grounding to prevent hallucinations, alignment with domain context for behavioral steering, operational controls (rate limits, step limits, token limits), and automated reasoning for formal policy adherence. Includes building an automated evaluation harness that measures guardrail accuracy, precision, recall, and latency against structured test datasets.

Key topics: content filters, topic policies, contextual grounding checks, alignment techniques, operational guardrails, Strands Agent hooks for step limiting, automated evaluation harness, adversarial test design.

---

### [Basic RAG](Basic%20RAG/)

**Evaluate retrieval quality and end-to-end RAG generation**

An abridged version of a deeper RAG evaluation repository, demonstrating how to evaluate both the retrieval task and the entire RAG system end-to-end. Covers information retrieval metrics (precision@k, recall@k, NDCG@k) to assess whether vector search returns the right documents, and LLM-as-Judge faithfulness scoring to measure whether generated answers are grounded in retrieved context.

Key topics: embeddings validation, chunking strategy evaluation, retrieval metrics (precision@k, recall@k, NDCG@k), end-to-end RAG testing, ChromaDB vector store.

---

### [MultiModal RAG](MultiModal%20RAG/)

**Evaluate retrieval across text, vision, and audio modalities**

A comprehensive multimodal RAGAS implementation combining traditional RAG metrics with custom multimodal evaluation using ImageBind embeddings. Evaluates retrieval performance across text, vision, and audio modalities using the Cinepile movie dataset. Compares retrieval strategies (text-only, vision-only, audio-only, full multimodal) and provides detailed performance analysis with visualizations.

Key topics: ImageBind unified embeddings, FAISS indices, cross-modal retrieval, multimodal faithfulness/relevancy metrics, strategy comparison, SageMaker integration.

---

### [Speech to Speech](Speech%20to%20Speech/)

**End-to-end evaluation of Nova Sonic speech-to-speech interactions**

Provides an automated evaluation pipeline for Speech-to-Speech (Nova Sonic) interactions through Playwright E2E testing, CloudWatch telemetry extraction, and LLM-as-Judge scoring. Includes a sample full-stack application (React + Python WebSocket), pre-configured test scenarios with audio files, a manual annotation UI for session-to-category mapping, and comprehensive evaluation reporting.

Key topics: Playwright automated testing, CloudWatch trace extraction, OpenTelemetry spans, LLM-as-Judge conversation evaluation, session annotation UI, multi-run evaluation merging.

---

### [Automated Reasoning Evaluations](Automated%20Reasoning%20Evaluations/)

**Verify LLM outputs against formal policy rules**

Demonstrates evaluating Automated Reasoning (AR) Checks in Amazon Bedrock Guardrails — a two-step pipeline that translates natural language into logical formulas and verifies them against policy rules via an SMT solver. Covers the 7 validation result types, building AR policies from regulatory documents, and evaluating policy quality with metrics including false valid rate, consistency accuracy, macro F1, and translation confidence.

Key topics: SMT solver verification, 7 validation result types, confusion matrix analysis, false valid rate (safety metric), translation fidelity gap, per-type precision/recall/F1.

---

### [Tool Calling](Tool%20Calling/)

**Evaluate agent tool-calling behavior without real tool execution**

Tackles one of the hardest problems in agentic AI evaluation: rigorously evaluating tool-calling behavior without actually executing tools. Uses the Amazon Bedrock Converse API to demonstrate six progressively sophisticated approaches — from static trajectory analysis ($0, instant) through schema validation, mock tool evaluation, LLM-as-Judge, multi-turn simulation, and fully synthetic Strands ToolSimulator + ActorSimulator evaluation.

Key topics: evaluation pyramid (cost vs. depth tradeoffs), static trajectory analysis, JSON Schema parameter validation, mock tool responses, LLM-as-Judge trajectory scoring, Strands ToolSimulator/ActorSimulator.

---

### [Chatbot](Chatbot/)

**Evaluate multi-turn conversational AI systems**

Demonstrates evaluating multi-turn conversational AI using Strands Agents Evals and DeepEval with Amazon Bedrock. Builds a travel booking assistant, simulates realistic multi-turn conversations, and evaluates against domain-specific pass/fail criteria derived from real failure modes. Covers conversation simulation (ActorSimulator, ConversationSimulator), custom binary evaluators, dimension-driven synthetic data generation, and end-to-end evaluation pipelines.

Key topics: multi-turn evaluation, ActorSimulator/ToolSimulator, ConversationSimulator, custom binary evaluators (ConversationalGEval, ConversationalDAGMetric), synthetic data from structured dimensions, pass rate per failure mode.

---

### [Red Teaming](Red%20Teaming/)

**Systematically probe AI systems with adversarial inputs**

Uses Promptfoo to automate red teaming across four workload types: general LLM applications, RAG pipelines with Amazon Bedrock Knowledge Bases, agentic systems built with Strands Agents on AgentCore, and Bedrock Guardrails configurations. Covers 50+ vulnerability categories including harmful content, privacy violations, security exploits, and misinformation — delivered through encoding tricks, multi-turn conversations, jailbreak patterns, and other evasion strategies.

Key topics: Promptfoo red teaming pipeline (plugins → strategies → graders), vulnerability categories, prompt injection, jailbreaking, data exfiltration, indirect injection via RAG, agent tool misuse.

---

### [Multiagent Shared Context Evaluation](Multiagent%20Shared%20Context%20Evaluation/)

**Measure memory and coordination quality in multi-agent systems**

Evaluates whether agents in a multi-agent system operate on the same current facts and constraints, and whether updates propagate correctly. Covers four orchestration patterns (hub-spoke with local memory, hub-spoke with AgentCore Memory, peer-to-peer dynamic swarm, peer-to-peer sequential pipeline) with metrics for context freshness, handoff completeness, state consistency, memory write accuracy, and coordination latency.

Key topics: semantic metrics (LLM-as-Judge), static metrics (latency, compression ratio), embedding metrics (peer alignment), TurnRecord/AgentRecord instrumentation, MetricsCollector, hub-spoke vs. peer-to-peer patterns.

---

### [Coding Assistant](Coding%20Assistant/)

Coming soon.

---

## Prerequisites

All modules require:
- AWS account with Amazon Bedrock model access
- Python 3.10+
- AWS credentials configured

Individual modules may have additional requirements (Node.js for Red Teaming/Promptfoo, GPU instance for MultiModal RAG, Docker for Speech to Speech). See each module's README for specific setup instructions.
