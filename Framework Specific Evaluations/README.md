# Framework Specific Evaluations

This section demonstrates how to evaluate generative AI workloads using popular evaluation frameworks and tools. Each module is self-contained and can be completed independently in any order. Choose the frameworks most relevant to your stack.

These modules assume you have completed the [Foundational Evaluations](../Foundational%20Evaluations/) and are familiar with core evaluation concepts like LLM-as-Judge, metrics design, and test case construction.

---

## Modules

### [Prompt Foo](Prompt%20Foo/)

**Eval-focused framework for input→output comparison**

PromptFoo is an open-source, vendor-agnostic CLI and library for evaluating LLM applications. You define inputs, send them through prompt templates to one or more providers, and compare outputs against expected results using assertions. This module walks through configuring YAML-based evaluations, writing test cases with expected-output assertions in CSV format, running evaluations from the CLI, and comparing model performance across multiple Bedrock providers in a single run.

Key topics: YAML configuration, prompt templates as Python functions, `__expected` assertion convention, multi-provider comparison, pass/fail reporting.

---

### [Strands](Strands/)

**Agent-focused evaluation using the Strands Evals SDK**

The Strands Agents Evaluation SDK provides structured evaluation for AI agents — measuring not just what an agent says, but how it arrives at its answers. This module covers defining test cases with the `Case` class, evaluating output quality with LLM-as-Judge rubrics, assessing tool-usage trajectories, building custom programmatic evaluators, and combining multiple evaluators in a single experiment for multi-dimensional scoring.

Key topics: `OutputEvaluator` with custom rubrics, `TrajectoryEvaluator` for tool selection assessment, custom `Evaluator` subclasses, `Experiment` management, `tools_use_extractor`.

---

### [AgentCore](AgentCore/)

**Run, observe, evaluate, and improve agents with Amazon Bedrock AgentCore**

This progressive six-notebook module treats AgentCore Runtime, Observability, and Evaluations as one workflow. It deploys one deterministic Strands city-information agent, follows sessions through OpenTelemetry traces, applies built-in evaluators at session, trace, and tool-call levels, adds stable ground truth and curated datasets, builds focused custom evaluators, and extends the same evidence into batch evaluation, sampled online monitoring, simulation, insights, recommendations, and controlled A/B testing.

Key topics: AgentCore CLI project workflow, Runtime deployment, CloudWatch Transaction Search, session/trace/tool-call evaluation levels, `EvaluationClient`, dataset runners, ground-truth reference inputs, binary LLM-as-a-Judge, code-based evaluators, evaluator calibration, batch evaluation, online sampling, simulated users, insights, recommendations, configuration bundles, and cleanup.

---

### [DSPy](DSPy/)

**Prompt optimization as an evaluation loop**

DSPy collapses the evaluate→improve cycle into a single automated loop. Instead of hand-writing prompts, you declare typed signatures (input/output contracts), define a metric that scores quality, and let an optimizer (`BootstrapFewShot`) find the best few-shot demonstrations automatically. The optimizer runs your metric on training examples, keeps the best-scoring outputs, and produces a portable JSON artifact. This module builds a city Q&A system, measures accuracy with percentage-error metrics, and demonstrates before/after improvement.

Key topics: `Signature` declarations, `dspy.Evaluate`, `BootstrapFewShot` optimization, `ChainOfThought` modules, enhanced metrics with LLM-as-judge faithfulness, portable JSON artifacts.

---

### [MLflow](MLflow/)

**Experiment tracking and model comparison with MLflow**

This module demonstrates using MLflow to evaluate the generation step of a RAG pipeline on Amazon Bedrock and track all results (metrics, parameters, artifacts, traces) in a single place. It runs 18 scorers across five categories — MLflow built-in LLM-as-Judge, custom Guidelines, custom `make_judge` scorers, DeepEval metrics via Bedrock, and code-based heuristics — then compares multiple model runs side-by-side. Optionally bootstraps a serverless SageMaker-managed MLflow App for persistent tracking.

Key topics: `mlflow.genai.evaluate()`, 18 evaluation scorers, RAG faithfulness/hallucination metrics, SageMaker MLflow App bootstrap, run comparison via `mlflow.search_runs`.

---

### [DeepEval](DeepEval/)

**Pytest-native LLM evaluation with 50+ metrics**

DeepEval is an open-source evaluation framework with 50+ built-in metrics spanning RAG, agents, multi-turn conversations, safety, and multimodal applications, plus custom metrics defined in natural language for any criteria. This module evaluates a RAG pipeline by scoring retrieval quality and generation quality separately, then cross-referencing the results to diagnose whether failures originate in the retriever or the generator. Compares two candidate models on a 20-example AWS Q&A dataset with mixed retrieval difficulty.

Key topics: retrieval evaluation, generation evaluation, custom LLM-as-a-judge metrics, failure diagnosis across RAG stages.

---

## Prerequisites

All modules require:
- AWS account with Amazon Bedrock model access
- Python 3.10+
- AWS credentials configured

Individual modules may have additional requirements (Node.js for PromptFoo and the AgentCore CLI). See each module's README for specific setup instructions.
