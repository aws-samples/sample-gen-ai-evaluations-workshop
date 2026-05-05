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

**Custom evaluations for agents deployed on Amazon Bedrock AgentCore Runtime**

This module deploys a Strands-based city search agent to AgentCore Runtime and evaluates it using custom LLM-as-Judge scoring across five quality dimensions (helpfulness, accuracy, clarity, professionalism, completeness). It also demonstrates extracting tool usage data from CloudWatch logs to compute tool selection precision/recall, and provides comprehensive visualization of evaluation results.

Key topics: AgentCore Runtime deployment, multi-dimensional LLM-as-Judge, CloudWatch log-based tool evaluation, X-Ray distributed tracing, account ID masking for safe commits.

---

### [AgentCore Runtime Evals](AgentCore%20Runtime%20Evals/)

**Native AgentCore Evaluations API with built-in evaluators**

Unlike the custom evaluation approach in the AgentCore module, this module uses the native AgentCore Evaluations API — built-in evaluators that analyze full agent execution traces directly from CloudWatch. The built-in evaluators (`Builtin.Helpfulness`, `Builtin.ToolSelectionAccuracy`) have access to every LLM call, tool invocation, and intermediate reasoning step, enabling assessment of the agent's reasoning process rather than just its final output.

Key topics: `evaluate()` API, built-in evaluators, session span retrieval, score/label/explanation interpretation, on-demand vs. online evaluation modes.

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

Coming soon.

---

## Prerequisites

All modules require:
- AWS account with Amazon Bedrock model access
- Python 3.10+
- AWS credentials configured

Individual modules may have additional requirements (Node.js for PromptFoo, Docker for AgentCore). See each module's README for specific setup instructions.
