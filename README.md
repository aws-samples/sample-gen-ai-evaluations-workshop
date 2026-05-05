![Workshop Structure](https://github.com/aws-samples/sample-gen-ai-evaluations-workshop/blob/main/evals%20workshop.png?v=2 "Evals are so cool!")
# Generative AI Evaluations Workshop

This workshop teaches systematic approaches to evaluating Generative AI workloads for production use. You'll learn to build evaluation frameworks that go beyond basic metrics to ensure reliable model performance while optimizing cost and performance.

[Click here for a slide deck](https://d2ot4ns4zf41bm.cloudfront.net/slides/Gen+AI+Evals+Workshop.pptx) which covers the basics of evaluations and includes an overview of this workshop.

## How to use this repository

We strongly recommend going in order through the [Foundational Evaluations](Foundational%20Evaluations/) modules. These cover the core of generative AI evaluations which will be critical in all workloads. After that, please feel free to select any of the workload- and framework-specific modules in any order, according to what is most relevant to you.

### Interactive Learning

As an alternative to working through the Jupyter notebooks directly, you can use the [Interactive Learning](Interactive%20Learning/README.md) module as a skill file in your AI coding assistant (Kiro, Claude, or similar). The assistant reads the workshop's notebooks and skill docs, then guides you through hands-on challenges — asking you to write code, explain concepts, and debug configurations rather than passively reading. Tell the agent which module you want to work on and it will present exercises one at a time, check your understanding, provide hints when you're stuck, and adapt to your pace.

## What You'll Learn

### [Foundational Evaluations](Foundational%20Evaluations/) - Do all of these in order!
- 01 Operational Metrics: evaluate how your workload is running in terms of cost and performance.
- 02 Quality Metrics: evaluate and tune the quality of your results.
- 03 Understanding Failures: discover failure patterns by reading agent traces.
- 04 Agentic Metrics: evaluate your agents and use agents for evaluation.

### Optional Modules - Do any of these in any order!
- [Workload Specific Evaluations](Workload%20Specific%20Evaluations/)
  - Intelligent Document Processing: evaluate structured data extraction accuracy with field-level precision/recall.
  - Guardrails: configure and test content filters, grounding checks, alignment, and operational controls.
  - Basic RAG: evaluate retrieval quality and end-to-end answer generation with precision@k, NDCG, and faithfulness scoring.
  - MultiModal RAG: evaluate retrieval across text, vision, and audio modalities using ImageBind embeddings.
  - Speech to Speech: end-to-end evaluation of Nova Sonic interactions using CloudWatch telemetry and LLM-as-Judge.
  - Automated Reasoning Evaluations: verify LLM outputs against formal policy rules using SMT solver-based guardrails.
  - Tool Calling: evaluate agent tool-calling behavior without real tool execution using five progressively sophisticated approaches.
  - Chatbot: evaluate multi-turn conversational AI with simulated users, custom binary evaluators, and synthetic data generation.
  - Red Teaming: systematically probe AI systems with adversarial inputs using Promptfoo across LLM apps, RAG, agents, and guardrails.
  - Multiagent Shared Context Evaluation: measure memory coordination quality in multi-agent systems across hub-spoke and peer-to-peer patterns.
  - Coding Assistant: coming soon.
  
- [Framework Specific Evaluations](Framework%20Specific%20Evaluations/)
  - Prompt Foo: configure YAML-based evaluations, write assertion test cases, and compare models across providers.
  - Strands: evaluate agents using the Strands Evals SDK with output quality, trajectory, and custom evaluators.
  - AgentCore: evaluate agents deployed on Amazon Bedrock AgentCore Runtime with LLM-as-Judge and CloudWatch log analysis.
  - AgentCore Runtime Evals: run native AgentCore Evaluations API with built-in evaluators for helpfulness and tool selection accuracy.
  - DSPy: optimize prompts automatically with BootstrapFewShot and measure improvement with custom metrics.
  - MLflow: track and compare evaluation experiments using MLflow with Amazon Bedrock.
  - DeepEval: coming soon.

- [Interactive Learning Mode](Interactive%20Learning/README.md): guided challenges, exercises, and real-time feedback with an interactive tutor.

## Prerequisites

- AWS account with Amazon Bedrock enabled
- Basic Python and ML familiarity
- No prior AI evaluation experience required

## Getting Started

1. Clone the repository
2. Configure AWS credentials
3. Work through the Foundational Evaluations modules in order
4. Pick from the Workload Specific and Framework Specific modules based on what's relevant to you

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.
