# Red Teaming for GenAI Applications

Red teaming is the practice of systematically probing an AI system with adversarial inputs to uncover vulnerabilities before real users do. Rather than testing whether a system works correctly under normal conditions, red teaming asks: *how does it fail when someone actively tries to break it?*

This module uses [Promptfoo](https://www.promptfoo.dev/), an open-source evaluation framework, to automate red teaming of applications built on Amazon Bedrock. We cover four workload types: general LLM applications powered by Bedrock foundation models, RAG pipelines using Amazon Bedrock Knowledge Bases, agentic systems built with Strands Agents on Amazon Bedrock AgentCore, and Amazon Bedrock Guardrails configurations.

## Why Red Team?

Foundation models are susceptible to attacks that traditional software testing does not cover. These include prompt injection (tricking the model into following attacker instructions), jailbreaking (bypassing safety filters), data exfiltration (leaking sensitive context), and harmful content generation. While Amazon Bedrock Guardrails provide built-in defenses against many of these attacks, red teaming validates that those guardrails — and your application logic around them — hold up under sustained, creative adversarial pressure. Red teaming surfaces these risks with concrete, reproducible test cases rather than theoretical threat models.

## How Promptfoo Red Teaming Works

Promptfoo's red teaming pipeline has three components:

1. **Plugins** generate adversarial inputs targeting specific vulnerability categories (e.g., PII extraction, prompt injection, harmful content). Each plugin produces test cases tailored to your application's stated purpose.

2. **Strategies** wrap those inputs in delivery techniques designed to bypass defenses — encoding tricks, multi-turn conversations, jailbreak patterns, and other evasion methods.

3. **Graders** evaluate model responses to determine whether the attack succeeded or the system held firm.

The separation between *what* is tested (plugins) and *how* it is delivered (strategies) allows broad coverage across dozens of vulnerability types while keeping configuration manageable.

### Configuration

Red team tests are defined in a `promptfooconfig.yaml` file that specifies:

- **Target**: the Amazon Bedrock model, API endpoint, or application under test
- **Purpose**: a plain-language description of what the application does (this drives how adversarial inputs are tailored)
- **Plugins**: which vulnerability categories to test (harmful content, PII, injection, etc.)
- **Strategies**: which attack delivery techniques to apply (encoding, jailbreaks, multi-turn)

> **Note:** Promptfoo uses a separate "attacker" model to generate adversarial inputs, distinct from the model being tested.

### Vulnerability Categories

Promptfoo covers 50+ vulnerability types organized into these families:

| Category | Examples |
|----------|----------|
| **Harmful content** | Hate speech, toxicity, dangerous advice, illegal activity |
| **Privacy** | PII leakage, COPPA/FERPA violations, training data extraction |
| **Security** | Prompt injection, SQL injection, shell injection, unauthorized access |
| **Misinformation** | Hallucination, excessive agency, impersonation |
| **Custom** | Organization-specific policies and behavioral constraints |

## Submodules

Each subfolder applies red teaming to a different workload type:

| Submodule | Description |
|-----------|-------------|
| [`04-12-01-llm-app-red-teaming`](04-12-01-llm-app-red-teaming/) | Red teaming a basic Bedrock-powered LLM application — foundational setup, plugin selection, and interpreting results |
| [`04-12-02-testing-bedrock-guardrails`](04-12-02-testing-bedrock-guardrails/) | Testing Amazon Bedrock Guardrails configurations against adversarial inputs to validate filter and policy effectiveness |
| [`04-12-03-RAG-red-teaming`](04-12-03-RAG-red-teaming/) | Red teaming RAG pipelines built with Amazon Bedrock Knowledge Bases — indirect prompt injection via retrieved documents, context poisoning |
| [`04-12-04-agent-red-teaming`](04-12-04-agent-red-teaming/) | Red teaming agentic applications built with Strands Agents and Amazon Bedrock AgentCore — tool misuse, privilege escalation, multi-step exploits |

## Prerequisites

- AWS account with Amazon Bedrock model access
- AWS CLI configured with appropriate credentials
- Python 3.10+
- Node.js 20+ (required for Promptfoo)
- Install Promptfoo: `npm install -g promptfoo`
- Install Python dependencies: `pip install -r requirements.txt` (or `uv pip install -r requirements.txt`)

## Getting Started

1. Read through this README to understand the red teaming concepts and Promptfoo's approach.
2. Start with [`04-12-01-llm-app-red-teaming`](04-12-01-llm-app-red-teaming/) for the foundational walkthrough.
3. Proceed through the remaining submodules in order — each builds on the prior concepts while introducing workload-specific considerations.
