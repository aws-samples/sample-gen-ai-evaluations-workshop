---
name: promptfoo-basic-evaluation
description: Set up and run your first promptfoo evaluation against Amazon Bedrock models. Activate when asked to "evaluate prompts with promptfoo", "set up promptfoo", "run model evaluations", "compare LLM outputs", or "test prompt quality".
---

# Basic Promptfoo Evaluation Setup and Execution

Build a complete promptfoo evaluation pipeline that tests an email classification prompt against multiple Bedrock models and validates outputs with assertions. Promptfoo treats models as functions (input→output), making it an eval-focused framework distinct from orchestration tools.

## Prerequisites
- Completion of foundational prompt engineering modules (concepts: prompt templates, model invocation, Bedrock access)
- Source notebook: `05-framework-specific-evaluations/05-01-Prompt-Foo/05-01-Promptfoo-basic.ipynb`
- AWS services: Amazon Bedrock (Nova Lite, Claude 3.5 Haiku)
- Tools: Node.js (for `npm install -g promptfoo`)

## Learning Objectives
By the end of this module, you will:
- Configure a promptfoo evaluation pipeline with YAML config, prompt files, and test datasets
- Execute evaluations against multiple Bedrock providers simultaneously
- Define test cases with expected outputs and interpret pass/fail assertion results
- Compare model performance across classification categories

## Setup

```python
# Install promptfoo CLI globally
# Run in terminal or notebook cell:
# !npm install -g promptfoo --loglevel=error --no-fund

# Project structure you'll create:
# ./prompts.py            - Prompt template functions
# ./dataset.csv           - Test cases with expected outputs
# ./promptfooconfig.yaml  - Evaluation configuration
```

## Section 1: Prompt Template Design

**Concept:** Promptfoo loads prompts from Python functions, enabling version control, reuse, and parameterization. Each function receives variables from test cases and returns the complete prompt string. This separation keeps prompt logic independent of evaluation config.

**Build:**

```python
# prompts.py
def classify_email(email_content):
    return f"""You are an AI assistant for GlobalMart's customer support team. Your task is to classify the following email into one of these categories: Order Issues, Product Inquiries, Technical Support, or Returns/Refunds.

Email content: {email_content}

Provide your classification as a single word or phrase, choosing from the categories mentioned above. Do not include any explanation or additional text.

Classification:"""
```

## Section 2: Test Case Dataset

**Concept:** Promptfoo reads test cases from CSV files where column names map to prompt variables. The special `__expected` column defines the ground-truth label for automatic assertion. This input→expected-output pattern is the core of eval-focused testing: you define what correct looks like, then measure how often the model gets there.

**Build:**

```csv
email_content,__expected
"Hi, I ordered a laptop last week, but I haven't received any shipping update. Can you help?",Order Issues
"I'm having trouble logging into my account. It keeps saying my password is incorrect even though I'm sure it's right.",Technical Support
"Do you have the latest iPhone model in stock? I couldn't find it on your website.",Product Inquiries
"I received my order yesterday, but the shirt is the wrong size. I'd like to return it for a refund.",Returns/Refunds
"Can you tell me when the next sale is? I'm looking to buy a new TV.",Product Inquiries
"My order arrived damaged. What should I do?",Returns/Refunds
"How do I track my recent order?",Order Issues
"I bought a blender from your store, but it's not working. Is there a warranty?",Technical Support
"I want to change the shipping address for my recent order. Is that possible?",Order Issues
"What's your return policy for electronics?",Returns/Refunds
```

## Section 3: YAML Configuration

**Concept:** The `promptfooconfig.yaml` ties together prompts, providers, and tests into a single evaluation run. You can test the same prompt against multiple models in one pass — critical for model selection decisions. The config is the orchestration layer of the eval pipeline.

**Build:**

```yaml
description: "GlobalMart Email Classification Evaluation"

prompts:
  - prompts.py:classify_email

providers:
  - id: bedrock:us.amazon.nova-lite-v1:0
    label: "Nova Lite"
    config:
      region: us-west-2

  - id: bedrock:us.anthropic.claude-3-5-haiku-20241022-v1:0
    label: "Haiku 3.5"
    config:
      region: us-west-2

tests:
  - file://dataset.csv
```

## Section 4: Running and Interpreting Evaluations

**Concept:** The `promptfoo eval` command executes your full test matrix (prompts × providers × test cases) and reports pass/fail per assertion. The `--no-cache` flag ensures fresh model calls each run — important during development when you need to observe variance. Results can be shared via `promptfoo share` for team collaboration.

**Build:**

```bash
# Run evaluation (fresh results, no progress bar for notebook compatibility)
promptfoo eval --no-progress-bar --no-cache

# Share results (requires free promptfoo account — one-time setup)
# promptfoo auth login --host https://api.promptfoo.app --api-key YOUR_KEY
# promptfoo share
```

After execution, examine:
1. **Overall accuracy** — percentage of emails correctly classified per model
2. **Per-category performance** — which categories each model handles best
3. **Misclassification patterns** — systematic errors revealing prompt weaknesses

## Challenges

### Challenge 1: Extend the Evaluation Pipeline

Add a fifth classification category ("Billing Questions") to the system. This requires changes across all three files.

**Success criteria:**
- Evaluation runs without errors with the new category
- At least 3 new test cases target the Billing Questions category
- Prompt template explicitly includes the new category in instructions
- Learner can explain why adding a category might reduce accuracy on existing categories and how to measure that

> **Deep-dive challenge:** See `CHALLENGE-deep-dive.md` in the module directory for an extended version that explores assertion types, custom scorers, and threshold-based pass criteria.

## Wrap-Up

**Key takeaways:**
- Promptfoo is an eval-focused framework: define inputs, expected outputs, and let it measure model performance as a function
- YAML config + Python prompts + CSV test cases form a portable, version-controllable evaluation pipeline
- Multi-provider testing in a single run enables direct model comparison without code changes

**What this does NOT cover:**
- Custom assertion functions (Python-based scoring beyond exact match)
- Red-teaming and adversarial test generation
- CI/CD integration for continuous evaluation
- Promptfoo's caching and cost optimization strategies

**Next steps:**
- Module 05-02: Advanced promptfoo assertions and custom scorers
- Module 05-03: Integrating evaluations into deployment pipelines
