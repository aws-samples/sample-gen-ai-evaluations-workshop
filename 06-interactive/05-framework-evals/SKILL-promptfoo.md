---
name: "Promptfoo Evaluation Framework"
description: "Use when learner needs to set up promptfoo YAML configs, define test cases with assertions, run eval CLI commands, and compare model outputs across providers using Amazon Bedrock"
---

# Promptfoo Evaluation Framework

Promptfoo is an **eval-focused framework** that treats the model as a function: you define inputs, send them through a prompt template to one or more providers, and compare the outputs against expected results. Unlike agent frameworks that orchestrate multi-step workflows, promptfoo focuses purely on the input→output evaluation loop. In this skill, you will build a complete evaluation pipeline—from YAML configuration through test execution—for an email classification system using Amazon Bedrock. By the end, you will be able to configure providers, write assertion-based test cases, run evaluations from the CLI, and interpret pass/fail results across multiple models.

## Prerequisites

- Completed Module 01 (evaluation fundamentals and terminology)
- Completed Module 02 (test case design patterns)
- AWS account with Amazon Bedrock model access (Nova Lite, Claude 3.5 Haiku)
- Node.js installed (for npm-based promptfoo installation)
- Familiarity with YAML syntax and CSV files

## Learning Objectives

- Configure a promptfoo YAML evaluation file specifying prompts, providers, and test cases
- Write test cases with expected-output assertions in CSV format
- Execute evaluations using the promptfoo CLI and interpret pass/fail results
- Compare model performance across multiple Bedrock providers in a single evaluation run

## Setup

Install promptfoo globally via npm:

```bash
npm install -g promptfoo --loglevel=error --no-fund
```

Verify the installation:

```bash
promptfoo --version
```

Create a working directory for this module:

```bash
mkdir -p ~/promptfoo-eval && cd ~/promptfoo-eval
```

Confirm your AWS credentials are configured and you have Bedrock model access in your target region (us-west-2 or us-east-1).

---

### Section 1: Prompt Templates as Python Functions

**Concept**

Before you can evaluate a model, you need a repeatable way to format inputs into prompts. Promptfoo supports prompt templates defined as Python functions, which gives you modularity—your prompt logic lives in code, separate from configuration. This separation means you can version-control prompts independently, reuse them across evaluations, and swap them without touching your test cases.

The function receives a variable (the input) and returns the fully-formatted prompt string. Promptfoo calls this function for each test case, injecting the test variable automatically.

**Build**

Create a file called `prompts.py` with the classification prompt:

```python
# prompts.py

def classify_email(email_content):
    return f"""You are an AI assistant for GlobalMart's customer support team. Your task is to classify the following email into one of these categories: Order Issues, Product Inquiries, Technical Support, or Returns/Refunds.

Email content: {email_content}

Provide your classification as a single word or phrase, choosing from the categories mentioned above. Do not include any explanation or additional text.

Classification:"""
```

This function takes `email_content` as input and returns a structured prompt that constrains the model to output only a category label.

---

### Section 2: Test Cases with Expected Outputs

**Concept**

Evaluations are only as good as your test data. Promptfoo uses a convention where test cases define input variables and expected outputs. When you store test cases in a CSV file, the column `__expected` (double underscore prefix) tells promptfoo what the correct output should be. At evaluation time, promptfoo compares the model's actual response against this expected value.

This is the core of the eval-focused paradigm: you define the ground truth, run the model, and measure how often reality matches expectation. The CSV format scales well—you can add hundreds of test cases without changing any code or configuration.

**Build**

Create a file called `dataset.csv` with labeled email examples:

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

Each row is one test case. The `email_content` column maps to the variable in your prompt function. The `__expected` column is the assertion target.

---

### Section 3: YAML Configuration — Bringing It Together

**Concept**

The `promptfooconfig.yaml` file is the central orchestration point. It declares four things: a description of the evaluation, which prompt functions to use, which model providers to test against, and where to find test cases. This single file defines your entire evaluation run.

The provider configuration is where promptfoo's multi-model comparison shines. You can list multiple Bedrock models and promptfoo will run every test case against every provider, producing a side-by-side comparison matrix. This lets you answer questions like "Does Nova Lite classify as accurately as Haiku 3.5?" in a single command.

**Build**

Create `promptfooconfig.yaml`:

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

Key structure:
- `prompts` — references the Python function using `filename:function_name` syntax
- `providers` — each entry specifies a Bedrock model ID, a human-readable label, and region config
- `tests` — points to the CSV file using the `file://` protocol

---

### Section 4: Running Evaluations and Interpreting Results

**Concept**

With configuration, prompts, and test cases in place, you execute the evaluation from the CLI. Promptfoo processes each test case through each provider, compares outputs to expected values, and reports pass/fail counts. The CLI flags control caching behavior and output formatting.

Understanding the results means looking at overall accuracy (what percentage passed), per-category performance (are some categories harder?), and per-provider differences (which model is more reliable?). This is where the eval-focused approach pays off—you get quantitative answers, not anecdotal impressions.

**Build**

Run the evaluation:

```bash
promptfoo eval --no-progress-bar --no-cache
```

Flag reference:
- `--no-progress-bar` — cleaner output in notebook/script environments
- `--no-cache` — forces fresh API calls (no cached responses from prior runs)

After the run completes, promptfoo prints a results table showing each test case, the model output, and whether it matched the expected value.

To share results or view them in a web UI:

```bash
promptfoo share
```

This generates a unique URL with an interactive results viewer showing the full comparison matrix.

Examine the output for:
1. **Overall accuracy** — what percentage of test cases passed per provider?
2. **Category patterns** — are certain categories consistently misclassified?
3. **Provider differences** — does one model outperform the other on specific categories?

---

## Challenges

### Challenge: Build a Multi-Category Product Evaluation

Design and run a promptfoo evaluation for a **new domain** (not email classification). Choose a text classification or extraction task relevant to your work, configure at least two Bedrock providers, and write a minimum of 8 test cases with expected outputs.

**Assessment criteria:**

1. Evaluation runs without errors using `promptfoo eval`
2. YAML config correctly references a custom prompt function and at least two providers
3. Test cases use the `__expected` column convention with at least 8 distinct inputs
4. Results demonstrate comparison across providers with pass/fail reporting
5. Learner can explain their choice of categories, why certain test cases were included, and what the results reveal about provider differences

---

## Wrap-Up

You have built a complete promptfoo evaluation pipeline: prompt templates as Python functions, CSV-based test cases with assertions, YAML configuration orchestrating multiple Bedrock providers, and CLI execution with results interpretation.

Key takeaways:
- Promptfoo treats evaluation as input→output comparison, making it ideal for classification, extraction, and formatting tasks
- The YAML config is the single source of truth for an evaluation run
- Multi-provider comparison reveals model differences without writing custom comparison code
- The `__expected` convention turns test cases into automated assertions

**Ready for more?** Take on the [Module 05 Deep-Dive Challenge](CHALLENGE-deep-dive.md) to push your framework evaluation skills further—combining promptfoo patterns with advanced assertion types and custom scoring.

**Feedback:** What worked well in this skill? What was confusing? Share with your instructor or post in the workshop channel.

**Next steps:** Explore the other framework evaluations in Module 05 to see how agent-focused and observability-focused tools compare to promptfoo's eval-focused approach.
