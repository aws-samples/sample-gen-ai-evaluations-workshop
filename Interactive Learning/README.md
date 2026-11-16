# Interactive Learning Mode

This directory contains AI-tutored interactive lessons for the AWS Evaluations
Workshop. Each SKILL file is a lesson plan that an AI coding assistant uses to
teach through questions, small implementation steps, and applied challenges.

## How to Use

1. Configure your AI coding tool to read [AGENTS.md](AGENTS.md).
2. Pick a skill from the map below.
3. Tell the assistant, "Teach me about [topic]."
4. Work through one question at a time; the assistant checks your reasoning
   before advancing.

## Two modes: Learn vs. Build

- **Learn mode (tutor)** — the SKILLs below teach evaluation techniques through hands-on, Socratic challenges. Say _"Teach me about [topic]."_
- **Build mode (`eval-builder`)** — a separate skill that **builds real evals for your own workload** using a failure-first approach: it discovers what's actually failing, maps failures to the right workshop modules, and scaffolds binary pass/fail judges. It also has a **Review mode** that grades your existing eval pipeline against what the workshop teaches.

## Build mode: `eval-builder`

[`eval-builder`](../plugins/eval-builder/) is a plugin for Kiro, Claude Code, and Codex. It runs in **your own** workload repo (not this one), reads the workshop's live module content as its source of truth, and produces an interactive HTML report.

**Install:**

- **Claude Code**
  ```bash
  claude plugin marketplace add aws-samples/sample-gen-ai-evaluations-workshop
  claude plugin install eval-builder@genai-evals-workshop
  ```
- **Codex**
  ```bash
  codex plugin marketplace add aws-samples/sample-gen-ai-evaluations-workshop
  codex plugin add eval-builder@genai-evals-workshop
  ```
- **Kiro** — clone this repo, then either run `kiro-cli powers install <clone>/plugins/eval-builder`, or in the Kiro IDE open **Powers → Add Custom Power → Import power from a folder** and select `plugins/eval-builder`.

Start a new session after installing. The plugin needs AWS credentials with Amazon Bedrock access in the account and Region of the workload you evaluate.

**Notes by tool:**

- **Kiro** asks once for permission to load the Power the first time you use it. Approve it. In headless runs (`kiro-cli chat --no-interactive`), add `--trust-tools=kiro_powers` or the Power can't load and Kiro answers without it.
- **Codex:** run the skill in an interactive session. In `codex exec`, Codex tends to work through the whole flow without stopping at the approval gates.
- **Any tool:** set `EVAL_WORKSHOP_DIR` to a local clone of this workshop to skip the clone the skill otherwise makes in `/tmp`.

**Use it:**

1. `cd` into the repository containing the workload you want to evaluate.
2. Invoke it (e.g. `/eval-builder`, or _"build evals for my app"_, or _"review my evals"_).
3. The skill follows a failure-first, gated workflow:
   - **Discover** — scans your repo, detects workload type(s), finds existing evals → confirms scope with you.
   - **Failure-first** — samples real outputs, categorizes what's failing, suggests prompt fixes before building infrastructure, maps failures to workshop modules → you answer scoping questions.
   - **Plan & Build** — proposes layered evals (foundational → workload-specific → framework) → **you approve** → scaffolds binary pass/fail judges into `evals/`.
   - **Run** — **you confirm** (Bedrock spend) → runs evals → writes `evals/report.html`.
4. If existing evals are found, it offers **Review mode**: grades them against workshop patterns, shows a coverage matrix, and recommends fixes + new evals.

## Skill Map

### Foundational Evaluations

| Skill | What You'll Build |
|---|---|
| [SKILL-operational](foundational%20evaluations/SKILL-operational.md) | CloudWatch metrics, dashboards, and alarms for LLM monitoring |
| [SKILL-quality](foundational%20evaluations/SKILL-quality.md) | Binary LLM-as-a-Judge checks and judge calibration |
| [SKILL-understanding-failures](foundational%20evaluations/SKILL-understanding-failures.md) | Trace review, failure-pattern discovery, and evaluator criteria |
| [SKILL-agentic](foundational%20evaluations/SKILL-agentic.md) | Agent trace evaluation, tool-selection metrics, and metric reuse |

### Workload-Specific Evaluations

| Skill | What You'll Build |
|---|---|
| [SKILL-structured-data](workload%20evals/SKILL-structured-data.md) | Document extraction accuracy scoring |
| [SKILL-guardrails](workload%20evals/SKILL-guardrails.md) | Bedrock Guardrails filters, grounding, alignment, and evaluation |
| [SKILL-rag-evaluation](workload%20evals/SKILL-rag-evaluation.md) | Retrieval and generation evaluation across text and multimodal RAG |
| [SKILL-speech-reasoning](workload%20evals/SKILL-speech-reasoning.md) | Speech-to-speech and automated reasoning verification |
| [SKILL-tool-calling](workload%20evals/SKILL-tool-calling.md) | Tool selection, parameter, trajectory, and simulation evaluation |
| [SKILL-chatbot](workload%20evals/SKILL-chatbot.md) | Multi-turn simulation, binary evaluators, and synthetic test data |
| [SKILL-red-teaming](workload%20evals/SKILL-red-teaming.md) | Adversarial testing for LLM apps, guardrails, RAG, and agents |
| [SKILL-multiagent-context](workload%20evals/SKILL-multiagent-context.md) | Shared-context quality across hub-spoke and peer-to-peer agents |
| [CHALLENGE-capstone](workload%20evals/CHALLENGE-capstone.md) | **Capstone:** integrate several workload evaluation techniques |

### Framework-Specific Evaluations

| Skill | What You'll Build |
|---|---|
| [SKILL-promptfoo](framework%20evals/SKILL-promptfoo.md) | Promptfoo YAML configuration, assertions, and model comparisons |
| [SKILL-agentcore](framework%20evals/SKILL-agentcore.md) | Run, observe, evaluate, and improve one AgentCore agent through built-ins, ground truth, custom evaluators, monitoring, and simulation |
| [SKILL-strands](framework%20evals/SKILL-strands.md) | Strands Evals cases, experiments, trajectory checks, and custom evaluators |
| [SKILL-dspy](framework%20evals/SKILL-dspy.md) | DSPy signatures, metrics, and prompt optimization loops |
| [SKILL-mlflow](framework%20evals/SKILL-mlflow.md) | MLflow experiment tracking, scorer composition, and run comparison |
| [CHALLENGE-deep-dive](framework%20evals/CHALLENGE-deep-dive.md) | **Deep-dive:** extend one framework beyond its guided module |

## Prerequisites

- AWS account with Amazon Bedrock model access
- Python 3.10 or later
- AWS credentials configured
- An AI coding assistant such as Claude Code, Kiro, or Codex

Individual skills list their additional dependencies. AgentCore requires
Node.js 20 or later, `uv`, and the current `@aws/agentcore` CLI.

## Dependencies

Complete the foundational skills in order. Workload and framework skills can
then be selected independently. See [curriculum.md](curriculum.md) for the full
learning path and the six-notebook AgentCore progression.

## Contributing a Skill

Use [meta/SKILL-BUILDER.md](meta/SKILL-BUILDER.md) to generate or revise a
lesson from source notebooks. Validate changes with:

```bash
bash meta/validate_skills.sh path/to/SKILL-name.md
```
