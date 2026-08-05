# Interactive Learning Mode

This directory contains AI-tutored interactive lessons for the AWS Evaluations Workshop. Each SKILL file is a self-contained module that an AI coding assistant can use to teach you hands-on evaluation techniques.

## How to Use

1. Set up your AI coding tool (Claude Code, Kiro, or Codex) — see `agents.md` for configuration
2. Pick a skill from the map below
3. Tell your AI assistant: "Teach me about [topic]" — it will match your request to the right SKILL

## Two modes: Learn vs. Build

- **Learn mode (tutor)** — the SKILLs below teach evaluation techniques through hands-on, Socratic challenges. Say _"Teach me about [topic]."_
- **Build mode (`eval-builder`)** — a separate skill that **builds real evals for your own workload** using a failure-first approach: it discovers what's actually failing, maps failures to the right workshop modules, and scaffolds binary pass/fail judges. It also has a **Review mode** that grades your existing eval pipeline against what the workshop teaches.

## Build mode: `eval-builder`

[`eval-builder/`](eval-builder/) is a portable skill for Claude Code and Kiro. It runs in **your own** workload repo (not this one), reads the workshop's live module content as its source of truth, and produces an interactive HTML report.

**Install (agent-driven — no manual file moving, no script):**

1. Clone this workshop repo and open it with your AI assistant (Kiro or Claude Code).
2. Say: **"install the eval-builder skill."**
3. The assistant copies `Interactive Learning/eval-builder/` into your global skills directory:
   - Kiro → `~/.kiro/skills/eval-builder/`
   - Claude Code → `~/.claude/skills/eval-builder/`

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
|-------|-------------------|
| [SKILL-operational](foundational%20evaluations/SKILL-operational.md) | CloudWatch metrics, dashboards, and alarms for LLM monitoring |
| [SKILL-quality](foundational%20evaluations/SKILL-quality.md) | LLM-as-Judge evaluation with binary pass/fail verdicts and judge calibration |
| [SKILL-agentic](foundational%20evaluations/SKILL-agentic.md) | Agent trace evaluation, tool selection metrics, metric reuse |

### Workload-Specific Evaluations
| Skill | What You'll Build |
|-------|-------------------|
| [SKILL-structured-data](workload%20evals/SKILL-structured-data.md) | Document extraction accuracy scoring |
| [SKILL-guardrails](workload%20evals/SKILL-guardrails.md) | Bedrock Guardrails: filters, grounding, alignment, evaluation |
| [SKILL-rag-evaluation](workload%20evals/SKILL-rag-evaluation.md) | RAG retrieval quality + multimodal evaluation |
| [SKILL-speech-reasoning](workload%20evals/SKILL-speech-reasoning.md) | Speech-to-speech + automated reasoning verification |
| [CHALLENGE-capstone](workload%20evals/CHALLENGE-capstone.md) | **Capstone:** Integrate guardrails + RAG + custom metric |

### Framework-Specific Evaluations
| Skill | What You'll Build |
|-------|-------------------|
| [SKILL-promptfoo](framework%20evals/SKILL-promptfoo.md) | PromptFoo CLI: YAML configs, assertions, eval runs |
| [SKILL-agentcore](framework%20evals/SKILL-agentcore.md) | AgentCore: deploy, invoke, evaluate with native API |
| [SKILL-strands](framework%20evals/SKILL-strands.md) | Strands Evals: Cases, Experiments, custom evaluators |
| [SKILL-dspy](framework%20evals/SKILL-dspy.md) | DSPy: signatures, metrics, optimization loops |
| [CHALLENGE-deep-dive](framework%20evals/CHALLENGE-deep-dive.md) | **Deep-dive:** Pick one framework, go beyond the notebook |

## Prerequisites

- AWS account with Bedrock model access (Claude, Titan)
- Python 3.11+ with boto3
- An AI coding assistant (Claude Code, Kiro, or Codex)

## Dependencies

Foundational skills (01-03) have no prerequisites. Workload and Framework skills recommend completing at least one foundational skill first. See [curriculum.md](curriculum.md) for the full dependency map.

## Contributing a New Skill

Use the skill-doc-builder in `.kiro/skills/skill-doc-builder/SKILL.md` to generate new skills from notebooks. Run `scripts/validate_skills.sh` to verify structural compliance.
