# Interactive Learning Mode

This directory contains AI-tutored interactive lessons for the AWS Evaluations Workshop. Each SKILL file is a self-contained module that an AI coding assistant can use to teach you hands-on evaluation techniques.

## How to Use

1. Set up your AI coding tool (Claude Code, Kiro, or Codex) — see `agents.md` for configuration
2. Pick a skill from the map below
3. Tell your AI assistant: "Teach me about [topic]" — it will match your request to the right SKILL

## Skill Map

### Foundational Evaluations
| Skill | What You'll Build |
|-------|-------------------|
| [SKILL-operational](foundational%20evaluations/SKILL-operational.md) | CloudWatch metrics, dashboards, and alarms for LLM monitoring |
| [SKILL-quality](foundational%20evaluations/SKILL-quality.md) | LLM-as-Judge and Jury evaluation with agreement scoring |
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
