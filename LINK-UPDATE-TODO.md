# Phase 3: Link Update To-Do List

Files to review for old path references. Check off each file once reviewed and updated.

## Files with confirmed old path references (from grep)

- [x] `README.md` (root) — updated links to `Interactive Learning/`
- [x] `CONTRIBUTING.md` — prose only ("Module 02", "Module 03" etc.), no file paths to update
- [x] `Interactive Learning/README.md` — updated links to new skill file locations
- [x] `Interactive Learning/claude/kiro.md` — updated relative paths to new structure
- [x] `Interactive Learning/foundational evaluations/SKILL-agentic.md` — updated `cd` command and asset table paths
- [x] `Interactive Learning/foundational evaluations/SKILL-operational.md` — prose only, no paths to update
- [x] `Interactive Learning/foundational evaluations/SKILL-quality.md` — prose only, no paths to update
- [x] `Interactive Learning/workload evals/SKILL-speech-reasoning.md` — prose only, no paths to update
- [x] `Interactive Learning/workload evals/SKILL-structured-data.md` — prose only, no paths to update
- [x] `Interactive Learning/framework evals/SKILL-promptfoo.md` — prose only, no paths to update
- [x] `Interactive Learning/framework evals/SKILL-strands.md` — prose only (references Module 03 conceptually), no paths to update
- [x] `Interactive Learning/framework evals/SKILL-agentcore.md` — prose only, no paths to update
- [x] `Interactive Learning/framework evals/SKILL-dspy.md` — prose only, no paths to update
- [x] `Interactive Learning/workload evals/SKILL-guardrails.md` — prose only, no paths to update
- [x] `Interactive Learning/workload evals/SKILL-rag-evaluation.md` — prose only, no paths to update

## Remaining files reviewed (no changes needed unless noted)

- [x] `CODE_OF_CONDUCT.md` — no repo references
- [x] `Foundational Evaluations/01-operational-metrics/README.md` — local references only, still valid
- [x] `Foundational Evaluations/02-quality-metrics/README.md` — no cross-module paths
- [x] `Foundational Evaluations/03-understanding-failures/README.md` — **UPDATED** cross-module links to use new paths
- [x] `Foundational Evaluations/04-agentic-metrics/README.md` — **UPDATED** notebook filename reference
- [x] `Foundational Evaluations/04-agentic-metrics/todo.md` — no repo references
- [x] `Framework Specific Evaluations/AgentCore/README.md` — **UPDATED** git hook path reference
- [x] `Framework Specific Evaluations/AgentCore Runtime Evals/README.md` — no cross-module paths
- [x] `Framework Specific Evaluations/DSPy/README.md` — no cross-module paths
- [x] `Framework Specific Evaluations/Mlflow/README.md` — no cross-module paths
- [x] `Framework Specific Evaluations/Prompt Foo/README.md` — no cross-module paths
- [x] `Framework Specific Evaluations/Strands/README.md` — no cross-module paths
- [x] `Workload Specific Evaluations/Automated Reasoning Evaluations/README.md` — no cross-module paths
- [x] `Workload Specific Evaluations/Automated Reasoning Evaluations/data/housing_code_structured_rules.md` — content doc, no paths
- [x] `Workload Specific Evaluations/Basic RAG/README.md` — no cross-module paths
- [x] `Workload Specific Evaluations/Chatbot/README.md` — no cross-module paths
- [x] `Workload Specific Evaluations/Guardrails/README.md` — no cross-module paths
- [x] `Workload Specific Evaluations/Guardrails/05 reasoning.md` — content doc, no paths
- [x] `Workload Specific Evaluations/Intelligent Document Processing/readme.md` — no cross-module paths
- [x] `Workload Specific Evaluations/MultiModal RAG/readme.md` — no cross-module paths
- [x] `Workload Specific Evaluations/Multiagent Shared Context Evaluation/with-strands-agent/metrics/README.md` — no cross-module paths
- [x] `Workload Specific Evaluations/Red Teaming/README.md` — no cross-module paths
- [x] `Workload Specific Evaluations/Red Teaming/01 LLM App Red Teaming/README.md` — no cross-module paths
- [x] `Workload Specific Evaluations/Red Teaming/02 Testing Bedrock Guardrails/README.md` — no cross-module paths
- [x] `Workload Specific Evaluations/Red Teaming/03 RAG Red Teaming/README.md` — no cross-module paths
- [x] `Workload Specific Evaluations/Red Teaming/04 Agent Red Teaming/README.md` — no cross-module paths
- [x] `Workload Specific Evaluations/Speech to Speech/README.md` — no cross-module paths
- [x] `Workload Specific Evaluations/Speech to Speech/sample_s2s_app/README.md` — no cross-module paths
- [x] `Workload Specific Evaluations/Speech to Speech/test-data/CONFIG_TEMPLATE.md` — no cross-module paths
- [x] `Workload Specific Evaluations/Speech to Speech/test-data/interview_practice_assistant/README.md` — no cross-module paths
- [x] `Workload Specific Evaluations/Speech to Speech/test-data/order_assistant/README.md` — no cross-module paths
- [x] `Workload Specific Evaluations/Speech to Speech/test-data/survey_assistant/README.md` — no cross-module paths
- [x] `Workload Specific Evaluations/Speech to Speech/test-evaluation-results/sample/evaluation_report.md` — no cross-module paths
- [x] `Workload Specific Evaluations/Speech to Speech/test/e2e/README.md` — local relative path (`../../sample_s2s_app/README.md`), still valid
- [x] `Workload Specific Evaluations/Tool Calling/README.md` — no cross-module paths

## Notebooks reviewed

- [x] `Foundational Evaluations/03-understanding-failures/01_Discovering_Failure_Patterns.ipynb` — **UPDATED** cross-module links
- [x] `Framework Specific Evaluations/AgentCore/01 Agentic Metrics AgentCore.ipynb` — old paths in output cells only (execution artifacts), not actionable
- [x] `Framework Specific Evaluations/AgentCore Runtime Evals/02 AgentCore Runtime on demand Evals.ipynb` — old paths in output cells only (execution artifacts), not actionable
- [x] All other notebooks — searched via grep, no old path references found in source cells

## Notes

- `RESTRUCTURE-LOG.md` intentionally contains old paths as historical reference — NOT updated.
- "Module 01", "Module 02" etc. in prose (not as file paths) left as-is — they're conceptual references.
- Notebook output cells containing absolute paths from prior execution runs are left as-is — these are artifacts that will be overwritten on next run.
- The AgentCore notebooks reference `03-agentic-metrics` in prose markdown cells describing what the notebook does — these are descriptive text about the original module, not navigational links.
