# Automated Reasoning Evaluations

This module demonstrates how to evaluate **Automated Reasoning (AR) Checks** in Amazon Bedrock Guardrails. AR Checks verify LLM outputs against formal policy rules using a two-step pipeline: LLM-based translation (natural language to logical formulas) followed by SMT-based verification.

## What You'll Learn

- How AR Checks work: the 7 validation result types and what they mean
- Building AR policies from a real regulatory document (SF Housing Code)
- Running claims through an AR guardrail and interpreting results
- Evaluating AR policy quality with metrics (accuracy, F1, false valid rate, translation confidence)
- Visualizing evaluation results (confusion matrix, radar chart, per-type performance)

## Notebooks

| Notebook | Description | Time |
|----------|-------------|------|
| `04-06-01-automated-reasoning-evaluation.ipynb` | **Hub notebook** — create AR policies, run a live demo, and evaluate with 34 essential test cases | ~20 min |
| `04-06-02` (planned) | Document preprocessing pipeline — chunking, filtering, anti-hallucination prompt design | ~30 min |
| `04-06-03` (planned) | Comprehensive evaluation — full 83-case test suite, 6 metric families, 8 visualizations | ~30 min |
| `04-06-04` (planned) | Multi-guardrail architecture and fidelity optimization via MCP | ~45 min |

## Key Concepts

### 7 Validation result Types

**Definitive answers:**
- `VALID` — claim satisfies all translated policy rules
- `INVALID` — claim violates at least one policy rule
- `IMPOSSIBLE` — contradictory premises detected

**Partial answers:**
- `SATISFIABLE` — some interpretations valid, some not
- `TRANSLATION_AMBIGUOUS` — multiple interpretations of the input

**No answer:**
- `NO_TRANSLATIONS` — no policy variables matched the input
- `TOO_COMPLEX` — SMT solver timed out

### Evaluation Dimensions

1. **Validation Correctness** — confusion matrix, per-type P/R/F1, false valid rate
2. **Translation Quality** — confidence scores, premise/claim translation rates
3. **Domain Coverage** — rule and section coverage across the policy
4. **Adversarial Robustness** — paraphrase consistency, negation flip rate
5. **Operational Metrics** — latency distribution (P50/P95/P99)
6. **Per-Category Accuracy** — breakdowns by test category, difficulty, and domain

## Data Files

| File | Description |
|------|-------------|
| `data/ar_tests_essential.json` | 34 essential test cases for the hub notebook |
| `data/ar_tests.json` | Full 83-case test suite (used by spoke notebook 04-06-03) |
| `data/housing_code_structured_rules.md` | Pre-processed rules extracted from the housing code PDF |
| `housing-code.pdf` | Source document: San Francisco Housing Code |

## Prerequisites

- AWS account with Amazon Bedrock access (us-west-2)
- Python 3.10+
- Install dependencies: `uv pip install -r requirements.txt` (or `pip install -r requirements.txt`)

## Getting Started

1. Open `04-06-01-automated-reasoning-evaluation.ipynb`
2. Choose your path:
   - **Path A** — Use an existing guardrail (fastest, ~5 min)
   - **Path B** — Bring your own AR policy ARNs (~2 min setup)
   - **Path C** — Build everything from scratch (~10-15 min)
3. Run all cells to see a live demo and evaluation results
