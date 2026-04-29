# Module 04 Capstone Challenge: Unified Evaluation Pipeline

This capstone challenge brings together everything you've learned across Module 04. You will build an end-to-end evaluation pipeline for a document-processing application that combines multiple evaluation approaches into a single, unified report. This is your opportunity to demonstrate mastery of guardrails validation, RAG retrieval quality assessment, and workload-specific evaluation metrics.

## Challenge

Build an evaluation pipeline for a document-processing application that combines: guardrails validation, RAG retrieval quality, and one workload-specific metric of your choice. Produce a unified evaluation report.

## Sub-groups

Your custom metric (Stage 4) should draw from one of the following tracks:

- **(a) IDP + Automated Reasoning** — Evaluate document extraction accuracy, reasoning correctness, or structured output validation
- **(b) Guardrails (5 notebooks → 1 track)** — Extend guardrails evaluation with advanced policy composition, multi-turn validation, or custom guardrail logic
- **(c) RAG + Multimodal RAG + Speech** — Evaluate speech transcription quality, multimodal retrieval, or audio/image understanding

The capstone requires you to integrate guardrails + RAG + one choice from the other tracks.

## Prerequisites

Before attempting this challenge, complete the following skills:

- [SKILL-guardrails.md](./SKILL-guardrails.md)
- [SKILL-rag-evaluation.md](./SKILL-rag-evaluation.md)
- [SKILL-structured-data.md](./SKILL-structured-data.md)
- [SKILL-speech-reasoning.md](./SKILL-speech-reasoning.md)

## Starter Template

| Section | Pre-built (provided) | Learner fills in |
|---|---|---|
| **1. Setup & Configuration** | Imports (boto3, json, pandas, datetime), helper utilities, `load_document()`, `format_report()` | AWS client configuration, model selection |
| **2. Guardrails Evaluation Stage** | `GuardrailsEvaluator` class skeleton with `evaluate()` method signature, sample guardrail policy | `evaluate()` body — run input/output through guardrail, capture pass/fail/score |
| **3. RAG Retrieval Quality Stage** | `RAGEvaluator` class skeleton with `retrieve_and_score()` signature, sample document corpus (3-5 docs), sample queries | Retrieval call, faithfulness metric, relevance metric |
| **4. Custom Metric Stage** | `CustomMetricEvaluator` base class with `evaluate()` signature | Entire implementation — choose from IDP accuracy, speech quality, reasoning correctness, or define their own |
| **5. Pipeline Orchestration** | `run_pipeline()` skeleton that calls stages in sequence | Data passing between stages, error handling, aggregation |
| **6. Unified Report** | `ReportGenerator` class with `render()` method, Markdown/HTML template | Populate template with per-stage results, add summary + recommendations |

## Pipeline Architecture

```
Input Document(s)
       │
       ▼
┌─────────────────┐
│ Guardrails Stage │ → guardrail_results: {policy, pass/fail, violations[]}
└────────┬────────┘
         ▼
┌─────────────────┐
│ RAG Eval Stage   │ → rag_results: {queries[], faithfulness[], relevance[], latency}
└────────┬────────┘
         ▼
┌─────────────────┐
│ Custom Metric    │ → custom_results: {metric_name, scores[], methodology}
└────────┬────────┘
         ▼
┌─────────────────┐
│ Report Generator │ → unified_report: per-stage summary table, aggregate score,
└─────────────────┘   pass/fail determination, recommendations
```

## Assessment Criteria

1. Runs without errors — pipeline executes all stages end-to-end
2. Integrates 3+ evaluation types — guardrails + RAG + custom metric all produce results
3. Uses guardrails checks — correctly applies policy and captures violations
4. Handles multimodal input — pipeline doesn't break on non-text documents (or gracefully skips)
5. Learner explains which metrics matter most for their use case and why they chose their custom metric

## Tips & Hints

- **Start with the skeleton** — Get `run_pipeline()` calling empty stage methods first, then fill in each stage incrementally.
- **Use the SKILL notebooks as reference** — Your guardrails and RAG stages can reuse patterns directly from SKILL-guardrails.md and SKILL-rag-evaluation.md.
- **Pick a custom metric you care about** — The best evaluations measure what actually matters for your application. If you're processing invoices, measure extraction accuracy. If you're building a voice assistant, measure transcription fidelity.
- **Error handling matters** — A robust pipeline should not crash if one stage fails. Consider try/except blocks and partial results.
- **The report is your deliverable** — Spend time making it readable. A clear summary table with pass/fail per stage and an aggregate score tells the story at a glance.
- **Test with edge cases** — Try an empty document, a very long document, or a non-text file to verify your multimodal handling.
