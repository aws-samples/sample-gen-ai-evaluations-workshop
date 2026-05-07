# Module 04 Capstone: Unified Evaluation Pipeline

## Overview

This capstone integrates all Module 04 skills into a single evaluation pipeline for a document-processing application. You will combine guardrails validation, RAG retrieval quality assessment, and one workload-specific metric of your choice into a system that produces a unified evaluation report.

**Challenge:** Build an evaluation pipeline for a document-processing application that combines: guardrails validation, RAG retrieval quality, and one workload-specific metric of your choice. Produce a unified evaluation report.

## Sub-Groups

Your custom metric (Stage 3) should draw from one of these tracks:

- **(a) IDP + Automated Reasoning** — field-level extraction accuracy (TP/TN/FA/FD/FN classification, precision/recall/F1) or AR policy verification (false valid rate, consistency accuracy)
- **(b) Guardrails** — multi-layer guardrail precision/recall using a confusion matrix harness (content filters + grounding + alignment)
- **(c) RAG + Multimodal RAG + Speech** — multimodal retrieval comparison (text-only vs vision vs combined), MAP/MRR diagnostics, or LLM-as-Judge speech quality scoring

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

## Starter Template

| Section | Pre-built (provided) | Learner fills in |
|---|---|---|
| **1. Setup & Configuration** | Imports (boto3, json, pandas, datetime), helper utilities, `load_document()`, `format_report()` | AWS client configuration, model selection |
| **2. Guardrails Evaluation Stage** | `GuardrailsEvaluator` class skeleton with `evaluate()` method signature, sample guardrail policy | `evaluate()` body — run input/output through guardrail, capture pass/fail/score |
| **3. RAG Retrieval Quality Stage** | `RAGEvaluator` class skeleton with `retrieve_and_score()` signature, sample document corpus (3-5 docs), sample queries | Retrieval call, faithfulness metric, relevance metric |
| **4. Custom Metric Stage** | `CustomMetricEvaluator` base class with `evaluate()` signature | Entire implementation — choose from IDP accuracy, speech quality, reasoning correctness, or define their own |
| **5. Pipeline Orchestration** | `run_pipeline()` skeleton that calls stages in sequence | Data passing between stages, error handling, aggregation |
| **6. Unified Report** | `ReportGenerator` class with `render()` method, Markdown/HTML template | Populate template with per-stage results, add summary + recommendations |

## Provided Scaffolding

```python
import boto3
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional

# --- Helper Utilities (provided) ---

def load_document(path: str) -> Dict[str, Any]:
    """Load a document for evaluation. Handles text and JSON formats."""
    with open(path) as f:
        content = f.read()
    return {
        "path": path,
        "content": content,
        "loaded_at": datetime.now().isoformat(),
        "format": "json" if path.endswith(".json") else "text"
    }

def format_report(results: Dict[str, Any], template: str = "markdown") -> str:
    """Format evaluation results into a report. Supports 'markdown' or 'html'."""
    if template == "markdown":
        lines = ["# Evaluation Report", f"**Generated:** {datetime.now().isoformat()}", ""]
        for stage, data in results.items():
            lines.append(f"## {stage}")
            lines.append(f"- Status: {data.get('status', 'unknown')}")
            lines.append(f"- Score: {data.get('score', 'N/A')}")
            lines.append("")
        return "\n".join(lines)
    return json.dumps(results, indent=2)


# --- Stage 1: Setup & Configuration ---
# TODO: Configure your AWS clients and select models
# Hint: You need bedrock_client and bedrock_runtime (see SKILL-guardrails.md Setup)

bedrock_client = None   # YOUR CODE: boto3.client(...)
bedrock_runtime = None  # YOUR CODE: boto3.client(...)
model_id = None         # YOUR CODE: select your evaluation model


# --- Stage 2: Guardrails Evaluation ---

SAMPLE_GUARDRAIL_POLICY = {
    "content_filters": ["SEXUAL", "VIOLENCE", "HATE", "MISCONDUCT", "PROMPT_ATTACK"],
    "denied_topics": ["financial_advice", "medical_diagnosis"],
    "grounding_threshold": 0.7
}

class GuardrailsEvaluator:
    """Evaluates document content against guardrail policies.

    Reference: SKILL-guardrails.md Section 1 (apply_guardrail pattern)
    and Section 5 (confusion matrix harness).
    """

    def __init__(self, guardrail_id: str, guardrail_version: str, client=None):
        self.guardrail_id = guardrail_id
        self.guardrail_version = guardrail_version
        self.client = client or bedrock_runtime

    def evaluate(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Run document content through guardrail, return pass/fail and violations.

        TODO: Implement using bedrock_runtime.apply_guardrail()
        Expected return: {
            "policy": SAMPLE_GUARDRAIL_POLICY,
            "pass": True/False,
            "violations": [...],
            "action": "NONE" or "GUARDRAIL_INTERVENED",
            "score": float  # 1.0 if pass, 0.0 if fail
        }
        """
        raise NotImplementedError("Implement guardrail evaluation")


# --- Stage 3: RAG Retrieval Quality ---

SAMPLE_CORPUS = [
    {"id": "doc1", "text": "Amazon Bedrock provides access to foundation models from leading AI companies."},
    {"id": "doc2", "text": "Guardrails help prevent harmful content and hallucinations in AI applications."},
    {"id": "doc3", "text": "RAG systems retrieve relevant documents to ground LLM responses in facts."},
    {"id": "doc4", "text": "Evaluation metrics like precision and recall measure retrieval effectiveness."},
    {"id": "doc5", "text": "Multimodal models can process text, images, and audio inputs together."},
]

SAMPLE_QUERIES = [
    {"query": "How do guardrails prevent hallucinations?", "relevant_ids": ["doc2", "doc3"]},
    {"query": "What metrics measure RAG quality?", "relevant_ids": ["doc4", "doc3"]},
    {"query": "Can Bedrock handle images?", "relevant_ids": ["doc1", "doc5"]},
]

class RAGEvaluator:
    """Evaluates RAG retrieval quality using IR metrics and faithfulness scoring.

    Reference: SKILL-rag-evaluation.md Section 1 (precision/recall/NDCG)
    and Section 2 (LLM-as-Judge faithfulness rubric).
    """

    def __init__(self, corpus: List[Dict], client=None):
        self.corpus = corpus
        self.client = client or bedrock_runtime

    def retrieve_and_score(self, queries: List[Dict], k: int = 3) -> Dict[str, Any]:
        """Retrieve documents for each query and compute quality metrics.

        TODO: Implement retrieval + scoring
        Expected return: {
            "queries": [...],
            "faithfulness": [float, ...],  # per-query faithfulness scores
            "relevance": [float, ...],     # per-query relevance scores (precision@k)
            "latency": float,              # total retrieval time in seconds
            "aggregate": {"map": float, "mrr": float}
        }
        """
        raise NotImplementedError("Implement RAG retrieval and scoring")


# --- Stage 4: Custom Metric ---

class CustomMetricEvaluator:
    """Base class for your custom evaluation metric.

    Choose one:
    - IDP accuracy: field-level TP/FD/FN classification (SKILL-structured-data.md Section 1-2)
    - Speech quality: LLM-as-Judge over conversation traces (SKILL-speech-reasoning.md Section 2)
    - Reasoning correctness: AR policy verification (SKILL-speech-reasoning.md Section 3-4)
    - Your own metric: define methodology and scoring

    Reference the relevant SKILL for implementation patterns.
    """

    def __init__(self, metric_name: str):
        self.metric_name = metric_name

    def evaluate(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Run your custom evaluation metric.

        TODO: Entire implementation is yours
        Expected return: {
            "metric_name": self.metric_name,
            "scores": [float, ...],
            "methodology": "description of how you computed this",
            "details": {...}  # metric-specific breakdown
        }
        """
        raise NotImplementedError("Implement your custom metric")


# --- Stage 5: Pipeline Orchestration ---

def run_pipeline(documents: List[str], config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Execute all evaluation stages in sequence.

    TODO: Implement data passing between stages, error handling, aggregation.
    Skeleton calls stages in order — you wire them together.

    Expected flow:
    1. Load documents
    2. Run GuardrailsEvaluator.evaluate() on each
    3. Run RAGEvaluator.retrieve_and_score() with relevant queries
    4. Run CustomMetricEvaluator.evaluate() on each
    5. Aggregate results
    6. Generate report
    """
    results = {}

    # Stage 1: Guardrails
    # YOUR CODE: instantiate evaluator, run on documents, collect results

    # Stage 2: RAG
    # YOUR CODE: instantiate evaluator, run queries, collect results

    # Stage 3: Custom Metric
    # YOUR CODE: instantiate your chosen evaluator, run, collect results

    # Aggregate
    # YOUR CODE: combine stage results, compute overall pass/fail

    return results


# --- Stage 6: Unified Report ---

REPORT_TEMPLATE = """# Unified Evaluation Report
**Generated:** {timestamp}
**Documents Evaluated:** {doc_count}

## Summary
| Stage | Status | Score | Key Finding |
|-------|--------|-------|-------------|
| Guardrails | {guardrails_status} | {guardrails_score} | {guardrails_finding} |
| RAG Quality | {rag_status} | {rag_score} | {rag_finding} |
| {custom_name} | {custom_status} | {custom_score} | {custom_finding} |

## Aggregate
- **Overall Score:** {overall_score}
- **Pass/Fail:** {pass_fail}

## Per-Stage Details

### Guardrails
{guardrails_details}

### RAG Retrieval Quality
{rag_details}

### {custom_name}
{custom_details}

## Recommendations
{recommendations}
"""

class ReportGenerator:
    """Generates unified evaluation reports from pipeline results."""

    def __init__(self, template: str = REPORT_TEMPLATE):
        self.template = template

    def render(self, pipeline_results: Dict[str, Any]) -> str:
        """Populate template with per-stage results, summary, and recommendations.

        TODO: Extract results from each stage, format into template,
        add summary scoring and actionable recommendations.
        """
        raise NotImplementedError("Implement report rendering")


# --- Entry Point ---

if __name__ == "__main__":
    # Example usage
    test_documents = ["data/sample_doc_1.json", "data/sample_doc_2.json"]
    results = run_pipeline(test_documents)
    report = ReportGenerator().render(results)
    print(report)
```

## Assessment Criteria

1. **Runs without errors** — pipeline executes all stages end-to-end
2. **Integrates 3+ evaluation types** — guardrails + RAG + custom metric all produce results
3. **Uses guardrails checks** — correctly applies policy and captures violations using `apply_guardrail()` (see SKILL-guardrails.md Section 1 and 5)
4. **Handles multimodal input** — pipeline doesn't break on non-text documents (or gracefully skips with a logged reason)
5. **Learner explains** which metrics matter most for their use case and why they chose their custom metric

## Hints

- **Guardrails stage:** Reuse the `apply_guardrail()` pattern from SKILL-guardrails.md Section 1. For scoring, adapt the confusion matrix approach from Section 5.
- **RAG stage:** The `IRMetricsCalculator` class from SKILL-rag-evaluation.md Section 1 gives you precision/recall/NDCG. The faithfulness rubric from Section 2 scores generation quality.
- **Custom metric options:**
  - IDP accuracy → `compare_and_classify()` from SKILL-structured-data.md Section 1, then `calculate_metrics()` from Section 2
  - AR verification → `run_ar_test()` + `compute_core_metrics()` from SKILL-speech-reasoning.md Section 4
  - Speech quality → `S2SEvaluator` pattern from SKILL-speech-reasoning.md Section 2
- **Error handling:** Wrap each stage in try/except so one failure doesn't kill the pipeline. Record the error in results and continue.
- **Multimodal:** Check `document["format"]` before processing. If a stage can't handle the format, return `{"status": "skipped", "reason": "unsupported format"}`.

## Submission

When complete, your pipeline should:
1. Accept 2+ documents as input
2. Produce a rendered Markdown report with all three stages populated
3. Include a brief written explanation (2-3 paragraphs) of your metric choices and what the results tell you about the document-processing system's quality
