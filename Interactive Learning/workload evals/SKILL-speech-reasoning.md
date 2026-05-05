---
name: Speech & Automated Reasoning Evaluations
description: Build evaluation pipelines for speech-to-speech applications and automated reasoning policies using CloudWatch traces, LLM-as-Judge, and SMT solver verification
---

In this skill you build two distinct evaluation pipelines that address specialized AI workloads. First, you construct an end-to-end speech-to-speech evaluation system that extracts telemetry from CloudWatch, maps sessions to validation data, and runs LLM-as-Judge scoring. Second, you implement automated reasoning (AR) evaluations that verify LLM outputs against formal policy rules using Bedrock Guardrails. By the end, you will understand how to measure quality for both conversational speech systems and logic-verified compliance systems.

## Prerequisites

- Completion of Module 01 (evaluation fundamentals) and Module 02 (LLM-as-Judge patterns)
- Familiarity with Amazon Bedrock, CloudWatch Logs, and boto3
- Python environment with `boto3`, `pandas`, `numpy`, `matplotlib`, `seaborn` installed
- AWS credentials with access to Bedrock and CloudWatch

## Learning Objectives

- Extract and process speech-to-speech telemetry traces from CloudWatch into structured evaluation datasets
- Design and execute LLM-as-Judge evaluations that compare session transcripts against validation categories
- Implement automated reasoning policy evaluations that measure translation fidelity, consistency accuracy, and false valid rate
- Interpret confusion matrices and per-type precision/recall/F1 metrics to diagnose AR policy quality

## Setup

Ensure your Python environment has the required dependencies:

```bash
pip install boto3 pandas numpy matplotlib seaborn python-dotenv
```

Configure AWS credentials. The notebooks expect a `.env` file or environment variables:

```python
import os
import boto3
from dotenv import load_dotenv

load_dotenv(".env")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
session = boto3.Session(region_name=AWS_REGION)
bedrock_client = session.client("bedrock")
bedrock_runtime_client = session.client("bedrock-runtime")
cloudwatch_logs_client = session.client("logs")
```

Verify Bedrock access:

```python
try:
    bedrock_client.list_guardrails(maxResults=1)
    print("Bedrock access verified")
except Exception as e:
    print(f"ERROR: Cannot access Bedrock. Check credentials.\n{e}")
```

---

### Section 1: Speech Evaluation Concepts

Speech-to-speech (S2S) systems present unique evaluation challenges because they operate across multiple modalities — audio input, language understanding, response generation, and speech synthesis. Traditional text-based metrics cannot capture the full quality picture. You need an evaluation pipeline that captures the entire conversation flow, maps it to expected behaviors, and scores it holistically.

The pipeline follows five stages: (1) run the S2S application to generate conversations, (2) capture telemetry traces in CloudWatch, (3) extract and structure those traces into an evaluation dataset, (4) annotate sessions by mapping them to validation categories, and (5) run LLM-as-Judge scoring against the validation data. Each stage produces artifacts that feed the next, creating a reproducible evaluation workflow.

CloudWatch serves as the telemetry backbone. The S2S application emits OpenTelemetry spans that record each turn of the conversation — user audio transcriptions, assistant responses, timing data, and session identifiers. By querying CloudWatch log groups, you retrieve all spans for a time window and group them by session ID.

**Build: Extract traces from CloudWatch and build an evaluation dataset**

```python
from s2s_evaluator import S2SEvaluator

EVAL_DATASET_PATH = "data/s2s_eval_data.jsonl"
evaluator = S2SEvaluator(boto3_session=session)

# Extract all traces from the last 24 hours
traces = evaluator.extract_traces_from_cloudwatch(hours_back=24)

# Process into structured evaluation dataset
eval_data = evaluator.process_and_store_eval_dataset(
    [traces], EVAL_DATASET_PATH
)
print(f"Saved {len(eval_data)} sessions to {EVAL_DATASET_PATH}")
```

---

### Section 2: Speech Metrics Implementation

Once traces are extracted, you must map each session to a validation category (e.g., "TechnicalInterview", "OrderAssistant") so the LLM judge can compare actual behavior against expected behavior. This mapping can be manual (via an annotation UI) or automated. The evaluation then scores each session turn-by-turn using configurable criteria from a JSON config file.

The LLM-as-Judge approach uses a model (e.g., Claude Haiku) to assess whether the assistant's responses match the expected conversation flow for that category. Results are aggregated across multiple runs to reduce variance, and a markdown report is generated with per-category scores.

Key metrics include: overall pass rate, per-category accuracy, and individual turn scores. Multiple evaluation runs can be merged to produce confidence intervals and identify flaky test cases.

**Build: Run LLM-as-Judge evaluation on mapped sessions**

```python
import json
from pathlib import Path

VALIDATION_DATASET_PATH = "data/s2s_validation_dataset.jsonl"
MAPPINGS_FILE = Path("config/manual_mappings.json")

eval_data = evaluator.load_eval_dataset(EVAL_DATASET_PATH)
manual_mappings = evaluator.load_manual_mappings(str(MAPPINGS_FILE))
validation_data = evaluator.load_validation_dataset(VALIDATION_DATASET_PATH)
config = evaluator.load_config("config/llm_judge_s2s_config.json")
judge = evaluator.initialize_judge(config)

# Run evaluation
results = evaluator.run_evaluation_iteration(
    category_filter=None,
    eval_data=eval_data,
    validation_data=validation_data,
    manual_mappings=manual_mappings
)

# Generate report
merged = evaluator.merge_results([results])
report = evaluator.generate_evaluation_report(merged)
print(f"Total evaluations: {merged.get('total_evaluations', 0)}")
```

---

### Section 3: Automated Reasoning Concepts

Automated Reasoning (AR) Checks in Amazon Bedrock Guardrails verify LLM outputs against formal policy rules using a two-step pipeline: (1) translation of natural language into logical formulas, and (2) verification of those formulas against policy rules via an SMT solver. This produces one of seven validation result types:

| Result | Meaning |
|--------|---------|
| `VALID` | Claim satisfies all policy rules |
| `INVALID` | Claim violates at least one rule |
| `SATISFIABLE` | Some interpretations valid, some not |
| `IMPOSSIBLE` | Contradictory premises detected |
| `TRANSLATION_AMBIGUOUS` | Multiple interpretations possible |
| `NO_TRANSLATIONS` | No policy variables matched |
| `TOO_COMPLEX` | SMT solver timed out |

The evaluation framework measures five key metrics: **False Valid Rate** (safety — target 0%), **Consistency Accuracy** (predictability — target >80%), **Macro F1** (balance across types — target >0.8), **Ideal Accuracy** (quality with perfect translation), and **Fidelity Gap** (room for improvement via variable descriptions).

Each test case has two expectation fields: `expected_finding_type` (what the policy actually produces) and `ideal_finding_type` (what it should produce with perfect translation). The gap between these measures translation fidelity.

**Build: Create AR policies and verify guardrail access**

```python
from botocore.config import Config

config = Config(retries={"max_attempts": 3})
bedrock_client = boto3.client("bedrock", config=config)
bedrock_runtime_client = boto3.client("bedrock-runtime", config=config)

# Verify guardrail access
GUARDRAIL_ID = "your-guardrail-id"
GUARDRAIL_VERSION = "DRAFT"

resp = bedrock_runtime_client.apply_guardrail(
    guardrailIdentifier=GUARDRAIL_ID,
    guardrailVersion=GUARDRAIL_VERSION,
    source="OUTPUT",
    content=[{"text": {"text": "test"}}]
)
print(f"Guardrail verified: {GUARDRAIL_ID}")
```

---

### Section 4: Reasoning Verification Implementation

The evaluation loop runs test cases through the AR policy test API, collects validation results, and computes metrics. For each test, you create a temporary test case, run the test workflow, poll for results, normalize the validation result type, and extract translation quality metrics (confidence, untranslated claims, rules triggered).

The metrics computation uses a confusion matrix approach: for each validation result type, compute precision (of all predictions of this type, how many correct?), recall (of all tests that should be this type, how many caught?), and F1 (harmonic mean). The aggregate Macro F1 weights all types equally to catch imbalances.

The most critical metric is False Valid Rate — when the policy certifies a non-compliant claim as valid. This is the AR equivalent of a safety system saying "all clear" when there's a real problem.

**Build: Run AR evaluation and compute core metrics**

```python
import numpy as np

FINDING_TYPES = [
    "translationAmbiguous", "impossible", "invalid",
    "satisfiable", "valid", "tooComplex", "noTranslations"
]
FINDING_PRIORITY = {ft: i for i, ft in enumerate(FINDING_TYPES)}

def compute_core_metrics(df, expected_col="expected_finding_type"):
    valid_df = df[df["api_error"].isna()]
    accuracy = valid_df["finding_correct"].mean()

    labels = [ft for ft in FINDING_TYPES
              if ft in valid_df[expected_col].values
              or ft in valid_df["actual_finding_type"].values]

    import pandas as pd
    cm = pd.crosstab(
        valid_df[expected_col], valid_df["actual_finding_type"],
        dropna=False
    ).reindex(index=labels, columns=labels, fill_value=0)

    per_type = {}
    for ft in labels:
        tp = cm.loc[ft, ft] if ft in cm.index and ft in cm.columns else 0
        fp = cm[ft].sum() - tp if ft in cm.columns else 0
        fn = cm.loc[ft].sum() - tp if ft in cm.index else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_type[ft] = {"precision": precision, "recall": recall, "f1": f1}

    types_with_data = [ft for ft in labels if ft in cm.index and cm.loc[ft].sum() > 0]
    macro_f1 = np.mean([per_type[ft]["f1"] for ft in types_with_data]) if types_with_data else 0

    # False valid rate
    expected_non_valid = valid_df[valid_df[expected_col] != "valid"]
    false_valid = expected_non_valid[expected_non_valid["actual_finding_type"] == "valid"]
    false_valid_rate = len(false_valid) / len(expected_non_valid) if len(expected_non_valid) > 0 else 0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "false_valid_rate": false_valid_rate,
        "per_type": per_type,
    }

# After running evaluation and collecting results into a DataFrame `df`:
# metrics = compute_core_metrics(df)
# print(f"Accuracy: {metrics['accuracy']:.1%}")
# print(f"Macro F1: {metrics['macro_f1']:.3f}")
# print(f"False Valid Rate: {metrics['false_valid_rate']:.1%}")
```

---

## Challenges

Design and implement a combined evaluation dashboard that reports on both a speech-to-speech system and an automated reasoning policy. Your dashboard should:

1. Extract S2S traces from CloudWatch and run at least one LLM-as-Judge evaluation pass
2. Run at least 10 AR test cases through the policy test API
3. Compute and display: S2S pass rate, AR consistency accuracy, AR macro F1, and AR false valid rate
4. Visualize results with at least one confusion matrix and one summary chart
5. Produce a markdown report summarizing findings from both evaluation domains

**Assessment criteria:**

1. Pipeline runs without errors end-to-end
2. Correctly extracts and processes CloudWatch traces into evaluation datasets
3. Implements LLM-as-Judge scoring with configurable validation categories
4. Computes AR metrics including false valid rate, accuracy, and per-type F1
5. Generates visualizations (confusion matrix, summary chart) from evaluation results
6. Produces a coherent markdown report covering both speech and reasoning evaluations
7. Learner can explain their approach and interpret the metrics

## Wrap-Up

You have now built evaluation pipelines for two specialized AI workloads: speech-to-speech systems (using CloudWatch telemetry and LLM-as-Judge) and automated reasoning policies (using SMT solver verification and classification metrics). These techniques extend the foundational evaluation patterns from earlier modules into production-grade quality measurement.

Key takeaways:
- Speech evaluations require end-to-end trace capture and category-based validation
- AR evaluations produce formal verification results that can be measured with classification metrics
- False Valid Rate is the most critical safety metric for reasoning systems
- The Fidelity Gap between consistency and ideal accuracy reveals optimization opportunities

For the Module 04 capstone challenge, see `CHALLENGE-capstone.md` — it integrates techniques from all workload-specific evaluations in this module into a comprehensive evaluation system.

To continue building your evaluation expertise, explore Module 05 for production monitoring and continuous evaluation patterns.
