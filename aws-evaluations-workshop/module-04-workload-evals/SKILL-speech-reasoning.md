---
name: Speech and Automated Reasoning Evaluation
description: Help me evaluate speech-to-speech applications with LLM-as-Judge and verify LLM outputs against formal policy rules using Automated Reasoning checks
---

# Evaluating Speech-to-Speech and Automated Reasoning

Two distinct evaluation domains that share a common theme: verifying AI system outputs against ground truth. Speech evaluation uses LLM-as-Judge to assess conversation quality from telemetry traces. Automated Reasoning uses formal verification (SMT solvers) to check claims against policy rules. Together they cover the spectrum from subjective quality assessment to mathematically provable correctness.

This module does NOT cover: building speech applications, training speech models, writing AR policy documents from scratch, or production guardrail deployment patterns.

## Prerequisites

- AWS account with Bedrock access (us-east-1 for AR, us-west-2 for speech)
- Python 3.10+
- Familiarity with boto3, CloudWatch Logs, and the Bedrock Guardrails API
- A deployed speech-to-speech application with OpenTelemetry tracing (for Section 1–2)
- Completed Module 04 Guardrails section or equivalent understanding of Bedrock Guardrails

## Learning Objectives

By the end of this module, you will be able to:

1. Extract and structure speech-to-speech conversation traces from CloudWatch into evaluation datasets
2. Implement an LLM-as-Judge pipeline that scores multi-turn voice interactions against validation criteria
3. Configure Automated Reasoning policies and run claims through the policy test API
4. Compute AR evaluation metrics (false valid rate, consistency accuracy, macro F1) and interpret validation results
5. Diagnose AR mismatches by analyzing how question context affects translation behavior

## Setup

```bash
uv venv .venv
source .venv/bin/activate
uv pip install boto3 pandas numpy matplotlib seaborn python-dotenv
```

```python
import boto3
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path

session = boto3.Session(region_name='us-east-1')
bedrock_client = session.client('bedrock')
bedrock_runtime = session.client('bedrock-runtime')

# Verify access
bedrock_client.list_guardrails(maxResults=1)
print("Bedrock access verified")
```

## Section 1: Speech Evaluation Concepts

**Concept:** Speech-to-speech applications produce conversations captured as telemetry spans in CloudWatch. Evaluating them requires three steps: (1) extract traces and reconstruct turn-by-turn conversations, (2) map sessions to validation categories so you know what "correct" looks like, and (3) run an LLM judge that scores each turn against category-specific criteria.

The key insight is that speech interactions have temporal structure — turn order, response latency, and tool call sequences all matter. A text-only evaluation misses these signals. The S2SEvaluator class handles this by preserving span metadata (timestamps, tool calls, session state transitions) alongside the text content.

**Build:**

```python
from s2s_evaluator import S2SEvaluator

evaluator = S2SEvaluator(boto3_session=session)

# Step 1: Extract traces from CloudWatch
traces = evaluator.extract_traces_from_cloudwatch(
    hours_back=24,
    log_group_name='aws/spans'
)

# Step 2: Process spans into structured eval dataset
eval_data = evaluator.process_and_store_eval_dataset(
    raw_traces=[traces],
    output_path="data/s2s_eval_data.jsonl"
)

print(f"Extracted {len(eval_data)} sessions")
for session_data in eval_data[:2]:
    print(f"  Session: {len(session_data.get('turns', []))} turns")
```

## Section 2: Speech Quality Metrics with LLM-as-Judge

**Concept:** The LLM judge evaluates each session against a validation dataset — predefined "golden" conversations that define expected behavior per category (e.g., TechnicalInterview, OrderAssistant). The judge scores on multiple dimensions: response relevance, tool call accuracy, conversation flow, and task completion. Running multiple iterations and merging results reduces variance from LLM non-determinism.

**Build:**

```python
# Load config and validation data
config = evaluator.load_config("config/llm_judge_s2s_config.json")
validation_data = evaluator.load_validation_dataset(
    "data/s2s_validation_dataset.jsonl"
)

# Initialize judge and run evaluation
judge = evaluator.initialize_judge(config)
results = evaluator.run_evaluation_iteration(
    category_filter=None  # Evaluate all categories
)

# Run multiple iterations for statistical confidence
all_runs = [evaluator.run_evaluation_iteration() for _ in range(3)]
merged = evaluator.merge_results(all_runs)

# Generate report
report = evaluator.generate_evaluation_report(merged)
print(report[:500])
```

## Section 3: Automated Reasoning Concepts

**Concept:** Automated Reasoning (AR) Checks verify LLM outputs against formal policy rules using a two-step pipeline: (1) an LLM translates natural language claims into logical formulas, (2) an SMT solver checks those formulas against compiled policy rules. This produces 7 validation result types:

| Result | Meaning | Action |
|--------|---------|--------|
| `VALID` | Claim satisfies all rules | Serve response |
| `INVALID` | Claim violates a rule | Block/rewrite |
| `SATISFIABLE` | Claim is consistent with rules | Serve with caveat |
| `IMPOSSIBLE` | Contradictory premises | Review input |
| `noTranslations` | Couldn't parse the claim | Out of scope |
| `partiallyTranslated` | Partial parse | Lower confidence |
| `unknown` | Solver timeout | Retry or flag |

The critical insight: **questions act as premises**. The AR translator uses both the user's question and the LLM's answer. The question establishes facts (premises), the answer makes claims. Changing the question can flip the validation result — this is a feature, not a bug.

**Build:**

```python
# AR finding type constants
FINDING_TYPES = [
    'valid', 'invalid', 'satisfiable', 'impossible',
    'noTranslations', 'partiallyTranslated', 'unknown'
]

def get_finding_type(finding):
    """Extract validation result from a union-typed AR finding."""
    for ft in FINDING_TYPES:
        if ft in finding:
            return ft, finding[ft]
    return "unknown", {}

def has_untranslated_parts(finding_detail):
    """Check if a finding has untranslated premises or claims."""
    translation = finding_detail.get('translation', {})
    return (len(translation.get('untranslatedClaims', [])) > 0 or
            len(translation.get('untranslatedPremises', [])) > 0)
```

## Section 4: Reasoning Verification Implementation

**Concept:** Evaluating an AR policy means running a test suite of claims with known expected outcomes and measuring how often the policy agrees. The key metrics are: false valid rate (safety — did it certify a bad claim?), consistency accuracy (predictability), and macro F1 (balance across all result types). You use the policy test API rather than `apply_guardrail` for evaluation because it produces more accurate translations.

**Build:**

```python
def run_ar_test(policy_arn, build_workflow_id, test_case):
    """Run a single claim through the AR policy test API."""
    # Create test case
    resp = bedrock_client.start_automated_reasoning_policy_test_workflow(
        policyArn=policy_arn,
        buildWorkflowId=build_workflow_id,
        testCases=[{
            'query': test_case['query'],
            'guardContent': [{'text': {'text': test_case['text']}}]
        }]
    )
    workflow_id = resp['testWorkflowId']

    # Poll for completion
    while True:
        status = bedrock_client.get_automated_reasoning_policy_test_workflow(
            policyArn=policy_arn, testWorkflowId=workflow_id
        )
        if status['status'] in ('COMPLETED', 'FAILED'):
            break
        time.sleep(2)

    return status.get('results', [{}])[0]


def compute_core_metrics(df, expected_col='expected_finding_type'):
    """Compute AR evaluation metrics from results DataFrame."""
    valid_df = df[df['api_error'].isna()]
    accuracy = (valid_df['actual_finding_type'] == valid_df[expected_col]).mean()

    # False valid rate — the safety-critical metric
    expected_invalid = valid_df[valid_df[expected_col] == 'invalid']
    false_valid = (expected_invalid['actual_finding_type'] == 'valid').sum()
    false_valid_rate = false_valid / len(expected_invalid) if len(expected_invalid) > 0 else 0

    # Macro F1 across all result types
    labels = list(set(valid_df[expected_col]) | set(valid_df['actual_finding_type']))
    cm = pd.crosstab(valid_df[expected_col], valid_df['actual_finding_type'])

    return {
        'accuracy': accuracy,
        'false_valid_rate': false_valid_rate,
        'confusion_matrix': cm,
        'n_tests': len(valid_df),
    }
```

## Challenges

1. **Speech Pipeline Extension:** Add a custom scoring dimension to the LLM judge config (e.g., "empathy" or "technical accuracy") and run a comparative evaluation showing how scores change across categories.

2. **AR Boundary Testing:** Create 5 test cases that probe exact numeric boundaries in your AR policy (e.g., "exactly 70 square feet" vs "69.9 square feet"). Predict the expected validation result for each, run them, and explain any mismatches.

3. **Context Sensitivity Investigation:** Take one AR test case and run it with three different question phrasings while keeping the answer identical. Document how the validation result changes and explain why based on the premise/claim distinction.

4. **Capstone:** See CHALLENGE-capstone.md for the Module 04 capstone that integrates guardrails, structured data, RAG, speech, and reasoning evaluation into a unified assessment pipeline.

## Wrap-Up

You built two complementary evaluation pipelines:

- **Speech evaluation** uses LLM-as-Judge over telemetry traces — subjective but scalable quality assessment for voice interactions
- **Automated Reasoning** uses formal verification — mathematically provable correctness checks against policy rules

The key difference: speech evaluation tells you "this conversation was good/bad" (probabilistic). AR evaluation tells you "this claim is valid/invalid" (deterministic). Production systems need both — AR for compliance-critical outputs, LLM-as-Judge for everything else.

Next: Complete the Module 04 capstone (CHALLENGE-capstone.md) to integrate all workload-specific evaluation techniques into a unified pipeline.
