---
name: llm-quality-evaluation
description: Build LLM-as-Judge evaluation systems with binary pass/fail verdicts. Activate when asked to "evaluate LLM outputs", "build a judge prompt", "calibrate a judge", "measure judge accuracy", or "add pass/fail evaluation".
---

# LLM Quality Evaluation: The Judge Pattern

Build automated evaluation systems that deliver binary pass/fail verdicts on LLM outputs using structured judge prompts, then calibrate the judge itself against human-labeled data — so you know *how much* to trust each verdict.

## Prerequisites
- Completion of Module 01 (concepts: operational metrics, CloudWatch dashboards, latency tracking)
- Source notebooks: `../../Foundational Evaluations/02-quality-metrics/01_LLM_as_Judge_analysis.ipynb`, `../../Foundational Evaluations/02-quality-metrics/03_Evaluating_your_Judge.ipynb`
- AWS services: Amazon Bedrock (Claude Sonnet 4.6)
- Python libraries: boto3, numpy, pandas, matplotlib

## Learning Objectives
By the end of this module, you will:
- Implement a structured LLM-as-Judge evaluation with binary pass/fail verdicts
- Run parallel judge evaluations across a dataset of model responses
- Aggregate verdicts into pass rates with binomial confidence intervals
- Calibrate a judge against human-labeled benchmarks using TPR/TNR
- Test judge repeatability so verdicts are stable across runs

## Setup

```python
import numpy as np
import pandas as pd
import boto3
import json
from concurrent.futures import ThreadPoolExecutor

bedrock = boto3.client("bedrock-runtime", region_name="us-west-2")
JUDGE_MODEL_ID = "us.anthropic.claude-sonnet-4-6"

def call_judge_model(prompt: str) -> str:
    """Call Bedrock to get a judge evaluation."""
    response = bedrock.converse(
        modelId=JUDGE_MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 1000, "temperature": 0.0}
    )
    return response["output"]["message"]["content"][0]["text"]
```

## Section 1: Designing a Judge Prompt with Binary Pass/Fail Verdicts

**Concept:** A single LLM evaluating another LLM's output is only as good as its rubric. Vague instructions like "rate quality 1-10" produce inconsistent scores — rating scales introduce implicit variation that hides the actual failure state. A good judge prompt evaluates **one failure mode**, defines pass and fail explicitly, and forces reasoning before the verdict.

**Build:**

```python
JUDGE_PROMPT_TEMPLATE = """Evaluate whether the model response is factually accurate given the context.

<question>{question}</question>
<model_response>{response}</model_response>
<context>{context}</context>

## Verdict Criteria (binary — do NOT use a rating scale)
- PASS: The numerical data in the response matches the context. Approximate values
  (e.g., "about 2.4 million" for 2,390,125) are acceptable. Additional commentary
  that does not contradict the data is fine.
- FAIL: The response contains a specific but wrong number, an incorrect calculation,
  reverses a comparison, claims data is unavailable when it is in the context, or
  does not answer the question asked.

Respond in this exact format:
<reasoning>Your step-by-step analysis</reasoning>
<verdict>pass or fail</verdict>
"""

def build_judge_prompt(question: str, response: str, context: str = "") -> str:
    return JUDGE_PROMPT_TEMPLATE.format(
        question=question, response=response, context=context
    )
```

**Anti-patterns to avoid:**
- `"Is this response good?"` — too vague, no criteria
- `"Rate this response from 1-5"` — scales hide failure modes behind implicit variation
- `"Check accuracy, completeness, tone, and helpfulness"` — multi-criteria overload; you can't tell which criterion failed. Build one judge per failure mode and combine results afterward.

## Section 2: Running Judge Evaluations at Scale

**Concept:** Evaluating one response is trivial. Evaluating thousands requires parallel execution and structured result parsing. The pattern is: build all prompts first, execute concurrently, then parse verdicts into a DataFrame for analysis.

**Build:**

```python
import re

def parse_verdict(raw: str) -> str | None:
    """Extract and normalize the pass/fail verdict from judge output."""
    match = re.search(r"<verdict>(.*?)</verdict>", raw, re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    verdict = match.group(1).strip().lower()
    return verdict if verdict in ("pass", "fail") else None

def run_judge_evaluations(responses: list[dict]) -> pd.DataFrame:
    """Evaluate responses in parallel and return verdicts as a DataFrame."""
    prompts = [
        build_judge_prompt(r["question"], r["model_response"], r.get("context", ""))
        for r in responses
    ]

    with ThreadPoolExecutor(max_workers=3) as executor:
        raw_results = list(executor.map(call_judge_model, prompts))

    df = pd.DataFrame({
        "question": [r["question"] for r in responses],
        "verdict": [parse_verdict(raw) for raw in raw_results],
    })
    df["passed"] = df["verdict"] == "pass"
    return df
```

## Section 3: Aggregating Verdicts — Pass Rates and Confidence Intervals

**Concept:** Binary verdicts aggregate into a pass rate, which is directly actionable: "91.7% of responses pass" tells you exactly how many cases need human review. But a pass rate from 20 samples is far less trustworthy than one from 1,000. Binomial confidence intervals quantify that uncertainty.

**Build:**

```python
def pass_rate_with_ci(passed: pd.Series, confidence: float = 0.95) -> dict:
    """Pass rate with a normal-approximation binomial confidence interval."""
    n = len(passed)
    p = passed.mean()
    z = 1.96  # 95% confidence
    se = np.sqrt(p * (1 - p) / n) if n > 0 else 0

    return {
        "pass_rate": round(p * 100, 1),
        "ci_lower": round(max(0, p - z * se) * 100, 1),
        "ci_upper": round(min(1, p + z * se) * 100, 1),
        "n": n,
        "failures": int((~passed).sum()),
    }

def pass_rate_by_type(df: pd.DataFrame) -> pd.DataFrame:
    """Break pass rates down by question type to find failure clusters."""
    return df.groupby("question_type").agg(
        pass_rate=("passed", "mean"),
        count=("passed", "count"),
        failures=("passed", lambda s: int((~s).sum())),
    ).sort_values("pass_rate")
```

Failures rarely distribute evenly — they cluster in specific question types. The per-type breakdown tells you where to focus prompt engineering effort.

## Section 4: Calibrating Your Judge Against Human Labels

**Concept:** A judge is just another prompt — it needs its own test set. Before trusting the judge in production, run it against human-labeled examples and measure agreement. True Positive Rate (TPR: of real passes, how many did the judge catch?) and True Negative Rate (TNR: of real fails, how many did the judge catch?) quantify how accurate your judge is as an estimator of real model error.

**Build:**

```python
def calibrate_judge(judge_labels: list[str], human_labels: list[str]) -> dict:
    """Compare judge verdicts to human ground truth."""
    df = pd.DataFrame({"judge": judge_labels, "human": human_labels})

    actual_pass = df[df["human"] == "pass"]
    actual_fail = df[df["human"] == "fail"]

    tpr = (actual_pass["judge"] == "pass").mean() if len(actual_pass) else 0
    tnr = (actual_fail["judge"] == "fail").mean() if len(actual_fail) else 0

    return {
        "accuracy": round((df["judge"] == df["human"]).mean(), 3),
        "tpr": round(tpr, 3),   # catches real passes
        "tnr": round(tnr, 3),   # catches real fails
        "disagreements": df[df["judge"] != df["human"]],
    }
```

**Workflow:** Split your human-labeled benchmark into few-shot (~15%, examples placed in the judge prompt), dev (~45%, iterative calibration), and test (~40%, final blind validation). Analyze disagreements on the dev set, refine the judge prompt, and only touch the test set once at the end. Notebook 03 walks through the full loop.

## Section 5: Judge Repeatability

**Concept:** A judge that flips its verdict on the same input across runs is unusable — you can't tell whether a change in pass rate comes from the model or from judge noise. Run the judge multiple times on identical inputs and measure verdict stability. Temperature 0 helps but does not guarantee determinism.

**Build:**

```python
def test_repeatability(prompt: str, n_runs: int = 5) -> dict:
    """Run the judge repeatedly on the same input and check verdict stability."""
    verdicts = [parse_verdict(call_judge_model(prompt)) for _ in range(n_runs)]
    counts = pd.Series(verdicts).value_counts()

    return {
        "verdicts": verdicts,
        "stable": len(counts) == 1,
        "majority": counts.index[0],
        "agreement": round(counts.iloc[0] / n_runs, 2),
    }
```

If verdicts flip between runs, your pass/fail definitions are ambiguous for that input — tighten the rubric or add a few-shot example covering the borderline case.

## Challenges

### Challenge: End-to-End Judge Build and Calibration

Given a new dataset of LLM responses (e.g., customer support answers), build a pass/fail judge and prove it can be trusted.

**Assessment criteria:**
1. Runs without errors on the provided dataset
2. Implements a single-failure-mode judge with explicit binary pass/fail definitions (no rating scales)
3. Aggregates verdicts into pass rates with confidence intervals, broken down by question type
4. Calibrates the judge against a small human-labeled set and reports TPR/TNR
5. Learner explains what the disagreement analysis revealed and how they refined the judge prompt in response — with evidence from their own results (e.g., "the judge failed responses that used approximations, so I added an approximation rule to the rubric")

**Starter structure:**
```python
# 1. Define ONE failure mode and explicit pass/fail criteria
# 2. Build the judge prompt with structured output (reasoning before verdict)
# 3. Run the judge across all responses in parallel
# 4. Aggregate: overall pass rate + per-type breakdown with CIs
# 5. Hand-label ~20 examples, run calibration, report TPR/TNR
# 6. Analyze disagreements, refine the prompt, re-measure
```

## Wrap-Up

**Key takeaways:**
- Binary pass/fail verdicts beat rating scales: they force clear criteria, are easy to aggregate, and every failure is directly actionable
- One failure mode per judge — combine multiple focused judges instead of one overloaded prompt
- A judge is just another prompt: calibrate it against human labels (TPR/TNR) and test its repeatability before trusting it

**What this does NOT cover:**
- Trace review, open coding, and failure-pattern discovery (Module 03: Understanding Failures)
- Fine-tuning judge models on domain-specific rubrics
- RAG pipeline construction (covered in Module 01 context section)

**Next steps:**
- Module 03: Agentic Metrics (evaluating multi-step agent behavior)
- Work through notebook 03's full calibration loop, including the held-out test set validation
- Build a monitoring dashboard (Module 01) tracking pass rates and judge agreement over time
