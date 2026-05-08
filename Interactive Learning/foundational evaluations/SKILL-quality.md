---
name: llm-quality-evaluation
description: Build LLM-as-Judge and LLM-as-Jury evaluation systems. Activate when asked to "evaluate LLM outputs", "build a judge prompt", "compare judge vs jury", "measure inter-judge agreement", or "add confidence intervals to evaluations".
---

# LLM Quality Evaluation: Judge and Jury Patterns

Build automated evaluation systems that score LLM outputs using structured judge prompts, then scale to multi-judge juries with statistical confidence — so you know *how much* to trust each score.

## Prerequisites
- Completion of Module 01 (concepts: operational metrics, CloudWatch dashboards, latency tracking)
- Source notebooks: `../../Foundational Evaluations/02-quality-metrics/01_LLM_as_Judge_analysis.ipynb`, `../../Foundational Evaluations/02-quality-metrics/02_LLM_as_Jury_evaluation_analysis.ipynb`
- AWS services: Amazon Bedrock (Claude 3.7 Sonnet)
- Python libraries: boto3, numpy, pandas, matplotlib

## Learning Objectives
By the end of this module, you will:
- Implement a structured LLM-as-Judge evaluation with a multi-criteria scoring rubric
- Run parallel judge evaluations across a dataset of model responses
- Build a multi-judge jury system that quantifies inter-judge reliability
- Calculate bootstrap confidence intervals from jury scores
- Configure dynamic pass/fail thresholds that adjust for judge disagreement

## Setup

```python
import numpy as np
import pandas as pd
import boto3
import json
from concurrent.futures import ThreadPoolExecutor

bedrock = boto3.client("bedrock-runtime", region_name="us-west-2")
JUDGE_MODEL_ID = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"

def call_judge_model(prompt: str) -> str:
    """Call Bedrock to get a judge evaluation."""
    response = bedrock.converse(
        modelId=JUDGE_MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 2048, "temperature": 0.0}
    )
    return response["output"]["message"]["content"][0]["text"]
```

## Section 1: Designing a Judge Prompt with Scoring Rubric

**Concept:** A single LLM evaluating another LLM's output is only as good as its rubric. Vague instructions like "rate quality 1-10" produce inconsistent scores. Structured rubrics with named criteria force the judge to evaluate each dimension independently, producing scores you can decompose and act on.

**Build:**

```python
JUDGE_PROMPT_TEMPLATE = """Evaluate the model response against these criteria:

<question>{question}</question>
<model_response>{response}</model_response>
<context>{context}</context>

Score each criterion 1-5:
1. **Data Accuracy**: Are facts correct given the context?
2. **Calculation Correctness**: Are mathematical operations sound?
3. **Analytical Depth**: Does it provide insight beyond retrieval?
4. **Completeness**: Does it address all parts of the question?

Respond in this exact format:
<scores>
accuracy: X/5
calculation: X/5
depth: X/5
completeness: X/5
</scores>
<reasoning>Your explanation</reasoning>
"""

def build_judge_prompt(question: str, response: str, context: str = "") -> str:
    return JUDGE_PROMPT_TEMPLATE.format(
        question=question, response=response, context=context
    )
```

## Section 2: Running Judge Evaluations at Scale

**Concept:** Evaluating one response is trivial. Evaluating thousands requires parallel execution and structured result parsing. The pattern is: build all prompts first, execute concurrently, then parse structured output into a DataFrame for analysis.

**Build:**

```python
def run_judge_evaluations(responses: list[dict]) -> list[dict]:
    """Evaluate responses in parallel using the judge prompt."""
    prompts = [
        build_judge_prompt(r["question"], r["model_response"], r.get("context", ""))
        for r in responses
    ]

    with ThreadPoolExecutor(max_workers=3) as executor:
        raw_results = list(executor.map(call_judge_model, prompts))

    parsed = []
    for raw in raw_results:
        scores = {}
        for line in raw.split("\n"):
            for criterion in ["accuracy", "calculation", "depth", "completeness"]:
                if criterion in line and "/" in line:
                    scores[criterion] = int(line.split("/")[0].strip()[-1])
        parsed.append(scores)
    return parsed
```

## Section 3: Multi-Judge Jury and Reliability Scoring

**Concept:** A single judge can be biased — lenient, harsh, or inconsistent across metrics. LLM-as-Jury uses multiple judges evaluating the same response. But not all judges are equally trustworthy. Reliability scoring profiles each judge on consistency (low variance), discrimination (uses full scale), and centrality (not systematically biased).

**Build:**

```python
def calculate_judge_reliability(scores_history: list[int]) -> dict:
    """Score a judge's reliability from their evaluation history."""
    arr = np.array(scores_history)
    mean = np.mean(arr)

    # Consistency: low coefficient of variation = more reliable
    cv = np.std(arr) / mean if mean > 0 else 0
    consistency = 1 - min(cv, 1)

    # Discrimination: uses multiple distinct scores (not all 4s)
    unique_ratio = len(np.unique(arr)) / 5  # scale is 1-5
    discrimination = min(unique_ratio, 1.0)

    # Centrality: not systematically extreme
    centrality = 1 - abs(mean - 3.0) / 2.0

    overall = 0.4 * consistency + 0.35 * discrimination + 0.25 * centrality
    return {
        "consistency": round(consistency, 3),
        "discrimination": round(discrimination, 3),
        "centrality": round(centrality, 3),
        "overall_reliability": round(overall, 3)
    }
```

## Section 4: Bootstrap Confidence Intervals from Jury Scores

**Concept:** Three judges score a response: 3, 4, 5. The average is 4 — but how confident are you? Bootstrap resampling answers this by simulating thousands of possible jury compositions from your actual scores, producing a confidence interval like "95% confident the true score is between 3.0 and 4.7." Wide intervals mean judges disagree; narrow intervals mean you can trust the score.

**Build:**

```python
def bootstrap_confidence_interval(
    scores: list, n_bootstrap: int = 1000, confidence: float = 0.95
) -> tuple[float, float, float]:
    """Calculate bootstrap CI for jury scores."""
    scores = np.array(scores)
    bootstrap_means = [
        np.mean(np.random.choice(scores, size=len(scores), replace=True))
        for _ in range(n_bootstrap)
    ]

    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_means, (alpha / 2) * 100)
    ci_upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)
    return float(np.mean(scores)), float(ci_lower), float(ci_upper)
```

## Section 5: Dynamic Thresholds — When Jury Beats Single Judge

**Concept:** Fixed pass/fail thresholds ignore uncertainty. A score of 3.1 with tight agreement (CI: 3.0–3.2) is genuinely passing. A score of 3.1 with wide disagreement (CI: 2.0–4.2) is unreliable. Dynamic thresholds adjust upward when judges disagree, preventing false positives. This is precisely when a jury beats a single judge: when you need to *quantify* how much to trust the evaluation.

**Build:**

```python
def confidence_based_decision(
    scores: list, base_threshold: float = 3.0, confidence_level: float = 0.95
) -> dict:
    """Make pass/fail decision accounting for judge disagreement."""
    mean, ci_low, ci_high = bootstrap_confidence_interval(scores, confidence=confidence_level)
    ci_width = ci_high - ci_low

    # Widen threshold when judges disagree
    adjusted_threshold = base_threshold + (ci_width * 0.5)

    if ci_low >= base_threshold:
        decision, conf = "PASS", 0.95
    elif ci_high < base_threshold:
        decision, conf = "FAIL", 0.95
    elif mean >= adjusted_threshold:
        decision, conf = "PASS", 0.7
    else:
        decision, conf = "FAIL", 0.7

    return {
        "decision": decision,
        "confidence": conf,
        "mean_score": round(mean, 2),
        "ci": (round(ci_low, 2), round(ci_high, 2)),
        "adjusted_threshold": round(adjusted_threshold, 2),
        "judges_agree": ci_width < 1.0
    }
```

## Challenges

### Challenge: End-to-End Judge vs. Jury Evaluation

Given a new dataset of LLM responses (e.g., customer support answers), implement both evaluation approaches and recommend which to use.

**Assessment criteria:**
1. Runs without errors on the provided dataset
2. Implements both single-judge (structured rubric) and multi-judge jury evaluation
3. Handles judge disagreement by computing confidence intervals and flagging uncertain cases
4. Uses an appropriate multi-criteria scoring rubric (not binary pass/fail)
5. Learner explains when jury evaluation beats a single judge — with evidence from their own results (e.g., "jury caught 3 false positives the single judge missed because CI crossed threshold")

**Starter structure:**
```python
# 1. Define your rubric (adapt criteria to your domain)
# 2. Run single-judge evaluation across all responses
# 3. Run 3-judge jury on the same responses
# 4. Compare: where do single-judge and jury disagree?
# 5. For disagreements, show CI width and threshold adjustment
# 6. Recommend: single judge (fast, cheap) vs jury (reliable, costly)
```

## Wrap-Up

**Key takeaways:**
- Structured rubrics with named criteria produce decomposable, actionable scores
- Bootstrap confidence intervals transform point estimates into ranges of trust
- Dynamic thresholds prevent false positives when judges disagree — this is the core advantage of jury over single judge

**What this does NOT cover:**
- Human-in-the-loop calibration workflows
- Cost optimization for multi-judge systems (token budgets)
- Fine-tuning judge models on domain-specific rubrics
- RAG pipeline construction (covered in Module 01 context section)
- Statistical tests beyond bootstrap (Krippendorff's alpha, Cohen's kappa)

**Next steps:**
- Module 03: Agentic Metrics (evaluating multi-step agent behavior)
- Extend jury to weighted voting using reliability scores from Section 3
- Build a monitoring dashboard (Module 01) tracking judge agreement over time
