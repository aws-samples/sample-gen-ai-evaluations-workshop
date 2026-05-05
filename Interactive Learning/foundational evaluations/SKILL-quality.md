---
name: LLM Quality Metrics - Judge and Jury Evaluation
description: Build LLM-as-Judge and LLM-as-Jury evaluation systems, compare agreement rates, calculate inter-judge reliability metrics, and recommend which approach to use for different evaluation scenarios.
---

In this module you will build two complementary evaluation systems for assessing LLM output quality. First, you'll implement an LLM-as-Judge pattern using structured scoring rubrics and prompt templates. Then you'll extend this into an LLM-as-Jury system with multiple judges, aggregate their scores, and measure inter-judge agreement. By the end, you'll be able to compare both approaches and recommend which to use based on the evaluation context.

## Prerequisites

- Completion of Module 01 (Programmatic Testing fundamentals)
- AWS account with Amazon Bedrock access
- Python 3.10+ with boto3, pandas, numpy, matplotlib, seaborn, scipy
- Basic understanding of prompt engineering

## Learning Objectives

- Implement an LLM-as-Judge evaluation pipeline with structured scoring rubrics
- Create judge prompt templates that produce consistent, metric-based scores
- Build an LLM-as-Jury system using multiple judge models
- Calculate inter-judge agreement rates and reliability metrics
- Compare Judge vs. Jury approaches and recommend which to deploy

## Setup

Install dependencies and configure the environment:

```python
!pip install -q pandas seaborn scipy matplotlib boto3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import boto3
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

bedrock = boto3.client("bedrock-runtime")
JUDGE_MODEL_ID = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"

def call_judge_model(prompt: str) -> str:
    """Call the judge model to evaluate a response."""
    try:
        response = bedrock.converse(
            modelId=JUDGE_MODEL_ID,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"temperature": 0.1, "maxTokens": 1000}
        )
        return response["output"]["message"]["content"][0]["text"]
    except Exception as e:
        return f"Error: {str(e)}"
```

### Section 1: The Problem with Binary Pass/Fail Evaluation

When evaluating LLM outputs, simple pass/fail judgments hide critical information. A single judge might give a response a 5/5 while another gives it 2/5. Which score should you trust? Binary evaluations are problematic because they can be driven by many factors, making it difficult for a judge to prioritize criteria. Details get lost, leading to false positives and negatives.

The solution is to move away from binary pass/fail and switch to a ranking system of custom metrics—and to add more judges for consensus.

**Build: Demonstrate the Trust Problem**

Create a simple judge evaluation and visualize how two judges can disagree on the same response:

```python
customer_question = "What kind of food are you serving in the cafeteria today?"
model_answer = "Today we are serving chicken fingers, pizza and mixed fruits."

context = """
Food|Price
Pizza|2.00
Chicken Fingers|6.00
Mixed Fruits|4.00
"""

judge_prompt = f"""You will be given a question about our pirate themed mini golf company's facilities.
Your task is to evaluate a model's response for accuracy, completeness, and analytical quality.

Here is the question:
<question>{customer_question}</question>

Here is the model's response:
<model_response>{model_answer}</model_response>

Here is the context from the data:
<dataset>{context}</dataset>

If the model response is accurate, complete, and meets analytical quality return "MET CRITERIA", else return "FAIL TO MEET CRITERIA" ONLY.
"""

response = call_judge_model(judge_prompt)
print(f"Model response: {response}")

# Visualize disagreement between two judges
judge_scores = {'Judge A (Model A)': 5, 'Judge B (Model B)': 2}

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
judges = list(judge_scores.keys())
scores = list(judge_scores.values())
colors = ['#3498db', '#e74c3c']

bars = ax.bar(judges, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax.axhline(y=3, color='gray', linestyle='--', label='Pass/Fail Threshold')
ax.set_ylim(0, 5.5)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Same Response, Different Scores', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.show()
```

### Section 2: Judge Reliability Scoring

Not all AI judges are created equal. Some are reluctant to vary their scores while others swing widely. By profiling judge behavior patterns, we can identify which judges to trust more. High agreement metrics (like Correctness at 1.0) are objective—like judging if a door is open or closed. Low agreement metrics (like Completeness at 0.85) are subjective—like judging if a meal is "complete."

**Build: Calculate Judge Reliability**

Implement a reliability scoring framework that evaluates judges on consistency, discrimination, and centrality:

```python
def calculate_judge_reliability(judge_scores_history):
    """
    Calculate reliability score for a judge based on:
    1. Consistency (low variance)
    2. Discrimination (uses full scale)
    3. Centrality (avoiding extreme bias)
    """
    reliability_components = {}

    # Consistency (coefficient of variation)
    cv = np.std(judge_scores_history) / np.mean(judge_scores_history) if np.mean(judge_scores_history) > 0 else 0
    reliability_components['consistency'] = 1 - min(cv, 1)

    # Discrimination (uses multiple scores)
    unique_scores = len(np.unique(judge_scores_history))
    reliability_components['discrimination'] = min(unique_scores / 5.0, 1.0)

    # Central tendency (not always too high or too low)
    mean_score = np.mean(judge_scores_history)
    distance_from_center = abs(mean_score - 3) / 2
    reliability_components['centrality'] = 1 - distance_from_center

    reliability_score = np.mean(list(reliability_components.values()))
    return reliability_score, reliability_components

# Compare two judges
judge_a_scores = [5, 5, 4, 5, 5, 5, 4, 5, 5, 5]  # Consistent but lacks discrimination
judge_b_scores = [3, 4, 5, 4, 3, 5, 4, 4, 3, 5]  # More varied, better discrimination

reliability_a, components_a = calculate_judge_reliability(judge_a_scores)
reliability_b, components_b = calculate_judge_reliability(judge_b_scores)

print(f"Judge A Reliability: {reliability_a:.3f}")
for k, v in components_a.items():
    print(f"  {k}: {round(v, 2)}")
print(f"\nJudge B Reliability: {reliability_b:.3f}")
for k, v in components_b.items():
    print(f"  {k}: {round(v, 2)}")
```

### Section 3: From Scores to Confidence (Jury Aggregation)

Instead of just averaging scores, we can use bootstrap confidence intervals to transform point estimates into ranges of trust. When judges agree (scores: 4,4,5), we're more confident than when they disagree (scores: 2,4,5), even if both average similarly. This is the foundation of the LLM-as-Jury approach.

**Build: Bootstrap Confidence Intervals**

Implement bootstrap resampling to calculate confidence intervals from multiple judge scores:

```python
def bootstrap_confidence_interval(scores, n_bootstrap=1000, confidence=0.95):
    """Calculate bootstrap confidence interval for jury scores."""
    bootstrap_means = []

    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(scores, size=len(scores), replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))

    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    ci_lower = np.percentile(bootstrap_means, lower_percentile)
    ci_upper = np.percentile(bootstrap_means, upper_percentile)

    return np.mean(scores), ci_lower, ci_upper, bootstrap_means

# Three judges score a response
example_scores = [4, 5, 4]
mean_score, ci_lower, ci_upper, bootstrap_dist = bootstrap_confidence_interval(example_scores)

print(f"Mean Score: {mean_score:.2f}")
print(f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.bar(range(len(example_scores)), example_scores, color='steelblue', alpha=0.7)
ax1.axhline(y=mean_score, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean_score:.2f}')
ax1.set_title('Original Judge Scores', fontweight='bold')
ax1.legend()

ax2.hist(bootstrap_dist, bins=30, alpha=0.7, color='green', edgecolor='black')
ax2.axvline(x=ci_lower, color='orange', linestyle='--', linewidth=2)
ax2.axvline(x=ci_upper, color='orange', linestyle='--', linewidth=2)
ax2.axvspan(ci_lower, ci_upper, alpha=0.2, color='orange', label=f'95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]')
ax2.set_title('Bootstrap Confidence Interval', fontweight='bold')
ax2.legend()
plt.tight_layout()
plt.show()
```

### Section 4: Agreement Rate Calculation

The key metric for a jury system is the agreement rate between judges. High agreement means the evaluation is objective and reliable. Low agreement signals subjectivity that requires additional buffer zones. We measure this per-metric to understand which dimensions of quality are most contested.

**Build: Calculate Agreement Rates Across Metrics**

```python
def calculate_agreement_rate(scores_matrix, threshold=1):
    """
    Calculate pairwise agreement rate for a jury.
    scores_matrix: list of lists, each inner list is one judge's scores for N items.
    threshold: max difference to count as 'agreement'.
    """
    n_judges = len(scores_matrix)
    n_items = len(scores_matrix[0])
    agreements = 0
    comparisons = 0

    for item_idx in range(n_items):
        for i in range(n_judges):
            for j in range(i + 1, n_judges):
                comparisons += 1
                if abs(scores_matrix[i][item_idx] - scores_matrix[j][item_idx]) <= threshold:
                    agreements += 1

    return agreements / comparisons if comparisons > 0 else 0

# Simulate jury scores across metrics
np.random.seed(42)
metrics = ['Correctness', 'Completeness', 'Relevance', 'Coherence']
n_items = 20
n_judges = 3

results = {}
for metric in metrics:
    base_scores = np.random.choice([3, 4, 5], size=n_items, p=[0.2, 0.5, 0.3])
    judge_scores = []
    for j in range(n_judges):
        noise = np.random.normal(0, 0.3 if metric != 'Completeness' else 0.7, size=n_items)
        scores = np.clip(np.round(base_scores + noise), 1, 5).astype(int)
        judge_scores.append(scores.tolist())
    results[metric] = calculate_agreement_rate(judge_scores)

print("Agreement Rates by Metric:")
for metric, rate in sorted(results.items(), key=lambda x: x[1]):
    print(f"  {metric}: {rate:.2f}")
```

### Section 5: Judge vs. Jury — When to Use Which

The single-judge approach is faster and cheaper but vulnerable to bias and inconsistency. The jury approach provides confidence intervals and catches disagreement but costs more. The decision depends on the stakes of the evaluation, the subjectivity of the metrics, and the budget.

Use a single judge when: metrics are objective (factual correctness), speed matters, and the judge has high reliability scores. Use a jury when: metrics are subjective (completeness, style), high-stakes decisions depend on the evaluation, or you need confidence intervals for reporting.

**Build: Compare Both Approaches on the Same Dataset**

```python
def compare_judge_vs_jury(responses, context_list, n_jury=3):
    """Run both single-judge and jury evaluation, compare results."""
    single_judge_scores = []
    jury_scores = []
    jury_confidence_widths = []

    for i, (resp, ctx) in enumerate(zip(responses, context_list)):
        # Single judge
        single_score = np.random.choice([3, 4, 5], p=[0.1, 0.4, 0.5])
        single_judge_scores.append(single_score)

        # Jury of n judges
        jury = [np.clip(single_score + np.random.normal(0, 0.5), 1, 5) for _ in range(n_jury)]
        mean, ci_low, ci_high, _ = bootstrap_confidence_interval(jury)
        jury_scores.append(mean)
        jury_confidence_widths.append(ci_high - ci_low)

    agreement = np.mean([1 if abs(s - j) <= 0.5 else 0
                         for s, j in zip(single_judge_scores, jury_scores)])

    print(f"Agreement Rate (Judge vs Jury mean): {agreement:.2%}")
    print(f"Average Jury Confidence Width: {np.mean(jury_confidence_widths):.2f}")
    print(f"\nRecommendation:")
    if agreement > 0.85:
        print("  → Single judge is sufficient for this task (high agreement).")
    else:
        print("  → Use jury approach (significant disagreement detected).")

# Simulate
responses = [f"Response {i}" for i in range(30)]
contexts = [f"Context {i}" for i in range(30)]
compare_judge_vs_jury(responses, contexts)
```

## Challenges

### Challenge 1: Build a Structured LLM-as-Judge

Write a judge prompt template and evaluation function that scores LLM responses on a multi-dimensional rubric. The judge should evaluate responses to customer support questions for a retail company.

**Success criteria:**
- Judge prompt includes at least 3 scoring dimensions (e.g., Correctness, Completeness, Relevance) each scored 1–5
- Prompt constrains the judge to output structured scores (e.g., XML tags or JSON) with a brief justification per dimension
- Evaluation function calls Bedrock with `temperature=0.1`, parses the structured output, and returns a dict of dimension scores
- Run the judge on at least 5 different question/answer pairs and print a summary table of scores per dimension

### Challenge 2: Calculate Judge Reliability

Extend the `calculate_judge_reliability` function from Section 2 by adding a 4th reliability component: **bias detection** (measuring whether a judge systematically scores higher or lower than peers). Profile a judge's behavior across multiple evaluations.

**Success criteria:**
- Function accepts a list of scores from a single judge and returns a reliability score (0–1) with component breakdowns
- Consistency component uses coefficient of variation (lower CV = higher consistency)
- Discrimination component measures how many unique score values the judge uses relative to the scale
- Run the same 5 question/answer pairs through the judge 3 times each and compute reliability from the 15 scores
- Print reliability score and explain whether this judge is trustworthy

### Challenge 3: Build an LLM-as-Jury System

Extend your single-judge into a jury of 3 judges using different models or temperature settings. Implement **weighted jury voting** where each judge's score is weighted by their reliability score from Challenge 2, and aggregate using bootstrap confidence intervals.

**Success criteria:**
- Jury uses at least 2 distinct Bedrock models (e.g., Nova Pro and Claude Haiku) or the same model at different temperatures
- Each jury member scores the same 5 responses on the same rubric from Challenge 1
- Bootstrap confidence interval function resamples jury scores 1000 times and returns mean, 95% CI lower, and 95% CI upper
- Print a comparison table: response ID, single-judge score, jury mean, CI width
- Identify which responses have the widest confidence intervals and explain why

### Challenge 4: Agreement Rate Analysis and Recommendation

Calculate pairwise agreement rates across your jury members for each scoring dimension. Produce a written recommendation on when to use single-judge vs. jury evaluation.

**Success criteria:**
- Agreement rate function computes pairwise agreement (scores within ±1) for each dimension across all jury members
- Print agreement rates per dimension and identify which dimensions are most/least contested
- Written recommendation (3–5 sentences) addresses: cost tradeoff, when jury adds value, which dimensions need jury consensus
- Recommendation references specific agreement rate numbers from your results

### Challenge 5: Full Judge and Jury Evaluation Pipeline

Given a new dataset of LLM outputs (at least 20 responses), implement both a single-judge and a jury evaluation pipeline. Compare agreement rates across at least 3 metrics, handle judge disagreement with confidence intervals, and produce a written recommendation on which approach to use.

**Success criteria:**
- Pipeline runs without errors on the provided dataset
- Implements both single-judge and multi-judge (jury) evaluation patterns
- Handles judge disagreement using bootstrap confidence intervals or equivalent
- Uses an appropriate multi-dimensional scoring rubric (at least 3 metrics)
- Learner explains when jury evaluation beats single-judge evaluation and why

### Tips

- Use `temperature=0.1` for judges to maximize consistency; use `temperature=0.7` only if you want to test variance
- The `bootstrap_confidence_interval` function from Section 3 is a good starting point for Challenge 3
- Wide confidence intervals signal subjective dimensions — these are where jury evaluation adds the most value
- Keep your test dataset consistent across all exercises so results are comparable

## Wrap-Up

In this module you built two evaluation systems—LLM-as-Judge for fast, single-model scoring and LLM-as-Jury for robust, multi-model consensus. You learned to measure judge reliability, calculate agreement rates, and use confidence intervals to quantify uncertainty in evaluations.

**Feedback:** How did this module go? What would you improve? Please share your thoughts.

**Profile update:** Consider adding "LLM Evaluation Systems" and "Inter-Judge Reliability" to your skills profile.

**Next module:** Module 03 explores automated evaluation pipelines at scale, including continuous monitoring and drift detection.
