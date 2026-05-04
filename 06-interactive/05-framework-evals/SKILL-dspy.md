---
name: "DSPy Prompt Optimization"
description: "Use when learner needs to define DSPy signatures, write evaluation metrics, run BootstrapFewShot optimization, inspect optimized prompts, and compare before/after scores using Amazon Bedrock"
---

# DSPy Prompt Optimization

DSPy is an **eval-focused framework** where prompt optimization IS the evaluation loop. Instead of hand-writing prompts and manually testing them, you declare what goes in and what comes out, define a metric that scores quality, and let the optimizer write the prompt for you. The optimizer runs your metric on training examples, keeps the best-scoring outputs as few-shot demonstrations, and produces a portable JSON artifact you can deploy. In this skill, you will build a city Q&A system that answers population questions about US cities, measure accuracy with a percentage-error metric, optimize prompts automatically with `BootstrapFewShot`, and compare before/after results. By the end, you will understand how DSPy collapses the evaluate→improve cycle into a single automated loop.

## Prerequisites

- Completed Module 01 (evaluation fundamentals and terminology)
- Completed Module 02 (quality metrics, including LLM-as-judge concepts)
- AWS account with Amazon Bedrock model access (Claude Haiku)
- Python 3.11+ with pip
- Familiarity with Python classes and type annotations

## Learning Objectives

- Define a DSPy Signature that declares input/output fields for a city Q&A task
- Implement an evaluation metric that scores model outputs against ground-truth data
- Execute BootstrapFewShot optimization and measure improvement over a baseline
- Inspect the optimized program to identify what demonstrations the optimizer selected
- Save and load optimized programs as portable JSON artifacts

## Setup

Install DSPy and dependencies:

```bash
pip install "dspy>=3.1,<4" boto3 pandas --quiet
```

Configure the DSPy language model to use Bedrock:

```python
import dspy

lm = dspy.LM("bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0")
dspy.configure(lm=lm)

# Verify connection
response = lm("Say 'hello' and nothing else.")
print(f"Connection OK: {response}")
```

Download the `city_pop.csv` dataset into your working directory. This file contains population and land area data for 300 US cities.

---

### Section 1: Signatures and Data — Declaring What Goes In and Out

**Concept**

Traditional prompt engineering means editing strings, testing manually, and hoping for the best. DSPy replaces this with a declarative approach: you define a **Signature** that specifies input fields, output fields, and a docstring instruction. The signature is not a prompt—it's a specification that DSPy compiles into a prompt. This separation is what makes optimization possible.

The second ingredient is structured training data. You need question/answer pairs that represent what "correct" looks like. DSPy wraps these as `Example` objects with explicit input markers, so the optimizer knows which fields to fill and which to score against.

Why does this matter for evaluation? Because the signature + examples define the contract. The metric measures how well the model fulfills that contract. And the optimizer searches for prompts that maximize the metric. Without a clean declaration, there's nothing to optimize.

**Build**

Load and prepare the city data, then define the signature and create train/test splits:

```python
import pandas as pd
import re

df = pd.read_csv("city_pop.csv", encoding="utf-8-sig")
df["city"] = df["city"].apply(lambda x: re.sub(r"\[.*?\]", "", x).strip())
df["population"] = df["population"].apply(lambda x: int(str(x).replace(",", "")))
df["land_area_mi2"] = pd.to_numeric(df["land_area_mi2"].str.replace(",", ""), errors="coerce")
df["density"] = (df["population"] / df["land_area_mi2"]).round(1)

print(f"{len(df)} cities loaded")
```

Define the signature and build examples:

```python
class CityQA(dspy.Signature):
    """Answer factual questions about US cities using your knowledge."""
    question: str = dspy.InputField(desc="A question about a US city's population or demographics")
    answer: str = dspy.OutputField(desc="A precise numeric answer to the question")

def make_examples(df_slice):
    examples = []
    for _, row in df_slice.iterrows():
        examples.append(dspy.Example(
            question=f"What is the population of {row['city']}, {row['state']}?",
            answer=str(row["population"])
        ).with_inputs("question"))
        examples.append(dspy.Example(
            question=f"What is the population density (people per square mile) of {row['city']}, {row['state']}?",
            answer=str(row["density"])
        ).with_inputs("question"))
    return examples

trainset = make_examples(df.iloc[:20])
testset = make_examples(df.iloc[20:30])
print(f"Training: {len(trainset)} | Test: {len(testset)}")
```

Run the baseline (zero prompt engineering) and observe the variance:

```python
baseline = dspy.Predict(CityQA)

for ex in testset[:3]:
    result = baseline(question=ex.question)
    print(f"Q: {ex.question}")
    print(f"  Got: {result.answer} | Expected: {ex.answer}\n")
```

---

### Section 2: Evaluation Metrics — Defining What "Good" Looks Like

**Concept**

An optimizer is only as good as its metric. If your metric is broken, the optimizer will exploit the brokenness—maximizing a flawed score rather than actual quality. This is why DSPy's approach works: you invest effort in defining a precise metric, and the framework handles the prompt engineering.

The metric function receives an example (ground truth) and a prediction (model output). During optimization, it also receives a `trace` parameter—when present, the metric returns a boolean (pass/fail) so the optimizer knows which demonstrations to keep. During evaluation, it returns a numeric score for finer-grained reporting.

Before running any optimizer, you must test your metric on edge cases. A model might return "approximately 780,995 people" instead of "780995"—if your parser can't handle that, the optimizer will learn to avoid verbose answers rather than learning to be accurate.

**Build**

Implement the percentage-error metric:

```python
def extract_number(text):
    """Extract the first number from text, handling commas and decimals."""
    text = str(text).replace(",", "")
    matches = re.findall(r"[\d]+\.?[\d]*", text)
    return float(matches[0]) if matches else None

def city_metric(example, prediction, trace=None):
    """Score based on % error: within 5% → 1.0, within 10% → 0.5, else → 0.0."""
    expected = extract_number(example.answer)
    predicted = extract_number(prediction.answer)

    if expected is None or predicted is None or expected == 0:
        return False if trace else 0.0

    pct_error = abs(predicted - expected) / expected

    if pct_error <= 0.05:
        score = 1.0
    elif pct_error <= 0.10:
        score = 0.5
    else:
        score = 0.0

    if trace is not None:
        return score >= 0.5
    return score
```

Run the evaluator across the full test set to establish a baseline score:

```python
evaluator = dspy.Evaluate(
    devset=testset,
    metric=city_metric,
    num_threads=1,
    display_progress=True,
    display_table=5,
)

baseline_score = float(evaluator(baseline))
print(f"Baseline average score: {baseline_score:.2f}")
```

Test the metric on edge-case formats to verify it parses correctly:

```python
test_responses = [
    "The population is approximately 780,995 people.",
    "780995",
    "About 780K",
]
expected = 780995
for resp in test_responses:
    extracted = extract_number(resp)
    print(f"  '{resp}' → extracted: {extracted}")
```

---

### Section 3: Optimization — Letting the Machine Write the Prompt

**Concept**

Here is where DSPy's eval-focused nature becomes concrete. `BootstrapFewShot` is a teleprompter (optimizer) that runs your model on training examples, scores each output with your metric, and keeps the highest-scoring outputs as few-shot demonstrations. The optimization loop IS an evaluation loop—every candidate prompt is scored, compared, and either kept or discarded.

The key parameters are: `max_bootstrapped_demos` (how many model-generated examples to include), `max_labeled_demos` (how many ground-truth examples to include), and `max_rounds` (how many passes over the training set). More demos means a longer prompt and higher cost per call, but potentially better accuracy.

This is fundamentally different from manual prompt engineering. You don't guess which examples to include—the optimizer finds them by measuring what works.

**Build**

Configure and run the optimizer:

```python
import time

def retry_on_throttle(fn, max_retries=3, base_delay=5):
    """Retry with exponential backoff on Bedrock throttling."""
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            if "ThrottlingException" in str(e) and attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                print(f"⏳ Throttled — retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise

optimizer = dspy.BootstrapFewShot(
    metric=city_metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
    max_rounds=2,
)

optimized = retry_on_throttle(lambda: optimizer.compile(baseline, trainset=trainset))
optimized_score = float(evaluator(optimized))

print(f"{'='*40}")
print(f"Baseline score:  {baseline_score:.2f}")
print(f"Optimized score: {optimized_score:.2f}")
print(f"Improvement:     {optimized_score - baseline_score:+.2f}")
print(f"{'='*40}")
```

Inspect what the optimizer selected—no black boxes:

```python
print(f"Number of demos: {len(optimized.demos)}\n")
for i, demo in enumerate(optimized.demos):
    print(f"--- Demo {i+1} ---")
    print(f"  Q: {demo.get('question', 'N/A')}")
    print(f"  A: {demo.get('answer', 'N/A')}\n")

# See the full prompt DSPy now sends
_ = optimized(question=testset[0].question)
dspy.inspect_history(n=1)
```

---

### Section 4: Saving, Loading, and Composing Modules

**Concept**

An optimized program is useless if it lives only in a notebook session. DSPy serializes optimized programs as JSON artifacts—portable files you can commit to git, deploy to production, or roll back if quality drops. The artifact contains the selected demonstrations and any learned parameters, tied to the signature that produced them.

Beyond saving, DSPy lets you compose modules. A `Module` wraps one or more predictors into a reusable building block with a `forward()` method. You can swap `Predict` for `ChainOfThought` (which adds step-by-step reasoning) with a one-line change, then optimize the entire module end-to-end. The same optimizer that improved a simple predictor works on complex multi-step pipelines.

**Build**

Save and reload the optimized program:

```python
import json as _json

optimized.save("optimized_city_qa.json")
print("Saved optimized program")

# Load into a fresh predictor and verify
loaded = dspy.Predict(CityQA)
loaded.load("optimized_city_qa.json")
loaded_score = float(evaluator(loaded))
print(f"Original: {optimized_score:.2f} | Loaded: {loaded_score:.2f}")
print(f"Round-trip OK: {abs(loaded_score - optimized_score) < 0.01}")
```

Compose a module with ChainOfThought and optimize it:

```python
class CityExpert(dspy.Module):
    def __init__(self):
        super().__init__()
        self.answer = dspy.ChainOfThought(CityQA)

    def forward(self, question):
        return self.answer(question=question)

module = CityExpert()
module_score = float(evaluator(module))

module_optimizer = dspy.BootstrapFewShot(
    metric=city_metric,
    max_bootstrapped_demos=3,
    max_labeled_demos=3,
)
optimized_module = retry_on_throttle(lambda: module_optimizer.compile(module, trainset=trainset))
optimized_module_score = float(evaluator(optimized_module))

print(f"Module (CoT) score:       {module_score:.2f}")
print(f"Module (CoT) + optimized: {optimized_module_score:.2f}")
print(f"Improvement:              {optimized_module_score - module_score:+.2f}")
```

---

### Section 5: Enhanced Metrics — LLM-as-Judge Inside the Loop

**Concept**

Numeric accuracy catches wrong numbers but misses hallucinated context. A model could say "Seattle's population is 780,995, located in Oregon"—the number is right, but the surrounding claim is fabricated. To catch this, you add a second evaluation dimension: an LLM judge that checks faithfulness.

This is the same LLM-as-judge concept from Module 02, but here the judge lives *inside* the metric function. Because DSPy's optimizer uses the metric to select demonstrations, a richer metric produces better optimization. The judge doesn't just report problems—it actively steers the optimizer away from hallucination-prone prompts.

The tradeoff is cost and latency: each evaluation now requires an additional LLM call. You weight the components (e.g., 70% numeric + 30% faithfulness) to balance accuracy and reliability.

**Build**

Define the faithfulness judge and enhanced metric:

```python
class FaithfulnessCheck(dspy.Signature):
    """Check if the answer is factually faithful and not hallucinated."""
    question: str = dspy.InputField()
    answer: str = dspy.InputField()
    expected_answer: str = dspy.InputField()
    is_faithful: bool = dspy.OutputField(desc="True if the answer is factually consistent")
    reason: str = dspy.OutputField(desc="Brief explanation of the judgment")

faithfulness_judge = dspy.Predict(FaithfulnessCheck)

def enhanced_metric(example, prediction, trace=None):
    """Numeric accuracy (70%) + LLM faithfulness judge (30%)."""
    expected = extract_number(example.answer)
    predicted = extract_number(prediction.answer)

    if expected is None or predicted is None or expected == 0:
        numeric_score = 0.0
    else:
        pct_error = abs(predicted - expected) / expected
        if pct_error <= 0.05:
            numeric_score = 1.0
        elif pct_error <= 0.10:
            numeric_score = 0.5
        else:
            numeric_score = 0.0

    try:
        judgment = faithfulness_judge(
            question=example.question,
            answer=prediction.answer,
            expected_answer=example.answer
        )
        judge_score = 1.0 if judgment.is_faithful else 0.0
    except Exception:
        judge_score = 0.0

    score = 0.7 * numeric_score + 0.3 * judge_score

    if trace is not None:
        return score >= 0.5
    return score
```

Re-evaluate with the enhanced metric and compare:

```python
enhanced_evaluator = dspy.Evaluate(
    devset=testset,
    metric=enhanced_metric,
    num_threads=1,
    display_progress=True,
    display_table=5,
)

enhanced_score = float(enhanced_evaluator(optimized_module))
print(f"Numeric metric:   {optimized_module_score:.2f}")
print(f"Enhanced metric:  {enhanced_score:.2f}")
```

---

## Challenges

### Challenge: Optimize a New Domain with a Custom Metric

Choose a different Q&A domain (not city populations). Define a DSPy Signature, implement a custom evaluation metric, run `BootstrapFewShot` optimization, and compare baseline vs. optimized scores. Your metric must have at least two scoring tiers (not just pass/fail).

**Assessment criteria:**

1. Optimization runs without errors and produces a saved JSON artifact
2. Custom metric implements tiered scoring with the `trace` parameter pattern for optimization compatibility
3. Before/after comparison shows measurable score difference (improvement or explanation of why not)
4. Optimized program demos are inspected and the learner identifies what the optimizer selected
5. Learner can explain why their metric design drives the optimizer toward better outputs

---

## Deep-Dive Challenge

DSPy is an **eval-focused** framework — prompt optimization IS the evaluation loop. Instead of hand-tuning prompts, you declare inputs/outputs, define a metric, and let the optimizer search for the best prompt. This deep-dive pushes you beyond notebook-level usage into advanced optimization and evaluation patterns.

### Workflow

| Stage | What you implement |
|---|---|
| Test suite definition | 10+ test cases with inputs, expected outputs, and edge cases |
| Metric configuration | 3+ metrics (at least 1 custom beyond built-in) |
| Execution | Run eval suite against a live model endpoint |
| Analysis | Score distribution, failure clustering, threshold tuning |
| Iteration | Modify prompts or config based on results, re-run, show improvement |

### "Beyond" Examples for DSPy

- Custom teleprompter
- Evaluation of optimized vs. unoptimized across 3+ tasks
- Metric composition

### Scoring Rubric

| Tier | Points | Criteria |
|---|---|---|
| **Functional** | 60-69 | Complete workflow runs end-to-end; uses only notebook-level features; results are valid |
| **Extended** | 70-84 | Adds 1 capability not in notebook; clear justification for the extension |
| **Advanced** | 85-94 | Adds 2+ capabilities; demonstrates iteration (before/after comparison); addresses a real evaluation gap |
| **Exceptional** | 95-100 | Novel approach; production-quality output (CI-ready, dashboarded, or automated); teaches the reviewer something new |

### Assessment Criteria

| Criterion | Weight | Description |
|---|---|---|
| Complete workflow execution | 25% | All stages implemented and runnable; produces valid output |
| Beyond-notebook features | 25% | Number and quality of capabilities not covered in source notebook |
| Justification & analysis | 20% | Why each metric/feature was chosen; what evaluation gap it addresses |
| Iteration evidence | 15% | Before/after comparison showing the pipeline caught or improved something |
| "What was left out" | 10% | Identifies limitations; names what they'd need to cover them |
| Code quality & documentation | 5% | Readable, commented, reproducible |

### Tips

1. **Start with the notebook** — get it running, then extend one piece at a time.
2. **Define your "beyond" early** — decide what you're adding before you start coding.
3. **Document as you go** — capture why you chose each metric and what gap it fills.
4. **Show iteration** — run your eval, change something, re-run, and compare results. This is the strongest signal of understanding.
5. **Name your limitations** — the rubric rewards honesty about what's missing.

---

## Wrap-Up

You built a complete DSPy optimization pipeline: declared a typed signature, measured quality with a percentage-error metric, ran `BootstrapFewShot` to automatically select few-shot demonstrations, inspected the results, composed a module with `ChainOfThought`, and enhanced the metric with an LLM-as-judge.

Key takeaways:
- DSPy's optimization loop IS an evaluation loop—every candidate prompt is scored and compared
- The metric function is the most important code you write; a broken metric produces broken optimization
- Optimized programs are portable JSON artifacts you can version, deploy, and roll back
- `ChainOfThought` and custom modules compose with the same optimizer—no new optimization code needed
- Enhanced metrics (numeric + judge) catch problems that single-dimension scores miss

**Ready for more?** Take on the [Module 05 Deep-Dive Challenge](CHALLENGE-deep-dive.md) to combine DSPy optimization with advanced evaluation patterns—try `MIPROv2` for instruction optimization, build multi-stage pipelines, or add evaluation gates to a CI/CD workflow.

**Feedback:** What worked well in this skill? What was confusing? Share with your instructor or post in the workshop channel.

**Next steps:** Explore the other framework evaluations in Module 05 to see how agent-focused tools (Strands, LangGraph) and config-driven tools (promptfoo) compare to DSPy's optimization-as-evaluation approach.
