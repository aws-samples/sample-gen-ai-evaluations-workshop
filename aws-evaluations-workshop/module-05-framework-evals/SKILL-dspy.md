---
name: DSPy Prompt Optimization
description: Evaluate and optimize LLM prompts programmatically using DSPy where the optimization loop IS the evaluation framework
---

# DSPy Prompt Optimization

## What You Will Learn

DSPy is an **eval-focused** framework — unlike tools that run evaluations after the fact, DSPy's optimization loop IS the evaluation. You define a metric, and the optimizer uses that metric to automatically find better prompts. No manual prompt engineering required.

In this module you will:
- Declare typed signatures instead of editing prompt strings
- Write evaluation metrics that drive automatic optimization
- Run `BootstrapFewShot` to let the machine write better prompts
- Inspect what the optimizer changed (no black boxes)
- Add reasoning with `ChainOfThought`
- Compose multi-step modules and optimize end-to-end
- Enhance metrics with LLM-as-judge faithfulness checks

## Prerequisites

- AWS account with Bedrock model access (Claude Haiku)
- Python 3.10+ with `dspy>=3.1`, `boto3`, `pandas`
- Familiarity with LLM API calls (Module 01)

## Does NOT Cover

- DSPy assertions or typed constraints
- Multi-hop retrieval pipelines
- Production deployment patterns
- Custom teleprompter implementations beyond BootstrapFewShot

---

## Section 1: Signatures — Declare, Don't Prompt

**Concept:** A DSPy Signature replaces hand-written prompt strings with a typed declaration of inputs and outputs. The docstring becomes the instruction; field descriptions guide output format. You stop editing strings and start declaring intent.

**Build:**

```python
import dspy

lm = dspy.LM("bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0")
dspy.configure(lm=lm)

class CityQA(dspy.Signature):
    """Answer factual questions about US cities using your knowledge."""
    question: str = dspy.InputField(
        desc="A question about a US city's population or demographics"
    )
    answer: str = dspy.OutputField(
        desc="A precise numeric answer to the question"
    )
```

---

## Section 2: Baseline — Measure Before You Optimize

**Concept:** Before optimizing anything, establish a baseline score. `dspy.Predict` runs your signature with zero prompt engineering — no examples, no tricks. This score is what you need to beat. Find the *errors*, not just the score.

**Build:**

```python
baseline = dspy.Predict(CityQA)

# Run baseline on test set
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

---

## Section 3: Metrics — Define What Good Looks Like

**Concept:** The metric is the most important piece in DSPy — the optimizer maximizes whatever you measure. During optimization (`trace is not None`), return a boolean so the optimizer knows pass/fail. During evaluation, return a float for granularity. A broken metric means the optimizer exploits the wrong signal.

**Build:**

```python
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

    # Optimization needs boolean; evaluation needs float
    if trace is not None:
        return score >= 0.5
    return score
```

---

## Section 4: Optimization — Let the Machine Write the Prompt

**Concept:** `BootstrapFewShot` runs your model on training examples, scores each output with your metric, and keeps the winners as few-shot demonstrations. The optimizer writes the prompt for you — using your metric as the objective function. This is where evaluation and optimization become the same loop.

**Build:**

```python
optimizer = dspy.BootstrapFewShot(
    metric=city_metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
    max_rounds=2,
)

optimized = optimizer.compile(baseline, trainset=trainset)

optimized_score = float(evaluator(optimized))
print(f"Baseline:  {baseline_score:.2f}")
print(f"Optimized: {optimized_score:.2f}")
print(f"Improvement: {optimized_score - baseline_score:+.2f}")
```

---

## Section 5: Inspection — No Black Boxes

**Concept:** After optimization, inspect what changed. The optimizer selects specific question/answer pairs as few-shot demonstrations. You can see exactly which demos it picked, view the full prompt via `dspy.inspect_history()`, and save the optimized program as a portable JSON artifact.

**Build:**

```python
# See what demos the optimizer selected
print(f"Number of demos: {len(optimized.demos)}")
for i, demo in enumerate(optimized.demos):
    print(f"Demo {i+1}: Q={demo['question'][:50]}... A={demo['answer']}")

# View the actual prompt sent to the model
_ = optimized(question=testset[0].question)
dspy.inspect_history(n=1)

# Save as portable artifact
optimized.save("optimized_city_qa.json")
```

---

## Section 6: ChainOfThought — Add Reasoning

**Concept:** `ChainOfThought` adds a `reasoning` field — the model thinks step-by-step before answering. One-line change from `Predict`. The reasoning is visible and debuggable, and the same optimizer works on CoT modules. More tokens, more cost, but better debuggability.

**Build:**

```python
cot = dspy.ChainOfThought(CityQA)

result = cot(question="What is the population of Seattle, WA?")
print(f"Reasoning: {result.reasoning}")
print(f"Answer:    {result.answer}")

cot_score = float(evaluator(cot))
print(f"ChainOfThought: {cot_score:.2f} vs Baseline: {baseline_score:.2f}")
```

---

## Section 7: Modules — Compose and Optimize End-to-End

**Concept:** A DSPy Module wraps multiple steps into a reusable building block. Define ingredients in `__init__`, define steps in `forward()`. The same optimizer that improved `Predict` works on modules — optimizing the entire pipeline as one unit.

**Build:**

```python
class CityExpert(dspy.Module):
    def __init__(self):
        super().__init__()
        self.answer = dspy.ChainOfThought(CityQA)

    def forward(self, question):
        return self.answer(question=question)

module = CityExpert()
module_optimizer = dspy.BootstrapFewShot(
    metric=city_metric,
    max_bootstrapped_demos=3,
    max_labeled_demos=3,
)
optimized_module = module_optimizer.compile(module, trainset=trainset)
optimized_module_score = float(evaluator(optimized_module))
print(f"Optimized Module: {optimized_module_score:.2f}")
```

---

## Section 8: Enhanced Metrics — LLM-as-Judge Inside the Loop

**Concept:** Numeric metrics miss qualitative failures (hallucinated context, wrong format). Add an LLM-as-judge as a second signal inside your metric. The judge lives inside the optimization loop — so the optimizer learns to avoid hallucinations, not just numeric errors. This combines Module 02's LLM-as-Judge concept with DSPy's optimization-as-evaluation paradigm.

**Build:**

```python
class FaithfulnessCheck(dspy.Signature):
    """Check if the answer is factually faithful and not hallucinated."""
    question: str = dspy.InputField()
    answer: str = dspy.InputField()
    expected_answer: str = dspy.InputField()
    is_faithful: bool = dspy.OutputField(
        desc="True if factually consistent with expected answer"
    )
    reason: str = dspy.OutputField(desc="Brief explanation of the judgment")

faithfulness_judge = dspy.Predict(FaithfulnessCheck)

def enhanced_metric(example, prediction, trace=None):
    """Numeric accuracy (70%) + LLM faithfulness judge (30%)."""
    numeric_score = city_metric(example, prediction)
    judgment = faithfulness_judge(
        question=example.question,
        answer=prediction.answer,
        expected_answer=example.answer,
    )
    faith_score = 1.0 if judgment.is_faithful else 0.0
    combined = 0.7 * numeric_score + 0.3 * faith_score
    if trace is not None:
        return combined >= 0.5
    return combined
```

---

## Wrap-Up

You built an evaluation-driven optimization pipeline:

| Approach | What It Does | When to Use |
|----------|-------------|-------------|
| `Predict` | Zero-shot baseline | Starting point |
| `BootstrapFewShot` | Auto-selects few-shot demos | First optimization pass |
| `ChainOfThought` | Adds visible reasoning | Need debuggability |
| Module + Optimize | End-to-end pipeline optimization | Multi-step tasks |
| Enhanced Metric | LLM judge inside the loop | Catch qualitative failures |

**Key insight:** In DSPy, you never hand-write prompts. You define what good looks like (metric), and the framework finds the prompt that maximizes it. The evaluation IS the optimization.

**Next:** See [CHALLENGE-deep-dive.md](./CHALLENGE-deep-dive.md) to extend this pipeline with your own custom metric and optimizer configuration.

## Assessment Criteria

The learner can:
- Declare a typed Signature with input/output fields
- Write a metric function that returns bool for optimization and float for evaluation
- Run `BootstrapFewShot` and compare optimized vs baseline scores
- Inspect optimizer output (demos selected, full prompt, saved artifact)
- Explain why DSPy's optimization loop IS an evaluation framework
