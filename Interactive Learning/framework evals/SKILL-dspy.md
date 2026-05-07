---
name: DSPy Prompt Optimization
description: Evaluate and optimize LLM prompts programmatically using DSPy where the optimization loop IS the evaluation framework
---

# DSPy Prompt Optimization

DSPy replaces hand-written prompts with typed declarations and automatic optimization. You define what good looks like (a metric), and the optimizer finds the prompt that maximizes it. The evaluation IS the optimization — no manual prompt engineering required.

## Prerequisites

- AWS account with Bedrock model access (Claude Haiku)
- Python 3.10+ with `boto3`
- Familiarity with LLM API calls (Module 01)

## Learning Objectives

- Declare typed Signatures that replace hand-written prompt strings
- Write metric functions that drive automatic optimization
- Run `BootstrapFewShot` to generate optimized few-shot prompts
- Inspect optimizer output to understand what changed
- Compose multi-step Modules with LLM-as-judge metrics

## Does NOT Cover

- DSPy assertions or typed constraints
- Multi-hop retrieval pipelines
- Production deployment patterns
- Custom optimizer implementations beyond BootstrapFewShot

## Setup

```bash
pip install "dspy>=2.5" boto3 pandas
```

```python
import dspy
import re

# Configure Bedrock LM
lm = dspy.LM("bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0")
dspy.configure(lm=lm)

# Helper: extract numeric value from text
def extract_number(text):
    """Extract first number (int or float) from a string."""
    match = re.search(r"[\d,]+\.?\d*", str(text).replace(",", ""))
    return float(match.group()) if match else None

# Sample dataset: US city populations
data = [
    ("What is the population of Seattle, WA?", "749256"),
    ("What is the population of Austin, TX?", "964177"),
    ("What is the population of Denver, CO?", "713252"),
    ("What is the population of Portland, OR?", "652503"),
    ("What is the population of Nashville, TN?", "689447"),
    ("What is the population of Boston, MA?", "675647"),
    ("What is the population of Miami, FL?", "442241"),
    ("What is the population of Atlanta, GA?", "498715"),
]

examples = [dspy.Example(question=q, answer=a).with_inputs("question") for q, a in data]
trainset, testset = examples[:5], examples[5:]
```

---

## Section 1: Signatures + Baseline

**Concept:** A DSPy Signature declares inputs and outputs as typed fields — the docstring becomes the instruction. Before optimizing, establish a baseline score with `dspy.Predict` (zero-shot, no examples). This is the number you need to beat.

**Build:**

```python
class CityQA(dspy.Signature):
    """Answer factual questions about US cities using your knowledge."""
    question: str = dspy.InputField(desc="A question about a US city's population")
    answer: str = dspy.OutputField(desc="A precise numeric answer")

baseline = dspy.Predict(CityQA)

def city_metric(example, prediction, trace=None):
    """Within 5% → 1.0, within 10% → 0.5, else → 0.0."""
    expected = extract_number(example.answer)
    predicted = extract_number(prediction.answer)
    if expected is None or predicted is None or expected == 0:
        return False if trace else 0.0
    pct_error = abs(predicted - expected) / expected
    score = 1.0 if pct_error <= 0.05 else (0.5 if pct_error <= 0.10 else 0.0)
    return score >= 0.5 if trace else score

evaluator = dspy.Evaluate(devset=testset, metric=city_metric, num_threads=1, display_progress=True)
baseline_score = float(evaluator(baseline))
print(f"Baseline: {baseline_score:.2f}")
```

---

## Section 2: Metrics + Optimization

**Concept:** The metric is the most important piece — the optimizer maximizes whatever you measure. `BootstrapFewShot` runs your model on training examples, scores each with your metric, and keeps winners as few-shot demonstrations. The optimizer writes the prompt for you.

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

## Section 3: Inspection + Chain-of-Thought

**Concept:** After optimization, inspect what changed — which demos were selected, the full prompt sent to the model. No black boxes. Then swap `Predict` for `ChainOfThought` to add visible step-by-step reasoning (one-line change, same optimizer works).

**Build:**

```python
# Inspect optimizer output
print(f"Demos selected: {len(optimized.demos)}")
for i, demo in enumerate(optimized.demos):
    print(f"  {i+1}: Q={demo['question'][:40]}... A={demo['answer']}")

# View actual prompt
_ = optimized(question=testset[0].question)
dspy.inspect_history(n=1)

# Save as artifact
optimized.save("optimized_city_qa.json")

# Chain-of-Thought: adds reasoning field
cot = dspy.ChainOfThought(CityQA)
result = cot(question="What is the population of Seattle, WA?")
print(f"Reasoning: {result.reasoning}")
print(f"Answer:    {result.answer}")

cot_score = float(evaluator(cot))
print(f"CoT: {cot_score:.2f} vs Baseline: {baseline_score:.2f}")
```

---

## Section 4: Modules + Composition

**Concept:** A DSPy Module wraps multiple steps into a reusable pipeline. Define components in `__init__`, orchestrate in `forward()`. The same optimizer that improved `Predict` optimizes entire modules end-to-end — all steps jointly.

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
    metric=city_metric, max_bootstrapped_demos=3, max_labeled_demos=3,
)
optimized_module = module_optimizer.compile(module, trainset=trainset)
print(f"Optimized Module: {float(evaluator(optimized_module)):.2f}")
```

---

## Section 5: Advanced Metrics + Evaluation Loop

**Concept:** Numeric metrics miss qualitative failures. Add an LLM-as-judge inside your metric — the optimizer then learns to avoid hallucinations, not just numeric errors. This combines Module 02's LLM-as-Judge with DSPy's optimization-as-evaluation paradigm.

**Build:**

```python
class FaithfulnessCheck(dspy.Signature):
    """Check if the answer is factually faithful and not hallucinated."""
    question: str = dspy.InputField()
    answer: str = dspy.InputField()
    expected_answer: str = dspy.InputField()
    is_faithful: bool = dspy.OutputField(desc="True if factually consistent")
    reason: str = dspy.OutputField(desc="Brief explanation")

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
    return combined >= 0.5 if trace else combined

# Full evaluation loop with enhanced metric
enhanced_evaluator = dspy.Evaluate(devset=testset, metric=enhanced_metric, num_threads=1)
final_optimizer = dspy.BootstrapFewShot(metric=enhanced_metric, max_bootstrapped_demos=4)
final_optimized = final_optimizer.compile(CityExpert(), trainset=trainset)
print(f"Enhanced optimized: {float(enhanced_evaluator(final_optimized)):.2f}")
```

---

## Challenges

Build a custom metric for a **new domain** (e.g., date extraction, unit conversion, or sentiment scoring) and run `BootstrapFewShot` to beat your baseline by ≥10%.

**Assessment criteria:**

- Defines a typed Signature with domain-appropriate input/output fields
- Implements a metric returning bool (optimization) and float (evaluation)
- Runs BootstrapFewShot and demonstrates ≥10% improvement over unoptimized baseline
- Inspects optimizer output (demos selected, prompt history, saved artifact)
- Explains why DSPy's optimization loop IS an evaluation framework

## Wrap-Up

| Approach | What It Does | When to Use |
|----------|-------------|-------------|
| `Predict` | Zero-shot baseline | Starting point |
| `BootstrapFewShot` | Auto-selects few-shot demos | First optimization pass |
| `ChainOfThought` | Adds visible reasoning | Need debuggability |
| Module + Optimize | End-to-end pipeline | Multi-step tasks |
| Enhanced Metric | LLM judge in the loop | Catch qualitative failures |

**Key insight:** In DSPy, you never hand-write prompts. You define what good looks like (metric), and the framework finds the prompt that maximizes it.

**Next:** See [CHALLENGE-deep-dive.md](./CHALLENGE-deep-dive.md) to extend this pipeline with custom optimizers and multi-stage evaluation.
