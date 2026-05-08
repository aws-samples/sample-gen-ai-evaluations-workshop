---
name: mlflow-rag-generation-evaluation
description: Evaluate RAG generation quality using MLflow's GenAI evaluation framework on Amazon Bedrock. Activate when asked to "evaluate with mlflow", "set up mlflow scorers", "compare model runs", "run genai evaluate", or "track LLM evaluation metrics".
---

# MLflow RAG Generation Evaluation

Use MLflow's GenAI evaluation framework to measure how faithfully Bedrock models ground answers in retrieved context. Register 18 scorers across five categories (code-based, built-in LLM-as-Judge, Guidelines, custom `make_judge`, DeepEval), run evaluations against multiple models, and compare results side-by-side — all tracked in MLflow.

## Prerequisites
- AWS account with Bedrock model access (Claude Sonnet 4.5, Claude Haiku 4.5)
- Python 3.10+ with `mlflow>=3.8.0`, `deepeval`, `datasets`, `boto3>=1.42.4`, `litellm`, `pandas`, `nest_asyncio`, `python-dotenv`
- Source notebook: `../../Framework Specific Evaluations/MLflow/01 Mlflow Evaluation.ipynb`

## Learning Objectives
By the end of this module, you will:
- Shape a HuggingFace RAG dataset into the `{inputs, expectations}` format required by `mlflow.genai.evaluate()`
- Build code-based scorers using the `@scorer` decorator for deterministic metrics
- Create custom LLM-as-Judge scorers with `make_judge` and integrate DeepEval metrics via Bedrock
- Run multi-model evaluations and compare results using `mlflow.search_runs`

## Setup

```python
import os
import json
import logging
import boto3
from botocore.config import Config
import nest_asyncio
from dotenv import load_dotenv

nest_asyncio.apply()
load_dotenv(dotenv_path=".env", override=True)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

# AWS and Bedrock configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_PROFILE = os.getenv("AWS_PROFILE")

session_kwargs = {"region_name": AWS_REGION}
if AWS_PROFILE:
    session_kwargs["profile_name"] = AWS_PROFILE

session = boto3.Session(**session_kwargs)
region = session.region_name or "us-east-1"

bedrock = session.client(
    service_name="bedrock-runtime",
    region_name=region,
    config=Config(retries={"max_attempts": 10, "mode": "standard"}),
)

# Export credentials for LiteLLM (used by LLM-as-Judge scorers)
_creds = session.get_credentials().get_frozen_credentials()
os.environ["AWS_ACCESS_KEY_ID"] = _creds.access_key
os.environ["AWS_SECRET_ACCESS_KEY"] = _creds.secret_key
os.environ["AWS_REGION"] = region
os.environ["AWS_DEFAULT_REGION"] = region
if _creds.token:
    os.environ["AWS_SESSION_TOKEN"] = _creds.token

# Models
BEDROCK_MODELS = {
    "claude-sonnet-4-5": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "claude-haiku-4-5": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
}
JUDGE_MODEL = "bedrock:/us.anthropic.claude-sonnet-4-5-20250929-v1:0"

# MLflow tracking — local store by default
import mlflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
EXPERIMENT_NAME = "bedrock-llm-evaluation"
mlflow.set_experiment(EXPERIMENT_NAME)

print(f"Bedrock client: region={region}")
print(f"Judge model: {JUDGE_MODEL}")
print(f"MLflow tracking: {MLFLOW_TRACKING_URI}")
```

## Section 1: Dataset Preparation for MLflow GenAI Evaluate

**Concept:** `mlflow.genai.evaluate()` expects data as a list of dictionaries with `inputs` (passed to your predict function) and `expectations` (consumed by scorers for ground-truth comparison). For RAG generation evaluation, `inputs` contains the question and retrieved context, while `expectations` holds the expected answer and extracted facts. This format decouples the data shape from the scorer logic.

**Build:**

```python
import re
import pandas as pd
from datasets import load_dataset

DATASET_NAME = "explodinggradients/ragas-wikiqa"
SAMPLE_SIZE = 5


def extract_facts(text: str) -> list[str]:
    """Split a ground-truth answer into up to 5 short factual claims."""
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip()) > 10]
    return [s[:200] for s in sentences[:5]]


def normalize_context(value) -> str:
    """Coerce a dataset context field (list or ndarray) to a plain string."""
    if isinstance(value, str):
        return value
    try:
        return "\n\n".join(str(v) for v in value)
    except TypeError:
        return str(value)


ds = load_dataset(DATASET_NAME, split="train")
df = ds.to_pandas().head(SAMPLE_SIZE)

eval_data = [
    {
        "inputs": {
            "question": str(row["question"]),
            "context": normalize_context(row["context"]),
        },
        "expectations": {
            "expected_response": str(row["correct_answer"]),
            "expected_facts": extract_facts(str(row["correct_answer"])),
        },
    }
    for _, row in df.iterrows()
]

print(f"Loaded {len(eval_data)} samples")
print(f"Example keys: inputs={list(eval_data[0]['inputs'].keys())}, "
      f"expectations={list(eval_data[0]['expectations'].keys())}")
```

**Why this matters:** The `{inputs, expectations}` structure is MLflow's contract. Get it wrong and scorers silently receive empty values. The `expected_facts` list enables per-claim verification by DeepEval's Faithfulness scorer.

## Section 2: Code-Based Scorers

**Concept:** Code-based scorers are deterministic functions decorated with `@scorer`. They run instantly (no LLM calls), produce reproducible results, and provide objective baselines alongside subjective LLM-as-Judge metrics. MLflow tracks them in the same run as all other scorers.

**Build:**

```python
from mlflow.genai import scorer


@scorer
def exact_match(outputs: str, expectations: dict) -> bool:
    """Case-insensitive, whitespace-trimmed string equality."""
    return outputs.strip().lower() == expectations["expected_response"].strip().lower()


@scorer
def is_concise(outputs: str) -> bool:
    """True if the response is 100 words or fewer."""
    return len(outputs.split()) <= 100


@scorer
def word_overlap(outputs: str, expectations: dict) -> float:
    """Fraction of expected words that appear anywhere in the output."""
    output_words = set(outputs.lower().split())
    expected_words = set(expectations["expected_response"].lower().split())
    if not expected_words:
        return 0.0
    return len(output_words & expected_words) / len(expected_words)


@scorer
def response_length(outputs: str) -> int:
    """Word count of the model's response."""
    return len(outputs.split())
```

**Why this matters:** Code-based scorers catch obvious failures (verbosity, zero overlap with ground truth) before you spend judge-model tokens. They also serve as sanity checks — if `word_overlap` is 0.0 but `Faithfulness` is 1.0, something is misconfigured.

## Section 3: LLM-as-Judge Scorers

**Concept:** LLM-as-Judge scorers use a separate model (the "judge") to evaluate response quality against criteria you define. MLflow provides three mechanisms: built-in scorers (pre-built rubrics), `Guidelines` (custom pass/fail criteria), and `make_judge` (full Jinja2 prompt templates). DeepEval adds claim-decomposition scorers that verify each factual claim individually — more rigorous than single yes/no judgments.

**Build:**

```python
from mlflow.genai import make_judge
from mlflow.genai.scorers import (
    Equivalence,
    Fluency,
    Guidelines,
    RelevanceToQuery,
)
from mlflow.genai.scorers.deepeval import (
    AnswerRelevancy,
    Bias,
    ContextualRelevancy,
    Faithfulness,
    Hallucination,
    Toxicity,
)

# Custom LLM-as-Judge scorers using make_judge
factual_consistency = make_judge(
    name="factual_consistency",
    model=JUDGE_MODEL,
    instructions=(
        "Evaluate whether the response is factually consistent with the provided context. "
        "The response should not introduce facts that contradict or are absent from the context.\n\n"
        "Inputs (contains `question` and `context`): {{ inputs }}\n\n"
        "Response: {{ outputs }}"
    ),
)

professionalism = make_judge(
    name="professionalism",
    model=JUDGE_MODEL,
    instructions=(
        "Evaluate the professionalism of the response. A professional response uses formal "
        "language, avoids slang, is well-structured, and maintains a neutral tone.\n\n"
        "Response: {{ outputs }}"
    ),
)

correctness = make_judge(
    name="correctness",
    model=JUDGE_MODEL,
    instructions=(
        "Evaluate whether the response is factually correct based on the expected answer.\n\n"
        "Response: {{ outputs }}\n\n"
        "Expected answer: {{ expectations }}"
    ),
)

# DeepEval scorers — claim-decomposition metrics via Bedrock
deepeval_scorers = [
    Faithfulness(model=JUDGE_MODEL),
    AnswerRelevancy(model=JUDGE_MODEL),
    ContextualRelevancy(model=JUDGE_MODEL),
    Hallucination(model=JUDGE_MODEL),
    Toxicity(model=JUDGE_MODEL),
    Bias(model=JUDGE_MODEL),
]

# Assemble all scorers
scorers_list = [
    # MLflow built-in
    RelevanceToQuery(model=JUDGE_MODEL),
    Equivalence(model=JUDGE_MODEL),
    Fluency(model=JUDGE_MODEL),
    # Guidelines-based
    Guidelines(
        name="answer_groundedness",
        guidelines="The answer must be grounded in the provided context and not hallucinate facts.",
        model=JUDGE_MODEL,
    ),
    Guidelines(
        name="answer_completeness",
        guidelines="The answer must fully address the question without omitting key information.",
        model=JUDGE_MODEL,
    ),
    # Custom judges
    factual_consistency,
    professionalism,
    correctness,
    # DeepEval
    *deepeval_scorers,
    # Code-based
    exact_match,
    is_concise,
    word_overlap,
    response_length,
]

print(f"Total scorers registered: {len(scorers_list)}")
```

**Why this matters:** Each scorer category serves a different purpose. Code-based scorers are fast and reproducible. Built-in scorers provide standard quality metrics. `make_judge` lets you encode domain-specific criteria. DeepEval decomposes responses into claims for rigorous RAG grounding verification. Combining all four gives you a comprehensive evaluation signal.

## Section 4: Running Multi-Model Evaluations

**Concept:** `mlflow.genai.evaluate()` takes a predict function, your dataset, and the scorer list. It calls the predict function once per sample, then runs all scorers against each response. By wrapping this in `mlflow.start_run()` for each model, you get separate tracked runs that can be compared side-by-side in the MLflow UI.

**Build:**

```python
import mlflow
import mlflow.genai


def make_predict_fn(model_id: str):
    """Return a predict function closed over a specific Bedrock model id."""

    def predict_fn(question: str, context: str = "") -> str:
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer concisely."
        try:
            response = bedrock.converse(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": 512, "temperature": 0.1},
            )
            return response["output"]["message"]["content"][0]["text"]
        except Exception as e:
            return f"ERROR: {e}"

    return predict_fn


run_ids = {}

for model_key, model_id in BEDROCK_MODELS.items():
    print(f"\n=== Evaluating {model_key} ({model_id}) ===")
    with mlflow.start_run(run_name=f"eval-{model_key}") as run:
        mlflow.log_param("model_id", model_id)
        mlflow.log_param("model_key", model_key)
        mlflow.log_param("dataset", DATASET_NAME)
        mlflow.log_param("sample_size", len(eval_data))

        results = mlflow.genai.evaluate(
            data=eval_data,
            predict_fn=make_predict_fn(model_id),
            scorers=scorers_list,
        )

        run_ids[model_key] = run.info.run_id
        print(f"Run {run.info.run_id} finished. Metrics: {list(results.metrics.keys())[:5]}...")
```

**Why this matters:** The predict function's parameter names (`question`, `context`) must match the keys under `inputs` in your eval data — MLflow maps them automatically. Running each model in its own MLflow run gives you isolated tracking with full parameter/metric/artifact lineage.

## Section 5: Comparing and Analyzing Results

**Concept:** After evaluation runs complete, `mlflow.search_runs()` queries the tracking store and returns a DataFrame with all parameters and metrics. This enables programmatic model comparison without opening the UI — useful for CI pipelines, automated model selection, or generating comparison reports.

**Build:**

```python
import pandas as pd

# Log per-sample artifacts for deeper inspection
summary_df = pd.DataFrame(
    [
        {
            "question": d["inputs"]["question"],
            "ground_truth": d["expectations"]["expected_response"],
            "context": d["inputs"].get("context", "")[:200],
        }
        for d in eval_data
    ]
)
summary_df.to_csv("/tmp/eval_inputs.csv", index=False)

for model_key, run_id in run_ids.items():
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact("/tmp/eval_inputs.csv", artifact_path="eval_tables")

# Compare models side-by-side
runs = mlflow.search_runs(
    experiment_names=[EXPERIMENT_NAME],
    order_by=["start_time DESC"],
)

metric_cols = [c for c in runs.columns if c.startswith("metrics.")]
param_cols = ["params.model_key", "params.sample_size"]
display_cols = ["run_id", "status"] + [c for c in param_cols if c in runs.columns] + metric_cols

print(runs[display_cols].head(10).to_string())
```

**What to look for in results:**
- **Faithfulness + answer_groundedness** — do answers stay within the provided context?
- **Hallucination** — does the model introduce unsupported claims?
- **word_overlap vs Equivalence** — deterministic overlap vs semantic equivalence (they often disagree)
- **is_concise + response_length** — verbosity patterns across models

**Viewing the MLflow UI:**

```bash
# Local store
mlflow ui --backend-store-uri ./mlruns
# Then open http://localhost:5000
```

## Challenges

### Challenge: Build a Domain-Specific Scorer

Design and integrate a custom scorer that evaluates a quality dimension not covered by the 18 scorers in this module. For example: citation accuracy (does the response reference specific passages from context?), answer specificity (does it give concrete details vs vague generalities?), or instruction adherence (does it follow the "answer concisely" directive?).

**Requirements:**
1. Implement the scorer using either `@scorer` (code-based) or `make_judge` (LLM-as-Judge)
2. Add it to `scorers_list` and run a full evaluation with it included
3. Compare the new scorer's results against existing scorers — does it surface failures the others miss?

**Assessment criteria:**
- Produces ≥1 new scorer with a novel evaluation criterion not already in the 18-scorer list
- Scorer runs without errors inside `mlflow.genai.evaluate()` and produces non-trivial results (not all-pass or all-fail)
- Learner can articulate why their chosen criterion matters for RAG generation quality and identify ≥1 case where existing scorers would miss the failure their scorer catches

> **Deep-dive challenge:** See [CHALLENGE-deep-dive.md](./CHALLENGE-deep-dive.md) for an extended exercise combining custom scorers with automated threshold-based pass/fail gates and CI integration.

## Wrap-Up

**Key takeaways:**
- MLflow's `genai.evaluate()` is a single entry point for running any combination of code-based, built-in, custom, and third-party scorers against a predict function
- The `{inputs, expectations}` data contract decouples dataset shape from scorer logic — change one without touching the other
- `make_judge` with Jinja2 templates lets you encode any evaluation criterion as an LLM-as-Judge scorer using Bedrock as the judge
- DeepEval's claim-decomposition approach (via `Faithfulness`, `Hallucination`) is more rigorous than single-judgment scorers for RAG grounding
- `mlflow.search_runs()` enables programmatic model comparison — essential for automated model selection in production pipelines

**What this does NOT cover:**
- Retrieval evaluation (recall@k, precision@k, MRR) — this module holds retrieval constant
- SageMaker MLflow App deployment (covered in the source notebook's bootstrap cells)
- MLflow Model Registry integration
- Continuous evaluation in CI/CD pipelines

**Next:** See [CHALLENGE-deep-dive.md](./CHALLENGE-deep-dive.md) for advanced scorer design, threshold gates, and production integration patterns.
