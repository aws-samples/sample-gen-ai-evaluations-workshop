"""
The workload under evaluation: BANKING77 intent classification.

BANKING77 (PolyAI, CC-BY-4.0) is 13,083 real online-banking customer service
queries, each labelled with one of 77 fine-grained intents. A sampled subset is
vendored under ``data/`` so the notebook runs without a network download.

    Casanueva et al., "Efficient Intent Detection with Dual Sentence Encoders"
    https://github.com/PolyAI-LDN/task-specific-datasets
    Licensed CC-BY-4.0.

Intent triage is a good stand-in for the workloads this module is about: very
high volume, short inputs, short outputs, objective ground truth, and 77
confusable classes that genuinely separate strong models from weak ones.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "data")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

with open(os.path.join(_DATA, "banking77_labels.json")) as _f:
    LABELS: List[str] = json.load(_f)

LABEL_BLOCK = "\n".join(LABELS)

# BANKING77 quirk: 'Refund_not_showing_up' is the only label carrying an
# uppercase character, so every comparison has to be case-insensitive. Miss
# this and all five models look like they failed that class.
_CANON: Dict[str, str] = {label.lower(): label for label in LABELS}

# Three labels are substrings of other labels:
#   exchange_rate      -> card_payment_wrong_exchange_rate,
#                         wrong_exchange_rate_for_cash_withdrawal
#   card_not_working   -> virtual_card_not_working
# So any substring search has to prefer the longest match.


def load_jsonl(name: str) -> List[dict]:
    path = os.path.join(_DATA, name)
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_eval_set() -> List[dict]:
    """154 held-out items, 2 per class, from the official BANKING77 test split."""
    return load_jsonl("banking77_eval.jsonl")


def load_optimizer_samples() -> List[dict]:
    """77 items, 1 per class, drawn from the official TRAIN split.

    Kept separate from the evaluation set so that any gain measured after
    optimization is measured on data the optimizer never saw.
    """
    return load_jsonl("banking77_optimizer_samples.jsonl")


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
# Templates use Advanced Prompt Optimization's {{variable}} placeholder syntax
# so that the exact same string can be submitted to an optimization job and
# rendered locally for evaluation.

# The baseline: a deliberately ordinary first attempt. Describe the job, list the
# intents, ask for the answer. No output-format contract, which is exactly the
# weakness the evaluation exposes.
BASELINE_TEMPLATE = f"""You are a customer service assistant for a digital bank.
Classify the customer's message into one of the following intents:

{LABEL_BLOCK}

Customer message: {{{{customerMessage}}}}

What is the intent?"""

# --- Variants submitted to Advanced Prompt Optimization ---------------------
#
# AdvPO rewrites the whole template by default. Two `<advpo:...>` tags scope that
# rewrite, and their interaction is worth knowing about:
#
#   <advpo:optimize>  marks the ONLY regions the optimizer may rewrite
#   <advpo:exclude>   marks regions the optimizer must leave alone
#
# Wrapping the 77-label vocabulary in <advpo:exclude> and providing no
# <advpo:optimize> region produced NO optimization at all: every model came back
# with a template byte-identical to the input apart from the tags being stripped.
# If you want to protect one section, mark what IS optimizable with
# <advpo:optimize> rather than only marking what is not.
#
# Variant 1: no tags. The optimizer may rewrite anything, including how the label
# vocabulary is presented. The exact-match metric protects the label strings
# themselves, because mangling them tanks the score.
ADVPO_TEMPLATE_FREE = BASELINE_TEMPLATE

# Variant 2: the instruction text is explicitly marked optimizable, while the
# label vocabulary and the {{customerMessage}} placeholder sit outside any
# optimize region and are therefore preserved verbatim.
ADVPO_TEMPLATE_SCOPED = f"""<advpo:optimize>You are a customer service assistant for a digital bank.
Classify the customer's message into one of the intents listed below.
Give the intent as your answer.</advpo:optimize>

{LABEL_BLOCK}

Customer message: {{{{customerMessage}}}}"""


_ADVPO_TAG = re.compile(r"</?advpo:(?:optimize|exclude)>\s*")


def render_prompt(template: str, message: str) -> str:
    """Turn a template into a concrete prompt.

    Strips the ``<advpo:...>`` control tags, which are instructions to the
    optimizer and must not be sent to the model, then substitutes the message.
    """
    text = _ADVPO_TAG.sub("", template)
    return text.replace("{{customerMessage}}", message)


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def _squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


def normalize_label(raw: str, strategy: str = "longest") -> Optional[str]:
    """Map a free-form model response onto a canonical BANKING77 label.

    1. If the whole response is a label, use it. This is the happy path and the
       only path a well-constrained prompt should need.
    2. Otherwise fall back to searching for a label inside the response:
         ``longest`` picks the longest matching label (disambiguates
             card_not_working vs virtual_card_not_working)
         ``first``   picks the earliest-mentioned label
    3. Otherwise return None, meaning unparseable.

    Step 2 is where evaluations quietly go wrong. A chatty model that answers
    correctly and then discusses three other intents can be scored against any
    of them depending on which fallback you chose. The notebook measures how
    much this single decision moves the numbers.
    """
    if raw is None:
        return None
    squashed = _squash(raw)
    if squashed in _CANON:
        return _CANON[squashed]

    hits = [low for low in _CANON if low in squashed]
    if not hits:
        return None
    if strategy == "longest":
        return _CANON[max(hits, key=len)]
    if strategy == "first":
        best = min(hits, key=lambda low: (squashed.find(low), -len(low)))
        return _CANON[best]
    raise ValueError(f"unknown strategy {strategy!r}")


def is_correct(raw: str, gold: str, strategy: str = "longest") -> bool:
    return normalize_label(raw, strategy) == gold


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def summarize(records, model_id: Optional[str] = None,
              strategy: str = "longest") -> dict:
    """Aggregate a list of InvocationRecords into the module's metrics.

    ``records`` items need ``.ok``, ``.text``, ``.input_tokens``,
    ``.output_tokens``, ``.latency_s``, ``.queue_wait_s``, ``.fallback_depth``
    and ``.meta['item']['label']``.
    """
    from model_config import calculate_cost

    total = len(records)
    ok = [r for r in records if r is not None and r.ok]
    errors = total - len(ok)

    correct = unparsed = 0
    cost = 0.0
    in_toks = out_toks = 0
    lat: List[float] = []
    waits: List[float] = []
    depths: Dict[int, int] = {}
    per_model: Dict[str, int] = {}
    confusions: Dict[tuple, int] = {}

    for r in ok:
        gold = r.meta["item"]["label"]
        pred = normalize_label(r.text, strategy)
        if pred is None:
            unparsed += 1
        if pred == gold:
            correct += 1
        elif pred is not None:
            key = (gold, pred)
            confusions[key] = confusions.get(key, 0) + 1
        mid = model_id or r.model
        cost += calculate_cost(mid, r.input_tokens, r.output_tokens)
        in_toks += r.input_tokens
        out_toks += r.output_tokens
        lat.append(r.latency_s)
        waits.append(r.queue_wait_s)
        depths[r.fallback_depth] = depths.get(r.fallback_depth, 0) + 1
        per_model[r.model] = per_model.get(r.model, 0) + 1

    n = len(ok) or 1
    return {
        "n_items": total,
        "n_scored": len(ok),
        "errors": errors,
        "correct": correct,
        "accuracy": correct / n,
        "unparsed": unparsed,
        "unparsed_rate": unparsed / n,
        "total_cost_usd": cost,
        "cost_per_1k_usd": (cost / n) * 1000,
        "avg_input_tokens": in_toks / n,
        "avg_output_tokens": out_toks / n,
        "latency_p50_s": _percentile(lat, 0.50),
        "latency_p95_s": _percentile(lat, 0.95),
        "queue_wait_mean_s": sum(waits) / n,
        "queue_wait_p95_s": _percentile(waits, 0.95),
        "fallback_depths": dict(sorted(depths.items())),
        "per_model_counts": per_model,
        "top_confusions": sorted(confusions.items(), key=lambda kv: -kv[1])[:10],
    }
