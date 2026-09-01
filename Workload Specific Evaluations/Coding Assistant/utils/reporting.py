"""Pandas aggregation + display helpers for the eval notebooks.

Two scorecards, one per evaluation axis:

  - autonomous_summary: per-agent pass rates across the four signals
    (review, tests, static, tool-call), reliability across seeds, and
    wall-clock efficiency. Tokens shown for the custom agent only.

  - pair_programmer_summary: per-agent IR metrics (precision@5,
    recall@10, MRR), answer correctness, citation grounding, and
    honesty on trap tasks.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _to_record(d: Any) -> Dict[str, Any]:
    if is_dataclass(d):
        return asdict(d)
    if isinstance(d, dict):
        return d
    raise TypeError(f"Cannot convert {type(d)} to record")


# ---------------------------------------------------------------------------
# Autonomous axis
# ---------------------------------------------------------------------------

def build_results_frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Autonomous results frame, one row per (agent, task, seed).

    Expected row keys include: agent, task_id, seed, review_pass,
    tests_pass, static_pass, tools_pass, sequence_pass, elapsed_s,
    tool_call_count, input_tokens, output_tokens, error.
    """
    df = pd.DataFrame([_to_record(r) for r in rows])
    if df.empty:
        return df
    signal_cols = [c for c in ["review_pass", "tests_pass", "static_pass", "tools_pass"] if c in df.columns]
    if signal_cols:
        df["overall_pass"] = df[signal_cols].all(axis=1)
    return df


def per_agent_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    agg: Dict[str, Any] = {"n_runs": ("task_id", "count")}
    for col in ("review_pass", "tests_pass", "static_pass", "tools_pass",
                "sequence_pass", "overall_pass"):
        if col in df.columns:
            agg[col.replace("_pass", "_rate")] = (col, "mean")
    if "elapsed_s" in df.columns:
        agg["avg_elapsed_s"] = ("elapsed_s", "mean")
    if "tool_call_count" in df.columns:
        agg["total_tool_calls"] = ("tool_call_count", "sum")
    if "input_tokens" in df.columns:
        agg["avg_input_tokens"] = ("input_tokens", "mean")
    if "output_tokens" in df.columns:
        agg["avg_output_tokens"] = ("output_tokens", "mean")
    return df.groupby("agent").agg(**agg).round(3)


def per_task_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    agg: Dict[str, Any] = {"n_runs": ("agent", "count")}
    for col in ("overall_pass", "review_pass", "tests_pass", "tools_pass"):
        if col in df.columns:
            agg[col.replace("_pass", "_rate")] = (col, "mean")
    out = df.groupby("task_id").agg(**agg).round(3)
    if "overall_rate" in out.columns:
        out = out.sort_values("overall_rate")
    return out


def reliability_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per (agent, task) pass-rate across seeds.

    Only meaningful for tasks that were run with multiple seeds; tasks
    with one seed will show n_seeds=1 and pass_rate ∈ {0, 1}. Useful
    table to spot stochastic flakiness.
    """
    if df.empty or "seed" not in df.columns:
        return pd.DataFrame()
    grouped = df.groupby(["agent", "task_id"]).agg(
        n_seeds=("seed", "nunique"),
        pass_rate=("overall_pass", "mean"),
    ).round(3)
    return grouped[grouped["n_seeds"] > 1].sort_values(["agent", "pass_rate"])


def efficiency_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Wall-clock per task and per correct task. Uniform across all 3 agents.

    Tokens are added as separate columns and will be NaN for agents that
    don't expose them (claude_code, kiro). Don't compare $/task across
    agents — wall-clock is the only uniform signal.
    """
    if df.empty:
        return df
    rows = []
    for agent, sub in df.groupby("agent"):
        passed = sub[sub.get("overall_pass", False)] if "overall_pass" in sub.columns else sub.iloc[0:0]
        seconds_per_task = sub["elapsed_s"].mean() if "elapsed_s" in sub.columns else float("nan")
        seconds_per_correct = (
            passed["elapsed_s"].mean() if not passed.empty else float("nan")
        )
        in_tok = sub["input_tokens"].mean() if "input_tokens" in sub.columns else float("nan")
        out_tok = sub["output_tokens"].mean() if "output_tokens" in sub.columns else float("nan")
        rows.append({
            "agent": agent,
            "seconds_per_task": round(seconds_per_task, 1) if pd.notna(seconds_per_task) else float("nan"),
            "seconds_per_correct_task": round(seconds_per_correct, 1) if pd.notna(seconds_per_correct) else float("nan"),
            "avg_input_tokens": round(in_tok, 0) if pd.notna(in_tok) else float("nan"),
            "avg_output_tokens": round(out_tok, 0) if pd.notna(out_tok) else float("nan"),
        })
    out = pd.DataFrame(rows).set_index("agent")
    return out


def failure_modes(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    fails = {}
    for col, label in [
        ("review_pass", "review_fail"),
        ("tests_pass", "tests_fail"),
        ("static_pass", "static_fail"),
        ("tools_pass", "tools_fail"),
    ]:
        if col in df.columns:
            fails[label] = (~df[col]).astype(int)
    if not fails:
        return pd.DataFrame()
    fails["agent"] = df["agent"]
    return pd.DataFrame(fails).groupby("agent").sum()


# ---------------------------------------------------------------------------
# Pair-programmer axis
# ---------------------------------------------------------------------------

def build_pair_programmer_frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """One row per (agent, task, question).

    Expected row keys: agent, task_id, question, precision_at_5,
    recall_at_10, mrr, answer_correct, citation_grounded, citations_found,
    citations_valid, is_trap, honesty_pass.
    """
    df = pd.DataFrame([_to_record(r) for r in rows])
    return df


def pair_programmer_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-agent rollup of IR + correctness + grounding + honesty."""
    if df.empty:
        return df
    agg: Dict[str, Any] = {"n_questions": ("question", "count")}
    for col, label in [
        ("precision_at_5", "precision_at_5"),
        ("recall_at_10", "recall_at_10"),
        ("mrr", "mrr"),
    ]:
        if col in df.columns:
            agg[label] = (col, "mean")
    for col, label in [
        ("answer_correct", "answer_accuracy"),
        ("citation_grounded", "citation_grounded_rate"),
        ("honesty_pass", "honesty_rate"),
    ]:
        if col in df.columns:
            agg[label] = (col, "mean")
    return df.groupby("agent").agg(**agg).round(3)
