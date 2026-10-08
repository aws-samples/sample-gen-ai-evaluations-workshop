"""Readable notebook output for Strands Evals reports."""

from __future__ import annotations

import json
from html import escape
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strands_evals import EvaluationReport


def _text(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def _score(value: float | None) -> str:
    return "Not reported" if value is None else f"{value:.2f}"


def evaluation_report_html(report: EvaluationReport, *, expand_details: bool = True) -> str:
    """Render SDK scores and full explanations without console truncation.

    Agent responses, tool results, and evaluator text are escaped before being
    included in HTML. This function only formats an existing report.
    """
    styles = """
    <style>
      .strands-eval-report { color: inherit; font: inherit; line-height: 1.5; }
      .strands-eval-report table {
        border-collapse: collapse; width: 100%; margin: 0.75rem 0 1rem;
      }
      .strands-eval-report th, .strands-eval-report td {
        text-align: left; vertical-align: top; padding: 0.5rem 0.75rem;
        border-bottom: 1px solid currentColor; overflow-wrap: anywhere;
      }
      .strands-eval-report details {
        border: 1px solid currentColor; border-radius: 6px;
        padding: 0.6rem 0.8rem; margin: 0.75rem 0;
      }
      .strands-eval-report summary { cursor: pointer; overflow-wrap: anywhere; }
      .strands-eval-report .eval-text {
        white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.4rem 0 1rem;
      }
      .strands-eval-report pre {
        white-space: pre-wrap; overflow-wrap: anywhere; font-size: 0.9em;
        max-height: 28rem; overflow: auto; padding: 0.5rem;
      }
    </style>
    """
    parts = [styles, '<section class="strands-eval-report">', "<h3>Evaluation results</h3>"]
    if not report.cases:
        parts.append("<p>No evaluation results were returned.</p></section>")
        return "".join(parts)

    parts.append(
        f"<p><strong>Overall score:</strong> {_score(report.overall_score)}"
        f" &middot; <strong>Case/evaluator results:</strong> {len(report.cases)}</p>"
    )
    parts.append(
        "<table><thead><tr>"
        '<th scope="col">Case</th><th scope="col">Evaluator</th>'
        '<th scope="col">Score</th><th scope="col">Outcome</th>'
        "</tr></thead><tbody>"
    )
    rows = []
    for index, case in enumerate(report.cases):
        name = str(case.get("name", f"Case {index + 1}"))
        evaluator = str(case.get("evaluator", "Not reported"))
        score = _score(report.scores[index] if index < len(report.scores) else None)
        passed = report.test_passes[index] if index < len(report.test_passes) else None
        outcome = "Pass" if passed is True else "Fail" if passed is False else "Not reported"
        reason = report.reasons[index] if index < len(report.reasons) else ""
        rows.append((case, name, evaluator, score, outcome, reason))
        parts.append(
            "<tr>"
            + "".join(f"<td>{escape(value)}</td>" for value in (name, evaluator, score, outcome))
            + "</tr>"
        )
    parts.append(
        "</tbody></table>"
        "<p>Scores and pass/fail decisions come from the evaluator. "
        "Read the explanations below to understand each result.</p>"
    )

    evidence_fields = [
        ("expected_output", "Expected response"),
        ("actual_output", "Actual response"),
        ("expected_trajectory", "Expected tool sequence"),
        ("actual_trajectory", "Recorded tool calls"),
    ]
    for case, name, evaluator, score, outcome, reason in rows:
        opened = " open" if expand_details else ""
        parts.append(
            f"<details{opened}><summary><strong>{escape(name)}</strong>"
            f" &middot; {escape(evaluator)} &middot; {escape(outcome)}"
            f" &middot; Score {score}</summary>"
            f'<p><strong>Explanation</strong></p><div class="eval-text">'
            f"{escape(reason or 'No explanation was returned.')}</div>"
        )
        if "input" in case:
            parts.append(
                '<p><strong>Input</strong></p><div class="eval-text">'
                f"{escape(_text(case['input']))}</div>"
            )
        for key, label in evidence_fields:
            if key in case and case[key] is not None:
                parts.append(
                    f"<details><summary>{label}</summary>"
                    f"<pre>{escape(_text(case[key]))}</pre></details>"
                )
        parts.append("</details>")
    parts.append("</section>")
    return "".join(parts)


def display_evaluation_report(report: EvaluationReport, *, expand_details: bool = True) -> None:
    """Display an existing report in Jupyter without rerunning evaluations."""
    from IPython.display import HTML, display

    display(HTML(evaluation_report_html(report, expand_details=expand_details)))
