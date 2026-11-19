"""Tests for skills/eval-builder/scripts/build_report.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "skills" / "eval-builder" / "scripts" / "build_report.py"
sys.path.insert(0, str(SCRIPT.parent))

import build_report  # noqa: E402

WORKLOAD = {
    "workload": "concierge",
    "eval_type": "agent",
    "checks": ["no_tag_leak", "booking_authorized", "missing_data_framing"],
    "summary": {"n": 3, "operational": {"latency_p95_s": 3.28, "total_tokens": 47557}},
    "judges": [
        {"name": "missing_data_framing", "model": "us.amazon.nova-pro-v1:0", "calibrated": False},
    ],
    "cases": [
        {"id": "q1", "input": "Population of Seattle?",
         "verdicts": {"no_tag_leak": True, "booking_authorized": True, "missing_data_framing": None}},
        {"id": "q3", "input": "Population of Boise?",
         "verdicts": {"no_tag_leak": False, "booking_authorized": True, "missing_data_framing": False},
         "reasons": {"no_tag_leak": "reply contains <thinking>"}},
        {"id": "u2", "input": "Book a tour", "verdicts": {}, "error": "ReadTimeoutError"},
    ],
}

REVIEW = {
    "summary": "Existing evals average a 1-5 multi-criteria judge.",
    "findings": [{"severity": "high", "eval": "evals/llm_judge.py", "issue": "1-5 scale, 4 criteria",
                  "evidence": "evals/llm_judge.py:11", "fix": "One binary judge per failure mode"}],
    "coverage": [{"failure_mode": "forgets order number", "covered": False, "by": "—",
                  "module": "Workload Specific Evaluations/Chatbot"}],
    "recommendations": ["Replace the averaged score with per-check pass rates"],
}


def _write(tmp_path: Path, payload: dict, name: str = "concierge.json") -> Path:
    results = tmp_path / "results"
    results.mkdir(exist_ok=True)
    (results / name).write_text(json.dumps(payload))
    return results


def test_na_verdicts_are_excluded_from_rates(tmp_path):
    [result] = build_report.load_results(_write(tmp_path, WORKLOAD))
    # missing_data_framing applied to one case (q3) and failed -> 0%, not 1/3
    assert result.per_check_pass_rate["missing_data_framing"] == 0.0
    # no_tag_leak: q1 pass, q3 fail; u2 errored and has no verdict -> 1/2
    assert result.per_check_pass_rate["no_tag_leak"] == 0.5


def test_errored_case_never_counts_as_pass(tmp_path):
    [result] = build_report.load_results(_write(tmp_path, WORKLOAD))
    overall = {c.id: c.overall_pass for c in result.cases}
    assert overall == {"q1": True, "q3": False, "u2": False}
    assert result.all_checks_pass_rate == pytest.approx(1 / 3)


def test_review_json_in_results_dir_is_not_a_workload(tmp_path):
    results = _write(tmp_path, WORKLOAD)
    (results / "review.json").write_text(json.dumps(REVIEW))
    assert [r.workload for r in build_report.load_results(results)] == ["concierge"]


def test_cli_renders_judges_operational_and_review(tmp_path):
    results = _write(tmp_path, WORKLOAD)
    review = tmp_path / "review.json"
    review.write_text(json.dumps(REVIEW))
    out = tmp_path / "report.html"

    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(results), "-o", str(out), "--review", str(review)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr

    page = out.read_text()
    assert "UNCALIBRATED" in page
    assert "latency_p95_s" in page
    assert "(n/a)" in page
    assert "ReadTimeoutError" in page
    assert "1-5 scale, 4 criteria" in page and "forgets order number" in page

    export = json.loads((tmp_path / "results.json").read_text())
    assert export["workloads"][0]["uncalibrated_judges"] == ["missing_data_framing"]


def test_cli_without_review_shows_placeholder(tmp_path):
    out = tmp_path / "report.html"
    proc = subprocess.run([sys.executable, str(SCRIPT), str(_write(tmp_path, WORKLOAD)), "-o", str(out)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "No review in this run" in out.read_text()


def test_cli_rejects_missing_dir(tmp_path):
    proc = subprocess.run([sys.executable, str(SCRIPT), str(tmp_path / "nope"), "-o", str(tmp_path / "r.html")],
                          capture_output=True, text=True)
    assert proc.returncode == 2
