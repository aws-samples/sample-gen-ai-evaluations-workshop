#!/usr/bin/env python3
"""Deterministic, self-contained HTML eval report generator.

Produces a polished single-file HTML report with 3 tabs:
  * Summary — dashboard cards + per-workload overview table
  * Results — expandable per-workload sections with per-test detail
  * Review — placeholder for Review+Build mode findings

Usage:
    python3 build_report.py <results_dir> -o <output.html>
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any



# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class CaseResult:
    """A single test case's evaluation result.

    A verdict of ``None`` means the check did not apply to this case (N/A).
    """

    id: str
    input_text: str
    verdicts: dict[str, bool | None]
    overall_pass: bool
    reasons: dict[str, str]
    error: str | None = None


@dataclass
class JudgeInfo:
    """Metadata about one LLM judge, including whether it is calibrated."""

    name: str
    model: str
    calibrated: bool
    tpr: float | None = None
    tnr: float | None = None
    n_labels: int | None = None


@dataclass
class WorkloadResult:
    """Results for one evaluated workload."""

    workload: str
    eval_type: str
    checks: list[str]
    cases: list[CaseResult]
    per_check_pass_rate: dict[str, float]
    all_checks_pass_rate: float | None
    n: int
    error: str | None
    source: str
    operational: dict[str, Any] = field(default_factory=dict)
    judges: list[JudgeInfo] = field(default_factory=list)
    slug: str = ""

    def __post_init__(self) -> None:
        self.slug = _slug(f"{self.workload}-{self.eval_type}")


@dataclass
class Aggregate:
    """Cross-workload summary."""

    n_workloads: int
    n_ok: int
    n_errored: int
    n_cases: int
    overall_rate: float | None
    workloads: list[WorkloadResult]



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slug(text: str) -> str:
    """Create a deterministic id-safe slug."""
    return "".join(c if c.isalnum() else "-" for c in text.lower()).strip("-")


def _esc(value: Any) -> str:
    """HTML-escape a value."""
    return html.escape("" if value is None else str(value), quote=True)


def _pct(rate: float | None) -> str:
    """Format a rate as a percentage string."""
    if rate is None:
        return "—"
    return f"{float(rate) * 100:.0f}%"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_results(results_dir: Path) -> list[WorkloadResult]:
    """Load and parse all per-workload result JSON files from a directory.

    ``review.json`` is not a workload result and is skipped here; pass it with
    ``--review`` instead.

    Args:
        results_dir: Directory containing per-workload result JSON files.

    Returns:
        Sorted list of WorkloadResult objects.

    Raises:
        FileNotFoundError: If results_dir doesn't exist.
    """
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {results_dir}")

    results: list[WorkloadResult] = []
    for path in sorted(results_dir.glob("*.json"), key=lambda p: p.name):
        if path.name == "review.json":
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            results.append(WorkloadResult(
                workload=path.stem, eval_type="unknown", checks=[], cases=[],
                per_check_pass_rate={}, all_checks_pass_rate=None, n=0,
                error=f"Failed to parse {path.name}: {exc}", source=path.name,
            ))
            continue
        results.append(_parse(raw, path.name))

    results.sort(key=lambda r: (r.workload, r.eval_type))
    return results



def _parse(raw: dict[str, Any], source: str) -> WorkloadResult:
    """Parse a raw JSON dict into a WorkloadResult."""
    summary = raw.get("summary") or {}
    cases = []
    for c in raw.get("cases") or []:
        verdicts = {k: (None if v is None else bool(v)) for k, v in (c.get("verdicts") or {}).items()}
        reasons = c.get("reasons", c.get("reason")) or {}
        if "overall_pass" in c:
            overall = bool(c["overall_pass"])
        else:  # all applicable checks pass, and at least one applied
            applied = [v for v in verdicts.values() if v is not None]
            overall = bool(applied) and all(applied) and not c.get("error")
        cases.append(CaseResult(
            id=str(c.get("id", "—")),
            input_text=str(c.get("input", "")),
            verdicts=verdicts,
            overall_pass=overall,
            reasons=dict(reasons),
            error=c.get("error"),
        ))

    checks = list(raw.get("checks") or [])
    if not checks:  # derive from the verdicts, keeping first-seen order
        for case in cases:
            checks.extend(k for k in case.verdicts if k not in checks)

    per_check = dict(summary.get("per_check_pass_rate") or {})
    for check in checks:  # compute any missing rate, excluding N/A cases
        if check not in per_check:
            applied = [case.verdicts[check] for case in cases if case.verdicts.get(check) is not None]
            per_check[check] = (sum(applied) / len(applied)) if applied else None

    all_rate = summary.get("all_checks_pass_rate")
    if all_rate is None and cases:
        all_rate = sum(c.overall_pass for c in cases) / len(cases)

    judges = [
        JudgeInfo(
            name=str(j.get("name", "—")),
            model=str(j.get("model", "—")),
            calibrated=bool(j.get("calibrated")),
            tpr=j.get("tpr"),
            tnr=j.get("tnr"),
            n_labels=j.get("n_labels"),
        )
        for j in raw.get("judges") or []
    ]

    return WorkloadResult(
        workload=str(raw.get("workload") or source),
        eval_type=str(raw.get("eval_type") or "unknown"),
        checks=checks,
        cases=cases,
        per_check_pass_rate=per_check,
        all_checks_pass_rate=all_rate,
        n=summary.get("n", len(cases)),
        error=raw.get("error"),
        source=source,
        operational=dict(summary.get("operational") or {}),
        judges=judges,
    )


def build_aggregate(results: list[WorkloadResult]) -> Aggregate:
    """Compute cross-workload summary.

    Args:
        results: All loaded workload results.

    Returns:
        Aggregate summary object.
    """
    ok = [r for r in results if not r.error]
    errored = [r for r in results if r.error]
    total_cases = sum(r.n for r in ok)
    weighted = sum(
        (r.all_checks_pass_rate or 0) * r.n for r in ok
        if r.all_checks_pass_rate is not None
    )
    overall = (weighted / total_cases) if total_cases else None
    return Aggregate(
        n_workloads=len(results), n_ok=len(ok), n_errored=len(errored),
        n_cases=total_cases, overall_rate=overall, workloads=results,
    )



# ---------------------------------------------------------------------------
# HTML rendering — styles
# ---------------------------------------------------------------------------

_CSS = """\
:root{--accent:#7c3aed;--accent2:#8b5cf6;--accent-bg:#f5f3ff;--ink:#111827;
--muted:#6b7280;--bg:#f9fafb;--card:#fff;--border:#e5e7eb;--ok:#059669;
--ok-bg:#ecfdf5;--fail:#dc2626;--fail-bg:#fef2f2;--warn:#d97706;--radius:12px;
--shadow:0 1px 3px rgba(0,0,0,.06),0 1px 2px rgba(0,0,0,.04)}
*{box-sizing:border-box;margin:0}
body{font:16px/1.6 'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
color:var(--ink);background:var(--bg)}
a{color:var(--accent);text-decoration:none}

/* Header */
.header{background:var(--card);border-bottom:1px solid var(--border);padding:28px 48px}
.header h1{font-size:24px;font-weight:700;color:var(--ink)}
.header p{color:var(--muted);font-size:14px;margin-top:4px}

/* Tabs */
.tabs{display:flex;gap:0;background:var(--card);border-bottom:1px solid var(--border);
padding:0 48px;position:sticky;top:0;z-index:10}
.tab{padding:16px 28px;font-size:15px;font-weight:600;color:var(--muted);
cursor:pointer;border-bottom:2px solid transparent;transition:all .15s}
.tab:hover{color:var(--ink)}
.tab.active{color:var(--accent);border-bottom-color:var(--accent)}

/* Layout */
main{max-width:1440px;margin:0 auto;padding:32px 48px 80px;width:95%}
.panel{display:none}.panel.active{display:block}

/* Cards grid */
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin-bottom:32px}
.card{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);
padding:20px;box-shadow:var(--shadow)}
.card .label{font-size:11px;font-weight:600;letter-spacing:.05em;
text-transform:uppercase;color:var(--muted);margin-bottom:8px}
.card .value{font-size:32px;font-weight:700;color:var(--accent);
font-variant-numeric:tabular-nums}
.card .value.ok{color:var(--ok)}.card .value.fail{color:var(--fail)}

/* Summary table */
.summary-table{width:100%;border-collapse:separate;border-spacing:0;
background:var(--card);border:1px solid var(--border);border-radius:var(--radius);
overflow:hidden;box-shadow:var(--shadow)}
.summary-table th{background:var(--bg);font-size:11px;font-weight:600;
letter-spacing:.04em;text-transform:uppercase;color:var(--muted);
padding:12px 16px;text-align:left;border-bottom:1px solid var(--border)}
.summary-table td{padding:14px 16px;border-bottom:1px solid var(--border);font-size:14px}
.summary-table tr:last-child td{border-bottom:none}
.mini-bar{display:inline-flex;align-items:center;gap:8px;width:100%}
.mini-bar .track{flex:1;height:6px;background:var(--border);border-radius:99px;overflow:hidden}
.mini-bar .fill{height:100%;border-radius:99px;background:var(--accent)}
.mini-bar .pct{font-size:12px;font-weight:700;min-width:36px;text-align:right;
font-variant-numeric:tabular-nums}

/* Workload accordion */
.workload{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);
margin-bottom:16px;box-shadow:var(--shadow);overflow:hidden}
.workload-header{padding:18px 20px;cursor:pointer;display:flex;align-items:center;gap:14px;
user-select:none}
.workload-header:hover{background:var(--bg)}
.workload-header .arrow{transition:transform .2s;color:var(--muted);font-size:12px}
.workload.open .workload-header .arrow{transform:rotate(90deg)}
.workload-header .wl-name{font-weight:700;font-size:16px}
.workload-header .wl-type{color:var(--muted);font-size:13px}
.workload-header .wl-badge{margin-left:auto;font-size:12px;font-weight:700;
padding:4px 12px;border-radius:99px}
.wl-badge.pass{background:var(--ok-bg);color:var(--ok)}
.wl-badge.fail{background:var(--fail-bg);color:var(--fail)}
.wl-badge.err{background:var(--fail-bg);color:var(--fail)}
.workload-body{display:none;padding:0 20px 20px;border-top:1px solid var(--border)}
.workload.open .workload-body{display:block}

/* Check bars */
.check-bars{margin:16px 0}
.check-row{display:grid;grid-template-columns:220px 1fr 56px 52px;gap:14px;
align-items:center;padding:10px 0}
.check-row .name{font-size:14px;font-weight:600;overflow:hidden;
text-overflow:ellipsis;white-space:nowrap}
.check-row .bar{height:12px;background:var(--border);border-radius:99px;overflow:hidden}
.check-row .bar-fill{height:100%;border-radius:99px;background:var(--accent);transition:width .3s}
.check-row .bar-fill.low{background:var(--warn)}
.check-row .bar-fill.crit{background:var(--fail)}
.check-row .rate{font-size:13px;font-weight:700;text-align:right;
font-variant-numeric:tabular-nums}

/* Gate */
.gate{display:flex;align-items:center;gap:14px;padding:16px 20px;
border-radius:var(--radius);margin:16px 0;font-weight:600}
.gate.pass{background:var(--ok-bg);border:1px solid #a7f3d0;color:var(--ok)}
.gate.fail{background:var(--fail-bg);border:1px solid #fecaca;color:var(--fail)}
.gate .gate-val{font-size:26px;font-variant-numeric:tabular-nums}

/* Cases table */
.cases-section h4{font-size:12px;font-weight:600;letter-spacing:.04em;
text-transform:uppercase;color:var(--muted);margin:20px 0 12px}
.case-item{border:1px solid var(--border);border-radius:8px;margin-bottom:8px;overflow:hidden}
.case-head{display:grid;grid-template-columns:90px 1fr auto;gap:14px;
padding:14px 16px;align-items:center;cursor:pointer}
.case-head:hover{background:var(--bg)}
.case-head .cid{font-family:ui-monospace,monospace;font-size:13px;color:var(--muted);font-weight:600}
.case-head .cinput{font-size:14px;color:var(--ink);overflow:hidden;
text-overflow:ellipsis;white-space:nowrap}
.case-head .cverd{font-size:12px;font-weight:700;padding:4px 12px;border-radius:99px}
.cverd.pass{background:var(--ok-bg);color:var(--ok)}
.cverd.fail{background:var(--fail-bg);color:var(--fail)}
.case-body{display:none;padding:12px 14px;border-top:1px solid var(--border);background:var(--bg)}
.case-item.open .case-body{display:block}
.checks-pills{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px}
.pill{font-size:11px;font-weight:600;padding:3px 10px;border-radius:99px}
.pill.pass{background:var(--ok-bg);color:var(--ok)}
.pill.fail{background:var(--fail-bg);color:var(--fail)}
.fail-reasons{margin-top:8px}
.fail-reasons .fr{font-size:12px;color:var(--fail);margin:4px 0;
padding-left:12px;border-left:3px solid #fecaca}
.fail-reasons .fr b{color:var(--ink)}

/* Error panel */
.error-panel{background:var(--fail-bg);border:1px solid #fecaca;border-radius:var(--radius);
padding:20px;color:var(--fail)}
.error-panel h3{font-size:15px;margin-bottom:8px}
.error-panel pre{background:var(--card);border:1px solid #fecaca;border-radius:8px;
padding:14px;font:12.5px/1.5 ui-monospace,monospace;color:var(--ink);
overflow:auto;white-space:pre-wrap}

/* Review placeholder */
.review-placeholder{text-align:center;padding:60px 20px;color:var(--muted)}
.review-placeholder .icon{font-size:48px;margin-bottom:16px;opacity:.5}
.review-placeholder p{font-size:14px;max-width:400px;margin:0 auto}

.check-row .napp{font-size:12px;color:var(--muted);text-align:right;font-variant-numeric:tabular-nums}
.pill.na{background:var(--border);color:var(--muted)}
.case-body .fr{font-size:12px;color:var(--fail);margin:4px 0;padding-left:12px;border-left:3px solid #fecaca}
.warn{background:#fffbeb;border:1px solid #fde68a;color:#92400e;border-radius:8px;
padding:10px 14px;font-size:13px;margin-bottom:10px}
.ops{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:8px}
.op{display:flex;justify-content:space-between;gap:10px;background:var(--bg);
border:1px solid var(--border);border-radius:8px;padding:8px 12px;font-size:13px}
.op span{color:var(--muted)}
.review-summary{font-size:15px;margin-bottom:8px}
.recs{padding-left:20px;font-size:14px}.recs li{margin:6px 0}
code{font:12px ui-monospace,monospace;color:var(--muted)}

footer{text-align:center;color:var(--muted);font-size:12px;padding:20px}
"""



_JS = """\
document.querySelectorAll('.tab').forEach(t=>t.addEventListener('click',()=>{
  document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(x=>x.classList.remove('active'));
  t.classList.add('active');
  document.getElementById(t.dataset.t).classList.add('active');
}));
document.querySelectorAll('.workload-header').forEach(h=>h.addEventListener('click',()=>{
  h.parentElement.classList.toggle('open');
}));
document.querySelectorAll('.case-head').forEach(h=>h.addEventListener('click',()=>{
  h.parentElement.classList.toggle('open');
}));
"""


# ---------------------------------------------------------------------------
# HTML rendering — panels
# ---------------------------------------------------------------------------


def _render_summary(agg: Aggregate) -> str:
    """Render the Summary tab panel."""
    rate_cls = "ok" if (agg.overall_rate and agg.overall_rate >= 0.8) else "fail"
    cards = f"""<div class="cards">
<div class="card"><div class="label">Workloads</div><div class="value">{agg.n_workloads}</div></div>
<div class="card"><div class="label">Total Cases</div><div class="value">{agg.n_cases}</div></div>
<div class="card"><div class="label">All-Checks Pass</div><div class="value {rate_cls}">{_pct(agg.overall_rate)}</div></div>
<div class="card"><div class="label">Errored</div><div class="value{"" if not agg.n_errored else " fail"}">{agg.n_errored}</div></div>
</div>"""

    rows = ""
    for r in agg.workloads:
        if r.error:
            rows += f'<tr><td><b>{_esc(r.workload)}</b></td><td>{_esc(r.eval_type)}</td><td>—</td><td style="color:var(--fail);font-weight:700">ERROR</td></tr>'
        else:
            rate = r.all_checks_pass_rate or 0
            w = max(0, min(100, rate * 100))
            rows += f'''<tr><td><b>{_esc(r.workload)}</b></td><td>{_esc(r.eval_type)}</td><td>{r.n}</td>
<td><div class="mini-bar"><div class="track"><div class="fill" style="width:{w:.0f}%"></div></div><div class="pct">{_pct(r.all_checks_pass_rate)}</div></div></td></tr>'''

    table = f"""<table class="summary-table"><thead><tr>
<th>Workload</th><th>Eval Type</th><th>Cases</th><th>All-Checks Pass Rate</th>
</tr></thead><tbody>{rows}</tbody></table>"""

    return f'<div class="panel active" id="summary">{cards}{table}</div>'



def _render_results(results: list[WorkloadResult]) -> str:
    """Render the Results tab panel with expandable workload sections."""
    sections = ""
    for r in results:
        if r.error:
            sections += f'''<div class="workload"><div class="workload-header">
<span class="arrow">▶</span><span class="wl-name">{_esc(r.workload)}</span>
<span class="wl-type">{_esc(r.eval_type)}</span>
<span class="wl-badge err">ERROR</span></div>
<div class="workload-body"><div class="error-panel"><h3>Evaluation failed</h3>
<pre>{_esc(r.error)}</pre></div></div></div>'''
            continue

        # Pass rate bars (N/A cases are excluded from each rate)
        bars = ""
        for check in r.checks:
            rate = r.per_check_pass_rate.get(check)
            applied = sum(1 for c in r.cases if c.verdicts.get(check) is not None)
            w = 0 if rate is None else max(0, min(100, rate * 100))
            fill_cls = "" if rate is None else " crit" if rate < 0.5 else " low" if rate < 0.8 else ""
            bars += f'''<div class="check-row"><div class="name" title="{_esc(check)}">{_esc(check)}</div>
<div class="bar"><div class="bar-fill{fill_cls}" style="width:{w:.0f}%"></div></div>
<div class="rate">{_pct(rate)}</div><div class="napp">n={applied}</div></div>'''

        # Operational metrics
        ops = ""
        if r.operational:
            items = "".join(
                f'<div class="op"><span>{_esc(k)}</span><b>{_esc(v)}</b></div>'
                for k, v in r.operational.items()
            )
            ops = f'<div class="cases-section"><h4>Operational</h4><div class="ops">{items}</div></div>'

        # Judge calibration
        judges = ""
        if r.judges:
            rows = ""
            for j in r.judges:
                status = (
                    '<span class="pill pass">calibrated</span>' if j.calibrated
                    else '<span class="pill fail">UNCALIBRATED</span>'
                )
                rows += (f"<tr><td>{_esc(j.name)}</td><td>{_esc(j.model)}</td><td>{status}</td>"
                         f"<td>{_pct(j.tpr)}</td><td>{_pct(j.tnr)}</td><td>{_esc(j.n_labels or '—')}</td></tr>")
            warn = ""
            if any(not j.calibrated for j in r.judges):
                warn = ('<div class="warn">Uncalibrated judges have not been checked against human '
                        'labels. Treat their pass rates as unverified.</div>')
            judges = f'''<div class="cases-section"><h4>LLM judges</h4>{warn}
<table class="summary-table"><thead><tr><th>Judge</th><th>Model</th><th>Status</th>
<th>TPR</th><th>TNR</th><th>Labels</th></tr></thead><tbody>{rows}</tbody></table></div>'''

        # Gate
        gate_cls = "pass" if (r.all_checks_pass_rate and r.all_checks_pass_rate >= 1.0) else "fail"
        gate = f'''<div class="gate {gate_cls}"><span class="gate-val">{_pct(r.all_checks_pass_rate)}</span>
<span>of {r.n} cases pass ALL checks</span></div>'''

        # Cases
        case_items = ""
        for case in r.cases:
            verd_cls = "pass" if case.overall_pass else "fail"
            verd_text = "PASS" if case.overall_pass else "FAIL"
            input_display = _esc(case.input_text) if case.input_text else "<em>no input recorded</em>"

            pills = ""
            for check in r.checks:
                if check not in case.verdicts:
                    continue
                v = case.verdicts[check]
                if v is True:
                    pills += f'<span class="pill pass">✓ {_esc(check)}</span>'
                elif v is False:
                    pills += f'<span class="pill fail">✗ {_esc(check)}</span>'
                else:
                    pills += f'<span class="pill na">– {_esc(check)} (n/a)</span>'
            if case.error:
                verd_cls, verd_text = "fail", "ERROR"
                pills += f'<div class="fr"><b>error:</b> {_esc(case.error)}</div>'

            reasons_html = ""
            if case.reasons:
                bits = "".join(
                    f'<div class="fr"><b>{_esc(k)}:</b> {_esc(v)}</div>'
                    for k, v in case.reasons.items()
                )
                reasons_html = f'<div class="fail-reasons">{bits}</div>'

            case_items += f'''<div class="case-item"><div class="case-head">
<span class="cid">{_esc(case.id)}</span>
<span class="cinput">{input_display}</span>
<span class="cverd {verd_cls}">{verd_text}</span></div>
<div class="case-body"><div class="checks-pills">{pills}</div>{reasons_html}</div></div>'''

        badge_cls = "pass" if (r.all_checks_pass_rate and r.all_checks_pass_rate >= 1.0) else "fail"
        badge_text = _pct(r.all_checks_pass_rate)

        sections += f'''<div class="workload"><div class="workload-header">
<span class="arrow">▶</span><span class="wl-name">{_esc(r.workload)}</span>
<span class="wl-type">{_esc(r.eval_type)} · {r.n} cases</span>
<span class="wl-badge {badge_cls}">{badge_text}</span></div>
<div class="workload-body"><div class="check-bars">{bars}</div>{gate}{judges}{ops}
<div class="cases-section"><h4>Test Cases</h4>{case_items}</div></div></div>'''

    return f'<div class="panel" id="results">{sections}</div>'



def _render_review(review: dict[str, Any] | None) -> str:
    """Render the Review tab from review.json, or a placeholder in Build mode."""
    if not review:
        return '''<div class="panel" id="review"><div class="review-placeholder">
<div class="icon">📋</div>
<p>No review in this run. In Review mode, the skill grades your existing evals
against the workshop's practices and shows findings, coverage, and fixes here.</p>
</div></div>'''

    parts = []
    if review.get("summary"):
        parts.append(f'<p class="review-summary">{_esc(review["summary"])}</p>')

    findings = review.get("findings") or []
    if findings:
        rows = "".join(
            f'<tr><td><span class="pill {"fail" if str(f.get("severity", "")).lower() in ("high", "critical") else "na"}">'
            f'{_esc(f.get("severity", "—"))}</span></td><td>{_esc(f.get("eval", "—"))}</td>'
            f'<td>{_esc(f.get("issue", ""))}</td><td><code>{_esc(f.get("evidence", ""))}</code></td>'
            f'<td>{_esc(f.get("fix", ""))}</td></tr>'
            for f in findings
        )
        parts.append(f'''<div class="cases-section"><h4>Findings ({len(findings)})</h4>
<table class="summary-table"><thead><tr><th>Severity</th><th>Eval</th><th>Issue</th>
<th>Evidence</th><th>Fix</th></tr></thead><tbody>{rows}</tbody></table></div>''')

    coverage = review.get("coverage") or []
    if coverage:
        def _cov(v: Any) -> str:
            if v is True or str(v).lower() == "yes":
                return '<span class="pill pass">covered</span>'
            if str(v).lower() == "partial":
                return '<span class="pill na">partial</span>'
            return '<span class="pill fail">gap</span>'
        rows = "".join(
            f'<tr><td>{_esc(c.get("failure_mode", "—"))}</td><td>{_cov(c.get("covered"))}</td>'
            f'<td>{_esc(c.get("by", "—"))}</td><td>{_esc(c.get("module", "—"))}</td></tr>'
            for c in coverage
        )
        parts.append(f'''<div class="cases-section"><h4>Coverage</h4>
<table class="summary-table"><thead><tr><th>Failure mode</th><th>Status</th>
<th>Covered by</th><th>Workshop module</th></tr></thead><tbody>{rows}</tbody></table></div>''')

    recs = review.get("recommendations") or []
    if recs:
        items = "".join(f"<li>{_esc(x)}</li>" for x in recs)
        parts.append(f'<div class="cases-section"><h4>Recommendations</h4><ol class="recs">{items}</ol></div>')

    return f'<div class="panel" id="review">{"".join(parts)}</div>'


def render_html(results: list[WorkloadResult], agg: Aggregate,
                review: dict[str, Any] | None = None) -> str:
    """Render the full self-contained HTML report.

    Args:
        results: All workload results.
        agg: Aggregate summary.
        review: Optional Review-mode findings (see --review).

    Returns:
        Complete HTML string.
    """
    tabs = '''<div class="tabs">
<div class="tab active" data-t="summary">Summary</div>
<div class="tab" data-t="results">Results</div>
<div class="tab" data-t="review">Review</div>
</div>'''

    header = f'''<div class="header">
<h1>Evaluation Report</h1>
<p>{agg.n_workloads} workloads · {agg.n_cases} cases · {agg.n_errored} errored</p>
</div>'''

    panels = _render_summary(agg) + _render_results(results) + _render_review(review)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Evaluation Report</title>
<style>{_CSS}</style>
</head>
<body>
{header}
{tabs}
<main>{panels}</main>
<footer>Generated by eval-builder · deterministic, offline, self-contained</footer>
<script>{_JS}</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Command-line arguments (defaults to sys.argv).

    Returns:
        Exit code (0 success, 2 for errors).
    """
    parser = argparse.ArgumentParser(description="Build eval report HTML.")
    parser.add_argument("results_dir", type=Path, help="Directory of per-workload *.json results")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output HTML path")
    parser.add_argument("--review", type=Path, help="Optional Review-mode findings JSON")
    args = parser.parse_args(argv)

    review = None
    if args.review:
        try:
            review = json.loads(args.review.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"error: cannot read review file {args.review}: {exc}", file=sys.stderr)
            return 2

    try:
        results = load_results(args.results_dir)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not results:
        print(f"error: no *.json files in {args.results_dir}", file=sys.stderr)
        return 2

    agg = build_aggregate(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(results, agg, review), encoding="utf-8")

    results_json = args.output.parent / "results.json"
    export = {
        "totals": {
            "n_workloads": agg.n_workloads, "n_ok": agg.n_ok,
            "n_errored": agg.n_errored, "n_cases": agg.n_cases,
            "overall_all_checks_pass_rate": agg.overall_rate,
        },
        "workloads": [
            {"workload": r.workload, "eval_type": r.eval_type,
             "all_checks_pass_rate": r.all_checks_pass_rate, "n": r.n,
             "per_check_pass_rate": r.per_check_pass_rate,
             "uncalibrated_judges": [j.name for j in r.judges if not j.calibrated],
             "error": r.error}
            for r in results
        ],
    }
    results_json.write_text(
        json.dumps(export, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"Wrote {args.output} ({len(results)} workloads)")
    print(f"Wrote {results_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
