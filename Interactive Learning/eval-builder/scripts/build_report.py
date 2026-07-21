#!/usr/bin/env python3
"""Deterministic, self-contained HTML report generator for eval-builder.

Consumes per-workload result JSON files (design.md §9) and produces:
  * a single self-contained ``report.html`` (inline CSS/JS, no external deps),
    with one tab per workload/eval-type plus an aggregate summary tab; and
  * an aggregated ``results.json`` written next to the HTML.

The output is a pure function of the input JSON: stable ordering, no
timestamps, no network, no LLM calls. Standard library only.

Usage:
    python3 build_report.py <results_dir> -o <output.html>

Input schema (design.md §9), one object per ``*.json`` file::

    {
      "workload": "customer-support-agent",
      "eval_type": "tool-calling",
      "checks": ["tool_selection", "efficiency", ...],
      "cases": [
        {"id": "tc-001",
         "verdicts": {"tool_selection": true, "efficiency": false},
         "overall_pass": false,
         "reason": {"efficiency": "duplicate lookup call"}}
      ],
      "summary": {"per_check_pass_rate": {"tool_selection": 0.9},
                  "all_checks_pass_rate": 0.55, "n": 20},
      "error": null
    }
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Loading & normalization
# ---------------------------------------------------------------------------


def load_results(results_dir: Path) -> list[dict[str, Any]]:
    """Load and normalize every ``*.json`` file in ``results_dir``.

    Args:
        results_dir: Directory containing per-workload result JSON files.

    Returns:
        A list of normalized result dicts, sorted deterministically by
        ``(workload, eval_type, source_filename)``.

    Raises:
        FileNotFoundError: If ``results_dir`` does not exist or is not a dir.
    """
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {results_dir}")

    results: list[dict[str, Any]] = []
    # Sort by filename first so that the source ordering is itself stable.
    for path in sorted(results_dir.glob("*.json"), key=lambda p: p.name):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            # A malformed file becomes an error panel rather than aborting.
            results.append(
                _normalize(
                    {"workload": path.stem, "eval_type": "unknown",
                     "error": f"Failed to parse {path.name}: {exc}"},
                    path.name,
                )
            )
            continue
        results.append(_normalize(raw, path.name))

    # Deterministic tab ordering, independent of filesystem enumeration order.
    results.sort(key=lambda r: (r["workload"], r["eval_type"], r["_source"]))
    return results


def _normalize(raw: dict[str, Any], source: str) -> dict[str, Any]:
    """Fill in defaults so downstream rendering never has to guard for keys."""
    checks = list(raw.get("checks") or [])
    summary = raw.get("summary") or {}
    per_check = dict(summary.get("per_check_pass_rate") or {})
    return {
        "workload": str(raw.get("workload") or source),
        "eval_type": str(raw.get("eval_type") or "unknown"),
        "checks": checks,
        "cases": list(raw.get("cases") or []),
        "summary": {
            "per_check_pass_rate": per_check,
            "all_checks_pass_rate": summary.get("all_checks_pass_rate"),
            "n": summary.get("n", len(raw.get("cases") or [])),
        },
        "error": raw.get("error"),
        "_source": source,
        "_id": _slug(f"{raw.get('workload') or source}-{raw.get('eval_type') or 'unknown'}"),
    }


def _slug(text: str) -> str:
    """Deterministic id-safe slug for tab/panel anchors."""
    return "".join(c if c.isalnum() else "-" for c in text.lower()).strip("-")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def build_aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute cross-workload summary numbers (deterministic)."""
    ok = [r for r in results if not r["error"]]
    errored = [r for r in results if r["error"]]
    total_cases = sum(int(r["summary"]["n"] or 0) for r in ok)
    # Overall all-checks-pass rate weighted by case count.
    weighted = 0.0
    for r in ok:
        rate = r["summary"]["all_checks_pass_rate"]
        n = int(r["summary"]["n"] or 0)
        if rate is not None:
            weighted += float(rate) * n
    overall_rate = (weighted / total_cases) if total_cases else None
    return {
        "workloads": [
            {k: v for k, v in r.items() if not k.startswith("_")} for r in results
        ],
        "totals": {
            "n_workloads": len(results),
            "n_ok": len(ok),
            "n_errored": len(errored),
            "n_cases": total_cases,
            "overall_all_checks_pass_rate": overall_rate,
        },
    }


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

_STYLE = """
:root{--indigo:#7c3aed;--indigo2:#8b5cf6;--ink:#1e1b2e;--muted:#6b7280;--bg:#f7f7fb;
--card:#fff;--line:#e6e4ef;--ok:#16a34a;--pill:#efeafe;--code:#f4f2fb;--red:#b91c1c}
*{box-sizing:border-box}
body{margin:0;font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;color:var(--ink);background:var(--bg)}
header.hero{background:linear-gradient(120deg,var(--indigo),var(--indigo2));color:#fff;padding:26px 40px}
header.hero h1{margin:0;font-size:24px}
header.hero p{margin:6px 0 0;opacity:.9;font-size:14px}
.tabs{display:flex;gap:4px;padding:0 40px;background:var(--card);border-bottom:1px solid var(--line);position:sticky;top:0;z-index:5;flex-wrap:wrap}
.tab{padding:14px 18px;cursor:pointer;font-weight:600;font-size:14px;color:var(--muted);border-bottom:3px solid transparent}
.tab.active{color:var(--indigo);border-bottom-color:var(--indigo)}
.tab .badge{font-size:11px;background:var(--pill);color:var(--indigo);border-radius:999px;padding:1px 7px;margin-left:6px}
.tab.err{color:var(--red)}
main{max-width:960px;margin:0 auto;padding:28px 40px 60px}
.panel{display:none}
.panel.active{display:block}
section.block{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px 20px;margin:0 0 14px}
h2.wt{margin:0 0 4px;font-size:20px}
.sub{color:var(--muted);font-size:13px;margin:0 0 4px}
h3.sh{margin:18px 2px 10px;font-size:13px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.chip{background:var(--pill);color:var(--indigo);border-radius:999px;padding:3px 10px;font-size:12px;font-weight:600}
.bar-row{display:grid;grid-template-columns:200px 1fr 56px;align-items:center;gap:12px;margin:8px 0;font-size:13px}
.bar-row .lbl{font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.prog{height:12px;background:var(--line);border-radius:999px;overflow:hidden}
.progfill{height:100%;background:linear-gradient(90deg,var(--indigo),var(--indigo2))}
.progfill.low{background:linear-gradient(90deg,#f59e0b,#ef4444)}
.pct{text-align:right;font-variant-numeric:tabular-nums;font-weight:700;color:var(--ink)}
.gate{border-radius:12px;padding:14px 16px;margin:6px 0;font-weight:600;display:flex;align-items:center;gap:10px}
.gate.pass{background:#eafaf0;border:1px solid #b7e4c7;color:var(--ok)}
.gate.fail{background:#fdecec;border:1px solid #f3c2c2;color:var(--red)}
.gate .big{font-size:22px;font-variant-numeric:tabular-nums}
table{border-collapse:collapse;width:100%;margin:10px 0;font-size:13px}
th,td{border:1px solid var(--line);padding:7px 10px;text-align:left;vertical-align:top}
th{background:var(--pill);font-weight:600}
td.mono{font-family:ui-monospace,Menlo,monospace;font-size:12.5px}
.v-pass{color:var(--ok);font-weight:700}
.v-fail{color:var(--red);font-weight:700}
details.case{border:1px solid var(--line);border-radius:10px;margin:8px 0;background:var(--card)}
details.case>summary{cursor:pointer;list-style:none;padding:11px 14px;font-weight:600;display:flex;align-items:center;gap:10px}
details.case>summary::-webkit-details-marker{display:none}
details.case>summary .chev{transition:transform .2s;color:var(--muted)}
details.case[open]>summary .chev{transform:rotate(90deg)}
.badge-pass{margin-left:auto;font-size:11px;font-weight:700;color:var(--ok);background:#eafaf0;border:1px solid #b7e4c7;border-radius:999px;padding:2px 10px}
.badge-fail{margin-left:auto;font-size:11px;font-weight:700;color:var(--red);background:#fdecec;border:1px solid #f3c2c2;border-radius:999px;padding:2px 10px}
.case-body{padding:2px 14px 14px}
.reason{color:var(--red);font-size:12.5px}
.errpanel{background:#fdecec;border:1px solid #f3c2c2;border-radius:12px;padding:18px 20px;color:var(--red)}
.errpanel h3{margin:0 0 8px;font-size:15px}
.errpanel pre{background:#fff;border:1px solid #f3c2c2;border-radius:8px;padding:12px;overflow:auto;font:12.5px/1.5 ui-monospace,Menlo,monospace;color:var(--ink);margin:0}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:12px 0}
.kv{background:var(--bg);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.kv h4{margin:0 0 6px;font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:var(--muted)}
.kv .val{font-size:24px;font-weight:700;color:var(--indigo);font-variant-numeric:tabular-nums}
footer{color:var(--muted);font-size:12px;text-align:center;padding:20px}
"""

_SCRIPT = """
(function(){
  var tabs=document.querySelectorAll('.tab');
  function show(id){
    document.querySelectorAll('.panel').forEach(function(p){p.classList.toggle('active',p.id===id);});
    tabs.forEach(function(t){t.classList.toggle('active',t.dataset.t===id);});
  }
  tabs.forEach(function(t){t.addEventListener('click',function(){show(t.dataset.t);});});
})();
"""


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _fmt_pct(rate: Any) -> str:
    if rate is None:
        return "—"
    return f"{float(rate) * 100:.0f}%"


def _bar(label: str, rate: Any) -> str:
    if rate is None:
        width, pct, low = 0.0, "—", ""
    else:
        width = max(0.0, min(1.0, float(rate))) * 100
        pct = f"{float(rate) * 100:.0f}%"
        low = " low" if float(rate) < 0.8 else ""
    return (
        f'<div class="bar-row"><div class="lbl" title="{_esc(label)}">{_esc(label)}</div>'
        f'<div class="prog"><div class="progfill{low}" style="width:{width:.1f}%"></div></div>'
        f'<div class="pct">{_esc(pct)}</div></div>'
    )


def _render_case_table(result: dict[str, Any]) -> str:
    checks = result["checks"]
    rows: list[str] = []
    for case in result["cases"]:
        verdicts = case.get("verdicts") or {}
        reasons = case.get("reason") or {}
        overall = bool(case.get("overall_pass"))
        badge = (
            '<span class="badge-pass">PASS</span>' if overall
            else '<span class="badge-fail">FAIL</span>'
        )
        case_id = _esc(case.get("id", "—"))
        cells: list[str] = []
        for check in checks:
            val = verdicts.get(check)
            if val is True:
                cells.append('<td class="v-pass">✓</td>')
            elif val is False:
                cells.append('<td class="v-fail">✗</td>')
            else:
                cells.append("<td>—</td>")
        # Failure reasons (only for checks that recorded one).
        reason_bits = [
            f"<div class='reason'><b>{_esc(k)}:</b> {_esc(v)}</div>"
            for k, v in reasons.items()
        ]
        reason_html = "".join(reason_bits) or "<span style='color:var(--muted)'>—</span>"
        header_cells = "".join(f"<th>{_esc(c)}</th>" for c in checks)
        rows.append(
            f'<details class="case"><summary><span class="chev">▶</span>'
            f'<span class="mono">{case_id}</span>{badge}</summary>'
            f'<div class="case-body"><table><thead><tr>{header_cells}</tr></thead>'
            f'<tbody><tr>{"".join(cells)}</tr></tbody></table>'
            f'<h3 class="sh">Failure reasons</h3>{reason_html}</div></details>'
        )
    if not rows:
        return "<p class='sub'>No cases recorded.</p>"
    return "".join(rows)


def _render_workload_panel(result: dict[str, Any]) -> str:
    wl = _esc(result["workload"])
    et = _esc(result["eval_type"])
    if result["error"]:
        return (
            f'<div class="panel" id="{_esc(result["_id"])}">'
            f'<section class="block"><h2 class="wt">{wl}</h2>'
            f'<p class="sub">{et}</p>'
            f'<div class="errpanel"><h3>⚠ This workload failed to evaluate</h3>'
            f'<pre>{_esc(result["error"])}</pre></div></section></div>'
        )

    summary = result["summary"]
    per_check = summary["per_check_pass_rate"]
    # Deterministic bar ordering: follow declared checks[] order, then any
    # extra keys present only in per_check_pass_rate, sorted.
    ordered = list(result["checks"])
    for k in sorted(per_check):
        if k not in ordered:
            ordered.append(k)
    bars = "".join(_bar(c, per_check.get(c)) for c in ordered)

    gate_rate = summary["all_checks_pass_rate"]
    gate_cls = "pass" if (gate_rate is not None and float(gate_rate) >= 1.0) else "fail"
    gate = (
        f'<div class="gate {gate_cls}"><span class="big">{_fmt_pct(gate_rate)}</span>'
        f'<span>of {int(summary["n"] or 0)} cases pass <b>all</b> checks '
        f'(all-checks-pass gate)</span></div>'
    )

    return (
        f'<div class="panel" id="{_esc(result["_id"])}">'
        f'<section class="block"><h2 class="wt">{wl}</h2>'
        f'<p class="sub">{et} · {int(summary["n"] or 0)} cases · {len(result["checks"])} checks</p>'
        f'<h3 class="sh">Per-check pass rate</h3>{bars}'
        f'<h3 class="sh">All-checks-pass gate</h3>{gate}'
        f'<h3 class="sh">Cases</h3>{_render_case_table(result)}'
        f'</section></div>'
    )


def _render_summary_panel(agg: dict[str, Any], results: list[dict[str, Any]]) -> str:
    totals = agg["totals"]
    cards = (
        f'<div class="cards">'
        f'<div class="kv"><h4>Workloads</h4><div class="val">{totals["n_workloads"]}</div></div>'
        f'<div class="kv"><h4>Evaluated OK</h4><div class="val">{totals["n_ok"]}</div></div>'
        f'<div class="kv"><h4>Errored</h4><div class="val">{totals["n_errored"]}</div></div>'
        f'<div class="kv"><h4>Total cases</h4><div class="val">{totals["n_cases"]}</div></div>'
        f'<div class="kv"><h4>Overall gate</h4><div class="val">'
        f'{_fmt_pct(totals["overall_all_checks_pass_rate"])}</div></div>'
        f'</div>'
    )
    rows: list[str] = []
    for r in results:
        if r["error"]:
            rows.append(
                f'<tr><td>{_esc(r["workload"])}</td><td>{_esc(r["eval_type"])}</td>'
                f'<td>—</td><td class="v-fail">ERROR</td></tr>'
            )
        else:
            rate = r["summary"]["all_checks_pass_rate"]
            cls = "v-pass" if (rate is not None and float(rate) >= 1.0) else ""
            rows.append(
                f'<tr><td>{_esc(r["workload"])}</td><td>{_esc(r["eval_type"])}</td>'
                f'<td>{int(r["summary"]["n"] or 0)}</td>'
                f'<td class="{cls}">{_fmt_pct(rate)}</td></tr>'
            )
    table = (
        '<table><thead><tr><th>Workload</th><th>Eval type</th>'
        '<th>Cases</th><th>All-checks-pass</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )
    return (
        '<div class="panel active" id="summary">'
        '<section class="block"><h2 class="wt">Summary</h2>'
        '<p class="sub">Aggregated across all workloads</p>'
        f'{cards}<h3 class="sh">Per-workload results</h3>{table}'
        '</section></div>'
    )


def render_html(results: list[dict[str, Any]], agg: dict[str, Any]) -> str:
    """Render the full self-contained HTML document (deterministic)."""
    tabs = ['<div class="tab active" data-t="summary">Summary</div>']
    for r in results:
        err_cls = " err" if r["error"] else ""
        badge = ' <span class="badge">⚠</span>' if r["error"] else ""
        label = f'{_esc(r["workload"])} · {_esc(r["eval_type"])}'
        tabs.append(
            f'<div class="tab{err_cls}" data-t="{_esc(r["_id"])}">{label}{badge}</div>'
        )

    panels = [_render_summary_panel(agg, results)]
    panels.extend(_render_workload_panel(r) for r in results)

    totals = agg["totals"]
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n"
        '<meta charset="utf-8" />\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1" />\n'
        "<title>eval-builder — Results Report</title>\n"
        f"<style>{_STYLE}</style>\n</head>\n<body>\n"
        '<header class="hero"><h1>eval-builder — Results Report</h1>'
        f'<p>{totals["n_workloads"]} workloads · {totals["n_cases"]} cases · '
        f'{totals["n_errored"]} errored</p></header>\n'
        f'<div class="tabs">{"".join(tabs)}</div>\n'
        f'<main>{"".join(panels)}</main>\n'
        '<footer>Generated by build_report.py · deterministic, offline</footer>\n'
        f"<script>{_SCRIPT}</script>\n</body>\n</html>\n"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a deterministic, self-contained HTML eval report."
    )
    parser.add_argument("results_dir", type=Path, help="Directory of *.json results")
    parser.add_argument(
        "-o", "--output", type=Path, required=True, help="Output report.html path"
    )
    args = parser.parse_args(argv)

    try:
        results = load_results(args.results_dir)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not results:
        print(f"error: no *.json files found in {args.results_dir}", file=sys.stderr)
        return 2

    agg = build_aggregate(results)

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_html(results, agg), encoding="utf-8")

    results_json = output.parent / "results.json"
    results_json.write_text(
        json.dumps(agg, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {output} ({len(results)} workload tab(s) + summary)")
    print(f"Wrote {results_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
