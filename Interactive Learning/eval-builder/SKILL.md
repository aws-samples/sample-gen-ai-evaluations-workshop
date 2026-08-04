---
name: eval-builder
description: >
  Build, run, and summarize evaluations for YOUR OWN GenAI workload (RAG, agent, tool-calling,
  chatbot, IDP) using AWS GenAI Evaluations Workshop best practices. Use when the user wants to
  build/create/scaffold evals for their application, evaluate their agent/RAG/chatbot, set up an
  eval harness, or asks for "/eval-builder". Triggers on requests like "build evals for my app",
  "evaluate my agent", "scaffold tests for my RAG pipeline", "create pass/fail judges for my
  chatbot", "review my evals", or "are my evals any good?". NOT for learning eval concepts,
  tutorials, or Socratic practice — that is the tutor (learn-mode). This skill takes action.
category: workflow
---

# eval-builder — build real evals for your own workload

## Purpose

`eval-builder` is the **build-mode** counterpart to the GenAI Evaluations Workshop tutor. The
tutor teaches concepts Socratically; this skill **takes action**: it inspects your repository,
discovers what's actually failing, maps failures to the right workshop modules, and — after you
approve — scaffolds a working eval harness and produces a single interactive HTML report.

## Core philosophy (from the workshop)

1. **Failure-first.** Don't start with generic metrics. Look at actual outputs, find what's
   breaking, build evaluators that target those specific failures.
2. **Fix before you automate.** If a failure can be resolved with a prompt edit, fix it now. Only
   build evaluation infrastructure for failures that need ongoing monitoring.
3. **Layered, not parallel.** Foundational quality always runs first. Workload-specific evals
   (tool-calling, RAG, guardrails…) build ON TOP of foundational findings. Framework-specific
   layers (Strands Evals, AgentCore Runtime Evals) deepen those findings — they don't start from
   scratch. Each layer feeds forward to the next.
4. **Binary pass/fail.** Numeric scales (1–5) hide failure modes. Each evaluator answers one
   question: does this output exhibit failure X? Yes or no.
5. **Evaluate the evaluator.** Every automated judge must be checked against human judgment before
   trusting it.
6. **Workshop is source of truth.** Read the workshop modules for patterns — never hardcode
   evaluator names, imports, or API patterns from memory.

## Reference strategy (no-drift)

The skill carries bundled `references/` for offline use. But if the workshop repo is available
(either the current directory IS the workshop, or it can be cloned to a temp location), **prefer
reading live module content** over bundled references — this eliminates drift. The decision:

- If `Interactive Learning/` or `Foundational Evaluations/` exists in the working tree or a parent
  → read live from it.
- Otherwise, offer to clone:
  `git clone --depth 1 https://github.com/aws-samples/sample-gen-ai-evaluations-workshop.git /tmp/eval-workshop-reference`
  If the user declines or it fails, fall back to bundled `references/`.
- At the end of a session that cloned, clean up: `rm -rf /tmp/eval-workshop-reference`.

When reading live, always read the README first to understand current structure, then navigate to
the relevant module for each step. Never hardcode evaluator names, imports, or patterns — derive
them from the module content.

## Flow overview

```
Phase 1 Discover ─[confirm scope]─►
Phase 2 Failure-First ─[fix-before-automate gate]─►
Phase 3 Plan & Build (layered) ─[approval gate]─►
Phase 4 Run ─[run gate]─► evals/report.html
```

> **⚠️ ONE PHASE PER TURN.** Do exactly one phase, then STOP and wait for the user. Never chain
> phases in a single response. Each STOP below is a hard stop: end your turn there.

---

## Phase 0 — Preflight

**Check installation.** This skill runs from the user's **global** skills directory:
- Kiro → `~/.kiro/skills/eval-builder/`
- Claude Code → `~/.claude/skills/eval-builder/`

If invoked from the workshop repo itself and not yet installed globally, have an agent copy
`Interactive Learning/eval-builder/` to the global skills dir. No install script — agent-executed.

**Then the mandatory first question (verbatim):**

> **"Just to confirm, are you in the repository with the workload(s) you'd like me to evaluate?
> That's the best way to run this skill."**

Do NOT proceed until confirmed.

---

## Phase 1 — Discover (discovery ONLY)

Goal: understand the workload(s) and any existing evals. **Read-only. No questions, no mapping,
no file writes.**

1. **Investigate the repository.** Read (do not modify):
   - Dependency manifests for eval-relevant libraries (boto3, strands, bedrock-agentcore, vector/
     retriever libs, jsonschema, deepeval, ragas, promptfoo).
   - Source for API signals — Bedrock converse/invoke_model, toolConfig/tools, retrieve/embeddings/
     KB, multi-agent handoffs, session state, doc/field extraction.
   - Existing evals: `evals/` dirs, judge/scorer code, datasets, eval configs, CI eval steps,
     framework usage (Strands Evals, AgentCore Runtime, promptfoo, mlflow, etc.).
   - Side-effecting tools — classify each tool as read-only vs. side-effecting (write/delete/pay/
     prod) from names/docstrings; flag dangerous tools in findings.
2. **Detect workload type(s)** using `references/module-index.md`.
3. **Present a Discovery findings report** — "Discovery complete. No files written." + a
   signal→evidence table. Include rows for: workload type, framework, tools (with ⚠️ for
   side-effecting), model/region, ground truth, guardrails, and **existing evals** (what/where/
   how many, or "none found").
4. **Mode selection:** if existing evals found → offer Review; if none → default Build; if
   ambiguous → ask. (Review mode evaluates the user's existing evals against the rubric and
   recommends fixes + new evals. Build mode scaffolds fresh.)

**STOP — confirm scope + mode.** Do not ask scoping questions, map modules, or propose evals.
End your turn.

---

## Phase 2 — Failure-First (sample → categorize → fix gate)

Only after user confirms scope+mode. Goal: understand what's *actually failing* before building
evals. **Still no file writes.**

### Build mode:

1. **Operational check.** Quick: is the workload running? Can it be invoked? Any obvious errors?
2. **Sample outputs.** Run 3–5 representative inputs through the workload (or read existing
   outputs/logs if available). Show them to the user.
3. **Categorize failures.** From the samples, identify and group failure modes by type. Prioritize
   by frequency × severity. Present:
   - Failure categories found (with examples).
   - Which are fixable immediately (prompt edit, config change) vs. need ongoing monitoring.
4. **Map to modules.** NOW map failures → workshop modules (via `references/module-index.md` or
   live workshop content). State what's N/A and why. The mapping is **informed by what's actually
   failing**, not generic.
5. **Fix-before-automate gate.** For failures fixable with a prompt/config edit: suggest the fix.
   Ask: "Want to apply these fixes first, then re-sample, before I build evals for the rest?"
   If yes → apply fixes (within `evals/` or the file the user designates), re-sample, update
   failure categories. If no → proceed with all failures.
6. **Ask build-scoping questions** (judge model + region, operational thresholds, anything else
   needed to scaffold). Offer to propose demo-reasonable thresholds for approval.

### Review mode:

1. **Read existing eval code** — judges, datasets, configs, aggregation, reporting.
2. **Map existing evals to modules** — which dimensions does each judge/scorer cover?
3. **Run the review rubric** (from the module's `## Review rubric` section) — binary pass/fail per
   check, with evidence + severity. Flag: Likert scales, multi-criteria mega-judges, missing
   calibration, no ground truth, etc.
4. **Present the review report** — per-eval verdicts, coverage matrix, gaps, and a prioritized
   fix list + new evals to fill gaps.

**STOP — wait for response.** In Build mode: user answers scoping Qs or approves fixes. In Review
mode: user acknowledges findings and decides whether to remediate.

---

## Phase 3 — Plan & Build (layered, approval-gated)

Only after Phase 2 is resolved. Goal: present a concrete plan with **layered evaluation paths**,
get approval, then build.

### Layered paths (not parallel — each feeds the next):

| Layer | Always? | What it covers |
|---|---|---|
| **A: Foundational Quality** | Always | Output correctness — binary pass/fail judges per failure mode discovered in Phase 2. Programmatic checks (deterministic, $0) first, LLM-as-judge for subjective dimensions. boto3 only. |
| **Agentic Process** | If tool-use / multi-step | HOW the agent reached the answer — tool selection, parameter accuracy, sequence, efficiency. Separate from WHAT it answered. |
| **B: Strands Evals SDK** | If strands-agents-evals available | Framework evaluators + experiment runner. Derive evaluators from workshop (not hardcoded). Informed by Layer A findings. |
| **C: AgentCore Runtime Evals** | If deployed to AgentCore + CW Transaction Search | Managed `evaluate()` API with built-in evaluators on full traces. Informed by Layer A findings. |

**Feed-forward:** Layer A's failure categories explicitly scope what Layers B/C evaluate. Framework
evals don't start from scratch — they deepen understanding of foundational failures.

### Build plan contents:
- Layers that apply (with justification from Phase 2 findings).
- Per failure mode: which layer handles it, the exact binary check (one per judge), deterministic
  vs. LLM-judge.
- Dataset (cases from ground truth / sampled outputs, enriched with expected values).
- Aggregation: per-check pass rate + all-checks-pass. Never averaged.
- The `evals/` layout to be written.
- For Review-mode remediation: specific fixes to existing evals (in-place edits) + new evals for gaps.

**STOP — approval gate.** Do NOT create or modify any file until the user approves.

### On approval, scaffold:

```
evals/
├── eval_config.yaml                    # cached: workloads, layers, model IDs, region, failure modes
├── datasets/<workload>.jsonl           # test cases / ground truth
├── judges/<workload>_<failuremode>.py  # ONE binary judge per failure mode
├── run_<workload>.py                   # cases → judges → aggregation → results/
├── results/<workload>.json             # per-workload structured results
├── results.json                        # aggregated (Phase 4)
└── report.html                         # interactive summary (Phase 4)
```

Rules:
- **Every LLM-as-judge is binary pass/fail** (`{"passed": bool, "reason": str}`), one failure mode
  per judge. Never 1–5. Never averaged scores.
- **Mirror workshop patterns** from references or live modules. Never invent Bedrock APIs.
- **Tools are mocked by default; real side-effecting tools are NEVER executed.** Evals assess the
  model's *decisions*, not real side effects. For irreversible tools, lean on deterministic checks.
- **Substitute user's model IDs + region.**
- Write `eval_config.yaml` first (enables repeatable re-runs without re-discovery).

---

## Phase 4 — Run & Summarize (run gate)

Only after build is complete. Goal: run evals and emit one shareable HTML summary.

1. **STOP — confirm before running.** Running calls the user's Bedrock and costs money.
2. On confirmation, run each `run_<workload>.py`. If one errors, record it and continue.
3. **Generate the report:**

   ```bash
   python3 scripts/build_report.py evals/results/ -o evals/report.html
   ```

   Self-contained `evals/report.html` with:
   - **Review tab** (if Review mode): rubric verdicts, coverage matrix, gaps.
   - **Results tab**: per-workload pass-rate bars, all-checks-pass gate, expandable per-case table
     with failure reasons.
   - **Summary tab**: aggregate across workloads.
   - Workloads with an `error` render an error panel.
   - Plus aggregated `evals/results.json` for CI/diffing.

4. **Feed-forward note:** after results, suggest: "These results surface new failure patterns.
   Want to iterate — re-categorize failures and tighten the evals?"

---

## Rules (hard constraints)

- **One phase per turn.** Do a single phase, hit its STOP, wait for the user.
- **Failure-first.** Never design evaluators without first looking at actual outputs and
  categorizing what's failing. Generic metrics without failure grounding are forbidden.
- **Fix before automate.** Suggest prompt/config fixes before building eval infrastructure for
  trivially-fixable failures.
- **Layered, feed-forward.** Foundational always first; workload-specific builds on it; framework
  layers deepen it. Each layer's findings inform the next.
- **Binary pass/fail only.** `{"passed": bool, "reason": str}`, one failure mode per judge. Never
  1–5 scales. Never averaged scores. Report pass rates.
- **Tools mocked; no real side effects.** Never execute side-effecting tools in evals. Evaluate
  model *decisions*, not outcomes.
- **Writes only under `evals/`, Phase 3+ only.** During Phases 1–2, write nothing.
- **Workshop is source of truth.** Prefer live workshop content (clone if needed); fall back to
  bundled `references/`. Never hardcode evaluator names/imports from memory.
- **No secrets to disk; use existing credentials only.**
- **Evaluate the evaluator.** Flag when a judge hasn't been validated against human labels.
