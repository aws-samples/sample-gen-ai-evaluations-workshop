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
   build ON TOP of foundational findings. Framework layers deepen them. Each layer feeds forward.
4. **Binary pass/fail.** Numeric scales (1–5) hide failure modes. Each evaluator answers one
   question: does this output exhibit failure X? Yes or no.
5. **Evaluate the evaluator.** Every automated judge must be checked against human judgment.
6. **Workshop is source of truth.** Always read from the workshop — never hardcode evaluator
   names, imports, or patterns from memory.

## Workshop access (no-drift, no bundled references)

The skill reads the **live workshop content** directly. This eliminates drift entirely — patterns
always come from the current workshop, not stale bundled copies.

**How to get the workshop content:**

1. If the workshop repo is already on disk (check for `Foundational Evaluations/` in the working
   tree, parent dirs, or a sibling directory) → read directly from it.
2. Otherwise, clone it:
   ```
   git clone --depth 1 https://github.com/aws-samples/sample-gen-ai-evaluations-workshop.git /tmp/eval-workshop-reference
   ```
   If the user declines or clone fails → inform the user the skill needs workshop access and
   cannot proceed without it. **There is no offline fallback.**
3. At session end (if cloned): `rm -rf /tmp/eval-workshop-reference`

**Always read the READMEs first** to understand current structure, then navigate to the relevant
module. Never hardcode evaluator names, imports, or API patterns — derive everything from what the
module currently contains.

## Module routing table

After detecting the workload type in Phase 1, read these paths from the workshop:

| Module | Path in workshop repo | When to read |
|--------|----------------------|--------------|
| Workshop overview | `README.md` | Always |
| Foundational overview | `Foundational Evaluations/README.md` | Always |
| Operational metrics | `Foundational Evaluations/01-operational-metrics/` | Always — operational check |
| Quality metrics / LLM-as-Judge | `Foundational Evaluations/02-quality-metrics/` | Always — evaluator design (binary pass/fail, judge calibration) |
| Understanding failures | `Foundational Evaluations/03-understanding-failures/` | Always — failure discovery and categorization |
| Agentic metrics | `Foundational Evaluations/04-agentic-metrics/` | When workload is an agent with tool use |
| Tool Calling | `Workload Specific Evaluations/Tool Calling/` | When workload calls external tools |
| Basic RAG | `Workload Specific Evaluations/Basic RAG/` | When workload uses retrieval-augmented generation |
| MultiModal RAG | `Workload Specific Evaluations/MultiModal RAG/` | When RAG includes vision/audio/image retrieval |
| Chatbot / multi-turn | `Workload Specific Evaluations/Chatbot/` | When workload has multi-turn conversations |
| Guardrails | `Workload Specific Evaluations/Guardrails/` | When workload uses content filters / grounding / alignment |
| Intelligent Document Processing | `Workload Specific Evaluations/Intelligent Document Processing/` | When workload extracts structured data from documents |
| Multiagent Shared Context | `Workload Specific Evaluations/Multiagent Shared Context Evaluation/` | When workload has multiple coordinating agents |
| Red Teaming | `Workload Specific Evaluations/Red Teaming/` | When adversarial / security testing is needed |
| Automated Reasoning | `Workload Specific Evaluations/Automated Reasoning Evaluations/` | When workload uses formal reasoning / policy verification |
| Speech to Speech | `Workload Specific Evaluations/Speech to Speech/` | When workload has voice / audio interaction |
| Strands Evals SDK | `Framework Specific Evaluations/Strands/` | When agent is built with Strands SDK |
| AgentCore | `Framework Specific Evaluations/AgentCore/` | When agent uses AgentCore deployment |
| AgentCore Runtime Evals | `Framework Specific Evaluations/AgentCore Runtime Evals/` | When agent deployed to AgentCore Runtime + CloudWatch |
| DeepEval | `Framework Specific Evaluations/DeepEval/` | When using DeepEval framework |
| DSPy | `Framework Specific Evaluations/DSPy/` | When using DSPy optimization |
| MLflow | `Framework Specific Evaluations/MLflow/` | When using MLflow tracking |
| Promptfoo | `Framework Specific Evaluations/Prompt Foo/` | When using Promptfoo test harness |

Multiple modules can apply to one workload. Foundational modules (operational, quality,
understanding-failures) **always** apply.

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

1. **Locate the workshop.** Check if the workshop repo is accessible (sibling dir, parent, or
   already cloned). If not, clone it now. Read `README.md` and
   `Foundational Evaluations/README.md` to confirm current structure.
2. **Investigate the workload repository.** Read (do not modify):
   - Dependency manifests for eval-relevant libraries.
   - Source for API signals — Bedrock converse/invoke_model, toolConfig/tools, retrieve/embeddings/
     KB, multi-agent handoffs, session state, doc/field extraction.
   - Existing evals: `evals/` dirs, judge/scorer code, datasets, eval configs, CI eval steps.
   - Side-effecting tools — classify each as read-only vs. side-effecting; flag dangerous ones.
3. **Detect workload type(s)** using the routing table above.
4. **Present a Discovery findings report** — "Discovery complete. No files written." + a
   signal→evidence table. Include rows for: workload type, framework, tools (with ⚠️ for
   side-effecting), model/region, ground truth, guardrails, and **existing evals** (what/where/
   how many, or "none found").
5. **Mode selection:** if existing evals found → offer Review; if none → default Build; if
   ambiguous → ask.

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
4. **Map to modules.** NOW map failures → workshop modules (using the routing table). Read the
   relevant module READMEs/notebooks to understand what patterns they teach. State what's N/A and
   why. The mapping is **informed by what's actually failing**, not generic.
5. **Fix-before-automate gate.** For failures fixable with a prompt/config edit: suggest the fix.
   Ask: "Want to apply these fixes first, then re-sample, before I build evals for the rest?"
   If yes → apply fixes, re-sample, update failure categories. If no → proceed with all failures.
6. **Ask build-scoping questions** (judge model + region, operational thresholds, anything else
   needed to scaffold). Offer to propose demo-reasonable thresholds for approval.

### Review mode:

1. **Read existing eval code** — judges, datasets, configs, aggregation, reporting.
2. **Read the relevant workshop modules** (from the routing table) to understand what they teach
   about how evals should be built — binary pass/fail, one failure mode per judge, judge
   calibration, deterministic checks first, pass-rate reporting, etc.
3. **Grade the existing evals against what the workshop teaches.** For each existing eval, assess:
   - Does it use binary pass/fail or a rating scale?
   - Is it one failure mode per judge or a multi-criteria mega-judge?
   - Is it calibrated/validated against human labels?
   - Does it use deterministic checks where possible before LLM judges?
   - Does it report pass rates (not averaged scores)?
   - What dimensions/failure modes does it cover vs. what the mapped module(s) teach?
4. **Present the review report** — per-eval findings (with evidence: file/line), a coverage matrix
   (what's covered vs. what the modules say should be), gaps, and a prioritized fix list + new
   evals to fill gaps.

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
| **B: Framework Evals** | If framework SDK available | Framework-specific evaluators (Strands Evals, DeepEval, DSPy, MLflow, Promptfoo). Derive evaluators from workshop — not hardcoded. Informed by Layer A findings. |
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
- For Review-mode remediation: specific fixes to existing evals (in-place edits) + new evals for
  gaps.

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
- **Mirror workshop patterns.** Read the relevant module from the cloned workshop and use its
  patterns directly. Never invent Bedrock APIs or hardcode evaluator names.
- **Tools are mocked by default; real side-effecting tools are NEVER executed.** Evals assess the
  model's *decisions*, not real side effects.
- **Substitute user's model IDs + region.**
- Write `eval_config.yaml` first (enables repeatable re-runs without re-discovery).

---

## Phase 4 — Run & Summarize (run gate)

Only after build is complete. Goal: run evals and emit one shareable HTML summary.

1. **STOP — confirm before running.** Running calls the user's Bedrock and costs money.
2. On confirmation, run each `run_<workload>.py`. If one errors, record it and continue.
3. **Generate the report** — a self-contained `evals/report.html` (inline CSS/JS, no deps) with:
   - **Review tab** (if Review mode): existing-eval findings, coverage matrix, gaps.
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
  categorizing what's failing.
- **Fix before automate.** Suggest prompt/config fixes before building eval infrastructure for
  trivially-fixable failures.
- **Layered, feed-forward.** Foundational always first; workload-specific builds on it; framework
  layers deepen it.
- **Binary pass/fail only.** `{"passed": bool, "reason": str}`, one failure mode per judge. Never
  1–5 scales. Never averaged scores. Report pass rates.
- **Tools mocked; no real side effects.** Never execute side-effecting tools in evals.
- **Writes only under `evals/`, Phase 3+ only.** During Phases 1–2, write nothing.
- **Workshop is the only source of truth.** Always read live from the cloned workshop. No bundled
  references, no hardcoded patterns from memory. If you can't access the workshop, stop and tell
  the user.
- **No secrets to disk; use existing credentials only.**
- **Evaluate the evaluator.** Flag when a judge hasn't been validated against human labels.
