---
name: eval-builder
description: >
  Build, run, and summarize evaluations for YOUR OWN GenAI workload (RAG, agent, tool-calling,
  chatbot, IDP) using AWS GenAI Evaluations Workshop best practices. Use when the user wants to
  build/create/scaffold evals for their application, evaluate their agent/RAG/chatbot, set up an
  eval harness, or asks for "/eval-builder". Triggers on requests like "build evals for my app",
  "evaluate my agent", "scaffold tests for my RAG pipeline", or "create pass/fail judges for my
  chatbot". NOT for learning eval concepts, tutorials, or Socratic practice — that is the tutor
  (learn-mode). This skill takes action: it discovers your workload, maps it to workshop modules,
  proposes a plan, and (after you approve) scaffolds and runs binary pass/fail evals.
category: workflow
---

# eval-builder — build real evals for your own workload

## Purpose

`eval-builder` is the **build-mode** counterpart to the GenAI Evaluations Workshop tutor.
Where the tutor teaches concepts Socratically (and withholds answers on purpose), this skill
**takes action**: it inspects the repository you are in, maps your workload to the right workshop
evaluation modules, and — after you approve a plan — scaffolds a working eval harness and produces
a single interactive HTML summary.

Use it when the user wants evals _built_ for their application. If the user wants to _learn_ how
evaluation works, redirect them to the workshop tutor (learn-mode) instead.

The skill is fully self-contained. It carries distilled, build-ready copies of the workshop's
evaluation patterns under `references/`. It never reads the workshop repo at runtime — everything
it needs travels with it.

The flow has **four phases, each ending in a STOP** where you wait for the user:

```
Phase 1 Discover ─[confirm scope]─► Phase 2 Map & Scope ─[answer questions]─►
Phase 3 Plan & Build ─[approval gate]─► Phase 4 Run ─[run gate]─► evals/report.html
```

> **⚠️ ONE PHASE PER TURN — the most important rule.** Do exactly one phase, then STOP and wait
> for the user's reply before starting the next phase. Never chain phases in a single response
> (e.g. never discover *and* ask scoping questions, never map *and* present a build plan). Each
> STOP below is a hard stop: end your turn there.

---

## Phase 0 — Preflight

**Check installation first.** This skill is meant to run from the user's **global** skills
directory so it works in any workload repo:

- Kiro → `~/.kiro/skills/eval-builder/`
- Claude Code → `~/.claude/skills/eval-builder/`

If the skill is being invoked from a clone of the workshop repo (i.e. not yet installed globally),
tell the user to have an agent install it once:

> "Open the cloned workshop repo with your agent and say _'install eval-builder'_. The agent will
> copy `Interactive Learning/eval-builder/` into your global skills directory
> (`~/.kiro/skills/eval-builder/` or `~/.claude/skills/eval-builder/`) using the copy mechanism
> appropriate to your OS. After that, `cd` into the repository that holds your workload and invoke
> the skill there."

There is **no install script** — installation is an agent-executed copy. Do not write or run one.

**Then, before anything else, ask the mandatory first question verbatim:**

> **"Just to confirm, are you in the repository with the workload(s) you'd like me to evaluate?
> That's the best way to run this skill."**

Do **not** proceed past this question until the user confirms. If they are not in the right repo,
have them `cd` into it and re-invoke.

---

## Phase 1 — Discover (discovery ONLY)

Goal: understand what workload(s) live in this repo. **Read-only: no file writes, no execution, no
questions, no module mapping yet.** This phase does one thing — discover — and then stops.

1. **Investigate the repository.** Read (do not modify):
   - dependency manifests (`requirements.txt`, `pyproject.toml`, `package.json`) for eval-relevant
     libraries (boto3, `strands`, `bedrock-agentcore`, vector-store/retriever libs, `jsonschema`);
   - source for API signals — Bedrock `converse` / `invoke_model`, `toolConfig` / `tools=[...]`,
     `retrieve(` / embeddings / knowledge base, multi-agent handoffs / shared memory, conversation
     / session state, document/field extraction;
   - any existing `evals/`, tests, datasets, or ground-truth fixtures.
2. **Detect workload type(s)** using `references/module-index.md` (detection signals → workload
   type → modules). More than one type can match a single repo.
3. **Present a Discovery findings report.** Lead with a line like _"Discovery complete. Here's what
   I found — no files were written."_ Then give a compact **signal → evidence table**, e.g.:

   | Signal | Evidence |
   |---|---|
   | Workload | e.g. single tool-calling agent (`agent.py`) |
   | Agent framework | e.g. `strands-agents`; `from strands import Agent, tool` |
   | Tool schemas | e.g. 10 `@tool` functions (list them) |
   | Model / region | e.g. `MODEL_ID`, `REGION` from the source |
   | Ground truth | e.g. `sample_queries.json` — N cases with `expected_tools` |
   | Guardrails | e.g. refund limit, ID pattern, policy/KB grounding |

   Briefly note what was **not** detected too (it shapes mapping later).

4. **STOP — confirm scope.** Ask the user to confirm this is the workload/scope they want
   evaluated (and to correct anything). **Do not ask scoping questions, do not map modules, do not
   propose evals yet.** End your turn and wait for confirmation.

---

## Phase 2 — Map & Scope (module mapping + build-scoping questions)

Only after the user confirms scope. Goal: map to the right workshop modules (this is the core
value) and gather what's needed to build. **Still no file writes, no build plan yet.**

1. **Map** each confirmed workload to workshop module(s) via `references/module-index.md`, and
   **state what is N/A and why** (e.g. "no `bedrock-agentcore`/`@app.entrypoint` → AgentCore N/A;
   single agent → multi-agent N/A; keyword search, not embeddings → RAG N/A"). Call out modules
   that are deferred / unavailable in this skill version. `operational` and `quality` are
   near-universal baselines; workload-specific modules layer on top.
2. **Ask the build-scoping questions** needed to build the evals (do not silently default):
   - **Judge model + region** — which Bedrock model should the LLM-as-judge use, and where?
   - **Operational thresholds** — per-call targets (max latency ms, max cost USD, max TTFT ms,
     throughput floor, error-rate ceiling). If the user has no SLAs, offer to **propose
     demo-reasonable numbers for them to approve** rather than baking in silent defaults.
   - **"Good" / known failure modes** — anything specific to catch beyond the obvious.
   - **Ground truth** — where inputs/expected values live, if any.
3. **STOP — wait for answers.** End your turn. Do not present a build plan or write files until the
   user has answered.

---

## Phase 3 — Plan & Build (approval gate, then scaffold)

Only after the user answers the scoping questions. Goal: present a concrete plan, get approval,
then build. **No writes until the user approves.**

1. **Present the build plan** using the answers:
   - one workload → one report tab; how trajectories are generated (mirror the matching
     `references/<module>.md`);
   - the exact **binary checks** per workload (one failure mode per judge; deterministic $0 checks
     first, then LLM-as-judge), plus operational threshold checks with the agreed numbers;
   - the **dataset** (e.g. cases from ground truth, enriched with expected values);
   - aggregation: per-check pass rate + all-checks-pass rate (never averaged into one score);
   - the exact `evals/` layout to be written (below).
2. **STOP — approval gate.** Do **not** create or modify any file until the user approves. If they
   request changes, revise and re-present; only proceed on explicit approval.
3. **On approval, scaffold.** **All writes are confined to a single `evals/` directory in the
   workload repo. Write nowhere else.**

```
evals/
├── eval_config.yaml                    # cached discovery: workloads, modules, model IDs, region
├── datasets/<workload>.jsonl           # test cases / ground truth
├── judges/<workload>_<failuremode>.py  # ONE binary judge per failure mode
├── run_<workload>.py                   # cases → judges → aggregation → results/<workload>.json
├── results/<workload>.json             # per-workload structured results (schema below)
├── results.json                        # aggregated results across workloads (written in Phase 4)
└── report.html                         # final interactive summary (written in Phase 4)
```

Rules for what you generate:

- **Every LLM-as-judge is binary pass/fail**, returning `{"passed": true|false, "reason": "..."}`,
  with **one failure mode per judge**. Never emit 1–5 Likert scales or averaged scores.
- **Mirror the patterns in `references/<module>.md` exactly.** Use the binary judge templates,
  aggregation, and code patterns from those files. Never invent Bedrock APIs — if it isn't in a
  reference, don't fabricate it.
- Substitute the user's **model IDs and region** (from Phase 2) into the templates.
- Write `eval_config.yaml` first so subsequent runs can skip re-discovery; offer to re-discover if
  the repo has changed materially.

Each `run_<workload>.py` runs every case through each binary judge, then aggregates:
**per-check pass rate** and an **all-checks-pass rate** (a case passes overall only if every
check passes). It writes a per-workload result object to `results/<workload>.json`:

```json
{
  "workload": "customer-support-agent",
  "eval_type": "tool-calling",
  "checks": ["tool_selection", "efficiency", "response_quality"],
  "cases": [
    {
      "id": "tc-001",
      "verdicts": {
        "tool_selection": true,
        "efficiency": false,
        "response_quality": true
      },
      "overall_pass": false,
      "reason": { "efficiency": "duplicate lookup call" }
    }
  ],
  "summary": {
    "per_check_pass_rate": { "tool_selection": 0.9, "efficiency": 0.6 },
    "all_checks_pass_rate": 0.55,
    "n": 20
  },
  "error": null
}
```

`error` is `null` on success; if a workload's run fails, capture the message here so Phase 4 can
surface it without aborting the other workloads.

---

## Phase 4 — Run & Summarize (run gate)

Only after the build is complete. Goal: run the evals and emit one shareable HTML summary.

1. **STOP — confirm before running.** Running the evals calls the user's AWS/Bedrock account and
   **incurs cost**. Do not run until the user explicitly confirms.
2. On confirmation, run each `run_<workload>.py` to produce `results/<workload>.json`. If one
   workload errors, record the error in its result object and continue with the others.
3. **Generate the summary via the bundled script** (deterministic; do not hand-roll HTML per run):

   ```bash
   python3 scripts/build_report.py evals/results/ -o evals/report.html
   ```

   This emits a self-contained `evals/report.html` (inline CSS/JS, no external deps) with **one tab
   per workload / eval type** — each showing per-check pass-rate bars, the all-checks-pass gate, and
   an expandable per-case table with failure reasons — plus an aggregated `evals/results.json` for
   CI/diffing. Workloads with an `error` render an error panel instead of aborting the report.

---

## Rules (hard constraints)

- **One phase per turn.** Do a single phase, hit its STOP, and wait for the user. The four stops
  are: Phase 1 → confirm scope; Phase 2 → wait for scoping answers; Phase 3 → approval gate before
  any write; Phase 4 → run gate before any execution. Never chain phases in one response.
- **Binary pass/fail only.** Every generated judge returns `{"passed": bool, "reason": str}` with
  **one failure mode per judge**. Never 1–5 rating scales; never averaged scores. Decompose quality
  into specific binary checks and report **pass rates**.
- **Writes only under `evals/`, and only in Phase 3+.** Never create or modify any file outside the
  workload repo's `evals/` directory. During Phases 1–2, write nothing at all.
- **Mirror `references/` patterns; never invent Bedrock APIs.** All judge prompts, metrics, and
  code patterns come from the distilled `references/<module>.md` files. If a pattern isn't there,
  don't fabricate it.
- **No secrets to disk; use existing credentials only.** Rely on the user's current AWS
  session/credentials; never write secrets into `evals/` or anywhere else.
