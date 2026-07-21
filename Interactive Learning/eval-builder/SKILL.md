---
name: eval-builder
description: >
  Build, run, and summarize evaluations for YOUR OWN GenAI workload (RAG, agent, tool-calling,
  chatbot, IDP) using AWS GenAI Evaluations Workshop best practices. Use when the user wants to
  build/create/scaffold evals for their application, evaluate their agent/RAG/chatbot, set up an
  eval harness, or asks for "/eval-builder". Triggers on requests like "build evals for my app",
  "evaluate my agent", "scaffold tests for my RAG pipeline", or "create pass/fail judges for my
  chatbot". NOT for learning eval concepts, tutorials, or Socratic practice — that is the tutor
  (learn-mode). This skill takes action: it discovers your workload, proposes a plan, and (after
  you approve) scaffolds and runs binary pass/fail evals.
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

The flow has four phases and two hard stops:

```
Phase 1 Discover ─► Phase 2 Map & Report ─[APPROVAL GATE]─► Phase 3 Build ─► Phase 4 Run ─[RUN GATE]─► evals/report.html
```

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

## Phase 1 — Discover

Goal: understand the workload(s) and what "good" means to the user. **No file writes, no execution
in this phase.**

1. **Investigate the repository.** Read (do not modify):
   - dependency manifests (`requirements.txt`, `pyproject.toml`, `package.json`) for eval-relevant
     libraries (boto3, `strands`, `bedrock-agentcore`, vector-store/retriever libs, `jsonschema`);
   - source for API signals — Bedrock `converse` / `invoke_model`, `toolConfig` / `tools=[...]`,
     `retrieve(` / embeddings / knowledge base, multi-agent handoffs / shared memory, conversation
     / session state, document/field extraction;
   - any existing `evals/`, tests, datasets, or ground-truth fixtures.
2. **Detect workload type(s)** using `references/module-index.md` (detection signals → workload
   type → modules). More than one type can match a single repo.
3. **Ask targeted questions to fill gaps:**
   - What does "good" look like for this workload? What are the known failure modes?
   - Which Bedrock **model IDs** and **provider** are in use (agent model, judge model)?
   - Which **AWS region**?
   - Where are inputs/ground truth, if any exist?
4. **If the workload type cannot be confidently determined, ask the user to confirm or choose** —
   never assume.

---

## Phase 2 — Map & Report ⟶ **STOP: await approval**

Goal: present a plan the user approves before anything is built.

1. **Map** each detected workload to workshop module(s) via `references/module-index.md`.
   `operational` and `quality` are near-universal baselines; workload-specific modules
   (tool-calling, agentcore, multiagent-context, rag, chatbot, structured-data) layer on top.
   Multiple matches → multiple report tabs later.
2. **Produce a report** containing:
   - (a) **Workloads found** (with the detection signals that identified each);
   - (b) **Relevant modules** and a one-line summary of what each recommends (drawn from the
     matching `references/<module>.md`);
   - (c) **Specific evals proposed** — for each workload, the exact binary checks (one failure mode
     per judge), the test-case structure, and the model IDs/region to use.
3. **Present the report and STOP.** Explicitly ask for approval. Do **not** create or modify any
   file until the user approves.
4. **If the user requests changes**, revise and re-present the report; only proceed on explicit
   approval.

---

## Phase 3 — Build

Goal: scaffold the approved evals, adapted to the user's workload. **All writes are confined to a
single `evals/` directory in the workload repo. Write nowhere else.**

Scaffold this layout:

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
- Substitute the user's **model IDs and region** (from discovery) into the templates.
- Write `eval_config.yaml` immediately after approval so subsequent runs can skip re-discovery;
  offer to re-discover if the repo has changed materially.

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

## Phase 4 — Run & Summarize ⟶ **STOP: confirm before run**

Goal: run the evals and emit one shareable HTML summary.

1. **STOP and confirm before executing anything.** Running the evals calls the user's AWS/Bedrock
   account and **incurs cost**. Do not run until the user explicitly confirms.
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

- **Binary pass/fail only.** Every generated judge returns `{"passed": bool, "reason": str}` with
  **one failure mode per judge**. Never 1–5 rating scales; never averaged scores. Decompose quality
  into specific binary checks and report **pass rates**.
- **Writes only under `evals/`.** Never create or modify any file outside the workload repo's
  `evals/` directory. During discovery (Phases 1–2), write nothing at all.
- **Confirm before executing.** Phase 2 has an approval gate (no build without approval); Phase 4
  has a run gate (no execution without confirmation, because it costs money).
- **Mirror `references/` patterns; never invent Bedrock APIs.** All judge prompts, metrics, and
  code patterns come from the distilled `references/<module>.md` files. If a pattern isn't there,
  don't fabricate it.
- **No secrets to disk; use existing credentials only.** Rely on the user's current AWS
  session/credentials; never write secrets into `evals/` or anywhere else.
