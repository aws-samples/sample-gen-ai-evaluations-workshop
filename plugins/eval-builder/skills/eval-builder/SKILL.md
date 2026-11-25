---
name: eval-builder
description: >
  Build, run, and review evaluations for YOUR OWN GenAI workload (agent, RAG, tool-calling,
  chatbot, coding assistant, IDP) using AWS GenAI Evaluations Workshop best practices. Use when
  the user wants to build/create/scaffold evals for their application, evaluate their agent/RAG/
  chatbot/code-review bot, set up an eval harness, or asks for "/eval-builder". Triggers on
  requests like "build evals for my app", "evaluate my agent", "scaffold tests for my RAG
  pipeline", "create pass/fail judges for my chatbot", "review my evals", or "are my evals any
  good?". NOT for learning eval concepts, tutorials, or Socratic practice — that is the tutor
  (learn mode). This skill takes action.
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

## Skill files

This skill ships one helper next to this `SKILL.md`:

- `scripts/build_report.py` — renders `evals/report.html` and `evals/results.json` from
  per-workload result files. **Always use it in Phase 4; never write your own report generator.**

Resolve its path from the directory that contains this `SKILL.md` (the skill directory your
tool loaded). Call it with an absolute path, for example
`python3 "<skill-dir>/scripts/build_report.py" evals/results -o evals/report.html`.
If you cannot determine the skill directory, search for it:
`find ~ -path "*eval-builder*/scripts/build_report.py" -not -path "*/node_modules/*" 2>/dev/null | head -1`.

## Workshop access (mandatory, no bundled references)

The skill reads the **live workshop content**. Patterns always come from the current workshop,
not stale copies. **You must locate the workshop in Phase 1 before anything else**, and you must
report where it is.

Find it in this order and use the first hit:

1. `$EVAL_WORKSHOP_DIR`, if set.
2. A directory containing `Foundational Evaluations/` in the working tree, a parent directory, or
   a sibling directory of the working tree.
3. `/tmp/eval-workshop-reference` from an earlier session (only if it contains
   `Foundational Evaluations/`).
4. Otherwise clone it:
   ```
   git clone --depth 1 https://github.com/aws-samples/sample-gen-ai-evaluations-workshop.git /tmp/eval-workshop-reference
   ```
   If the clone fails or the user declines, stop and tell the user the skill needs workshop
   access. **There is no offline fallback, and you must not continue from memory.**

If you cloned it, remove `/tmp/eval-workshop-reference` at the end of the session.

**Read the READMEs first** (`README.md`, `Foundational Evaluations/README.md`,
`Workload Specific Evaluations/README.md`, `Framework Specific Evaluations/README.md`), then the
modules that apply. Every pattern you use in Phase 3 must cite the workshop file it came from.

### Known pitfalls in workshop code — do not copy these

The workshop notebooks pin older library versions and model families. When you mirror their
patterns, apply these corrections:

- **Converse responses:** never read `response["output"]["message"]["content"][0]["text"]`.
  Reasoning models (for example Claude Sonnet 5) return a `reasoningContent` block first. Take
  the first block that has a `"text"` key, and set `maxTokens` high enough for the model to
  finish reasoning (several thousand for reasoning models). Treat a response with no text block
  as a judge **error**, never as a pass or a fail.
- **`strands-agents-tools`:** v0.9.0 removed `calculator`, `file_read`, `file_write`,
  `current_time`, and the code interpreter tool. Check the installed version before you import
  them.
- **`strands-agents-evals`:** `Experiment.run_evaluations()` returns one `EvaluationReport` in
  1.x, not a list. `run_display()` is interactive and waits for input; use `display()` in
  scripts.

## Module routing

The routing table maps workload signals to workshop modules. **Before you use a row, check that
its path exists in the workshop** — modules are added and renamed. If the READMEs list a module
that is not in this table, consider it too.

| Module | Path in workshop repo | When to read |
|--------|----------------------|--------------|
| Workshop overview | `README.md` | Always |
| Foundational overview | `Foundational Evaluations/README.md` | Always |
| Operational metrics | `Foundational Evaluations/01-operational-metrics/` | Always — operational check |
| Quality metrics / LLM-as-Judge | `Foundational Evaluations/02-quality-metrics/` | Always — binary judges, judge calibration |
| Understanding failures | `Foundational Evaluations/03-understanding-failures/` | Always — failure discovery and categorization |
| Agentic metrics | `Foundational Evaluations/04-agentic-metrics/` | Agent with tool use |
| Tool Calling | `Workload Specific Evaluations/Tool Calling/` | Workload calls tools/functions |
| MCP Tool Calling | `Workload Specific Evaluations/MCP Tool Calling/` | Tools are reached through MCP servers |
| Basic RAG | `Workload Specific Evaluations/Basic RAG/` | Retrieval-augmented generation |
| MultiModal RAG | `Workload Specific Evaluations/MultiModal RAG/` | RAG over images, audio, or video |
| Chatbot / multi-turn | `Workload Specific Evaluations/Chatbot/` | Multi-turn conversations with session state |
| Coding Assistant | `Workload Specific Evaluations/Coding Assistant/` | Code generation, bug fixing, or PR/code review |
| Guardrails | `Workload Specific Evaluations/Guardrails/` | Content filters, grounding, alignment checks |
| Intelligent Document Processing | `Workload Specific Evaluations/Intelligent Document Processing/` | Structured extraction from documents |
| Multiagent Shared Context | `Workload Specific Evaluations/Multiagent Shared Context Evaluation/` | Multiple coordinating agents |
| Red Teaming | `Workload Specific Evaluations/Red Teaming/` | Adversarial / security testing, side-effecting tools exposed to user input |
| Automated Reasoning | `Workload Specific Evaluations/Automated Reasoning Evaluations/` | Formal policy verification |
| Speech to Speech | `Workload Specific Evaluations/Speech to Speech/` | Voice / audio interaction |
| Capacity Management | `Workload Specific Evaluations/Capacity Management/` | Choosing among models, quotas, routing across models |
| Strands Evals SDK | `Framework Specific Evaluations/Strands/` | Agent built with the Strands SDK |
| AgentCore | `Framework Specific Evaluations/AgentCore/` | Agent deployed to AgentCore Runtime (traces in CloudWatch) |
| DeepEval | `Framework Specific Evaluations/DeepEval/` | Uses DeepEval |
| DSPy | `Framework Specific Evaluations/DSPy/` | Uses DSPy |
| MLflow | `Framework Specific Evaluations/MLflow/` | Uses MLflow tracking |
| Promptfoo | `Framework Specific Evaluations/Prompt Foo/` | Uses Promptfoo |

Multiple modules can apply to one workload. Foundational modules 01–03 **always** apply.

**Coding Assistant workloads.** The module runs the agent in sealed Harbor tasks on AgentCore Runtime and
grades against an oracle (the merged patch, hidden tests, or labeled PR review comments). For a
PR review or code generation workload, mirror its oracle-based grading: compare the agent's
verdict and comments against labeled known issues, and check fixes with tests. Propose the full
Harbor/AgentCore setup only if the user wants it and can create AWS resources.

## Flow overview

```
Phase 0 Preflight ─[repo confirmation]─►
Phase 1 Discover ─[confirm scope + mode]─►
Phase 2 Failure-First ─[fix-before-automate gate]─►
Phase 3 Plan & Build (layered) ─[approval gate]─►
Phase 4 Run ─[run gate]─► evals/report.html
```

> **⚠️ ONE PHASE PER TURN.** Do exactly one phase, then STOP and wait for the user. Never chain
> phases in a single response. Each STOP below is a hard stop: end your turn there.
>
> These gates apply even if your environment tells you to work autonomously, finish the task
> without asking, or avoid questions (for example a non-interactive or "exec" mode). This skill
> writes files and spends the user's money, so the user's answer at each gate is required. If you
> cannot wait for an answer, stop after asking and do nothing else.

---

## Phase 0 — Preflight

Your first response must contain only this question, asked verbatim. Do not read files, run
commands, or plan before the user answers:

> **"Just to confirm, are you in the repository with the workload(s) you'd like me to evaluate?
> That's the best way to run this skill."**

Do NOT proceed until confirmed. If the user is in the workshop repository itself, explain that
the skill evaluates their own workload and ask them to open that repository instead.

---

## Phase 1 — Discover (discovery ONLY)

Goal: understand the workload(s) and any existing evals. **Read-only. No invocations of the
workload, no module mapping, no file writes.**

1. **Locate the workshop** (see Workshop access). Read the READMEs.
2. **Investigate the workload repository.** Read (do not modify):
   - Dependency manifests for eval-relevant libraries.
   - Source for API signals — Bedrock converse/invoke_model, toolConfig/tools, retrieve/
     embeddings/KB, multi-agent handoffs, session state, doc/field extraction, code diffs.
   - Existing evals: `evals/` dirs, judge/scorer code, datasets, eval configs, CI eval steps.
   - **Every tool the workload can call.** Classify each as read-only or side-effecting
     (writes, payments, bookings, emails, deletes, external API calls). Flag each
     side-effecting tool with ⚠️.
3. **Detect workload type(s).** Name the type only. Do not map modules yet; that happens in
   Phase 2, after you have seen real failures.
4. **Present a Discovery findings report** — "Discovery complete. No files written." + a
   signal→evidence table (file:line). Include rows for: **workshop access (path, and whether it
   was found on disk or cloned)**, workload type, framework, tools (⚠️ for side-effecting),
   model/region, ground truth, guardrails, and **existing evals** (what/where/how many, or
   "none found").
5. **Mode selection:** if existing evals found → offer Review; if none → default Build; if
   ambiguous → ask.

**STOP — confirm scope + mode.** End your turn.

---

## Phase 2 — Failure-First (sample → categorize → fix gate)

Only after the user confirms scope and mode. Goal: understand what's *actually failing* before
building evals. **No file writes in the repository.** Scratch files go in a temporary directory
outside the repository.

### Safety rule for sampling

Before you invoke the workload, **replace every side-effecting tool with a record-only mock**
(for example, monkeypatch the function or pass a stub tool list) so the call is recorded but
nothing real happens. Redirecting a log file is not enough. If you cannot mock a side-effecting
tool, sample only inputs that cannot reach it, and say so. Never let sampling charge, book,
send, write, or delete anything real.

### Build mode

1. **Operational check.** Is the workload runnable? Can it be invoked? Any obvious errors?
2. **Sample outputs.** Run representative inputs (the repository's own samples first, then a few
   probes for edge cases you found in Phase 1) or read existing outputs/logs. Show them.
3. **Categorize failures.** Group failure modes by type, prioritized by frequency × severity,
   each with an example. Mark which are fixable now (prompt/config edit) and which need ongoing
   monitoring. Also list behaviors that pass but would be costly to break (regressions to lock
   down).
4. **Map to modules.** Map failures → workshop modules using the routing table. Read the
   relevant modules. State what's N/A and why.
5. **Fix-before-automate gate.** Suggest the prompt/config fixes. Ask: "Want to apply these fixes
   first, then re-sample, before I build evals for the rest?" Applying a fix is the only time you
   may edit the user's application code, and only after they say yes.
6. **Ask build-scoping questions** — judge model + region, operational thresholds, dataset size,
   optional framework layer, judge calibration. Offer demo-reasonable defaults for approval.

**Model IDs:** propose only model or inference profile IDs that you have confirmed exist in the
user's account and region:
```
aws bedrock list-inference-profiles --region <region> --query "inferenceProfileSummaries[].inferenceProfileId"
aws bedrock list-foundation-models --region <region> --query "modelSummaries[].modelId"
```
Never write a model ID from memory. If you cannot list models, ask the user for the ID.
Recommend a judge from a different model family than the model under test when one is
available, and say why (self-preference bias).

### Review mode

1. **Read existing eval code** — judges, datasets, configs, aggregation, reporting.
2. **Read the relevant workshop modules** to see what they teach about how evals should be
   built — binary pass/fail, one failure mode per judge, judge calibration, deterministic checks
   first, pass-rate reporting.
3. **Sample the workload** (with the safety rule above) to find failures the existing evals miss.
4. **Grade the existing evals against what the workshop teaches.** For each existing eval:
   - Binary pass/fail or a rating scale?
   - One failure mode per judge, or a multi-criteria mega-judge?
   - Calibrated/validated against human labels?
   - Deterministic checks where possible before LLM judges?
   - Pass rates, or averaged scores?
   - Is the judge the same model as the system under test?
   - Which failure modes does it cover vs. what the sampled failures and mapped modules call for?
5. **Present the review report** — per-eval findings with evidence (file:line), a coverage
   matrix, gaps, and a prioritized fix list plus new evals to fill gaps.

**STOP — wait for response.** Build mode: the user answers scoping questions or approves fixes.
Review mode: the user acknowledges findings and decides whether to remediate.

---

## Phase 3 — Plan & Build (layered, approval-gated)

Only after Phase 2 is resolved. Goal: present a concrete plan with **layered evaluation paths**,
get approval, then build.

### Layered paths (not parallel — each feeds the next)

| Layer | Always? | What it covers |
|---|---|---|
| **A: Foundational Quality** | Always | Output correctness — binary pass/fail checks per failure mode discovered in Phase 2. Deterministic checks ($0) first, LLM-as-judge for subjective dimensions. boto3 only. |
| **Agentic Process** | If tool-use / multi-step | HOW the agent reached the answer — tool selection, parameter accuracy, sequence, efficiency. Separate from WHAT it answered. |
| **B: Framework Evals** | If the framework SDK is in use | Framework-specific evaluators (Strands Evals, DeepEval, DSPy, MLflow, Promptfoo). Derived from the workshop. Informed by Layer A findings. |
| **C: AgentCore Evaluations** | If deployed to AgentCore Runtime with traces in CloudWatch | Built-in and custom AgentCore evaluators on recorded sessions. Informed by Layer A findings. |

**Feed-forward:** Layer A's failure categories scope what Layers B/C evaluate.

### Build plan contents

- Layers that apply, justified by Phase 2 findings.
- Per failure mode: which layer handles it, the exact binary question (one per check),
  deterministic vs. LLM judge, and the workshop file the pattern comes from.
- Dataset (cases from ground truth / sampled outputs, enriched with expected values).
- Aggregation: per-check pass rate (N/A cases excluded) + all-checks-pass. Never averaged.
- The `evals/` layout to be written, and any new dependencies (they go in
  `evals/requirements.txt`).
- Review-mode remediation: specific fixes to existing evals + new evals for gaps.

**STOP — approval gate.** Do NOT create or modify any file until the user approves.

### On approval, scaffold

```
evals/
├── README.md                           # how to install, re-run, and rebuild the report
├── eval_config.yaml                    # workloads, layers, model IDs, region, failure modes
├── requirements.txt                    # extra dependencies the evals need (never edit the app's own manifest)
├── datasets/<workload>.jsonl           # test cases / ground truth
├── checks/<workload>_<failuremode>.py  # deterministic checks ($0)
├── judges/<workload>_<failuremode>.py  # ONE binary LLM judge per failure mode
├── run_<workload>.py                   # cases → checks → judges → results/<workload>.json
├── results/<workload>.json             # per-workload results (schema below)
├── results/review.json                 # Review mode only (schema below)
├── results.json                        # written by build_report.py
└── report.html                         # written by build_report.py
```

Rules:
- **Every LLM judge is binary pass/fail** (`{"passed": bool, "reason": str}`), one failure mode
  per judge. Never 1–5. Never averaged scores. A judge call that errors or cannot be parsed is
  an **error**, not a verdict.
- **Mirror workshop patterns** and apply the known-pitfall corrections above.
- **Side-effecting tools are mocked; real ones are NEVER executed.** Evals assess the model's
  *decisions*, not real side effects. Add an assertion that fails the run if a real side effect
  happened.
- **Substitute the user's model IDs and region** (verified, as in Phase 2).
- **No time bombs in datasets.** Never hard-code dates or other values that go stale (for
  example a booking date). Store them relative to the run date (such as `{today+14}`) and fill
  them in when the eval runs, so the dataset stays valid next month.
- **Re-runnable without you.** Write `evals/README.md` with the exact setup and run commands
  (for example `pip install -r evals/requirements.txt`, then `python evals/run_<workload>.py`,
  then the `build_report.py` command). Never depend on a scratch environment or temp file from
  this session.
- Write `eval_config.yaml` first.
- In Review mode, edit the existing eval files in place only as approved in the plan.

### Results schema (`evals/results/<workload>.json`)

`build_report.py` reads this shape. Unknown keys are ignored.

```json
{
  "workload": "concierge",
  "eval_type": "agent",
  "checks": ["no_tag_leak", "booking_authorized", "missing_data_framing"],
  "summary": {
    "n": 28,
    "operational": {"latency_p95_s": 3.28, "total_tokens": 47557, "judge_calls": 54}
  },
  "judges": [
    {"name": "missing_data_framing", "model": "<judge model id>", "calibrated": false,
     "tpr": null, "tnr": null, "n_labels": null}
  ],
  "cases": [
    {"id": "q3", "input": "What's the population of Boise?",
     "verdicts": {"no_tag_leak": false, "booking_authorized": true, "missing_data_framing": false},
     "reasons": {"no_tag_leak": "reply contains <thinking>"},
     "error": null}
  ],
  "error": null
}
```

- A verdict of `null` means the check did not apply to that case (N/A). It is excluded from the
  check's pass rate.
- A case with `error` set never counts as a pass.
- Set the workload-level `error` if the run failed entirely; the report shows an error panel.
- Omit `per_check_pass_rate` and `all_checks_pass_rate`; the report computes them.

### Review schema (`evals/results/review.json`, Review mode only)

```json
{
  "summary": "One-paragraph verdict on the existing evals.",
  "findings": [
    {"severity": "high", "eval": "evals/llm_judge.py", "issue": "1-5 scale over four criteria",
     "evidence": "evals/llm_judge.py:11", "fix": "One binary judge per failure mode"}
  ],
  "coverage": [
    {"failure_mode": "forgets order number", "covered": false, "by": null,
     "module": "Workload Specific Evaluations/Chatbot"}
  ],
  "recommendations": ["Replace the averaged score with per-check pass rates"]
}
```

`covered` is `true`, `false`, or `"partial"`.

---

## Phase 4 — Run & Summarize (run gate)

Only after the build is complete.

1. **STOP — confirm before running.** Running calls the user's models and costs money.
2. On confirmation, run each `run_<workload>.py`. If one errors, record the error in its results
   file and continue.
3. **Generate the report with the bundled script** (absolute path, see Skill files):
   ```
   python3 "<skill-dir>/scripts/build_report.py" evals/results -o evals/report.html [--review evals/results/review.json]
   ```
   It writes `evals/report.html` (Summary, Results, and Review tabs; self-contained) and
   `evals/results.json` (aggregate for CI/diffing).
4. Summarize: pass rate per check, the all-checks-pass rate, the worst failures with examples,
   any uncalibrated judges, and any errors.
5. **Feed-forward:** "These results surface new failure patterns. Want to iterate — re-categorize
   failures and tighten the evals?"

---

## Rules (hard constraints)

- **One phase per turn.** Do a single phase, hit its STOP, wait for the user.
- **Workshop first.** Locate and read the workshop in Phase 1. If you can't access it, stop.
  Never build from memory.
- **Failure-first.** Never design evaluators without first looking at actual outputs.
- **Fix before automate.** Suggest prompt/config fixes before building eval infrastructure for
  trivially fixable failures.
- **Layered, feed-forward.** Foundational first; workload-specific builds on it; framework layers
  deepen it.
- **Binary pass/fail only.** `{"passed": bool, "reason": str}`, one failure mode per judge. Never
  1–5 scales. Never averaged scores. Report pass rates.
- **No real side effects, ever** — not in sampling, not in evals. Mock side-effecting tools.
- **Writes only under `evals/`, Phase 3+ only.** The one exception is an application fix the user
  approved at the Phase 2 gate. Never edit the application's dependency manifests; put eval
  dependencies in `evals/requirements.txt`.
- **Verified model IDs only.** List them from the account; never invent one.
- **Use `scripts/build_report.py` for the report.** Don't write your own.
- **No secrets to disk; use existing credentials only.**
- **Evaluate the evaluator.** Mark every judge that hasn't been validated against human labels
  as uncalibrated in the results and in your summary.
