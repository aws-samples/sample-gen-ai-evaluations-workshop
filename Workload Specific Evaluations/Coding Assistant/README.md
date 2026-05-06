# Coding Assistant Evaluations

**Run a real two-axis eval against three coding agents on a curated, prebuilt task set.** This workshop ships the task set, the rubrics, and the gold-standard calibration data already written; you read them, run the eval against Claude Code and Kiro, then fill in a custom-agent harness skeleton and add your own agent to the scorecard.

The two axes:

1. **Pair-programmer** — when you ask a question about the codebase, does the agent surface the right files (an IR problem) and answer correctly?
2. **Autonomous** — given a task, does it produce a mergeable diff reliably?

Why prebuilt data: if Claude writes the tasks *and* gets graded on them, you measure self-preference, not capability. The tasks, rubrics, and gold-standard examples in this workshop were authored by hand against a pinned SHA of [`aws-samples/sample-agentic-platform`](https://github.com/aws-samples/sample-agentic-platform). You can read them, trust them, and reproduce results.

## What you'll have at the end

1. A reproducible 9-task eval set: 5 normal autonomous tasks across difficulty bands, 2 trap tasks (issue describes a bug that doesn't exist — tests honesty), 2 nav-only tasks (deliverable is an answer, not a diff).
2. Calibrated rubrics: 7 ground-truth rubrics (one per non-nav task) + 8 hand-authored gold-standard diff/verdict pairs that prove the LLM judge agrees with a human reviewer ≥ 80 % of the time.
3. A working custom-agent harness (`my_agent/`) you filled in. Strands + Bedrock + your own tools.
4. **Pair-programmer scorecard** — precision@5, recall@10, MRR, answer accuracy, citation grounding, honesty.
5. **Autonomous scorecard** — pass rates by difficulty, reliability across seeds, sequence-aware tool-call quality, wall-clock efficiency.
6. A reusable PR-reviewer CI artifact.

## Why not SWE-Bench

Public benchmarks are fine for vendor comparison but wrong for deciding whether an agent will work on *your* codebase. Two reasons:

- **Relevance**: SWE-Bench tasks don't look like your code, use your MCP tooling, or follow your review standards.
- **Contamination**: SWE-Bench sources from public repos — providers almost certainly have the issues, PRs, and discussion in training data. Scores mix capability with leakage at an unknowable ratio.

This workshop uses a public stand-in (`aws-samples/sample-agentic-platform`) so everyone can follow along, but the structure generalizes: swap the repo URL in `tasks.yaml`, hand-curate new tasks following the schema docs in notebooks 02 and 03, run the eval.

## Notebooks

| | what you do |
|---|---|
| **01** | Verify env (AWS, Bedrock, CLIs on PATH). |
| **02** | Inspect the prebuilt tasks (`scaffolding/tasks/tasks.yaml`); learn the schema. |
| **03** | Inspect the prebuilt rubrics + gold standard; learn how rubrics calibrate the LLM judge. |
| **04** | Run the LLM judge against the gold set; verify ≥ 80 % agreement. |
| **05** | Fill in the `my_agent/` harness skeleton. The skeleton handles CLI plumbing; you write the agent loop and tools. |
| **06** | **Pair-programmer eval** — IR + correctness + grounding + honesty across all 3 agents. |
| **07** | **Autonomous eval + reliability + report** — full grid, two scorecards. |

## The two scorecards

### Pair-programmer (notebook 06)

| Metric | What it asks |
|---|---|
| precision@5 | Of the first 5 files the agent touched, how many were relevant? |
| recall@10 | Of the relevant files, how many surfaced in the first 10? |
| MRR | How quickly did the first relevant file appear? |
| answer accuracy | LLM judge against ground-truth answer |
| citation grounded | All `path:line` refs in answer exist and (optionally) support the claim |
| honesty | On trap tasks, did the agent refuse to fabricate a fix? |

### Autonomous (notebook 07)

| Metric | What it asks |
|---|---|
| review pass rate | Aligned LLM judge on rubric — mergeable by your standards? |
| tests pass rate | Repo's own pytest |
| static pass rate | Repo's own ruff |
| tools pass | Required tools called, forbidden ones avoided |
| sequence pass | Required tool called *before* first edit, query results consumed |
| reliability | Pass-rate across 3 seeds on the hardest tasks |
| seconds/task, seconds/correct task | Uniform wall-clock efficiency |
| input/output tokens | Custom agent only — Kiro can't be intercepted |

**Cost caveat**: Kiro is a closed desktop product whose Bedrock traffic can't be redirected. Wall-clock is the only uniform efficiency signal across all 3 agents. Do not read it as $/task — use it for within-agent tuning.

## Repo layout

```
Coding Assistant/
  01 … 07 *.ipynb                       the 7 notebooks
  scaffolding/                          PREBUILT — read, don't write
    task_schema_example.yaml            schema reference for tasks
    rubric_schema_example.md            schema reference for rubrics
    gold_entry_schema_example.yaml      schema reference for gold entries
    prompts.md                          copy-paste prompt library
    tasks/tasks.yaml                    9 curated tasks (5 normal + 2 trap + 2 nav)
    ground_truth/<task_id>.md           7 rubrics (one per non-nav task)
    gold_standard/<slug>.yaml           8 hand-authored verdicts (good/bad pairs)
    gold_standard/diffs/<slug>.diff     synthetic diffs paired with the verdicts
  my_agent/                             HARNESS SKELETON — you fill in
    __main__.py                           CLI plumbing (don't edit)
    trace.py                              trace recorder (don't edit)
    agent.py                              CodingAgent.run loop (TODO)
    model.py                              Bedrock model wiring (TODO)
    tools.py                              tool implementations (TODO)
  validators/
    tasks.py                              enforces qa_pairs/traps/nav_only structure
    rubrics.py
    gold_standard.py
    agent.py                              CLI contract check
    traces.py                             sequence-aware trace scoring
    retrieval.py                          precision@k / recall@k / MRR
    qa.py                                 judge_answer / check_citations / judge_honesty
    calibration.py
  utils/
    workspace.py                          per-(agent, task) sandboxes
    runners.py                            autonomous mode for all 3 agents
    qa_runner.py                          Q&A mode for all 3 agents
    checks.py                             pytest + ruff
    reporting.py                          both scorecards
  pr_reviewer/                          aligned PR reviewer (library + CLI)
  .github-action-example/
    pr-review.yml                       drop-in GitHub Action
  requirements.txt
```

## Prereqs

- AWS credentials with access to Bedrock and the model `us.anthropic.claude-sonnet-4-5-20250929-v1:0`.
- Python 3.10+, `uv`, `git`.
- Optional CLIs to evaluate as baselines:
  - `claude` (Claude Code CLI)
  - `kiro-cli` (Kiro CLI binary — note: the alias `kiro` doesn't work in subprocesses)
- `pip install -r requirements.txt`.

Notebook 01 verifies all of these.
