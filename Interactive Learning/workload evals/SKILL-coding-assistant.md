---
name: Coding Assistant Evaluation
description: Help me evaluate coding agents (a PR review agent and a coding agent that writes the fix) with datapoints from git history, sealed Harbor tasks, oracles, hidden tests, and binary pass/fail gates. Activate when asked to "evaluate a coding agent", "evaluate a PR review agent", "grade a code fix", "build SWE-bench style tasks", or "use Harbor tasks".
---

# Evaluating Coding Assistants

A coding agent works on a full repository, not a prompt. To evaluate it, you need an example with a known correct answer, an environment where the agent cannot find that answer, and a grader that a different correct answer also passes. This module teaches you to build these from your own git history. It uses two agent types: a **PR review agent** that finds defects in a change, and a **coding agent** that writes the fix for an issue.

The module has two paths:

- **Path A (no Docker, no AWS):** reason over the prepared task files (`instruction.md`, `oracle.json`, `oracle.patch`, `rubric.toml`, `validation_spec.toml`) and over a sample `reward_details.json`. Every code block in this file is Path A. For real agent outputs, use the recorded runs in `Workload Specific Evaluations/Coding Assistant/datasets/recorded-runs/` (Codex and Claude Code, with trajectories and verifier files) and the notebook `04 Replay recorded runs (no Docker).ipynb`.
- **Path B (optional, hands-on):** run the oracle, Codex, and optionally Claude Code in Harbor on Docker with the notebooks of the module. See Section 5.

## Prerequisites

- Python 3.11 or later (for `tomllib`). Path A needs nothing else.
- Run the code from the repository root. The module is in `Workload Specific Evaluations/Coding Assistant/`.
- Completed SKILL-quality (binary LLM judges) and SKILL-agentic (agent trajectories), or equivalent understanding.
- Path B only: Docker with approximately 15 GB of free disk space, Python 3.12, Harbor, the Codex CLI, and AWS credentials with Amazon Bedrock access in `us-east-1`.

## Learning Objectives

By the end of this module, you will be able to:

1. Explain why a coding datapoint comes from git history: roll back to the parent commit, seal the environment, and use the review or the merged patch as the oracle
2. Separate the files that the agent sees from the files that only the verifier sees, and find a leak with a leakage scan that tests itself with a canary
3. Grade a PR review with location matching, coverage, and precision, and explain why precision is undefined when the agent reports no findings
4. Grade a code fix with a stack of binary gates: hidden tests, a validation agent, a task rubric, a taste judge, and patch bloat
5. Calibrate a task with the oracle and a no-op agent, and sort each failure into an agent, datapoint, scorer, or environment cause

## Setup

```python
import json
import re
import shutil
import tempfile
import tomllib
from pathlib import Path

MODULE = Path("Workload Specific Evaluations/Coding Assistant")
assert MODULE.is_dir(), "Run this from the repository root."

PR_TASKS = MODULE / "datasets/awslabs-mcp-pr-review/tasks"
CODING_TASKS = MODULE / "datasets/strands-evals-coding/tasks"
REJECT = PR_TASKS / "awslabs__mcp__pr4615-reject"      # PR review: the reviewer blocked this change
APPROVE = PR_TASKS / "awslabs__mcp__pr4121-approve"    # PR review: two reviewers approved this change
ROOT_FIX = CODING_TASKS / "strands-agents__evals__pr371-fix"     # coding: a 7-line fix
HISTORY_FIX = CODING_TASKS / "strands-agents__evals__pr338-fix"  # coding: a 40-line fix


def read_toml(path):
    return tomllib.loads(Path(path).read_text())


def read_json(path):
    return json.loads(Path(path).read_text())


for task in (REJECT, APPROVE, ROOT_FIX, HISTORY_FIX):
    files = sorted(str(p.relative_to(task)) for p in task.rglob("*") if p.is_file())
    print(f"{task.name}: {len(files)} files")
```

Each task is one [Harbor](https://github.com/laude-institute/harbor) task directory: `task.toml`, `instruction.md`, `environment/` (the Dockerfile of the agent container), `tests/` (the verifier), and `solution/` (the oracle agent).

## Section 1: Datapoints From Git History

**Concept:** Each merged PR already records the code before the change, the change, the review, and the fix that the reviewer approved. That is a labeled example. A datapoint is built in three steps:

1. **Roll back** to the parent commit of the change. The agent sees the code as the reviewer saw it.
2. **Seal** the environment. Remove every path to the answer: later commits, the remote, GitHub, and the oracle.
3. **Use the review or the merged patch as the oracle.** For a PR review agent, the oracle is the list of defects. For a coding agent, it is the merged fix and hidden tests.

The two agent types use the same history, at different points:

| | PR review datapoint | Coding datapoint |
|---|---|---|
| Start state | The base commit, with the change applied | The base commit, with no change |
| Instruction | The same for all tasks: review the change | An issue report, different for each task |
| Agent output | A list of findings and a verdict | Changes to the code |
| Oracle | The defects that the reviewers found | The merged fix and hidden tests |

**Explore which files each side sees:**

```python
def who_sees(task_dir):
    """Harbor puts instruction.md and the environment/ image in front of the agent. tests/ and solution/ are verifier only."""
    for path in sorted(p for p in task_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(task_dir)
        if rel == Path("instruction.md") or rel.parts[0] == "environment":
            side = "agent"
        elif rel.name == "task.toml":
            side = "harbor (settings)"
        else:
            side = "verifier only"
        print(f"  {side:<18} {rel}")

for task in (REJECT, ROOT_FIX):
    print(task.name)
    who_sees(task)
```

**Tutor questions:** Ask the learner which single file in `ROOT_FIX` would end the eval if the agent could read it, and why. Then ask why the PR review instruction is identical for every task, while the coding instruction is different for each task.

## Section 2: Sealing the Environment and Scanning for Leaks

**Concept:** A seal has three layers. If the agent can find the answer, the score measures search, not skill.

| Layer | Path to the answer | Control | How to check it |
|-------|--------------------|---------|-----------------|
| Git | `git log --all` shows the fix commit | Clone only the base commit; remove the remote, the remote refs, and the reflog | In the image: one commit, no remote |
| Network | `gh pr view` shows the review | `network_mode = "allowlist"` with only the LLM endpoint | `task.toml`, and a run-time egress check |
| Verifier | The agent reads `tests/` or `solution/` | A separate verifier, or `tests/` copied in only after the agent stops | `task.toml` |

**Read the network seal from `task.toml`:**

```python
for task in (REJECT, ROOT_FIX):
    cfg = read_toml(task / "task.toml")
    env, verifier = cfg["environment"], cfg["verifier"]
    print(task.name)
    print(f"  agent:    {env['network_mode']} {env.get('allowed_hosts')}")
    print(f"  verifier: {verifier.get('environment_mode', 'shared')}, allowed {verifier.get('allowed_hosts', 'public')}")
    assert env["network_mode"] == "allowlist"
    assert not any("github" in host for host in env["allowed_hosts"])
```

The PR review verifier runs in a **separate** container (it needs only the findings file). The coding verifier runs in the **agent's** container after the agent stops, because it must run the agent's code. Ask the learner what this means for when `tests/` may appear in that container.

**A leakage scan that tests itself.** Search every agent-visible file for strings that would give away the answer: PR and issue numbers, the merge commit, the hidden test file name, and long lines that the fix adds. First, plant a canary in a temporary copy of the task. If the scan does not find the canary, the scan is broken, and "no leaks" means nothing.

```python
def agent_visible_files(task_dir):
    yield task_dir / "instruction.md"
    yield from (p for p in sorted((task_dir / "environment").rglob("*")) if p.is_file())


def scan(task_dir, needles):
    hits = []
    for path in agent_visible_files(task_dir):
        text = path.read_text(errors="ignore").lower()
        hits += [(str(path.relative_to(task_dir)), n) for n in needles if n and n.lower() in text]
    return hits


def leakage_scan(task_dir, needles, canary="CANARY-7f3a91"):
    with tempfile.TemporaryDirectory() as tmp:  # never write into the real task
        copy = Path(tmp) / task_dir.name
        shutil.copytree(task_dir, copy)
        (copy / "environment" / "canary.txt").write_text(canary)
        assert scan(copy, [canary]), "The scan did not find the canary. The scan has a defect."
    return scan(task_dir, needles)


def coding_needles(task_dir):
    meta = read_toml(task_dir / "task.toml")["metadata"]
    origin = meta["origin"]
    needles = [f"#{n}" for n in origin["pr_numbers"] + origin["issue_numbers"]]
    needles += [f"pull/{n}" for n in origin["pr_numbers"]] + [origin["merge_commit"][:12]]
    needles += [Path(meta["verifier"]["hidden_tests"]).stem]
    patch = (task_dir / "solution" / "oracle.patch").read_text()
    needles += [l[1:].strip() for l in patch.splitlines() if l.startswith("+") and not l.startswith("+++") and len(l.strip()) >= 40]
    return needles


for task in (ROOT_FIX, HISTORY_FIX):
    needles = coding_needles(task)
    print(f"{task.name}: {len(needles)} needles, leaks: {leakage_scan(task, needles) or 'none'}")
```

**Git seal (Path B, needs Docker):** after the image is built, this must print `1 0 0` for a coding task (one commit, no remote, clean tree) and `1 0 <n>` with `n > 0` for a PR review task (the change is applied as uncommitted edits).

```bash
docker run --rm eval-strands-agents__evals__pr371-fix sh -c \
  'echo "$(git log --all --oneline | wc -l) $(git remote | wc -l) $(git status --porcelain | wc -l)"'
```

**Tutor questions:** Ask the learner what a leakage scan cannot find (for example, a public blog post about the fix that the allowlist would block anyway, or a test name that hints at the cause). Then ask why a passing `task.toml` check is not proof that the network is sealed at run time.

## Section 3: Grading a PR Review Agent

**Concept:** The agent writes findings: a file, a line, a statement, and a severity. The scorer matches each oracle finding to at most one agent finding in two steps. First, the agent line must be inside a **location span** of the oracle finding. Then an LLM judge decides if the two statements describe the same defect. With no judge (location only), the match is **strict**.

From the matches come these signals:

| Signal | Question | Empty denominator |
|---|---|---|
| `coverage_required` | Of the high (blocking) oracle findings, what fraction did the agent find? | 1.0 (nothing to miss) |
| `coverage_all` | Of all oracle findings, what fraction did the agent find? | 1.0 |
| `precision_strict` / `precision_adjudicated` | Of the agent findings, what fraction matched? | **Undefined** (no findings) |
| `uncredited` | How many agent findings matched nothing? | 0 |

An unmatched agent finding is **uncredited**, not wrong: reviewers do not find every defect.

**Look at the oracle of the reject task:**

```python
oracle = read_json(REJECT / "tests" / "oracle.json")
print("expected verdict:", oracle["expected_verdict"])
for f in oracle["findings"]:
    spans = ", ".join(f"{l['line_start']}-{l['line_end']}" for l in f["locations"])
    flag = " (reference only)" if f["reference_only"] else ""
    print(f"  {f['severity']:<6} {f['locations'][0]['path'].split('/')[-1]}:{spans}{flag}")
```

**A strict scorer and the pass/fail rule.** The verdict comes from the findings: `block` if any finding is high. A task passes if the verdict is correct and, for a reject task, every high finding is covered.

```python
def strict_review_score(oracle, findings):
    claimed, matched = set(), []
    for o in oracle["findings"]:
        for i, a in enumerate(findings):
            inside = any(l["path"] == a["path"] and l["line_start"] <= a["line"] <= l["line_end"] for l in o["locations"])
            if inside and i not in claimed:
                claimed.add(i)
                matched.append(o)
                break
    high = [o for o in oracle["findings"] if o["severity"] == "high"]
    ratio = lambda num, den: num / den if den else 1.0
    verdict = "block" if any(a["severity"] == "high" for a in findings) else "approve"
    result = {
        "coverage_required": ratio(sum(o["severity"] == "high" for o in matched), len(high)),
        "coverage_all": ratio(len(matched), len(oracle["findings"])),
        "precision_strict": len(matched) / len(findings) if findings else None,  # undefined, not 0
        "uncredited": len(findings) - len(matched),
        "verdict": verdict,
    }
    ok = verdict == oracle["expected_verdict"] and (verdict == "approve" or result["coverage_required"] == 1.0)
    return result, "PASS" if ok else "FAIL"


DETECTOR = "src/mysql-mcp-server/awslabs/mysql_mcp_server/mutable_sql_detector.py"
agents = {
    "silent": [],
    "blocks everything": [{"path": "README.md", "line": 1, "severity": "high", "statement": "Risky change."}],
    "finds the bypass": [{"path": DETECTOR, "line": 250, "severity": "high", "statement": "A '#x' comment hides the verb."}],
}
for task in (REJECT, APPROVE):
    for name, findings in agents.items():
        print(f"{task.name:<32} {name:<18}", *strict_review_score(read_json(task / "tests" / "oracle.json"), findings))
```

**Tutor questions:** Ask the learner to predict each of the six rows before running the cell. Then ask: on the approve task, the silent agent is correct, so why must its precision be undefined and not 0 or 1? (In the module, `score.py` leaves an undefined precision out of `reward.json`, because Harbor accepts only numbers there, and writes it as `null` in `reward_details.json`. The dataset metric skips it.) Finally, ask which dataset metric catches the "blocks everything" agent: block recall, or false block rate?

## Section 4: Grading a Code Fix With Binary Gates

**Concept:** A fix that passes the tests is not always a fix that a maintainer would merge. Following [Senior SWE-Bench](https://senior-swe-bench.snorkel.ai/blog/2026-06-16-how-it-works), the verifier runs a stack of mechanisms. Teach each one as a **binary gate**: it passes or it fails. The reward is 1 only if every gate passes.

| Gate | What it checks | Pass if |
|------|----------------|---------|
| `verifiers` | Hidden fail-to-pass (f2p) and pass-to-pass (p2p) tests, and the base unit test suite | every test passes |
| `validation` | Tests that a validation agent writes from `validation_spec.toml`, adapted to the agent's change | every story passes |
| `rubric` | Task criteria from `rubric.toml` that a test cannot check well, each judged yes or no | more than `rubric_min` of the criteria |
| `patch_bloat` | Source lines of the agent change divided by source lines of the merged fix | less than `bloat_max` |
| `relative_taste`, `practice_alignment` | A taste judge answers ten yes/no checks: five compare the change with the merged fix, five check its fit with the conventions of the repository | at least `taste_min_pass_rate` of the checks in each group |

`correct` is the first two gates (the change works). `reward` (a "tasteful solve") is all six. Agent-written tests are reported, but they do not gate.

**Read the gate thresholds, the rubric, and the validation stories:**

```python
models = read_toml(ROOT_FIX / "tests" / "judge" / "models.toml")
GATES = models["gates"]
print("gates:", GATES)

for c in read_toml(ROOT_FIX / "tests" / "judge" / "rubric.toml")["criteria"]:
    print(f"  rubric [{c['kind']}] {c['id']}: {c['text'][:90]}")

spec = read_toml(ROOT_FIX / "tests" / "validate" / "validation_spec.toml")
for story in spec["stories"]:
    print(f"  story {story['id']}: {len(story['cases'])} cases")
```

**Patch bloat, from the patches only.** Count changed source lines. Tests, documentation, changelogs, blank lines, and comments do not count.

```python
NOT_SOURCE = re.compile(r"(^|/)(tests?|docs?)/|(^|/)test_[^/]*\.py$|_test\.py$|\.(md|rst|txt)$|CHANGELOG", re.I)


def source_sloc(patch):
    count, counted = 0, False
    for line in patch.splitlines():
        if line.startswith("diff --git"):
            counted = not NOT_SOURCE.search(line.split(" b/")[-1])
        elif counted and line[:1] in "+-" and not line.startswith(("+++", "---")):
            body = line[1:].strip()
            count += bool(body) and not body.startswith("#")
    return count


for task in (ROOT_FIX, HISTORY_FIX):
    oracle_sloc = source_sloc((task / "solution" / "oracle.patch").read_text())
    reference_sloc = source_sloc((task / "tests" / "judge" / "reference.patch").read_text())
    print(f"{task.name}: oracle {oracle_sloc} SLOC, reference {reference_sloc} SLOC, bloat {oracle_sloc / reference_sloc:.2f}")
```

**Apply the gates to a sample trial.** This `reward_details.json`-shaped example is illustrative (not from a real run). It has the same keys that `tests/lib/aggregate.py` reads.

```python
sample = {
    "verifiers": {"fail_to_pass": 1.0, "pass_to_pass": 1.0, "existing_suite": 1.0, "pass": True},
    "validation": {"status": "accepted", "passed_stories": 2, "total_stories": 2},
    "rubric": {"rubric_score": 0.67},
    "taste": {"relative_taste": 0.4, "practice_alignment": 0.8},  # fraction of yes/no checks that pass
    "bloat": {"agent_sloc": 34, "reference_sloc": 15, "patch_bloat": 2.27},
}


def gates(d):
    v = d["validation"]
    checks = {
        "verifiers": d["verifiers"]["pass"],
        "validation": v["status"] == "accepted" and v["passed_stories"] == v["total_stories"],
        "rubric": d["rubric"]["rubric_score"] > GATES["rubric_min"],
        "patch_bloat": d["bloat"]["patch_bloat"] is not None and d["bloat"]["patch_bloat"] < GATES["bloat_max"],
        "practice_alignment": d["taste"]["practice_alignment"] >= GATES["taste_min_pass_rate"],
        "relative_taste": d["taste"]["relative_taste"] >= GATES["taste_min_pass_rate"],
    }
    return checks, checks["verifiers"] and checks["validation"], all(checks.values())


checks, correct, tasteful = gates(sample)
for name, ok in checks.items():
    print(f"  {'pass' if ok else 'FAIL'}  {name}")
print(f"correct={correct} tasteful_solve={tasteful}")
```

> **Note:** Every taste dimension is one yes/no question (for example, "Is the change focused on the task, with no unrelated changes?"), not a 1-5 score. LLM judges that score on a scale cluster at 4 or 5 whatever the quality, and a yes/no question is easier to calibrate against a human label. Senior SWE-Bench uses 1-5 scores; this module converts each to a yes/no check with about the same bar.

**Tutor questions:** Ask the learner to name the gates that fail in the sample, and what the agent probably did (hint: compare the 34 changed source lines with the 15 of the merged fix). Then ask why the rubric does not gate a solve alone: a no-op agent gets some rubric credit, because an empty change satisfies "no interface change". Finally, ask what a hidden f2p test must not do so that a *different* correct fix also passes (it must use only the public API, and check behavior, not one design).

## Section 5: Calibrate, Run, and Diagnose

**Concept:** Before a score tells you anything about an agent, test the grader on two known answers:

| Run | Expected result | If not |
|-----|-----------------|--------|
| **Oracle** (`solution/`: submits the oracle findings or applies `oracle.patch`) | A perfect score: every gate passes | The scorer or a test is broken |
| **No-op agent** (does nothing) | `fail_to_pass = 0`, `correct = 0`, `reward = 0`; `pass_to_pass = 1` and `existing_suite = 1` | An f2p test does not test the bug, or a p2p test is broken |

Senior SWE-Bench runs each 3 times and keeps a task only if all 3 agree. If an LLM stage fails (for example, throttling) the trial is **invalid**: there is no `reward.json`, and it is not counted as an agent failure.

**Check the calibration rules on two sample outcomes (Path A):**

```python
def calibration_ok(run, flat):
    if run == "oracle":
        return flat["reward"] == 1.0
    return (flat["fail_to_pass"] == 0.0 and flat["correct"] == 0.0 and flat["reward"] == 0.0
            and flat["pass_to_pass"] == 1.0 and flat["existing_suite"] == 1.0)


nop = {"reward": 0.0, "correct": 0.0, "fail_to_pass": 0.17, "pass_to_pass": 1.0, "existing_suite": 1.0}
oracle_run = {"reward": 1.0, "correct": 1.0, "fail_to_pass": 1.0, "pass_to_pass": 1.0, "existing_suite": 1.0}
print("oracle:", calibration_ok("oracle", oracle_run), " no-op:", calibration_ok("nop", nop))
```

**Sort each failure into one cause.** The score tells you that the agent failed; the trajectory and the patch tell you why.

| Cause | Example | What to do |
|-------|---------|------------|
| The agent | It patched the symptom in the evaluator, not the cause in the extractor | Improve the agent (prompt, tools, procedure) |
| The datapoint | An approved PR that a later commit fixed: the agent that blocked it was right | Correct the oracle, or select a different PR |
| The scorer | A correct finding is one line outside the span; a hidden test checks one design | Widen the span; rewrite the test on the public API |
| The environment | A tool is missing in the image | Repair the Dockerfile and rebuild |

**Path B (optional, needs Docker and AWS):** open the notebooks in `Workload Specific Evaluations/Coding Assistant/` in order. Notebook 01 does the setup checks. Notebooks 02 and 03 build the images (an image that exists is skipped), check the seal, run the oracle, and run Codex. The helpers are in `coding_eval.py`. The verifier files that every task shares are in `datasets/_shared/`; after you edit one, run `python datasets/_shared/sync.py` from the module directory. To check the module code without Docker, run `python -m pytest tests_local` there.

**Tutor questions:** Ask the learner what the no-op `fail_to_pass = 0.17` above means, and which file they would open first. Then ask how many trials per task they need before they call a difference between two agents real, with two tasks and pass@1.

## Challenges

### Challenge: Turn One of Your Own PRs Into a Sealed Datapoint

Pick a merged PR from a repository you know (or use a hypothetical one, and write its diff). Build a datapoint for **one** agent type: a PR review datapoint or a coding datapoint.

1. Write `instruction.md`. For a coding datapoint, write it as an issue report: the symptom and the expected behavior, without the file, the function, or the cause.
2. Write the oracle. For a PR review datapoint: `oracle.json` with at least one high finding that has two location spans. For a coding datapoint: the names and one-line intent of at least 2 f2p tests and 1 p2p test that use only the public API.
3. Define at least 5 leakage needles, and run `leakage_scan` from Section 2 on your task directory (a directory with `instruction.md` and an `environment/Dockerfile` is enough).
4. Write one concrete example of a change that fails each of two taste checks (for example, minimality and documentation fit), and one that passes, so a human can label the judge's answers.

**Constraint:** Two reviewers approved your PR, but six weeks later another commit changed the same lines. Decide whether you can use it as an approve datapoint, and what you would check first.

**Assessment criteria:**
1. `leakage_scan` passes its canary test and reports 0 hits for ≥5 needles that include the PR number and at least one line from the fix
2. The oracle is complete: for PR review, ≥1 high finding with ≥2 spans and an `expected_verdict` that agrees with the high findings; for coding, ≥2 f2p and ≥1 p2p tests, each with a reason why it fails or passes on the base commit
3. The instruction names no file, function, PR number, or reviewer (the tutor checks the text against the needles)
4. ≥2 taste checks have one failing and one passing example
5. The approve-datapoint decision names the later commit as evidence and states one check (for example, compare the later diff with the approved lines) before the datapoint is used

For a cross-module integrative challenge, see `CHALLENGE-capstone.md`.

## Wrap-Up

**Key takeaways:**
- Your PR history is a labeled dataset: roll back to the parent commit, seal the environment, and use the review or the merged patch as the oracle
- A seal has three layers (git, network, verifier). Check each one, and test the leakage scan itself with a canary
- A PR review is graded on coverage and precision separately; precision is undefined when there are no findings
- A code fix passes only if every binary gate passes: tests, validation stories, rubric, bloat, and taste
- Run the oracle (must pass) and a no-op agent (must fail) before you trust any agent score

**Key metrics to track:**
- PR review: accuracy, block recall, and false approve rate (the most expensive error)
- Coding: resolved rate (mean `correct`) and tasteful-solve rate (mean `reward`); the gap between them is fixes that work but would not be merged

**This module does NOT cover:**
- Making datapoints at scale with `eval-harvest` (the module uses prepared datapoints)
- Running trials on Amazon EC2 or Lambda MicroVMs instead of local Docker
- Feature tasks, where the interface is not known before the agent builds it

**Next steps:**
- Build one approve and one reject datapoint from your own repository
- Explore `CHALLENGE-capstone.md` for an integrative challenge that combines evaluation approaches across modules
