# Three ways to run the Coding Assistant evaluation: Docker, Local, AgentCore

This page compares the three ways to run the tasks of this module and says what was measured. It is for testers who want to pick a mode. The numbers come from the project's internal run notes (one laptop, one AWS account, one day, 2026-11-20). **measured** means timed or counted in those notes. **estimate** means derived or summed, not timed end to end. Setup and commands are in each mode's own guide: [the module README](../README.md), [agentcore/README.md](../agentcore/README.md) and [local/README.md](../local/README.md).

Background: the module grades a coding agent (Codex on Amazon Bedrock) on bug-fix and pull-request-review tasks. The notebooks run each trial in Docker with Harbor. Two lighter ways were built and checked against it. **Local** runs everything on a Mac without Docker. **AgentCore** runs each trial in its own Amazon Bedrock AgentCore Runtime session, so the laptop holds almost nothing.

## 1. Summary

| | Docker (baseline) | Local | AgentCore |
|---|---|---|---|
| What runs where | Harbor and Docker on your laptop; Codex on Bedrock | Your Mac only: sealed git clone, `uv` virtual environment, `srt` sandbox, one small runner | One AgentCore Runtime session per trial; your laptop runs one Python script |
| Setup steps (manual) | 19 (5 manual, 2 more conditional) | 6 (1 manual: AWS credentials) | 4 (2 manual); 3 (1 manual) inside a workshop checkout |
| Time to first result | 13.3 min, estimate (11.5 min setup, then 1 min 46 s to the first PR-review reward) | about 3 min, estimate, from an empty machine; the first review trial itself took 48.7 s, measured | 4.3 min, measured, from an empty prefix to the first PR-review reward |
| Time for 12 Codex trials | trial times add up to 40.4 min, run 2 at a time; wall clock for exactly these 12 was not recorded | 37.7 min, measured, one at a time | 7.6 min, measured, all at once; 60 trials in two waves of 30 took about 16 min |
| Disk, steady | 15.8 GB | 2.2 GB | 0.15 GB on the laptop, plus 7.1 GB of images in ECR |
| Bedrock cost, 12 Codex trials (6 coding, 6 review) | $2.33 agent, measured, plus about $0.2 judges, estimate | about $2.6 ($2.46 agent plus $0.15 judges), estimate | about $3.4, estimate. The 60-trial sample cost about $22, plus or minus 30%, estimate |
| Trials at once | 2 | 1 | 30 in the 60-trial sample; 36 in an earlier run, with no throttling |
| Rewards compared with Docker | reference | Grader: 11 of 11 deterministic keys equal, PR review 6 of 6 identical. Codex: coding reward 1 in 6 of 6 trials (Docker 5 of 6), approve task 3 of 3, reject task 0 of 3 (Docker 0 of 3) | Grader: 12 of 12 deterministic keys equal. Codex: coding reward 1 in 50 of 53 valid trials (94%), reject task 23 of 185 (12%) |
| Security gates | Seal and network allowlist pass; the agent runs as root with session credentials | Seal and hiding of hidden tests observed; 16 negative network gates blocked; two LLM red-team runs found no breach. System files outside your home directory stay readable | Seal pass; 43 firewall probes held; two LLM red-team runs found no breach. The optional `hard` grader turns stage 1 red for 4 of 8 planted tampers |
| Main limit | Docker, Harbor and about 16 GB of free disk; disk use can spike to failure | macOS on Apple silicon only; one trial at a time; the sandbox is a deny-list, not a container boundary | Needs an AWS account where you may create IAM, ECR, S3, CodeBuild and AgentCore resources; 7 of 60 trials were lost before the fixes |

The three modes agree on every deterministic grading field. The reward differences between them are inside run-to-run noise at these sample sizes (section 4).

**Which to use.** AgentCore if you have an AWS account you control and run more than a handful of trials. Local if you have a Mac, Bedrock access and no Docker. Docker if neither fits (Linux or Windows, or no AWS account that can create resources).

## 2. The three options

**Docker (baseline, as shipped).** Four notebooks call `harbor run`. Each trial runs in a Docker container built from the task's Dockerfile, with a proxy allowlist for Bedrock. Nothing was simplified. It needs Docker with compose, Harbor 0.26.0, Node, Codex, the GitHub CLI, Jupyter and AWS credentials.

**Local.** One runner written with the Python standard library only (464 lines) makes a sealed git clone per task and a `uv` environment per trial from the task's own lock file. It runs Codex inside `srt` (Seatbelt on macOS) with an allowlist of the two Bedrock hosts. It drops Docker, Harbor, the GitHub CLI, Jupyter and sudo. The module's verifier files run unchanged from a temporary copy with host paths.

**AgentCore.** Each trial is one AgentCore Runtime session. The image is built on CodeBuild from the task's own Dockerfile plus a 29-line layer. The agent runs as an unprivileged user in an empty network namespace whose only exit is a root-owned proxy. `p2_v2.py all` creates the ECR repository, S3 bucket, log group, three scoped IAM roles and the CodeBuild project by API (8 idempotent steps), then builds, deploys and runs. There is no Docker, Harbor or CloudFormation stack. The grader runs in the same session after the agent is stopped.

## 3. Details

### 3.1 Results kept and lost

| | Docker | Local | AgentCore |
|---|---|---|---|
| `reward.json` keys | 14 (coding) | same | same |
| Grader only, deterministic keys | oracle 1, empty 0 | 11 of 11 equal; PR review 6 of 6 identical; oracle 1.0 and empty 0.0 on all 4 tasks | 12 of 12 equal; in the packaged variants (`par`, `hard`) 8 of 8 |
| Coding agent, reward 1 | Codex 5 of 6 | 6 of 6 | 50 of 53 valid (94%): pr371 24 of 27, pr338 26 of 26. The 3 zeros share one failing pass-to-pass test, so they are an agent mistake, not judge noise |
| PR-review approve task | 3 of 3 | 3 of 3 | 28 of 28 valid at 30 at once |
| PR-review reject task | 0 of 3 | 0 of 3 | 23 of 185 (12%). With a 12% rate, 0 of 3 has probability 0.67, so no real difference is shown |
| Invalid trials | 0 of 24 | 0 of 12 Codex trials, 0 of 8 control runs | 7 of 60 lost before the fixes (section 3.5) |
| Lost compared with Docker | none | `trajectory.json` and `harbor view` (the raw Codex JSONL is kept) | same as Local; the sentence "inside a Docker container" in `instruction.md` is now slightly wrong |

### 3.2 Setup and time

| | Docker | Local | AgentCore |
|---|---|---|---|
| Cold setup | 11.5 min of machine time, 44 min real with fixes | `setup.sh` 36 s plus clones 31 s, about 1.1 min to the first trial; 2.7 min with all environments warm. A fresh-checkout run took about 3 min and about 2 GB | bootstrap about 13 s, image build 2.2 to 6.3 min per task in parallel, deploy 9 s |
| Re-run | 3 s check | 1 s | bootstrap 4.5 s; `all` starts trials after about 10 s |
| AWS needs | Bedrock only | Bedrock only, with an IAM user key (role and IAM Identity Center credentials should fail at the first call, from AWS documentation; not run) | IAM, ECR, S3, CodeBuild and AgentCore permissions; derived from the code, not tested with a restricted caller |

Time per trial for Codex (measured; pr371 and pr338 are the coding tasks):

| | Docker | Local | AgentCore |
|---|---|---|---|
| Environment start | 5.0 to 5.5 s | 19 to 21 s coding, 13 to 15 s review | 1.2 to 2.6 s; 25 to 33 s for 30 cold sessions at once |
| Agent, pr371 / pr338 | 133.0 / 148.4 s | 85.2 / 115.4 s | median 231.5 / 249.9 s |
| Trial total, pr371 / pr338 / review | 322 / 367 / 60 s | 286 / 327 / 52 to 63 s | 374 / 406 / 39 s; medians over 60 trials 347 / 374 s |
| Coding verifier | 185 s mean | 192 s (pr338), 180 s (pr371) | 159 to 167 s at 12 at once; 87 s serial, 45 s with `par`; median 95 s at 30 at once |

**The run notes disagree on the pytest stage of the verifier:** 92 s with 12 trials at once, 22 s serial on the same image content. The cause was not found. Local measured about 100 s per coding verifier against 26 to 29 s for plain `pytest`, and the junit option is ruled out as the cause. The question is open in all three modes.

### 3.3 Disk, cost and parallelism

| | Docker | Local | AgentCore |
|---|---|---|---|
| Disk peak / steady | 16.9 / 15.8 GB | 2.8 / 2.2 GB (`du` apparent sizes) | 0.15 GB on the laptop; ECR 7.1 GB |
| RAM and CPU during a run | VM guest mean 0.77 GB, peak 1.02 GB; CPU mean 6% of 4 vCPU; host VM process not measured | not measured | laptop: one Python process, a few MB; session 2 vCPU and 8 GB, idle 0.49 GB |
| Idle cost | none | none | about $0.02 per day (ECR) |
| Agent cost per trial | coding $0.242 to $0.263, review $0.132 to $0.140, measured | pr338 $0.304, pr371 $0.214, approve $0.134, reject $0.168; estimate, using the price implied by the Docker runs' own cost field | coding $0.32 and $0.45, review $0.11 to $0.12; estimate from tokens |
| Judge cost per coding trial | $0.030 to $0.039, estimate | $0.020 to $0.030, estimate | about $0.03 to $0.04, estimate |
| 12 Codex trials | $2.33 agent, measured, plus about $0.2 judges | $2.46 agent plus $0.15 judges | about $3.4: agent $3.0, judges about $0.2, session time about $0.2 |
| Parallelism | 2 at once; more is limited by disk | 1 | 12 in 7.6 min, 24 in 8.8 min, 36 at once with no throttling; the 60-trial sample ran as two waves of 30 |

Input tokens per trial compared with Docker: AgentCore uses 30% to 70% more on coding tasks; Local is between 0.65 and 1.39 times, inside a plus or minus 50% band. No PR-review judge token counts are logged in any mode.

### 3.4 Security gates

| Gate | Docker | Local | AgentCore |
|---|---|---|---|
| Seal (one commit, no remotes, hidden tests absent) | enforced by the image; pass | host clones pass 4 of 4; hiding of module files from the agent observed under `srt` | enforced by the image; pass. Hidden tests are baked into the image, root-only |
| Network allowlist | proxy; 6 of 6 checks pass | `srt` proxy, Bedrock hosts only; 16 negative gates blocked. The fixed probe script has not been re-run (two lines in the first run were probe bugs). The sandboxed verifier ran the whole judge chain; its own blocked paths were not probed | one Bedrock host per phase, fail closed; 43 probes held |
| Credentials | the agent is root with session credentials | one-hour federated token; cannot be limited per Mantle model | agent role: Codex model only; grader role: judge model only; negative test passed |
| Tamper: root `conftest.py` forces tests to pass | stage 1 green; reward 0 only through the validation agent | same; not re-run under `srt` | same by default; with `hard`, stage 1 is red |
| Tamper: hook inside package source | not tested | a `sitecustomize.py` in the environment defeats two gates; reward 0 through the rubric and taste judges | stage 1 green in every variant, `hard` included; the validation stage discards it |
| Red team | none | two LLM runs, no breach; system files outside the home directory stay readable | two LLM runs and 43 probes, no breach; not a certified boundary |

### 3.5 Scale and robustness (AgentCore)

**60-trial sample.** Two waves of 30 Codex trials, default grader, one account, one day.

| | pr371-fix | pr338-fix | both |
|---|---|---|---|
| trials run / lost | 30 / 3 | 30 / 4 | 60 / 7 (11.7%) |
| reward 1, of trials with a reward | 24 of 27 (89%) | 26 of 26 (100%) | 50 of 53 (94%, 95% interval 85% to 98%) |
| reward 1, of all trials (lost counted as 0) | 24 of 30 | 26 of 30 | 50 of 60 (83%) |

The median trial took 365 s (90th percentile 399 s): agent 243 s, verifier 95 s. No throttling. The cost was about $22, plus or minus 30%, estimate.

**The 7 lost trials and the fixes** (on this branch, only `agentcore/` files changed):

| Lost | Cause | Fix |
|---|---|---|
| 4 | `RuntimeClientError` 500 when the session starts (AgentCore side) | `Session.start()` retries the first call on a new session, 3 attempts with waits of 2.5, 5 and 10 s. Later calls are not retried, so agent work is never repeated |
| 2 | Reading results back rejected an absolute symlink that Codex creates in its temporary directory, and the graded result was thrown away | The archive is extracted member by member through the `tarfile` data filter. A rejected member is skipped and listed; the rest is kept |
| 1 | The taste judge raised a `TypeError` on a malformed tool call (section 5); the summary showed no reason | An invalid trial now shows its reason, and the summary lists it with a command to rerun it. The module code is not changed |

Checks: five unit tests without AWS pass. A live run of the fixed runner (4 trials) lost none. The retry and skip paths did not fire in that run, so they are verified by unit tests only. No 30-session run was made after the fixes.

**Other robustness checks.** Missing or bad credentials fail in under 1 s and create nothing. A missing permission names the action. Bootstrap is not transactional, but every step is idempotent, so a re-run finishes it. Four faults were found and fixed: no boto3 version check, `build` returning success after a failed build, Ctrl-C waiting up to 25 min, and `cleanup` aborting on a runtime that was still deleting. Still open: prefix validation, a visible timeout error, the 12-fold retry on read timeouts, a safer default for `cleanup` (the default prefix is shared), no concurrency cap, and runtime log groups without a retention period.

**Variants.** `par` runs pytest beside the LLM judging stages and cuts the verifier from 87 s to 45 s. `hard` grades a fresh root-owned checkout without the agent's changes to `tests/`, `conftest.py`, `sitecustomize.py`, `*.pth` and pytest configuration; it adds 0.1 s. Both leave the deterministic fields unchanged. `hard` does not stop a hook placed inside package source.

### 3.6 Checks on the branch

- **AgentCore:** a clean clone ran `all pr4121:1` and `all pr371:1` (both reward 1) and tore down to zero leftovers, for about $1.
- **Local:** a fresh-checkout run of `local/` at commit `345f2c9` passed. It ran from a path under the home directory in a plain terminal, took about 2 GB and about 3 min of setup. L0: `sandbox-ok`. L1 base suites: 1864 passed (pr371), 1728 passed (pr338), 311 passed and 4 failed (pr4121, a known macOS `/etc` symlink effect), 669 passed and 1 skipped (pr4615). L5s: two oracle runs OK. L6a: Codex works inside `srt`.

## 4. Do the three modes grade differently?

No design-level difference was found. The study is in the internal run notes.

- The first request of the taste and rubric judges is byte-identical in all three designs (8,327 input tokens). Local differs only by the host path written into the prompts.
- For one patch, the taste judge scores 0.4 to 1.0 in AgentCore (27 repeats, mean 0.80, standard deviation 0.19). A replay of the judge on the host (819 calls) has mean 0.81, which is not significantly different (p=0.93). So the differences you see are sampling noise of about 0.19 per call.
- One small lead, not a finding: how much clutter the judge sees in the repository (`__pycache__` folders, very long paths) moved the share of taste scores below the 0.6 reward gate between 1.4% and 6.4% (p=0.002, not corrected for several comparisons).
- In practice about 5 to 7% of trials on a pr338-style patch get reward 0 from that gate, in every mode. With 3 trials per task, the chance of at least one is about 20%.
- The Docker run's single taste score of 0.2 is consistent with a rare low value of the same judge (6 of 819 host replays, 0 of 27 AgentCore repeats). It cannot be placed more precisely without repeating the Docker run, which was not done.
- PR-review judge: identical decisions on the findings that all three designs scored; 15 of 15 decision sets were 3-0 unanimous.

## 5. Finding for the module owners: `judges.py` line 106

In the taste judge, `datasets/strands-evals-coding/tasks/strands-agents__evals__pr371-fix/tests/lib/judges.py` line 106 computes `result[k]['passed']` for each check. If the model submits a check such as `minimality` as a string holding malformed JSON instead of an object, this raises `TypeError: string indices must be integers, not 'str'`. The trial is then marked `INVALID TRIAL` and no reward is written. It happened in 1 of 60 AgentCore trials and in 1 of 820 host replays. The same line is in the `pr338-fix` copy and in `datasets/_shared/coding/tests/lib/judges.py`; the files are identical. This page's change does not edit them.

Suggested fix: validate the submitted input against the schema in `Conversation.ask` (`lib/bedrock.py`) and return a tool error, so the model resubmits, as it already does for other tool errors. The rubric judge uses the same `ask` and has the same exposure.

## 6. Known limits

- **Docker:** host disk fell from 8.0 to 0.4 GiB in 3 minutes during one run and a notebook failed; colima needs `fstrim` to return the space. `docker compose`, Node, Jupyter and a host Codex are prerequisites that the old guide did not list.
- **Local:** macOS on Apple silicon only. One trial at a time. `srt` cannot start inside another sandbox, so run it from a plain terminal, and keep the repository under your home directory. The sandbox is "read everything, deny some paths", so the agent can read system files outside the home directory (`/etc/passwd`, `/Applications`). A `sitecustomize.py` in the environment defeats two grader gates; the Docker path is expected to have the same exposure but it was not tested. The 4 pr4121 base tests that fail on macOS are not run by the verifier.
- **AgentCore:** hidden tests sit in the image. The verifier runs agent-influenced code with the judge-only role. AgentCore may log command text that includes credentials; this was not checked. Runtime deletion takes up to 9 min. `agentcore/scripts/teardown.sh` removes everything a run created; `--print` lists the commands instead.
- **All modes:** only Codex was run in Local and AgentCore. The reject task needs about 85 trials per design to tell 12% from 30%. Judge temperature is not set in any mode.

## 7. What is not shown

1. **Claude Code** was not run in Local or AgentCore. Only Codex was tested there.
2. **Docker repeats.** The Docker run's one low taste score has no repeats, so whether its judge has a fatter low tail is open.
3. **A fresh AWS account.** AgentCore ran in the account used for all earlier work. Service control policies were not tested.
4. **A restricted caller.** The AgentCore permission table is derived from the code; every run used an admin-level user.
5. **Linux and Windows** for Local. `setup.sh` refuses anything but macOS on Apple silicon.
6. **Tamper with a hook inside package source** is not stopped by `hard`. The validation stage caught it in every mode, but only one fake agent per tamper was tried. The tamper cases were not re-run under `srt`.
7. **The fixed AgentCore runner at scale.** The retry and skip paths are covered by unit tests only.
8. **Resources.** RAM and CPU during a Local run and the host VM of Docker were not measured. Local disk numbers are apparent sizes. Dollar figures are estimates; no bill was read.
9. **The slow pytest stage** (92 to 100 s in three places, 22 s in one) has no explanation.
10. **Judge differences:** the tool outputs the judges saw were not logged in any design, the Docker run kept no judge transcripts, and the PR-review judge prompts could not be compared. The clutter effect was tested only in a host replay on one patch.
11. **Sample size.** Local and Docker have 3 trials per task; AgentCore has 60 coding trials. A coding reward that is almost always 1 cannot show a worse design.
12. **Notebooks 02 and 03** call Harbor and were not ported to Local.
