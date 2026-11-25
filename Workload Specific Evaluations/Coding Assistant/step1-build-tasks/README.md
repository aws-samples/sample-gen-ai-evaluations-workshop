# Step 1: Build evaluation tasks from your repository

Public benchmarks are not your code. This step turns your repository's own history into Harbor tasks.
[Step 2](../README.md) then runs coding agents on those tasks on Amazon Bedrock AgentCore. The
notebook [`../step1-build-tasks.ipynb`](../step1-build-tasks.ipynb) walks through this step. Everything it
runs lives here, in scripts you can reuse as they are.

There are two kinds of task:

| kind | the agent... | graded by | how to build it |
|---|---|---|---|
| **PR review** | reviews a change and reports defects | the review the change really got | [eval-harvest](https://github.com/aws-samples/sample-code-eval-harvester)'s review workflow (`survey`, `capture`, `emit`), driven by a coding agent |
| **Coding** | makes a merged pull request's change, from the commit before it | the tests that pull request added or changed | the scripts in this folder (`coding_tasks.py`, standard-library Python) |

## Coding tasks

```bash
# 1. one task per merged pull request (reads git history only; no GitHub API call)
step1-build-tasks/build_coding_tasks.sh https://github.com/pallets/itsdangerous.git 378

# 2. find which hidden tests the change makes pass, on AgentCore (needs Step 2's setup: .venv and bootstrap)
step1-build-tasks/calibrate_tasks.sh step1-build-tasks/work/dataset/tasks/pallets__itsdangerous__pr378-code
```

Everything Step 1 needs for coding tasks is in this folder: `coding_tasks.py` builds and calibrates the
tasks, and `task_template/` holds what every task carries (its Dockerfile, the staged `test.sh`, the
grader, the oracle's `solve.sh`). Nothing is installed from another repository. You need `git`, a Python
3.11 or later (Step 2's `.venv` is used when it exists), and for calibration Step 2's AgentCore setup.

`calibrate_tasks.sh` ends with a gate: Harbor's `nop` agent (changes nothing) must score 0 and its
`oracle` agent (applies the real change) must score 1, both graded in stages. A task that passes the
gate goes into Step 2 like the sample tasks: `harbor run -p step1-build-tasks/work/dataset/tasks ...`.

### What a task holds

| path | what it is | can the agent see it? |
|---|---|---|
| `environment/` | the repository at the commit before the change, as one fresh commit at `/work/repo` (no history), with its dependencies installed | yes |
| `instruction.md` | the pull request's title, the hidden test files and how they run; after calibration, the tests that must pass. Never the PR description, which usually says how the fix works | yes |
| `tests/` | the changed test files as the pull request left them, and the grader | only after the agent stops |
| `solution/` | the change itself, for the `oracle` agent | no |

### How it is graded

Like the sample coding tasks, in stages. `tests/test.sh --stage untrusted` runs the hidden tests as
the agent user and writes only a test report. `tests/test.sh --stage trusted` then runs as a
different user, after every agent process has ended, and computes the score from that report and the
task's own files. So no file the agent can write becomes its score.

### Use your own repository

1. Copy [`coding-profile.toml`](coding-profile.toml) and change it for your repository:

   | key | meaning |
   |---|---|
   | `image` | the base image; it must have, or install, your toolchain |
   | `install` | runs at build time in `/work/repo` (the agent and the grader have no internet later) |
   | `command` | your test command, with `{files}` (the hidden test files) and `{junit}` (where to write the JUnit XML report) |
   | `run` | globs of the test files the command runs |
   | `tests` | optional: globs of the test side, if yours is not in `tests/`, `test/`, `spec/` or `*_test.*` files |
   | `agent_setup` | the agents' own tools, installed at build time |

   The other keys: `display_command` (the command as the instruction shows it), `workdir` (where the
   command runs, relative to the repository root), `timeout_sec`, `env`, and `allowed_hosts` (what the
   agent may reach; default: the Bedrock endpoints of Codex and Claude Code).
2. Pick merged pull requests that change code and its tests together. The script skips the others
   with a reason:

   | reason | meaning |
   |---|---|
   | `no-test-change` | the pull request changes no test file |
   | `no-source-change` | it changes only tests: there is nothing to write |
   | `no-run-files` | no changed test file matches your `run` globs |
   | `no-fail-to-pass` (at calibration) | no hidden test fails without the change, so the tests cannot tell a fix from no fix |

3. Run both scripts with `PROFILE=<your profile>` and your repository's URL or local clone.

### Cost and time

Each task builds one image on CodeBuild (several minutes the first time). Calibration runs four short
trials per task (nop and oracle, twice), with no model calls.

### Files

| file | what it does |
|---|---|
| `build_coding_tasks.sh` | clones the repository and runs `coding_tasks.py build` for each pull request |
| `calibrate_tasks.sh` | builds the images, runs nop and oracle on AgentCore, runs `coding_tasks.py calibrate`, then the gate |
| `coding_tasks.py` | `build`: the task from a merged pull request; `calibrate`: the fail-to-pass and pass-to-pass tests from Harbor's runs |
| `check_gate.py` | the gate: nop 0, oracle 1, both graded `staged` |
| `task_template/` | copied into every task: `Dockerfile.tmpl`, `test.sh` (the `--stage` contract `harbor_env` grades with), `grader.py` (standard library, any `python3` from 3.7), `solve.sh` |
| `coding-profile.toml` | the sample repository's profile |
| `tests/` | offline tests (no AWS): `.venv/bin/python -m pytest -q step1-build-tasks/tests` |
