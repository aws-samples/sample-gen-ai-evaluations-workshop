# Step 1: Build evaluation tasks from your repository

Public benchmarks are not your code. This step turns your repository's own history into Harbor tasks.
[Step 2](../README.md) then runs coding agents on those tasks on Amazon Bedrock AgentCore. The
notebook [`../step1-build-tasks.ipynb`](../step1-build-tasks.ipynb) walks through this step. Everything it
runs lives here, in scripts you can reuse as they are.

There are two kinds of task:

| kind | the agent... | graded by | how to build it |
|---|---|---|---|
| **PR review** | reviews a change and reports defects | the review the change really got | [eval-harvest](https://github.com/aws-samples/sample-code-eval-harvester)'s review workflow (`survey`, `capture`, `emit`), driven by a coding agent |
| **Coding** | makes a merged pull request's change, from the commit before it | the tests that pull request added or changed | `eval-harvest code`, with the scripts in this folder |

## Coding tasks

```bash
# 1. one task per merged pull request (reads git history only; no GitHub API call)
step1-build-tasks/build_coding_tasks.sh https://github.com/pallets/itsdangerous.git 378

# 2. find which hidden tests the change makes pass, on AgentCore (needs Step 2's setup: .venv and bootstrap)
step1-build-tasks/calibrate_tasks.sh step1-build-tasks/work/dataset/tasks/pallets__itsdangerous__pr378-code
```

The scripts run eval-harvest with `uvx`, from the repository `EVAL_HARVEST` names (default: aws-samples). Coding-task
support is in [aws-samples/sample-code-eval-harvester#16](https://github.com/aws-samples/sample-code-eval-harvester/pull/16);
until it is merged, set `EVAL_HARVEST` to that pull request's branch.

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

   The full list is in eval-harvest's `man/eval-harvest-code.md`.
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
