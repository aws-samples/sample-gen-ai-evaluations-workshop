# Coding Assistant Evaluations on your Mac

This folder evaluates a coding agent (Codex CLI on Amazon Bedrock) on four tasks made from real pull requests: two PR reviews and two bug fixes. Everything runs on your Mac, with no container and no VM. It uses four small pieces:

1. a sealed `git clone` per task (the same git steps as each task's Dockerfile),
2. a `uv` virtual environment per trial (built from the task's own `requirements.lock`),
3. an operating-system sandbox for the agent: [`srt`](https://github.com/anthropics/sandbox-runtime), which uses `sandbox-exec` (Seatbelt) on macOS, and
4. one Python runner, `runner/p1.py` (standard library only, about 500 lines).

**Status: preview, tested on macOS arm64 only.** The grader was checked against reference reward files of the same verifier files, the sandbox is verified by scripted probes and an LLM red team, and 12 real Codex trials were run (see [Agent trials](#agent-trials)). All numbers come from one laptop and one owner run, n=3 per task.

To walk through it step by step in Jupyter (setup, the sandbox checks, one trial of each kind, the controls and clean-up), open the notebook [05 Local mode walkthrough](05%20Local%20mode%20walkthrough.ipynb). It runs the same commands as the [tester guide](TESTER-GUIDE.md).

## How a task runs here

Each task directory in `datasets/` is written for a container: an `environment/Dockerfile` with the start state, and a `tests/test.sh` that runs at fixed paths. The runner does the same steps on your Mac:

| Job | What the task files describe | What the runner does |
|---|---|---|
| Dependencies | the build steps of `environment/Dockerfile` (Debian, Python 3.12) | `uv venv` per trial from the task's `requirements.lock` (macOS wheels, CPython 3.12) |
| Git seal (agent cannot see the fix) | reset to the base commit, remove the remote, run `gc` | the same git commands, run once per task on the host. Each trial gets a copy of that clone |
| Network allowlist | `allowed_hosts` in `task.toml` | `srt` proxy: the agent can reach `bedrock-mantle.us-east-1.api.aws` and `bedrock-runtime.us-east-1.amazonaws.com` on port 443, nothing else |
| Hidden tests and oracle patch stay away from the agent | the verifier files are given to the grader only after the agent stops | they stay in this folder, which the agent's sandbox cannot read |
| Verifier phase | `tests/test.sh`, after the agent stops | the same `tests/test.sh`, run after the agent stops, from a temp copy in which `/work/repo`, `/tests`, `/logs` and `/tmp` are replaced by the trial's host paths. The files in `datasets/` are not changed |
| Run and record | one folder per trial | `runner/p1.py` writes `jobs/<job>/<trial>/` |
| Agent | Codex 0.162.1 | the same version, with the flags in `CODEX_FLAGS` of `runner/p1.py` |
| Credentials | not set by the task files | the agent gets a one-hour federated token limited to three model IDs. The verifier gets a two-hour session token |

What the agent sees: file paths are host paths instead of `/work/repo`, and the prompt line "You are inside of a Docker container." in `instruction.md` reads "You are working on a laptop, in a sandbox with its own copy of the repository." Judge prompts name the host paths of the two files they read. Model IDs, criteria and every other prompt line are as in the task files.

## Architecture

```
 you (plain terminal, default AWS profile)
   |
   |  setup.sh        installs Node 22, srt 0.0.83, Codex 0.162.1, ripgrep into ./ (about 40 s)
   |  runner/p1.py
   v
 aws sts  ->  agent token (1 h, 3 model IDs)      verifier token (2 h)
   |
   |  prepare:  git clone + seal  ->  $P1_WORK/base/<task>/   (once per task)
   |  trial:    copy of the sealed clone + uv venv  ->  $P1_WORK/trials/<trial>/
   v
 +-------------------------------- srt (Seatbelt) ---------------------------------+
 |  Codex CLI  reads and writes only  $P1_WORK/trials/<trial>/                       |
 |             reaches only  bedrock-mantle ...:443  and  bedrock-runtime ...:443    |
 +-----------------------------------------------------------------------------------+
   |  agent stops
   v
 verifier: rendered copy of tests/ ($P1_WORK/verifier/<trial>/), run by test.sh
           (hidden tests, base suite, validation agent, rubric judge, taste judge)
           inside a second srt profile that allows bedrock-runtime only
   |
   v
 jobs/<job>/<trial>/result.json, verifier/reward.json, agent/codex.txt, agent/sessions/
```

`$P1_WORK` is `~/.coding-assistant-local/<id>/work`, where `<id>` is the first 8 hex digits of the SHA-1 of this folder's path (`env.sh` computes it), so two checkouts never share a work folder. Set `P1_WORK` to choose another path. It is outside this folder on purpose: the path of this folder can contain spaces, and the verifier scripts use host paths without quotes.

## Supported platforms

- **macOS on Apple silicon (arm64): tested.** `setup.sh` refuses to run on anything else. Tested on macOS 26.7.3.
- **Intel Mac: not supported.** `setup.sh` downloads arm64 binaries.
- **Linux: not tested.** `srt` itself supports Linux (it needs `bubblewrap` and `socat` instead of `sandbox-exec`). The runner would also need changes: it copies the clone with `cp -c` (APFS clone), points Codex to the CA file `/etc/ssl/cert.pem`, denies reads of `/Users` and `/Volumes`, and `setup.sh` downloads Darwin builds of Node and ripgrep.
- **Windows: not supported.**

## Setup

Six steps, one of them manual. About 1 minute to the first trial and about 3 minutes with every task environment warmed up (measured on one machine, see [TESTER-GUIDE.md](TESTER-GUIDE.md)).

1. **(manual)** Put AWS credentials in your default profile: an IAM user's access key. The runner calls `aws sts get-session-token` and `aws sts get-federation-token`, which AWS does not allow with credentials from an assumed role or IAM Identity Center. See "AWS permissions" in the tester guide.
2. Install `uv`: `brew install uv`. You also need the AWS CLI v2, `git`, `curl` and `tar`.
3. Open a plain terminal (not a terminal inside an agent tool or another sandbox), then `cd` to this folder.
4. `./setup.sh`
5. `source env.sh`
6. Prepare the four sealed clones:

```bash
p1() { uv run --no-project --python 3.12 runner/p1.py "$@"; }
p1 prepare pr371-fix pr338-fix pr4121-approve pr4615-reject
```

Then run one trial, or the whole test ladder:

```bash
p1 trial pr4121-approve --agent codex --job try1 --federated scripts/agent-policy.json
./run_all.sh
```

Everything the setup installs goes into this folder (`node/`, `npm/`, `uv-cache/`, `uv-python/`, `home/`, `bin/`) and into `~/.coding-assistant-local`. Nothing is installed system-wide, and nothing needs `sudo`. All of it is ignored by git. The [tester guide](TESTER-GUIDE.md) has the commands, the AWS permissions, the test ladder, how to read results and how to clean up.

## What is verified, and what is not

Measured on one Mac (macOS 26.7.3, arm64, 11 cores, 18 GiB), in November 2026, us-east-1.

**Verified**

- **Grader equals the reference.** The task's verifier files run unmodified. For the same patch, compared with reference reward files made in a Linux container, the deterministic reward keys are equal in 11 of 11 coding cases and the PR-review reward files are identical in 6 of 6. The judge keys are within 0.25 in 10 of 11. The exception is the taste judge, which is noisy by itself (temperature is not set).
- **Controls.** The oracle patch scores 1, the empty patch scores 0, for both coding tasks.
- **Same tasks and prompts.** The 41 prompt and config files hash identically to the task files. Only the paths and one sentence differ, as listed above.
- **Base test suites.** pr371 1864 passed and pr338 1728 passed, as in the image. For pr4121, 4 of 315 tests fail on macOS because `/etc` is a symlink to `/private/etc`. The verifier does not run them.
- **Sandbox, scripted probes** (run by the owner in a plain terminal): both Bedrock hosts are reachable and 16 other network targets are blocked; reads of this folder, other trials, the decoy `~/.aws` and `~/.ssh` and the real home are denied; writes outside the trial directory are denied. 0 breaches. An LLM red team (two runs, 38 and 31 commands) found no way out. After two probe-script fixes, the network probes (L2) were re-run and held: both Bedrock hosts reachable, 16 other targets blocked.
- **Setup cost.** 36 s for `setup.sh`, 31 s for the four sealed clones, about 2.7 min cold with all environments built. Disk 2.2 GB steady, 2.8 GB peak (apparent sizes). No container, no VM, no idle processes.

**Known gaps, by design or by tool**

- A planted root `conftest.py` that forces every test to pass defeats the hidden-test stage. The reward still stays 0 because the validation agent and the judges catch it.
- A `sitecustomize.py` planted in the trial's venv defeats the hidden-test stage and the validation agent. The reward stays 0 only through the rubric and taste judges.
- System files outside `/Users` stay readable inside the sandbox (`/etc/passwd`, `/Applications`), and `lsof` lists other processes' sockets.
- The agent's federated token allows any model on the Bedrock Mantle endpoint, because the Mantle action cannot be limited per model.
- `srt` 0.0.83 is a research preview, and `sandbox-exec` is deprecated by Apple but still works.
- Claude Code as the agent is not supported here. Only Codex.

### Agent trials

Measured on one laptop (macOS 26.7.3, arm64, 11 cores, 18 GiB), serial (one trial at a time), n=3 per task, 20 November 2026, us-east-1. Dollar figures are estimates: the price per token is $2.50 per 1M uncached input, $0.20 cached, $10.00 output.

| Task | Reward (3 trials) | Agent seconds | Trial total seconds | Agent $ per trial (est.) |
|---|---|---|---|---|
| pr371-fix | 1, 1, 1 | 85.2 ± 3.4 | 286.3 ± 15.1 | 0.214 |
| pr338-fix | 1, 1, 1 | 115.4 ± 3.6 | 326.5 ± 6.0 | 0.304 |
| pr4121-approve | coverage 1.0 in 3 of 3 | 36.5 ± 3.6 | 51.5 ± 3.6 | 0.134 |
| pr4615-reject | coverage 0.0 in 3 of 3 | 48.2 ± 10.6 | 63.4 ± 12.5 | 0.168 |

(± is the population standard deviation of 3 trials.)

- **12 Codex trials: 0 invalid**, every agent exited 0 and every trial produced a `reward.json`. Wall clock for the 12 trials: 2,260 s (37.7 min). With the oracle and empty controls the whole L7 stage took about 54 min.
- **Controls inside the sandbox:** the oracle patch scores 1 and the empty patch scores 0 on all four tasks.
- **First agent reward:** 48.7 s after the trial started (pr4121 review). From an empty machine to the first reward: about 3 min.
- **Cost:** agent $2.46 for the 12 trials, judges about $0.15, so one full `./run_all.sh` costs about $3 (estimate).
- **Not measured:** RAM and CPU during a trial, and the byte count of the downloads.
- **Reading the numbers:** n=3 per task shows large gaps only. The PR-review reject task scores 0 in 3 of 3 trials, and a coding reward that is 1 everywhere cannot tell two agents apart.

If you run it, please send your results with the template at the end of the tester guide.

## Files

See [MANIFEST.md](MANIFEST.md). The tester guide is [TESTER-GUIDE.md](TESTER-GUIDE.md).
