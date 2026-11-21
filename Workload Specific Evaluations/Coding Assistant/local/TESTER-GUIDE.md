# Tester guide

You are testing the Coding Assistant evaluation that runs on your Mac only: no container, no VM. Read [README.md](README.md) first for what it is. This guide tells you what to install, what to run, what each test stage proves, what it costs, and how to clean up. Numbers marked **measured** come from one laptop (macOS 26.7.3, 11 cores, 18 GiB), one trial at a time, n=3 per task, 20 November 2026. Numbers marked **estimate** were not measured.

All commands start in this folder (the one with `runner/p1.py`).

## 1. Prerequisites

| Need | Detail |
|---|---|
| A Mac with Apple silicon | macOS arm64. Tested on macOS 26.7.3. Intel Macs, Linux and Windows are not supported yet |
| A plain terminal | Terminal.app or iTerm. **Not** a terminal inside an agent tool, an IDE sandbox or another sandbox: `sandbox-exec` fails there with `sandbox_apply: Operation not permitted` |
| `uv` | `brew install uv`. Tested with 0.12.3 |
| AWS CLI v2 | the runner calls `aws sts ...`. Check with `aws --version` |
| `git`, `curl`, `tar` | the ones that ship with macOS are fine (tested with Apple git 2.50.1) |
| AWS credentials in the default profile | an **IAM user** access key, see section 2. Region `us-east-1` is fixed by the tasks |
| Model access in `us-east-1` | `openai.gpt-6-sol` (the agent), `us.anthropic.claude-haiku-5-5` (the judges) |
| Disk | about 3 GB free (section 4) |
| Network | one download of about 1 GB during setup (estimate). During a run, only the two Bedrock hosts are used from the sandbox |

You do **not** need a container engine, `gh`, a system Python or `sudo`. Jupyter is needed only for the walkthrough notebook, and its start command installs it with `uv`.

## 2. AWS permissions

The runner mints two short-lived credentials from your default profile for each trial and passes them only to the processes it starts:

| Credential | Call | Lifetime | Used by |
|---|---|---|---|
| Verifier token | `aws sts get-session-token` | 2 hours | the grader (judge models) |
| Agent token | `aws sts get-federation-token --name p1-agent --policy scripts/agent-policy.json` | 1 hour | Codex, inside the sandbox |

What your IAM user needs, derived from `runner/p1.py`, `scripts/agent-policy.json` and the tasks' verifier code (`Converse` calls):

- `sts:GetFederationToken`. Without `--federated`, the agent gets the verifier token instead and this one is not needed. `run_all.sh` always uses `--federated`.
- `sts:GetSessionToken` needs no permission (AWS does not let a policy control it).
- `bedrock-mantle:CreateInference`. Codex reaches `openai.gpt-6-sol` through the Bedrock Mantle endpoint. This action cannot be limited to one model.
- `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream` on the Claude Haiku 5.5 inference profile and its foundation models (judges and validation agent), and on `openai.gpt-6-sol`.

The federated agent token is the **intersection** of your user's policy and `scripts/agent-policy.json`, so the user must allow at least what that file allows. A minimal identity policy for a dedicated test user:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {"Effect": "Allow", "Action": "sts:GetFederationToken", "Resource": "*"},
    {"Effect": "Allow", "Action": "bedrock-mantle:CreateInference", "Resource": "*"},
    {"Effect": "Allow",
     "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
     "Resource": [
       "arn:aws:bedrock:*::foundation-model/anthropic.claude-haiku-5-5*",
       "arn:aws:bedrock:*::foundation-model/anthropic.claude-sonnet-5-5*",
       "arn:aws:bedrock:*::foundation-model/openai.gpt-6-sol*",
       "arn:aws:bedrock:*:*:inference-profile/us.anthropic.claude-haiku-5-5",
       "arn:aws:bedrock:*:*:inference-profile/us.anthropic.claude-sonnet-5-5"
     ]}
  ]
}
```

This policy is derived from the code and the owner's IAM user. It was not run as a stand-alone policy. If a call fails with `AccessDenied`, the error names the missing action.

**Credentials from an assumed role or IAM Identity Center do not work.** AWS does not allow `GetSessionToken` or `GetFederationToken` with role credentials ([GetFederationToken](https://docs.aws.amazon.com/STS/latest/APIReference/API_GetFederationToken.html), [AssumeRole](https://docs.aws.amazon.com/STS/latest/APIReference/API_AssumeRole.html)). The runner also ignores `AWS_*` variables and `AWS_PROFILE` and reads the default profile. This is not tested here, because the owner's run used an IAM user. If your only credentials are a role or SSO, say so in your feedback: that is a known gap, not your mistake.

The verifier runs code from the agent's change while it holds the 2-hour token, with your user's full Bedrock permissions. Use a dedicated user with only the permissions above and no administrator rights.

**Nothing is created in AWS.** The runner makes `sts` calls (temporary credentials, which are not resources and expire on their own) and Bedrock invocations. There is no S3, IAM, CloudFormation or other resource to delete. If model invocation logging is enabled in your account, the prompts and answers of these calls are logged there like any other Bedrock call.

## 3. Setup and run

```bash
./setup.sh          # Node 22, srt 0.0.83, Codex 0.162.1, ripgrep, GNU-style timeout shim: all inside this folder. About 40 s
source env.sh       # own HOME, caches, PATH, work dir. Run it in every new shell before the commands below
p1() { uv run --no-project --python 3.12 runner/p1.py "$@"; }    # shorthand for the runner

p1 prepare pr371-fix pr338-fix pr4121-approve pr4615-reject      # sealed clones, once. About 30 s
p1 base-suite pr371-fix                                          # base tests in a fresh venv: expect 1864 passed
```

`source env.sh` changes `HOME` in that shell to `home/` in this folder. Open a new terminal when you are done.

One trial (environment start, Codex inside the sandbox, verifier, record):

```bash
p1 trial pr4121-approve --agent codex --job try1 --federated scripts/agent-policy.json
p1 trial pr371-fix      --agent codex --job try1 --federated scripts/agent-policy.json
```

Controls without an agent, and several trials in a row:

```bash
p1 trial pr371-fix --agent oracle --job ctl --tag oracle         # the reference patch: must score 1
p1 trial pr371-fix --agent nop    --job ctl --tag nop            # no change: must score 0
p1 job --tasks pr371-fix,pr338-fix,pr4121-approve,pr4615-reject --n 3 --agent codex --job full --federated scripts/agent-policy.json
```

`--agent` also takes `patch:/absolute/path/to/agent.patch` and `tamper:A` to `tamper:E` (planted cheats, see `TAMPER` in `runner/p1.py`). Trials run one after another. A second run with the same `--job` and trial name replaces the earlier record.

The whole test ladder in one go, stopping at the first failure:

```bash
./run_all.sh                        # L0 L1 L2 L4 L5s L6a L6 L7: the owner's run of L5s to L7 took 93 min; L0 to L4 add a few minutes
STAGES="L5s L6a L6" ./run_all.sh    # a subset, in ladder order
CONTINUE=1 ./run_all.sh             # keep going after a failed stage (do not use for L6 and L7)
```

Run only one `run_all.sh` or `p1 job` at a time: they share the lock `.bedrock.lock` and wait for each other.

## 4. Time, disk and cost

Measured on one Mac with 11 cores and 18 GiB, one trial at a time (serial).

| | Value | Source |
|---|---|---|
| `setup.sh`, cold | 36 s (re-run: 1 s) | measured |
| `prepare`, 4 sealed clones | 31 s cold | measured |
| First environment of each task (fills the uv cache) | 43 s, 22 s, 19 s, 14 s | measured |
| Environment start per coding trial | 21 to 23 s (venv rebuilt from the cache) | measured |
| Grader, coding task, oracle patch | 202 s (pr371), 250 s (pr338) | measured, outside the sandbox |
| Grader, PR-review task | 0.1 s without findings, 18 s with findings | measured |
| Agent loop (Codex), per trial, mean ± sd of 3 | pr371 85.2 ± 3.4 s, pr338 115.4 ± 3.6 s, pr4121 review 36.5 ± 3.6 s, pr4615 review 48.2 ± 10.6 s | measured |
| Whole trial (environment start, agent, verifier) | pr371 286 s, pr338 327 s, pr4121 52 s, pr4615 63 s | measured |
| Verifier, coding task | 180 to 192 s (the hidden and base test stage is about 100 s of it) | measured |
| Time to first agent reward | 48.7 s after the trial starts (pr4121 review); about 3 min from an empty machine | measured |
| 12 Codex trials (L7) | 37.7 min; with the oracle and empty controls the L7 stage took about 54 min | measured |
| Disk, steady / peak | about 2.2 GB / 2.8 GB apparent size (the uv cache is 1 GB of that) | measured |
| Memory and CPU during a run | not measured (nothing runs between trials) | |
| Judge cost, one coding trial | $0.02 to $0.03 | measured |
| Agent cost, Codex | $0.21 to $0.30 per coding trial, $0.13 to $0.17 per PR review; $2.46 for the 12 trials | estimate: tokens measured, price $2.50 per 1M uncached input, $0.20 cached, $10.00 output |
| Cost of `./run_all.sh` | about $3 (L6 $0.41, L7 $2.68 with judges and controls, L5s a few cents) | estimate |

Trial directories are removed after the record is copied, except those of `base-suite` and `sandbox-run` (stages L1 to L4), which stay at 90 to 260 MB each. A long session can use several GB. Section 8 shows how to clear them.

## 5. The test ladder

`./run_all.sh` runs these stages in order. Each writes `logs/run_all/<stage>.log`.

| Stage | What it runs | What it proves | Time |
|---|---|---|---|
| L0 | `srt -- true` | The sandbox starts in your terminal. If it prints `sandbox_apply: Operation not permitted`, you are inside another sandbox: use a plain terminal | seconds |
| L1 | `prepare` and `base-suite` for four tasks | The git seal steps work on your machine, the macOS wheels install, and the base test suites pass as expected (pr371 1864, pr338 1728, pr4615 669 passed and 1 skipped, pr4121 311 passed and 4 failed on macOS: a known `/etc` symlink effect, not a failure) | about 3 to 4 min cold (estimate) |
| L2 | `scripts/l2_inner.sh` inside the agent's sandbox | The agent can reach `bedrock-mantle` and `bedrock-runtime` and nothing else: GitHub, PyPI, npm, plain HTTP, IP addresses, IPv6, UDP, DNS and raw sockets are blocked. Fails if a line says `REACHED` or `NOT reachable`. Output: `logs/L2-gates.txt` | not timed (under 1 min) |
| L4 | `scripts/l4_seal.py` | Each sealed clone has 1 commit, no remote and no reflog. No hidden test, oracle patch or canary string is reachable from a trial directory. Output: `logs/L4-static.json`. The stage prints JSON and always exits 0: **read it**, every `seal_ok` must be `true` and every `*_hits` and `hidden_in_trial_dir` list empty. Needs L1 first | seconds (estimate) |
| L5s | oracle patches through the verifier, inside its sandbox profile | The sandboxed grader works: the oracle scores 1 with a non-empty `agent.patch` (pr371) and full coverage (pr4121) | about 4 min (estimate) |
| L6a | `node` start-up, then a no-tool "reply OK" Codex call inside the agent's sandbox | `node` starts and Codex can reach Bedrock from inside the sandbox. If not, it tries more variants and says which one works | about 1 min (estimate) |
| L6 | two real Codex trials (pr4121 review, pr371 fix) | An agent runs end to end in the sandbox and the grader scores it. Fails if Codex exits non-zero or used 0 tokens | about 6 min (measured: 336 s) |
| L7 | 4 tasks x 3 Codex trials, then oracle and empty controls | The full set | about 54 min (measured) |

The investigators also ran three stages that are not in this folder because they need files that are not shipped: **L3** (32 scripted break-out probes inside the sandbox), **L3rt** (an LLM red team with Claude Sonnet 5.5 as the attacker) and **L5** (the grader on 22 fixed patches compared with reference reward files made in a Linux container). Their results are summarised in README.md. If you want them, ask the maintainers.

## 6. How to read the results

Each trial is a folder `jobs/<job>/<trial>/`:

| File | Meaning |
|---|---|
| `result.json` | timings per phase and `agent_result`: `exit`, `seconds`, `input_tokens`, `cached_input_tokens`, `output_tokens`, `steps`, `tool_calls` |
| `verifier/reward.json` | the reward. Coding tasks: `reward` (0 or 1) plus `correct`, `fail_to_pass`, `pass_to_pass`, `validation_score`, `rubric_score`, `relative_taste`, `patch_bloat` and others. PR-review tasks: `coverage_all`, `coverage_required` and the `credited_*` keys |
| `verifier/agent.patch`, `agent.stat` | the agent's change as the verifier saw it. An empty patch for a non-empty change means the sandboxed verifier is broken |
| `verifier/render.diff` | the path substitutions applied to the verifier files |
| `agent/codex.txt` | Codex's JSON event stream. `Reconnecting... (Connection failed ...)` lines mean it could not reach Bedrock |
| `agent/sessions/` | Codex's own session logs (every step) |

A trial is **valid** only if `agent_result.exit` is 0, `input_tokens` is above 0 and `verifier/reward.json` exists. The runner stops the job with `STOP: the agent did not run` otherwise. A missing `reward.json` is an invalid trial, not a score of 0.

Read the rewards of a job from Python, in this folder (a trial with no `reward.json` prints `None`: it is invalid):

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --no-project --python 3.12 python -c '
import json, pathlib
for t in sorted(pathlib.Path("jobs/L6").glob("*/result.json")):
    r = t.parent / "verifier" / "reward.json"
    print(t.parent.name, json.loads(r.read_text()) if r.exists() else None)'
```

For the number of trials that matter: judge fields move between runs of the same patch (`relative_taste` varied from 0.8 to 1.0 on one patch), and the PR-review reject task passed 12% of the time in a separate 185-trial study, so one trial per task shows that the pipeline works but does not rank agents.

## 7. Known limits

- **`sandbox-exec` is deprecated by Apple** but still works on macOS 26. `srt` 0.0.83 is a research preview and is pinned in `setup.sh`.
- **Run in a plain terminal.** macOS does not let a process that is already inside a Seatbelt sandbox start another one.
- **`srt` blocks writes to `.git/config` and `.git/hooks`.** The verifier's `test.sh` runs `git config user.email ...`, which fails with `could not write config file .git/config: Operation not permitted`. This is harmless: the runner sets the git identity through `GIT_AUTHOR_*` and `GIT_COMMITTER_*` variables. Do not treat that line in `test.sh.out` as a failure.
- **CA file for Codex.** The runner sets `CODEX_CA_CERTIFICATE` and `SSL_CERT_FILE` to `/etc/ssl/cert.pem`, and Codex worked with that setting in all 12 trials. Whether Codex needs it was not tested. If Codex cannot connect, `scripts/diag_codex.sh` (stage L6a) tries alternatives. Turning on `enableWeakerNetworkIsolation` is a real weakening and is never enabled automatically.
- **Run the stages through `run_all.sh`.** Do not redirect the output of the sandboxed runner to a file under `/Users` yourself.
- **Keep the repository under your home directory.** The sandbox denies reads of your home, `/Users`, `/Volumes` and `/tmp`. A checkout somewhere else (for example `/opt`) would be readable by the agent, including the hidden tests.
- **The work directory is outside this folder** (`~/.coding-assistant-local/<id>/work`, `<id>` = the first 8 hex digits of the SHA-1 of this folder's path, so each checkout has its own) and must not contain a space. `env.sh` sets it; set `P1_WORK` to override it. The runner refuses to start if the path has a space.
- **BSD tools.** The agent sees macOS `sed`, `find`, `tar` and bash 3.2. An agent that uses `sed -i` without a suffix or `grep -P` gets errors that it would not get on Linux.
- **Isolation limits.** System files outside `/Users` stay readable inside the sandbox. The sandbox is not a certified security boundary: one build, one attacker model, honest QA framing. Do not run an adversarial agent here without a review.
- **Cheats.** A planted root `conftest.py` that forces tests to pass defeats the hidden-test stage. A planted `sitecustomize.py` in the venv defeats it and the validation agent. The final reward stays 0 through the rubric and taste judges.
- **Parallel trials** are not supported (one lock) and not measured.
- **Linux, Intel Mac, Windows and Claude Code as agent** are not supported.

## 8. Clean up

Nothing exists in AWS. Delete local files only:

```bash
rm -rf "$P1_WORK"                                      # sealed clones, trial directories, verifier copies (can be several GB)
rm -rf jobs logs .bedrock.lock                         # records and logs of this folder
rm -rf node npm npm-cache uv-cache uv-python home bin  # toolchain and caches (uv-cache is about 1 GB)
rmdir "$(dirname "$P1_WORK")" ~/.coding-assistant-local 2>/dev/null   # empty parents of the work directory
rmdir /tmp/claude 2>/dev/null                          # created by the runner for srt, emptied after each run
```

To keep the toolchain and only free the space of trials: `rm -rf "$P1_WORK/trials"`. If you never ran `source env.sh` in the shell you clean from, `$P1_WORK` is empty: run `source env.sh` first (it prints nothing; `echo "$P1_WORK"` shows the path). All of these paths are ignored by git (`.gitignore` in this folder), so `git status` stays clean. Short-lived AWS credentials expire on their own (1 and 2 hours).

## 9. Troubleshooting

| You see | Cause and fix |
|---|---|
| `sandbox-exec: sandbox_apply: Operation not permitted` | You are inside another sandbox. Use a plain terminal |
| `aws: command not found` | Install AWS CLI v2 |
| `Cannot call GetFederationToken with session credentials`, or `credentials failed` | Your default profile is a role or SSO session. Use an IAM user key (section 2) |
| `AccessDenied` on `sts:GetFederationToken`, `bedrock:InvokeModel` or `bedrock-mantle:CreateInference` | Add the action named in the message (section 2) |
| `work dir ... contains a space` | Run `source env.sh` in this shell |
| `waiting for ... .bedrock.lock, held by: ...` | Another run is active. If none is, `rm -rf .bedrock.lock` |
| Codex output: `Reconnecting... Connection failed: error sending request`, 0 tokens, or `node` aborts (`SIGABRT`) | Run `STAGES=L6a ./run_all.sh` and send `logs/diag-*.log` |
| `uv: command not found` after `source env.sh` | `uv` is not in `/opt/homebrew/bin`, `/usr/local/bin` or `/usr/bin`. Install it with Homebrew |

## 10. Feedback and issue template

Send this to the module maintainers (an issue on the repository, or the channel they name). Remove anything you do not want to share. Do not paste credentials or account IDs.

```markdown
**Coding Assistant (Mac) feedback**

- Date, macOS version, chip, RAM:
- `uv --version`, `aws --version`, `git --version`:
- Credential type in the default profile (IAM user key / assumed role / IAM Identity Center):
- Stages run (`./run_all.sh`, or STAGES=...): 
- Result per stage (pass / fail / skipped), with seconds from `logs/run_all/*.log`:
  - L0:
  - L1:
  - L2:
  - L4:
  - L5s:
  - L6a:
  - L6:
  - L7:
- First failing stage and the last 20 lines of its log:
- For a trial: task, `result.json` agent_result (exit, seconds, tokens, steps), reward.json:
- Setup time (`setup.sh` seconds, `prepare` seconds), disk used (`du -sh local ~/.coding-assistant-local`):
- Anything that differs from this guide (a command that did not work, a missing prerequisite, a confusing message):
- Would you use this setup for your own evals? Why or why not:
```
