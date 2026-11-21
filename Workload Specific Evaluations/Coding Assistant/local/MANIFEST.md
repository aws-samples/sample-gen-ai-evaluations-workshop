# Manifest: files of this folder

## Files in this folder

The files below, plus the copies listed after them. No binary is included.

| File | Bytes | What it is |
|---|---|---|
| `README.md` | 11,159 | what it is, how a task runs on the Mac, setup, what is verified |
| `TESTER-GUIDE.md` | 19,501 | prerequisites, AWS permissions, commands, test ladder, results, clean-up, feedback template |
| `MANIFEST.md` | this file | file list |
| `.gitignore` | 276 | ignores everything the setup and the runner generate |
| `setup.sh` | 2,025 | installs Node 22.23.5, srt 0.0.83, Codex 0.162.1 and ripgrep 15.2.0 into this folder, and writes the GNU-style `timeout` shim to `bin/timeout` |
| `env.sh` | 948 | `source` it in each shell: own `HOME`, caches, `PATH`, and the work directory `P1_WORK` |
| `run_all.sh` | 4,621 | the test ladder (stages L0, L1, L2, L4, L5s, L6a, L6, L7) |
| `runner/p1.py` | 32,816 | the runner: `prepare`, `trial`, `job`, `base-suite`, `sandbox-run`, `verifier-venv`. Standard library only |
| `scripts/agent-policy.json` | 671 | session policy for the agent's federated token (three model IDs) |
| `scripts/diag_codex.sh` | 3,018 | stage L6a: why Codex fails inside the sandbox, if it does |
| `scripts/l2_inner.sh` | 3,316 | stage L2: network probes, run inside the agent's sandbox |
| `scripts/l4_seal.py` | 4,130 | stage L4: seal check, hidden-test reachability, leakage scan |
| `05 Local mode walkthrough.ipynb` | 120,531 | the walkthrough notebook: the approach, the task format, setup, the sandbox checks, trials, controls, metrics, clean-up |

The task data and settings are in this folder too, and the runner only reads them:

| Path | What it is |
|---|---|
| `datasets/` | the four task directories, the PR-review candidates, `rubric.md` and `risk-map.toml`, and `_shared/` (one source for the verifier files that are the same in each task; `python datasets/_shared/sync.py --check` reports a copy that differs) |
| `model_config.py` | all model IDs: the agent and the judges |
| `codex-bedrock.toml` | the Codex settings that the runner copies into each trial |

## Generated when you use it (ignored by git)

| Path | Created by | Size |
|---|---|---|
| `node/`, `npm/`, `npm-cache/` | `setup.sh` | about 0.7 GB |
| `bin/` (`rg`, `timeout`) | `setup.sh` | 4 MB |
| `uv-cache/`, `uv-python/` | the first environment build | about 1.1 GB |
| `home/` | `env.sh` (the `HOME` of the toolchain) | under 1 MB |
| `jobs/<job>/<trial>/` | `p1 trial`, `p1 job` | about 10 MB per 22 trials |
| `logs/` | `run_all.sh`, `base-suite` | under 1 MB |
| `.bedrock.lock/` | `p1 job`, `run_all.sh` while they run | none |
| `~/.coding-assistant-local/<id>/work/` | the runner (`<id>`: a short hash of this folder's path): sealed clones, trial directories, verifier copies | 0.2 GB sealed clones, 0.1 to 0.6 GB per trial directory |

## Not included

Stages L3 (scripted break-out probes), L3rt (LLM red team) and L5 (grader on 22 fixed patches against reference reward files) are not included. They need test fixtures (decoy credential files, a canary file) or reference reward files that are not in this folder. Their results are summarised in `README.md`.
