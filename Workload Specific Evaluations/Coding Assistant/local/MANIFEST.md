# Manifest: files of the local mode

## Files in this folder

12 files, about 80 KB. No file is larger than 31 KB, and no binary is included.

| File | Bytes | What it is |
|---|---|---|
| `README.md` | 11,418 | what the mode is, how it differs from the Docker path, setup, what is verified |
| `TESTER-GUIDE.md` | 19,298 | prerequisites, AWS permissions, commands, test ladder, results, clean-up, feedback template |
| `MANIFEST.md` | this file | file list |
| `.gitignore` | 197 | ignores everything the setup and the runner generate |
| `setup.sh` | 2,025 | installs Node 22.23.5, srt 0.0.83, Codex 0.162.1 and ripgrep 15.2.0 into this folder, and writes the GNU-style `timeout` shim to `bin/timeout` |
| `env.sh` | 745 | `source` it in each shell: own `HOME`, caches, `PATH`, and the work directory `P1_WORK` |
| `run_all.sh` | 4,621 | the test ladder (stages L0, L1, L2, L4, L5s, L6a, L6, L7) |
| `runner/p1.py` | 30,778 | the runner: `prepare`, `trial`, `job`, `base-suite`, `sandbox-run`, `verifier-venv`. Standard library only |
| `scripts/agent-policy.json` | 671 | session policy for the agent's federated token (three model IDs) |
| `scripts/diag_codex.sh` | 3,018 | stage L6a: why Codex fails inside the sandbox, if it does |
| `scripts/l2_inner.sh` | 3,316 | stage L2: network probes, run inside the agent's sandbox |
| `scripts/l4_seal.py` | 2,946 | stage L4: seal check, hidden-test reachability, leakage scan |

The module's own files (`datasets/`, `coding_eval.py`, `codex-bedrock.toml`, `model_config.py`) are used in place and are not changed or copied.

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
| `~/.coding-assistant-local/work/` | the runner: sealed clones, trial directories, verifier copies | 0.2 GB sealed clones, 0.1 to 0.6 GB per trial directory |

## Not in this mode

Stages L3 (scripted break-out probes), L3rt (LLM red team) and L5 (grader on 22 fixed patches against recorded Docker-path rewards) are not included. They need test fixtures (decoy credential files, a canary file) or reward files from the Docker path that are not part of the module. Their results are summarised in `README.md`.
