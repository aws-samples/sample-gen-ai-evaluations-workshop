# Pruning: keep one option, delete the other two

Temporary: this file is deleted by the steps below.

Each of `docker/`, `local/` and `agentcore/` is complete by itself. Nothing in one of them reads a file outside its own folder, and every path inside is resolved from the folder itself (`Path(__file__)`, the notebook's working folder, or `$(dirname "$0")`). So promoting the survivor to the module root with `git mv` needs no path edit. `README.md` (the chooser), `docs/`, `tools/` and this file exist only for the test week.

## 1. Before you delete: mirror the fixes of the test week

Fixes made in one copy of a shared file (`model_config.py`, `codex-bedrock.toml`, anything in `datasets/`) are lost if they are only in a deleted folder. From the module folder (`Workload Specific Evaluations/Coding Assistant`):

```bash
python3 tools/check_copies.py          # lists each shared file whose copies differ, and which folders agree
```

Copy the fixed version into the survivor (for example `cp local/datasets/_shared/pr-review/tests/score.py agentcore/datasets/_shared/pr-review/tests/score.py`), then run `python datasets/_shared/sync.py` in the survivor's folder and commit.

## 2. Keep one, delete the rest, promote the survivor

Run from the module folder, on a clean `git status`. Set `KEEP` to `docker`, `local` or `agentcore`.

```bash
KEEP=local                                                 # docker | local | agentcore
python3 tools/check_escapes.py --dir "$KEEP" --as "$KEEP"  # must print "clean"
for o in docker local agentcore; do [ "$o" = "$KEEP" ] || git rm -r -q "$o"; done
git rm -r -q README.md docs tools PRUNING.md
git -C "$KEEP" ls-files | cut -d/ -f1 | sort -u | while IFS= read -r f; do git mv "$KEEP/$f" "./$f"; done
ls -A "$KEEP"                                              # only ignored, generated files are left (toolchain, jobs/, logs/)
git status --short | grep -vE '^(R|D) '                    # expect one line: M README.md (the survivor's README replaces the chooser)
```

The last `git mv` moves every tracked top-level entry of the survivor, hidden files included (`.gitignore`, and `.env.example` for docker). Files that git ignores stay in `$KEEP/`. Delete them, or move what you want to keep (for example `local/jobs/`), then `rmdir "$KEEP"`.

Check the result before you commit:

| Survivor | Check (from the module folder) |
|---|---|
| docker | `python -m pytest -q tests_local`; `python datasets/_shared/sync.py --check`; open notebook 04 (no Docker, no AWS) |
| local | `source ./env.sh && uv run --no-project --python 3.12 runner/p1.py --help` (the runner refuses to start without `env.sh`, which sets `P1_WORK`); `LOCAL_NB_DRY_RUN=1` run of the notebook (see local/README.md); `python3 datasets/_shared/sync.py --check` |
| agentcore | `python3 -m py_compile p2_v2.py`; `python3 p2_v2.py policies`; `python3 scripts/test_runner_v2.py` |

Then commit, for example `git commit -m "Coding Assistant: keep the $KEEP option"`.

## 3. Edit afterwards

These were found while the folders were split. None of them breaks a run; they are text that is stale or that only the deleted options made true.

**Links and text elsewhere in the repository**

| File | What to change |
|---|---|
| `Interactive Learning/workload evals/SKILL-coding-assistant.md` (lines 18, 42, 369) | They name `Coding Assistant/docker/`. Drop `/docker` in all three. If the survivor is not docker, also rewrite "Path B" (lines 13, 20 and 369): it describes notebooks 01 to 03, Harbor and Docker. Path A still works: it reads `datasets/` only, which every survivor has |
| `Workload Specific Evaluations/README.md` (the Coding Assistant entry, lines 139 to 145) | The link `Coding%20Assistant/` stays valid. If the survivor is not docker, rewrite the text: it describes Harbor, local Docker and notebooks 02 to 04 |
| `README.md` at the repository root (line 40) | Same: "run in Harbor on local Docker. A replay notebook..." is true only for docker |
| `plugins/eval-builder/skills/eval-builder/SKILL.md` (line 129) | Same: "sealed Harbor tasks on Docker" |
| `Interactive Learning/curriculum.md` (line 51) | The folder link stays valid. Nothing to do |

Check with `grep -rn "Coding Assistant/docker\|Coding%20Assistant/docker\|lite-modes" .` from the repository root: it must find nothing.

**Comments in the shared files (identical in each option, so they were not edited)**

The copies in `local/` and `agentcore/` are byte-identical to `docker/` on purpose, so their comments still describe Harbor and the notebooks 01 to 03. If the survivor is not docker, reword them:

- `model_config.py`: the module docstring ("all Coding Assistant notebooks", "reach the Harbor tasks", "harbor run") and the comments "Notebooks 01, 02, and 03", "notebooks 02 and 03", "the Harbor verifier", "(notebook 02)", "(notebook 03)". `CLAUDE_CODE_MODEL_ID` is used only by docker.
- `codex-bedrock.toml`: the header comment ("Harbor uploads this file ... harbor run --ae").
- `datasets/_shared/sync.py`: the docstring (Harbor, notebooks 02 and 03).
- `datasets/*/tasks/*/task.toml`: comments such as "Harbor's default shared mode" and "the notebook".
- `datasets/*/tasks/*/environment/Dockerfile`, `tests/test.sh`, `tests/score.py`, `tests/judge.toml`, `solution/solve.sh`, `tests/lib/aggregate.py`: comments that name Harbor or a notebook. Edit the `_shared/` source, then run `python datasets/_shared/sync.py`.
- `datasets/strands-evals-coding/tasks/*/instruction.md` line 20 says "You are inside of a Docker container." The agent reads it. **local** replaces that exact sentence in `render_instruction` of `runner/p1.py`: change both together, or not at all.
- For local and agentcore, `datasets/recorded-runs/` and `datasets/awslabs-mcp-pr-review/metric.py` were not copied (neither the runner nor the notebook reads them). Nothing to do.

**Per survivor**

- **docker**: nothing else. Its README describes Docker and Harbor as its own tools.
- **local**:
  - The default work folder is `~/.coding-assistant-local/<8 hex of the SHA-1 of the folder path>/work`. Promoting the folder changes its path, so the old work folders of the test week are orphaned: `rm -rf ~/.coding-assistant-local` once (it is rebuilt).
  - The notebook name `05 Local mode walkthrough.ipynb` reads like part of a series. A rename (for example `Coding agent eval on your Mac.ipynb`) needs the links in `README.md` and `MANIFEST.md` and the start command in the notebook's first cell changed.
  - `runner/p1.py` comments still name Harbor or Docker (for example line 2, "no Docker, no Harbor", and line 426, "a Harbor-like layout"; `grep -n -i "harbor\|docker" runner/p1.py`). Text only.
  - `MANIFEST.md` lists byte sizes; refresh them after edits.
- **agentcore**:
  - The notebook name `05 AgentCore walkthrough.ipynb` reads like part of a series; a rename needs the link in `README.md` changed.
  - `p2_v2.py` and the notebook tag resources `Project=evals-streamline-spike` and use the prefixes `evals-p2b-` and `evals-nb-`. Rename them only after every resource of the old names is deleted (`scripts/teardown.sh <prefix>`), or the cleanup will miss them.
