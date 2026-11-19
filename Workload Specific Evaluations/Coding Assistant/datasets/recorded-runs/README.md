# Recorded runs

Real Harbor jobs of notebooks 02 and 03, for `04 Replay recorded runs (no Docker).ipynb`. Recorded on
2026-11-18 in us-east-1 with `coding_eval.record_job`, one trial for each task:

| Agent | Model | Recorded |
|---|---|---|
| `codex/` | Codex 0.162.1 with `openai.gpt-6-sol` | `pr-review/` (notebook 02), `coding/` (notebook 03) |
| `claude-code/` | Claude Code 2.1.327 with `us.anthropic.claude-sonnet-5-5` | `pr-review/`, `coding/` |

Judges and verifier: `us.anthropic.claude-haiku-5-5`. The coding runs use the PR 338 issue text that states its hidden requirements (mixed naive and aware times; "at or before"). Taste uses the yes/no checks of `tests/lib/judges.py`.

Each trial has `result.json` (selected fields only), `agent/trajectory.json`, and the verifier files.
The raw agent sessions, `junit.xml`, and `pytest.log` are left out. The repositories are open source
(awslabs/mcp, strands-agents/evals), so the trajectories show their public code and metadata.

To refresh: run notebooks 02 and 03, then `ce.record_job("jobs/<job>", ce.RECORDED_DIR / "<agent>" / "<pr-review|coding>")`.
