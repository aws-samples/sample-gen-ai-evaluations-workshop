# Prompt Library

Copy-paste-ready prompts the notebooks tell users to give to Claude Code
(or Kiro, or Claude.ai). Each is parameterized — swap the bracketed
placeholders for your repo / file paths / etc.

Phrases like "put output at path X" aren't incidental: the validation
cells in the notebooks read artifacts from those exact paths.

---

## 02 — Draft the task set

> Please open `[target repo clone path]` and draft a set of 6-8
> coding-agent evaluation tasks in `scaffolding/tasks/tasks.yaml`. Follow
> the schema in `scaffolding/task_schema_example.yaml` exactly.
>
> For each task you draft:
>   - Actually open the files in the repo and verify the rough edge you
>     describe is really there. Include file paths and approximate line
>     numbers in the issue_description.
>   - Span difficulties: roughly 2 easy, 3-4 medium, 1-2 hard.
>   - Cover a mix of categories: targeted single-file edit, multi-file
>     feature, test writing, refactor of shared abstractions, defensive
>     coding.
>   - For at least two tasks, require use of the repo's code-graph MCP
>     server (at `src/agentic_platform/mcp_server/code_graph_mcp_server/`)
>     — set those tasks' `expected_tools.required` to include
>     `find_callers` or `find_dependencies`.
>
> Do NOT write the rubrics yet — that's a later step. Just tasks.yaml.
> When you're done, print a summary of the tasks you drafted.

---

## 03 — Draft the rubrics

> For each task in `scaffolding/tasks/tasks.yaml`, draft a ground-truth
> review rubric at `scaffolding/ground_truth/<task_id>.md`. Follow the
> format in `scaffolding/rubric_schema_example.md`.
>
> Each rubric must:
>   - Have 3-5 dimensions with concrete, task-specific criteria (not
>     generic "the code is good").
>   - Include a scope_discipline dimension — PRs that make drive-by
>     edits beyond the task should fail.
>   - Include 2-4 red flags drawn from actual risks for that task
>     (e.g. "adds a top-level dependency just for this fix").
>   - Use binary pass/fail language — avoid "mostly", "sort of", etc.
>
> When you're done, print the list of files you created.

---

## 04 — Build the gold-standard set

> Help me build the calibration set for the PR reviewer. I'll pick 3-5
> merged PRs from `[repo url]` that exercise the same rubrics we've
> written. For each one:
>
> 1. Use `gh pr diff <N>` to fetch the diff, save it to
>    `scaffolding/gold_standard/diffs/<slug>.diff`.
> 2. Open the diff and read the corresponding rubric in
>    `scaffolding/ground_truth/<task_id>.md`.
> 3. Walk through each dimension and give me your best read on pass/fail
>    with a one-line reason.
>
> I'll review your verdicts and override where I disagree — your
> verdicts are not the gold standard, the final ones I sign off on are.
>
> Write the final entry at
> `scaffolding/gold_standard/<slug>.yaml` using the schema in
> `scaffolding/gold_entry_schema_example.yaml`.

---

## 05 — Iterate on the rubric when calibration fails

> The automated PR reviewer disagreed with me on these dimensions:
>
> [paste the disagreements frame from notebook 05]
>
> For each disagreement, decide whether the fix is:
>   (a) tighten the rubric wording so the automated reviewer interprets
>       it the way I do, or
>   (b) accept that my verdict was a judgment call and update the gold
>       standard.
>
> Propose specific rubric edits for the (a) cases and show them as diffs
> against the current rubric files. Don't apply them until I approve.

---

## 06 — Scaffold the custom coding agent

> Build a minimal but real coding agent at
> `scaffolding/my_agent/` with this contract:
>
>   python -m scaffolding.my_agent \
>       --task-id <id> --tasks-file <path> --repo <path> \
>       --out <diff.patch> --trace-out <trace.json>
>
> Requirements:
>   - Use Strands Agents + `strands.models.BedrockModel` with
>     `us.anthropic.claude-sonnet-4-5-20250929-v1:0`.
>   - Expose file tools (read, write, edit, list_dir), a bash tool, and
>     an MCP client to the repo's code_graph_mcp_server (stdio transport,
>     launched via `uv run python -m agentic_platform.mcp_server.code_graph_mcp_server.server stdio`).
>   - The `--out` file must be a unified git diff of the agent's changes
>     against the pinned SHA.
>   - The `--trace-out` file must be a JSON list of
>     `{"tool": <name>, "input": <dict>}` entries, one per tool call.
>   - Exit code 0 if the agent emitted TASK_COMPLETE, 2 otherwise.
>
> When done, I'll run a contract check against a no-op task to verify
> the CLI shape.
