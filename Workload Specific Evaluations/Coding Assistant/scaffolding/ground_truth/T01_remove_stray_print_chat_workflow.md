# T01 — Remove stray print() from langgraph_chat workflow

**Scope**: `src/agentic_platform/agent/langgraph_chat/chat_workflow.py`

The PR is judged on the dimensions below. Each dimension is pass/fail.
"Partial" is not a rating — when in doubt, fail with a one-line reason.

## Dimensions

### 1. correctness
- Line 51 (`print(response)`) is gone in the diff.
- The surrounding `LangGraphChat.run()` method still returns the same
  `Message(role='assistant', text=response.text)` object — the print
  removal must not perturb the return value.
- The diff is a deletion, not a comment-out. `# print(response)` is
  also a fail.

### 2. scope_discipline
- The diff touches exactly one file: `chat_workflow.py`.
- No new imports, no logger introduction, no formatting churn on
  unrelated lines.
- No edits to other agents (`agentic_chat`, `agentic_rag`, `jira_agent`)
  or to other prints in the repo — those are separate work.

### 3. minimality
- The change is a single-line deletion; the diff size is < 5 lines
  (excluding the standard `--- / +++` headers).
- No reformatting of lines 50 or 52.

## Red flags (any one → overall fail)

- Replaces `print(response)` with `logger.debug(...)` — the task
  explicitly says no logger substitution.
- Adds `import logging` or `logger = logging.getLogger(__name__)`.
- Touches files outside `chat_workflow.py`.
- Edits other `print()` calls in the repo (those belong to T02).
