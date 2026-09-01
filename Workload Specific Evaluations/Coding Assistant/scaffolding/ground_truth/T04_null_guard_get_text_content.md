# T04 — Add null-guard for get_text_content() in agent invoke methods

**Scope**: `src/agentic_platform/agent/agentic_chat/agent/agentic_chat_agent.py`

The PR is judged on the dimensions below. Each dimension is pass/fail.

## Dimensions

### 1. correctness
- The synchronous `invoke` method (around lines 48-63) checks whether
  `request.message.get_text_content()` returned `None` BEFORE
  dereferencing `.text`. On `None`, it raises `ValueError` with a
  clear message (e.g. `"AgenticRequest.message has no text content"`).
- The async `invoke_stream` method (around lines 65-77) has the
  equivalent check before passing `text_content.text` into
  `self.agent.stream_async(...)`.
- The error type is `ValueError` (or a subclass) — not `Exception`,
  not a custom unraised type, not silently substituting `""`.

### 2. scope_discipline
- Only `agentic_chat_agent.py` is modified.
- `agentic_rag_agent.py` and `jira_agent.py` are NOT touched in this PR
  — the issue scopes the fix to the agentic_chat agent specifically.
- `memory_models.py` is not modified — the existing
  `get_text_content -> Optional[TextContent]` signature is the
  source of truth.
- No changes to `controller/agentic_chat_controller.py`, the streaming
  converter, or the api_error_decorator.

### 3. error_handling_quality
- The check is explicit (`if text_content is None:`), not a try/except.
- The raised error includes context useful to the API caller — at
  minimum the field name (`message`) and what was missing (text).
- The error is raised, not caught locally — `api_error_decorator`
  upstream is responsible for the HTTP translation.

## Red flags (any one → overall fail)

- Catches the AttributeError after the fact (`try: text_content.text except: ...`)
  instead of a `None` guard.
- Substitutes an empty string or default text when text_content is None
  — that's silently dropping the bug, not fixing it.
- Wraps the entire invoke method in a try/except to swallow all errors.
- Modifies `Message.get_text_content()` in `memory_models.py` to never
  return `None` — that changes the contract for every other caller.
- Drive-by edits the other two agents — out of scope for this PR.
