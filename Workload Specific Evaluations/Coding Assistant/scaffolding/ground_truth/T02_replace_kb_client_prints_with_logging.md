# T02 — Replace debug print() calls in kb_client.py with logging

**Scope**: `src/agentic_platform/service/retrieval_gateway/client/kb_client.py`

The PR is judged on the dimensions below. Each dimension is pass/fail.

## Dimensions

### 1. correctness
- All three `print()` calls (originally lines 42, 44, 59) are gone.
- They are replaced by calls on a module-level
  `logger = logging.getLogger(__name__)`. `logging.getLogger("kb")` or a
  hard-coded name is a fail — must use `__name__`.
- The level is `logger.debug(...)` — these are diagnostic, not warnings
  or info.
- The interpolated content matches what was being printed (results,
  item, all_results) so the diagnostic is preserved, not lost.

### 2. scope_discipline
- The diff touches exactly one file: `kb_client.py`.
- No edits to the pagination loop, the request builder, or the filter
  conversion helpers.
- No reformatting of unrelated lines.

### 3. logging_setup
- `import logging` is added to the imports block at the top of the file
  (don't put it mid-file).
- The `logger = logging.getLogger(__name__)` line lives at module scope,
  near the existing module-level `knowledgebase_id` / `bedrock_client`
  declarations — not inside a method.
- No `logging.basicConfig(...)` call is added — this is a library
  module, not an entrypoint, so it should not configure root logging.

## Red flags (any one → overall fail)

- Adds `logging.basicConfig(...)` anywhere in this file.
- Uses `logger.info` or `logger.warning` instead of `debug` — these
  messages will spam the gateway's logs in production.
- Removes the diagnostic content entirely (e.g. `logger.debug("retrieve called")`
  with no result/item context) — that's information loss, not a fix.
- Edits other `print()` calls elsewhere in the repo (those are T01 or
  separate work).
