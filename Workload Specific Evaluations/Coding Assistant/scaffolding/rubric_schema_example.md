# T01 — Example task title

**Scope**: `<file path the PR should touch>`

The PR is judged on the dimensions below. Each dimension is pass/fail.
"Partial" is not a rating — when in doubt, fail with a one-line reason.

## Dimensions

### 1. correctness
- Concrete bullet describing what "pass" requires.
- Additional bullet calling out an easy-to-miss constraint.
- Specific red-flag to reject on (e.g. "returns None instead of the apology string is a fail").

### 2. scope_discipline
- No behavior changes beyond what the task specifies.
- No drive-by edits to unrelated files.
- Public signatures unchanged unless task requires.

### 3. test_coverage
- Concrete list of cases the tests must cover.
- Tests pass under `make test-unit` (or whatever your repo uses).
- Tests live in the expected tests/ path mirroring the source.

## Red flags (any one → overall fail)

- Specific smell that should never land. Be concrete — "adds a dependency" is less useful than "pulls in tenacity or backoff".
- Another specific smell, ideally drawn from real past bad PRs.
- One more if relevant. 2-4 red flags is typical.
