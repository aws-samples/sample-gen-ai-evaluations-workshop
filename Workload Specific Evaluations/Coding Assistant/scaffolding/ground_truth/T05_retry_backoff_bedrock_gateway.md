# T05 — Add exponential backoff to BedrockGatewayClient

**Scope**: `src/agentic_platform/core/client/llm_gateway/bedrock_gateway_client.py`

The PR is judged on the dimensions below. Each dimension is pass/fail.

## Dimensions

### 1. correctness
- BOTH config blocks are updated:
  - the local-environment path (originally line 33).
  - the gateway path (originally line 43).
- Each `Config(...)` now sets `retries={'max_attempts': N, 'mode': 'adaptive'}`
  with `N >= 3`. `mode='standard'` is acceptable; `mode='legacy'` or
  omitting `mode` entirely is a fail (legacy mode does not implement
  exponential backoff with jitter).
- The gateway path keeps `signature_version=botocore.UNSIGNED`. Removing
  it breaks auth.
- The `_add_headers` event handler (lines 65-71) is untouched.
- Public method signatures (`__init__`, `chat_invoke`, `get_client`,
  `embed_invoke`) are unchanged.

### 2. scope_discipline
- Only `bedrock_gateway_client.py` is modified.
- No edits to `litellm_gateway_client.py`, `llm_gateway_client.py`, or
  any consuming agent.
- No new imports — botocore is already imported.

### 3. resilience_quality
- The retry policy is centralized in the `Config` object, not bolted on
  via a hand-rolled `for attempt in range(N): ...` loop. botocore handles
  this natively.
- No `time.sleep()` calls added.
- Both the local and gateway paths use the same `max_attempts` and
  `mode` — the two paths must behave consistently.

## Red flags (any one → overall fail)

- Adds a new dependency: `tenacity`, `backoff`, `retry`, or anything else
  pulled into requirements.
- Implements a hand-rolled retry loop around `chat_invoke` instead of
  using botocore's Config — duplicates work the SDK already does and
  diverges from how the rest of the platform's boto3 clients behave.
- Removes `signature_version=botocore.UNSIGNED` from the gateway path
  — silent auth breakage.
- Changes the public API of `BedrockGatewayClient` (e.g. adds a
  `max_attempts` constructor argument) without the issue asking for it
  — gratuitous surface-area expansion.
- Sets `mode='legacy'` or omits `mode` — does not actually deliver
  backoff.
