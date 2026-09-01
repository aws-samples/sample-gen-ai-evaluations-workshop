# T03 — Make the Bedrock region configurable in kb_client.py

**Scope**: `src/agentic_platform/service/retrieval_gateway/client/kb_client.py`

The PR is judged on the dimensions below. Each dimension is pass/fail.

## Dimensions

### 1. correctness
- Line 14's hardcoded `region_name="us-west-2"` is replaced with a
  read of the `AWS_REGION` env var.
- The replacement uses `os.getenv('AWS_REGION', '<default>')` so behaviour
  in deployments without the env var is preserved. The default may be
  `'us-west-2'` (preserves the previous behaviour exactly) or
  `'us-east-1'` (matches the bedrock_kb MCP server convention) — either
  is acceptable.
- `os` is already imported on line 1; the diff does not need to add the
  import.

### 2. scope_discipline
- Only `kb_client.py` is modified.
- The pagination loop, request builder, and filter logic are untouched.
- The `bedrock-agent-runtime` client name and any other arguments to
  `boto3.client(...)` are unchanged.

### 3. config_convention
- The fix reuses the existing `AWS_REGION` convention. Introducing a new
  env var like `BEDROCK_REGION` or `KB_REGION` is a fail — the issue
  description explicitly forbids it.
- No reading of region from a file or settings object — env var only.

## Red flags (any one → overall fail)

- Introduces a new env var name (`BEDROCK_REGION`, `KB_REGION`, etc).
- Edits the `bedrock_kb_mcp_server/server.py` file in the same PR — that
  file already does the right thing, and touching it widens the blast
  radius unnecessarily.
- Wraps the boto3 call in a `try / except` to swallow region errors —
  that's masking misconfiguration, not fixing it.
- Adds a `Config(region_name=...)` object instead of just passing
  `region_name=` directly — extra ceremony with no upside.
