---
version: v1
---

# Review rubric — awslabs/mcp

Built by the overlay method: a standard review base, overlaid with the conventions awslabs/mcp states
in `DESIGN_GUIDELINES.md`, `CONTRIBUTING.md`, `.github/pull_request_template.md`, `.github/CODEOWNERS`,
`.ruff.toml`, and its CI workflows (bandit, semgrep, checkov, detect-secrets, license-header, pre-commit).

## Standard base

Grade a review on whether it catches substantive defects — correctness, security, data loss,
breaking changes — and separates them from nits (style or preference) that do not block a merge.

## Your team's conventions (overlay)

- **Security first for tools that touch customer data or AWS accounts.** Servers that execute
  user-provided input (SQL, code, specs, URLs) must validate it with explicit allowlists, layered
  checks, and timeouts; read-only modes must actually be read-only. A bypass of a guard, injection,
  SSRF, credential exposure, or missing TLS is a blocking defect.
- **Authentication follows the AWS conventions**: honour `AWS_PROFILE` / default credentials and
  `AWS_REGION`, give clear errors on auth failure, and document required IAM permissions.
- **Error handling**: catch and log exceptions with context, report via MCP context
  (`ctx.error` / `ToolError`), return meaningful messages, distinguish client vs server errors.
  Silently swallowed failures or wrong results returned as success are defects.
- **Tool and parameter definitions**: Pydantic models and `Field` descriptions are the agent-facing
  contract; a wrong or misleading description, or a renamed/removed field without a documented
  migration, is a breaking change (the PR template asks "Is this a breaking change?").
- **Tests**: changes come with pytest unit tests that mock AWS (moto/mocks, no real credentials);
  behaviour changes without tests, or tests that do not exercise the change, are worth flagging.
- **Style**: `ruff` / `ruff format` / `pyright`, license headers, loguru logging. Violations here
  are nits unless CI would fail on them.
- **Docs**: new tools, env vars, and IAM permissions are documented in the server README and
  CHANGELOG. Missing docs is normally a nit; wrong docs that would mislead a user is a defect.
