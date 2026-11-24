"""Repository checks that run in CI. They need no AWS credentials.

Usage, from the repository root:

    python .github/scripts/check_repo.py notebooks     # every notebook is valid nbformat
    python .github/scripts/check_repo.py model-ids     # no new hard-coded Bedrock model IDs
    python .github/scripts/check_repo.py aws-ids       # no real AWS account IDs in ARNs

CONTRIBUTING.md asks that Bedrock model IDs live in Foundational Evaluations/model_config.py
or a module's own config file. Files named model_config.py or *_config.py are config files. Files that already hard-code IDs are listed in
.github/model-id-baseline.txt so this check can block new ones without breaking existing
modules. When you move a file to the shared config, delete its line from the baseline.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / ".github" / "model-id-baseline.txt"
SKIP_PARTS = {".git", "node_modules", ".venv", "cdk.out", "datasets"}

MODEL_ID = re.compile(
    r"""["'](?:(?:us|eu|apac|global|jp|au)\.)?"""
    r"(?:anthropic|amazon|meta|mistral|cohere|ai21|deepseek|openai|qwen|writer|stability|twelvelabs)"
    r"""\.[A-Za-z0-9][\w.\-:]*["']"""
)
ACCOUNT_ARN = re.compile(r"arn:aws[\w-]*:[\w-]+:[\w-]*:(\d{12}):")
PLACEHOLDER_ACCOUNTS = {"123456789012", "000000000000", "111111111111", "111122223333"}


def files(*suffixes: str):
    for path in sorted(ROOT.rglob("*")):
        rel = path.relative_to(ROOT)
        if path.is_file() and path.suffix in suffixes and not SKIP_PARTS.intersection(rel.parts):
            yield path, rel.as_posix()


def code_text(path: Path) -> str:
    """Source code of a .py file, or the code cells of a notebook."""
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix != ".ipynb":
        return text
    cells = json.loads(text)["cells"]
    return "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")


def check_notebooks() -> int:
    import nbformat

    errors = 0
    for path, rel in files(".ipynb"):
        try:
            nbformat.validate(nbformat.read(path, as_version=4))
        except Exception as exc:  # noqa: BLE001 - report every invalid notebook, not just the first
            errors += 1
            print(f"::error file={rel}::invalid notebook: {str(exc).splitlines()[0]}")
    return errors


def check_model_ids() -> int:
    baseline = {
        line.strip()
        for line in BASELINE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    errors, seen = 0, set()
    for path, rel in files(".py", ".ipynb"):
        if path.name.endswith("_config.py"):
            continue
        ids = sorted({m.strip("\"'") for m in MODEL_ID.findall(code_text(path))})
        if not ids:
            continue
        seen.add(rel)
        if rel not in baseline:
            errors += 1
            print(
                f"::error file={rel}::hard-coded Bedrock model ID(s) {', '.join(ids)}. "
                "Import them from Foundational Evaluations/model_config.py (see CONTRIBUTING.md)."
            )
    for rel in sorted(baseline - seen):
        print(f"::notice::{rel} no longer hard-codes model IDs; remove it from .github/model-id-baseline.txt")
    return errors


def check_aws_ids() -> int:
    errors = 0
    for path, rel in files(".py", ".ipynb", ".md", ".json", ".yaml", ".yml", ".ts", ".js", ".sh", ".toml"):
        text = path.read_text(encoding="utf-8", errors="replace")
        for account in sorted(set(ACCOUNT_ARN.findall(text)) - PLACEHOLDER_ACCOUNTS):
            errors += 1
            print(f"::error file={rel}::ARN with AWS account ID {account}. Use a placeholder such as 123456789012.")
    return errors


CHECKS = {"notebooks": check_notebooks, "model-ids": check_model_ids, "aws-ids": check_aws_ids}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in CHECKS:
        sys.exit(f"usage: {sys.argv[0]} {{{'|'.join(CHECKS)}}}")
    failures = CHECKS[sys.argv[1]]()
    print(f"{sys.argv[1]}: {failures} problem(s)")
    sys.exit(1 if failures else 0)
