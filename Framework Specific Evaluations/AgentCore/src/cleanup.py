import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = MODULE_ROOT / "agentcore" / "agentcore.json"
GENERATED_DIR = MODULE_ROOT / "generated"
RESOURCE_ARRAYS = (
    "runtimes",
    "memories",
    "knowledgeBases",
    "credentials",
    "evaluators",
    "onlineEvalConfigs",
    "agentCoreGateways",
    "policyEngines",
    "configBundles",
    "abTests",
    "harnesses",
    "datasets",
    "payments",
)


def empty_project_config(config: dict) -> dict:
    cleaned = dict(config)
    for key in RESOURCE_ARRAYS:
        cleaned[key] = []
    return cleaned


def deploy_empty_project() -> None:
    original = CONFIG_PATH.read_bytes()
    config = json.loads(original)
    empty_config = empty_project_config(config)
    CONFIG_PATH.write_text(
        json.dumps(empty_config, indent=2) + "\n",
        encoding="utf-8",
    )
    try:
        subprocess.run(
            ["agentcore", "deploy", "-y"],
            cwd=MODULE_ROOT,
            check=True,
        )
    finally:
        CONFIG_PATH.write_bytes(original)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Delete AgentCore resources managed by this workshop project."
    )
    parser.add_argument("--yes", action="store_true", help="Confirm resource deletion.")
    parser.add_argument(
        "--keep-generated",
        action="store_true",
        help="Keep local generated traces and evaluation results.",
    )
    args = parser.parse_args()

    if not args.yes:
        print("Cleanup changes AWS resources. Re-run with --yes to continue.")
        return 2

    try:
        deploy_empty_project()
    except subprocess.CalledProcessError as error:
        print(f"AgentCore cleanup deploy failed with exit code {error.returncode}.", file=sys.stderr)
        return error.returncode or 1

    if not args.keep_generated:
        shutil.rmtree(GENERATED_DIR, ignore_errors=True)

    print("Managed project resources were removed and the workshop config was restored.")
    print("Review retained CloudWatch logs, KMS keys, and externally managed resources separately.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

