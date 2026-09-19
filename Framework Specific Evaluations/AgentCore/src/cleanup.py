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


def deploy_empty_project(target_name: str) -> None:
    original = CONFIG_PATH.read_bytes()
    config = json.loads(original)
    empty_config = empty_project_config(config)
    CONFIG_PATH.write_text(
        json.dumps(empty_config, indent=2) + "\n",
        encoding="utf-8",
    )
    try:
        subprocess.run(
            ["agentcore", "deploy", "--target", target_name, "--yes"],
            cwd=MODULE_ROOT,
            check=True,
        )
    finally:
        CONFIG_PATH.write_bytes(original)


def clear_generated_outputs() -> None:
    GENERATED_DIR.mkdir(exist_ok=True)
    for path in GENERATED_DIR.iterdir():
        if path.name == ".gitkeep":
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Delete AgentCore resources managed by this workshop project."
    )
    parser.add_argument("--yes", action="store_true", help="Confirm resource deletion.")
    parser.add_argument(
        "--target",
        default="default",
        help="Deployment target to clean up (default: default).",
    )
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
        deploy_empty_project(args.target)
    except subprocess.CalledProcessError as error:
        print(f"AgentCore cleanup deploy failed with exit code {error.returncode}.", file=sys.stderr)
        return error.returncode or 1

    if not args.keep_generated:
        clear_generated_outputs()

    print("Managed project resources were removed and the workshop config was restored.")
    print("Review retained CloudWatch logs, KMS keys, and externally managed resources separately.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
