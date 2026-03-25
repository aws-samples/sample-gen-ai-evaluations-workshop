"""
Helper to generate the correct frontend/websocket URLs when running in SageMaker Studio
and automatically update the relevant .env files.

Run this script before starting the app:
    python sagemaker_helper.py
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent


def get_proxy_base_url() -> str:
    """Returns the SageMaker Studio space base URL, or None if not in Studio."""
    try:
        with open("/opt/ml/metadata/resource-metadata.json", "r") as f:
            data = json.load(f)
            domain_id = data["DomainId"]
            space_name = data["SpaceName"]
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error reading SageMaker metadata: {e}")
        sys.exit(1)

    import boto3
    sagemaker_client = boto3.client("sagemaker")
    response = sagemaker_client.describe_space(DomainId=domain_id, SpaceName=space_name)
    return response["Url"]  # e.g. https://d-xxxx.studio.us-west-2.sagemaker.aws/jupyter/default


def get_frontend_url(base_url: str, port: int = 3000) -> str:
    if base_url is None:
        return f"http://localhost:{port}"
    return base_url + f"/proxy/{port}/"


def get_websocket_url(base_url: str, port: int = 8081, path: str = "/ws") -> str:
    if base_url is None:
        return f"ws://localhost:{port}{path}"
    # Convert https:// -> wss://
    # No path suffix — SageMaker proxy handles the WebSocket upgrade at the port level
    ws_base = base_url.replace("https://", "wss://").replace("http://", "ws://")
    return ws_base + f"/proxy/{port}/"

def update_env_file(filepath: Path, updates: dict):
    """Update key=value pairs in an env file, preserving all other content."""
    content = filepath.read_text()
    for key, value in updates.items():
        pattern = rf"^{re.escape(key)}=.*$"
        replacement = f"{key}={value}"
        if re.search(pattern, content, flags=re.MULTILINE):
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        else:
            content += f"\n{replacement}\n"
    filepath.write_text(content)
    print(f"  Updated {filepath.relative_to(HERE)}")


if __name__ == "__main__":
    base_url = get_proxy_base_url()

    if base_url is None:
        print("Not running in SageMaker Studio — localhost URLs will be used, no changes needed.")
        sys.exit(0)

    frontend_url = get_frontend_url(base_url, port=3000)
    websocket_url = get_websocket_url(base_url, port=8081, path="/ws")

    print(f"\nSageMaker Studio detected.")
    print(f"  Frontend URL : {frontend_url}")
    print(f"  WebSocket URL: {websocket_url}\n")

    # Update react-client .env
    react_env = HERE / "sample_s2s_app/react-client/.env"
    update_env_file(react_env, {"REACT_APP_WEBSOCKET_URL": websocket_url})

    # Update playwright test .env.test
    test_env = HERE / "test/.env.test"
    update_env_file(test_env, {"FRONTEND_URL": frontend_url})

    print("\nDone. You can now start the React app and run Playwright tests.")
