"""
Environment helper for running the S2S evaluation pipeline on both
local machines and SageMaker Studio.

Detects the runtime environment automatically and provides the correct
URLs and commands for starting servers.

Usage in notebook:
    from environment import env
    env.frontend_url        # http://localhost:3000 or SageMaker proxy URL
    env.websocket_url       # ws://localhost:8081/ws or SageMaker proxy URL
    env.is_sagemaker        # True / False
    env.frontend_start_cmd  # command string to start the frontend
    env.frontend_cwd        # working directory for the frontend command
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Ports (single source of truth)
# ---------------------------------------------------------------------------
FRONTEND_PORT = 3000
WEBSOCKET_PORT = 8081
WEBSOCKET_PATH = "/ws"


def _get_sagemaker_base_url() -> str | None:
    """Return the SageMaker Studio space base URL, or None if not in Studio."""
    try:
        with open("/opt/ml/metadata/resource-metadata.json", "r") as f:
            data = json.load(f)
            domain_id = data["DomainId"]
            space_name = data["SpaceName"]
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, KeyError) as exc:
        print(f"Error reading SageMaker metadata: {exc}")
        sys.exit(1)

    import boto3
    client = boto3.client("sagemaker")
    response = client.describe_space(DomainId=domain_id, SpaceName=space_name)
    return response["Url"]  # e.g. https://d-xxxx.studio.us-west-2.sagemaker.aws/jupyter/default


class _Environment:
    """Lazy-initialised singleton that holds runtime environment info."""

    def __init__(self):
        self._base_url: str | None = ...  # sentinel — not yet resolved
        self._resolved = False

    # -- internal -----------------------------------------------------------

    def _resolve(self):
        if self._resolved:
            return
        self._base_url = _get_sagemaker_base_url()
        self._resolved = True

    # -- public properties --------------------------------------------------

    @property
    def is_sagemaker(self) -> bool:
        self._resolve()
        return self._base_url is not None

    @property
    def frontend_url(self) -> str:
        """URL for humans to open in a browser.

        On SageMaker this is the authenticated proxy URL; locally it is
        plain localhost.
        """
        self._resolve()
        if self._base_url is None:
            return f"http://localhost:{FRONTEND_PORT}"
        return f"{self._base_url}/proxy/{FRONTEND_PORT}/"

    @property
    def websocket_url(self) -> str:
        self._resolve()
        if self._base_url is None:
            return f"ws://localhost:{WEBSOCKET_PORT}{WEBSOCKET_PATH}"
        ws_base = self._base_url.replace("https://", "wss://").replace("http://", "ws://")
        return f"{ws_base}/proxy/{WEBSOCKET_PORT}/"

    # -- frontend start strategy --------------------------------------------

    @property
    def frontend_start_cmd(self) -> str:
        """
        Local  → ``npm run start``  (CRA dev server with hot-reload)
        SageMaker → ``npm run build && python3 -m http.server 3000 --directory build``
                     (builds first if needed, then serves static files)
        """
        if self.is_sagemaker:
            return (
                f"npm run build && python3 -m http.server {FRONTEND_PORT} "
                f"--directory build"
            )
        return "npm run start"

    @property
    def frontend_cwd(self) -> str:
        """Working directory for the frontend command."""
        return str((HERE / "sample_s2s_app" / "react-client").resolve())

    # -- backend start strategy ---------------------------------------------

    @property
    def _venv_prefix(self) -> str:
        """Shell prefix that activates the project venv if it exists."""
        venv_activate = HERE / "venv" / "bin" / "activate"
        if venv_activate.exists():
            return f"source {venv_activate} && "
        return ""

    @property
    def backend_start_cmd(self) -> str:
        """Backend command — activates venv first so dependencies are found.

        Local     → run_server_with_telemetry.sh (websockets + OpenTelemetry)
        SageMaker → run_aiohttp_server.py (aiohttp, binds 0.0.0.0, with OTel)
        """
        if self.is_sagemaker:
            return (
                f'{self._venv_prefix}'
                f'opentelemetry-instrument python run_aiohttp_server.py 2>&1 | tee "telemetry.log"'
            )
        return (
            f'{self._venv_prefix}'
            f'./run_server_with_telemetry.sh 2>&1 | tee "telemetry.log"'
        )

    @property
    def backend_cwd(self) -> str:
        return str((HERE / "sample_s2s_app" / "python-server").resolve())

    def summary(self):
        """Print a human-readable summary of the detected environment."""
        mode = "SageMaker Studio" if self.is_sagemaker else "Local"
        print(f"🌐 Environment : {mode}")
        print(f"   Frontend URL : {self.frontend_url}")
        print(f"   WebSocket URL: {self.websocket_url}")
        print(f"   Frontend cmd : {self.frontend_start_cmd}")


# Module-level singleton
env = _Environment()
