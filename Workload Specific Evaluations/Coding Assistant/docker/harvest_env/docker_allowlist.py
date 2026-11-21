"""A local Docker Harbor environment that enforces ``allowlist`` networking without kernel support.

Harbor's own Docker environment enforces ``no-network`` and ``allowlist`` with an nftables sidecar,
and only when the Docker host kernel has ``CONFIG_NFT_FIB_INET`` (``docker.py:836-865``). Docker
Desktop's VM kernel does not, so on a Mac Harbor disables egress control and then rejects any task
whose network policy is not ``public`` (``environments/base.py:782-790``).

This subclass keeps Harbor's behaviour wherever the kernel supports it, and otherwise enforces the
policy at the network layer instead:

- ``main`` is attached only to an ``internal: true`` Docker network, which has no route off the host.
- An ``egress-proxy`` sidecar (Squid) sits on that network and on an ordinary bridge network. It is
  the only thing ``main`` can reach, and it forwards ``CONNECT`` only to hosts in ``allowed_hosts``.
- ``main`` gets ``HTTPS_PROXY``/``HTTP_PROXY`` pointing at the sidecar, which is how Claude Code, the
  AWS SDKs, ``pip``, and ``npm`` find their way out.

A process that ignores the proxy variables gets nowhere — the internal network is the enforcement,
the proxy is the one exit. Hostname and wildcard-hostname entries are supported; IP and CIDR entries
are not (Squid matches the ``CONNECT`` host), and the capabilities say so, so Harbor refuses a task
that needs them rather than running it unenforced.

Load it into an unmodified Harbor by import path:

    harbor run -e harvest_env.docker_allowlist:ProxyAllowlistDockerEnvironment ...
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Final, override

from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.task.config import NetworkMode, NetworkPolicy

PROXY_SERVICE: Final = "egress-proxy"
PROXY_PORT: Final = 3128
PROXY_IMAGE: Final = "harvest-env-egress-proxy:1"
PROXY_CONTEXT: Final = Path(__file__).parent / "egress_proxy"
INTERNAL_NETWORK: Final = "harvest-allowlist-internal"
EGRESS_NETWORK: Final = "harvest-allowlist-egress"


class ProxyAllowlistDockerEnvironment(DockerEnvironment):
    """Harbor's Docker environment, with a proxy-enforced allowlist where nftables is unavailable."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # Any: Harbor's constructor signature, passed through
        self._proxy_compose_dir: tempfile.TemporaryDirectory[str] | None = None
        self._proxy_compose_path: Path | None = None
        super().__init__(*args, **kwargs)

    @property
    def _proxy_enforced(self) -> bool:
        """Whether the proxy, rather than Harbor's nftables sidecar, enforces this environment's policy.

        Harbor's native egress control wins wherever it works; the proxy covers only hosts where it was
        disabled for lack of kernel support and the task still asks for a restriction. A property, not
        an attribute: the base constructor reads ``capabilities`` (and so this) before it returns.
        """
        native = getattr(self, "_enable_egress_control", False)
        policy = getattr(self, "network_policy", None)
        return not native and policy is not None and policy.network_mode != NetworkMode.PUBLIC

    @property
    @override
    def capabilities(self) -> EnvironmentCapabilities:
        native = super().capabilities
        if not self._proxy_enforced:
            return native
        return native.model_copy(
            update={
                "disable_internet": True,
                "network_allowlist": True,
                "network_allowlist_hostnames": True,
                "network_allowlist_wildcard_hostnames": True,
                "dynamic_network_policy": True,
            }
        )

    @property
    @override
    def _docker_compose_paths(self) -> list[Path]:
        paths = super()._docker_compose_paths
        if self._proxy_enforced and self._proxy_compose_path is not None:
            paths.append(self._proxy_compose_path)
        return paths

    @override
    async def start(self, force_build: bool) -> None:
        if self._proxy_enforced:
            self._proxy_compose_path = self._write_proxy_compose_file()
        await super().start(force_build)

    @override
    async def stop(self, delete: bool) -> None:
        try:
            await super().stop(delete)
        finally:
            if self._proxy_compose_dir is not None:
                self._proxy_compose_dir.cleanup()
                self._proxy_compose_dir = None
                self._proxy_compose_path = None

    @override
    async def _apply_network_policy(self, network_policy: NetworkPolicy) -> None:
        if not self._proxy_enforced:
            await super()._apply_network_policy(network_policy)
            return
        command = ["exec", "--no-TTY", PROXY_SERVICE, "policy", *self.policy_arguments(network_policy)]
        await self._run_docker_compose_command(command)

    @staticmethod
    def policy_arguments(network_policy: NetworkPolicy) -> list[str]:
        """The sidecar's ``policy`` command arguments for a Harbor network policy."""
        match network_policy.network_mode:
            case NetworkMode.PUBLIC:
                return ["allow-all"]
            case NetworkMode.NO_NETWORK:
                return ["deny-all"]
            case NetworkMode.ALLOWLIST:
                return ["allow", *network_policy.allowed_hosts]
        raise ValueError(f"Invalid network mode: {network_policy.network_mode}")

    def _write_proxy_compose_file(self) -> Path:
        """The overlay that isolates ``main`` and adds the proxy, appended after every other file."""
        proxy_url = f"http://{PROXY_SERVICE}:{PROXY_PORT}"
        no_proxy = f"localhost,127.0.0.1,{PROXY_SERVICE}"
        arguments = self.policy_arguments(self.network_policy)
        document = {
            "services": {
                "main": {
                    "networks": [INTERNAL_NETWORK],
                    "depends_on": [PROXY_SERVICE],
                    "environment": {
                        "HTTPS_PROXY": proxy_url,
                        "HTTP_PROXY": proxy_url,
                        "https_proxy": proxy_url,
                        "http_proxy": proxy_url,
                        "NO_PROXY": no_proxy,
                        "no_proxy": no_proxy,
                    },
                },
                PROXY_SERVICE: {
                    "build": {"context": str(PROXY_CONTEXT)},
                    "image": PROXY_IMAGE,
                    "environment": {"POLICY_MODE": arguments[0], "ALLOWED_HOSTS": " ".join(arguments[1:])},
                    "networks": [INTERNAL_NETWORK, EGRESS_NETWORK],
                },
            },
            "networks": {INTERNAL_NETWORK: {"internal": True}, EGRESS_NETWORK: {}},
        }
        self._proxy_compose_dir = tempfile.TemporaryDirectory(prefix="harvest-allowlist-")
        path = Path(self._proxy_compose_dir.name) / "docker-compose-proxy-allowlist.json"
        path.write_text(json.dumps(document, indent=2), encoding="utf-8")
        return path
