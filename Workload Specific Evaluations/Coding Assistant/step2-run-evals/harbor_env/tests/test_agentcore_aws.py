"""Session.put writes exactly the bytes given, even when a command runs twice."""
import getpass
import os
import subprocess

import pytest

import agentcore_aws as aws


class LocalSession(aws.Session):
    def __init__(self, repeat):
        self.repeat, self.calls = repeat, 0

    def cmd(self, command, timeout=300, retries=12):
        for _ in range(2 if self.repeat else 1):  # a retried command runs again
            p = subprocess.run(["bash", "-c", command], capture_output=True, text=True)
        return p.stdout + p.stderr, p.returncode, 0.1


@pytest.mark.parametrize("size", [0, 10, 40000, 100_001])
@pytest.mark.parametrize("repeat", [False, True])
def test_put_is_exact_and_private(tmp_path, size, repeat):
    data = os.urandom(size)
    target = tmp_path / "f"
    LocalSession(repeat).put(str(target), data, "0600", getpass.getuser())
    assert target.read_bytes() == data and oct(target.stat().st_mode & 0o777) == "0o600"
    assert not (tmp_path / "f.b64.d").exists()


class FakeCtl:
    def __init__(self, metadata):
        self.metadata, self.updates = metadata, 0

    def list_agent_runtimes(self, **k):
        return {"agentRuntimes": []}

    def create_agent_runtime(self, **k):
        return {"agentRuntimeId": "rid"}

    def get_agent_runtime(self, agentRuntimeId):
        return {"status": "READY", "metadataConfiguration": self.metadata}

    def update_agent_runtime(self, **k):
        self.updates += 1
        self.metadata = k["metadataConfiguration"]


@pytest.mark.parametrize("default,updates", [({"requireMMDSV2": True}, 0), (None, 1), ({"requireMMDSV2": False}, 1)])
def test_deploy_updates_only_when_imdsv2_is_not_required(monkeypatch, default, updates):
    ctl = FakeCtl(default)
    monkeypatch.setattr(aws, "ctl", ctl)
    monkeypatch.setattr(aws, "acct", lambda: "111122223333")
    monkeypatch.setattr(aws.iam, "get_role", lambda RoleName: {"Role": {"Arn": "arn:aws:iam::111122223333:role/x"}})
    monkeypatch.setattr(aws.time, "sleep", lambda s: None)
    r = aws.deploy("t_a0123456789", "V2")
    assert ctl.updates == updates and r["imdsv2_required"] is True and r["status"] == "READY"
