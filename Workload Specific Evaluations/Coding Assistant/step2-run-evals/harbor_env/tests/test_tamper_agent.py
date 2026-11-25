"""Offline tests of the tamper agent's variant switches."""
import asyncio
import base64
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harbor.environments.base import ExecResult  # noqa: E402

from harbor_env.tests import tamper_agent as ta  # noqa: E402


@pytest.mark.parametrize("keep", [True, False])
@pytest.mark.parametrize("stage_out", [None, "invalid", "changed"])
def test_plant_code_bakes_the_variant_in_and_compiles(keep, stage_out):
    code = ta.plant_code(keep, stage_out)
    assert f"KEEP_FDS = {keep}" in code and f"STAGE_OUT = {stage_out!r}" in code and "__KEEP_FDS__" not in code
    assert "__STAGE_OUT__" not in code
    compile(code, "planted.py", "exec")


@pytest.mark.parametrize("value, mode, variant", [(None, None, 2), ("", None, 2), ("false", None, 2), ("true", "invalid", 3),
                                                  ("invalid", "invalid", 3), ("changed", "changed", 4), ("CHANGED", "changed", 4)])
def test_stage_out_option(tmp_path, value, mode, variant):
    agent = ta.TamperAgent(logs_dir=tmp_path, stage_out=value)
    assert agent._stage_out == mode and agent.variant() == variant


def test_unknown_stage_out_is_refused(tmp_path):
    with pytest.raises(ValueError, match="stage_out"):
        ta.TamperAgent(logs_dir=tmp_path, stage_out="bogus")


@pytest.mark.parametrize("value, keep", [("true", True), ("1", True), (True, True), ("false", False), (False, False), ("0", False)])
def test_keep_fds_option(tmp_path, value, keep):
    assert ta.TamperAgent(logs_dir=tmp_path, keep_fds=value)._keep_fds is keep


def test_default_is_variant_2(tmp_path):
    assert ta.TamperAgent(logs_dir=tmp_path)._keep_fds is False


class FakeEnv:
    def __init__(self):
        self.commands = []

    async def exec(self, command, timeout_sec=None, **kw):
        self.commands.append(command)
        return ExecResult(stdout="1 file changed\n", stderr="", return_code=0)


@pytest.mark.parametrize("keep, stage_out, variant", [(True, None, 1), (False, None, 2), (False, "invalid", 3), (False, "changed", 4)])
def test_run_plants_the_chosen_variant(tmp_path, keep, stage_out, variant):
    agent = ta.TamperAgent(logs_dir=tmp_path, keep_fds=keep, stage_out=stage_out)
    env = FakeEnv()
    asyncio.run(agent.run("fix it", env, None))
    b64 = re.search(r"echo (\S+) \| base64 -d >> src/strands_evals/__init__.py", env.commands[0]).group(1)
    code = base64.b64decode(b64).decode()
    assert f"KEEP_FDS = {keep}" in code and f"STAGE_OUT = {stage_out!r}" in code
    assert (tmp_path / "tamper.txt").read_text().startswith(f"variant: {variant} ")


@pytest.mark.parametrize("stage_out", ["invalid", "changed"])
def test_stage_out_loop_writes_rewards_and_forces_an_invalid_trial(tmp_path, stage_out):
    # the planted loop for 1 s, paths moved into tmp_path
    top, stage, repo = tmp_path / "logs-verifier", tmp_path / "stage", tmp_path / "repo"
    stage.mkdir()
    repo.mkdir()
    (stage / "junit.xml").write_text("<testsuite/>\n")
    (stage / "validation.json").write_text('{"status": "accepted"}')
    code = (ta.plant_code(False, stage_out).replace('os.path.isdir("/tests")', "True").replace("+ 300", "+ 1.0")
            .replace('"/logs/verifier"', repr(str(top))).replace('"/work/repo/', '"' + str(repo) + "/"))
    subprocess.run([sys.executable, "-I", "-c", code], env={**os.environ, "VERIFIER_LOGS": str(stage)}, check=True, timeout=30)
    time.sleep(2.0)  # the detached writer ends after 1 s
    for d in (top, stage):
        assert json.loads((d / "reward.json").read_text())["tampered"] == 1.0 and (d / "reward.txt").read_text() == "1.0"
    if stage_out == "invalid":
        assert not (stage / "junit.xml").exists() and json.loads((stage / "validation.json").read_text())["status"] == "error"
    else:
        assert (repo / "tamper_changed.txt").is_file() and (stage / "junit.xml").exists()
