"""score.py of the PR review tasks (notebook 02), with a judge that needs no model."""

import json
import re
import sys
import types

import pytest

from conftest import PR_TASKS

REJECT = next(t for t in PR_TASKS if t.name.endswith("reject"))
APPROVE = next(t for t in PR_TASKS if t.name.endswith("approve"))


def oracle_submission(task):
    """The findings that solution/solve.sh submits (the oracle agent)."""
    line = next(l for l in (task / "solution" / "solve.sh").read_text().splitlines() if l.startswith("submission = "))
    return json.loads(line.removeprefix("submission = "))


def agent(score, entries):
    return [score._agent_finding(i, e) for i, e in enumerate(entries)]


def accept_all(_oracle, _agent):
    return True


def test_precision_is_undefined_with_no_findings(score):
    for task in (REJECT, APPROVE):
        oracle = score.load_oracle(task / "tests" / "oracle.json")
        reward = score.Score.compute_reward(oracle, [], accept_all)
        assert reward["precision_strict"] is None
        assert reward["precision_adjudicated"] is None
        assert reward["uncredited"] == 0.0
        assert set(reward) == set(score.REWARD_KEYS)


def test_approve_oracle_coverage_is_perfect(score):
    oracle = score.load_oracle(APPROVE / "tests" / "oracle.json")
    reward = score.Score.compute_reward(oracle, agent(score, oracle_submission(APPROVE)), accept_all)
    assert reward["coverage_all"] == 1.0 and reward["coverage_required"] == 1.0
    assert reward["precision_adjudicated"] is None


def test_reject_oracle_gets_full_credit(score):
    oracle = score.load_oracle(REJECT / "tests" / "oracle.json")
    reward = score.Score.compute_reward(oracle, agent(score, oracle_submission(REJECT)), accept_all)
    assert reward["coverage_all"] == 1.0
    assert reward["coverage_required"] == 1.0
    assert reward["precision_adjudicated"] == 1.0
    assert reward["precision_strict"] == 1.0
    assert reward["uncredited"] == 0.0


def test_silent_agent_fails_reject_on_coverage(score):
    oracle = score.load_oracle(REJECT / "tests" / "oracle.json")
    reward = score.Score.compute_reward(oracle, [], accept_all)
    assert reward["coverage_required"] == 0.0 and reward["coverage_all"] == 0.0


def test_main_leaves_undefined_keys_out_of_reward_json(score, tmp_path, monkeypatch):
    findings = tmp_path / "findings.json"
    findings.write_text("[]")
    recording = tmp_path / "recording.json"
    recording.write_text('{"decisions": []}')
    monkeypatch.setattr(score, "_ORACLE_PATH", APPROVE / "tests" / "oracle.json")
    monkeypatch.setattr(score, "_REWARD_PATH", tmp_path / "reward.json")
    monkeypatch.setattr(score, "_REWARD_DETAILS_PATH", tmp_path / "reward_details.json")
    monkeypatch.setattr(score, "_JUDGE_RECORD_PATH", tmp_path / "judge.json")
    monkeypatch.setenv("EVAL_HARVEST_AGENT_FINDINGS", str(findings))
    monkeypatch.setenv("EVAL_HARVEST_JUDGE_RECORDING", str(recording))
    score.main()
    reward = json.loads((tmp_path / "reward.json").read_text())
    details = json.loads((tmp_path / "reward_details.json").read_text())
    # Harbor accepts only numbers in reward.json.
    assert all(isinstance(v, (int, float)) for v in reward.values())
    assert "precision_adjudicated" not in reward and "precision_strict" not in reward
    assert details["precision_adjudicated"] is None and details["precision_strict"] is None
    assert set(details) == set(score.REWARD_KEYS)


def _live_judge(score, monkeypatch):
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.delenv("JUDGE_MODEL_ID", raising=False)
    return score.LiveJudge.from_config(REJECT / "tests" / "judge.toml")


def test_judge_record_with_no_call(score, monkeypatch):
    judge = _live_judge(score, monkeypatch)
    record = score.judge_record(judge)
    assert record == {"model": judge.model, "temperature_configured": 0.0, "temperature": None, "calls": 0,
                      "votes": 3, "exact_matches": 0, "decisions": []}


@pytest.mark.parametrize("rejects", [True, False])
def test_judge_temperature_fallback(score, monkeypatch, rejects):
    """A model that rejects the temperature gets the call again without it, and judge.json says null."""
    from botocore.exceptions import ClientError

    sent = []

    class FakeClient:
        def converse(self, modelId, messages, inferenceConfig):
            sent.append(dict(inferenceConfig))
            if rejects and "temperature" in inferenceConfig:
                raise ClientError({"Error": {"Code": "ValidationException",
                                             "Message": "temperature is not supported"}}, "Converse")
            return {"output": {"message": {"content": [{"text": "yes"}]}}}

    monkeypatch.setitem(sys.modules, "boto3", types.SimpleNamespace(client=lambda *a, **k: FakeClient()))
    judge = _live_judge(score, monkeypatch)
    oracle = score.load_oracle(REJECT / "tests" / "oracle.json")
    reworded = [{**e, "statement": e["statement"] + " (reworded)"} for e in oracle_submission(REJECT)]
    reward = score.Score.compute_reward(oracle, agent(score, reworded), judge)
    assert reward["coverage_all"] == 1.0
    record = score.judge_record(judge)
    assert record["calls"] == len(oracle) * judge.votes
    if rejects:
        assert record["temperature"] is None
        assert "temperature" in sent[0] and all("temperature" not in c for c in sent[1:])
    else:
        assert record["temperature"] == 0.0
        assert all(c["temperature"] == 0.0 for c in sent)


def test_test_sh_fallback_reward_is_numeric():
    """If score.py cannot run, test.sh writes a zero reward. Harbor must be able to parse it."""
    text = (REJECT / "tests" / "test.sh").read_text()
    payload = re.search(r"printf '(\{.*\})'", text).group(1)
    assert all(isinstance(v, (int, float)) for v in json.loads(payload).values())


def _fake_boto3(monkeypatch, answers):
    """A boto3 stand-in whose model gives the answers in order, then repeats the last one."""
    calls = []

    class FakeClient:
        def converse(self, modelId, messages, inferenceConfig):
            calls.append(messages)
            text = answers[min(len(calls), len(answers)) - 1]
            return {"output": {"message": {"content": [{"text": text}]}}}

    monkeypatch.setitem(sys.modules, "boto3", types.SimpleNamespace(client=lambda *a, **k: FakeClient()))
    return calls


def test_oracle_is_matched_without_model_calls(score, monkeypatch):
    """The oracle submits the reference statements, so the result cannot depend on the model."""
    calls = _fake_boto3(monkeypatch, ["no"])  # a model that always says no
    judge = _live_judge(score, monkeypatch)
    oracle = score.load_oracle(REJECT / "tests" / "oracle.json")
    reward = score.Score.compute_reward(oracle, agent(score, oracle_submission(REJECT)), judge)
    assert reward["coverage_all"] == 1.0 and reward["precision_adjudicated"] == 1.0
    assert calls == [] and score.judge_record(judge)["exact_matches"] >= len(oracle)


def test_exact_match_ignores_case_and_whitespace(score, monkeypatch):
    calls = _fake_boto3(monkeypatch, ["no"])
    judge = _live_judge(score, monkeypatch)
    o = score.load_oracle(REJECT / "tests" / "oracle.json")[0]
    a = score.AgentFinding(index=0, path="x", line=1, statement="  " + o.statement.upper().replace(" ", "\n "), severity="high")
    assert judge(o, a) is True and calls == []


@pytest.mark.parametrize("answers, accepted", [(["yes", "no", "yes"], True), (["no", "yes", "no"], False)])
def test_majority_vote(score, monkeypatch, answers, accepted):
    """One flipped answer out of three does not change the decision, and judge.json records the split."""
    calls = _fake_boto3(monkeypatch, answers)
    judge = _live_judge(score, monkeypatch)
    o = score.load_oracle(REJECT / "tests" / "oracle.json")[0]
    a = score.AgentFinding(index=0, path="x", line=1, statement="a different wording", severity="high")
    assert judge(o, a) is accepted
    assert len(calls) == 3
    decision = score.judge_record(judge)["decisions"][0]
    assert decision["answers"] == [x == "yes" for x in answers] and decision["accepted"] is accepted
