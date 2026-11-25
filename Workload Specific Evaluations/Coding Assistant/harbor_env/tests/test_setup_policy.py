"""The agent role allows only the models named with --agent-model."""
import pytest

from harbor_env import setup

PROFILE = "arn:aws:bedrock:us-east-1:111122223333:inference-profile/us.amazon.nova-2-lite-v1:0"
MODEL = "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-2-lite-v1:0"


@pytest.fixture(autouse=True)
def no_aws(monkeypatch):
    def arns(model):
        assert not model.startswith("openai."), "an OpenAI model is not an inference profile"
        return [PROFILE, MODEL]
    monkeypatch.setattr(setup, "model_arns", arns)


def test_openai_model_gets_only_mantle():
    (st,) = setup.policy_agent_models(["openai.gpt-6-sol"])["Statement"]
    assert st["Action"] == "bedrock-mantle:CreateInference"
    assert st["Condition"] == {"StringEquals": {"bedrock-mantle:Model": ["openai.gpt-6-sol"]}}


def test_converse_model_gets_only_invoke_on_its_arns():
    (st,) = setup.policy_agent_models(["us.amazon.nova-2-lite-v1:0"])["Statement"]
    assert st["Action"] == ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"] and st["Resource"] == [PROFILE, MODEL]


def test_both_kinds_and_no_codex_default():
    sts = setup.policy_agent_models(["us.amazon.nova-2-lite-v1:0", "openai.gpt-6-sol"])["Statement"]
    assert sorted(s["Action"] if isinstance(s["Action"], str) else s["Action"][0] for s in sts) == \
        ["bedrock-mantle:CreateInference", "bedrock:InvokeModel"]
    assert all(s["Condition"]["StringEquals"]["bedrock-mantle:Model"] == ["openai.gpt-6-sol"] for s in sts if "Condition" in s)


def test_no_model_is_refused():
    with pytest.raises(SystemExit, match="--agent-model"):
        setup.policy_agent_models([])


def test_teardown_matches_only_its_own_prefix(monkeypatch):
    monkeypatch.setattr(setup.aws.Names, "prefix", "evals-hb-")
    assert setup.own_runtime("evals_hb_evalspr371fix_a446eed6c6a")
    assert setup.own_runtime("evals_hb_mcppr4121approve_vd657b743f7")
    assert not setup.own_runtime("evals_hb_2_evalspr371fix_a446eed6c6a") and not setup.own_runtime("evals_hb_x")
    assert setup.own_log_group("/aws/codebuild/evals-hb-build")
    assert setup.own_log_group("/aws/bedrock-agentcore/runtimes/evals_hb_evalspr371fix_a446eed6c6a-AbC123xYz9-DEFAULT")
    assert not setup.own_log_group("/aws/codebuild/evals-hb-build2")
    assert not setup.own_log_group("/aws/bedrock-agentcore/runtimes/evals_hb_2_evalspr371fix_a446eed6c6a-AbC123xYz9-DEFAULT")
