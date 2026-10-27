# Model Upgrade: Evaluation-Driven Migration and Prompt Optimization

![Architecture: a notebook hill-climb orchestrator invokes agents on AgentCore Runtime (concurrent microVM sessions), which call a single AgentCore Gateway inference seam that routes by the model field to Amazon Bedrock (GPT-5.6 Luna source, GPT-6 Luna target) with OpenAI first-party and Azure OpenAI as example legs; session traces land in CloudWatch (AgentCore Observability), AgentCore Evaluations scores them, and accuracy plus judge reasoning feed back to the notebook for the next generation.](images/architecture.png)

When a newer model becomes available, two questions decide whether you should move your
workload to it:

1. **Does the upgrade hold or improve quality?** Running your existing prompt on the new
   model is not enough — it tells you nothing you can defend. You need to *score* the prompt
   on both models, on your own gold-standard data, and see the new model match or beat the
   old one on accuracy (and ideally on latency and cost too).
2. **Is the old prompt even the right prompt for the new model?** A prompt tuned for one
   model is rarely optimal for another. The real gain from an upgrade often comes from
   re-optimizing the prompt *for the model you are migrating to* — not from the raw model
   swap.

This module builds an evaluation-driven migration loop that answers both. It reuses the
travel-booking chatbot and gold-standard scenarios from the
[Chatbot](../Chatbot/) module as the workload under test, scores it the same way
(session-level goal completion plus binary per-criterion rubrics), and adds a
**notebook-driven hill-climbing optimizer** that generates candidate prompts, runs them in
parallel across [Amazon Bedrock AgentCore](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/what-is-bedrock-agentcore.html)
Runtime, and climbs toward the prompt that scores highest on the target model.

The architecture is deliberately AWS-native, open-source, and AgentCore-native so you can
flex between models, providers, and prompts without rewriting the agent.

---

## The Migration Example

The **default live path keeps both ends on Amazon Bedrock**, so anyone with only Bedrock
access can run the whole module end to end. We migrate an OpenAI-on-Bedrock model from one
version to the next:

| Role | Model | Where it runs | Reached via |
|---|---|---|---|
| **Source** (current production) | GPT-5.6 Luna (`us.openai.gpt-5.6-luna`) | Amazon Bedrock | Gateway inference target → `bedrock-runtime` connector |
| **Target** (upgrade candidate) | GPT-6 Luna (`us.openai.gpt-6-luna`) | Amazon Bedrock | Gateway inference target → `bedrock-runtime` connector |

Both are real, Bedrock-hosted, Converse-capable models. The module reads the IDs from
`model_config.py`, so bumping either end to a newer model is a one-line change.

**Migrating to or from a non-Bedrock model.** Many teams run their current model on **OpenAI
first-party** or **Azure OpenAI** and want to migrate to (or from) Bedrock. The single
Gateway seam makes that a config change, not an agent-code change — add the matching
inference target and point one end of the migration at it. The module ships both as
ready-to-use examples:

- `gateway/target-openai.json` + `model_config.EXAMPLE_SOURCE_OPENAI_1P` — OpenAI first-party
  (`https://api.openai.com`), key stored in an AgentCore API-key credential provider.
- `gateway/target-azure-openai.json` + `model_config.EXAMPLE_SOURCE_AZURE_OPENAI` — Azure
  OpenAI (your resource endpoint, deployment-name model id, `api-key` header).

The examples carry placeholder key/endpoint slots for a customer to fill in; the default
live path uses neither, so the module runs with Bedrock access alone.

---

## How It Works

### The single Gateway seam

Every agent instance points at **one AgentCore Gateway**. The Gateway is a managed entry
point for agentic traffic and supports three target categories — MCP, HTTP, and
**inference**. We use an **inference target** so the agent's model calls go through the
Gateway, and the Gateway selects the backend by the request's `model` field.

```
                      ┌─────────────────────────────┐
  AgentCore Runtime   │     AgentCore Gateway        │     ┌────────────────────────┐
  (one deployed agent)│   (single inference seam)    │────▶│ Bedrock (bedrock-runtime│  ← default:
  N concurrent        │                              │     │ connector): Luna models │    source+target
  sessions (microVMs) │  routes by the `model` field │     └────────────────────────┘
  ───────────────────▶│                              │     ┌────────────────────────┐
                      │                              │────▶│ OpenAI / Azure (example)│  ← optional
                      └─────────────────────────────┘     └────────────────────────┘
```

Because the agent only ever talks to the Gateway, switching the model under test — or
adding a provider leg — never touches agent code. That is what makes the harness reusable
across providers.

**The Bedrock leg uses the built-in `bedrock-runtime` connector**, which targets the native
Amazon Bedrock `InvokeModel` API. This is deliberate and deprecation-safe: the older
`bedrock-mantle` endpoint is being retired, and the `bedrock-runtime` connector is the
forward path. The connector auto-handles model discovery, model-id translation, and path
rewriting, and the gateway transparently translates an OpenAI Chat Completions request into
`InvokeModel` for OpenAI-family Bedrock models — so the agent speaks one dialect (OpenAI
Chat Completions) to one endpoint regardless of which model is under test.

### Verified live setup (and the gotchas it encodes)

`gateway/setup_gateway.py` provisions the whole seam via boto3 (`bedrock-agentcore-control`),
because the GA AgentCore CLI's `add gateway-target` covers MCP/HTTP/Runtime targets, not
inference targets. The script and `gateway/target-bedrock.json` encode four things learned
from a live end-to-end run (both Luna models returning `200` through the gateway):

1. **Use the `bedrock-runtime` connector, not a hand-built `bedrock-mantle` provider config.**
   The connector removes the manual path/model-mapping fiddliness and is deprecation-safe.
2. **The gateway IAM role needs `bedrock:InvokeModel` plus read-only `bedrock:List*`/`Get*`.**
   The connector enumerates models/profiles at target-activation time (it calls
   `ListFoundationModels`, `ListInferenceProfiles`, `ListProvisionedModelThroughputs`, …);
   without the `List*`/`Get*` grants the target goes `FAILED` at discovery.
3. **Model id = the target-qualified inference-profile id**, e.g.
   `bedrock/us.openai.gpt-6-luna` (the `us.` form the connector lists), not a bare base id.
4. **These reasoning models take `max_completion_tokens`, not `max_tokens`**, and want a
   generous budget (they spend reasoning tokens before the visible answer).

Provision and tear down:

```bash
python gateway/setup_gateway.py create         # Bedrock-only (no keys needed)
python gateway/setup_gateway.py create --with-openai   # also add the OpenAI example target
python gateway/setup_gateway.py status
python gateway/setup_gateway.py teardown        # removes targets, gateway, role, cred provider
```

### Live end-to-end smoke test

`scripts/e2e_smoke_test.py` provisions a throwaway AgentCore project and exercises the
**Runtime + Observability + Evaluations** path end to end against real resources, then tears
it down (`--keep` leaves it up). Running it surfaced two more findings baked into the module:

- **Pick the judge model carefully.** A `claude-sonnet-5` LLM-as-a-judge evaluator fails to
  create because that model rejects the default `temperature` parameter
  (`temperature is deprecated for this model`). The module's judge is **Nova Micro**
  (`amazon.nova-micro-v1:0`), which creates and scores cleanly and is cheap.
- **`eval ondemand` selects sessions from ingested Observability traces**, and trace
  ingestion lags the agent turn by a couple of minutes — budget an ingestion wait before
  grading, and confirm `sessionsEvaluated > 0` rather than trusting an empty `results: []`.

What the smoke test proves live today: Runtime deploys to `READY`, a live `agentcore invoke`
returns a correct multi-turn travel-agent response, and a custom AgentCore evaluator is
created and deployed.

### Iterate evaluators locally — no deploy

You do **not** need a CloudFormation deploy to develop or debug an evaluator. The slow,
flaky part of a live run is span *ingestion*: `eval ondemand` collects spans from the
CloudWatch log group `/aws/bedrock-agentcore/runtimes/{agent_id}-DEFAULT` over a lookback
window, and ingestion lags the agent turn by a couple of minutes — which is why a grade run
fired too early returns `sessionsEvaluated: 0` (an empty `results: []`, not an evaluator
error). Budget an ingestion wait and confirm `sessionsEvaluated > 0`.

The evaluator itself is just a Python function: `EvaluatorInput.session_spans` is a
`List[Dict]`, and `custom_code_based_evaluator` wraps `func(EvaluatorInput, context) ->
EvaluatorOutput`. So you can grade a session in milliseconds with no AWS by constructing (or
loading) spans and calling the handler directly:

```bash
python scripts/local_eval_demo.py   # scores sessions in-memory; no deploy, no CloudWatch
```

Use the local demo to iterate on evaluator logic fast; run the full `e2e_smoke_test.py` only
when you want to confirm the end-to-end Runtime → Observability → Evaluations plumbing.

### Fan-out is sessions, not instances

AgentCore Runtime scales **per session**: each session runs in its own isolated microVM
with dedicated compute, memory, and filesystem. You **deploy the agent once** and achieve
parallel fan-out by invoking that one runtime with **many concurrent session IDs** — the
platform spins the microVMs up and down for you.

So the mental model of "spin up 30 runtime instances" maps precisely to **30 concurrent
`invoke_agent_runtime` calls with 30 distinct session IDs against one deployed runtime**.
In this module:

```
3 candidate prompts × 10 concurrent sessions each = 30 sessions
each session runs a slice of the gold-standard scenarios
→ all traces land in AgentCore Observability
→ AgentCore custom evaluators score the sessions
→ the notebook collects scores + judge reasoning and decides the next generation
```

### The hill-climbing loop (owned by the notebook)

```
 1. Seed      : start from the current production prompt (scored on both models as the baseline)
 2. Mutate    : an LLM proposes K candidate prompts, informed by the judge's reasoning on
                where the current best prompt loses points (true hill climbing, not random)
 3. Fan out   : run all K candidates in parallel across AgentCore Runtime sessions, over a
                slice of the gold scenarios, through the Gateway to the TARGET model
 4. Score     : AgentCore custom evaluators return goal-completion + per-criterion pass/fail
                + reasoning; the notebook also records latency and cost per candidate
 5. Select    : keep the highest-scoring candidate as the new incumbent
 6. Check     : if accuracy gain over the previous generation is below a threshold → stop
 7. Report    : "we ran N generations; here is the prompt that scores highest on the target
                model, at this accuracy / latency / cost vs the source baseline"
```

The orchestration stays in the notebook on purpose — it keeps the control loop inspectable
and lets you watch *why* each candidate moved the score, rather than handing the decision to
a managed job.

---

## Scoring

Reused verbatim from the [Chatbot](../Chatbot/) module so the number means the same thing
here as there — this module is about the migration/optimization loop, not a new metric:

| Layer | Evaluator | What it yields |
|---|---|---|
| **Session-level** (headline accuracy) | `GoalSuccessRateEvaluator` (assertion mode) | Did the conversation satisfy its explicit success assertions? → the goal-completion rate that drives hill climbing |
| **Turn / output-level** (the reasoning signal) | `OutputEvaluator` with binary rubrics | Per-criterion pass/fail (e.g. "confirmed passenger name before booking", "never invented a flight"), each with judge reasoning |

Alongside accuracy, every candidate reports **latency** (per-turn and per-session) and
**cost** (token spend × model price), so the final recommendation can show the upgrade is
the same or better on all three axes — not just accuracy. Because the task is multi-turn,
latency and cost are measured across the whole conversation, not a single call.

---

## Why Not Bedrock Advanced Prompt Optimization?

The [Capacity Management](../Capacity%20Management/) module optimizes prompts with Bedrock's
managed **Advanced Prompt Optimization (AdvPO)** job. This module builds its own
hill-climbing loop instead, for reasons that are specific to a cross-provider migration:

- **AdvPO optimizes for Bedrock-hosted models only.** A migration whose *source* is a
  first-party third-party model (GPT-4.1 on OpenAI) cannot be driven end-to-end by a
  Bedrock-only optimizer.
- **We want the judge's reasoning in the loop.** Hill climbing here is *informed* mutation —
  the optimizer reads why the current best prompt lost points and targets those failures.
- **We want a multi-turn, agentic task.** The optimization target is a simulated multi-turn
  conversation scored by AgentCore evaluators, not a single-turn input/output pair.

AdvPO remains the right tool when your models are all on Bedrock and your task is
single-turn — see the Capacity Management module for that path. The two modules are
complementary, not redundant.

---

## File Structure

```
Model Upgrade/
├── README.md                              # This file
├── NN Model Upgrade Evaluation.ipynb      # The module (lead notebook)
├── requirements.txt
├── model_config.py                        # source/target model IDs, pricing, region, fan-out knobs
├── hillclimb.py                           # candidate generation, selection, stop rule, results objects
├── gateway/                               # AgentCore Gateway (inference targets: Bedrock + OpenAI)
│   └── ...                                # CDK / CLI config for the single inference seam
├── agent/                                 # the travel-booking agent packaged for AgentCore Runtime
│   └── ...                                # reuses the Chatbot workload; model call routed via Gateway
├── evaluators/                            # AgentCore custom evaluators (goal completion + binary rubrics)
├── data/
│   ├── gold_scenarios.jsonl               # gold-standard scenarios (from the Chatbot module)
│   └── committed_run/                     # full hill-climbing run results, committed for fast reading
└── scripts/
    └── run_optimization.py                # end-to-end driver for a full (turned-up) run
```

> Final file names/numbers are set when the notebook lands. `NN` is the next module number
> in the Workload Specific Evaluations README.

---

## Runnable by Default, Scalable by Design

Following the repo convention (Capacity Management commits real optimizer results; AgentCore
commits fixtures), this module ships a **small runnable default**:

- a few candidate prompts per generation, a small slice of gold scenarios, a few generations
- committed results from a full run under `data/committed_run/`, so the notebook renders the
  complete hill-climbing story without forcing an expensive run

Every knob that controls scale (candidates per generation, sessions, scenario slice size,
generation count) lives in `model_config.py`. Turning the module onto your **own** data and
a full-size fan-out is meant to be a config change, not a rewrite — that is the intended way
to use it in anger.

---

## Prerequisites

- AWS account with Amazon Bedrock access and the target model enabled (defaults to
  `us-east-1`)
- AgentCore CLI and CDK bootstrap (see the [AgentCore](../../Framework%20Specific%20Evaluations/AgentCore/)
  module for the install + bootstrap steps the Gateway/Runtime scaffolding reuses)
- An OpenAI API key for the source model (GPT-4.1), stored as a secret the Gateway's OpenAI
  inference target reads — never hard-code it in the notebook
- Python 3.10+
- Completion of the [Foundational Evaluations](../../Foundational%20Evaluations/) and the
  [Chatbot](../Chatbot/) module (this module assumes its workload and scoring)

## Cost and Cleanup

Each hill-climbing generation invokes the agent (many sessions × multi-turn conversations)
and the judge (per session). A full turned-up run is real inference spend on both the source
and target providers; the committed results exist so you can learn the loop without paying
for it. Run the Gateway/Runtime cleanup helper after the module to remove deployed resources.

> Cost note: even a "small" local optimization run accumulates real token spend at volume on
> both providers. Treat the committed run as the default and turn fan-out up deliberately.

## Official References

- [What is Amazon Bedrock AgentCore](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/what-is-bedrock-agentcore.html)
- [AgentCore Gateway — core concepts (MCP, HTTP, inference targets)](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-core-concepts.html)
- [AgentCore Runtime — sessions and microVM isolation](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-sessions.html)
- [AgentCore Evaluations](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/evaluations.html)
- [Claude Sonnet 5.5 model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-sonnet-5-5.html)
