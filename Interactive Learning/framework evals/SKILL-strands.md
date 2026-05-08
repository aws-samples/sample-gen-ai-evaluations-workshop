---
name: Agent Evaluation with Strands Evals
description: Evaluate agent performance using the Strands Agents Evals SDK. Activate when asked to "evaluate an agent with strands", "set up strands evals", "create trajectory evaluators", "build custom agent evaluators", or "compare agent tool usage patterns".
---

# Agent Evaluation with Strands Evals SDK

Evaluate agents, not just models. This module uses the Strands Agents Evals SDK to assess agent behavior across output quality, tool-use trajectories, and custom domain criteria. This is an AGENT-FOCUSED evaluation framework — distinct from Module 03 which uses Strands to *build* agents. Here you use Strands Evals to *evaluate* them.

## Prerequisites

- Python 3.10+
- `strands-agents`, `strands-agents-tools`, `strands-agents-evals` installed
- Amazon Bedrock access (Claude Haiku or similar)
- Familiarity with Strands Agent creation (Module 03)

## Learning Objectives

1. **Define** test cases using the Strands `Case` class with inputs, expected outputs, and expected trajectories
2. **Configure** an `OutputEvaluator` with a custom rubric to judge response quality
3. **Capture** agent tool-use trajectories using the `tools_use_extractor`
4. **Assess** tool-use patterns with a `TrajectoryEvaluator` and scoring rubric
5. **Build** a custom `Evaluator` subclass for domain-specific checks
6. **Combine** multiple evaluators in a single `Experiment` for multi-dimensional assessment
7. **Run** experiments and interpret pass/fail results with score reasoning
8. **Persist** experiment results for regression tracking over time

## Setup

```python
from strands import Agent, tool
from strands.models import BedrockModel
from strands_evals import Case, Experiment
from strands_evals.evaluators import OutputEvaluator, TrajectoryEvaluator
from strands_evals.extractors import tools_use_extractor
from botocore.config import Config

model = BedrockModel(
    model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0",
    boto_client_config=Config(connect_timeout=5, read_timeout=30, retries={"max_attempts": 1})
)
```

## Section 1: Cases and Output Evaluation

**Concept:** A `Case` is the atomic unit of evaluation — it bundles an input prompt, expected output, optional expected trajectory, and metadata. Cases are typed generics (`Case[InputType, OutputType]`) so evaluators know what they're comparing. Without structured cases, evaluation devolves into ad-hoc spot checks. The `OutputEvaluator` uses an LLM judge to score agent responses against a rubric you define — the rubric is your evaluation contract that tells the judge exactly what "good" means.

**Build:**

```python
from strands_evals import Case

test_cases = [
    Case[str, str](
        name="city-population-seattle",
        input="What is the population of Seattle, WA?",
        expected_output="Population: 780995",
        metadata={"category": "city_info", "city": "Seattle"}
    ),
    Case[str, str](
        name="city-area-phoenix",
        input="What is the area of Phoenix, AZ in square miles?",
        expected_output="Area: 518.0 square miles",
        expected_trajectory=["web_search"],
        metadata={"category": "city_info", "city": "Phoenix"}
    ),
]

accuracy_evaluator = OutputEvaluator(
    rubric="""
    Evaluate the response based on:
    1. Population/Area Accuracy - Is the value within 10% of expected?
    2. Completeness - Does the response include the requested information?
    
    Score 1.0 if values are acceptable.
    Score 0.5 if partially correct.
    Score 0.0 if wrong or missing.
    """,
    include_inputs=True
)
```

## Section 2: Task Functions and Running Experiments

**Concept:** A task function bridges your agent and the evaluation harness. It receives a `Case`, runs the agent, and returns the output in the format evaluators expect. Creating a fresh agent per case prevents context contamination between test runs. An `Experiment` groups cases with evaluators and orchestrates the evaluation run — it handles iteration, error collection, and report generation. The result is a structured report with per-case scores, pass/fail status, and reasoning.

**Build:**

```python
def get_agent_response(case: Case) -> str:
    """Run agent on a test case, return response string."""
    agent = Agent(
        system_prompt="You are a helpful assistant. Use tools to find current data.",
        tools=[web_search, get_page],
        model=model,
        callback_handler=None  # Suppress output during eval
    )
    response = agent(case.input)
    return str(response)

experiment = Experiment[str, str](
    cases=test_cases,
    evaluators=[accuracy_evaluator]
)

reports = experiment.run_evaluations(get_agent_response)
reports[0].run_display()  # Visual report with scores and reasoning
```

## Section 3: Trajectory Evaluation — Judging Tool Usage

**Concept:** Output correctness doesn't guarantee good agent behavior. An agent might get the right answer through wasteful tool calls or skip tools it should use. The `TrajectoryEvaluator` scores HOW the agent solved the problem — which tools were called, in what order, and whether that sequence was efficient. Use `tools_use_extractor` to capture the actual trajectory from agent messages.

**Build:**

```python
trajectory_evaluator = TrajectoryEvaluator(
    rubric="""
    Evaluate tool usage:
    1. Were expected tools used?
    2. Were tools called in a logical order?
    3. Were unnecessary tool calls avoided?
    
    Score 1.0 if all expected tools used appropriately.
    Score 0.5 if task completed but tools misused.
    Score 0.0 if critical tools missing.
    """,
    include_inputs=True
)

# Register tool descriptions so the judge understands available tools
sample_agent = Agent(tools=[web_search, get_page], callback_handler=None)
tool_descriptions = tools_use_extractor.extract_tools_description(sample_agent, is_short=True)
trajectory_evaluator.update_trajectory_description(tool_descriptions)

def get_response_with_trajectory(case: Case) -> dict:
    """Capture both output and tool-use trajectory."""
    agent = Agent(
        system_prompt="You are a helpful assistant.",
        tools=[web_search, get_page],
        model=model,
        callback_handler=None
    )
    response = agent(case.input)
    trajectory = tools_use_extractor.extract_agent_tools_used_from_messages(agent.messages)
    return {"output": str(response), "trajectory": trajectory}
```

## Section 4: Custom Evaluators

**Concept:** When rubric-based LLM judging isn't precise enough, subclass `Evaluator` to implement deterministic checks. Custom evaluators return a list of `EvaluationOutput` with score, pass/fail, reason, and label. Use these for format validation, regex checks, numeric thresholds — anything where you want exact, reproducible scoring without LLM variance.

**Build:**

```python
import re
from strands_evals.evaluators import Evaluator
from strands_evals.types import EvaluationData, EvaluationOutput

class XMLFormatEvaluator(Evaluator[str, str]):
    """Check that agent output contains required XML tags."""
    
    def evaluate(self, evaluation_case: EvaluationData[str, str]) -> list[EvaluationOutput]:
        response = evaluation_case.actual_output
        has_response = bool(re.search(r'<response>.*?</response>', response, re.DOTALL))
        has_pop = bool(re.search(r'<pop>\d+</pop>', response))
        has_area = bool(re.search(r'<area>[\d.]+</area>', response))
        
        score = sum([has_response, has_pop, has_area]) / 3.0
        missing = [t for t, p in [("<response>", has_response), ("<pop>", has_pop), ("<area>", has_area)] if not p]
        
        return [EvaluationOutput(
            score=score,
            test_pass=score >= 0.66,
            reason=f"Missing: {', '.join(missing)}" if missing else "All XML tags present",
            label="complete" if score == 1.0 else "incomplete"
        )]
```

## Section 5: Multi-Evaluator Experiments and Persistence

**Concept:** Real agent evaluation is multi-dimensional — accuracy, tool usage, and format compliance are independent axes. Combining evaluators in one experiment gives a holistic view without running the agent multiple times. Each evaluator produces its own report, so you can identify exactly which dimension failed. Saving experiments to files lets you track performance over time, detect regressions after prompt changes, and compare agent versions.

**Build:**

```python
comprehensive_cases = [
    Case[str, str](
        name="comprehensive-seattle",
        input="What is the population and area of Seattle, WA?",
        expected_output="Population: 780995, Area: 83.8 square miles",
        expected_trajectory=["web_search"],
        metadata={"category": "comprehensive"}
    )
]

experiment = Experiment[str, str](
    cases=comprehensive_cases,
    evaluators=[
        accuracy_evaluator,       # Output quality
        trajectory_evaluator,     # Tool usage patterns
        XMLFormatEvaluator()      # Format compliance
    ]
)

reports = experiment.run_evaluations(get_response_with_trajectory)
for i, report in enumerate(reports):
    print(f"\nEvaluator {i+1}:")
    report.run_display()

# Save experiment results for regression tracking
experiment.to_file("agent_evaluation_results.json")

# Load for comparison after agent changes
# previous = Experiment.from_file("agent_evaluation_results.json")
# Compare scores between runs to detect regressions
```

## Challenges

See **CHALLENGE-deep-dive.md** for the Module 05 deep-dive challenge that extends across all framework evaluations in this module.

**Assessment criteria:**
1. Runs experiment end-to-end without errors
2. Defines at least 3 test cases with expected outputs and trajectories
3. Implements one custom evaluator with deterministic scoring logic
4. Combines ≥2 evaluators in a single experiment
5. Learner explains when to use OutputEvaluator vs TrajectoryEvaluator vs custom

## Wrap-Up

**Key Takeaways:**
- Strands Evals separates WHAT to evaluate (Cases) from HOW to evaluate (Evaluators) from orchestration (Experiments)
- OutputEvaluator judges response quality; TrajectoryEvaluator judges tool-use behavior — use both
- Custom evaluators give deterministic, reproducible scoring for format/structural checks
- Persist experiments to detect regressions across agent iterations

**This module does NOT cover:**
- Building agents with Strands (→ Module 03: Agentic Metrics)
- Prompt-level evaluation without agents (→ Module 05: PromptFoo)
- Guardrails and safety evaluation (→ Module 04)
- Async evaluation with `run_evaluations_async()` for large test suites
- `HelpfulnessEvaluator` and `GoalSuccessRateEvaluator` (advanced Strands Evals features)

**Next Steps:**
- Compare Strands Evals results with PromptFoo results on the same agent for framework tradeoff analysis
- Scale test suites using async evaluation for CI/CD integration
