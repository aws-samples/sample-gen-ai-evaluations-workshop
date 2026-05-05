---
name: "Strands Agents Evals SDK"
description: "Evaluate AI agents using the Strands Evals SDK — build test cases, run output and trajectory evaluations, create custom evaluators, and combine multiple evaluators for comprehensive agent assessment"
---

# Strands Agents Evaluation SDK

In this lesson you build a complete agent evaluation harness using the Strands Agents Evals SDK. This is an **agent-focused** evaluation framework — unlike Module 03 where you used Strands to *build* agents, here you use the Strands Evals SDK to *evaluate* agent behavior systematically. You will define test cases, judge output quality with LLM-as-judge rubrics, assess tool-usage trajectories, and create custom evaluators for domain-specific checks. By the end, you will run multi-evaluator experiments that score agents across accuracy, efficiency, and format compliance.

## Prerequisites

- Completed Module 03 (Agentic Metrics) — familiarity with the web search agent
- AWS account with Amazon Bedrock access (Claude Haiku enabled)
- Python 3.10+ environment with pip
- Basic understanding of LLM-as-judge evaluation concepts (Module 02)

## Learning Objectives

- Construct structured test cases using the Strands Evals `Case` class with inputs, expected outputs, and expected trajectories
- Configure and run `OutputEvaluator` with custom rubrics to score agent response quality
- Configure and run `TrajectoryEvaluator` to assess whether agents select appropriate tools in logical sequences
- Implement a custom `Evaluator` subclass for domain-specific evaluation logic
- Combine multiple evaluators in a single `Experiment` to produce multi-dimensional agent assessments

## Setup

Install the required packages:

```bash
pip install strands-agents strands-agents-tools strands-agents-evals ddgs pandas typing-extensions --quiet
```

Create a file `city_pop.csv` with sample data (or reuse from Module 03):

```csv
city,state,population,land_area_mi2
New York,NY,"8,478,072",300.5
Los Angeles,CA,"3,878,704",469.5
Chicago,IL,"2,721,308",227.7
```

Set up imports and the Bedrock model:

```python
from strands import Agent, tool
from strands.models import BedrockModel
from strands_evals import Case, Experiment
from strands_evals.evaluators import OutputEvaluator, TrajectoryEvaluator
from strands_evals.extractors import tools_use_extractor
from botocore.config import Config
from ddgs import DDGS
import requests
from bs4 import BeautifulSoup
import pandas as pd

quick_config = Config(
    connect_timeout=5,
    read_timeout=30,
    retries={"max_attempts": 1}
)

model_name = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
chatbot_model = BedrockModel(
    model_id=model_name,
    boto_client_config=quick_config
)
```

Define the agent tools and system prompt (reused from Module 03):

```python
@tool
def web_search(topic: str) -> str:
    """Search DuckDuckGo for a given topic."""
    try:
        results = DDGS(timeout=5).text(topic, max_results=5)
        if not results:
            return "No search results found"
        result_string = ""
        for i, result in enumerate(results):
            result_string += f"Result {i+1}: {result.get('title', 'No title')}\nURL: {result.get('href', 'No URL')}\nSnippet: {result.get('body', 'No description')}\n\n"
        return result_string
    except Exception as e:
        return f"Search error: {str(e)}"

@tool
def get_page(url: str) -> str:
    """Fetch and return the raw text from a URL."""
    response = requests.get(url)
    response.raise_for_status()
    bs = BeautifulSoup(response.text, 'html.parser')
    return bs.text

system_prompt = '''You are a helpful tour guide. Customers may ask you about the population and size of cities.
You should use tools to retrieve all data, to make sure it is as current as possible.
Please put your human friendly response in 'response' XML tags, and then follow with your data results in 'pop' and 'area' XML tags, for programatic processing.
The values in the 'pop' and 'area' XML tags should only be numbers, no words or commas.'''
```

### Section 1: Structuring Test Cases with the Case Class

When evaluating agents, you need repeatable, structured inputs paired with known-good expected outputs. Without this structure, evaluation becomes ad-hoc — you run the agent, eyeball the result, and move on. That approach doesn't scale and doesn't catch regressions.

The Strands Evals `Case` class solves this by bundling the input prompt, expected output, optional expected tool trajectory, and metadata into a single object. This lets you build reusable test suites that can be run automatically across agent versions. The metadata field enables filtering and grouping results by category, difficulty, or any dimension you care about.

**Build:** Create test cases from the gold standard dataset.

```python
gold_standard_city_pop = pd.read_csv('city_pop.csv')
gold_standard_city_pop['city'] = gold_standard_city_pop['city'].str.replace(r'\[.*?\]', '', regex=True)
gold_standard_city_pop['population'] = gold_standard_city_pop['population'].astype(str).str.replace(',', '').astype(float)
gold_standard_city_pop['land_area_mi2'] = gold_standard_city_pop['land_area_mi2'].astype(str).str.replace(',', '').astype(float)

test_cases = []
for idx, row in gold_standard_city_pop.head(3).iterrows():
    test_cases.append(
        Case[str, str](
            name=f"city-{idx+1}-{row['city'].lower().replace(' ', '-')}",
            input=f"What is the population and area in square miles of {row['city']}, {row['state']}?",
            expected_output=f"Population: {int(row['population'])}, Area: {row['land_area_mi2']} square miles",
            metadata={"category": "city_info", "city": row['city'], "state": row['state']}
        )
    )

for case in test_cases:
    print(f"Test: {case.name}")
    print(f"  Input: {case.input}")
    print(f"  Expected: {case.expected_output}\n")
```

### Section 2: Output Evaluation with LLM-as-Judge Rubrics

Checking whether an agent's free-form text response is "correct" is fundamentally harder than comparing two numbers. The agent might phrase things differently, include extra context, or use different units. Simple string matching fails here.

The `OutputEvaluator` addresses this by using a powerful LLM as a judge. You provide a rubric — a set of criteria and scoring rules — and the judge model assesses the agent's actual output against the expected output. This gives you nuanced scoring (not just pass/fail) with reasoning you can inspect. The `include_inputs=True` flag gives the judge access to the original question for context-aware evaluation.

**Build:** Create an output evaluator with a custom accuracy rubric and run an experiment.

```python
accuracy_evaluator = OutputEvaluator(
    rubric="""
    Evaluate the response based on:
    1. Population Accuracy - Is the population value close to the expected value? (within 10% is acceptable)
    2. Area Accuracy - Is the area value close to the expected value? (within 10% is acceptable)
    3. Completeness - Does the response include both population AND area information?
    
    Score 1.0 if both values are acceptable.
    Score 0.5 if only one value is acceptable.
    Score 0.0 if both values are wrong or either missing.
    """,
    include_inputs=True
)

def get_agent_response(case: Case) -> str:
    """Run the agent on a test case and return the response."""
    agent = Agent(
        system_prompt=system_prompt,
        tools=[web_search, get_page],
        model=chatbot_model,
        callback_handler=None
    )
    response = agent(case.input)
    return str(response)

experiment = Experiment[str, str](
    cases=test_cases,
    evaluators=[accuracy_evaluator]
)

reports = experiment.run_evaluations(get_agent_response)
reports[0].run_display()
```

### Section 3: Trajectory Evaluation — Assessing Tool Usage

Output quality tells you *what* the agent produced, but not *how* it got there. An agent might return the right answer by luck, or it might use tools inefficiently — making five searches when one would suffice. Trajectory evaluation closes this gap.

The `TrajectoryEvaluator` examines the sequence of tools the agent invoked and judges whether that sequence was appropriate. You specify expected trajectories in your test cases, and the evaluator uses an LLM judge to assess tool selection, ordering, and efficiency. The `tools_use_extractor` utility pulls the actual tool-call sequence from the agent's message history, making capture automatic.

**Build:** Define trajectory test cases, capture tool usage, and evaluate.

```python
trajectory_test_cases = [
    Case[str, str](
        name="web-search-required",
        input="What is the population of Phoenix, AZ?",
        expected_trajectory=["web_search"],
        metadata={"category": "tool_usage"}
    ),
    Case[str, str](
        name="multi-tool-query",
        input="Find detailed information about Houston, TX population from an official source.",
        expected_trajectory=["web_search", "get_page"],
        metadata={"category": "tool_usage"}
    )
]

trajectory_evaluator = TrajectoryEvaluator(
    rubric="""
    Evaluate the tool usage trajectory:
    1. Correct tool selection - Were the expected tools used?
    2. Logical sequence - Were tools used in a sensible order?
    3. Efficiency - Were unnecessary tools avoided?
    
    Score 1.0 if all expected tools were used appropriately.
    Score 0.7 if expected tools were used but with some inefficiency.
    Score 0.5 if some expected tools were missing but task was completed.
    Score 0.0 if wrong tools were used or critical tools were missing.
    """,
    include_inputs=True
)

sample_agent = Agent(tools=[web_search, get_page], callback_handler=None)
tool_descriptions = tools_use_extractor.extract_tools_description(sample_agent, is_short=True)
trajectory_evaluator.update_trajectory_description(tool_descriptions)

def get_response_with_trajectory(case: Case) -> dict:
    """Run the agent and capture both output and tool usage trajectory."""
    agent = Agent(
        system_prompt=system_prompt,
        tools=[web_search, get_page],
        model=chatbot_model,
        callback_handler=None
    )
    response = agent(case.input)
    trajectory = tools_use_extractor.extract_agent_tools_used_from_messages(agent.messages)
    return {"output": str(response), "trajectory": trajectory}

trajectory_experiment = Experiment[str, str](
    cases=trajectory_test_cases,
    evaluators=[trajectory_evaluator]
)

trajectory_reports = trajectory_experiment.run_evaluations(get_response_with_trajectory)
trajectory_reports[0].run_display()
```

### Section 4: Custom Evaluators for Domain-Specific Checks

LLM-as-judge evaluators are powerful but sometimes overkill. If you need to check a deterministic property — like whether the response contains specific XML tags with numeric values — a programmatic evaluator is faster, cheaper, and more reliable.

Strands Evals lets you subclass `Evaluator` and implement your own `evaluate` method. You receive the full evaluation data (input, expected output, actual output) and return an `EvaluationOutput` with a score, pass/fail flag, and reasoning. This pattern is ideal for format validation, regex checks, latency thresholds, or any rule that doesn't require judgment.

**Build:** Implement a custom evaluator that validates XML tag formatting.

```python
import re
from strands_evals.evaluators import Evaluator
from strands_evals.types import EvaluationData, EvaluationOutput

class XMLFormatEvaluator(Evaluator[str, str]):
    """Custom evaluator that checks for proper XML tag formatting in responses."""
    
    def evaluate(self, evaluation_case: EvaluationData[str, str]) -> EvaluationOutput:
        response = evaluation_case.actual_output
        
        has_response_tag = bool(re.search(r'<response>.*?</response>', response, re.DOTALL))
        has_pop_tag = bool(re.search(r'<pop>\d+</pop>', response))
        has_area_tag = bool(re.search(r'<area>[\d.]+</area>', response))
        
        tags_present = sum([has_response_tag, has_pop_tag, has_area_tag])
        score = tags_present / 3.0
        
        missing = []
        if not has_response_tag: missing.append("<response>")
        if not has_pop_tag: missing.append("<pop>")
        if not has_area_tag: missing.append("<area>")
        
        reason = "All required XML tags present and properly formatted" if score == 1.0 else f"Missing or malformed tags: {', '.join(missing)}"
        
        return EvaluationOutput(
            score=score,
            test_pass=score >= 0.66,
            reason=reason,
            label="complete" if score == 1.0 else "incomplete"
        )
```

### Section 5: Combining Evaluators in Multi-Dimensional Experiments

Real agent evaluation is never one-dimensional. An agent that returns accurate data in an unparseable format is just as problematic as one that formats beautifully but hallucinates numbers. You need to assess multiple quality dimensions simultaneously.

Strands Evals supports this by accepting a list of evaluators in a single `Experiment`. Each evaluator produces its own report, giving you a multi-dimensional scorecard. This lets you identify specific failure modes — maybe your agent scores 0.9 on accuracy but 0.3 on format compliance, telling you exactly where to focus improvement efforts.

**Build:** Run a comprehensive experiment combining all three evaluators.

```python
comprehensive_test_cases = [
    Case[str, str](
        name="comprehensive-seattle",
        input="What is the population and area in square miles of Seattle, WA?",
        expected_output="Population: 780995, Area: 83.8 square miles",
        expected_trajectory=["web_search"],
        metadata={"category": "comprehensive", "city": "Seattle"}
    )
]

comprehensive_experiment = Experiment[str, str](
    cases=comprehensive_test_cases,
    evaluators=[
        accuracy_evaluator,
        trajectory_evaluator,
        XMLFormatEvaluator()
    ]
)

comprehensive_reports = comprehensive_experiment.run_evaluations(get_response_with_trajectory)

for i, report in enumerate(comprehensive_reports):
    print(f"\nEvaluator {i+1} Results:")
    report.run_display()

# Save for later comparison
comprehensive_experiment.to_file("comprehensive_evaluation.json")
```

## Challenges

**Challenge: Evaluate a Modified Agent Across All Dimensions**

Create a second agent with a different system prompt (e.g., remove the XML formatting instructions or change the model to a different Bedrock model). Build an experiment that evaluates both agents on the same 5 test cases using all three evaluator types (output accuracy, trajectory, and format compliance). Compare the results to determine which agent performs better overall.

**Assessment criteria:**

1. Code runs without errors and produces evaluation reports for both agents
2. Uses `Case` objects with both `expected_output` and `expected_trajectory` fields
3. Implements at least one custom evaluator (can extend `XMLFormatEvaluator` or create a new one)
4. Combines 3+ evaluators in a single `Experiment` for each agent
5. Produces a comparison summary showing which agent scores higher on each dimension
6. Learner can explain their rubric design choices and what the trajectory results reveal about agent behavior

---

## Deep-Dive Challenge

Strands is an **agent-focused** framework — it treats the system as a workflow with multiple steps, tool calls, and state transitions. You evaluate process quality in addition to final output. This deep-dive pushes you beyond notebook-level usage into advanced agent evaluation patterns.

### Workflow

| Stage | What you implement |
|---|---|
| Agent instrumentation | Capture traces/spans for a multi-step agent workflow |
| Step-level metrics | Metrics per agent step (tool selection accuracy, retrieval quality, reasoning correctness) |
| End-to-end metrics | Task completion, total latency, cost |
| Failure analysis | Identify where and why the agent fails (wrong tool, bad retrieval, hallucination) |
| Observability | Dashboard or structured log output showing per-step and aggregate health |

### "Beyond" Examples for Strands

- Multi-agent comparison
- Tool-use efficiency scoring
- Conversation-level coherence tracking

### Scoring Rubric

| Tier | Points | Criteria |
|---|---|---|
| **Functional** | 60-69 | Complete workflow runs end-to-end; uses only notebook-level features; results are valid |
| **Extended** | 70-84 | Adds 1 capability not in notebook; clear justification for the extension |
| **Advanced** | 85-94 | Adds 2+ capabilities; demonstrates iteration (before/after comparison); addresses a real evaluation gap |
| **Exceptional** | 95-100 | Novel approach; production-quality output (CI-ready, dashboarded, or automated); teaches the reviewer something new |

### Assessment Criteria

| Criterion | Weight | Description |
|---|---|---|
| Complete workflow execution | 25% | All stages implemented and runnable; produces valid output |
| Beyond-notebook features | 25% | Number and quality of capabilities not covered in source notebook |
| Justification & analysis | 20% | Why each metric/feature was chosen; what evaluation gap it addresses |
| Iteration evidence | 15% | Before/after comparison showing the pipeline caught or improved something |
| "What was left out" | 10% | Identifies limitations; names what they'd need to cover them |
| Code quality & documentation | 5% | Readable, commented, reproducible |

### Tips

1. **Start with the notebook** — get it running, then extend one piece at a time.
2. **Define your "beyond" early** — decide what you're adding before you start coding.
3. **Document as you go** — capture why you chose each metric and what gap it fills.
4. **Show iteration** — run your eval, change something, re-run, and compare results. This is the strongest signal of understanding.
5. **Name your limitations** — the rubric rewards honesty about what's missing.

---

## Wrap-Up

You built a complete agent evaluation pipeline using the Strands Evals SDK. You learned to structure test cases with the `Case` class, evaluate output quality with LLM-as-judge rubrics, assess tool-usage trajectories, create custom programmatic evaluators, and combine multiple evaluators for comprehensive scoring.

Key distinction from Module 03: there you used Strands to *build* the agent. Here you used the Strands Evals SDK to *evaluate* agent behavior — measuring not just what the agent says, but how it arrives at its answers.

For a deeper challenge that extends this work, see **CHALLENGE-deep-dive.md** — it asks you to design a full evaluation suite for a multi-step agent workflow, combining the techniques from this module with cross-framework comparison.

**Suggested next module:** If you haven't completed SKILL-promptfoo.md yet, try it next to see how a CLI-driven evaluation framework compares to the SDK-based approach you used here.
