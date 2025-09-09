# Agentic Evaluator
Agentic systems empowers LLM's to make sequences of decisions often through a cycle of reasoning, tool use and response generation.

In the context of evaluation its very important to consider the level of "autonomy" that is intended for our system or use case.

More autonomy or agency means we instruct the agents to keep making tool calls, refining its plan until it's confident that it has fulfilled the request.

Less agency means the system might cede control, stop or ask for human input if anything is unclear.

It's important to have a clear understanding of whether a product if less agentic or not in order to have a clear understanding of what constitutes an error verses intended behavior.
When building AI agents, evaluating their performance is crucial during this process. It's important to consider various qualitative and quantitative factors, including response quality, task completion, success, and inaccuracies or hallucinations. In evaluations, it's also important to consider comparing different agent configurations to optimize for specific desired outcomes. Given the dynamic and non-deterministic nature of LLMs, it's also important to have rigorous and frequent evaluations to ensure a consistent baseline for tracking improvements or regressions.


This guide covers approaches to evaluating agents. Effective evaluation is essential for measuring agent performance, tracking improvements, and ensuring your agents meet quality standards.

### 1. Create and activate virtual environment

```bash
# Create virtual environment with Python 3.12+
uv venv --python 3.12

# Activate the virtual environment

# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```
## Running Ecommerce Bot

```python
uvicorn backend.main:app --reload
# Open http://127.0.0.1:8000
```

See the [Unicorn Web Store](https://gitlab.aws.dev/applied-ai-and-data-architects/lululemon_agentic_demo/-/tree/main/experiments/backend?ref_type=heads) for more info on this project.
## 1. Create Test Cases
##### 1. [01_generate_synthetic_data.ipynb](./bootstrap_data/generate_synthetic_data.ipynb)

**Generate Synthetic Data for Customer Support**

When developing your test cases, consider building a diverse suite that spans multiple categories.

Some common categories to consider include: 1. Knowledge Retrieval - Facts, definitions, explanations 2. Reasoning - Logic problems, deductions, inferences 3. Tool Usage - Tasks requiring specific tool selection 4. Conversation - Multi-turn interactions 5. Edge Cases - Unusual or boundary scenarios 6. Safety - Handling of sensitive topics
In this notebook, you'll learn how to:

- Create realistic customer support scenarios and queries
- Generate diverse conversation examples covering different departments and scenarios
- Use AI to create high-quality synthetic datasets
- Structure data for effective evaluation workflows
- Review and curate generated data for quality assurance

## 2. Structured Testing
##### [tasks_and_evals.ipynb](./02_tasks_and_evals.ipynb)

**Building Evaluation Tasks and Running Evals**

This notebook covers:

- Setting up evaluation tasks in Braintrust
- Defining metrics for customer support quality (accuracy, helpfulness, tone)
- Running systematic evaluations on your chatbot
- Analyzing results and identifying areas for improvement
- Comparing different model configurations and prompting strategies
### Steps Initial Data analysis(error analysis) & Open Coding

In this section, you'll learn how to instrument your recipe chatbot to log traces to Braintrust with LiteLLM. From there, you'll learn how to use Braintrust to:

- Optimize your system prompts
- Build synthetic user queries
- Perform error analysis using the open and axial coding methods.

