# Contributing Guidelines

Thank you for your interest in contributing to our project. Whether it's a bug report, new feature, correction, or additional
documentation, we greatly value feedback and contributions from our community.

Please read through this document before submitting any issues or pull requests to ensure we have all the necessary
information to effectively respond to your bug report or contribution.


## Reporting Bugs/Feature Requests

We welcome you to use the GitHub issue tracker to report bugs or suggest features.

When filing an issue, please check existing open, or recently closed, issues to make sure somebody else hasn't already
reported the issue. Please try to include as much information as you can. Details like these are incredibly useful:

* A reproducible test case or series of steps
* The version of our code being used
* Any modifications you've made relevant to the bug
* Anything unusual about your environment or deployment


## Contributing via Pull Requests
Contributions via pull requests are much appreciated. Before sending us a pull request, please ensure that:

1. You are working against the latest source on the *main* branch.
2. You check existing open, and recently merged, pull requests to make sure someone else hasn't addressed the problem already.
3. You open an issue to discuss any significant work - we would hate for your time to be wasted.

To send us a pull request, please:

1. Fork the repository.
2. Modify the source; please focus on the specific change you are contributing. If you also reformat all the code, it will be hard for us to focus on your change.
3. Ensure local tests pass.
4. Commit to your fork using clear commit messages.
5. Send us a pull request, answering any default questions in the pull request interface.
6. Pay attention to any automated CI failures reported in the pull request, and stay involved in the conversation.

GitHub provides additional document on [forking a repository](https://help.github.com/articles/fork-a-repo/) and
[creating a pull request](https://help.github.com/articles/creating-a-pull-request/).


## Finding contributions to work on
Looking at the existing issues is a great way to find something to contribute on. As our projects, by default, use the default GitHub issue labels (enhancement/bug/duplicate/help wanted/invalid/question/wontfix), looking at any 'help wanted' issues is a great place to start.


## Code of Conduct
This project has adopted the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct).
For more information see the [Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq) or contact
opensource-codeofconduct@amazon.com with any additional questions or comments.


## Security issue notifications
If you discover a potential security issue in this project we ask that you notify AWS/Amazon Security via our [vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a public github issue.


## Workshop Development Setup

### IAM Permissions

Attach this policy to the learner role (or Studio execution role):

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": [
      "bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream",
      "bedrock:Converse", "bedrock:ConverseStream",
      "bedrock:CreateGuardrail", "bedrock:TagResource", "bedrock:ApplyGuardrail",
      "bedrock:ListGuardrails", "bedrock:GetGuardrail",
      "cloudwatch:PutMetricData",
      "cloudwatch:PutDashboard", "cloudwatch:PutMetricAlarm",
      "logs:FilterLogEvents", "ecr:DescribeRepositories"
    ],
    "Resource": "*"
  }]
}
```

### Environment Setup

```bash
# Core (all modules)
pip install boto3 pandas numpy

# Module 02 — Quality Metrics
pip install matplotlib seaborn scipy

# Foundational Module 04 — Agentic Metrics
pip install strands-agents ddgs beautifulsoup4

# Optional Workload Evaluations
pip install llama-index faiss-cpu chromadb python-dotenv PyPDF2

# Optional Framework Evaluations
pip install strands-agents strands-agents-tools strands-agents-evals "dspy[litellm]>=3.1,<4" bedrock-agentcore ddgs

# Requires Node.js 18+
npm install -g promptfoo
```

### Model IDs

Do not hard-code Bedrock model IDs in notebooks or helper code. The default and judge models are defined once, in [`Foundational Evaluations/model_config.py`](Foundational%20Evaluations/model_config.py), so a model can be replaced in one place when it is retired. Import `DEFAULT_MODEL_ID` and `JUDGE_MODEL_ID` from that file; its docstring shows the import for notebooks in other modules.

A module can keep its own `model_config.py` for settings that are specific to it, such as the candidate set in Capacity Management or the prompts in Multiagent Shared Context. If the module also needs the workshop's default or judge model, that file should read them from the shared file instead of repeating the IDs.

Claude Sonnet 5 rejects the `temperature` and `top_p` parameters, so do not send them.


## Licensing

See the [LICENSE](LICENSE) file for our project's licensing. We will ask you to confirm the licensing of your contribution.
