## Task

`ToolParameterAccuracyEvaluator` and `ToolSelectionAccuracyEvaluator` flag correct tool calls as
hallucinated.

repro: an agent with two tools, `add_numbers` and `multiply_numbers`. the user asks "what is
(3 + 4) * 5?". the agent calls `add_numbers(a=3, b=4)`, gets `7`, then calls
`multiply_numbers(a=7, b=5)`. both calls are correct, but the judge says that the `7` in the second
call does not come from the conversation.

it looks like a tool call is judged without the results of the tool calls that the agent made before
it in the same turn. results from earlier turns are there.

a tool call must be judged with the results that the agent had when it made the call: the results of
the calls that finished before it started. it must not see its own result, or the results of calls
that were still running or that came after it.

## General instructions

- The code repo is at `/work/repo`.
- You are inside of a Docker container. The Python dependencies of the repository are installed, and
  `python` and `pytest` are on `PATH`. The network is not available, so you cannot install packages,
  and you cannot call a model on Amazon Bedrock.
- You are expected to act autonomously as a software engineer to complete tasks you are given.
- Do not stop until you feel you have completed the task and your code changes can be merged.
- You may need to use software engineering skills like analyzing the codebase, running the tests, and
  building small reproductions to complete the task. Not all tasks will be solvable by reading source
  code alone.
