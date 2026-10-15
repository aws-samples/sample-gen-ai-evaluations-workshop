"""A small tool-use loop on the Amazon Bedrock Converse API, shared by the validation agent and the judges.

The verifier runs this in the agent's container after the agent stops. The verifier phase can reach
only the Bedrock runtime endpoint (see [verifier] in task.toml). Only boto3 and the standard library are
used.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

TESTS = Path('/tests')
REPO = Path('/work/repo')
LOGS = Path('/logs/verifier')
MAX_TOOL_OUTPUT = 12_000

MODELS = tomllib.loads((TESTS / 'judge' / 'models.toml').read_text())


def model_for(role: str) -> str:
    """The model of a stage: VERIFIER_MODEL_ID from the environment, or the default in models.toml."""
    return os.environ.get('VERIFIER_MODEL_ID') or MODELS[role]['model']


def client():
    return boto3.client(
        'bedrock-runtime',
        config=Config(retries={'max_attempts': 8, 'mode': 'adaptive'}, read_timeout=300),
    )


class Tool:
    """A tool that the model can call: a name, a description, a JSON schema, and an implementation."""

    def __init__(self, name: str, description: str, schema: dict, impl: Callable[[dict], str]):
        self.name, self.description, self.schema, self.impl = name, description, schema, impl

    def spec(self) -> dict:
        return {'toolSpec': {'name': self.name, 'description': self.description, 'inputSchema': {'json': self.schema}}}


def _clip(text: str) -> str:
    if len(text) <= MAX_TOOL_OUTPUT:
        return text
    return text[:MAX_TOOL_OUTPUT] + f'\n[... {len(text) - MAX_TOOL_OUTPUT} more characters not shown]'


def _inside(path: str, roots: list[Path]) -> Path:
    resolved = (REPO / path).resolve() if not path.startswith('/') else Path(path).resolve()
    if not any(resolved == r or r in resolved.parents for r in roots):
        raise ValueError(f'{path} is outside the readable directories: {", ".join(map(str, roots))}')
    return resolved


def read_only_tools(roots: list[Path]) -> list[Tool]:
    """Tools to read the repository and the verifier files. Relative paths are relative to the repository."""

    def read_file(args: dict) -> str:
        path = _inside(args['path'], roots)
        lines = path.read_text(errors='replace').splitlines()
        start = max(int(args.get('start_line', 1)), 1)
        end = min(int(args.get('end_line', start + 399)), len(lines))
        return '\n'.join(f'{n:5} {lines[n - 1]}' for n in range(start, end + 1)) or '[empty file]'

    def list_directory(args: dict) -> str:
        path = _inside(args.get('path', '.'), roots)
        return '\n'.join(p.name + ('/' if p.is_dir() else '') for p in sorted(path.iterdir()) if p.name != '.git')

    def search_code(args: dict) -> str:
        path = _inside(args.get('path', '.'), roots)
        cmd = ['rg', '--line-number', '--max-count', '20', '--glob', '!.git', args['pattern'], str(path)]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
        return out or '[no matches]'

    return [
        Tool('read_file', 'Read lines of a file (default: 400 lines from start_line).',
             {'type': 'object', 'properties': {'path': {'type': 'string'}, 'start_line': {'type': 'integer'},
                                               'end_line': {'type': 'integer'}}, 'required': ['path']}, read_file),
        Tool('list_directory', 'List a directory.',
             {'type': 'object', 'properties': {'path': {'type': 'string'}}}, list_directory),
        Tool('search_code', 'Search with a ripgrep regular expression.',
             {'type': 'object', 'properties': {'pattern': {'type': 'string'}, 'path': {'type': 'string'}},
              'required': ['pattern']}, search_code),
    ]


class Conversation:
    """One model conversation with tools. The model ends a turn by calling the submit tool."""

    def __init__(self, role: str, system: str, tools: list[Tool], submit_name: str, submit_schema: dict):
        cfg = MODELS[role]
        self.role, self.model, self.max_turns = role, model_for(role), int(cfg['max_turns'])
        self.max_tokens, self.temperature = int(cfg.get('max_tokens', 8000)), float(cfg.get('temperature', 0))
        self.system, self.tools = system, {t.name: t for t in tools}
        self.submit_name = submit_name
        self.submit = Tool(submit_name, 'Submit your final result. Call this exactly once, when you are done.',
                           submit_schema, lambda args: 'submitted')
        self.messages: list[dict] = []
        self.usage = {'input_tokens': 0, 'output_tokens': 0, 'turns': 0, 'temperature': self.temperature}
        self.transcript = LOGS / f'{role}_transcript.jsonl'
        self._bedrock = client()

    def _log(self, record: dict) -> None:
        with self.transcript.open('a') as f:
            f.write(json.dumps(record, default=str) + '\n')

    def _converse(self) -> dict:
        """Call the model. Send the temperature only if the model accepts it."""
        config: dict = {'maxTokens': self.max_tokens}
        if self.usage['temperature'] is not None:
            config['temperature'] = self.usage['temperature']
        try:
            return self._bedrock.converse(
                modelId=self.model,
                system=[{'text': self.system}],
                messages=self.messages,
                toolConfig={'tools': [t.spec() for t in [*self.tools.values(), self.submit]]},
                inferenceConfig=config,
            )
        except ClientError as error:
            # Some models (for example Claude Haiku 5.5) reject the temperature parameter. Then use the
            # model's default, and record it: usage['temperature'] is None in the stage output.
            if 'temperature' in config and 'temperature' in str(error):
                self.usage['temperature'] = None
                return self._converse()
            raise

    def ask(self, text: str) -> dict:
        """Send a user message. Run tool calls until the model calls the submit tool. Return its input."""
        self.messages.append({'role': 'user', 'content': [{'text': text}]})
        self._log({'user': text})
        for turn in range(self.max_turns):
            response = self._converse()
            self.usage['input_tokens'] += response['usage']['inputTokens']
            self.usage['output_tokens'] += response['usage']['outputTokens']
            self.usage['turns'] += 1
            message = response['output']['message']
            self.messages.append(message)
            self._log({'assistant': message['content']})
            calls = [c['toolUse'] for c in message['content'] if 'toolUse' in c]
            if not calls:
                self.messages.append({'role': 'user', 'content': [{'text': f'Use the tools. When you are done, call {self.submit_name}.'}]})
                continue
            results = []
            submitted = None
            for call in calls:
                if call['name'] == self.submit_name:
                    submitted = call['input']
                    output = 'submitted'
                elif call['name'] in self.tools:
                    try:
                        output = _clip(self.tools[call['name']].impl(call['input']))
                    except Exception as error:  # the model gets the error and can try again
                        output = f'[error] {error}'
                else:
                    output = f'[error] unknown tool {call["name"]}'
                print(f'  [{self.role}] {call["name"]} {json.dumps(call["input"])[:120]}', file=sys.stderr)
                results.append({'toolResult': {'toolUseId': call['toolUseId'], 'content': [{'text': output}]}})
            if turn >= self.max_turns - 3:
                results.append({'text': f'You have {self.max_turns - turn - 1} turns left. Call {self.submit_name} now.'})
            self.messages.append({'role': 'user', 'content': results})
            if submitted is not None:
                self._log({'submitted': submitted})
                return submitted
        raise RuntimeError(f'{self.role}: no {self.submit_name} call after {self.max_turns} turns')


def write_json(name: str, data: Any) -> None:
    (LOGS / name).write_text(json.dumps(data, indent=2, default=str))


def git(*args: str, check: bool = True) -> str:
    return subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True, check=check).stdout
