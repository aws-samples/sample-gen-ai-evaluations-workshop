"""Validation agent: write behavioral tests adapted to the agent's patch, run them, and judge them.

Procedure (after Senior SWE-Bench):
1. The validation agent reads the agent's patch and tests/validate/validation_spec.toml. For each story it
   writes one pytest file with one test function `test_<story>(inputs, expected)`.
2. The driver runs each file once for each case in the spec. A story passes if all its cases pass.
3. The validation judge reviews the scripts and the results. It accepts them, asks for one revision, or
   discards the trial (for example, if the tests mock the code under test).
4. Integrity check: the validation agent must not change the repository. If it did, the trial is discarded.

Writes /logs/verifier/validation.json. `status` is `accepted`, `discarded`, or `error`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

from bedrock import LOGS, REPO, TESTS, Conversation, Tool, git, read_only_tools, write_json

SPEC = tomllib.loads((TESTS / 'validate' / 'validation_spec.toml').read_text())
WORK = Path('/tmp/validation')
SCRIPTS = WORK / 'scripts'
CASES = WORK / 'cases'

CONFTEST = '''import json, os, pathlib
def pytest_generate_tests(metafunc):
    if {"inputs", "expected"} <= set(metafunc.fixturenames):
        story = pathlib.Path(metafunc.definition.fspath).stem.removeprefix("test_")
        cases = json.loads((pathlib.Path(os.environ["VALIDATION_CASES"]) / f"{story}.json").read_text())
        metafunc.parametrize("inputs,expected", [(c["inputs"], c["expected"]) for c in cases],
                             ids=[c["name"] for c in cases])
'''

SYSTEM = '''You are a test engineer. A developer submitted a change to the repository in /work/repo. Your job is
to write behavioral tests that check if the change fulfills the user stories that you get. You implement
the test procedure faithfully. You do not fix the code, and you do not make a test pass by changing what it
checks. If the change does not implement a story, or implements it incorrectly, the test must fail.

Rules:
- Exercise the real code. Do not mock the code under test. Mock only what the procedure allows.
- Use the `inputs` and `expected` values. Do not hard-code values that match `expected`.
- Do not skip tests, catch AssertionError, or weaken assertions (for example, `is not None` instead of `==`).
- Do not change files in /work/repo. Write files only with write_test_file.
- Read the diff first. Adapt the tests to the shape of the change: the names, the signatures, and where
  the change puts the data. Read only the source that you need.
- Work efficiently. Do not check the cases one by one, and do not decide if the change is correct: the
  driver runs your tests and the result is the signal. You may run your tests one time to find errors in
  your test code (import errors, typos), with the command that you get. A test that fails because the
  change is wrong is a correct test. Then call finish.'''


def stories_text() -> str:
    parts = []
    for story in SPEC['stories']:
        steps = '\n'.join(f'  - {s.strip()}' for s in story['procedure'])
        example = json.dumps({'inputs': story['cases'][0]['inputs'], 'expected': story['cases'][0]['expected']})
        parts.append(f"### Story `{story['id']}`\n{story['description'].strip()}\n\nProcedure:\n{steps}\n\n"
                     f"Example case (the driver runs {len(story['cases'])} cases):\n{example}")
    return '\n\n'.join(parts)


def bash_tool() -> Tool:
    def bash(args: dict) -> str:
        r = subprocess.run(['bash', '-lc', args['command']], cwd=REPO, capture_output=True, text=True, timeout=180)
        return f'exit {r.returncode}\n{r.stdout}{r.stderr}'
    return Tool('bash', 'Run a shell command in /work/repo (timeout 180 s). Use it to read code and try imports.',
                {'type': 'object', 'properties': {'command': {'type': 'string'}}, 'required': ['command']}, bash)


def write_tool() -> Tool:
    def write(args: dict) -> str:
        name = Path(args['story_id']).name
        if name not in {s['id'] for s in SPEC['stories']}:
            return f'[error] unknown story id {name}'
        (SCRIPTS / f'{name}.py').write_text(args['content'])
        return f'wrote {SCRIPTS / name}.py'
    return Tool('write_test_file', 'Write the pytest file of one story. It replaces the file if it exists.',
                {'type': 'object', 'properties': {'story_id': {'type': 'string'}, 'content': {'type': 'string'}},
                 'required': ['story_id', 'content']}, write)


def run_stories() -> dict:
    """Run each story file for all of its cases. Return the results by story."""
    results = {}
    env = {'VALIDATION_CASES': str(CASES), 'PATH': '/opt/venv/bin:/usr/local/bin:/usr/bin:/bin', 'HOME': '/tmp'}
    for story in SPEC['stories']:
        sid, script = story['id'], SCRIPTS / f"{story['id']}.py"
        junit = WORK / f'{sid}.xml'
        junit.unlink(missing_ok=True)
        if not script.exists():
            results[sid] = {'passed': 0, 'total': len(story['cases']), 'pass': False, 'log': 'no test file'}
            continue
        r = subprocess.run(['python', '-m', 'pytest', str(script), '-q', '-p', 'no:cacheprovider', '--rootdir', str(SCRIPTS),
                            '-c', str(SCRIPTS / 'pytest.ini'), f'--junitxml={junit}'],
                           cwd=REPO, capture_output=True, text=True, timeout=600, env=env)
        cases = {}
        if junit.exists():
            for tc in ET.parse(junit).getroot().iter('testcase'):
                bad = tc.find('failure') is not None or tc.find('error') is not None or tc.find('skipped') is not None
                cases[tc.get('name', '')] = not bad
        passed = sum(cases.values())
        total = len(story['cases'])
        # A story passes only if every case ran and passed. A collection error gives no cases.
        results[sid] = {'passed': passed, 'total': total, 'pass': passed == total and len(cases) == total,
                        'cases': cases, 'log': (r.stdout + r.stderr)[-3000:]}
    return results


def results_text(results: dict) -> str:
    return '\n\n'.join(f"Story {sid}: {r['passed']}/{r['total']} cases passed\n{r['log']}" for sid, r in results.items())


JUDGE_SYSTEM = '''You review the work of a validation agent: a test engineer who wrote behavioral tests for user
stories, adapted to a developer's change. You do not grade the developer's change. You grade whether the
tests are a faithful and honest implementation of the procedures. A test that fails because the change is
wrong is correct work. Read the scripts, the spec, the results, and the code as you need.

Score each criterion 0-3:
- behavioral_authenticity: the tests exercise the real code. 0 = mocks the code under test, fabricates a pass
  condition, skips tests, swallows AssertionError, or weakens assertions.
- procedure_fidelity: each test follows its story procedure and uses inputs/expected. 0 = ignores the
  procedure or asserts hard-coded values.
- story_coverage: every story has a test that checks what the story describes. 0 = a story is missing.
- execution_soundness: failures come from the behavior under test, not from broken test code (import
  errors, wrong signatures, typos). 0 = a story fails only because the test code is broken.
- assertion_strength: assertions are as strict as `expected` implies.

Decide: `accept` (all scores >= 1 and no fix is needed), `revise` (a fix in the test code is needed; give
precise feedback), or `discard` (a score is 0 for a reason that a revision cannot fix, for example
collusion with the developer).'''

JUDGE_SCHEMA = {
    'type': 'object',
    'properties': {
        'scores': {'type': 'object', 'properties': {k: {'type': 'integer', 'minimum': 0, 'maximum': 3} for k in (
            'behavioral_authenticity', 'procedure_fidelity', 'story_coverage', 'execution_soundness',
            'assertion_strength')}},
        'decision': {'type': 'string', 'enum': ['accept', 'revise', 'discard']},
        'feedback': {'type': 'string'},
    },
    'required': ['scores', 'decision', 'feedback'],
}


def judge(results: dict, final: bool) -> dict:
    scripts = '\n\n'.join(f'## {p.name}\n```python\n{p.read_text()}\n```' for p in sorted(SCRIPTS.glob('*.py')) if p.name != 'conftest.py')
    convo = Conversation('validation_judge', JUDGE_SYSTEM, read_only_tools([REPO, TESTS / 'validate', SCRIPTS, LOGS]),
                         'submit_review', JUDGE_SCHEMA)
    note = 'This is the final review: decide accept or discard only.' if final else ''
    verdict = convo.ask(f'# Stories and procedures\n{stories_text()}\n\n# Test scripts\n{scripts}\n\n'
                        f'# Results\n{results_text(results)}\n\nThe developer patch is /logs/verifier/agent.patch. {note}')
    if final and verdict['decision'] == 'revise':
        verdict['decision'] = 'accept' if min(verdict['scores'].values()) >= 1 else 'discard'
    return {**verdict, 'usage': convo.usage}


def ask_or_keep(agent: Conversation, text: str) -> None:
    """Ask the validation agent. If it uses all of its turns, keep the test files that it wrote."""
    try:
        agent.ask(text)
    except RuntimeError:
        if not any(p.name != 'conftest.py' for p in SCRIPTS.glob('*.py')):
            raise
        print('validation agent: out of turns; the test files that it wrote are used', flush=True)


def main() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    SCRIPTS.mkdir(parents=True)
    CASES.mkdir()
    (SCRIPTS / 'conftest.py').write_text(CONFTEST)
    (SCRIPTS / 'pytest.ini').write_text('[pytest]\n')
    for story in SPEC['stories']:
        (CASES / f"{story['id']}.json").write_text(json.dumps(story['cases']))
    before = git('status', '--porcelain') + git('rev-parse', 'HEAD')

    patch = (LOGS / 'agent.patch').read_text()
    stat = (LOGS / 'agent.stat').read_text().strip()
    diff_note = (f'The change touches these files:\n```\n{stat}\n```\nThe full diff is in /logs/verifier/agent.patch.'
                 if patch.strip() else 'The developer made NO change. Write the tests anyway; they must fail naturally.')
    agent = Conversation('validation_agent', SYSTEM, [bash_tool(), write_tool(), *read_only_tools([REPO, LOGS, TESTS / 'validate'])],
                         'finish', {'type': 'object', 'properties': {'notes': {'type': 'string'}}})
    general = SPEC.get('instructions', '').strip()
    ask_or_keep(agent, f'{diff_note}\n\n# Instructions for all stories\n{general}\n\n'
              'For each story, write one pytest file with exactly one function `test_<story_id>(inputs, expected)`. '
              f'The file {SCRIPTS}/conftest.py already exists: it injects `inputs` and `expected` for each case from '
              f'{CASES}/<story_id>.json. Do not change it, and do not write your own parametrization. To run a story: '
              f'`VALIDATION_CASES={CASES} python -m pytest {SCRIPTS}/<story_id>.py -q -p no:cacheprovider '
              f'--rootdir {SCRIPTS} -c {SCRIPTS}/pytest.ini`. Then call finish.\n\n' + stories_text())

    results = run_stories()
    reviews = [judge(results, final=False)]
    if reviews[0]['decision'] == 'revise':
        ask_or_keep(agent, f'The tests ran. Results:\n{results_text(results)}\n\nReviewer feedback:\n{reviews[0]["feedback"]}\n\n'
                  'Fix the test code where the feedback is correct. Do not change what a test checks to make it pass. '
                  'Then call finish.')
        results = run_stories()
        reviews.append(judge(results, final=True))

    tampered = (git('status', '--porcelain') + git('rev-parse', 'HEAD')) != before
    if tampered:
        git('checkout', '-q', '--', '.', check=False)
    decision = 'discard' if tampered else reviews[-1]['decision']
    passed = sum(r['pass'] for r in results.values())
    write_json('validation.json', {
        'status': 'accepted' if decision == 'accept' else 'discarded',
        'discard_reason': 'the validation agent changed the repository' if tampered else
                          (reviews[-1]['feedback'] if decision == 'discard' else ''),
        'validation_score': round(passed / len(results), 4),
        'passed_stories': passed, 'total_stories': len(results),
        'stories': {k: {kk: vv for kk, vv in v.items() if kk != 'log'} for k, v in results.items()},
        'reviews': reviews, 'agent_usage': agent.usage, 'revised': len(reviews) > 1,
    })
    shutil.copytree(SCRIPTS, LOGS / 'validation_scripts', dirs_exist_ok=True)
    print(f'validation: {decision}, {passed}/{len(results)} stories pass')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:  # an infrastructure failure: the trial is invalid, not a fail for the agent
        write_json('validation.json', {'status': 'error', 'error': repr(error)})
        print(f'validation error: {error!r}')
