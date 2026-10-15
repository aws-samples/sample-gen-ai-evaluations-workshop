"""Tests that the coding agent wrote. Not gating: a senior engineer adds a regression test without being told.

The agent's test files are the test_*.py files under tests/ that the agent added or changed. They run two
times: on the agent's code, where they must pass, and on the base code with the same tests, where at least
one must fail (the tests reproduce the bug). Writes agent_tests.json. The working tree is the agent's
tree again at the end.

Usage: python3 agent_tests.py <base commit> <agent commit>
"""

from __future__ import annotations

import subprocess
import sys
import xml.etree.ElementTree as ET

from bedrock import LOGS, REPO, git, write_json


def run(files: list[str], junit: str) -> dict:
    subprocess.run(['python', '-m', 'pytest', *files, '-q', '-p', 'no:cacheprovider', f'--junitxml={junit}'],
                   cwd=REPO, capture_output=True, text=True, timeout=900)
    passed = failed = 0
    try:
        for tc in ET.parse(junit).getroot().iter('testcase'):
            if tc.find('skipped') is not None:
                continue
            if tc.find('failure') is None and tc.find('error') is None:
                passed += 1
            else:
                failed += 1
    except (FileNotFoundError, ET.ParseError):
        pass
    return {'passed': passed, 'failed': failed}


def main(base: str, agent: str) -> None:
    changed = git('diff', '--name-only', '--diff-filter=AM', base, agent, '--', 'tests').split()
    files = [f for f in changed if f.split('/')[-1].startswith('test_') and f.endswith('.py')]
    if not files:
        write_json('agent_tests.json', {'files': [], 'tests_added': 0, 'pass_on_fix': 0, 'reproduces_bug': 0})
        return
    on_fix = run(files, '/tmp/agent_tests_fix.xml')
    # The base code with the agent's tests: put every changed non-test path back to the base commit.
    for status_line in git('diff', '--name-status', base, agent).splitlines():
        status, path = status_line.split('\t')[0], status_line.split('\t')[-1]
        if path.startswith('tests/'):
            continue
        if status.startswith('A'):
            (REPO / path).unlink(missing_ok=True)
        else:
            git('checkout', base, '--', path, check=False)
    on_base = run(files, '/tmp/agent_tests_base.xml')
    git('checkout', agent, '--', '.', check=False)
    write_json('agent_tests.json', {
        'files': files, 'tests_added': on_fix['passed'] + on_fix['failed'],
        'on_fix': on_fix, 'on_base': on_base,
        'pass_on_fix': int(on_fix['failed'] == 0 and on_fix['passed'] > 0),
        'reproduces_bug': int(on_fix['failed'] == 0 and on_base['failed'] > 0),
    })


if __name__ == '__main__':
    try:
        main(sys.argv[1], sys.argv[2])
    except Exception as error:
        write_json('agent_tests.json', {'status': 'error', 'error': repr(error)})
    print((LOGS / 'agent_tests.json').read_text())
