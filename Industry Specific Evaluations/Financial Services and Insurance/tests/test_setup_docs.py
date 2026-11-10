"""The setup instructions state the Python version the pinned requirements need.

numpy 2.5.3 in requirements.txt requires Python 3.12 or later, so a kernel or
virtual environment on an older Python fails at the first install. The README,
the requirements file and each notebook's setup cell must say so, and the
README's venv command must name a 3.12+ interpreter: on many machines
``python`` does not exist and ``python3`` is an older version.
"""

import json
import re
from pathlib import Path

import pytest

MODULE_ROOT = Path(__file__).resolve().parent.parent
README = (MODULE_ROOT / "README.md").read_text()
MINIMUM = "Python 3.12 or later"


def test_readme_states_the_minimum_python():
    setup = README[README.index("### Environment"):README.index("## Scenario 1")]
    assert MINIMUM in setup


def test_readme_venv_command_names_a_python_3_12_or_later_interpreter():
    commands = re.findall(r"^(\S+) -m venv \.venv$", README, re.MULTILINE)
    assert commands, "no venv command in the README"
    for interpreter in commands:
        match = re.fullmatch(r"python3\.(\d+)", interpreter)
        assert match and int(match.group(1)) >= 12, interpreter


def test_requirements_state_the_minimum_python():
    assert MINIMUM in (MODULE_ROOT / "requirements.txt").read_text()


@pytest.mark.parametrize("path", sorted(MODULE_ROOT.glob("*.ipynb")), ids=lambda p: p.name)
def test_notebook_setup_cell_states_the_minimum_python(path):
    cells = json.loads(path.read_text())["cells"]
    setup = next("".join(c["source"]) for c in cells
                 if c["cell_type"] == "markdown" and "".join(c["source"]).startswith("## Setup"))
    assert MINIMUM in setup
