"""Validators that check user-built artifacts match the expected structure.

Each notebook step has the user direct their coding assistant to produce an
artifact (a task yaml, a rubric markdown, a gold-standard verdict, etc.).
These validators run in the notebook and give actionable feedback when
something is wrong, so the user can go back and correct Claude.
"""
