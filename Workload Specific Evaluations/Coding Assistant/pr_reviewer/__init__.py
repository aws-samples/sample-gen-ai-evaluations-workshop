"""Aligned PR reviewer — used as an LLM judge in the coding-assistant eval,
and usable standalone in CI/CD. See reviewer.review() and rubric.Rubric."""

from .reviewer import ReviewDimension, ReviewResult, review
from .rubric import Rubric

__all__ = ["Rubric", "ReviewDimension", "ReviewResult", "review"]
