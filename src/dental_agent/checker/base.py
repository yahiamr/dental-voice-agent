"""The checker interface and its answer type. Stage: checker.

What this file is:
    What every checker backend must provide, and the answer it returns.

What it contains:
    - CheckerError: the one error a backend raises.
    - Checker: the protocol each backend follows.
    - CheckAnswer: one backend's answers to one request.

How to understand it:
    A backend raises CheckerError on any failure: a timeout, a network error, or an answer it
    can't trust. It never returns a partial answer. CheckerChain catches the error and tries the
    next backend, so code above the chain never sees a failure.

Used by:
    checker/jev.py, checker/anthropic.py, checker/chain.py, tests/fakes.py
"""

from typing import Literal, Protocol

from pydantic import BaseModel, Field

Layer = Literal["jev", "anthropic"]
Role = Literal["caller", "reply"]  # Whose words are checked: the caller's, or a model reply.


class CheckerError(Exception):
    """A backend failed, or gave an answer that can't be trusted.

    timed_out is true only when the backend ran out of the time it was given. The chain uses it
    to tell a short turn from a broken backend.
    """

    def __init__(self, message: str, timed_out: bool = False) -> None:
        super().__init__(message)
        self.timed_out = timed_out


class CheckAnswer(BaseModel):
    """One backend's answers to one request."""

    layer: Layer
    answers: dict[str, bool]
    # The probability of yes for each question. Jev gives these; the Haiku checker does not.
    probabilities: dict[str, float] = Field(default_factory=dict)
    model: str  # The model version the response names, for the trace.
    seconds: float = 0.0  # Set by the chain: the whole ask, including failed attempts.
    # Set by the chain: what a second backend answered for a question in the grey zone. Its
    # answer replaces this one's in answers.
    second_opinions: dict[str, bool] = Field(default_factory=dict)


class Checker(Protocol):
    """A backend that answers yes-or-no questions about one piece of text."""

    name: Layer

    def ask(
        self, text: str, questions: dict[str, str], role: Role, time_limit_s: float
    ) -> CheckAnswer:
        """Answer every question, or raise CheckerError."""
        ...
