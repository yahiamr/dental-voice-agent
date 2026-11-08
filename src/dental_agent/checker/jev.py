"""The Jev backend: TypeSafe's decision model on OpenRouter. Stage: checker.

What this file is:
    Asks Jev the checker's questions and reads its probabilities.

What it contains:
    - JevChecker(model, key, cutoff, opener): ask() posts the questions and returns a
      CheckAnswer, or raises CheckerError.

How to understand it:
    It posts {model, state, questions} to OpenRouter's alpha decisions endpoint with the
    standard library, so it adds no dependency. Each question is Jev's yes-or-no type, which Jev
    calls "noul", and the answer is the probability of yes. A probability of the cut-off or more
    counts as yes.

    Jev's answer alone can silence an emergency phrase, so the response is checked strictly. It
    fails when the status is outside 2xx, when an answer is missing or was not requested, when a
    probability is not a number from 0 to 1 (a bool is refused), or when the model is missing.
    The timeout covers connecting and reading, but not the name lookup.

    opener is urllib.request.urlopen by default. Tests pass a fake one, so they need no network.

Uses:
    checker/base.py

Used by:
    factories.py, tests/test_checker.py
"""

import json
import time
import urllib.request
from collections.abc import Callable
from typing import Any

from dental_agent.checker.base import CheckAnswer, CheckerError, Role

DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"


class JevChecker:
    """Asks Jev on OpenRouter, and checks its answer strictly."""

    name = "jev"

    def __init__(
        self,
        model: str,
        key: str,
        cutoff: float,
        opener: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        self.model = model
        self._key = key
        self.cutoff = cutoff
        self._opener = opener

    def ask(
        self, text: str, questions: dict[str, str], role: Role, time_limit_s: float
    ) -> CheckAnswer:
        """Ask Jev every question, or raise CheckerError."""
        start = time.monotonic()
        try:
            body = self._post(text, questions, time_limit_s)
        except CheckerError:
            raise
        except Exception as error:  # Timeouts, network errors, HTTP errors, broken JSON.
            raise CheckerError(
                f"{type(error).__name__}: {error}", timed_out=_is_timeout(error)
            ) from error
        probabilities = _read_probabilities(body, questions)
        return CheckAnswer(
            layer="jev",
            answers={name: p >= self.cutoff for name, p in probabilities.items()},
            probabilities=probabilities,
            model=body["model"],
            seconds=time.monotonic() - start,
        )

    def _post(self, text: str, questions: dict[str, str], time_limit_s: float) -> dict:
        """Send one request, and return the JSON reply."""
        payload = {
            "model": self.model,
            "state": text,
            "questions": {
                name: {"type": "noul", "instructions": instructions, "criteria": {}}
                for name, instructions in questions.items()
            },
        }
        request = urllib.request.Request(
            DECISIONS_URL,
            json.dumps(payload).encode(),
            {"Content-Type": "application/json", "Authorization": f"Bearer {self._key}"},
        )
        with self._opener(request, timeout=time_limit_s) as response:
            if not 200 <= response.status < 300:
                raise CheckerError(f"HTTP {response.status}")
            return json.loads(response.read())


TIMEOUT_STATUSES = (408, 504)  # Request Timeout, and Gateway Timeout from an upstream.


def _is_timeout(error: Exception) -> bool:
    """Return True if the request ran out of time.

    urllib raises TimeoutError, bare or inside a URLError. OpenRouter can also answer 408 or 504
    when its upstream ran out of time.
    """
    if isinstance(error, TimeoutError):
        return True
    if isinstance(error, urllib.error.HTTPError):
        return error.code in TIMEOUT_STATUSES
    return isinstance(error, urllib.error.URLError) and isinstance(error.reason, TimeoutError)


def _read_probabilities(body: Any, questions: dict[str, str]) -> dict[str, float]:
    """Return each question's probability, or raise CheckerError if anything looks wrong."""
    if not isinstance(body, dict) or not isinstance(body.get("model"), str):
        raise CheckerError("The response names no model.")
    answers = body.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(questions):
        raise CheckerError("The response's answers don't match the questions asked.")
    probabilities = {}
    for name in questions:
        value = answers[name].get("noul") if isinstance(answers[name], dict) else None
        # bool is a kind of int in Python, so it must be refused by name.
        if isinstance(value, bool) or not isinstance(value, int | float) or not 0 <= value <= 1:
            raise CheckerError(f"The answer to {name} is not a probability.")
        probabilities[name] = float(value)
    return probabilities
