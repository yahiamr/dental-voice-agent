"""The checker chain: tries each backend in order. Stage: checker.

What this file is:
    The only checker object the rest of the bot uses.

What it contains:
    - CheckerChain(backends, time_limits, clock, second_opinion): ask() returns the first
      answer, or None.
    - MIN_TIME_S: below this much time left, nothing is tried.
    - MAX_STRIKES: failures in a row that switch a backend off for the rest of the call.

How to understand it:
    ask() tries each backend that isn't switched off in this call, in order. Each attempt gets
    its own limit or the time left minus the time already spent, whichever is less. When under
    MIN_TIME_S of the turn is left, it stops. A backend's own short limit never stops the chain,
    so the fallback is still tried.

    On any exception it tries the next backend. The failure is a strike against the backend
    unless it timed out (CheckerError.timed_out) with a limit cut short by the time left: then
    the turn ran short, and the timeout says nothing sure about the backend. Any other failure,
    such as a bad key, is a strike however much time it had. MAX_STRIKES strikes in a row
    switch the backend off for the rest of the call, and any answer clears its strikes. So one
    slow answer or one rate-limit error costs one fallback, and an outage costs at most
    MAX_STRIKES slow attempts per call. The strike counts are the only state, and start_call()
    clears them at the start of each call.

    ask() never raises. When nothing answers it returns None, and the fixed rules decide alone.
    last_attempts records each attempt of the last ask (name, ok, seconds, and for a failure
    the error and whether it was a strike) for the trace.
    An error is the exception type plus at most 80 characters, so no response body or checked
    text reaches the trace.

    Library timeouts apply to each blocking step, so an attempt can run a little over its
    limit. The limits are targets, and the trace records the real time.

    second_opinion maps a question to its grey zone, a band of probabilities around the
    cut-off, such as {"medical_advice": (0.3, 0.5), "claims_change": (0.5, 0.65)}. When an
    answer's probability for that question falls in the band, the chain asks the next working
    backend that one question alone, and its answer replaces the first. The advice band sits
    just under the cut-off: a reply judging the caller's own symptom scored 0.41 on a live call
    and passed. The claims band sits just over it: correct notice-period refusals scored 0.51
    to 0.59 and were replaced. Only Jev gives probabilities, so only a Jev answer can be in a
    grey zone. A failed or unaffordable second
    opinion leaves the first answer as it was, and counts as a strike like any attempt.

Uses:
    checker/base.py

Used by:
    factories.py, deps.py, session.py, graph/nodes.py (from Step 3)
"""

import time
from collections.abc import Callable

from dental_agent.checker.base import CheckAnswer, Checker, CheckerError, Role

MIN_TIME_S = 0.2  # Too little time for any backend to answer.
MAX_STRIKES = 2  # Failures in a row before a backend is switched off for the call.
ERROR_CHARS = 80  # The longest error message kept in the trace.


class CheckerChain:
    """Tries each checker backend in order, and never raises."""

    def __init__(
        self,
        backends: list[Checker],
        time_limits: dict[str, float],
        clock: Callable[[], float] = time.monotonic,
        second_opinion: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        missing = [backend.name for backend in backends if backend.name not in time_limits]
        if missing:
            raise ValueError(f"No time limit for checker backends: {', '.join(missing)}")
        self.backends = backends
        self.time_limits = time_limits
        self._clock = clock
        # Question to its grey zone, the probabilities from low up to high that get a second
        # opinion from the next backend.
        self.second_opinion = second_opinion or {}
        # Failures in a row, per backend. MAX_STRIKES switches a backend off for the call.
        self._strikes: dict[str, int] = {}
        self.last_attempts: list[dict] = []

    @property
    def names(self) -> list[str]:
        """The backends this chain holds, in order."""
        return [backend.name for backend in self.backends]

    def start_call(self) -> None:
        """Forget the strikes, and so which backends were switched off, at the start of a call."""
        self._strikes.clear()

    def ask(
        self, text: str, questions: dict[str, str], role: Role, time_left_s: float
    ) -> CheckAnswer | None:
        """Return the first backend's answer, or None if none answered. Never raises."""
        self.last_attempts = []
        start = self._clock()
        for index, backend in enumerate(self.backends):
            if self._strikes.get(backend.name, 0) >= MAX_STRIKES:
                continue  # Switched off for the rest of this call.
            remaining = time_left_s - (self._clock() - start)
            # The minimum is for the time left in the turn. A backend's own short limit is used
            # as given, so it can't stop the backends after it from being tried.
            if remaining < MIN_TIME_S:
                break
            answer = self._attempt(backend, text, questions, role, remaining)
            if answer is None:
                continue
            answer = self._second_opinions(
                answer, self.backends[index + 1 :], text, questions, role, time_left_s, start
            )
            return answer.model_copy(update={"seconds": self._clock() - start})
        return None

    def _second_opinions(
        self,
        answer: CheckAnswer,
        later: list[Checker],
        text: str,
        questions: dict[str, str],
        role: Role,
        time_left_s: float,
        start: float,
    ) -> CheckAnswer:
        """Ask a later backend about each answer in its grey zone, and let its answer stand."""
        answers = dict(answer.answers)
        opinions: dict[str, bool] = {}
        for question, (low, high) in self.second_opinion.items():
            probability = answer.probabilities.get(question)
            if question not in questions or probability is None:
                continue
            if not low <= probability < high:
                continue
            for backend in later:
                if self._strikes.get(backend.name, 0) >= MAX_STRIKES:
                    continue
                remaining = time_left_s - (self._clock() - start)
                if remaining < MIN_TIME_S:
                    break
                second = self._attempt(
                    backend, text, {question: questions[question]}, role, remaining, question
                )
                if second is not None:
                    opinions[question] = answers[question] = second.answers[question]
                    break
        if not opinions:
            return answer
        return answer.model_copy(update={"answers": answers, "second_opinions": opinions})

    def _attempt(
        self,
        backend: Checker,
        text: str,
        questions: dict[str, str],
        role: Role,
        remaining: float,
        second_opinion: str | None = None,
    ) -> CheckAnswer | None:
        """Ask one backend, record the attempt and its strike, and return its answer or None."""
        limit = min(self.time_limits[backend.name], remaining)
        cut_short = limit < self.time_limits[backend.name]
        attempt_start = self._clock()
        record: dict = {"name": backend.name}
        if second_opinion:
            record["second_opinion"] = second_opinion
        try:
            answer = backend.ask(text, questions, role, limit)
        except Exception as error:  # Any failure means try the next backend.
            # Running out of a limit the turn cut short is not a fair test of the backend.
            # Any other failure, such as a bad key, is a strike however much time it had.
            timed_out = isinstance(error, CheckerError) and error.timed_out
            strike = not (cut_short and timed_out)
            if strike:
                self._strikes[backend.name] = self._strikes.get(backend.name, 0) + 1
            self.last_attempts.append(
                record
                | {
                    "ok": False,
                    "seconds": self._clock() - attempt_start,
                    "error": f"{type(error).__name__}: {str(error)[:ERROR_CHARS]}",
                    "strike": strike,
                }
            )
            return None
        self._strikes[backend.name] = 0  # Strikes must be in a row.
        self.last_attempts.append(record | {"ok": True, "seconds": self._clock() - attempt_start})
        return answer
