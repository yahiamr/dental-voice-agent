"""A Speech object that plays the caller. Stage: 4.

What this file is:
    Feeds scripted caller lines into the normal session loop.

What it contains:
    - ScenarioSpeech(lines, auto_yes=True, before_yes=None): transcribe, speak, spoken and
      ran_out.
    - SPELLED: how a spelled-back name is recognised.

How to understand it:
    It looks like a speech backend to the session, so scenarios run the same code as a real
    call. The model's wording and question order change between runs, so a fixed script would
    drift out of step. Two answers are therefore given by the driver itself, without using a
    scripted line:
    - "yes", on the turn after a confirmation question, when auto_yes is true. With auto_yes
      false, the next scripted line answers the question instead, which is how "yes, but make it
      Thursday" is tested.
    - "Yes, that's right.", after the bot spells a name back letter by letter, as the prompt
      tells it to for a new caller.
    before_yes runs once, with the question, just before the first automatic yes. The slot-taken
    scenario uses it to give the slot to someone else at that moment.
    Running out of lines counts as the caller hanging up, and ran_out records it, because a
    scenario whose lines run out did not reach its end.

Uses:
    speech/base.py

Used by:
    evals/evaluate.py, tests/test_scenarios.py
"""

import re
from collections.abc import Callable

from dental_agent.speech.base import SpeakResult, SpeechResult

# Two or more capital letters joined by hyphens, as in "A-N-A, R-U-I-Z".
SPELLED = re.compile(r"\b[A-ZÁÉÍÓÚÑ](?:-[A-ZÁÉÍÓÚÑ])+\b")


class ScenarioSpeech:
    """Plays a scripted caller and records what the bot said."""

    uses_tone = False
    backend_name = "scenario"

    def __init__(
        self,
        lines: list[str],
        auto_yes: bool = True,
        before_yes: Callable[[str], object] | None = None,
    ) -> None:
        self.lines = list(lines)
        self.auto_yes = auto_yes
        self.before_yes = before_yes
        self.spoken: list[str] = []
        self.ran_out = False
        self._answer = ""  # A reply the driver gives itself on the next turn.
        self._question = ""  # The confirmation question the next "yes" answers.

    def transcribe(self, expect_yes: bool) -> SpeechResult:
        """Return the driver's own answer if one is due, else the next scripted line."""
        if self._answer:
            text, self._answer = self._answer, ""
            if text == "yes" and self.before_yes is not None:
                hook, self.before_yes = self.before_yes, None
                hook(self._question)
            return SpeechResult(kind="speech", text=text, provider="scenario")
        if not self.lines:
            self.ran_out = True
            return SpeechResult(kind="closed")
        return SpeechResult(kind="speech", text=self.lines.pop(0), provider="scenario")

    def speak(self, text: str, asks_confirmation: bool = False) -> SpeakResult:
        """Record the line, and note whether the driver must answer it itself."""
        self.spoken.append(text)
        if asks_confirmation and self.auto_yes:
            self._answer, self._question = "yes", text
        elif SPELLED.search(text) and text.rstrip().endswith("?"):
            self._answer = "Yes, that's right."
        return SpeakResult(ok=True)
