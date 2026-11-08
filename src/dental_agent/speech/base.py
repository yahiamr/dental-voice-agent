"""The Speech interface. Stage: 1.

What this file is:
    The contract every speech backend follows.

What it contains:
    - Speech, a protocol with transcribe(expect_yes) and speak(text, asks_confirmation=False).
    - SpeechResult, whose kind is speech, no_speech, error or closed.
    - SpeakResult, with ok and timings.

How to understand it:
    The session calls only these two methods, so text, Deepgram, local speech and the test
    drivers all fit the same loop. expect_yes=True waits longer for the caller to finish, so
    "yes... but Thursday" is not cut in half.

Used by:
    session.py, every backend, tests/fakes.py, evals/driver.py
"""

from typing import Literal, Protocol

from pydantic import BaseModel, Field


class SpeechResult(BaseModel):
    """What transcribe heard."""

    # speech: words were heard. no_speech: silence. error: the backend failed.
    # closed: the input ended, as when the caller hangs up.
    kind: Literal["speech", "no_speech", "error", "closed"]
    text: str = ""
    confidence: float = 1.0
    # Keys are speech_end_at and transcript_at, used to measure delay.
    timings: dict[str, float] = Field(default_factory=dict)
    provider: str = ""


class SpeakResult(BaseModel):
    """Whether speak worked."""

    ok: bool
    # Keys are first_audio_at and audio_end_at.
    timings: dict[str, float] = Field(default_factory=dict)


class Speech(Protocol):
    """What every speech backend must offer."""

    # True for voice backends, so the greeting can mention the tone.
    uses_tone: bool

    def transcribe(self, expect_yes: bool) -> SpeechResult:
        """Listen for one reply. expect_yes waits longer for the caller to finish."""
        ...

    def speak(self, text: str, asks_confirmation: bool = False) -> SpeakResult:
        """Say one reply. asks_confirmation lets a test driver see confirmation questions."""
        ...
