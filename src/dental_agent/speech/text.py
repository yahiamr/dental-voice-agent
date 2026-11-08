"""Keyboard input and printed output. Stage: 1.

What this file is:
    The text backend, used by dental-agent chat.

What it contains:
    - TextSpeech, with backend_name "text" for the trace.

How to understand it:
    It lets you run a full call with no audio or Deepgram key. An empty line counts as
    no_speech, so silence handling can be tried at the keyboard.

Uses:
    speech/base.py

Used by:
    factories.py
"""

from dental_agent.speech.base import SpeakResult, SpeechResult


class TextSpeech:
    """Reads the caller from the keyboard and prints replies to the screen."""

    uses_tone = False  # There is no tone in text mode.
    backend_name = "text"  # Written to the trace, so a text call is not logged as voice.

    def transcribe(self, expect_yes: bool) -> SpeechResult:
        """Read one line. expect_yes changes nothing here, because Enter ends the turn."""
        try:
            text = input("You: ").strip()
        except EOFError:
            # Ctrl-D, or piped input running out, means the caller has left.
            return SpeechResult(kind="closed", provider="text")
        if not text:
            return SpeechResult(kind="no_speech", provider="text")
        return SpeechResult(kind="speech", text=text, confidence=1.0, provider="text")

    def speak(self, text: str, asks_confirmation: bool = False) -> SpeakResult:
        """Print one reply."""
        print(f"Bot: {text}")
        return SpeakResult(ok=True)
