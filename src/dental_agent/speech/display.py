"""Shows a voice call in the terminal. Stage: 4.

What this file is:
    The screen side of a voice call: what each side said, and what the bot is doing now.

What it contains:
    - CLEAR: the code that returns to the start of the line and erases it.
    - CallDisplay: caller(text) and bot(text) print a transcript line; listening(),
      words(text), thinking() and speaking() set the status line; tick() moves the spinner on;
      close() clears the line and stops.

How to understand it:
    A voice call otherwise prints nothing while it runs, so a viewer cannot follow it without
    the sound. The display keeps one status line at the bottom, rewritten in place:
    "● Listening: <the words heard so far>", "⠋ Thinking…" with a spinner, or "♪ Speaking…".
    A transcript line ("You: …" or "Bot: …") clears the status line, prints itself, and draws
    the status line again below it, so the two never mix.

    The spinner runs on its own thread, redrawing every interval seconds, and only while the
    bot is thinking. Printing takes microseconds and the audio runs on its own threads, so the
    display adds no delay, and the trace's timings never include it.

    When the output is not a terminal, as when it is piped to a file, there is no status line
    and no thread: only the transcript lines are printed, so the output stays clean. A lock
    keeps the spinner thread and the call's own thread from writing at the same time.

Uses:
    The standard library only.

Used by:
    speech/deepgram.py, factories.py
"""

import shutil
import sys
import threading
from typing import TextIO

CLEAR = "\r\x1b[2K"  # Back to the start of the line, then erase it.
DIM = "\x1b[2m"
RESET = "\x1b[0m"
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
INTERVAL_S = 0.1  # How often the spinner moves on.


class CallDisplay:
    """The transcript of a voice call, and one status line that is rewritten in place."""

    def __init__(
        self,
        stream: TextIO | None = None,
        animate: bool | None = None,
        width: int | None = None,
        interval: float | None = INTERVAL_S,
    ):
        self.stream = stream if stream is not None else sys.stdout
        # Animate only on a terminal, so piped output holds just the transcript.
        self.animate = self.stream.isatty() if animate is None else animate
        self._width = width
        self._interval = interval
        self._lock = threading.Lock()
        self._state: str | None = None  # listening, thinking or speaking.
        self._words = ""
        self._frame = 0
        self._closed = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---------- Transcript lines ----------

    def caller(self, text: str) -> None:
        """Print what the caller said."""
        self._line(f"You: {text}")

    def bot(self, text: str) -> None:
        """Print what the bot is about to say."""
        self._line(f"Bot: {text}")

    # ---------- The status line ----------

    def listening(self) -> None:
        """Show that the microphone is open. Words from the last turn are dropped."""
        self._set("listening", words="")

    def words(self, text: str) -> None:
        """Show the words heard so far in this turn."""
        self._set("listening", words=text)

    def thinking(self) -> None:
        """Show the spinner while the agent works."""
        self._set("thinking")

    def speaking(self) -> None:
        """Show that the bot is speaking."""
        self._set("speaking")

    def tick(self) -> None:
        """Move the spinner on one frame. Does nothing unless the bot is thinking."""
        with self._lock:
            if self._closed or self._state != "thinking":
                return
            self._frame = (self._frame + 1) % len(SPINNER)
            self._draw()

    def close(self) -> None:
        """Clear the status line and stop the spinner. Later calls do nothing."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._stop.set()
            if self.animate:
                self.stream.write(CLEAR)
                self.stream.flush()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=1)

    # ---------- Inside ----------

    def _line(self, text: str) -> None:
        """Print one transcript line above the status line."""
        with self._lock:
            if self._closed:
                return
            prefix = CLEAR if self.animate else ""
            self.stream.write(f"{prefix}{text}\n")
            if self.animate and self._state is not None:
                self._draw()
            self.stream.flush()

    def _set(self, state: str, words: str | None = None) -> None:
        """Change the status, and start the spinner thread the first time."""
        if not self.animate:
            return
        with self._lock:
            if self._closed:
                return
            self._state = state
            if words is not None:
                self._words = words
            self._draw()
        self._start_thread()

    def _status(self) -> str:
        """The status line's text, before it is fitted to the width."""
        if self._state == "listening":
            return f"● Listening: {self._words}" if self._words else "● Listening…"
        if self._state == "thinking":
            return f"{SPINNER[self._frame]} Thinking…"
        return "♪ Speaking…"

    def _draw(self) -> None:
        """Rewrite the status line. The caller holds the lock."""
        text = _fit(self._status(), self._columns())
        self.stream.write(f"{CLEAR}{DIM}{text}{RESET}")
        self.stream.flush()

    def _columns(self) -> int:
        """The usable width: one less than the terminal, so the line never wraps."""
        width = self._width or shutil.get_terminal_size((80, 24)).columns
        return max(width - 1, 10)

    def _start_thread(self) -> None:
        """Start the spinner thread once, if an interval is set."""
        if self._interval is None or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._spin, name="call-display", daemon=True)
        self._thread.start()

    def _spin(self) -> None:
        """Move the spinner on until the display is closed."""
        while not self._stop.wait(self._interval):
            self.tick()


def _fit(text: str, columns: int) -> str:
    """Fit a status line to the width, keeping the end, where the newest words are."""
    if len(text) <= columns:
        return text
    head, _, words = text.partition(": ")
    if not words:
        return text[: columns - 1] + "…"
    room = columns - len(head) - 3  # The head, ": " and the "…" before the kept words.
    return f"{head}: …{words[-room:]}" if room > 0 else text[:columns]
