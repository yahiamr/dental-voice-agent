"""Tests for the voice call display. Stage: 4.

What this file is:
    Tests speech/display.py, which shows a voice call in the terminal: the transcript of both
    sides, and one status line for listening (with the words heard so far), thinking and
    speaking.

What it contains:
    - Without a terminal: only the transcript lines, and no status codes.
    - With a terminal: the status line, the live words, the spinner, the width limit, and a
      transcript line that clears and then redraws the status line.
    - close: clears the line and makes later calls do nothing.

How to understand it:
    The display writes to a StringIO here, with animate=True standing in for a terminal and
    interval=None so no thread runs. Tests call tick() to move the spinner on by hand.

Used by:
    pytest
"""

import io
import time

from dental_agent.speech.display import CLEAR, CallDisplay


def display(animate=True, width=60):
    stream = io.StringIO()
    return CallDisplay(stream=stream, animate=animate, width=width, interval=None), stream


def last_status(stream):
    """The text after the last clear code, which is what the terminal shows on the status line."""
    return stream.getvalue().rsplit(CLEAR, 1)[-1]


# ---------- Without a terminal ----------


def test_without_a_terminal_only_the_transcript_is_printed():
    shown, stream = display(animate=False)
    shown.listening()
    shown.words("I'd like")
    shown.caller("I'd like a cleaning.")
    shown.thinking()
    shown.bot("Which day suits you?")
    shown.speaking()
    shown.close()
    assert stream.getvalue() == "You: I'd like a cleaning.\nBot: Which day suits you?\n"


# ---------- With a terminal ----------


def test_listening_shows_the_words_heard_so_far():
    shown, stream = display()
    shown.listening()
    assert "Listening" in last_status(stream)
    shown.words("I'd like a cleaning")
    assert "Listening: I'd like a cleaning" in last_status(stream)


def test_long_words_keep_their_end_and_fit_the_width():
    shown, stream = display(width=30)
    shown.listening()
    shown.words("I'd like to move my check-up to Tuesday the 17th of November")
    status = last_status(stream)
    visible = status.replace("\x1b[2m", "").replace("\x1b[0m", "")
    assert len(visible) <= 29
    assert visible.endswith("November")


def test_thinking_shows_a_spinner_that_moves_on_each_tick():
    shown, stream = display()
    shown.thinking()
    first = last_status(stream)
    shown.tick()
    second = last_status(stream)
    assert "Thinking" in first and "Thinking" in second
    assert first != second


def test_a_tick_redraws_only_while_thinking():
    shown, stream = display()
    shown.speaking()
    before = stream.getvalue()
    shown.tick()
    assert stream.getvalue() == before


def test_speaking_has_its_own_status():
    shown, stream = display()
    shown.speaking()
    assert "Speaking" in last_status(stream)


def test_a_transcript_line_clears_the_status_then_redraws_it():
    shown, stream = display()
    shown.thinking()
    shown.bot("One moment.")
    text = stream.getvalue()
    # The line is printed on a cleared line, and the status comes back below it.
    assert CLEAR + "Bot: One moment.\n" in text
    assert "Thinking" in text.rsplit("Bot: One moment.\n", 1)[1]


def test_new_words_start_fresh_on_each_turn():
    shown, stream = display()
    shown.listening()
    shown.words("yes")
    shown.caller("yes")
    shown.listening()
    assert "yes" not in last_status(stream)


# ---------- close ----------


def test_close_clears_the_status_line_and_ignores_later_calls():
    shown, stream = display()
    shown.thinking()
    shown.close()
    assert stream.getvalue().endswith(CLEAR)
    after = stream.getvalue()
    shown.bot("Late line.")
    shown.thinking()
    shown.tick()
    assert stream.getvalue() == after


def test_a_running_spinner_stops_on_close():
    stream = io.StringIO()
    shown = CallDisplay(stream=stream, animate=True, width=60, interval=0.01)
    shown.thinking()
    shown.close()
    after = stream.getvalue()
    time.sleep(0.05)
    assert stream.getvalue() == after
