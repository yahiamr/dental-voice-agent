"""Failure tests. Stage: 1.

What this file is:
    Breaks things on purpose during a call and checks the caller is told the truth.

What it contains:
    - Model timeout.
    - speak failing after a commit.
    - A graph exception, then a clean next turn.
    - A commit error with and without a record.
    - A slot taken between prepare and commit.

How to understand it:
    Every test asserts that the words spoken match the calendar, including the closing sentence.

Used by:
    pytest
"""

from datetime import date

import pytest
from tests.conftest import TEST_NOW, build_test_deps
from tests.fakes import FailingCalendar, FakeSpeech, ScriptedModel, text_reply, tool_reply

from dental_agent.booking.seed import seed
from dental_agent.graph import nodes
from dental_agent.models import EndReason, Offer, Outcome, PendingAction
from dental_agent.session import run_call
from dental_agent.speech.base import SpeechResult

PHONE = "+34600999999"
SAVE = ("save_details", {"name": "Ana Ruiz", "service": "first_visit"})
FIND = (
    "find_slots",
    {"service": "first_visit", "date_from": "2026-11-16", "date_to": "2026-11-20"},
)
PREPARE = ("prepare_booking", {"slot_id": "s1"})
ERROR = SpeechResult(kind="error")
BOOKING = [tool_reply(SAVE, FIND), tool_reply(PREPARE)]


@pytest.fixture
def world(tmp_path, clock, clinic):
    """Build a failing-capable calendar and a function that runs a call on it."""
    calendar = FailingCalendar(tmp_path / "calendar.sqlite")
    seed(calendar, clock, clinic)

    def run(speech, script=()):
        # Copied, because a model uses up its script.
        fresh = [item.model_copy(deep=True) for item in script]
        model = ScriptedModel(script=[*fresh, text_reply("Summary.")])
        deps = build_test_deps(tmp_path, clinic, clock, calendar, model=model, speech=speech)
        return run_call(deps, PHONE)

    return calendar, run


def spoken_or_printed(record):
    """Return everything the caller was told."""
    return " ".join(item["text"] for item in record.transcript if item["role"] == "bot")


def test_a_model_timeout_is_told_as_trouble(world, clinic):
    _, run = world
    # The model has one response, which the first turn uses, so the second turn fails.
    record = run(FakeSpeech(["Hi", "Hello"]))
    assert clinic.line("trouble") in spoken_or_printed(record)
    assert record.actions == []


def test_speech_failing_right_after_a_commit_still_tells_the_truth(world):
    calendar, run = world
    speech = FakeSpeech(["Book me", "yes"])
    # The greeting and confirmation question work, and then speech stops.
    speech.fail_after = 2
    record = run(speech, BOOKING)
    stored = calendar.actions_for_call(record.conversation_id)
    assert len(stored) == 1
    assert stored[0].result_text in spoken_or_printed(record)
    assert record.outcome is Outcome.COMPLETED


def test_a_booking_is_closed_with_what_the_calendar_shows(world):
    calendar, run = world
    record = run(FakeSpeech(["Book me", "yes", "Bye"]), [*BOOKING, tool_reply(("end_call", {}))])
    stored = calendar.actions_for_call(record.conversation_id)[0]
    assert record.transcript[-1]["text"] == f"To confirm: {stored.result_text} Goodbye."


def test_a_graph_exception_then_a_normal_turn(world, clinic, monkeypatch):
    calendar, run = world
    original = calendar.actions_for_call
    calls = {"count": 0}

    def fails_once(conversation_id):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("read failed")
        return original(conversation_id)

    monkeypatch.setattr(calendar, "actions_for_call", fails_once)
    record = run(FakeSpeech(["Hi", "Hello"]), [text_reply("Second reply")])
    said = spoken_or_printed(record)
    assert clinic.line("trouble") in said
    assert "Nothing has been changed on this call so far." in said
    assert "Second reply" in said


def test_a_graph_exception_clears_a_waiting_confirmation(world, monkeypatch):
    # The question was heard, but the turn answering it fails. The next yes must not commit.
    _, run = world
    original = nodes.detect
    calls = {"count": 0}

    def fails_once(*args):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("guard failed")
        return original(*args)

    def break_guard():
        calls["count"] = 0
        monkeypatch.setattr(nodes, "detect", fails_once)

    speech = FakeSpeech(["Book me", break_guard, "yes", "yes"])
    record = run(speech, [*BOOKING, text_reply("What can I do for you?")])
    assert record.actions == []


def test_an_unreadable_calendar_after_a_graph_error_ends_the_call(world, clinic):
    calendar, run = world

    def break_reads():
        calendar.fail_reads = True

    record = run(FakeSpeech([break_reads, "Hi"]))
    assert record.end_reason is EndReason.CALENDAR_FAILED
    assert record.outcome is Outcome.FAILED
    assert clinic.line("call_clinic") in spoken_or_printed(record)
    assert clinic.line("emergency") in spoken_or_printed(record)


def test_an_unreadable_calendar_at_finish_replaces_the_closing(world):
    calendar, run = world

    def break_reads():
        calendar.fail_reads = True

    record = run(FakeSpeech(["Book me", "yes", break_reads, SpeechResult(kind="closed")]), BOOKING)
    # A caller who hangs up hears no closing sentence, but the outcome still changes.
    assert record.outcome is Outcome.FAILED
    assert len(record.actions) == 1


def test_a_commit_error_with_a_record_tells_the_caller_it_worked(world):
    calendar, run = world

    def break_write():
        calendar.fail_book = "after"

    record = run(FakeSpeech(["Book me", break_write, "yes"]), BOOKING)
    assert "You're booked" in spoken_or_printed(record)
    assert record.outcome is Outcome.COMPLETED


def test_a_commit_error_with_no_record_sends_the_caller_to_the_clinic(world, clinic):
    calendar, run = world

    def break_write():
        calendar.fail_book = "before"

    record = run(FakeSpeech(["Book me", break_write, "yes"]), BOOKING)
    said = spoken_or_printed(record)
    assert "You're booked" not in said
    assert clinic.line("commit_unknown") in said
    assert record.outcome is Outcome.FAILED


def test_sustained_speech_failure_prints_and_ends(world, capsys):
    _, run = world
    speech = FakeSpeech(["Hi", "Hello", "never"])
    speech.fail_speak = True
    record = run(speech, [text_reply("One"), text_reply("Two")])
    assert record.end_reason is EndReason.FAILURES
    assert "One" in capsys.readouterr().out
    assert all(item.get("printed_only", True) for item in record.transcript)


def test_transcription_errors_end_the_call_with_no_booking(world):
    _, run = world
    record = run(FakeSpeech(["Book me", ERROR, ERROR, "yes"]), BOOKING)
    assert record.end_reason is EndReason.SILENCE
    assert record.actions == []


def test_a_failed_repeat_of_the_confirmation_cannot_be_confirmed(world):
    _, run = world
    speech = FakeSpeech(["Book me", ERROR, "yes"])
    speech.fail_after = 2  # The first confirmation question works, and its repeat fails.
    record = run(speech, [*BOOKING, text_reply("Sorry.")])
    assert record.actions == []


def test_a_slot_taken_between_prepare_and_commit(world, clinic):
    calendar, run = world

    def someone_else_books():
        # The caller's offer is the earliest free slot, so take that one.
        week = (date(2026, 11, 16), date(2026, 11, 20))
        slot = calendar.free_slots(*week, None, None, 1, TEST_NOW)[0]
        offer = Offer(slot_id="s1", db_slot_id=slot.db_slot_id, service="check_up", spoken="t")
        action = PendingAction(action_id="other", kind="book", offer=offer)
        calendar.book(action, "c-0", "+34600888888", "Luis Mora", "check_up", "t", "B", TEST_NOW)

    record = run(FakeSpeech(["Book me", someone_else_books, "yes"]), BOOKING)
    said = spoken_or_printed(record)
    assert clinic.line("slot_taken") in said
    assert "You're booked" not in said
    assert record.actions == []
