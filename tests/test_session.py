"""Tests for the session loop. Stage: 1.

What this file is:
    Runs whole calls with FakeSpeech.

What it contains:
    - Silence and transcription errors, with and without a confirmation question waiting.
    - trouble_count reaching 2.
    - A low-confidence reply that cannot commit.
    - A failed confirmation question.
    - Each row of the Endings table.

Used by:
    pytest
"""

import json
import time

import pytest
from tests.conftest import PAST_NEW, add_past_first_visit, build_test_deps
from tests.fakes import (
    FailingCalendar,
    FakeSpeech,
    ScriptedModel,
    text_reply,
    tool_reply,
)

from dental_agent.booking.seed import seed
from dental_agent.checker import CheckerChain
from dental_agent.models import EndReason, Outcome
from dental_agent.session import run_call
from dental_agent.speech.base import SpeechResult

PHONE = "+34600999999"
SAVE = ("save_details", {"name": "Ana Ruiz", "service": "first_visit"})
FIND = (
    "find_slots",
    {"service": "first_visit", "date_from": "2026-11-16", "date_to": "2026-11-20"},
)
PREPARE = ("prepare_booking", {"slot_id": "s1"})
END = ("end_call", {})
SILENCE = SpeechResult(kind="no_speech")
ERROR = SpeechResult(kind="error")


def unsure(text):
    """Build a reply the speech service was not sure about."""
    return SpeechResult(kind="speech", text=text, confidence=0.2)


def booking():
    """Return the model responses that prepare a booking in one turn."""
    return [tool_reply(SAVE, FIND), tool_reply(PREPARE)]


@pytest.fixture
def run(tmp_path, clinic, clock, calendar):
    """Run a whole call, returning the record, the speech object and the deps."""

    def go(
        inputs,
        script=(),
        speech=None,
        calendar_override=None,
        on_call=None,
        phone=PHONE,
        checker=None,
    ):
        speech = speech or FakeSpeech(inputs)
        model = ScriptedModel(script=[*script, text_reply("Summary.")], on_call=on_call)
        deps = build_test_deps(
            tmp_path,
            clinic,
            clock,
            calendar_override or calendar,
            model=model,
            speech=speech,
            checker=checker,
        )
        return run_call(deps, phone), speech, deps

    return go


@pytest.fixture
def failing_calendar(tmp_path, clock, clinic):
    """A seeded calendar that can fail on request."""
    calendar = FailingCalendar(tmp_path / "failing.sqlite")
    seed(calendar, clock, clinic)
    return calendar


def test_a_call_opens_with_the_greeting(run, clinic):
    _, speech, _ = run([])
    assert speech.spoken[0] == clinic.line("greeting")


def test_a_voice_greeting_mentions_the_tone(run):
    speech = FakeSpeech([])
    speech.uses_tone = True
    _, spoken, _ = run([], speech=speech)
    assert spoken.spoken[0].endswith("Please speak after the tone.")


def test_the_conversation_id_has_the_expected_shape(run):
    record, _, _ = run([])
    assert record.conversation_id.startswith("c-20261109-100000-")
    assert len(record.conversation_id.split("-")[3]) == 4


def test_a_booking_call_from_start_to_finish(run):
    record, speech, deps = run(["Book me", "yes", "Bye"], [*booking(), tool_reply(END)])
    assert len(speech.confirmations) == 1
    assert record.outcome is Outcome.COMPLETED
    assert record.end_reason is EndReason.DONE
    assert [item["kind"] for item in record.actions] == ["book"]
    assert speech.spoken[-1].startswith("To confirm: You're booked")
    assert deps.graph.calendar.actions_for_call(record.conversation_id)


def test_the_transcript_holds_everything_said(run, clinic):
    record, _, _ = run(["Hi"], [text_reply("Hello there")])
    texts = [item["text"] for item in record.transcript]
    assert texts[0] == clinic.line("greeting")
    assert "Hi" in texts
    assert "Hello there" in texts


def test_the_conversation_file_is_written_each_turn(run, tmp_path):
    record, _, _ = run(["Hi"], [text_reply("Hello")])
    path = tmp_path / "runs" / "conversations" / f"{record.conversation_id}.json"
    data = json.loads(path.read_text())
    assert data["outcome"] == "abandoned"
    # The patient type is known from the number, even when the caller said nothing else.
    assert data["summary"]["key_points"] == ["Patient type: new", "No changes were made."]


def test_the_trace_file_starts_with_a_start_line(run, tmp_path):
    record, _, _ = run(["Hi"], [text_reply("Hello")])
    path = tmp_path / "runs" / "traces" / f"{record.conversation_id}.jsonl"
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines[0]["kind"] == "start"
    assert lines[0]["llm_model"]
    assert any(line.get("turn") == 1 for line in lines)


def test_the_model_adds_a_summary_note(run):
    record, _, _ = run(["Hi"], [text_reply("Hello")])
    assert record.summary.note == "Summary."


def test_the_first_turn_expects_no_yes(run):
    _, speech, _ = run(["Book me", "yes"], booking())
    assert speech.expect_yes_flags[0] is False
    assert speech.expect_yes_flags[1] is True


def test_silence_with_nothing_waiting_asks_if_the_caller_is_there(run, clinic):
    _, speech, _ = run([SILENCE, "Hi"], [text_reply("Hello")])
    assert clinic.line("still_there") in speech.spoken


def test_a_transcription_error_asks_again(run, clinic):
    _, speech, _ = run([ERROR, "Hi"], [text_reply("Hello")])
    assert clinic.line("not_caught") in speech.spoken


def test_silence_does_not_count_as_a_turn(run):
    _, _, deps = run([SILENCE, "Hi"], [text_reply("Hello")])
    assert deps.graph.model.seen  # The model ran once, for the one real turn.
    assert len(deps.graph.model.seen) == 2  # One turn, plus the summary sentence.


def test_silence_with_a_confirmation_waiting_repeats_it(run):
    _, speech, _ = run(["Book me", SILENCE, "yes", "Bye"], [*booking(), tool_reply(END)])
    assert len(speech.confirmations) == 2
    assert speech.confirmations[0] == speech.confirmations[1]


def test_a_yes_after_a_repeated_confirmation_commits(run):
    record, _, _ = run(["Book me", SILENCE, "yes", "Bye"], [*booking(), tool_reply(END)])
    assert [item["kind"] for item in record.actions] == ["book"]


def test_a_yes_to_are_you_still_there_cannot_commit(run):
    # After silence with nothing waiting, a yes answers the wrong question.
    record, _, _ = run([SILENCE, "yes"], [text_reply("How can I help?")])
    assert record.actions == []


def test_a_transcription_error_with_a_confirmation_waiting_repeats_it(run):
    _, speech, _ = run(["Book me", ERROR], booking())
    assert len(speech.confirmations) == 2


def test_a_low_confidence_yes_repeats_the_confirmation_and_cannot_commit(run):
    record, speech, _ = run(["Book me", unsure("yes")], booking())
    assert len(speech.confirmations) == 2
    assert record.actions == []


def test_low_confidence_counts_only_while_a_yes_is_expected(run):
    _, speech, _ = run([unsure("Hi")], [text_reply("Hello")])
    assert "Hello" in speech.spoken


def test_two_silences_in_a_row_end_the_call(run, clinic):
    record, speech, _ = run([SILENCE, SILENCE, "never heard"])
    assert record.end_reason is EndReason.SILENCE
    assert record.outcome is Outcome.ABANDONED
    assert clinic.line("emergency") in speech.spoken
    assert speech.spoken[-1] == "Nothing was changed on this call. Goodbye."


def test_a_normal_reply_resets_the_trouble_count(run):
    record, _, _ = run(
        [SILENCE, "Hi", SILENCE, "Hello", "Bye"],
        [text_reply("One"), text_reply("Two"), tool_reply(END)],
    )
    assert record.end_reason is EndReason.DONE


def test_a_repeated_confirmation_does_not_reset_the_trouble_count(run):
    record, speech, _ = run(["Book me", SILENCE, SILENCE, "yes"], booking())
    assert record.end_reason is EndReason.SILENCE
    assert len(speech.confirmations) == 2
    assert record.actions == []


def test_two_model_failures_end_the_call(run, clinic):
    # No script is left for the model, so both turns fail.
    record, speech, _ = run(["Hi", "Hello", "never"], [])
    assert record.end_reason is EndReason.FAILURES
    assert record.outcome is Outcome.FAILED
    assert speech.spoken[-3:] == [
        clinic.line("call_clinic"),
        clinic.line("emergency"),
        "Nothing was changed on this call. Goodbye.",
    ]


def test_a_failed_confirmation_cannot_be_confirmed(run):
    speech = FakeSpeech(["Book me", "yes"])
    speech.fail_confirmations = True
    record, _, _ = run([], [*booking(), text_reply("Sorry about that.")], speech=speech)
    assert record.actions == []


def test_a_failed_confirmation_stays_in_history_as_printed(run):
    speech = FakeSpeech(["Book me"])
    speech.fail_confirmations = True
    record, _, _ = run([], booking(), speech=speech)
    printed = [item for item in record.transcript if item.get("printed_only")]
    assert printed[0]["text"].startswith("Just to confirm")


def test_the_turn_limit_ends_the_call(run, clinic, tmp_path):
    lines = ["Hi"] * 31
    replies = [text_reply(f"Reply {number}") for number in range(30)]
    record, speech, _ = run(lines, replies)
    assert record.end_reason is EndReason.TURN_LIMIT
    assert record.outcome is Outcome.FAILED
    assert clinic.line("cannot_finish") in speech.spoken
    assert clinic.line("emergency") in speech.spoken


def test_the_round_limit_row(run, clinic):
    facts = ("get_clinic_facts", {"topic": "prices"})
    record, speech, _ = run(["Hi"], [tool_reply(facts)] * 3)
    assert record.end_reason is EndReason.ROUND_LIMIT
    assert record.outcome is Outcome.FAILED
    assert speech.spoken[1:] == [
        clinic.line("cannot_finish"),
        clinic.line("call_clinic"),
        clinic.line("emergency"),
        "Nothing was changed on this call. Goodbye.",
    ]


def test_the_urgent_row(run, clinic):
    record, speech, _ = run(["he can't breathe"])
    assert record.end_reason is EndReason.URGENT
    assert record.outcome is Outcome.URGENT
    assert speech.spoken[-1] == clinic.line("urgent_message")


def test_the_done_row_with_no_action(run):
    record, speech, _ = run(["Bye"], [tool_reply(END)])
    assert record.end_reason is EndReason.DONE
    assert record.outcome is Outcome.NO_ACTION
    assert speech.spoken[-1] == "Nothing was changed on this call. Goodbye."


def test_the_caller_left_row_says_nothing(run, clinic):
    record, speech, _ = run([])
    assert record.end_reason is EndReason.CALLER_LEFT
    assert record.outcome is Outcome.ABANDONED
    assert speech.spoken == [clinic.line("greeting")]


def test_ctrl_c_counts_as_the_caller_leaving(run):
    record, _, _ = run([KeyboardInterrupt()])
    assert record.end_reason is EndReason.CALLER_LEFT


def test_a_caller_who_leaves_after_a_booking_is_completed(run):
    record, _, _ = run(["Book me", "yes"], booking())
    assert record.outcome is Outcome.COMPLETED
    assert record.end_reason is EndReason.CALLER_LEFT


def test_the_commit_failed_row(run, failing_calendar, clinic):
    failing_calendar.fail_book = "before"
    record, speech, _ = run(["Book me", "yes"], booking(), calendar_override=failing_calendar)
    assert record.end_reason is EndReason.COMMIT_FAILED
    assert record.outcome is Outcome.FAILED
    assert speech.spoken[-2:] == [clinic.line("commit_unknown"), clinic.line("emergency")]


def test_the_crash_row_prints_the_clinic_line(run, clinic, capsys):
    record, _, _ = run([RuntimeError("boom")])
    assert record.end_reason is EndReason.CRASH
    assert record.outcome is Outcome.FAILED
    printed = capsys.readouterr().out
    assert clinic.line("call_clinic") in printed
    assert clinic.line("emergency") in printed


def test_a_known_number_starts_as_the_existing_patient_on_record(run):
    record, _, _ = run(["Hi"], [text_reply("Hello Lucía")], phone="+34600000001")
    assert record.details.patient_type == "existing"
    assert record.details.name == "Lucía Moreno"


def test_a_new_number_starts_as_a_new_patient_with_no_name(run):
    record, _, _ = run(["Hi"], [text_reply("Hello")])
    assert record.details.patient_type == "new"
    assert record.details.name is None


def test_the_trace_names_the_speech_backend_really_used(run, tmp_path):
    import json

    speech = FakeSpeech(["Hi"])
    speech.backend_name = "fake"
    record, _, deps = run([], [text_reply("Hello")], speech=speech)
    trace = deps.graph.settings.runs_dir / "traces" / f"{record.conversation_id}.jsonl"
    start = json.loads(trace.read_text().splitlines()[0])
    assert start["speech_backend"] == "fake"


def test_a_past_first_visit_makes_the_caller_existing_at_start(run, calendar):
    add_past_first_visit(calendar)
    record, _, _ = run(["Hi"], [text_reply("Hello")], phone=PAST_NEW)
    assert record.details.patient_type == "existing"
    assert record.details.name == "Teo Lara"


def test_a_new_patient_on_record_keeps_their_name_and_stays_new(run):
    record, _, _ = run(["Hi"], [text_reply("Hello")], phone="+34600000007")
    assert record.details.patient_type == "new"
    assert record.details.name == "Nora Blanco"


class CountingChain(CheckerChain):
    """An empty checker chain that counts start_call, to prove the session calls it."""

    def __init__(self) -> None:
        super().__init__([], {})
        self.starts = 0

    def start_call(self) -> None:
        self.starts += 1
        super().start_call()


def test_the_session_starts_the_checker_once_per_call(run):
    chain = CountingChain()
    run(["Hi"], [text_reply("Hello")], checker=chain)
    assert chain.starts == 1


def test_the_start_line_lists_the_checker_backends_and_settings(run, tmp_path):
    record, _, _ = run(["Hi"], [text_reply("Hello")])
    path = tmp_path / "runs" / "traces" / f"{record.conversation_id}.jsonl"
    start = json.loads(path.read_text().splitlines()[0])
    assert start["checker_backends"] == []  # The test chain is empty.
    assert start["checker_jev_model"] == "typesafe/jev-1.14"
    assert start["checker_cutoff"] == 0.5
    assert start["checker_jev_timeout_s"] == 1.0


# Stage 5: a short spoken line while a tool round runs, so a voice caller is not left in silence.
SEARCH = ("search_clinic_info", {"question": "parking"})


def voice_speech(inputs):
    """FakeSpeech that behaves like a voice backend, which is where the filler is spoken."""
    speech = FakeSpeech(inputs)
    speech.uses_tone = True
    return speech


def test_a_voice_tool_round_speaks_a_filler_line_before_the_answer(run, clinic):
    speech = voice_speech(["Where can I park?"])
    script = [tool_reply(SEARCH), text_reply("There is a car park nearby.")]
    record, _, deps = run([], script, speech=speech)
    assert speech.spoken[1] in clinic.filler_lines
    assert speech.spoken[2] == "There is a car park nearby."
    # The transcript keeps the order the caller heard.
    assert record.transcript[2]["text"] in clinic.filler_lines
    # The trace marks it as a filler, not speech, so delay is measured to the real answer.
    lines = [json.loads(line) for line in deps.graph.trace.path.read_text().splitlines()]
    events = [event for line in lines for event in line.get("events", [])]
    assert [event["kind"] for event in events].count("filler") == 1
    assert deps.graph.on_tool_round is None  # The hook never outlives the call.


def test_the_filler_is_said_once_a_turn_however_many_rounds(run, clinic):
    speech = voice_speech(["Where can I park, and is Tuesday free?"])
    run([], [tool_reply(SEARCH), tool_reply(FIND), text_reply("Here you go.")], speech=speech)
    assert sum(line in clinic.filler_lines for line in speech.spoken) == 1


def test_no_filler_before_a_confirmation_question(run, clinic):
    # A prepare round goes straight to the system's question, so a filler would only delay it.
    speech = voice_speech(["Book me"])
    run([], [tool_reply(SAVE, FIND), tool_reply(PREPARE)], speech=speech)
    fillers = [line for line in speech.spoken if line in clinic.filler_lines]
    assert len(fillers) == 1  # Only for the save and search round, not the prepare round.
    assert speech.spoken[-1].startswith("Just to confirm")


def test_no_filler_when_the_only_round_prepares(run, clinic):
    speech = voice_speech(["Book me"])
    run([], [tool_reply(PREPARE)], speech=speech)
    assert not any(line in clinic.filler_lines for line in speech.spoken)


def test_text_chat_has_no_filler(run, clinic):
    _, speech, _ = run(["Where can I park?"], [tool_reply(SEARCH), text_reply("Nearby.")])
    assert not any(line in clinic.filler_lines for line in speech.spoken)


def test_a_failed_filler_does_not_break_the_turn(run, clinic):
    class FailingFiller(FakeSpeech):
        def speak(self, text, asks_confirmation=False):
            if text in clinic.filler_lines:
                raise RuntimeError("speaker unplugged")
            return super().speak(text, asks_confirmation)

    speech = FailingFiller(["Where can I park?"])
    speech.uses_tone = True
    run([], [tool_reply(SEARCH), text_reply("There is a car park nearby.")], speech=speech)
    assert speech.spoken[-1] == "There is a car park nearby."


def test_an_emergency_never_hears_a_filler_before_the_112_line(run, clinic):
    # The urgent phrase stops the turn before any tool runs, and flag_urgent never gets one.
    speech = voice_speech(["I can't breathe"])
    run([], [tool_reply(("flag_urgent", {"level": "life_threatening"}))], speech=speech)
    assert not any(line in clinic.filler_lines for line in speech.spoken)
    assert any("112" in line for line in speech.spoken)


class SlowFiller(FakeSpeech):
    """Voice speech whose filler line takes a while, as a real one does."""

    uses_tone = True

    def __init__(self, inputs, filler_lines):
        super().__init__(inputs)
        self.filler_lines = filler_lines

    def speak(self, text, asks_confirmation=False):
        if text in self.filler_lines:
            time.sleep(0.2)
        return super().speak(text, asks_confirmation)


def test_a_turn_with_no_reply_waits_for_its_filler_before_writing_the_trace(run, clinic):
    # A search round starts the filler, then the call ends with nothing more to say.
    speech = SlowFiller(["Where can I park? That's all, bye."], clinic.filler_lines)
    _, _, deps = run([], [tool_reply(SEARCH), tool_reply(END)], speech=speech)
    lines = [json.loads(line) for line in deps.graph.trace.path.read_text().splitlines()]
    turn = next(line for line in lines if line.get("turn") == 1)
    assert "filler" in [event["kind"] for event in turn["events"]]


def test_a_filler_that_could_not_be_spoken_is_not_in_the_transcript(run, clinic):
    class FailingFiller(FakeSpeech):
        uses_tone = True

        def speak(self, text, asks_confirmation=False):
            if text in clinic.filler_lines:
                raise RuntimeError("speaker unplugged")
            return super().speak(text, asks_confirmation)

    speech = FailingFiller(["Where can I park?"])
    record, _, _ = run([], [tool_reply(SEARCH), text_reply("Nearby.")], speech=speech)
    assert not any(item["text"] in clinic.filler_lines for item in record.transcript)
