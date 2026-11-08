"""Tests for stored records. Stage: 1.

What this file is:
    Proves the saved files are correct and safe.

What it contains:
    - The conversation file, summary and outcome for each ending.
    - Partial details.
    - Trace writes during failures.
    - No key appears in a trace.

Used by:
    pytest
"""

import json

from tests.fakes import ScriptedModel, text_reply

from dental_agent.models import ActionRecord, Details, EndReason, Outcome
from dental_agent.state import CallState
from dental_agent.storage.conversation import ConversationRecord
from dental_agent.storage.summary import key_points, optional_note
from dental_agent.storage.trace import TraceSink


def record(**changes):
    """Build a conversation record with sample values."""
    values = {
        "conversation_id": "c-1",
        "started_at": "2026-11-09T10:00:00+01:00",
        "caller_phone": "+34600000001",
    }
    return ConversationRecord(**(values | changes))


def action():
    """Build a stored booking."""
    return ActionRecord(
        action_id="a",
        conversation_id="c-1",
        kind="book",
        appointment_id=1,
        spoken="Tuesday 10 November at 10:00",
        result_text="You're booked.",
        created_at="now",
    )


def test_save_writes_the_conversation_file(tmp_path):
    item = record(transcript=[{"role": "bot", "text": "Hello"}])
    item.save(tmp_path)
    data = json.loads((tmp_path / "conversations" / "c-1.json").read_text())
    assert data["conversation_id"] == "c-1"
    assert data["transcript"][0]["text"] == "Hello"


def test_save_replaces_the_previous_version(tmp_path):
    item = record()
    item.save(tmp_path)
    item.outcome = Outcome.COMPLETED
    item.save(tmp_path)
    data = json.loads((tmp_path / "conversations" / "c-1.json").read_text())
    assert data["outcome"] == "completed"


def test_save_leaves_no_temporary_file(tmp_path):
    record().save(tmp_path)
    names = [path.name for path in (tmp_path / "conversations").iterdir()]
    assert names == ["c-1.json"]


def test_a_failed_save_keeps_the_last_complete_file(tmp_path, monkeypatch):
    item = record(outcome=Outcome.COMPLETED)
    item.save(tmp_path)

    def broken_replace(*_):
        raise OSError("disk full")

    monkeypatch.setattr("dental_agent.storage.conversation.os.replace", broken_replace)
    item.outcome = Outcome.FAILED
    item.save(tmp_path)  # A failed save must not raise, because the call is still running.
    data = json.loads((tmp_path / "conversations" / "c-1.json").read_text())
    assert data["outcome"] == "completed"


def test_partial_details_are_stored(tmp_path):
    item = record(details=Details(name="Ana Ruiz"), end_reason=EndReason.CALLER_LEFT)
    item.save(tmp_path)
    data = json.loads((tmp_path / "conversations" / "c-1.json").read_text())
    assert data["details"]["name"] == "Ana Ruiz"
    assert data["details"]["service"] is None
    assert data["end_reason"] == "caller_left"


def test_key_points_state_the_details_and_the_action():
    state = CallState(details=Details(name="Ana Ruiz", service="check_up"))
    points = key_points(state, [action()])
    assert "Caller: Ana Ruiz" in points
    assert "You're booked." in points


def test_key_points_say_when_nothing_changed():
    assert "No changes were made." in key_points(CallState(), [])


def test_key_points_note_an_urgent_call():
    state = CallState(urgent_level="life_threatening")
    assert "Urgent: life_threatening" in key_points(state, [])


def test_optional_note_returns_the_model_sentence():
    model = ScriptedModel(script=[text_reply("Ana booked a check-up.")])
    assert optional_note(model, ["Caller: Ana"], 3) == "Ana booked a check-up."


def test_optional_note_keeps_only_the_first_sentence():
    # On a real call the model added doubts about an emergency in a second paragraph.
    reply = "Lucia called about a cleaning.\n\nYou may want to verify the urgency."
    model = ScriptedModel(script=[text_reply(reply)])
    assert optional_note(model, ["Caller: Lucia"], 3) == "Lucia called about a cleaning."


def test_optional_note_keeps_only_one_sentence_on_one_line():
    model = ScriptedModel(script=[text_reply("First sentence. Second sentence.")])
    assert optional_note(model, ["Caller: Ana"], 3) == "First sentence."


def test_optional_note_returns_none_when_the_model_fails():
    # An exhausted script raises an error, which stands in for a failed model call.
    assert optional_note(ScriptedModel(script=[]), ["Caller: Ana"], 3) is None


def trace_lines(path):
    """Read a trace file."""
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_flush_writes_one_line_per_turn(tmp_path):
    sink = TraceSink()
    sink.bind("c-1", tmp_path)
    sink.event("tool_call", name="find_slots")
    sink.flush(turn=1, tools_used=True)
    line = trace_lines(tmp_path / "traces" / "c-1.jsonl")[0]
    assert line["conversation_id"] == "c-1"
    assert line["turn"] == 1
    assert line["tools_used"] is True
    assert line["events"][0]["kind"] == "tool_call"


def test_flush_clears_the_buffer(tmp_path):
    sink = TraceSink()
    sink.bind("c-1", tmp_path)
    sink.event("speech")
    sink.flush(1, False)
    sink.flush(2, False)
    lines = trace_lines(tmp_path / "traces" / "c-1.jsonl")
    assert len(lines[1]["events"]) == 0


def test_the_start_line_keeps_only_listed_fields(tmp_path):
    # A key passed by mistake must never reach the trace file.
    sink = TraceSink()
    sink.bind("c-1", tmp_path)
    sink.write_start({"llm_model": "m", "anthropic_api_key": "sk-secret"})
    text = (tmp_path / "traces" / "c-1.jsonl").read_text()
    assert '"llm_model"' in text
    assert "sk-secret" not in text


def test_a_failed_trace_write_does_not_raise(tmp_path, caplog):
    sink = TraceSink()
    # A file where a folder is needed makes every write fail.
    (tmp_path / "traces").write_text("in the way")
    sink.bind("c-1", tmp_path)
    sink.event("speech")
    sink.flush(1, False)
    assert "trace" in caplog.text.lower()


def test_events_stay_buffered_during_a_failed_turn(tmp_path):
    # Events live outside graph state, so an exception does not lose them.
    sink = TraceSink()
    sink.bind("c-1", tmp_path)
    try:
        sink.event("error", detail="boom")
        raise RuntimeError
    except RuntimeError:
        sink.flush(1, False)
    assert trace_lines(tmp_path / "traces" / "c-1.jsonl")[0]["events"][0]["detail"] == "boom"


def test_the_note_prompt_keeps_an_urgent_flag_apart_from_the_booking():
    # A real note read "called about a filling, marked as life-threatening urgency".
    model = ScriptedModel(script=[text_reply("Lucía booked a filling.")])
    optional_note(model, ["Urgent: life_threatening", "You're booked."], 3)
    prompt = str(model.seen[0])
    assert "urgent flag is about the caller's health" in prompt
