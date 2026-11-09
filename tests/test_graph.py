"""Tests for the turn graph. Stage: 1 and 3.

What this file is:
    Runs whole turns with ScriptedModel and proves every path.

What it contains:
    - Every edge in the flowchart, and each user journey in its stage.
    - A prepare cannot commit in the same turn.
    - An unheard confirmation question cannot be confirmed.
    - A checked lead-in from the model goes before the confirmation question.
    - A yes to a different question cannot commit.
    - A non-yes deletes the action.
    - A cut-off response runs no tools.
    - Exit order and the deadline.
    - Routing functions alone.
    - The checker in check_emergency: Jev decides, the Haiku checker joins the phrases, the main
      model's flag still wins, and with no answer the phrases decide.

How to understand it:
    This is the most important test file, because it proves the safety rules together. Read it
    with the flowchart.

Used by:
    pytest
"""

import json
import time
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage
from tests.conftest import TEST_NOW, build_test_deps
from tests.fakes import (
    FailingCalendar,
    ScriptedChecker,
    ScriptedModel,
    cut_off_reply,
    text_reply,
    tool_reply,
)

from dental_agent.booking.seed import seed
from dental_agent.checker import CheckAnswer, CheckerChain, CheckerError
from dental_agent.graph import routing
from dental_agent.graph.build import build_graph, graph_diagram
from dental_agent.models import (
    EndReason,
    FixedReason,
    PendingAction,
    ToolExit,
)
from dental_agent.session import start_fields
from dental_agent.state import CallState

NEW_PHONE = "+34600999999"
KNOWN_PHONE = "+34600000001"
SAVE = ("save_details", {"name": "Ana Ruiz", "service": "first_visit"})
FIND = (
    "find_slots",
    {"service": "first_visit", "date_from": "2026-11-16", "date_to": "2026-11-20"},
)
PREPARE = ("prepare_booking", {"slot_id": "s1"})


class Call:
    """One call driven directly through the graph, turn by turn."""

    def __init__(self, deps, phone=NEW_PHONE):
        self.deps = deps
        self.graph = build_graph(deps.graph)
        self.config = {"configurable": {"thread_id": "c-test"}}
        # The same start as a real call, so a known number is that existing patient.
        graph = deps.graph
        self.start = start_fields(graph.calendar, graph.clinic, graph.clock.now(), "c-test", phone)

    def say(self, text, awaiting_yes=False):
        """Run one turn and return the new state."""
        turn_input = {"last_caller_text": text, "awaiting_yes": awaiting_yes} | self.start
        self.start = {}
        return CallState.model_validate(self.graph.invoke(turn_input, self.config))

    @property
    def actions(self):
        """Return what this call wrote to the calendar."""
        return self.deps.graph.calendar.actions_for_call("c-test")


@pytest.fixture
def make_call(tmp_path, clinic, clock, calendar):
    """Build a call from a list of scripted model responses."""

    def build(*script, phone=NEW_PHONE, calendar_override=None, on_call=None, checker=None):
        model = ScriptedModel(script=list(script), on_call=on_call)
        used = calendar_override or calendar
        deps = build_test_deps(tmp_path, clinic, clock, used, model=model, checker=checker)
        return Call(deps, phone)

    return build


@pytest.fixture
def failing_calendar(tmp_path, clock, clinic):
    """A seeded calendar that can fail on request."""
    calendar = FailingCalendar(tmp_path / "failing.sqlite")
    seed(calendar, clock, clinic)
    return calendar


def booking_turn():
    """Return the model responses that prepare a booking in one turn."""
    return tool_reply(SAVE, FIND), tool_reply(PREPARE)


def test_a_plain_reply_passes_through(make_call):
    state = make_call(text_reply("Which day suits you?")).say("Hi")
    assert state.reply_text == "Which day suits you?"
    assert state.turn == 1


def test_history_records_both_sides_of_the_turn(make_call):
    state = make_call(text_reply("Hello")).say("Hi")
    assert state.messages == [
        {"role": "caller", "text": "Hi"},
        {"role": "bot", "text": "Hello"},
    ]


def test_the_checkpointer_keeps_state_between_turns(make_call):
    call = make_call(
        tool_reply(("save_details", {"name": "Ana Ruiz"})),
        text_reply("Thanks"),
        text_reply("Noted"),
    )
    call.say("I'm Ana")
    state = call.say("I need a check-up")
    assert state.details.name == "Ana Ruiz"
    assert state.turn == 2
    assert len(state.messages) == 4


def test_turn_messages_are_dropped_when_the_turn_ends(make_call):
    state = make_call(tool_reply(("save_details", {"name": "Ana"})), text_reply("Ok")).say("Hi")
    assert state.turn_messages == []


def test_a_prepare_reads_back_and_writes_nothing(make_call):
    call = make_call(*booking_turn())
    state = call.say("I'm Ana, new, first visit please")
    assert state.asks_confirmation
    assert state.reply_text.startswith("Just to confirm: a first-visit consultation for Ana Ruiz")
    assert call.actions == []


def test_a_checked_lead_in_goes_before_the_confirmation_question(make_call):
    lead = "A first-visit consultation takes about 45 minutes."
    call = make_call(tool_reply(SAVE, FIND), tool_reply(PREPARE, text=lead))
    state = call.say("I'm Ana, new, first visit please. How long does it take?")
    question = state.pending_action.confirmation_text
    assert question.startswith("Just to confirm:")
    # The code's question comes last, so the caller's yes answers it.
    assert state.reply_text == f"{lead} {question}"
    assert state.asks_confirmation


def test_a_lead_in_that_fails_its_check_is_dropped(make_call):
    call = make_call(tool_reply(SAVE, FIND), tool_reply(PREPARE, text="Friday is quiet."))
    state = call.say("I'm Ana, new, first visit please")
    assert state.reply_text == state.pending_action.confirmation_text
    guards = [e for e in call.deps.graph.trace.events if e.get("name") == "check_lead_in"]
    assert guards[-1]["decision"] == "reject"
    assert guards[-1]["reason"] == "time"


def test_only_text_sent_with_the_prepare_call_is_spoken(make_call):
    # Text from an earlier tool round, such as "Let me check", is never a lead-in.
    call = make_call(tool_reply(SAVE, FIND, text="Let me check."), tool_reply(PREPARE))
    state = call.say("I'm Ana, new, first visit please")
    assert state.reply_text == state.pending_action.confirmation_text


def test_a_yes_after_a_lead_in_still_commits(make_call):
    lead = "A first-visit consultation takes about 45 minutes."
    call = make_call(tool_reply(SAVE, FIND), tool_reply(PREPARE, text=lead))
    call.say("I'm Ana, new, first visit please")
    state = call.say("yes", awaiting_yes=True)
    assert len(call.actions) == 1
    assert state.reply_text.startswith("You're booked")


def test_a_yes_after_a_heard_confirmation_commits(make_call):
    call = make_call(*booking_turn())
    first = call.say("I'm Ana, new, first visit please")
    state = call.say("yes", awaiting_yes=True)
    assert [item.kind for item in call.actions] == ["book"]
    assert state.reply_text.startswith("You're booked for a first-visit consultation on")
    assert first.pending_action.offer.spoken in state.reply_text
    assert state.pending_action is None


def test_a_commit_uses_no_model_call(make_call):
    call = make_call(*booking_turn())
    call.say("Book me")
    before = len(call.deps.graph.model.seen)
    call.say("yes", awaiting_yes=True)
    assert len(call.deps.graph.model.seen) == before


def test_a_commit_puts_the_new_appointment_in_found_appointments(make_call):
    call = make_call(*booking_turn())
    call.say("Book me")
    state = call.say("yes", awaiting_yes=True)
    assert [item.service for item in state.found_appointments] == ["first_visit"]
    assert state.offers == []


def test_a_prepare_cannot_commit_in_the_same_turn(make_call):
    # The caller's "yes" arrives with the very turn that prepares the action.
    call = make_call(*booking_turn())
    call.say("yes", awaiting_yes=False)
    assert call.actions == []


def test_an_unheard_confirmation_cannot_be_confirmed(make_call):
    # The session passes awaiting_yes false when speaking the confirmation question failed.
    call = make_call(*booking_turn(), text_reply("What can I do for you?"))
    call.say("Book me")
    state = call.say("yes", awaiting_yes=False)
    assert call.actions == []
    assert state.pending_action is None


def test_a_yes_to_a_different_question_cannot_commit(make_call):
    call = make_call(text_reply("Is there anything else?"), text_reply("Good"))
    call.say("Hi")
    call.say("yes", awaiting_yes=False)
    assert call.actions == []


def test_a_non_yes_deletes_the_action(make_call):
    call = make_call(*booking_turn(), text_reply("What time suits you?"))
    call.say("Book me")
    state = call.say("no, make it later", awaiting_yes=True)
    assert state.pending_action is None
    assert call.actions == []


def test_a_slot_taken_before_the_commit_gives_the_reason_line(make_call, clinic):
    call = make_call(*booking_turn())
    first = call.say("Book me")
    action = first.pending_action.model_copy(update={"action_id": "someone-else"})
    call.deps.graph.calendar.book(
        action, "c-other", "+34600888888", "Luis Mora", "first_visit", "t", "Booked.", TEST_NOW
    )
    state = call.say("yes", awaiting_yes=True)
    assert state.reply_text == clinic.line("slot_taken")
    assert state.pending_action is None
    assert state.offers == [
        item for item in state.offers if item.db_slot_id != action.offer.db_slot_id
    ]
    assert state.details.name == "Ana Ruiz"
    assert call.actions == []


def test_a_refused_commit_ends_the_turn_without_a_model_call(make_call):
    call = make_call(*booking_turn())
    first = call.say("Book me")
    action = first.pending_action.model_copy(update={"action_id": "someone-else"})
    call.deps.graph.calendar.book(
        action, "c-other", "+34600888888", "Luis Mora", "first_visit", "t", "Booked.", TEST_NOW
    )
    before = len(call.deps.graph.model.seen)
    call.say("yes", awaiting_yes=True)
    assert len(call.deps.graph.model.seen) == before


def test_a_commit_error_with_a_record_counts_as_written(make_call, failing_calendar):
    call = make_call(*booking_turn(), calendar_override=failing_calendar)
    call.say("Book me")
    failing_calendar.fail_book = "after"
    state = call.say("yes", awaiting_yes=True)
    assert state.reply_text.startswith("You're booked")
    assert state.fixed_reason is None


def test_a_commit_error_with_no_record_ends_the_call(make_call, failing_calendar, clinic):
    call = make_call(*booking_turn(), calendar_override=failing_calendar)
    call.say("Book me")
    failing_calendar.fail_book = "before"
    state = call.say("yes", awaiting_yes=True)
    assert state.fixed_reason is FixedReason.COMMIT_FAILED
    assert state.reply_text == clinic.line("commit_unknown")
    assert state.end_reason is EndReason.COMMIT_FAILED


def test_an_existing_service_changes_the_confirmation(make_call):
    find = (
        "find_slots",
        {"service": "check_up", "date_from": "2026-11-16", "date_to": "2026-11-20"},
    )
    save = (
        "save_details",
        {"name": "Lucía Moreno", "patient_type": "existing", "service": "check_up"},
    )
    call = make_call(tool_reply(save, find), tool_reply(PREPARE), phone=KNOWN_PHONE)
    state = call.say("Another check-up please")
    assert "in addition to your existing appointment" in state.reply_text


def test_a_new_prepare_replaces_one_that_was_never_heard(make_call):
    call = make_call(
        *booking_turn(), tool_reply(FIND), tool_reply(("prepare_booking", {"slot_id": "s3"}))
    )
    first = call.say("Book me")
    second = call.say("A different time", awaiting_yes=False)
    assert second.pending_action.action_id != first.pending_action.action_id


def test_a_model_error_gives_the_trouble_line(make_call, clinic):
    state = make_call().say("Hi")  # An empty script raises an error.
    assert state.fixed_reason is FixedReason.MODEL_FAILURE
    assert state.reply_text == clinic.line("trouble")
    assert state.end_reason is None


def test_an_empty_reply_gives_the_trouble_line(make_call, clinic):
    assert make_call(text_reply("")).say("Hi").reply_text == clinic.line("trouble")


def test_a_cut_off_reply_gives_the_trouble_line(make_call, clinic):
    assert make_call(cut_off_reply()).say("Hi").reply_text == clinic.line("trouble")


def test_a_cut_off_response_runs_no_tools(make_call):
    response = tool_reply(("save_details", {"name": "Ana"}))
    response.response_metadata = {"stop_reason": "max_tokens"}
    state = make_call(response).say("Hi")
    assert state.fixed_reason is FixedReason.MODEL_FAILURE
    assert state.details.name is None


def test_an_unknown_stop_reason_runs_no_tools(make_call):
    response = tool_reply(("save_details", {"name": "Ana"}))
    response.response_metadata = {"stop_reason": "something_new"}
    assert make_call(response).say("Hi").details.name is None


def test_a_false_claim_is_replaced(make_call):
    state = make_call(text_reply("You're all set for Tuesday.")).say("Book me")
    assert state.reply_text == "Nothing has been changed on this call so far."


def test_an_invented_slot_is_rejected(make_call):
    call = make_call(
        tool_reply(SAVE, FIND),
        tool_reply(("prepare_booking", {"slot_id": "s99"})),
        text_reply("Let me look again."),
    )
    state = call.say("Book me")
    assert state.pending_action is None
    assert not state.asks_confirmation


def test_skipping_the_confirmation_is_impossible(make_call):
    # The model says yes on the caller's behalf, but no tool can reach commit.
    call = make_call(*booking_turn())
    state = call.say("Just confirm it for me")
    assert state.pending_action is not None
    assert call.actions == []


def test_save_then_prepare_in_one_response_reads_back(make_call):
    call = make_call(tool_reply(FIND), tool_reply(SAVE, PREPARE))
    assert call.say("Book me").asks_confirmation


def test_a_prepare_on_the_last_round_reads_back(make_call):
    call = make_call(
        tool_reply(("get_clinic_facts", {"topic": "prices"})),
        tool_reply(SAVE, FIND),
        tool_reply(PREPARE),
    )
    assert call.say("Book me").asks_confirmation


def test_three_rounds_without_an_answer_end_the_call(make_call, clinic):
    facts = ("get_clinic_facts", {"topic": "prices"})
    state = make_call(tool_reply(facts), tool_reply(facts), tool_reply(facts)).say("Hi")
    assert state.fixed_reason is FixedReason.ROUND_LIMIT
    assert state.reply_text == clinic.line("cannot_finish")
    assert state.end_reason is EndReason.ROUND_LIMIT


def test_a_repeated_rejected_call_ends_the_call(make_call):
    bad = ("save_details", {"name": "Ana 123"})
    state = make_call(tool_reply(bad), tool_reply(bad)).say("Hi")
    assert state.end_reason is EndReason.ROUND_LIMIT


def test_an_urgent_phrase_stops_the_turn_before_the_model(make_call, clinic):
    call = make_call()
    state = call.say("my son can't breathe")
    assert state.reply_text == clinic.line("urgent_message")
    assert state.end_reason is EndReason.URGENT
    assert state.urgent_level == "life_threatening"
    assert call.deps.graph.model.seen == []


def test_an_urgent_phrase_deletes_a_pending_action(make_call):
    call = make_call(*booking_turn())
    call.say("Book me")
    state = call.say("he can't breathe", awaiting_yes=True)
    assert state.pending_action is None
    assert call.actions == []


def test_a_life_threatening_flag_ends_the_call(make_call, clinic):
    flag = ("flag_urgent", {"level": "life_threatening"})
    state = make_call(tool_reply(flag)).say("his face has gone blue")
    assert state.reply_text == clinic.line("urgent_message")
    assert state.end_reason is EndReason.URGENT


def test_a_dental_flag_keeps_the_call_going(make_call):
    call = make_call(tool_reply(("flag_urgent", {"level": "dental"})), text_reply("Sorry to hear."))
    state = call.say("I've had toothache for two days")
    assert state.urgent_level == "dental"
    assert state.end_reason is None


def test_end_call_ends_with_an_empty_reply(make_call):
    state = make_call(tool_reply(("end_call", {}))).say("Bye")
    assert state.reply_text == ""
    assert state.end_reason is EndReason.DONE


def test_a_model_call_past_the_deadline_stops_at_run_tools(make_call, clock, clinic):
    state = make_call(tool_reply(FIND), on_call=lambda: clock.advance(13)).say("Book me")
    assert state.fixed_reason is FixedReason.DEADLINE
    assert state.reply_text == clinic.line("trouble")
    assert state.offers == []


def test_a_tool_past_the_deadline_stops_at_call_model(make_call, clock, monkeypatch, calendar):
    original = calendar.free_slots

    def slow(*args):
        clock.advance(13)
        return original(*args)

    monkeypatch.setattr(calendar, "free_slots", slow)
    call = make_call(tool_reply(FIND), text_reply("never reached"))
    assert call.say("Book me").fixed_reason is FixedReason.DEADLINE
    assert len(call.deps.graph.model.seen) == 1


def test_each_turn_gets_a_new_deadline(make_call, clock):
    call = make_call(text_reply("One"), text_reply("Two"))
    call.say("Hi")
    clock.advance(100)
    assert call.say("Hello").reply_text == "Two"


def test_a_run_after_a_graph_error_starts_cleanly(make_call, monkeypatch):
    # An exception inside a node must not stop the next turn from starting at start_turn.
    call = make_call(text_reply("First"), text_reply("Second"))
    call.say("Hi")
    calendar = call.deps.graph.calendar
    original = calendar.actions_for_call

    def broken(_):
        raise RuntimeError("read failed")

    monkeypatch.setattr(calendar, "actions_for_call", broken)
    with pytest.raises(RuntimeError):
        call.say("Hello")
    monkeypatch.setattr(calendar, "actions_for_call", original)
    state = call.say("Hello again")
    assert state.reply_text == "Second"
    assert state.turn == 2


def test_guard_decisions_reach_the_trace(make_call):
    call = make_call(*booking_turn())
    call.say("Book me")
    call.say("yes", awaiting_yes=True)
    kinds = [event["kind"] for event in call.deps.graph.trace.events]
    assert "guard" in kinds
    assert "calendar_write" in kinds
    assert "model_call" in kinds


def test_after_check_emergency_goes_to_use_fixed_line_only_for_a_fixed_reason():
    assert (
        routing.after_check_emergency(CallState(fixed_reason=FixedReason.URGENT))
        == "use_fixed_line"
    )
    assert routing.after_check_emergency(CallState()) == "check_confirmation"


def test_after_run_tools_routes_each_exit():
    prepared = CallState(tool_exit=ToolExit.PREPARED)
    assert routing.after_run_tools(prepared) == "ask_confirmation"
    assert routing.after_run_tools(CallState(tool_exit=ToolExit.NONE)) == "call_model"
    assert routing.after_run_tools(CallState(tool_exit=ToolExit.END_CALL)) == "end_turn"
    assert routing.after_run_tools(CallState(tool_exit=ToolExit.URGENT)) == "use_fixed_line"
    assert routing.after_run_tools(CallState(tool_exit=ToolExit.ROUND_LIMIT)) == "use_fixed_line"
    late = CallState(fixed_reason=FixedReason.DEADLINE)
    assert routing.after_run_tools(late) == "use_fixed_line"


def test_after_write_to_calendar_routes_a_refusal_to_use_fixed_line():
    refused = CallState(fixed_reason=FixedReason.COMMIT_REFUSED)
    assert routing.after_write_to_calendar(refused) == "use_fixed_line"
    assert routing.after_write_to_calendar(CallState()) == "end_turn"


def test_the_diagram_is_drawn_from_the_real_graph(make_call):
    diagram = graph_diagram(make_call().deps.graph)
    assert "flowchart" in diagram or "graph TD" in diagram
    for node in (
        "start_turn",
        "check_confirmation",
        "write_to_calendar",
        "call_model",
        "run_tools",
        "end_turn",
    ):
        assert node in diagram


def test_checkpoints_load_with_no_blocked_type_warning(make_call, caplog):
    # LangGraph warns that unlisted types will be blocked in a future version.
    call = make_call(*booking_turn(), text_reply("Noted"))
    call.say("Book me")
    call.say("no", awaiting_yes=True)
    assert "unregistered type" not in caplog.text


def test_the_system_message_stays_the_same_within_a_turn(make_call):
    # Thinking blocks are tied to the system prompt, so a change mid-turn makes the API refuse.
    save = ("save_details", {"name": "Ana Ruiz"})
    call = make_call(tool_reply(save), text_reply("Thanks."))
    call.say("I'm Ana")
    first_round, second_round = call.deps.graph.model.seen
    assert first_round[0].content == second_round[0].content


def test_the_next_turn_sees_the_new_facts(make_call):
    save = ("save_details", {"name": "Ana Ruiz"})
    call = make_call(tool_reply(save), text_reply("Thanks."), text_reply("Which service?"))
    call.say("I'm Ana")
    call.say("I'd like an appointment")
    assert "- name: Ana Ruiz" in call.deps.graph.model.seen[-1][0].content


def test_ai_messages_are_resent_unchanged_within_a_turn(make_call):
    # Thinking blocks must go back exactly as they came.
    response = tool_reply(("save_details", {"name": "Ana"}))
    call = make_call(response, text_reply("Thanks."))
    call.say("I'm Ana")
    assert call.deps.graph.model.seen[1][-2].tool_calls == response.tool_calls


def test_a_reply_sent_with_a_tool_call_is_used_when_the_final_response_is_empty(make_call):
    # Seen in a real call: the model answered next to save_details, then sent nothing.
    from langchain_core.messages import AIMessage

    save = ("save_details", {"name": "Ana Ruiz"})
    empty = AIMessage(content="", response_metadata={"stop_reason": "end_turn"})
    call = make_call(tool_reply(save, text="Thanks, Ana. Which service would you like?"), empty)
    state = call.say("I'm Ana Ruiz")
    assert state.reply_text == "Thanks, Ana. Which service would you like?"
    assert state.fixed_reason is None


def test_an_empty_turn_with_no_earlier_text_is_still_a_model_failure(make_call):
    from langchain_core.messages import AIMessage

    empty = AIMessage(content="", response_metadata={"stop_reason": "end_turn"})
    call = make_call(tool_reply(("save_details", {"name": "Ana Ruiz"})), empty)
    assert call.say("I'm Ana Ruiz").fixed_reason is FixedReason.MODEL_FAILURE


# ---------- the checker in check_emergency ----------

STROKE_LAST_YEAR = "I had a stroke last year, is the clinic wheelchair accessible?"


# Jev's answer about a model reply that claims nothing and gives no advice.
CLEAN_REPLY = CheckAnswer(
    layer="jev", answers={"claims_change": False, "medical_advice": False}, model="m"
)


def jev_says(yes: bool, probability: float, *then: CheckAnswer) -> CheckerChain:
    """A chain whose only backend is Jev: this emergency answer, then any answers in then."""
    answer = CheckAnswer(
        layer="jev",
        answers={"emergency": yes},
        probabilities={"emergency": probability},
        model="typesafe/jev-1.14-20261014",
    )
    return CheckerChain([ScriptedChecker("jev", [answer, *then])], {"jev": 1.0})


def haiku_says(yes: bool) -> CheckerChain:
    """A chain whose only backend is the Haiku checker, answering once."""
    answer = CheckAnswer(layer="anthropic", answers={"emergency": yes}, model="claude-haiku-5-5")
    return CheckerChain([ScriptedChecker("anthropic", [answer])], {"anthropic": 2.0})


def checker_events(call) -> list[dict]:
    return [e for e in call.deps.graph.trace.events if e["kind"] == "checker"]


def test_a_jev_yes_ends_the_call_with_no_phrase_and_drops_the_model(make_call, clinic):
    # CUJ 1: no emergency phrase matches, but Jev says yes. The main model ran alongside the
    # check, and its reply is dropped.
    call = make_call(text_reply("Which day suits you?"), checker=jev_says(True, 0.79))
    state = call.say("my lips are turning blue")
    assert state.reply_text == clinic.line("urgent_message")
    assert state.end_reason is EndReason.URGENT
    assert state.urgent_level == "life_threatening"


def test_a_jev_no_over_a_phrase_match_lets_the_call_continue(make_call):
    # CUJ 2: "stroke" matches a phrase, but Jev says no, so the main model takes the turn.
    call = make_call(
        text_reply("Yes, the clinic has step-free access."),
        checker=jev_says(False, 0.04, CLEAN_REPLY),  # The model's reply is checked too.
    )
    state = call.say(STROKE_LAST_YEAR)
    assert state.end_reason is None
    assert state.urgent_level is None
    assert len(call.deps.graph.model.seen) == 1
    guard = next(e for e in call.deps.graph.trace.events if e.get("name") == "check_emergency")
    assert guard["phrase_hit"] is True
    assert guard["decision"] == "pass"


def test_with_the_haiku_checker_a_phrase_still_ends_the_call(make_call, clinic):
    # CUJ 3: the Haiku checker gives no probability, so its "no" can't silence a phrase.
    call = make_call(checker=haiku_says(False))
    state = call.say(STROKE_LAST_YEAR)
    assert state.end_reason is EndReason.URGENT
    assert state.reply_text == clinic.line("urgent_message")


def test_the_main_models_flag_wins_over_a_jev_no(make_call, clinic):
    # CUJ 4: Jev says no, and the main model then flags a life-threatening emergency.
    flag = ("flag_urgent", {"level": "life_threatening"})
    call = make_call(tool_reply(flag), checker=jev_says(False, 0.06))
    state = call.say("he has gone grey and he isn't answering me")
    assert state.end_reason is EndReason.URGENT
    assert state.reply_text == clinic.line("urgent_message")


def test_with_no_checker_answer_the_phrases_decide(make_call, clinic):
    # CUJ 9: every backend fails, so the phrases decide alone, as on main.
    failing = CheckerChain([ScriptedChecker("jev", [CheckerError("down")])], {"jev": 1.0})
    call = make_call(checker=failing)
    state = call.say("my son can't breathe")
    assert state.end_reason is EndReason.URGENT
    event = checker_events(call)[0]
    assert event["layer"] == "fixed"
    assert event["attempts"][0]["ok"] is False


def test_with_no_checker_answer_a_reworded_emergency_reaches_the_main_model(make_call):
    # Two failures: the caller's words, then the model's reply.
    down = [CheckerError("down"), CheckerError("down")]
    failing = CheckerChain([ScriptedChecker("jev", down)], {"jev": 1.0})
    call = make_call(text_reply("Can you tell me more?"), checker=failing)
    state = call.say("my lips are turning blue")
    assert state.end_reason is None
    assert len(call.deps.graph.model.seen) == 1


def test_a_full_outage_spends_both_strikes_in_one_turn_and_jev_is_skipped_after(make_call):
    # A turn makes two checker requests, the caller's words and the model's reply, so a Jev that
    # is down is tried twice in the first turn and never again.
    jev = ScriptedChecker("jev", [CheckerError("down"), CheckerError("down")])
    calm = CheckAnswer(layer="anthropic", answers={"emergency": False}, model="m")
    fine = CheckAnswer(
        layer="anthropic", answers={"claims_change": False, "medical_advice": False}, model="m"
    )
    haiku = ScriptedChecker("anthropic", [calm, fine, calm, fine])
    chain = CheckerChain([jev, haiku], {"jev": 1.0, "anthropic": 2.0})
    call = make_call(text_reply("Hello."), text_reply("Of course."), checker=chain)
    call.say("Hi")
    assert jev.calls == 2
    call.say("Thanks")
    assert jev.calls == 2
    assert haiku.calls == 4


def test_a_checker_alarm_deletes_a_pending_action(make_call):
    yes = CheckAnswer(layer="jev", answers={"emergency": True}, model="m")
    no = CheckAnswer(layer="jev", answers={"emergency": False}, model="m")
    chain = CheckerChain([ScriptedChecker("jev", [no, yes])], {"jev": 1.0})
    call = make_call(*booking_turn(), checker=chain)
    call.say("Book me")
    state = call.say("my lips are turning blue", awaiting_yes=True)
    # A non-yes deletes the action anyway, so the ending proves the checker raised the alarm.
    assert state.end_reason is EndReason.URGENT
    assert state.pending_action is None
    assert call.actions == []


def test_the_checker_event_records_the_answer_and_never_the_text(make_call):
    call = make_call(checker=jev_says(True, 0.79))
    call.say("my lips are turning blue")
    event = checker_events(call)[0]
    assert event["step"] == "check_emergency"
    assert event["layer"] == "jev"
    assert event["probabilities"] == {"emergency": 0.79}
    assert event["model"] == "typesafe/jev-1.14-20261014"
    assert event["attempts"][0]["name"] == "jev"
    assert "my lips are turning blue" not in str(event)


def test_the_checker_is_given_the_time_left_in_the_turn(make_call):
    no_emergency = CheckAnswer(layer="jev", answers={"emergency": False}, model="m")
    nothing = CheckAnswer(
        layer="jev", answers={"claims_change": False, "medical_advice": False}, model="m"
    )
    backend = ScriptedChecker("jev", [no_emergency, nothing])
    call = make_call(text_reply("Hello"), checker=CheckerChain([backend], {"jev": 1.0}))
    call.say("Hi")
    # The fixed clock doesn't move, so the whole 12 s deadline is left and the 1 s limit applies
    # to both requests: the caller's words first, then the model's reply.
    assert backend.limits == [1.0, 1.0]
    assert backend.texts == ["Hi", "Hello"]


# ---------- the checker in the reply and lead-in checks ----------


def checker_answers(*answers: dict) -> CheckerChain:
    """A Jev-only chain that answers each request in turn with these answers."""
    results = [CheckAnswer(layer="jev", answers=a, model="m") for a in answers]
    return CheckerChain([ScriptedChecker("jev", results)], {"jev": 1.0})


NO_EMERGENCY = {"emergency": False}


def test_a_claim_in_other_words_with_no_record_becomes_the_status(make_call):
    # CUJ 6: no claim phrase matches, but the checker flags a claim.
    chain = checker_answers(NO_EMERGENCY, {"claims_change": True, "medical_advice": False})
    call = make_call(text_reply("I've taken care of the cancellation for you."), checker=chain)
    state = call.say("Can you cancel my check-up?")
    assert state.reply_text == "Nothing has been changed on this call so far."


def test_advice_in_other_words_becomes_the_medical_line(make_call, clinic):
    # CUJ 7: no dosing word matches, but the checker flags advice.
    chain = checker_answers(NO_EMERGENCY, {"claims_change": False, "medical_advice": True})
    call = make_call(
        text_reply("Try rinsing with warm salt water until your visit."), checker=chain
    )
    state = call.say("My gum is sore")
    assert state.reply_text == clinic.line("medical")


def test_a_lead_in_flagged_as_advice_is_dropped_before_the_question(make_call):
    # CUJ 8: the lead-in gives advice, so only the confirmation question is spoken.
    chain = checker_answers(NO_EMERGENCY, {"claims_change": False, "medical_advice": True})
    call = make_call(
        tool_reply(SAVE, FIND),
        tool_reply(PREPARE, text="Rinse with salt water until then."),
        checker=chain,
    )
    state = call.say("Book me a first visit")
    assert state.reply_text.startswith("Just to confirm")
    assert "salt water" not in state.reply_text


def test_no_lead_in_means_no_reply_check_request(make_call):
    # Without a lead-in, ask_confirmation asks the checker nothing: only the emergency check runs.
    backend = ScriptedChecker("jev", [CheckAnswer(layer="jev", answers=NO_EMERGENCY, model="m")])
    call = make_call(*booking_turn(), checker=CheckerChain([backend], {"jev": 1.0}))
    call.say("Book me")
    assert backend.calls == 1


def test_the_reply_check_writes_a_checker_event_with_no_text(make_call):
    chain = checker_answers(NO_EMERGENCY, {"claims_change": False, "medical_advice": False})
    call = make_call(text_reply("Which day suits you?"), checker=chain)
    call.say("Hello")
    events = [e for e in call.deps.graph.trace.events if e["kind"] == "checker"]
    assert [e["step"] for e in events] == ["check_emergency", "check_model_reply"]
    assert "Which day suits you?" not in str(events[1])


# ---------- the emergency check alongside the main model ----------


def timed_jev(yes: bool, order: list[str], wait_s: float, *then: CheckAnswer) -> CheckerChain:
    """A Jev chain whose emergency answer takes wait_s, recording when it starts and ends."""

    def slow():
        order.append("check starts")
        time.sleep(wait_s)
        order.append("check ends")

    answer = CheckAnswer(
        layer="jev", answers={"emergency": yes}, probabilities={"emergency": 0.9}, model="m"
    )
    backend = ScriptedChecker("jev", [answer, *then])
    backend.on_call = lambda: slow() if backend.calls == 1 else None
    return CheckerChain([backend], {"jev": 2.0})


def test_without_a_phrase_the_check_runs_while_the_model_thinks(make_call):
    order = []
    call = make_call(
        text_reply("Which day suits you?"),
        on_call=lambda: (order.append("model starts"), time.sleep(0.3)),
        checker=timed_jev(False, order, 0.3, CLEAN_REPLY),
    )
    started = time.monotonic()
    state = call.say("I'd like a cleaning")
    elapsed = time.monotonic() - started
    assert state.reply_text == "Which day suits you?"
    assert order.index("model starts") < order.index("check ends")
    assert elapsed < 0.55  # One after the other would take at least 0.6 s.


def test_an_emergency_answered_after_the_model_drops_its_reply(make_call, clinic):
    order = []
    call = make_call(text_reply("Sure, which day?"), checker=timed_jev(True, order, 0.2))
    state = call.say("my lips are turning blue")
    assert state.reply_text == clinic.line("urgent_message")
    assert state.end_reason is EndReason.URGENT
    assert state.urgent_level == "life_threatening"
    assert "Sure, which day?" not in [item.get("text") for item in state.messages]


def test_a_dropped_reply_runs_none_of_its_tools(make_call):
    order = []
    call = make_call(*booking_turn(), checker=timed_jev(True, order, 0.2))
    state = call.say("my lips are turning blue, book me a check-up")
    assert state.end_reason is EndReason.URGENT
    kinds = [event["kind"] for event in call.deps.graph.trace.events]
    assert "tool_call" not in kinds
    assert state.pending_action is None


def test_a_yes_waits_for_the_check_before_anything_is_written(make_call):
    no = CheckAnswer(layer="jev", answers={"emergency": False}, model="m")
    order = []
    call = make_call(*booking_turn(), checker=timed_jev(False, order, 0.0, no))
    call.say("Book me")
    # The second turn's emergency answer is a slow yes, to a plain "yes".
    later = CheckAnswer(layer="jev", answers={"emergency": True}, model="m")
    backend = call.deps.graph.checker.backends[0]
    backend.results = [later]
    backend.on_call = lambda: time.sleep(0.2)
    state = call.say("yes", awaiting_yes=True)
    assert state.end_reason is EndReason.URGENT
    assert call.actions == []


def test_a_phrase_match_is_still_checked_before_the_model_starts(make_call):
    # A phrase can end the call, so the model is not started until the checker has answered.
    order = []
    call = make_call(
        text_reply("Yes, the clinic has step-free access."),
        on_call=lambda: order.append("model starts"),
        checker=timed_jev(False, order, 0.1, CLEAN_REPLY),
    )
    call.say(STROKE_LAST_YEAR)
    assert order == ["check starts", "check ends", "model starts"]


def test_with_the_checker_off_the_phrases_decide_before_the_model(make_call, clinic):
    call = make_call()
    state = call.say("my son can't breathe")
    assert state.reply_text == clinic.line("urgent_message")
    assert call.deps.graph.model.seen == []


def test_the_check_events_are_traced_once_per_turn(make_call):
    order = []
    call = make_call(text_reply("Hello."), checker=timed_jev(False, order, 0.05, CLEAN_REPLY))
    call.say("Hi")
    guards = [e for e in call.deps.graph.trace.events if e.get("name") == "check_emergency"]
    steps = [e["step"] for e in checker_events(call)]
    assert len(guards) == 1 and guards[0]["decision"] == "pass"
    assert steps.count("check_emergency") == 1


def test_after_await_emergency_routes_each_case():
    urgent = CallState(fixed_reason=FixedReason.URGENT)
    assert routing.after_await_emergency(urgent) == "use_fixed_line"
    pending = PendingAction(action_id="x", kind="book", offer=None)
    assert routing.after_await_emergency(CallState(pending_action=pending)) == "write_to_calendar"
    with_tools = CallState(turn_messages=[tool_reply(SAVE)])
    assert routing.after_await_emergency(with_tools) == "run_tools"
    with_text = CallState(turn_messages=[text_reply("Hello.")])
    assert routing.after_await_emergency(with_text) == "check_model_reply"


def test_an_emergency_is_spoken_without_waiting_for_a_slow_model(make_call, clinic):
    # A real call: Jev answered in 0.28 s, but the 112 message waited for a 2.06 s model reply
    # that was then dropped.
    order = []
    call = make_call(
        text_reply("Sure, which day?"),
        on_call=lambda: time.sleep(1.0),
        checker=timed_jev(True, order, 0.05),
    )
    started = time.monotonic()
    # The real transcript. "Choking" is an emergency phrase, but "shocking" is not.
    state = call.say("I'm shocking and bleeding a lot.")
    assert time.monotonic() - started < 0.6
    assert state.reply_text == clinic.line("urgent_message")
    assert state.end_reason is EndReason.URGENT
    assert "model_dropped" in [event["kind"] for event in call.deps.graph.trace.events]


def test_a_quick_no_still_waits_for_the_models_reply(make_call):
    order = []
    call = make_call(
        text_reply("Which day suits you?"),
        on_call=lambda: time.sleep(0.3),
        checker=timed_jev(False, order, 0.05, CLEAN_REPLY),
    )
    state = call.say("I'd like a cleaning")
    assert state.reply_text == "Which day suits you?"


# Stage 3: reschedule and cancel. Lucía (KNOWN_PHONE) has one check-up at least 3 days ahead.
LOOKUP = ("lookup_appointment", {})
FIND_MOVE = (
    "find_slots",
    {
        "service": "check_up",
        "date_from": "2026-11-16",
        "date_to": "2026-11-20",
        "appointment_id": "a1",
    },
)
RESCHEDULE = ("prepare_reschedule", {"appointment_id": "a1", "slot_id": "s1"})
CANCEL = ("prepare_cancel", {"appointment_id": "a1"})


def move_call(make_call):
    """Return a call by Lucía whose first turn prepares moving her check-up."""
    return make_call(tool_reply(LOOKUP, FIND_MOVE), tool_reply(RESCHEDULE), phone=KNOWN_PHONE)


def cancel_call(make_call):
    """Return a call by Lucía whose first turn prepares cancelling her check-up."""
    return make_call(tool_reply(LOOKUP), tool_reply(CANCEL), phone=KNOWN_PHONE)


def test_a_reschedule_reads_back_both_times_with_the_name(make_call):
    call = move_call(make_call)
    state = call.say("Can I move my check-up to next week?")
    action = state.pending_action
    assert action.kind == "move"
    assert state.reply_text == (
        f"Just to confirm: move your check-up for Lucía Moreno from {action.appointment.spoken} "
        f"to {action.offer.spoken}. Shall I go ahead?"
    )
    assert call.actions == []


def test_a_yes_moves_the_appointment(make_call):
    call = move_call(make_call)
    first = call.say("Move my check-up please")
    old = first.pending_action.appointment
    new = first.pending_action.offer
    state = call.say("yes", awaiting_yes=True)
    assert [item.kind for item in call.actions] == ["move"]
    assert state.reply_text == f"Your check-up is now on {new.spoken}."
    calendar = call.deps.graph.calendar
    assert calendar.get_appointment(old.db_id).db_slot_id == new.db_slot_id
    assert calendar.slot_is_free(old.db_slot_id)
    # The refreshed list shows the appointment at its new time.
    assert [item.db_slot_id for item in state.found_appointments] == [new.db_slot_id]
    assert state.offers == []


def test_a_cancel_reads_back_with_the_name(make_call):
    call = cancel_call(make_call)
    state = call.say("Please cancel my check-up")
    action = state.pending_action
    assert action.kind == "cancel"
    assert state.reply_text == (
        f"Just to confirm: cancel your check-up for Lucía Moreno on {action.appointment.spoken}. "
        "Shall I go ahead?"
    )


def test_a_yes_cancels_the_appointment(make_call):
    call = cancel_call(make_call)
    first = call.say("Please cancel my check-up")
    appointment = first.pending_action.appointment
    state = call.say("yes", awaiting_yes=True)
    assert [item.kind for item in call.actions] == ["cancel"]
    assert state.reply_text == f"Your check-up on {appointment.spoken} is cancelled."
    assert call.deps.graph.calendar.get_appointment(appointment.db_id).status == "cancelled"
    assert state.found_appointments == []


def test_a_no_keeps_the_appointment(make_call):
    call = make_call(
        tool_reply(LOOKUP), tool_reply(CANCEL), text_reply("OK, I've left it."), phone=KNOWN_PHONE
    )
    first = call.say("Please cancel my check-up")
    state = call.say("no, leave it", awaiting_yes=True)
    assert call.actions == []
    assert state.pending_action is None
    appointment = first.pending_action.appointment
    assert call.deps.graph.calendar.get_appointment(appointment.db_id).status == "active"


def test_an_appointment_cancelled_before_the_yes_gives_the_not_owner_line(make_call, clinic):
    call = cancel_call(make_call)
    first = call.say("Please cancel my check-up")
    with call.deps.graph.calendar._connect() as connection:
        connection.execute(
            "UPDATE appointments SET status = 'cancelled' WHERE id = ?",
            (first.pending_action.appointment.db_id,),
        )
    state = call.say("yes", awaiting_yes=True)
    assert state.reply_text == clinic.line("not_owner")
    assert state.pending_action is None
    assert call.actions == []
    # The list is read again, so the gone appointment is not shown, or used as proof, later.
    assert state.found_appointments == []


def test_a_new_slot_taken_before_the_yes_gives_the_slot_taken_line(make_call, clinic):
    call = move_call(make_call)
    first = call.say("Move my check-up please")
    offer = first.pending_action.offer
    other = PendingAction(action_id="someone-else", kind="book", offer=offer)
    call.deps.graph.calendar.book(
        other, "c-other", "+34600888888", "Luis Mora", "check_up", "t", "Booked.", TEST_NOW
    )
    state = call.say("yes", awaiting_yes=True)
    assert state.reply_text == clinic.line("slot_taken")
    assert all(item.db_slot_id != offer.db_slot_id for item in state.offers)
    old = first.pending_action.appointment
    assert call.deps.graph.calendar.get_appointment(old.db_id).db_slot_id == old.db_slot_id


def test_a_not_active_error_at_write_time_maps_to_not_owner(make_call, clinic, monkeypatch):
    # Another call cancels the row between the check and the write.
    from dental_agent.booking.calendar import NotActive

    call = cancel_call(make_call)
    call.say("Please cancel my check-up")

    def cancelled_elsewhere(*args, **kwargs):
        raise NotActive("1")

    monkeypatch.setattr(call.deps.graph.calendar, "cancel", cancelled_elsewhere)
    state = call.say("yes", awaiting_yes=True)
    assert state.reply_text == clinic.line("not_owner")
    assert state.fixed_reason is FixedReason.COMMIT_REFUSED


def test_a_cancel_lead_in_may_name_the_appointments_service(make_call):
    # The lead-in rule refuses other services, so it must be given the check-up, not None.
    lead = "A check-up takes about 30 minutes."
    call = make_call(tool_reply(LOOKUP), tool_reply(CANCEL, text=lead), phone=KNOWN_PHONE)
    state = call.say("How long is a check-up? Actually, cancel mine")
    assert state.reply_text == f"{lead} {state.pending_action.confirmation_text}"


def test_an_appointment_for_a_service_no_longer_offered_still_reads_back(make_call):
    # The service ID comes from the calendar row, so it may be missing from clinic.yaml.
    call = cancel_call(make_call)
    with call.deps.graph.calendar._connect() as connection:
        connection.execute(
            "UPDATE appointments SET service = 'old_whitening' WHERE patient_id = "
            "(SELECT id FROM patients WHERE phone = ?)",
            (KNOWN_PHONE,),
        )
    state = call.say("Please cancel my appointment")
    assert state.reply_text.startswith("Just to confirm: cancel your old whitening for Lucía")
    state = call.say("yes", awaiting_yes=True)
    assert state.reply_text.startswith("Your old whitening on ")


# Stage 5: a reply that repeats this turn's search results is the clinic's own text.
AFTERCARE_DOC = """# Handbook

## After a filling

If you had a local anaesthetic, your lip and tongue may feel numb for two to three hours.
"""


def with_handbook(call, tmp_path):
    """Give the call a keyword index of a one-piece handbook."""
    from dental_agent.clinic.index import build_index
    from dental_agent.clinic.retriever import KeywordRetriever

    docs = tmp_path / "handbook.md"
    docs.write_text(AFTERCARE_DOC)
    build_index([docs], tmp_path / "rag.sqlite", None)
    call.deps.graph.retriever = KeywordRetriever(tmp_path / "rag.sqlite")
    return call


def test_a_reply_repeating_the_search_result_is_spoken_despite_the_advice_flag(make_call, tmp_path):
    chain = checker_answers(NO_EMERGENCY, {"claims_change": False, "medical_advice": True})
    reply = "Your lip and tongue may feel numb for two to three hours after a local anaesthetic."
    call = with_handbook(
        make_call(
            tool_reply(("search_clinic_info", {"question": "numb filling"})),
            text_reply(reply),
            checker=chain,
        ),
        tmp_path,
    )
    state = call.say("How long does the numbness last after a filling?")
    assert state.reply_text == reply
    guards = [e for e in call.deps.graph.trace.events if e.get("name") == "check_model_reply"]
    assert guards[-1]["reason"] == "sourced"


def test_advice_beyond_the_search_result_still_becomes_the_medical_line(
    make_call, tmp_path, clinic
):
    chain = checker_answers(NO_EMERGENCY, {"claims_change": False, "medical_advice": True})
    reply = "Your lip may feel numb for two to three hours. Rinse with warm salt water tonight."
    call = with_handbook(
        make_call(
            tool_reply(("search_clinic_info", {"question": "numb filling"})),
            text_reply(reply),
            checker=chain,
        ),
        tmp_path,
    )
    state = call.say("How long does the numbness last after a filling?")
    assert state.reply_text == clinic.line("medical")


def test_the_prompts_own_referral_sentence_does_not_undo_a_sourced_reply(make_call, tmp_path):
    # A live run repeated the handbook, then added the referral the prompt asks for.
    chain = checker_answers(NO_EMERGENCY, {"claims_change": False, "medical_advice": True})
    reply = (
        "Your lip and tongue may feel numb for two to three hours after a local anaesthetic. "
        "The dentist can advise you further at your visit."
    )
    call = with_handbook(
        make_call(
            tool_reply(("search_clinic_info", {"question": "numb filling"})),
            text_reply(reply),
            checker=chain,
        ),
        tmp_path,
    )
    state = call.say("How long does the numbness last after a filling?")
    assert state.reply_text == reply


def test_a_clinic_line_with_its_number_filled_in_does_not_undo_a_sourced_reply(
    make_call, tmp_path, clinic
):
    chain = checker_answers(NO_EMERGENCY, {"claims_change": False, "medical_advice": True})
    reply = (
        "Your lip and tongue may feel numb for two to three hours after a local anaesthetic. "
        + clinic.line("call_clinic")
    )
    call = with_handbook(
        make_call(
            tool_reply(("search_clinic_info", {"question": "numb filling"})),
            text_reply(reply),
            checker=chain,
        ),
        tmp_path,
    )
    state = call.say("How long does the numbness last after a filling?")
    assert state.reply_text == reply


def test_fixed_lines_alone_never_excuse_advice_without_a_search(make_call, clinic):
    chain = checker_answers(NO_EMERGENCY, {"claims_change": False, "medical_advice": True})
    call = make_call(
        text_reply("Rinse with salt water. The dentist can advise you at your visit."),
        checker=chain,
    )
    assert call.say("My gum hurts").reply_text == clinic.line("medical")


def test_the_next_turn_can_repeat_the_last_search_without_the_medical_line(make_call, tmp_path):
    chain = checker_answers(
        NO_EMERGENCY,
        {"claims_change": False, "medical_advice": False},
        NO_EMERGENCY,
        {"claims_change": False, "medical_advice": True},
    )
    reply = "Your lip and tongue may feel numb for two to three hours after a local anaesthetic."
    call = with_handbook(
        make_call(
            tool_reply(("search_clinic_info", {"question": "numb filling"})),
            text_reply("Yes, I found that in the handbook."),
            text_reply(reply),
            checker=chain,
        ),
        tmp_path,
    )
    call.say("How long does the numbness last after a filling?")
    state = call.say("Sorry, how long did you say?")
    assert state.reply_text == reply


# ---------- Reply sources: the handbook text the advice exemption may accept ----------


def _search_message(pieces):
    """A search_clinic_info result as the model sees it."""
    data = [{"heading": h, "text": t, "source": "patient-handbook.md"} for h, t in pieces]
    return ToolMessage(content=json.dumps({"ok": True, "data": data}), tool_call_id="call-1")


def test_reply_sources_put_each_pieces_heading_before_its_text():
    from dental_agent.graph.nodes import search_texts

    message = _search_message([("After your visit › After a filling", "Your lip may feel numb.")])
    assert search_texts([message]) == ["After your visit › After a filling Your lip may feel numb."]


def test_the_last_searchs_pieces_keep_their_headings_too():
    from dental_agent.graph.nodes import _reply_sources

    state = CallState(
        conversation_id="c-test",
        last_search=[{"heading": "After a filling", "text": "Your lip may feel numb."}],
    )
    # The fixed lines are added by check_reply itself, so only the search's text comes back.
    assert _reply_sources(state, None) == ["After a filling Your lip may feel numb."]


def test_the_trace_shows_a_second_opinion():
    from dental_agent.graph.nodes import _trace_checker
    from dental_agent.storage.trace import TraceSink

    trace = TraceSink()
    deps = SimpleNamespace(trace=trace, checker=SimpleNamespace(last_attempts=[]))
    answer = CheckAnswer(
        layer="jev",
        answers={"claims_change": False, "medical_advice": True},
        probabilities={"claims_change": 0.02, "medical_advice": 0.41},
        model="m",
        second_opinions={"medical_advice": True},
    )
    _trace_checker(deps, "check_model_reply", answer)
    assert trace.events[-1]["second_opinions"] == {"medical_advice": True}
