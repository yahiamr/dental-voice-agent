"""Tests for message building. Stage: 1.

What this file is:
    Proves the model receives the right facts in the right order.

What it contains:
    - The state block's contents.
    - The message order.

Used by:
    pytest
"""

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from dental_agent.models import ActionRecord, Appointment, Details, Offer, PendingAction
from dental_agent.prompts.build import build_messages, state_block
from dental_agent.state import CallState

OFFER = Offer(slot_id="s1", db_slot_id=7, service="check_up", spoken="Tuesday 10 November at 10:00")


def action():
    """Build a stored booking."""
    return ActionRecord(
        action_id="a",
        conversation_id="c",
        kind="book",
        appointment_id=1,
        spoken="Tuesday 10 November at 10:00",
        result_text="You're booked.",
        created_at="now",
    )


def test_the_block_has_todays_date(clock, clinic):
    assert "Monday 9 November 2026" in state_block(CallState(), [], clock, clinic)


def test_the_block_lists_fourteen_days(clock, clinic):
    block = state_block(CallState(), [], clock, clinic)
    assert "Friday 20 November 2026: 2026-11-20" in block
    assert "2026-11-23" not in block


def test_the_block_shows_collected_details(clock, clinic):
    state = CallState(details=Details(name="Ana Ruiz", preferred_days=[date(2026, 11, 10)]))
    assert "Ana Ruiz" in state_block(state, [], clock, clinic)


def test_the_block_shows_offers_with_their_ids(clock, clinic):
    block = state_block(CallState(offers=[OFFER]), [], clock, clinic)
    assert "s1: Tuesday 10 November at 10:00" in block


def test_the_block_hides_database_ids(clock, clinic):
    assert "db_slot_id" not in state_block(CallState(offers=[OFFER]), [], clock, clinic)


def test_the_block_lists_actions_written_in_this_call(clock, clinic):
    assert "You're booked." in state_block(CallState(), [action()], clock, clinic)


def test_the_block_says_when_no_action_was_written(clock, clinic):
    assert "Actions written in this call: none" in state_block(CallState(), [], clock, clinic)


def test_the_block_shows_a_pending_action(clock, clinic):
    pending = PendingAction(action_id="x", kind="book", offer=OFFER)
    block = state_block(CallState(pending_action=pending), [], clock, clinic)
    assert "Pending action: book" in block


def test_the_block_names_an_existing_caller_and_their_services(clock, clinic):
    details = Details(name="Lucía Moreno", patient_type="existing")
    state = CallState(patient_record=True, details=details)
    block = state_block(state, [], clock, clinic)
    assert "Caller: Lucía Moreno, an existing patient" in block
    bookable = block.split("Services this caller can book:")[1].split("\n\n")[0]
    assert "cleaning" in bookable
    assert "first_visit" not in bookable


def test_the_block_shows_a_new_caller_can_book_only_a_first_visit(clock, clinic):
    block = state_block(CallState(details=Details(patient_type="new")), [], clock, clinic)
    assert "Caller: a new patient, name not given yet" in block
    bookable = block.split("Services this caller can book:")[1].split("\n\n")[0]
    assert "first_visit" in bookable
    assert "cleaning" not in bookable


def test_the_block_gives_the_urgent_line(clock, clinic):
    assert "910 000 002" in state_block(CallState(), [], clock, clinic)


def test_the_block_lists_service_ids(clock, clinic):
    # With the IDs here, the model needs no tool round to look them up.
    block = state_block(CallState(), [], clock, clinic)
    assert "first_visit: first-visit consultation" in block


def test_messages_are_in_the_right_order():
    worked = AIMessage(content="thinking")
    state = CallState(
        messages=[{"role": "caller", "text": "Hi"}, {"role": "bot", "text": "Hello"}],
        last_caller_text="A check-up please",
        turn_messages=[worked],
    )
    messages = build_messages(state, "facts")
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage) and messages[1].content == "Hi"
    assert isinstance(messages[2], AIMessage) and messages[2].content == "Hello"
    assert isinstance(messages[3], HumanMessage) and messages[3].content == "A check-up please"
    assert messages[4] is worked


def test_only_one_system_message_holds_the_prompt_and_the_block(clock, clinic):
    # Facts written by code must never look like the caller's words.
    state = CallState(last_caller_text="Hi")
    facts = state_block(state, [], clock, clinic)
    messages = build_messages(state, facts)
    assert sum(isinstance(item, SystemMessage) for item in messages) == 1
    assert "Mirador Dental" in messages[0].content
    assert "Current Facts" in messages[0].content
    assert "Monday 9 November 2026" in messages[0].content


def test_the_block_says_a_booked_first_visit_blocks_other_bookings(clock, clinic):
    details = Details(name="Nora Blanco", patient_type="new")
    state = CallState(patient_record=True, first_visit_booked=True, details=details)
    block = state_block(state, [], clock, clinic)
    assert "Caller: Nora Blanco, a new patient with a first-visit consultation booked" in block
    bookable = block.split("Services this caller can book:")[1].split("\n\n")[0]
    assert "none until their first visit" in bookable


def found(db_id, spoken):
    """An appointment found for the caller, with its per-call ID."""
    return Appointment(
        appointment_id=f"a{db_id}",
        db_id=db_id,
        patient_name="Lucía Moreno",
        service="check-up",
        db_slot_id=db_id,
        start_at=datetime(2026, 11, 10, 10, 0, tzinfo=ZoneInfo("Europe/Madrid")),
        status="active",
        spoken=spoken,
    )


def test_a_found_appointment_booked_in_this_call_is_marked(clock, clinic):
    # After a booking, the caller's appointments are reloaded, so the new one is listed twice:
    # once as found and once as an action. The mark stops the model reading two bookings.
    state = CallState(
        found_appointments=[found(1, "Tuesday 10 November at 10:00"), found(2, "Friday 13")]
    )
    block = state_block(state, [action()], clock, clinic)
    assert (
        "a1: check-up for Lucía Moreno on Tuesday 10 November at 10:00 (booked in this call)"
        in block
    )
    assert "a2: check-up for Lucía Moreno on Friday 13\n" in block + "\n"


def test_a_found_appointment_inside_the_notice_period_is_marked(clock, clinic):
    # Tool results are dropped when a turn ends. Without this mark, a real call showed the model
    # taking back a correct refusal on the next turn and offering the cancel again.
    soon = found(1, "Monday 9 November at 16:00").model_copy(
        update={"start_at": clock.now() + timedelta(hours=6)}
    )
    later = found(2, "Tuesday 10 November at 10:00")  # Exactly 24 hours ahead, so outside.
    state = CallState(found_appointments=[soon, later])
    block = state_block(state, [], clock, clinic)
    assert (
        "a1: check-up for Lucía Moreno on Monday 9 November at 16:00 (inside the 24-hour "
        "notice period, so it cannot be moved or cancelled on this line)" in block
    )
    assert "a2: check-up for Lucía Moreno on Tuesday 10 November at 10:00\n" in block + "\n"


def test_an_appointment_booked_in_this_call_has_no_notice_mark(clock, clinic):
    soon = found(1, "Monday 9 November at 16:00").model_copy(
        update={"start_at": clock.now() + timedelta(hours=6)}
    )
    block = state_block(CallState(found_appointments=[soon]), [action()], clock, clinic)
    assert "(booked in this call)" in block
    assert "notice period, so" not in block
