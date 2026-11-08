"""Tests for action validation. Stage: 1 and 4.

What this file is:
    Proves each business rule and the order the checks run in.

What it contains:
    - One test per error code.
    - The check order.
    - A repeat new caller.
    - The notice exemption.
    - Each row of the first-visit table.

Used by:
    pytest
"""

from datetime import date, timedelta

import pytest
from tests.conftest import TEST_NOW

from dental_agent.guards.validate import validate_action
from dental_agent.models import ActionContext, Details, ErrorCode, Offer, PendingAction

NEW_PHONE = "+34600999999"
KNOWN_PHONE = "+34600000001"
NORA = "+34600000007"  # New, with a first visit booked ahead.
WEEK = (date(2026, 11, 16), date(2026, 11, 20))


def setup(calendar, service="first_visit", patient_type="new", phone=NEW_PHONE, name="Ana Ruiz"):
    """Build a valid booking, which each test then breaks in one way."""
    slot = calendar.free_slots(*WEEK, None, None, 1, TEST_NOW)[0]
    offer = Offer(slot_id="s1", db_slot_id=slot.db_slot_id, service=service, spoken="a time")
    action = PendingAction(action_id="x", kind="book", offer=offer)
    context = ActionContext(
        conversation_id="c-1",
        caller_phone=phone,
        details=Details(name=name, patient_type=patient_type, service=service),
        offers=[offer],
        found_appointments=[],
    )
    return action, context


def check(action, context, calendar, clinic, now=TEST_NOW):
    """Run validate_action."""
    return validate_action(action, context, calendar, now, clinic)


def test_a_valid_booking_passes(calendar, clinic):
    action, context = setup(calendar)
    assert check(action, context, calendar, clinic) is None


@pytest.mark.parametrize("field", ["name", "service"])
def test_a_missing_detail_returns_missing_field(calendar, clinic, field):
    action, context = setup(calendar)
    setattr(context.details, field, None)
    assert check(action, context, calendar, clinic) is ErrorCode.MISSING_FIELD


def test_a_slot_that_was_not_offered_is_refused(calendar, clinic):
    # An invented slot must be refused even if it is a real free slot.
    action, context = setup(calendar)
    context.offers = []
    assert check(action, context, calendar, clinic) is ErrorCode.SLOT_NOT_OFFERED


def test_an_offer_for_another_service_is_refused(calendar, clinic):
    action, context = setup(calendar)
    context.offers = [context.offers[0].model_copy(update={"service": "cleaning"})]
    assert check(action, context, calendar, clinic) is ErrorCode.SLOT_NOT_OFFERED


def test_a_taken_slot_is_refused(calendar, clinic):
    action, context = setup(calendar)
    calendar.book(
        action.model_copy(update={"action_id": "other"}),
        "c-0",
        "+34600111111",
        "Luis Mora",
        "first_visit",
        "a time",
        "Booked.",
        TEST_NOW,
    )
    assert check(action, context, calendar, clinic) is ErrorCode.SLOT_TAKEN


def test_a_slot_too_close_to_now_is_refused(calendar, clinic):
    # The slot was fine when offered, but the clock moved past the booking window.
    action, context = setup(calendar)
    slot = calendar.get_slot(action.offer.db_slot_id)
    late = slot.start_at - timedelta(hours=1)
    assert check(action, context, calendar, clinic, now=late) is ErrorCode.SLOT_TAKEN


def test_details_are_checked_before_the_slot(calendar, clinic):
    action, context = setup(calendar)
    context.details.name = None
    context.offers = []
    assert check(action, context, calendar, clinic) is ErrorCode.MISSING_FIELD


def test_the_first_visit_rule_comes_before_the_slot(calendar, clinic):
    action, context = setup(calendar, service="cleaning", patient_type="new")
    context.offers = []
    assert check(action, context, calendar, clinic) is ErrorCode.FIRST_VISIT_REQUIRED


@pytest.mark.parametrize(
    ("phone", "service", "expected"),
    [
        (KNOWN_PHONE, "cleaning", None),  # A number on record is an existing patient.
        (KNOWN_PHONE, "first_visit", ErrorCode.EXISTING_PATIENT),  # No first visit for them.
        (NEW_PHONE, "first_visit", None),  # A new number books a first visit.
        (NEW_PHONE, "cleaning", ErrorCode.FIRST_VISIT_REQUIRED),  # And only a first visit.
        (NORA, "cleaning", ErrorCode.FIRST_VISIT_REQUIRED),  # New until the first visit.
        (NORA, "first_visit", ErrorCode.FIRST_VISIT_BOOKED),  # One first visit at a time.
    ],
)
def test_the_number_decides_which_services_can_be_booked(
    calendar, clinic, phone, service, expected
):
    action, context = setup(calendar, service=service, phone=phone)
    assert check(action, context, calendar, clinic) is expected


def test_a_claimed_patient_type_cannot_change_the_rule(calendar, clinic):
    # Only the number counts, so "existing" on a new number still needs a first visit.
    action, context = setup(calendar, service="cleaning", patient_type="existing")
    assert check(action, context, calendar, clinic) is ErrorCode.FIRST_VISIT_REQUIRED


def test_a_caller_with_a_booked_first_visit_cannot_book_another(calendar, clinic):
    action, context = setup(calendar)
    calendar.book(
        action, "c-1", NEW_PHONE, "Ana Ruiz", "first_visit", "a time", "Booked.", TEST_NOW
    )
    other = calendar.free_slots(*WEEK, None, None, 1, TEST_NOW)[0]
    second = Offer(slot_id="s2", db_slot_id=other.db_slot_id, service="first_visit", spoken="b")
    context.offers = [second]
    repeat = PendingAction(action_id="y", kind="book", offer=second)
    assert check(repeat, context, calendar, clinic) is ErrorCode.FIRST_VISIT_BOOKED


def test_a_new_patient_who_cancelled_their_first_visit_can_book_one_again(calendar, clinic):
    with calendar._connect() as connection:
        connection.execute(
            "UPDATE appointments SET status = 'cancelled' WHERE patient_id = "
            "(SELECT id FROM patients WHERE phone = ?)",
            (NORA,),
        )
    action, context = setup(calendar, service="first_visit", phone=NORA)
    assert check(action, context, calendar, clinic) is None
