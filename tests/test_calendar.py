"""Tests for the calendar. Stage: 1 and 3.

What this file is:
    Proves the database protects the bookings.

What it contains:
    - Double booking raises SlotTaken.
    - A slot can be rebooked after a cancel.
    - A repeated action ID returns the first result.
    - A failed move changes nothing.
    - move and cancel raise NotActive.
    - Seeding is repeatable.

Used by:
    pytest
"""

from datetime import date, timedelta

import pytest
from tests.conftest import PAST_NEW, TEST_NOW, add_past_first_visit

from dental_agent.booking.calendar import Calendar, NotActive, SlotTaken
from dental_agent.booking.seed import seed
from dental_agent.clock import MADRID, FixedClock
from dental_agent.models import Offer, PendingAction

NEW_PHONE = "+34600999999"
WEEK = (date(2026, 11, 16), date(2026, 11, 20))  # The Monday to Friday after TEST_NOW.


def free_slot(calendar):
    """Return one free slot in the test week."""
    return calendar.free_slots(*WEEK, None, None, 1, TEST_NOW)[0]


def pending(slot, action_id="action-1", service="check_up"):
    """Build a booking waiting for a yes."""
    offer = Offer(slot_id="s1", db_slot_id=slot.db_slot_id, service=service, spoken="a time")
    return PendingAction(action_id=action_id, kind="book", offer=offer)


def book(calendar, action, phone=NEW_PHONE, name="Ana Ruiz", conversation_id="c-1"):
    """Book with fixed text, so each test shows only what matters to it."""
    return calendar.book(
        action, conversation_id, phone, name, action.offer.service, "a time", "Booked.", TEST_NOW
    )


def test_booking_returns_a_record_and_takes_the_slot(calendar):
    slot = free_slot(calendar)
    record = book(calendar, pending(slot))
    assert record.kind == "book"
    assert record.new_slot_id == slot.db_slot_id
    assert not calendar.slot_is_free(slot.db_slot_id)


def test_double_booking_raises_slot_taken(calendar):
    slot = free_slot(calendar)
    book(calendar, pending(slot, "action-1"))
    with pytest.raises(SlotTaken):
        book(calendar, pending(slot, "action-2"), phone="+34600888888", name="Luis Mora")


def test_a_refused_booking_leaves_no_patient_row_behind(calendar):
    slot = free_slot(calendar)
    book(calendar, pending(slot, "action-1"))
    with pytest.raises(SlotTaken):
        book(calendar, pending(slot, "action-2"), phone="+34600888888", name="Luis Mora")
    assert not calendar.has_patient("+34600888888")


def test_a_repeated_action_id_returns_the_first_result(calendar):
    # A retried commit must not create a second appointment.
    slot = free_slot(calendar)
    first = book(calendar, pending(slot))
    second = book(calendar, pending(slot))
    assert second == first
    assert len(calendar.actions_for_call("c-1")) == 1


def test_a_repeat_new_caller_reuses_the_patient_row(calendar):
    # The same phone and name booking twice must not break the unique rule.
    first, second = calendar.free_slots(*WEEK, None, None, 2, TEST_NOW)
    book(calendar, pending(first, "action-1"))
    book(calendar, pending(second, "action-2"))
    assert calendar.patient_count(NEW_PHONE) == 1


def test_a_second_booking_on_one_number_keeps_the_first_name(calendar):
    # One number is one patient, so a later name does not create a second patient.
    first, second = calendar.free_slots(*WEEK, None, None, 2, TEST_NOW)
    book(calendar, pending(first, "action-1"), name="Ana Ruiz")
    book(calendar, pending(second, "action-2"), name="Pablo Ruiz")
    assert calendar.patient_count(NEW_PHONE) == 1
    assert calendar.patient_name(NEW_PHONE) == "Ana Ruiz"


def test_conversation_id_is_stored(calendar):
    book(calendar, pending(free_slot(calendar)), conversation_id="c-42")
    assert [item.conversation_id for item in calendar.actions_for_call("c-42")] == ["c-42"]


def test_get_action_returns_none_for_an_unknown_id(calendar):
    assert calendar.get_action("missing") is None


def test_appointments_for_lists_active_appointments_earliest_first(calendar):
    later, earlier = reversed(calendar.free_slots(*WEEK, None, None, 2, TEST_NOW))
    book(calendar, pending(later, "action-1"))
    book(calendar, pending(earlier, "action-2", service="cleaning"))
    found = calendar.appointments_for(NEW_PHONE, 5)
    assert [item.service for item in found] == ["cleaning", "check_up"]


def test_has_active_service_is_true_only_for_that_service(calendar):
    book(calendar, pending(free_slot(calendar)))
    assert calendar.has_active_service(NEW_PHONE, "check_up")
    assert not calendar.has_active_service(NEW_PHONE, "cleaning")


def test_free_slots_never_returns_a_taken_slot(calendar):
    slot = free_slot(calendar)
    book(calendar, pending(slot))
    found = calendar.free_slots(*WEEK, None, None, 500, TEST_NOW)
    assert slot.db_slot_id not in [item.db_slot_id for item in found]


def test_free_slots_keeps_to_the_time_window(calendar):
    found = calendar.free_slots(*WEEK, None, ("15:00", "19:00"), 500, TEST_NOW)
    assert found
    assert all(15 <= item.start_at.hour < 19 for item in found)


def test_free_slots_keeps_to_the_chosen_weekdays(calendar):
    found = calendar.free_slots(*WEEK, ["tue"], None, 500, TEST_NOW)
    assert found
    assert {item.start_at.strftime("%a") for item in found} == {"Tue"}


def test_free_slots_starts_at_the_earliest_allowed_time(calendar):
    earliest = TEST_NOW + timedelta(hours=2)
    found = calendar.free_slots(TEST_NOW.date(), TEST_NOW.date(), None, None, 500, earliest)
    assert found
    assert all(item.start_at >= earliest for item in found)


def test_free_slots_returns_the_earliest_first_up_to_the_limit(calendar):
    found = calendar.free_slots(*WEEK, None, None, 2, TEST_NOW)
    assert len(found) == 2
    assert found[0].start_at < found[1].start_at


def test_stored_times_keep_the_madrid_offset(calendar):
    assert free_slot(calendar).start_at.utcoffset() == TEST_NOW.astimezone(MADRID).utcoffset()


def test_seeding_is_repeatable(tmp_path, clinic):
    def rows(name):
        calendar = Calendar(tmp_path / name)
        seed(calendar, FixedClock(TEST_NOW), clinic)
        return calendar.appointments_for("+34600000001", 5), calendar.sample_patients(8)

    assert rows("one.sqlite") == rows("two.sqlite")


def test_the_first_sample_patient_has_one_check_up(calendar):
    phone, _name = calendar.sample_patients(1)[0]
    found = calendar.appointments_for(phone, 5)
    assert [item.service for item in found] == ["check_up"]


def test_seed_creates_eight_sample_patients(calendar):
    assert len(calendar.sample_patients(10)) == 8


def test_seed_skips_weekends_and_holidays(calendar):
    # 25 December 2026 is a Friday and a national holiday.
    day = date(2026, 12, 25)
    assert calendar.free_slots(day, day, None, None, 5, TEST_NOW) == []
    saturday = date(2026, 11, 14)
    assert calendar.free_slots(saturday, saturday, None, None, 5, TEST_NOW) == []


def test_seed_refuses_to_overwrite_data(calendar, clock, clinic):
    with pytest.raises(SystemExit):
        seed(calendar, clock, clinic)


def test_seed_with_reset_replaces_the_data(calendar, clock, clinic):
    book(calendar, pending(free_slot(calendar)))
    seed(calendar, clock, clinic, reset=True)
    assert not calendar.has_patient(NEW_PHONE)


KNOWN_PHONE = "+34600000001"  # Lucía Moreno in the sample data.


def test_patient_name_returns_the_name_on_record(calendar):
    assert calendar.patient_name(KNOWN_PHONE) == "Lucía Moreno"
    assert calendar.patient_name(NEW_PHONE) is None


def test_one_number_is_one_patient(calendar):
    # A booking under another name on a known number still belongs to the record's patient.
    calendar.book(
        pending(free_slot(calendar)), "c-1", KNOWN_PHONE, "Someone Else", "check_up",
        "a time", "Booked.", TEST_NOW,
    )  # fmt: skip
    assert calendar.patient_count(KNOWN_PHONE) == 1
    names = {item.patient_name for item in calendar.appointments_for(KNOWN_PHONE, 10)}
    assert names == {"Lucía Moreno"}


def test_the_database_refuses_a_second_patient_on_one_number(calendar):
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError), calendar._connect() as connection:
        connection.execute(
            "INSERT INTO patients (phone, name, created_at) VALUES (?, ?, ?)",
            (KNOWN_PHONE, "Another Name", "now"),
        )


def test_reset_rebuilds_the_tables_from_the_schema(tmp_path, clock, clinic):
    # An old file made two patients per number possible. Reset must apply the new schema.
    import sqlite3

    path = tmp_path / "old.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE patients (id INTEGER PRIMARY KEY, phone TEXT NOT NULL, "
            "name TEXT NOT NULL, created_at TEXT NOT NULL, UNIQUE (phone, name))"
        )
    calendar = Calendar(path)
    seed(calendar, clock, clinic, reset=True)
    with pytest.raises(sqlite3.IntegrityError), calendar._connect() as connection:
        connection.execute(
            "INSERT INTO patients (phone, name, created_at) VALUES (?, ?, ?)",
            (KNOWN_PHONE, "Another Name", "now"),
        )


NORA = "+34600000007"  # New, with a first visit booked ahead.
RAUL = "+34600000008"  # Existing, with one appointment ahead.


def set_status(calendar, phone, status):
    """Change every appointment on this number, as a cancel would."""
    with calendar._connect() as connection:
        connection.execute(
            "UPDATE appointments SET status = ? WHERE patient_id = "
            "(SELECT id FROM patients WHERE phone = ?)",
            (status, phone),
        )


def test_sample_patients_start_existing_or_new(calendar):
    assert calendar.patient_status(KNOWN_PHONE) == "existing"
    assert calendar.patient_status(NORA) == "new"
    assert calendar.patient_status(RAUL) == "existing"
    assert calendar.patient_status(NEW_PHONE) is None


def all_appointments(calendar):
    """Return (phone, start) for every appointment in the calendar."""
    with calendar._connect() as connection:
        return connection.execute(
            "SELECT patients.phone, slots.start_at FROM appointments "
            "JOIN patients ON patients.id = appointments.patient_id "
            "JOIN slots ON slots.id = appointments.slot_id"
        ).fetchall()


def test_the_seed_gives_every_patient_exactly_one_appointment(calendar):
    with calendar._connect() as connection:
        patients = [row[0] for row in connection.execute("SELECT phone FROM patients")]
    phones = [phone for phone, _start in all_appointments(calendar)]
    assert sorted(phones) == sorted(patients)


def test_the_seed_has_no_past_appointments(calendar):
    starts = [start for _phone, start in all_appointments(calendar)]
    assert all(start > TEST_NOW.isoformat() for start in starts)


def test_a_first_booking_creates_a_new_patient(calendar):
    book(calendar, pending(free_slot(calendar), service="first_visit"))
    assert calendar.patient_status(NEW_PHONE) == "new"


def test_a_past_first_visit_marks_the_patient_existing(calendar):
    add_past_first_visit(calendar)
    assert calendar.mark_attended(PAST_NEW, TEST_NOW, "first_visit")
    assert calendar.patient_status(PAST_NEW) == "existing"


def test_a_first_visit_still_ahead_leaves_the_patient_new(calendar):
    assert not calendar.mark_attended(NORA, TEST_NOW, "first_visit")
    assert calendar.patient_status(NORA) == "new"


def test_a_cancelled_first_visit_never_counts_as_attended(calendar):
    add_past_first_visit(calendar)
    set_status(calendar, PAST_NEW, "cancelled")
    assert not calendar.mark_attended(PAST_NEW, TEST_NOW, "first_visit")
    assert calendar.patient_status(PAST_NEW) == "new"


def test_an_existing_patient_stays_existing_with_no_appointments(calendar):
    set_status(calendar, KNOWN_PHONE, "cancelled")
    calendar.mark_attended(KNOWN_PHONE, TEST_NOW, "first_visit")
    assert calendar.patient_status(KNOWN_PHONE) == "existing"


# Stage 3: move and cancel.


def lucia_appointment(calendar):
    """Return Lucía's one seeded check-up."""
    return calendar.appointments_for(KNOWN_PHONE, 5)[0]


def change(kind, appointment, slot=None, action_id="change-1"):
    """Build a move or cancel waiting for a yes."""
    offer = None
    if slot is not None:
        offer = Offer(
            slot_id="s1", db_slot_id=slot.db_slot_id, service=appointment.service, spoken="later"
        )
    return PendingAction(action_id=action_id, kind=kind, appointment=appointment, offer=offer)


def move(calendar, action, conversation_id="c-1"):
    """Move with fixed text."""
    return calendar.move(action, conversation_id, "later", "Moved.", TEST_NOW)


def cancel(calendar, action, conversation_id="c-1"):
    """Cancel with fixed text."""
    return calendar.cancel(action, conversation_id, "a time", "Cancelled.", TEST_NOW)


def test_a_move_changes_the_slot_and_frees_the_old_one(calendar):
    appointment = lucia_appointment(calendar)
    slot = free_slot(calendar)
    record = move(calendar, change("move", appointment, slot))
    assert record.kind == "move"
    assert record.old_slot_id == appointment.db_slot_id
    assert record.new_slot_id == slot.db_slot_id
    assert record.appointment_id == appointment.db_id
    moved = lucia_appointment(calendar)
    assert moved.db_id == appointment.db_id  # The same row, so the record keeps its history.
    assert moved.db_slot_id == slot.db_slot_id
    assert calendar.slot_is_free(appointment.db_slot_id)


def test_a_move_to_a_taken_slot_raises_and_changes_nothing(calendar):
    appointment = lucia_appointment(calendar)
    slot = free_slot(calendar)
    book(calendar, pending(slot))
    with pytest.raises(SlotTaken):
        move(calendar, change("move", appointment, slot))
    assert lucia_appointment(calendar).db_slot_id == appointment.db_slot_id
    assert calendar.get_action("change-1") is None


def test_a_cancel_frees_the_slot_and_keeps_the_row(calendar):
    appointment = lucia_appointment(calendar)
    record = cancel(calendar, change("cancel", appointment))
    assert record.kind == "cancel"
    assert record.old_slot_id == appointment.db_slot_id
    assert record.new_slot_id is None
    assert calendar.appointments_for(KNOWN_PHONE, 5) == []
    assert calendar.slot_is_free(appointment.db_slot_id)
    with calendar._connect() as connection:
        row = connection.execute(
            "SELECT status FROM appointments WHERE id = ?", (appointment.db_id,)
        ).fetchone()
    assert row["status"] == "cancelled"


def test_a_cancelled_slot_can_be_booked_again(calendar):
    appointment = lucia_appointment(calendar)
    cancel(calendar, change("cancel", appointment))
    freed = calendar.get_slot(appointment.db_slot_id)
    book(calendar, pending(freed, "action-2"))
    assert not calendar.slot_is_free(appointment.db_slot_id)


def test_a_repeated_move_or_cancel_returns_the_first_result(calendar):
    appointment = lucia_appointment(calendar)
    first = cancel(calendar, change("cancel", appointment))
    assert cancel(calendar, change("cancel", appointment)) == first
    other = calendar.appointments_for(RAUL, 5)[0]
    action = change("move", other, free_slot(calendar), action_id="change-2")
    assert move(calendar, action) == move(calendar, action)


def test_move_and_cancel_raise_not_active_for_a_cancelled_appointment(calendar):
    appointment = lucia_appointment(calendar)
    set_status(calendar, KNOWN_PHONE, "cancelled")
    with pytest.raises(NotActive):
        cancel(calendar, change("cancel", appointment))
    with pytest.raises(NotActive):
        move(calendar, change("move", appointment, free_slot(calendar), action_id="change-2"))
    assert calendar.get_action("change-1") is None
    assert calendar.get_action("change-2") is None


def test_move_and_cancel_raise_not_active_when_the_appointment_moved_since_lookup(calendar):
    # The action holds the slot the call last saw. A row moved since then is left alone.
    appointment = lucia_appointment(calendar)
    move(calendar, change("move", appointment, free_slot(calendar)))
    moved_to = lucia_appointment(calendar).db_slot_id
    with pytest.raises(NotActive):
        cancel(calendar, change("cancel", appointment, action_id="change-2"))
    with pytest.raises(NotActive):
        move(calendar, change("move", appointment, free_slot(calendar), action_id="change-3"))
    assert lucia_appointment(calendar).db_slot_id == moved_to
    assert calendar.get_action("change-2") is None
    assert calendar.get_action("change-3") is None


def test_appointments_for_can_leave_out_past_appointments(calendar):
    add_past_first_visit(calendar)
    assert len(calendar.appointments_for(PAST_NEW, 5)) == 1
    assert calendar.appointments_for(PAST_NEW, 5, after=TEST_NOW) == []


def test_changes_are_listed_for_their_call(calendar):
    appointment = lucia_appointment(calendar)
    cancel(calendar, change("cancel", appointment), conversation_id="c-9")
    assert [item.kind for item in calendar.actions_for_call("c-9")] == ["cancel"]


def test_get_appointment_reads_the_row_in_any_status(calendar):
    appointment = lucia_appointment(calendar)
    assert calendar.get_appointment(appointment.db_id) == appointment
    set_status(calendar, KNOWN_PHONE, "cancelled")
    assert calendar.get_appointment(appointment.db_id).status == "cancelled"
    assert calendar.get_appointment(999999) is None
