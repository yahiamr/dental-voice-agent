"""Tests for the tools. Stage: 1, 4 and 5.

What this file is:
    Proves each tool follows its contract.

What it contains:
    - Every tool's output and message.
    - Offer replacement and IDs that never repeat.
    - no_slots clearing offers.
    - The lookup cap and name rejection.
    - invalid_input, tool_failed and skipped calls.
    - The repeat rule.

Used by:
    pytest
"""

from tests.conftest import TEST_NOW

from dental_agent.models import Details, ErrorCode, ToolCall, ToolExit
from dental_agent.state import CallState, apply_updates
from dental_agent.tools.dispatcher import dispatch, run_tool_round

WINDOW = {"service": "first_visit", "date_from": "2026-11-16", "date_to": "2026-11-20"}
READY = Details(name="Ana Ruiz", patient_type="new", service="first_visit")


def call(name, args=None, call_id="1"):
    """Build one tool call."""
    return ToolCall(id=call_id, name=name, args=args or {})


def run(name, state, deps, args=None):
    """Run one tool through dispatch, as the graph would."""
    return dispatch(call(name, args), state, deps.graph)


def searched(state, deps):
    """Return state after a successful slot search."""
    return apply_updates(state, run("find_slots", state, deps, WINDOW).updates)


def test_find_slots_returns_two_earliest_slots(state, deps):
    outcome = run("find_slots", state, deps, WINDOW)
    assert outcome.result.ok
    assert [item["slot_id"] for item in outcome.result.data] == ["s1", "s2"]
    assert "November" in outcome.result.data[0]["spoken"]


def test_find_slots_hides_the_database_id_from_the_model(state, deps):
    outcome = run("find_slots", state, deps, WINDOW)
    assert set(outcome.result.data[0]) == {"slot_id", "spoken"}


def test_a_new_search_replaces_earlier_offers(state, deps):
    first = searched(state, deps)
    second = apply_updates(first, run("find_slots", first, deps, WINDOW).updates)
    assert len(second.offers) == 2
    assert [item.slot_id for item in second.offers] == ["s3", "s4"]


def test_slot_ids_never_repeat_in_a_call(state, deps):
    first = searched(state, deps)
    second = searched(first, deps)
    assert not {item.slot_id for item in first.offers} & {item.slot_id for item in second.offers}


def test_no_slots_clears_offers(state, deps):
    with_offers = searched(state, deps)
    saturday = {"service": "first_visit", "date_from": "2026-11-14", "date_to": "2026-11-14"}
    outcome = run("find_slots", with_offers, deps, saturday)
    assert outcome.result.error_code is ErrorCode.NO_SLOTS
    assert outcome.updates["offers"] == []


def test_find_slots_rejects_an_unknown_service(state, deps):
    args = WINDOW | {"service": "brain_surgery"}
    assert run("find_slots", state, deps, args).result.error_code is ErrorCode.UNKNOWN_SERVICE


def test_find_slots_rejects_a_window_that_ends_before_it_starts(state, deps):
    args = WINDOW | {"date_from": "2026-11-20", "date_to": "2026-11-16"}
    assert run("find_slots", state, deps, args).result.error_code is ErrorCode.INVALID_WINDOW


def test_find_slots_rejects_a_window_in_the_past(state, deps):
    args = WINDOW | {"date_from": "2026-10-01", "date_to": "2026-10-02"}
    assert run("find_slots", state, deps, args).result.error_code is ErrorCode.INVALID_WINDOW


def test_find_slots_rejects_a_window_past_the_booking_limit(state, deps):
    args = WINDOW | {"date_from": "2027-03-01", "date_to": "2027-03-05"}
    assert run("find_slots", state, deps, args).result.error_code is ErrorCode.INVALID_WINDOW


def test_find_slots_respects_the_time_window(state, deps):
    updated = apply_updates(
        state, run("find_slots", state, deps, WINDOW | {"time_window": "afternoon"}).updates
    )
    slot = deps.graph.calendar.get_slot(updated.offers[0].db_slot_id)
    assert slot.start_at.hour >= 15


def test_lookup_returns_this_callers_appointments(deps):
    state = CallState(conversation_id="c", caller_phone="+34600000001")
    outcome = run("lookup_appointment", state, deps)
    assert outcome.result.ok
    assert outcome.result.data[0]["appointment_id"] == "a1"
    assert outcome.updates["found_appointments"][0].spoken


def test_lookup_with_no_appointment_returns_no_appointment(state, deps):
    assert run("lookup_appointment", state, deps).result.error_code is ErrorCode.NO_APPOINTMENT


def test_lookup_is_capped_at_five(deps):
    # The seed gives each patient one appointment, so this test adds six more.
    phone = "+34600000101"
    with deps.graph.calendar._connect() as connection:
        patient = connection.execute("SELECT id FROM patients WHERE phone = ?", (phone,)).fetchone()
        free = connection.execute(
            "SELECT id FROM slots WHERE start_at > ? AND id NOT IN "
            "(SELECT slot_id FROM appointments WHERE status = 'active') ORDER BY start_at LIMIT 6",
            (TEST_NOW.isoformat(),),
        ).fetchall()
        connection.executemany(
            "INSERT INTO appointments (patient_id, slot_id, service, status, created_at) "
            "VALUES (?, ?, 'cleaning', 'active', 'now')",
            [(patient[0], row[0]) for row in free],
        )
    state = CallState(conversation_id="c", caller_phone=phone)
    assert len(run("lookup_appointment", state, deps).result.data) == 5


def test_get_clinic_facts_returns_prices(state, deps):
    data = run("get_clinic_facts", state, deps, {"topic": "prices"}).result.data
    assert {"service": "check-up", "price": "40 euros"} in data


def test_get_clinic_facts_rejects_an_unknown_topic(state, deps):
    outcome = run("get_clinic_facts", state, deps, {"topic": "weather"})
    assert outcome.result.error_code is ErrorCode.UNKNOWN_TOPIC


def test_search_without_an_index_returns_search_unavailable(state, deps):
    outcome = run("search_clinic_info", state, deps, {"question": "Where can I park?"})
    assert outcome.result.error_code is ErrorCode.SEARCH_UNAVAILABLE
    assert outcome.result.message == deps.graph.clinic.line("no_search")


def test_save_details_merges_new_fields(state, deps):
    first = apply_updates(state, run("save_details", state, deps, {"name": "Ana Ruiz"}).updates)
    second = apply_updates(first, run("save_details", first, deps, {"service": "cleaning"}).updates)
    assert second.details.name == "Ana Ruiz"
    assert second.details.service == "cleaning"


def test_save_details_rejects_a_name_with_digits(state, deps):
    outcome = run("save_details", state, deps, {"name": "Ana 123"})
    assert outcome.result.error_code is ErrorCode.INVALID_NAME


def test_save_details_rejects_a_name_of_more_than_four_words(state, deps):
    outcome = run("save_details", state, deps, {"name": "one two three four five"})
    assert outcome.result.error_code is ErrorCode.INVALID_NAME


def test_save_details_rejects_an_unknown_service(state, deps):
    outcome = run("save_details", state, deps, {"service": "tattoo"})
    assert outcome.result.error_code is ErrorCode.UNKNOWN_SERVICE


def test_prepare_booking_stages_an_action_and_writes_nothing(state, deps):
    ready = searched(state.model_copy(update={"details": READY}), deps)
    slot_id = ready.offers[0].slot_id
    outcome = run("prepare_booking", ready, deps, {"slot_id": slot_id})
    assert outcome.result.ok
    assert outcome.updates["pending_action"].offer.slot_id == slot_id
    assert deps.graph.calendar.slot_is_free(ready.offers[0].db_slot_id)


def test_prepare_booking_rejects_an_invented_slot(state, deps):
    ready = searched(state.model_copy(update={"details": READY}), deps)
    outcome = run("prepare_booking", ready, deps, {"slot_id": "s99"})
    assert outcome.result.error_code is ErrorCode.SLOT_NOT_OFFERED


def test_prepare_booking_names_the_missing_fields(state, deps):
    ready = searched(state, deps)
    outcome = run("prepare_booking", ready, deps, {"slot_id": "s1"})
    assert outcome.result.error_code is ErrorCode.MISSING_FIELD
    assert "name" in outcome.result.message


def test_prepare_booking_notes_a_service_already_booked(deps):
    details = Details(name="Lucía Moreno", patient_type="existing", service="check_up")
    state = CallState(conversation_id="c", caller_phone="+34600000001", details=details)
    args = WINDOW | {"service": "check_up"}
    ready = apply_updates(state, run("find_slots", state, deps, args).updates)
    outcome = run("prepare_booking", ready, deps, {"slot_id": "s1"})
    assert outcome.updates["pending_action"].same_service_booked


def test_reschedule_is_unsupported_for_now(state, deps):
    args = {"appointment_id": "a1", "slot_id": "s1"}
    outcome = run("prepare_reschedule", state, deps, args)
    assert outcome.result.error_code is ErrorCode.UNSUPPORTED_REQUEST
    assert outcome.result.message == deps.graph.clinic.line("unsupported")


def test_cancel_is_unsupported_for_now(state, deps):
    outcome = run("prepare_cancel", state, deps, {"appointment_id": "a1"})
    assert outcome.result.error_code is ErrorCode.UNSUPPORTED_REQUEST


def test_flag_urgent_records_the_level(state, deps):
    outcome = run("flag_urgent", state, deps, {"level": "dental"})
    assert outcome.updates["urgent_level"] == "dental"


def test_bad_arguments_return_invalid_input(state, deps):
    outcome = run("find_slots", state, deps, {"service": "cleaning"})
    assert outcome.result.error_code is ErrorCode.INVALID_INPUT


def test_an_unknown_tool_returns_invalid_input(state, deps):
    assert run("delete_everything", state, deps).result.error_code is ErrorCode.INVALID_INPUT


def test_a_tool_that_raises_returns_tool_failed(state, deps, monkeypatch):
    def broken(*_):
        raise RuntimeError("database gone")

    monkeypatch.setattr(deps.graph.calendar, "free_slots", broken)
    assert run("find_slots", state, deps, WINDOW).result.error_code is ErrorCode.TOOL_FAILED


def test_save_details_then_prepare_works_in_one_response(state, deps):
    # Each call's updates apply before the next call runs.
    ready = searched(state, deps)
    calls = [
        call("save_details", {"name": "Ana", "patient_type": "new", "service": "first_visit"}, "1"),
        call("prepare_booking", {"slot_id": "s1"}, "2"),
    ]
    result = run_tool_round(calls, ready, deps.graph)
    assert result.exit is ToolExit.PREPARED
    assert result.updates["pending_action"] is not None


def test_a_call_after_a_prepare_is_skipped(state, deps):
    ready = searched(state.model_copy(update={"details": READY}), deps)
    calls = [
        call("prepare_booking", {"slot_id": "s1"}, "1"),
        call("find_slots", WINDOW, "2"),
    ]
    result = run_tool_round(calls, ready, deps.graph)
    assert result.results[1][1].error_code is ErrorCode.SKIPPED


def test_every_call_gets_a_result(state, deps):
    calls = [call("end_call", call_id="1"), call("find_slots", WINDOW, "2")]
    result = run_tool_round(calls, state, deps.graph)
    assert [pair[0] for pair in result.results] == ["1", "2"]


def test_end_call_runs_alone(state, deps):
    calls = [call("save_details", {"name": "Ana"}, "1"), call("end_call", call_id="2")]
    result = run_tool_round(calls, state, deps.graph)
    assert result.exit is ToolExit.END_CALL
    assert result.results[0][1].error_code is ErrorCode.SKIPPED
    assert "details" not in result.updates


def test_a_life_threatening_flag_beats_end_call(state, deps):
    calls = [call("end_call", call_id="1"), call("flag_urgent", {"level": "life_threatening"}, "2")]
    result = run_tool_round(calls, state, deps.graph)
    assert result.exit is ToolExit.URGENT
    assert result.results[0][1].error_code is ErrorCode.SKIPPED


def test_a_dental_flag_does_not_stop_the_round(state, deps):
    calls = [call("flag_urgent", {"level": "dental"}, "1"), call("find_slots", WINDOW, "2")]
    result = run_tool_round(calls, state, deps.graph)
    assert result.exit is ToolExit.NONE
    assert result.results[1][1].ok


def test_a_prepare_on_the_last_round_still_reads_back(state, deps):
    ready = searched(state.model_copy(update={"details": READY}), deps)
    last = ready.model_copy(update={"tool_rounds": 2})
    result = run_tool_round([call("prepare_booking", {"slot_id": "s1"})], last, deps.graph)
    assert result.exit is ToolExit.PREPARED


def test_reaching_the_round_limit_exits(state, deps):
    last = state.model_copy(update={"tool_rounds": 2})
    result = run_tool_round([call("find_slots", WINDOW)], last, deps.graph)
    assert result.exit is ToolExit.ROUND_LIMIT


def test_the_same_rejected_call_twice_exits(state, deps):
    bad = call("save_details", {"name": "Ana 123"})
    first = run_tool_round([bad], state, deps.graph)
    assert first.exit is ToolExit.NONE
    after = apply_updates(state, first.updates)
    assert run_tool_round([bad], after, deps.graph).exit is ToolExit.ROUND_LIMIT


def test_skipped_calls_do_not_count_as_rejections(state, deps):
    calls = [call("end_call", call_id="1"), call("find_slots", WINDOW, "2")]
    assert run_tool_round(calls, state, deps.graph).updates.get("rejected_calls", {}) == {}


KNOWN = "+34600000001"  # Lucía Moreno in the sample data.


def known_caller():
    """State for a caller whose number is on record, as the session sets it at Start."""
    details = Details(name="Lucía Moreno", patient_type="existing")
    return CallState(conversation_id="c", caller_phone=KNOWN, patient_record=True, details=details)


def test_find_slots_refuses_a_first_visit_for_an_existing_patient(deps):
    outcome = run("find_slots", known_caller(), deps, WINDOW)
    assert outcome.result.error_code is ErrorCode.EXISTING_PATIENT
    assert "offers" not in outcome.updates


def test_find_slots_refuses_other_services_for_a_new_patient(state, deps):
    outcome = run("find_slots", state, deps, WINDOW | {"service": "cleaning"})
    assert outcome.result.error_code is ErrorCode.FIRST_VISIT_REQUIRED


def test_find_slots_offers_other_services_to_an_existing_patient(deps):
    outcome = run("find_slots", known_caller(), deps, WINDOW | {"service": "cleaning"})
    assert outcome.result.ok


def test_save_details_cannot_change_the_name_on_record(deps):
    state = known_caller()
    outcome = run("save_details", state, deps, {"name": "Yaya Abu"})
    assert outcome.result.error_code is ErrorCode.NAME_ON_RECORD
    assert "Lucía Moreno" in outcome.result.message
    assert "details" not in outcome.updates


def test_save_details_accepts_the_name_on_record_again(deps):
    outcome = run("save_details", known_caller(), deps, {"name": "Lucía Moreno"})
    assert outcome.result.ok


def test_the_model_cannot_set_the_patient_type(state, deps):
    # The type comes from the number, so the tool ignores a type the model sends.
    outcome = run("save_details", state, deps, {"patient_type": "existing"})
    assert outcome.updates["details"].patient_type is None


def test_find_slots_refuses_a_second_first_visit_for_a_new_patient(deps):
    details = Details(name="Nora Blanco", patient_type="new")
    state = CallState(
        conversation_id="c", caller_phone="+34600000007", patient_record=True, details=details
    )
    outcome = run("find_slots", state, deps, WINDOW)
    assert outcome.result.error_code is ErrorCode.FIRST_VISIT_BOOKED
