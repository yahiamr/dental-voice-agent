"""Tools that stage an action. Stage: 1 and 3.

What this file is:
    The tools the model calls when the caller wants a change. They do not make the change.

What it contains:
    - prepare_booking, prepare_reschedule and prepare_cancel.

How to understand it:
    Each one calls validate_action and returns a PendingAction as an update. After that, code
    takes over: ask_confirmation speaks it, and write_to_calendar writes it only after a yes on
    the next turn.
    Only one pending action exists at a time. A move or cancel names its appointment by the
    per-call ID that lookup_appointment gave, and a move names its new slot by an offer ID from
    find_slots. An ID the model made up finds nothing, and the action is refused.

Uses:
    guards/validate.py, state.py, models.py

Used by:
    tools/__init__.py
"""

from uuid import uuid4

from dental_agent.deps import GraphDeps
from dental_agent.guards.validate import validate_action
from dental_agent.models import ErrorCode, PendingAction, ToolOutcome, ToolResult
from dental_agent.state import CallState
from dental_agent.tools.schemas import (
    PrepareBookingInput,
    PrepareCancelInput,
    PrepareRescheduleInput,
    failure,
)


def prepare_booking(args: PrepareBookingInput, state: CallState, deps: GraphDeps) -> ToolOutcome:
    """Stage a booking for an offered slot. Nothing is written to the calendar here."""
    # Look the ID up in this call's offers, so an invented ID finds nothing.
    offer = next((item for item in state.offers if item.slot_id == args.slot_id), None)
    if offer is None:
        return failure(ErrorCode.SLOT_NOT_OFFERED, deps.clinic)

    action = PendingAction(
        action_id=str(uuid4()),
        kind="book",
        offer=offer,
        # This adds "in addition to your existing appointment" to the confirmation question.
        same_service_booked=deps.calendar.has_active_service(state.caller_phone, offer.service),
    )

    # Apply the same checks that write_to_calendar applies again on the next turn.
    error = validate_action(
        action, state.to_context(), deps.calendar, deps.clock.now(), deps.clinic
    )
    if error is ErrorCode.MISSING_FIELD:
        # Name the fields, so the agent knows exactly what to ask for.
        return failure(error, deps.clinic, fields=", ".join(state.missing_details()))
    if error:
        return failure(error, deps.clinic)

    result = ToolResult(ok=True, data={"action_id": action.action_id})
    return ToolOutcome(result=result, updates={"pending_action": action})


def prepare_reschedule(
    args: PrepareRescheduleInput, state: CallState, deps: GraphDeps
) -> ToolOutcome:
    """Stage moving a found appointment to an offered slot. Nothing is written here."""
    appointment = state.found(args.appointment_id)
    offer = next((item for item in state.offers if item.slot_id == args.slot_id), None)
    if appointment is not None and offer is None:
        return failure(ErrorCode.SLOT_NOT_OFFERED, deps.clinic)
    action = PendingAction(
        action_id=str(uuid4()), kind="move", appointment=appointment, offer=offer
    )
    return _stage(action, state, deps)


def prepare_cancel(args: PrepareCancelInput, state: CallState, deps: GraphDeps) -> ToolOutcome:
    """Stage cancelling a found appointment. Nothing is written here."""
    appointment = state.found(args.appointment_id)
    action = PendingAction(action_id=str(uuid4()), kind="cancel", appointment=appointment)
    return _stage(action, state, deps)


def _stage(action: PendingAction, state: CallState, deps: GraphDeps) -> ToolOutcome:
    """Check a move or cancel and return it as the pending action, or return the refusal.

    An appointment that was never found reaches validate_action as None, which refuses it with
    not_owner.
    """
    error = validate_action(
        action, state.to_context(), deps.calendar, deps.clock.now(), deps.clinic
    )
    if error:
        return failure(error, deps.clinic)
    result = ToolResult(ok=True, data={"action_id": action.action_id})
    return ToolOutcome(result=result, updates={"pending_action": action})
