"""Tools that stage an action. Stage: 1 and 4.

What this file is:
    The tools the model calls when the caller wants a change. They do not make the change.

What it contains:
    - prepare_booking, prepare_reschedule and prepare_cancel.

How to understand it:
    Each one calls validate_action and returns a PendingAction as an update. After that, code
    takes over: ask_confirmation speaks it, and write_to_calendar writes it only after a yes on
    the next turn.
    Only one pending action exists at a time. Until Stage 4, the reschedule and cancel tools
    return unsupported_request.

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
        include_name=True,  # A booking confirmation question always says the name.
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
    """Not available until Stage 4, so the agent says the clinic line."""
    return failure(ErrorCode.UNSUPPORTED_REQUEST, deps.clinic)


def prepare_cancel(args: PrepareCancelInput, state: CallState, deps: GraphDeps) -> ToolOutcome:
    """Not available until Stage 4, so the agent says the clinic line."""
    return failure(ErrorCode.UNSUPPORTED_REQUEST, deps.clinic)
