"""CallState: the memory of one call. Stage: 1.

What this file is:
    The Pydantic model that holds everything LangGraph remembers about a call between turns.

What it contains:
    - CallState: identity, caller details, offers, found appointments, the pending action, turn
      counters, the reply and the messages.
    - to_context(), which builds the ActionContext that validate_action needs.
    - found(appointment_id), the found appointment with this per-call ID, or None.
    - Derived helpers such as missing_details().

How to understand it:
    The checkpointer saves this object after each step and loads it for the next turn, under a
    thread_id equal to the conversation ID. Every field replaces its old value, so there is no
    merging logic to learn. Two message lists matter: messages is the spoken history kept for
    the whole call, and turn_messages is working data for one turn. Actions are not stored here.
    They are read from the calendar, so state cannot disagree with it. The file sits outside
    graph/ so that tools/ can import it without an import cycle.

Uses:
    models.py

Used by:
    graph/, tools/, prompts/build.py, session.py
"""

from datetime import datetime
from typing import Any

from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field

from dental_agent.models import (
    ActionContext,
    Appointment,
    Details,
    EndReason,
    FixedReason,
    Offer,
    PendingAction,
    ToolExit,
    UrgentLevel,
)


class CallState(BaseModel):
    """Everything the graph remembers about one call. Every field replaces its old value."""

    # Identity and position.
    conversation_id: str = ""
    turn: int = 0
    caller_phone: str = ""  # Tools read the number from here, so the model cannot supply one.
    patient_record: bool = False  # True if this number has a record, set at Start.
    # True if a new patient already has a first visit booked, set at Start. Tools still check
    # the calendar, so this only helps the model explain.
    first_visit_booked: bool = False

    # The caller's latest words, kept apart from the message list.
    last_caller_text: str = ""

    # What the call has found out.
    details: Details = Field(default_factory=Details)
    offers: list[Offer] = Field(default_factory=list)  # A new search replaces these.
    found_appointments: list[Appointment] = Field(default_factory=list)

    # The change waiting for a yes, and whether its confirmation question was actually spoken.
    pending_action: PendingAction | None = None
    awaiting_yes: bool = False

    # How the call may end.
    urgent_level: UrgentLevel | None = None
    end_reason: EndReason | None = None

    # This turn's reply for the session.
    reply_text: str = ""
    asks_confirmation: bool = False
    fixed_reason: FixedReason | None = None

    # The spoken history as {role, text} entries, and this turn's working messages.
    messages: list[dict[str, str]] = Field(default_factory=list)
    turn_messages: list[AnyMessage] = Field(default_factory=list)

    # The Current Facts text for this turn. It stays fixed until the turn ends, because the
    # model's thinking blocks are tied to the exact system prompt they were made with.
    turn_facts: str = ""

    # Limits for this turn.
    tool_rounds: int = 0
    turn_deadline_at: datetime | None = None
    rejected_calls: dict[str, int] = Field(default_factory=dict)
    tool_exit: ToolExit = ToolExit.NONE

    # Counters for per-call IDs, so an ID never repeats in a call.
    next_slot_number: int = 1
    next_appointment_number: int = 1

    def to_context(self) -> ActionContext:
        """Build the facts validate_action needs."""
        return ActionContext(
            conversation_id=self.conversation_id,
            caller_phone=self.caller_phone,
            details=self.details,
            offers=self.offers,
            found_appointments=self.found_appointments,
        )

    def found(self, appointment_id: str) -> Appointment | None:
        """Return the appointment lookup_appointment gave this ID, or None for an invented one."""
        return next(
            (item for item in self.found_appointments if item.appointment_id == appointment_id),
            None,
        )

    def missing_details(self) -> list[str]:
        """Return what is still needed before a booking can be prepared."""
        return self.details.missing_for_booking()


def apply_updates(state: CallState, updates: dict[str, Any]) -> CallState:
    """Return a copy of state with these fields changed. Tools never change state themselves."""
    return state.model_copy(update=updates)
