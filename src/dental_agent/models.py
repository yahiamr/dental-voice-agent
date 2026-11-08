"""Every shared data type. Stage: 1.

What this file is:
    The shapes of the data passed between parts of the system.

What it contains:
    - Booking data: Details, Offer, Appointment, PendingAction, ActionRecord, ActionContext.
    - Tool data: ToolCall, ToolResult, ToolOutcome, RoundResult.
    - Other results: ReplyDecision, SearchResult.
    - Enums: ErrorCode, ToolExit, FixedReason, EndReason, Outcome, UrgentLevel.

How to understand it:
    Read this file early, because every other file uses these names. A PendingAction is the key
    type: one booking, move or cancel that is waiting for a yes. Each Offer and Appointment has
    a short ID for the model, such as s1, and a separate database ID the model never sees. This
    file imports nothing from LangChain or LangGraph, which lets guards/ and booking/ use it
    without loading a framework.

Used by:
    Almost every file
"""

from datetime import date, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class ErrorCode(StrEnum):
    """Why a tool or a check refused. Each code has one message under Tool Contracts."""

    INVALID_INPUT = "invalid_input"
    TOOL_FAILED = "tool_failed"
    SKIPPED = "skipped"
    INVALID_NAME = "invalid_name"
    UNKNOWN_SERVICE = "unknown_service"
    INVALID_WINDOW = "invalid_window"
    NO_SLOTS = "no_slots"
    NO_APPOINTMENT = "no_appointment"
    UNKNOWN_TOPIC = "unknown_topic"
    SEARCH_UNAVAILABLE = "search_unavailable"
    SLOT_NOT_OFFERED = "slot_not_offered"
    SLOT_TAKEN = "slot_taken"
    MISSING_FIELD = "missing_field"
    FIRST_VISIT_REQUIRED = "first_visit_required"
    EXISTING_PATIENT = "existing_patient"  # An existing patient cannot book a first visit.
    FIRST_VISIT_BOOKED = "first_visit_booked"  # A new patient already has one booked.
    NAME_ON_RECORD = "name_on_record"  # The name of a number on record cannot change.
    NOT_OWNER = "not_owner"
    POLICY_BLOCKED = "policy_blocked"
    UNSUPPORTED_REQUEST = "unsupported_request"


class ToolExit(StrEnum):
    """Why a tool round stopped the agent loop."""

    NONE = "none"  # Nothing special, so the model gets another round.
    PREPARED = "prepared"  # An action was staged, so code reads it back.
    URGENT = "urgent"  # A life-threatening flag was raised.
    END_CALL = "end_call"  # The agent ended the call.
    ROUND_LIMIT = "round_limit"  # Too many rounds, or the same rejected call twice.


class FixedReason(StrEnum):
    """Why use_fixed_line ran. Each reason picks a line."""

    URGENT = "urgent"
    MODEL_FAILURE = "model_failure"
    DEADLINE = "deadline"
    ROUND_LIMIT = "round_limit"
    COMMIT_REFUSED = "commit_refused"
    COMMIT_FAILED = "commit_failed"


class EndReason(StrEnum):
    """Why a call ended. Each value is a row of the Endings table."""

    DONE = "done"
    CALLER_LEFT = "caller_left"
    URGENT = "urgent"
    SILENCE = "silence"
    FAILURES = "failures"
    TURN_LIMIT = "turn_limit"
    ROUND_LIMIT = "round_limit"
    COMMIT_FAILED = "commit_failed"
    CALENDAR_FAILED = "calendar_failed"
    CRASH = "crash"


class Outcome(StrEnum):
    """The result of a call, stored in the conversation file."""

    COMPLETED = "completed"
    NO_ACTION = "no_action"
    URGENT = "urgent"
    ABANDONED = "abandoned"
    FAILED = "failed"


class UrgentLevel(StrEnum):
    """How urgent a call is. Only life_threatening ends the call."""

    DENTAL = "dental"
    LIFE_THREATENING = "life_threatening"


class Details(BaseModel):
    """What the caller has told us so far. Every field is optional until it is known."""

    name: str | None = None  # From the record for a known number, or as the caller said it.
    # Set by code at Start from the number, never by the model: one number is one patient.
    patient_type: Literal["new", "existing"] | None = None
    service: str | None = None  # A service ID from clinic.yaml.
    preferred_days: list[date] = Field(default_factory=list)
    time_window: Literal["morning", "afternoon", "any"] | None = None

    def missing_for_booking(self) -> list[str]:
        """Return the fields a booking still needs, as words the agent can use."""
        # The patient type is not asked, because code sets it from the number at Start.
        needed = {"name": self.name, "service": self.service}
        return [label for label, value in needed.items() if not value]


class Slot(BaseModel):
    """One bookable time, as stored in the calendar."""

    db_slot_id: int
    start_at: datetime
    end_at: datetime


class Offer(BaseModel):
    """A slot shown to the model. It sees slot_id, and never the database ID."""

    slot_id: str  # A per-call ID such as s1, which never repeats in a call.
    db_slot_id: int
    service: str
    spoken: str  # The exact text for this time. reply_check looks for this string.


class Appointment(BaseModel):
    """An appointment. appointment_id stays empty until a tool gives it a per-call ID."""

    appointment_id: str = ""  # A per-call ID such as a1.
    db_id: int
    patient_name: str
    service: str
    db_slot_id: int
    start_at: datetime
    status: Literal["active", "cancelled"]
    spoken: str = ""


class PendingAction(BaseModel):
    """One change waiting for a yes. Only one can exist at a time."""

    action_id: str  # A UUID. A repeated commit with this ID returns the first result.
    kind: Literal["book", "move", "cancel"]
    appointment: Appointment | None = None  # The target of a move or cancel.
    offer: Offer | None = None  # The new slot for a book or move.
    same_service_booked: bool = False  # Adds "in addition to your existing appointment".
    confirmation_text: str = ""  # Filled in by the ask_confirmation node.


class ActionRecord(BaseModel):
    """One change written to the calendar, as stored in the actions table."""

    action_id: str
    conversation_id: str
    kind: Literal["book", "move", "cancel"]
    appointment_id: int
    old_slot_id: int | None = None
    new_slot_id: int | None = None
    spoken: str  # The text the claim rule looks for.
    result_text: str  # The full sentence, so a repeated commit can return it.
    created_at: str


class ActionContext(BaseModel):
    """The facts validate_action needs, taken from CallState."""

    conversation_id: str
    caller_phone: str
    details: Details
    offers: list[Offer]
    found_appointments: list[Appointment]


class ToolResult(BaseModel):
    """What a tool returns to the model."""

    ok: bool
    data: Any = None
    error_code: ErrorCode | None = None
    message: str | None = None  # Tells the model what to do next.


class ToolOutcome(BaseModel):
    """A tool's result, plus the CallState fields it wants changed."""

    result: ToolResult
    updates: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    """One tool call taken from the model's response."""

    id: str
    name: str
    args: dict[str, Any]


class RoundResult(BaseModel):
    """Everything one tool round produced."""

    results: list[tuple[str, ToolResult]]  # Pairs of tool call ID and result.
    updates: dict[str, Any]
    exit: ToolExit


class ReplyDecision(BaseModel):
    """What check_reply or check_lead_in decided about the model's text."""

    text: str
    replaced: bool
    # Why the text was replaced or dropped. "question", "time" and "service" apply to a lead-in
    # only. "phone" means a clinic number was corrected and the text still passed.
    reason: (
        Literal[
            "claim",
            "dosing",
            "question",
            "time",
            "service",
            "model_claim",
            "model_advice",
            "phone",
        ]
        | None
    ) = None


class SearchResult(BaseModel):
    """One document section returned by a search."""

    text: str
    source: str
    score: float
