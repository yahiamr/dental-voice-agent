"""Input models for each tool. Stage: 1 and 3.

What this file is:
    A Pydantic class per tool, describing the arguments the model must send.

What it contains:
    - An input model for each of the 10 tools under Tool Contracts.
    - FindSlotsInput.appointment_id, set when the search is for moving a found appointment, and
      FindSlotsInput.time, an exact HH:MM time the caller asked for.

How to understand it:
    These classes produce the tool schemas sent to the model, and they check what comes back.
    Fields use plain strings on purpose. If service were an enum, a wrong value would fail as
    bad input. As a plain string it reaches the tool, which can return unknown_service with a
    helpful message.

Used by:
    tools/__init__.py, tools/dispatcher.py
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from dental_agent.clinic.data import Clinic
from dental_agent.models import ErrorCode, ToolOutcome, ToolResult

# What the model is told after each error, so it knows what to do next.
ERROR_MESSAGES = {
    ErrorCode.INVALID_INPUT: "Those tool arguments are not valid. Check the fields and try again.",
    ErrorCode.TOOL_FAILED: (
        "That tool failed. Say sorry briefly, and try once more or offer the clinic number."
    ),
    ErrorCode.SKIPPED: "Not run, because an earlier call in this response ended the round.",
    ErrorCode.INVALID_NAME: "That name could not be stored. Ask the caller to repeat it.",
    ErrorCode.UNKNOWN_SERVICE: ("That service is not offered. Use get_clinic_facts for the list."),
    ErrorCode.INVALID_WINDOW: (
        "Those dates are not bookable. Ask for dates within the booking window."
    ),
    ErrorCode.NO_SLOTS: "Nothing is free then. Offer to search later dates.",
    ErrorCode.NO_APPOINTMENT: (
        "This number has no appointment still to come. One that has already started cannot be "
        "changed here. You cannot search by name or by another number, so do not offer to. Say "
        "they can call again from the number they booked with, or call the clinic."
    ),
    ErrorCode.UNKNOWN_TOPIC: "Use services, prices, hours or rules.",
    ErrorCode.SLOT_NOT_OFFERED: "That slot is not on offer for this service. Search again.",
    ErrorCode.SLOT_TAKEN: "That slot has gone. Search again.",
    ErrorCode.MISSING_FIELD: "Collect {fields} first.",
    ErrorCode.FIRST_VISIT_REQUIRED: (
        "New patients must book the first-visit consultation. Explain this and offer one."
    ),
    ErrorCode.EXISTING_PATIENT: (
        "This caller is already a patient, so they cannot book a first-visit consultation. "
        "Offer a check-up or another service."
    ),
    ErrorCode.FIRST_VISIT_BOOKED: (
        "This new patient already has a first-visit consultation booked, and can book nothing "
        "else until they attend it. Use lookup_appointment to tell them when it is."
    ),
    ErrorCode.NAME_ON_RECORD: (
        "This number belongs to {name}, and the name cannot change. You can only book for "
        "{name}. Anyone else should call from their own number or call the clinic."
    ),
    ErrorCode.NOT_OWNER: "That appointment is not available to this caller.",
    ErrorCode.POLICY_BLOCKED: (
        "This is within the notice period. Explain this and give the clinic number."
    ),
}

# These two messages are the caller-facing lines, so they come from clinic.yaml.
CLINIC_LINE_MESSAGES = {
    ErrorCode.SEARCH_UNAVAILABLE: "no_search",
    ErrorCode.UNSUPPORTED_REQUEST: "unsupported",
}


def failure(code: ErrorCode, clinic: Clinic, updates: dict | None = None, **values) -> ToolOutcome:
    """Build the outcome for a refused tool call, with that error's one message."""
    if code in CLINIC_LINE_MESSAGES:
        message = clinic.line(CLINIC_LINE_MESSAGES[code])
    else:
        message = ERROR_MESSAGES[code].format(**values)
    result = ToolResult(ok=False, error_code=code, message=message)
    return ToolOutcome(result=result, updates=updates or {})


class SaveDetailsInput(BaseModel):
    """Store what the caller has said. Send only the fields you learned."""

    # There is no patient type here: code sets it from the number, so the model cannot.
    name: str | None = None
    service: str | None = Field(None, description="A service ID, such as check_up.")
    preferred_days: list[date] | None = None
    time_window: Literal["morning", "afternoon", "any"] | None = None


class FindSlotsInput(BaseModel):
    """Find the earliest free slots for a service. A new search replaces earlier offers."""

    service: str = Field(description="A service ID, such as check_up.")
    date_from: date
    date_to: date
    weekdays: list[Literal["mon", "tue", "wed", "thu", "fri"]] | None = None
    time_window: Literal["morning", "afternoon", "any"] = "any"
    appointment_id: str | None = Field(
        default=None,
        description="Set when searching for a new time for this found appointment.",
    )
    time: str | None = Field(
        default=None,
        pattern=r"^([01]?[0-9]|2[0-3]):[0-5][0-9]$",
        description="Set when the caller asks for one exact time, as HH:MM, such as 10:00.",
    )

    @field_validator("time")
    @classmethod
    def _two_digit_hour(cls, value: str | None) -> str | None:
        """Write 9:00 as 09:00, so it compares equal to the stored times."""
        if value is None:
            return None
        hour, minute = value.split(":")
        return f"{int(hour):02d}:{minute}"


class LookupAppointmentInput(BaseModel):
    """List this caller's active appointments."""


class GetClinicFactsInput(BaseModel):
    """Get facts from the clinic's data file."""

    # A plain string, so a bad topic reaches the tool and returns unknown_topic.
    topic: str = Field(description="One of: services, prices, hours, rules.")


class SearchClinicInfoInput(BaseModel):
    """Search the clinic's documents. Use key terms and synonyms."""

    question: str


class PrepareBookingInput(BaseModel):
    """Stage a booking for an offered slot. Code reads it back and asks for a yes."""

    slot_id: str


class PrepareRescheduleInput(BaseModel):
    """Stage moving a found appointment to an offered slot."""

    appointment_id: str
    slot_id: str


class PrepareCancelInput(BaseModel):
    """Stage cancelling a found appointment."""

    appointment_id: str


class FlagUrgentInput(BaseModel):
    """Record an urgent call. life_threatening ends the call with the emergency message."""

    level: Literal["dental", "life_threatening"]


class EndCallInput(BaseModel):
    """End the call, once the caller has finished."""
