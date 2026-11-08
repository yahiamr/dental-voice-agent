"""Tools that change call state. Stage: 1.

What this file is:
    Tools that update what the call knows, with no calendar access.

What it contains:
    - save_details: stores what the caller said. It rejects names with digits or more than 4
      words, and for a number on record any name other than the one on record. It cannot set
      the patient type, which comes from the number.
    - flag_urgent: records dental or life_threatening.
    - end_call: ends the call.

How to understand it:
    flag_urgent covers emergencies in wording outside the phrase list, so guards/urgent.py is
    not the only way to catch one.

Uses:
    models.py, state.py

Used by:
    tools/__init__.py
"""

import re

from dental_agent.deps import GraphDeps
from dental_agent.models import ErrorCode, ToolOutcome, ToolResult, UrgentLevel
from dental_agent.state import CallState
from dental_agent.tools.schemas import EndCallInput, FlagUrgentInput, SaveDetailsInput, failure

MAX_NAME_WORDS = 4  # A longer "name" is probably a misheard sentence.


def save_details(args: SaveDetailsInput, state: CallState, deps: GraphDeps) -> ToolOutcome:
    """Merge the fields the model sent into the details collected so far."""
    # exclude_unset keeps only what the model sent, so old values are not wiped.
    sent = args.model_dump(exclude_unset=True, exclude_none=True)

    # Names are stored as heard, but a name with digits or many words is a mistake.
    name = sent.get("name")
    if name is not None and (re.search(r"\d", name) or len(name.split()) > MAX_NAME_WORDS):
        return failure(ErrorCode.INVALID_NAME, deps.clinic)

    # One number is one patient, so the name on record never changes during a call.
    on_record = state.details.name if state.patient_record else None
    if name is not None and on_record and name.strip().casefold() != on_record.casefold():
        return failure(ErrorCode.NAME_ON_RECORD, deps.clinic, name=on_record)

    service = sent.get("service")
    if service is not None and deps.clinic.service(service) is None:
        return failure(ErrorCode.UNKNOWN_SERVICE, deps.clinic)

    details = state.details.model_copy(update=sent)
    result = ToolResult(ok=True, data=details.model_dump(mode="json"))
    return ToolOutcome(result=result, updates={"details": details})


def flag_urgent(args: FlagUrgentInput, state: CallState, deps: GraphDeps) -> ToolOutcome:
    """Record how urgent the call is, for wording the phrase check cannot catch."""
    level = UrgentLevel(args.level)
    return ToolOutcome(
        result=ToolResult(ok=True, data={"level": level}), updates={"urgent_level": level}
    )


def end_call(args: EndCallInput, state: CallState, deps: GraphDeps) -> ToolOutcome:
    """End the call. The dispatcher sees this tool's name and stops the round."""
    return ToolOutcome(result=ToolResult(ok=True))
