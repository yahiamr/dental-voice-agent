"""The tools the model can call. Stage: 1, 4 and 5.

What this file is:
    The registry of tools. Unlike the other __init__.py files, this one holds code.

What it contains:
    - TOOLS, which maps each name to its schema and function.

How to understand it:
    The call_model step passes these schemas to model.bind_tools. No tool writes the calendar,
    and no tool takes a phone number, because tools read the number from state. So the model
    cannot act on someone else's number.

Uses:
    tools/schemas.py, tools/read_tools.py, tools/action_tools.py, tools/session_tools.py

Used by:
    graph/nodes.py, tools/dispatcher.py
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from dental_agent.tools import action_tools, read_tools, schemas, session_tools


@dataclass
class Tool:
    """One tool: its input model, and the function that runs it."""

    schema: type[BaseModel]
    run: Callable


# Every tool the model can call. Nothing here writes to the calendar.
TOOLS: dict[str, Tool] = {
    "save_details": Tool(schemas.SaveDetailsInput, session_tools.save_details),
    "find_slots": Tool(schemas.FindSlotsInput, read_tools.find_slots),
    "lookup_appointment": Tool(schemas.LookupAppointmentInput, read_tools.lookup_appointment),
    "get_clinic_facts": Tool(schemas.GetClinicFactsInput, read_tools.get_clinic_facts),
    "search_clinic_info": Tool(schemas.SearchClinicInfoInput, read_tools.search_clinic_info),
    "prepare_booking": Tool(schemas.PrepareBookingInput, action_tools.prepare_booking),
    "prepare_reschedule": Tool(schemas.PrepareRescheduleInput, action_tools.prepare_reschedule),
    "prepare_cancel": Tool(schemas.PrepareCancelInput, action_tools.prepare_cancel),
    "flag_urgent": Tool(schemas.FlagUrgentInput, session_tools.flag_urgent),
    "end_call": Tool(schemas.EndCallInput, session_tools.end_call),
}


def tool_schemas() -> list[dict[str, Any]]:
    """Return the tools in the format model.bind_tools accepts, for any provider."""
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": tool.schema.__doc__,
                "parameters": tool.schema.model_json_schema(),
            },
        }
        for name, tool in TOOLS.items()
    ]
