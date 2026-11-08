"""Runs the model's tool calls. Stage: 1.

What this file is:
    Takes every tool call from one model response and decides which ones run.

What it contains:
    - run_tool_round(calls, state, deps) -> RoundResult: handles a whole response.
    - dispatch(call, state, deps) -> ToolOutcome: runs one call safely.

How to understand it:
    One response can hold several calls, so the code scans them first and applies an exit order:
    urgent, end_call, prepared, round_limit. Each call's updates apply before the next one runs,
    so save_details then prepare_booking works in one response. A skipped call still gets a
    result, because every tool call needs a matching answer. Tools return updates and never
    change state themselves.

Uses:
    tools/__init__.py, tools/schemas.py, models.py, state.py

Used by:
    graph/nodes.py (run_tools)
"""

import json
import logging
from typing import Any

from pydantic import ValidationError

from dental_agent.deps import GraphDeps
from dental_agent.models import (
    ErrorCode,
    RoundResult,
    ToolCall,
    ToolExit,
    ToolOutcome,
    ToolResult,
)
from dental_agent.state import CallState, apply_updates
from dental_agent.tools import TOOLS
from dental_agent.tools.schemas import ERROR_MESSAGES, failure

logger = logging.getLogger(__name__)

# Every tool that stages an action.
PREPARE_TOOLS = {"prepare_booking", "prepare_reschedule", "prepare_cancel"}

# The same rejected call this many times in one turn counts as hitting the round limit.
MAX_SAME_REJECTION = 2


def run_tool_round(calls: list[ToolCall], state: CallState, deps: GraphDeps) -> RoundResult:
    """Run every tool call from one model response, and decide what happens next."""
    # Scan first, because one call can decide whether the others run at all.
    to_run, planned_exit = _plan_round(calls)

    working = state  # Each call's updates apply here, so the next call sees them.
    all_updates: dict[str, Any] = {}
    results: list[tuple[str, ToolResult]] = []
    rejected = dict(state.rejected_calls)
    repeated = False
    prepared = False

    for call in calls:
        # A skipped call still needs a result, because the next model call must have an
        # answer for every tool call.
        if call.id not in to_run:
            skipped = ToolResult(
                ok=False, error_code=ErrorCode.SKIPPED, message=ERROR_MESSAGES[ErrorCode.SKIPPED]
            )
            results.append((call.id, skipped))
            continue

        outcome = dispatch(call, working, deps)
        results.append((call.id, outcome.result))
        working = apply_updates(working, outcome.updates)
        all_updates.update(outcome.updates)

        if not outcome.result.ok:
            # Count this rejection, keyed by tool name and arguments.
            key = _call_key(call)
            rejected[key] = rejected.get(key, 0) + 1
            repeated = repeated or rejected[key] >= MAX_SAME_REJECTION
        elif call.name in PREPARE_TOOLS:
            prepared = True

    if rejected != state.rejected_calls:
        all_updates["rejected_calls"] = rejected

    # The first exit that applies wins, so a valid prepare on the last round still reads back.
    if planned_exit is not ToolExit.NONE:
        exit_reason = planned_exit
    elif prepared:
        exit_reason = ToolExit.PREPARED
    elif repeated or state.tool_rounds + 1 >= deps.settings.max_tool_rounds:
        exit_reason = ToolExit.ROUND_LIMIT
    else:
        exit_reason = ToolExit.NONE
    return RoundResult(results=results, updates=all_updates, exit=exit_reason)


def dispatch(call: ToolCall, state: CallState, deps: GraphDeps) -> ToolOutcome:
    """Run one tool call. It never raises, so one bad call cannot end the turn."""
    tool = TOOLS.get(call.name)
    if tool is None:
        outcome = failure(ErrorCode.INVALID_INPUT, deps.clinic)
    else:
        try:
            # Check the arguments against the tool's input model.
            args = tool.schema.model_validate(call.args)
            outcome = tool.run(args, state, deps)
        except ValidationError:
            outcome = failure(ErrorCode.INVALID_INPUT, deps.clinic)
        except Exception as error:  # Any tool failure must reach the model as a typed result.
            logger.warning("Tool %s failed: %s", call.name, error)
            outcome = failure(ErrorCode.TOOL_FAILED, deps.clinic)
    deps.trace.event(
        "tool_call",
        name=call.name,
        arguments=call.args,
        ok=outcome.result.ok,
        error_code=outcome.result.error_code,
    )
    return outcome


def _plan_round(calls: list[ToolCall]) -> tuple[set[str], ToolExit]:
    """Return the IDs of the calls that should run, and any exit already decided."""
    # 1: a life-threatening flag wins, and only that call runs.
    for call in calls:
        if call.name == "flag_urgent" and call.args.get("level") == "life_threatening":
            return {call.id}, ToolExit.URGENT
    # 2: if end_call is present, only that call runs.
    for call in calls:
        if call.name == "end_call":
            return {call.id}, ToolExit.END_CALL
    # 3: if a prepare is present, the calls before it run, then the prepare.
    running: set[str] = set()
    for call in calls:
        running.add(call.id)
        if call.name in PREPARE_TOOLS:
            break
    return running, ToolExit.NONE


def _call_key(call: ToolCall) -> str:
    """Return a text key for a tool name and arguments, used by the repeat rule."""
    return call.name + ":" + json.dumps(call.args, sort_keys=True, default=str)
