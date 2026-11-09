"""One small function per graph step. Stage: 1, 2, 3 and 5.

What this file is:
    The steps of a turn. Each function reads state and returns only the fields it changes.

What it contains:
    - start_turn: clears this turn's fields and sets the deadline.
    - check_emergency: checks the life-threatening phrases and asks the checker. With no phrase
      match, the checker runs on another thread while the turn goes on.
    - await_emergency: collects that answer and combines it with combine_emergency, before
      anything is written, run or spoken. On an emergency, the model's work is dropped.
    - call_model, while a check is running, stops waiting for the model if the check raises
      the alarm first, so the 112 message is spoken at once.
    - check_confirmation: runs the yes rule. Anything but a yes deletes the pending action.
    - write_to_calendar: validates again, writes the booking, move or cancel, and builds the
      result sentence. A slot taken or an appointment no longer active at write time becomes the
      matching reason line.
    - call_model: calls the model. This is the only step that does. If the final response is
      empty, it uses text the model sent with an earlier tool call this turn.
    - run_tools: runs one round of tool calls.
    - ask_confirmation: builds the confirmation question from the stored action, with the
      patient's name. A move or cancel uses the name and service on the appointment. A checked
      sentence from the model may go before it. The checker is asked about that sentence only
      when there is one.
    - check_model_reply: checks the model's reply, with the checker's claim and advice flags,
      and this turn's search results, so the handbook's own text is not called advice.
    - search_texts(turn_messages): the document text search returned this turn.
    - use_fixed_line: picks a fixed line when something stopped the turn.
    - end_turn: adds the turn to messages. Every path ends here.

How to understand it:
    Read these with the flowchart under The Turn Graph in the design. The steps hold little
    logic themselves. Rules live in guards/, words in sentences.py, and actions in tools/. The
    key safety fact is that only write_to_calendar writes to the calendar, and no tool leads to
    it. The only way there is the check_confirmation step, which runs on the caller's next turn.

    Only call_model calls the main model. A check step may call deps.checker, which never
    raises and never writes or speaks.

    The emergency check and the main model run at the same time on most turns, to save about
    0.25 s. There is no race, because the model's call has no side effects: await_emergency
    sits after every model call and before write_to_calendar, so nothing the model asked for is
    spoken, run or written until the check has answered. If the check raises the alarm while
    the model is still thinking, call_model stops waiting for it, so the 112 message is not
    held up by a reply that would be dropped. A real call showed that wait: 2.06 s. A phrase match keeps the old order,
    because a phrase may end the call: the check answers first and the model never starts. Each check step writes its own trace events: a guard
    event, and a checker event with the layer, probabilities, model and attempts, but no text.

Uses:
    state.py, deps.py, guards/, sentences.py, tools/dispatcher.py, prompts/build.py

Used by:
    graph/build.py
"""

import json
import logging
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import timedelta
from typing import Any

from langchain_core.messages import ToolMessage

from dental_agent.booking.calendar import NotActive, SlotTaken
from dental_agent.checker import caller_questions, reply_questions
from dental_agent.deps import GraphDeps, normalise_stop
from dental_agent.guards.reply_check import check_lead_in, check_reply
from dental_agent.guards.urgent import combine_emergency, detect
from dental_agent.guards.validate import validate_action
from dental_agent.guards.yes_rule import is_yes
from dental_agent.models import (
    ActionRecord,
    EndReason,
    ErrorCode,
    FixedReason,
    PendingAction,
    ToolCall,
    ToolExit,
    UrgentLevel,
)
from dental_agent.prompts.build import build_messages, state_block
from dental_agent.sentences import confirmation_text, result_text
from dental_agent.state import CallState
from dental_agent.tools import tool_schemas
from dental_agent.tools.dispatcher import run_tool_round
from dental_agent.tools.read_tools import with_call_ids

logger = logging.getLogger(__name__)

# How many appointments write_to_calendar loads into state after a write.
MAX_FOUND = 5
# Runs the emergency check alongside the main model. A few threads, because one call runs at a
# time and each turn has at most one check in flight.
_CHECKS = ThreadPoolExecutor(max_workers=4, thread_name_prefix="emergency-check")
AWAIT_GRACE_S = 1.0  # Extra wait for the check beyond the turn's time, against a hang.


def start_turn(state: CallState, deps: GraphDeps) -> dict[str, Any]:
    """Clear this turn's fields and start its deadline."""
    deadline = deps.clock.now() + timedelta(seconds=deps.settings.turn_deadline_s)
    return {
        "tool_rounds": 0,
        "turn_deadline_at": deadline,
        "rejected_calls": {},
        "tool_exit": ToolExit.NONE,
        "turn_messages": [],
        "turn_facts": "",
        "reply_text": "",
        "asks_confirmation": False,
        "fixed_reason": None,
    }


def check_emergency(state: CallState, deps: GraphDeps) -> dict[str, Any]:
    """Check for an emergency before anything else, including a pending yes.

    With no phrase match, the checker is asked on another thread while the turn goes on, and
    await_emergency collects its answer before anything is spoken, run or written.
    """
    clinic = deps.clinic
    text = state.last_caller_text
    phrase_hit = detect(text, clinic.urgent_phrases, clinic.negation_tokens) is not None
    deps.pending_emergency.pop(state.conversation_id, None)  # Never one from an earlier turn.
    limit = _time_left(state, deps)
    if phrase_hit or not deps.checker.backends:
        # A phrase match may end the call, so the main model is not started until the checker
        # has answered. With no checker, the phrases decide at once.
        answer = deps.checker.ask(text, caller_questions(), "caller", limit)
        return _decide_emergency(deps, text, phrase_hit, answer)
    deps.pending_emergency[state.conversation_id] = _CHECKS.submit(
        deps.checker.ask, text, caller_questions(), "caller", limit
    )
    return {}


def await_emergency(state: CallState, deps: GraphDeps) -> dict[str, Any]:
    """Collect the emergency check that ran alongside the model, before acting on anything.

    It runs before write_to_calendar and after every model call, so no reply is spoken, no tool
    runs and nothing is written until the answer is in. On an emergency the model's work for
    this turn is dropped. With no check in flight, as on later tool rounds, it does nothing.
    """
    future = deps.pending_emergency.pop(state.conversation_id, None)
    if future is None:
        return {}
    try:
        # The chain keeps to the turn's time itself. The grace only guards against a hang.
        answer = future.result(timeout=_time_left(state, deps) + AWAIT_GRACE_S)
    except Exception as error:  # A lost answer means the phrases decide, and none matched.
        logger.warning("Emergency check failed: %s", error)
        answer = None
    updates = _decide_emergency(deps, state.last_caller_text, False, answer)
    if updates:
        # The model's reply and tool calls are never used.
        updates |= {"turn_messages": [], "reply_text": ""}
    return updates


def _raises(check: Future) -> bool:
    """Return True if a finished emergency check raised the alarm. No phrase matched."""
    answer = check.result()
    layer = answer.layer if answer else None
    return combine_emergency(False, layer, answer.answers["emergency"] if answer else None)


def _decide_emergency(deps: GraphDeps, text: str, phrase_hit: bool, answer: Any) -> dict[str, Any]:
    """Combine the phrase match and the checker's answer, and trace both."""
    layer = answer.layer if answer else None
    raised = combine_emergency(phrase_hit, layer, answer.answers["emergency"] if answer else None)
    deps.trace.event(
        "guard",
        name="check_emergency",
        input=text,
        decision="reject" if raised else "pass",
        phrase_hit=phrase_hit,
    )
    _trace_checker(deps, "check_emergency", answer)
    if not raised:
        return {}
    # The pending action is dropped, so nothing can commit during an emergency.
    return {
        "urgent_level": UrgentLevel.LIFE_THREATENING,
        "pending_action": None,
        "fixed_reason": FixedReason.URGENT,
    }


def check_confirmation(state: CallState, deps: GraphDeps) -> dict[str, Any]:
    """Let a pending action through only on a clear yes to a question the caller heard."""
    confirmed = (
        state.pending_action is not None and state.awaiting_yes and is_yes(state.last_caller_text)
    )
    decision = "pass" if confirmed else "reject"
    deps.trace.event(
        "guard", name="check_confirmation", input=state.last_caller_text, decision=decision
    )
    # Clearing awaiting_yes means one confirmation question can be answered once only.
    # A non-yes deletes the action, so an old action cannot be confirmed later.
    pending = state.pending_action if confirmed else None
    return {"pending_action": pending, "awaiting_yes": False}


def write_to_calendar(state: CallState, deps: GraphDeps) -> dict[str, Any]:
    """Write the confirmed action. This is the only node that writes to the calendar."""
    action = state.pending_action
    try:
        # A repeated commit finds its own record, and must not see its own slot as taken.
        record = deps.calendar.get_action(action.action_id)
        if record is None:
            # Rules run again, because a slot can be taken between prepare and commit.
            error = validate_action(
                action, state.to_context(), deps.calendar, deps.clock.now(), deps.clinic
            )
            if error:
                return _refuse(state, deps, action, error)
            record = _write(state, deps, action)
    except SlotTaken:
        # Another call won the slot in the moment between the check and the write.
        return _refuse(state, deps, action, ErrorCode.SLOT_TAKEN)
    except NotActive:
        # The appointment was cancelled elsewhere in the moment between the check and the write.
        return _refuse(state, deps, action, ErrorCode.NOT_OWNER)
    except Exception as error:  # The write may or may not have happened, so check the record.
        logger.warning("Commit failed: %s", error)
        record = _find_record(deps, action.action_id)
        if record is None:
            deps.trace.event("error", where="write_to_calendar", detail=str(error))
            return {"pending_action": None, "fixed_reason": FixedReason.COMMIT_FAILED}

    deps.trace.event(
        "calendar_write", action_id=record.action_id, action_kind=record.kind, spoken=record.spoken
    )
    return {
        "pending_action": None,
        "offers": [],
        "reply_text": record.result_text,
        "fixed_reason": None,
        **_refreshed_appointments(state, deps),
    }


def call_model(state: CallState, deps: GraphDeps) -> dict[str, Any]:
    """Ask the model what to do next. This is the only node that calls the model."""
    if _past_deadline(state, deps):
        return {"fixed_reason": FixedReason.DEADLINE}
    # The facts are written once per turn, on the first round. Actions come from the calendar,
    # so the facts cannot disagree with it.
    facts = state.turn_facts
    if not facts:
        actions = deps.calendar.actions_for_call(state.conversation_id)
        facts = state_block(state, actions, deps.clock, deps.clinic)
    messages = build_messages(state, facts)
    started = time.monotonic()
    model = deps.model.bind_tools(tool_schemas())
    check = deps.pending_emergency.get(state.conversation_id)
    try:
        if check is None:
            response = model.invoke(messages)
        else:
            # The emergency check is still running. If it raises an alarm first, the model's
            # reply is not waited for: it would be dropped, and the 112 message must not wait.
            reply = _CHECKS.submit(model.invoke, messages)
            wait([reply, check], return_when=FIRST_COMPLETED)
            if not reply.done() and _raises(check):
                deps.trace.event("model_dropped", reason="emergency")
                # await_emergency turns this into the emergency line.
                return {"turn_facts": facts, "fixed_reason": FixedReason.MODEL_FAILURE}
            response = reply.result()
    except Exception as error:  # Any provider failure must end in a fixed line.
        logger.warning("Model call failed: %s", error)
        deps.trace.event("error", where="call_model", detail=str(error))
        return {"fixed_reason": FixedReason.MODEL_FAILURE}

    stop = normalise_stop(response)
    deps.trace.event(
        "model_call",
        tokens=response.usage_metadata,
        delay_s=round(time.monotonic() - started, 3),
        stop=stop,
    )
    # A cut-off response is never trusted, so none of its tool calls run.
    if stop == "cut_off":
        return {"fixed_reason": FixedReason.MODEL_FAILURE}
    updates: dict[str, Any] = {
        "turn_messages": [*state.turn_messages, response],
        "turn_facts": facts,
    }
    if response.tool_calls:
        # Only final text is spoken, so text sent alongside a tool call is ignored.
        return updates
    text = response.text.strip()
    if not text:
        # A model can answer next to a tool call, then send nothing once it has the result.
        # That earlier answer is used, and still goes through check_model_reply.
        text = _text_sent_with_tools(state)
        if text:
            deps.trace.event("reply_from_tool_response", text=text)
    if not text:
        return {"fixed_reason": FixedReason.MODEL_FAILURE}
    return updates | {"reply_text": text}


def run_tools(state: CallState, deps: GraphDeps) -> dict[str, Any]:
    """Run one round of the model's tool calls."""
    if _past_deadline(state, deps):
        return {"fixed_reason": FixedReason.DEADLINE}
    calls = [
        ToolCall(id=item["id"], name=item["name"], args=item["args"])
        for item in state.turn_messages[-1].tool_calls
    ]
    if deps.on_tool_round is not None:
        # This runs after await_emergency, so a filler line can never come before 112.
        try:
            deps.on_tool_round([call.name for call in calls])
        except Exception as error:  # A filler line must never cost the turn.
            logger.warning("Tool round hook failed: %s", error)
    round_result = run_tool_round(calls, state, deps)
    # Every tool call gets one matching answer, which the next model call requires.
    answers = [
        ToolMessage(
            content=json.dumps(result.model_dump(mode="json", exclude_none=True)),
            tool_call_id=call_id,
        )
        for call_id, result in round_result.results
    ]
    updates = {
        **round_result.updates,
        "turn_messages": [*state.turn_messages, *answers],
        "tool_rounds": state.tool_rounds + 1,
        "tool_exit": round_result.exit,
    }
    if round_result.exit is ToolExit.URGENT:
        updates["fixed_reason"] = FixedReason.URGENT
    elif round_result.exit is ToolExit.ROUND_LIMIT:
        updates["fixed_reason"] = FixedReason.ROUND_LIMIT
    elif round_result.exit is ToolExit.END_CALL:
        updates["end_reason"] = EndReason.DONE
    return updates


def ask_confirmation(state: CallState, deps: GraphDeps) -> dict[str, Any]:
    """Build the confirmation question from the stored action.

    The model may have sent one sentence with its prepare call, such as an answer to a side
    question. It is spoken first only if check_lead_in passes it. The code's question always
    comes last, so the caller's yes answers that question.
    """
    action = state.pending_action
    service_id = _action_service(action)
    old = action.appointment
    question = confirmation_text(
        action.kind,
        deps.clinic.service_name(service_id),
        _action_spoken(action),
        # A move or cancel says the name on the appointment, the patient on record.
        name=old.patient_name if old else state.details.name,
        in_addition=action.same_service_booked,
        old_spoken=old.spoken if action.kind == "move" else None,
    )

    # The response that holds the prepare call is the last one the model sent this turn.
    sent = [item for item in state.turn_messages if getattr(item, "tool_calls", None)]
    model_text = sent[-1].text.strip() if sent else ""
    lead_in = ""
    if model_text:
        actions = deps.calendar.actions_for_call(state.conversation_id)
        # The checker is asked only when there is a lead-in, so a plain prepare adds no wait.
        claim, advice = _ask_reply_checker(state, deps, model_text, "ask_confirmation")
        decision = check_lead_in(
            model_text,
            service_id,
            actions,
            state.found_appointments,
            deps.clinic,
            model_claim=claim,
            model_advice=advice,
            sources=_reply_sources(state, deps),
        )
        deps.trace.event(
            "guard",
            name="check_lead_in",
            input=model_text,
            decision="reject" if decision.replaced else "pass",
            reason=decision.reason,
        )
        lead_in = decision.text

    return {
        # The stored question has no lead-in, so a repeat after silence asks only the question.
        "pending_action": action.model_copy(update={"confirmation_text": question}),
        "reply_text": f"{lead_in} {question}" if lead_in else question,
        "asks_confirmation": True,
    }


def check_model_reply(state: CallState, deps: GraphDeps) -> dict[str, Any]:
    """Check the model's reply before the session speaks it."""
    actions = deps.calendar.actions_for_call(state.conversation_id)
    claim, advice = _ask_reply_checker(state, deps, state.reply_text, "check_model_reply")
    decision = check_reply(
        state.reply_text,
        actions,
        state.found_appointments,
        deps.clinic,
        deps.settings.max_reply_sentences,
        model_claim=claim,
        model_advice=advice,
        sources=_reply_sources(state, deps),
    )
    deps.trace.event(
        "guard",
        name="check_model_reply",
        input=state.reply_text,
        decision="reject" if decision.replaced else "pass",
        reason=decision.reason,
    )
    return {"reply_text": decision.text}


def use_fixed_line(state: CallState, deps: GraphDeps) -> dict[str, Any]:
    """Pick the fixed line for why this turn stopped."""
    reason = state.fixed_reason
    deps.trace.event("fixed_line", reason=reason)
    # write_to_calendar has already set the refusal line, so it is left alone.
    if reason is FixedReason.COMMIT_REFUSED:
        return {}
    # Each reason maps to its line, and to an ending if that line ends the call.
    lines = {
        FixedReason.URGENT: ("urgent_message", EndReason.URGENT),
        FixedReason.MODEL_FAILURE: ("trouble", None),
        FixedReason.DEADLINE: ("trouble", None),
        FixedReason.ROUND_LIMIT: ("cannot_finish", EndReason.ROUND_LIMIT),
        FixedReason.COMMIT_FAILED: ("commit_unknown", EndReason.COMMIT_FAILED),
    }
    key, end_reason = lines[reason]
    return {"reply_text": deps.clinic.line(key), "end_reason": end_reason}


def end_turn(state: CallState, deps: GraphDeps) -> dict[str, Any]:
    """Add the turn to the history. Every path through the graph ends here."""
    history = [*state.messages, {"role": "caller", "text": state.last_caller_text}]
    # A reply can be empty, as after end_call, and an empty message would fail at the provider.
    if state.reply_text:
        history.append({"role": "bot", "text": state.reply_text})
    return {"messages": history, "turn": state.turn + 1, "turn_messages": []}


def _text_sent_with_tools(state: CallState) -> str:
    """Return the latest text the model sent alongside a tool call this turn, or ""."""
    for message in reversed(state.turn_messages):
        if getattr(message, "tool_calls", None) and message.text.strip():
            return message.text.strip()
    return ""


def _ask_reply_checker(
    state: CallState, deps: GraphDeps, text: str, step: str
) -> tuple[bool, bool]:
    """Ask the checker about a model reply or lead-in. With no answer, both flags are false."""
    answer = deps.checker.ask(text, reply_questions(), "reply", _time_left(state, deps))
    _trace_checker(deps, step, answer)
    if answer is None:
        return False, False
    return answer.answers["claims_change"], answer.answers["medical_advice"]


def _time_left(state: CallState, deps: GraphDeps) -> float:
    """Return the seconds left before this turn's deadline, for capping a checker call."""
    if state.turn_deadline_at is None:  # Only when a step runs outside start_turn.
        return deps.settings.turn_deadline_s
    return (state.turn_deadline_at - deps.clock.now()).total_seconds()


def _trace_checker(deps: GraphDeps, step: str, answer: Any) -> None:
    """Record what the checker answered, with no text. A layer of fixed means none answered."""
    deps.trace.event(
        "checker",
        step=step,
        layer=answer.layer if answer else "fixed",
        probabilities=answer.probabilities if answer else {},
        model=answer.model if answer else None,
        second_opinions=answer.second_opinions if answer else {},
        attempts=deps.checker.last_attempts,
    )


def _past_deadline(state: CallState, deps: GraphDeps) -> bool:
    """Return True if this turn has run out of time."""
    return deps.clock.now() > state.turn_deadline_at


def _reply_sources(state: CallState, deps: GraphDeps) -> list[str]:
    """Return the handbook text the reply checks may accept: this turn's search results and the
    last search's pieces, which Current Facts repeats on the next turn, each with its heading.
    check_reply adds the clinic's fixed lines, such as "The dentist can advise you at your
    visit", but only when this returns something, so with no search in the call a fixed line
    can never excuse advice of the model's own."""
    last = [_with_heading(item) for item in state.last_search]
    return search_texts(state.turn_messages) + last


def search_texts(turn_messages: list[Any]) -> list[str]:
    """Return the text of every document piece search_clinic_info returned this turn.

    The reply checks accept the clinic's own handbook text even when the checker calls it
    advice, so they need exactly what search gave the model, and nothing else: each piece's
    heading and text.
    """
    texts = []
    for message in turn_messages:
        if not isinstance(message, ToolMessage):
            continue
        try:
            data = json.loads(message.content).get("data")
        except (TypeError, ValueError, AttributeError):
            continue
        if isinstance(data, list):
            texts += [
                _with_heading(item)
                for item in data
                if isinstance(item, dict) and "heading" in item and "text" in item
            ]
    return texts


def _with_heading(item: dict[str, str]) -> str:
    """Return a piece's text with its heading in front, as the model saw them together.

    The heading has no full stop, so it joins the first sentence: a reply may take "after a
    filling" from the heading "After a filling" and the rest from the sentence below it.
    """
    heading = item.get("heading", "").strip()
    return f"{heading} {item['text']}" if heading else item["text"]


def _write(state: CallState, deps: GraphDeps, action: PendingAction) -> ActionRecord:
    """Build the result sentence and write it with the change, in one transaction."""
    service_id = _action_service(action)
    spoken = _action_spoken(action)
    sentence = result_text(action.kind, deps.clinic.service_name(service_id), spoken)
    now = deps.clock.now()
    calendar = deps.calendar
    if action.kind == "move":
        return calendar.move(action, state.conversation_id, spoken, sentence, now)
    if action.kind == "cancel":
        return calendar.cancel(action, state.conversation_id, spoken, sentence, now)
    return calendar.book(
        action,
        state.conversation_id,
        state.caller_phone,
        state.details.name,
        service_id,
        spoken,
        sentence,
        now,
    )


def _action_service(action: PendingAction) -> str:
    """Return the service an action is about: the offer's for a booking, else the appointment's."""
    return action.offer.service if action.kind == "book" else action.appointment.service


def _action_spoken(action: PendingAction) -> str:
    """Return the time an action is about: the new slot, or the appointment for a cancel.

    It is the action's stored spoken text, which the claim rule looks for in later replies.
    """
    return action.appointment.spoken if action.kind == "cancel" else action.offer.spoken


def _refuse(
    state: CallState, deps: GraphDeps, action: PendingAction, error: ErrorCode
) -> dict[str, Any]:
    """End the turn with the reason line, keeping the caller's details for the next try."""
    deps.trace.event(
        "guard", name="write_to_calendar", input=action.action_id, decision="reject", code=error
    )
    # Drop the refused offer, so it cannot be chosen again. A cancel has no offer.
    gone = action.offer.db_slot_id if action.offer else None
    # A refused move or cancel may mean the appointment changed elsewhere. Reading the list
    # again stops Current Facts showing it as it was, and stops its old time proving a claim.
    refreshed = _refreshed_appointments(state, deps) if action.kind != "book" else {}
    return {
        "pending_action": None,
        "offers": [item for item in state.offers if item.db_slot_id != gone],
        "reply_text": deps.clinic.line(error.value),
        "fixed_reason": FixedReason.COMMIT_REFUSED,
        **refreshed,
    }


def _find_record(deps: GraphDeps, action_id: str) -> ActionRecord | None:
    """Look for the stored record after an error. A failed read counts as not found."""
    try:
        return deps.calendar.get_action(action_id)
    except Exception as error:  # The caller is told to check with the clinic.
        logger.warning("Could not read action %s: %s", action_id, error)
        return None


def _refreshed_appointments(state: CallState, deps: GraphDeps) -> dict[str, Any]:
    """Reload the caller's appointments still to come, so they can be discussed truthfully."""
    try:
        rows = deps.calendar.appointments_for(state.caller_phone, MAX_FOUND, after=deps.clock.now())
    except Exception as error:  # The write or refusal stands, so this must not undo the turn.
        logger.warning("Could not refresh appointments: %s", error)
        return {}
    found = with_call_ids(rows, state.next_appointment_number)
    return {
        "found_appointments": found,
        "next_appointment_number": state.next_appointment_number + len(found),
    }
