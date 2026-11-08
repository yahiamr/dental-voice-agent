"""The outer loop of one call. Stage: 1 and 5.

What this file is:
    Runs a call from greeting to goodbye. It is the only code that touches speech, the
    conversation file and the trace file.

What it contains:
    - run_call(deps, caller_phone), with Start, Turn and Finish steps inside try/finally.
    - start_fields(calendar, conversation_id, caller_phone): what the first graph run is told
      about the caller. One number is one patient, so a number on record is that existing
      patient with their name, and any other number is a new patient.
    - Start: checks keys and data files, creates the conversation record, speaks the greeting.
      The trace's start line names the speech backend really in use.
    - Turn: transcribe, graph.invoke, speak, save. On a voice call, a filler line such as "One
      moment." plays on its own thread while a tool round and the next model call run: once a
      turn, only when a model call follows, never before a confirmation question or 112. It is
      traced as a filler, so delay figures time the real reply, and the turn waits for it
      before writing the trace. A filler that could not be spoken is left out of the
      transcript. Text chat has none.
    - Finish: reads actions from the calendar, speaks the closing lines, writes the outcome and
      summary.
    - Handling for silence, transcription errors, low confidence, graph errors and the Endings
      table.

How to understand it:
    Think of two layers. The session deals with the outside world, and the graph decides what to
    say. The session keeps only awaiting_yes, trouble_count and transcript, because the
    checkpointer owns CallState. trouble_count goes up on any failure, resets after a normal
    reply, and ends the call at 2. awaiting_yes becomes true only after a confirmation question
    was actually spoken, so nobody can confirm something they never heard.

Uses:
    deps.py, graph/build.py, speech/base.py, storage/, sentences.py

Used by:
    cli.py, evals/evaluate.py
"""

import logging
import random
import secrets
import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from dental_agent.deps import Deps
from dental_agent.graph.build import build_graph
from dental_agent.guards.validate import caller_standing, first_visit_id
from dental_agent.models import ActionRecord, Details, EndReason, FixedReason, Outcome
from dental_agent.sentences import closing_text, status_text
from dental_agent.state import CallState
from dental_agent.storage.conversation import ConversationRecord
from dental_agent.storage.summary import key_points, optional_note

logger = logging.getLogger(__name__)

# Two problems in a row end the call.
MAX_TROUBLE = 2

# Added to the greeting when the backend plays a tone.
TONE_NOTE = " Please speak after the tone."

# A round with one of these tools goes straight to a fixed reply, such as the confirmation
# question or the goodbye, with no model call after it, so a filler line would only delay it.
NO_FILLER_TOOLS = frozenset(
    {"prepare_booking", "prepare_reschedule", "prepare_cancel", "end_call", "flag_urgent"}
)

# Settings copied to the trace's start line, so a run can be explained later.
TRACED_SETTINGS = (
    "llm_provider",
    "llm_model",
    "speech_backend",
    "llm_timeout_s",
    "turn_deadline_s",
    "max_tool_rounds",
    "eos_wait_ms",
    "eos_wait_yes_ms",
    "silence_timeout_s",
    "min_speech_ms",
    "yes_confidence_min",
    "mic_reopen_ms",
    "speech_timeout_s",
    "checker_enabled",
    "checker_jev_model",
    "checker_anthropic_model",
    "checker_jev_timeout_s",
    "checker_anthropic_timeout_s",
    "checker_cutoff",
)


@dataclass(frozen=True)
class Ending:
    """One row of the Endings table: what Finish says, and how the outcome is chosen."""

    lines: tuple[str, ...]  # Fixed lines spoken before the closing sentence.
    closing: bool  # Whether the closing sentence is spoken.
    no_action_outcome: Outcome  # The outcome when the call changed nothing.


# With an action, these rows store completed. Urgent and calendar_failed are handled apart.
ENDINGS = {
    EndReason.DONE: Ending((), True, Outcome.NO_ACTION),
    EndReason.CALLER_LEFT: Ending((), False, Outcome.ABANDONED),
    EndReason.URGENT: Ending((), False, Outcome.URGENT),
    EndReason.SILENCE: Ending(("emergency",), True, Outcome.ABANDONED),
    EndReason.FAILURES: Ending(("call_clinic", "emergency"), True, Outcome.FAILED),
    EndReason.TURN_LIMIT: Ending(("call_clinic", "emergency"), True, Outcome.FAILED),
    EndReason.ROUND_LIMIT: Ending(("call_clinic", "emergency"), True, Outcome.FAILED),
    EndReason.COMMIT_FAILED: Ending(("emergency",), True, Outcome.FAILED),
    EndReason.CALENDAR_FAILED: Ending(("call_clinic", "emergency"), False, Outcome.FAILED),
    EndReason.CRASH: Ending(("call_clinic", "emergency"), False, Outcome.FAILED),
}


def start_fields(
    calendar: Any, clinic: Any, now: datetime, conversation_id: str, caller_phone: str
) -> dict[str, Any]:
    """Return what the first graph run is given about the caller.

    One number is one patient. A first visit whose time has passed marks the patient existing
    first. Then the record gives the name and the status: new until the first visit is
    attended, existing after. A number with no record is a new patient, name not yet asked.
    """
    calendar.mark_attended(caller_phone, now, first_visit_id(clinic))
    status, booked = caller_standing(calendar, caller_phone, clinic)
    name = calendar.patient_name(caller_phone)
    return {
        "conversation_id": conversation_id,
        "caller_phone": caller_phone,
        "patient_record": name is not None,
        "first_visit_booked": booked,
        "details": Details(name=name, patient_type=status or "new"),
    }


def run_call(deps: Deps, caller_phone: str) -> ConversationRecord:
    """Run one call from greeting to goodbye, and return what was stored."""
    return _Call(deps, caller_phone).run()


class _Call:
    """One running call. It holds only what the checkpointer does not."""

    def __init__(self, deps: Deps, caller_phone: str) -> None:
        self.deps = deps
        self.services = deps.graph
        self.speech = deps.speech
        self.clinic = deps.graph.clinic
        self.settings = deps.graph.settings
        self.caller_phone = caller_phone

        now = self.services.clock.now()
        # The time gives order, and the random part keeps two calls in one second apart.
        self.conversation_id = f"c-{now:%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"
        self.record = ConversationRecord(
            conversation_id=self.conversation_id,
            started_at=now.isoformat(),
            caller_phone=caller_phone,
        )
        self.services.trace.bind(self.conversation_id, self.settings.runs_dir)
        self.graph = build_graph(self.services)
        # thread_id tells the checkpointer which call's state to load and save.
        self.config = {"configurable": {"thread_id": self.conversation_id}}

        self.awaiting_yes = False  # True only after a confirmation question was actually spoken.
        self.trouble_count = 0  # Problems in a row. Reaching 2 ends the call.
        self.trouble_line_said = False  # The failures ending starts with this line.
        self.turn = 0
        self.end_reason: EndReason | None = None
        self.last_actions: list[ActionRecord] = []  # What the calendar last showed.
        self.first_run_fields: dict[str, Any] = {}  # Sent with the first graph run only.
        self.filler: threading.Thread | None = None  # This turn's filler line, while it plays.
        self.filler_said = False  # At most one filler line a turn.
        # Only a voice call fills silence: in text chat the reply simply appears.
        if self.speech.uses_tone:
            self.services.on_tool_round = self._fill_silence

    def run(self) -> ConversationRecord:
        """Start, take turns until the call ends, then finish on every handled exit."""
        try:
            self._start()
            while self.end_reason is None:
                self._take_turn()
        except KeyboardInterrupt:
            self.end_reason = EndReason.CALLER_LEFT
        except Exception as error:  # Anything unexpected still gets a stored record.
            logger.exception("The call crashed")
            self.services.trace.event("error", where="session", detail=str(error))
            self.end_reason = EndReason.CRASH
        finally:
            self._finish()
        return self.record

    def _start(self) -> None:
        """Record the run's settings, and greet the caller."""
        fields = {name: getattr(self.settings, name) for name in TRACED_SETTINGS}
        # The backend really in use. Text chat ignores the voice setting, so the setting alone
        # would be wrong in a text call.
        fields["speech_backend"] = getattr(self.speech, "backend_name", fields["speech_backend"])
        # A new call forgets which checker backends failed in the last one. The start line lists
        # the backends the chain actually holds, which can differ from the setting.
        self.services.checker.start_call()
        fields["checker_backends"] = self.services.checker.names
        clock_time = self.services.clock.now().isoformat()
        self.services.trace.write_start(
            fields | {"git_commit": _git_commit(), "clock_time": clock_time}
        )
        self.first_run_fields = start_fields(
            self.services.calendar,
            self.clinic,
            self.services.clock.now(),
            self.conversation_id,
            self.caller_phone,
        )
        greeting = self.clinic.line("greeting")
        self._say(greeting + TONE_NOTE if self.speech.uses_tone else greeting)
        self.services.trace.flush(turn=0, tools_used=False)
        self.record.save(self.settings.runs_dir)

    def _take_turn(self) -> None:
        """Listen once, act on what was heard, then save."""
        # The limit is checked first, so the caller is not asked to speak for nothing.
        if self.turn >= self.settings.max_turns:
            self.end_reason = EndReason.TURN_LIMIT
            self._say(self.clinic.line("cannot_finish"))
            return

        heard = self.speech.transcribe(expect_yes=self.awaiting_yes)
        self.services.trace.event(
            "speech",
            result=heard.kind,
            confidence=heard.confidence,
            timings=heard.timings,
            provider=heard.provider,
        )
        if heard.kind == "closed":
            self.end_reason = EndReason.CALLER_LEFT
        elif heard.kind in ("no_speech", "error"):
            self._nothing_usable(heard.kind)
        else:
            self.record.transcript.append({"role": "caller", "text": heard.text})
            # An unsure reply to a confirmation question counts as not heard, so it cannot commit.
            if self.awaiting_yes and heard.confidence < self.settings.yes_confidence_min:
                self._nothing_usable("error")
            else:
                self._run_graph(heard.text)

        # A turn with no reply never reached _say, so its filler may still be playing.
        self._wait_for_filler()
        tools_used = any(item["kind"] == "tool_call" for item in self.services.trace.events)
        self.services.trace.flush(turn=self.turn, tools_used=tools_used)
        self.record.save(self.settings.runs_dir)

    def _nothing_usable(self, kind: str) -> None:
        """Handle silence or a bad transcript. These never reach the graph or count as turns."""
        self.trouble_count += 1
        if self.trouble_count >= MAX_TROUBLE:
            self.end_reason = EndReason.SILENCE
            return
        if self.awaiting_yes:
            # Repeat the confirmation question, so the next yes still answers it.
            self._repeat_confirmation()
        else:
            key = "still_there" if kind == "no_speech" else "not_caught"
            if not self._say(self.clinic.line(key)):
                self._add_failure()

    def _repeat_confirmation(self) -> None:
        """Say the waiting confirmation question again."""
        pending = self._saved_state().pending_action
        spoke = self._say(pending.confirmation_text, asks_confirmation=True)
        # If it could not be heard, it cannot be confirmed.
        self.awaiting_yes = spoke
        if not spoke:
            self._add_failure()

    def _run_graph(self, text: str) -> None:
        """Run one graph turn, then speak its reply."""
        self.filler_said = False
        turn_input = {"last_caller_text": text, "awaiting_yes": self.awaiting_yes}
        turn_input |= self.first_run_fields
        try:
            result = self.graph.invoke(turn_input, self.config)
        except Exception as error:  # A failed turn must not end the call by itself.
            logger.warning("A graph run failed: %s", error)
            self.services.trace.event("error", where="graph", detail=str(error))
            self.first_run_fields = {}
            self._graph_error()
            return
        self.first_run_fields = {}

        # graph.invoke returns a dict, so turn it back into the typed state.
        state = CallState.model_validate(result)
        self.turn = state.turn
        self.record.details = state.details
        self.record.urgent_level = state.urgent_level
        self._read_actions()

        spoke = self._say(state.reply_text, state.asks_confirmation) if state.reply_text else True
        self.awaiting_yes = state.asks_confirmation and spoke

        model_failed = state.fixed_reason in (FixedReason.MODEL_FAILURE, FixedReason.DEADLINE)
        if model_failed:
            # The reply was the trouble line, so the failures ending need not repeat it.
            self.trouble_line_said = True
        if model_failed or not spoke:
            self._add_failure()
        elif state.reply_text:
            self.trouble_count = 0
        # An ending chosen by the graph outranks one from the trouble count.
        if state.end_reason:
            self.end_reason = state.end_reason

    def _graph_error(self) -> None:
        """Recover from a failed graph run.

        The next run starts from start_turn. awaiting_yes becomes false, so a stale confirmation
        question cannot be confirmed.
        """
        self.awaiting_yes = False
        self._say(self.clinic.line("trouble"))
        self.trouble_line_said = True
        self._add_failure()
        if self.end_reason:
            return
        # Tell the caller where things stand, because the failed turn may have changed something.
        try:
            actions = self.services.calendar.actions_for_call(self.conversation_id)
        except Exception as error:  # With no way to check, the caller is sent to the clinic.
            logger.warning("Could not read actions: %s", error)
            self.end_reason = EndReason.CALENDAR_FAILED
            return
        self._say(status_text([item.result_text for item in actions]))

    def _add_failure(self) -> None:
        """Count a failure, and end the call if it is the second problem in a row."""
        self.trouble_count += 1
        if self.trouble_count >= MAX_TROUBLE:
            self.end_reason = EndReason.FAILURES

    def _read_actions(self) -> None:
        """Keep the latest actions, so a later failed read still has something to store."""
        try:
            self.last_actions = self.services.calendar.actions_for_call(self.conversation_id)
        except Exception as error:  # Finish handles an unreadable calendar.
            logger.warning("Could not read actions: %s", error)
        self.record.actions = [_stored_action(item) for item in self.last_actions]

    def _fill_silence(self, tool_names: list[str]) -> None:
        """Start a short filler line, once a turn, while a tool round and model call run.

        The graph calls this from run_tools. The line plays on its own thread, so the tools and
        the next model call carry on, and _say waits for it before the real reply. It is traced
        as a filler, not as speech, so the delay figures still time the real reply.
        """
        if self.filler_said or NO_FILLER_TOOLS.intersection(tool_names):
            return
        self.filler_said = True
        line = random.choice(self.clinic.filler_lines)

        # Recorded now, so the transcript keeps the order the caller heard.
        entry = {"role": "bot", "text": line, "printed_only": False}
        self.record.transcript.append(entry)

        def speak() -> None:
            try:
                result = self.speech.speak(line)
                ok, timings = result.ok, result.timings
            except Exception as error:  # A failed filler is only a missed courtesy.
                logger.warning("Filler line failed: %s", error)
                ok, timings = False, {}
            if not ok:
                # The caller never heard it. Nothing else writes the transcript until
                # _wait_for_filler has joined this thread, so removing it here is safe.
                self.record.transcript.remove(entry)
            self.services.trace.event("filler", text=line, ok=ok, timings=timings)

        self.filler = threading.Thread(target=speak, name="filler", daemon=True)
        self.filler.start()

    def _wait_for_filler(self) -> None:
        """Let a filler line finish, so two lines never play at once."""
        if self.filler is not None:
            self.filler.join()
            self.filler = None

    def _say(self, text: str, asks_confirmation: bool = False) -> bool:
        """Speak one line and record it. A line that cannot be spoken is printed."""
        self._wait_for_filler()
        try:
            result = self.speech.speak(text, asks_confirmation=asks_confirmation)
            spoke, timings = result.ok, result.timings
        except Exception as error:  # A speech failure must not end the call by itself.
            logger.warning("Speech failed: %s", error)
            spoke, timings = False, {}
        if not spoke:
            print(text)
        self.record.transcript.append({"role": "bot", "text": text, "printed_only": not spoke})
        self.services.trace.event("speech", result="spoken", ok=spoke, timings=timings)
        return spoke

    def _print_line(self, text: str) -> None:
        """Print a line with no speech at all, because the session itself has failed."""
        print(text)
        self.record.transcript.append({"role": "bot", "text": text, "printed_only": True})

    def _finish(self) -> None:
        """Say the closing lines, then write the outcome and summary."""
        self._wait_for_filler()
        self.services.on_tool_round = None  # The hook belongs to this call only.
        reason = self.end_reason or EndReason.CRASH
        ending = ENDINGS[reason]

        # Step 1: read what this call really did. The closing sentence depends on it.
        try:
            actions = self.services.calendar.actions_for_call(self.conversation_id)
            calendar_read = True
        except Exception as error:  # Stored actions remain the ones last read.
            logger.warning("Could not read actions at Finish: %s", error)
            actions, calendar_read = self.last_actions, False

        # Step 2: speak the remaining lines, then the closing sentence.
        if reason is EndReason.FAILURES and not self.trouble_line_said:
            self._say(self.clinic.line("trouble"))
        for key in ending.lines:
            if reason is EndReason.CRASH:
                self._print_line(self.clinic.line(key))
            else:
                self._say(self.clinic.line(key))
        results = [item.result_text for item in actions]
        if ending.closing and (reason is not EndReason.COMMIT_FAILED or results):
            if calendar_read:
                self._say(closing_text(results))
            else:
                self._say(closing_text(results, read_failed_line=self.clinic.clinic_line))

        # Step 3: write the outcome, details and code summary.
        self.record.actions = [_stored_action(item) for item in actions]
        self.record.end_reason = reason
        self.record.outcome = _outcome(reason, ending, actions, calendar_read)
        self.record.ended_at = self.services.clock.now().isoformat()
        state = self._saved_state()
        self.record.details = state.details
        self.record.urgent_level = state.urgent_level
        self.record.summary.key_points = key_points(state, actions)

        # Step 4: try the optional model sentence, which is allowed to fail.
        self.record.summary.note = optional_note(
            self.services.model, self.record.summary.key_points, self.settings.summary_deadline_s
        )

        # Step 5: save everything, and close the trace.
        self.record.save(self.settings.runs_dir)
        self.services.trace.flush(turn=-1, tools_used=False)
        self.services.trace.close()

    def _saved_state(self) -> CallState:
        """Return the checkpointer's state. It stays usable if a graph never ran."""
        try:
            return CallState.model_validate(self.graph.get_state(self.config).values)
        except Exception as error:  # A summary must still be written.
            logger.warning("Could not read saved state: %s", error)
            return CallState(caller_phone=self.caller_phone)


def _stored_action(action: ActionRecord) -> dict[str, Any]:
    """Return the fields of an action that the conversation file keeps."""
    return action.model_dump(include={"action_id", "kind", "spoken", "result_text"})


def _outcome(
    reason: EndReason, ending: Ending, actions: list[ActionRecord], calendar_read: bool
) -> Outcome:
    """Choose the stored outcome, following the Endings table."""
    if not calendar_read or reason is EndReason.CALENDAR_FAILED:
        return Outcome.FAILED
    if reason is EndReason.URGENT:
        return Outcome.URGENT
    return Outcome.COMPLETED if actions else ending.no_action_outcome


def _git_commit() -> str:
    """Return the current commit, which also identifies the prompt, or "none"."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "none"
    return result.stdout.strip()
