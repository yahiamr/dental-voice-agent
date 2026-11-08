"""The command-line entry point. Stage: 1 and 2.

What this file is:
    What runs when you type dental-agent.

What it contains:
    - seed [--reset]: creates the sample data.
    - index [--with-embeddings]: builds the search index.
    - chat: runs a call by keyboard.
    - voice [--save-fixtures <dir>]: runs a call by microphone.
    - --caller-phone and --new-caller, which simulate caller ID.
    - --checker on|off, which turns the checker on or off for one call, over CHECKER_ENABLED.

How to understand it:
    This file does little. It reads the arguments, makes sure the data exists, prints sample
    numbers, builds Deps and calls run_call. Start here to follow a call from the top. There is
    no real phone line, so --caller-phone stands in for caller ID. A text call hangs up with
    Ctrl-D and a voice call with Ctrl-C. --save-fixtures saves each voice turn's audio and
    transcript, and the microphone is released when the call ends.

Uses:
    factories.py, session.py, booking/seed.py

Used by:
    You
"""

import argparse
import random
import re
import sys
from collections.abc import Sequence
from datetime import timedelta

from dental_agent import factories
from dental_agent.booking.calendar import Calendar
from dental_agent.booking.seed import seed
from dental_agent.clinic.data import load_clinic
from dental_agent.clock import Clock
from dental_agent.config import Settings
from dental_agent.deps import Deps
from dental_agent.session import run_call

# A Spanish number: +34 and nine digits, with no spaces.
PHONE_PATTERN = re.compile(r"\+34\d{9}")

MIN_FUTURE_DAYS = 14  # Startup warns if fewer days of slots remain.
SAMPLE_COUNT = 8  # How many sample numbers are printed: six existing and two new.


def main(argv: Sequence[str] | None = None) -> None:
    """Read the command and run it."""
    args = _parser().parse_args(argv)
    args.run(args)


def _parser() -> argparse.ArgumentParser:
    """Describe the commands and flags."""
    parser = argparse.ArgumentParser(
        prog="dental-agent", description="A voice agent for a fictional dental clinic."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    seed_command = commands.add_parser("seed", help="Create the sample data.")
    seed_command.add_argument("--reset", action="store_true", help="Replace existing data.")
    seed_command.set_defaults(run=_seed)

    index_command = commands.add_parser("index", help="Build the search index.")
    index_command.add_argument("--with-embeddings", action="store_true")
    index_command.set_defaults(run=_index)

    chat_command = commands.add_parser("chat", help="Run a call by keyboard.")
    _add_caller_flags(chat_command)
    _add_checker_flag(chat_command)
    chat_command.set_defaults(run=lambda args: _call(args, "text"))

    voice_command = commands.add_parser("voice", help="Run a call by microphone.")
    _add_caller_flags(voice_command)
    _add_checker_flag(voice_command)
    voice_command.add_argument("--save-fixtures", metavar="DIR")
    voice_command.set_defaults(run=lambda args: _call(args, "voice"))
    return parser


def _add_caller_flags(command: argparse.ArgumentParser) -> None:
    """Add the flags that simulate caller ID. The two cannot be used together."""
    group = command.add_mutually_exclusive_group()
    group.add_argument("--caller-phone", type=_phone, help="Pretend to call from this number.")
    group.add_argument("--new-caller", action="store_true", help="Use a number with no record.")


def _add_checker_flag(command: argparse.ArgumentParser) -> None:
    """Add --checker, which overrides CHECKER_ENABLED for one call. Without it, .env decides."""
    command.add_argument(
        "--checker",
        choices=["on", "off"],
        help="Turn the checker on or off for this call. The default comes from CHECKER_ENABLED.",
    )


def _phone(text: str) -> str:
    """Check a phone number. argparse turns the error into a usage message."""
    if not PHONE_PATTERN.fullmatch(text):
        raise argparse.ArgumentTypeError("use +34 and nine digits, such as +34600000001")
    return text


def _seed(args: argparse.Namespace) -> None:
    """Create the sample data. This needs no API key."""
    settings = Settings()
    calendar = Calendar(settings.calendar_path)
    seed(calendar, Clock(), load_clinic(settings.clinic_path), reset=args.reset)
    print(f"Sample data written to {settings.calendar_path}.")


def _index(args: argparse.Namespace) -> None:
    """Build the search index. Stage 5 adds this."""
    print("Indexing is not available yet.")


def _call(args: argparse.Namespace, mode: str) -> None:
    """Run one call, creating the data first if this is the first run."""
    settings = Settings()
    if args.checker:
        settings = settings.model_copy(update={"checker_enabled": args.checker == "on"})
    deps = factories.build_deps(settings, mode, fixture_dir=getattr(args, "save_fixtures", None))
    print(_checker_line(settings, deps.graph.checker.names))
    _prepare_data(deps)
    calendar = deps.graph.calendar
    samples = calendar.sample_patients(SAMPLE_COUNT)
    print("Sample numbers to try with --caller-phone:")
    for phone, name in samples:
        print(f"  {phone}  {name}")
    phone = _caller(args, calendar, samples[0][0])
    # Voice has no keyboard input to close, so Ctrl-C hangs up instead.
    hang_up = "Ctrl-C" if mode == "voice" else "Ctrl-D"
    print(f"Calling from {phone}. Press {hang_up} to hang up.\n")
    try:
        record = run_call(deps, phone)
    finally:
        # Release the microphone. Text speech has nothing to release.
        close = getattr(deps.speech, "close", None)
        if close is not None:
            close()
    print(f"\nSaved to {deps.graph.settings.runs_dir}/conversations/{record.conversation_id}.json")


def _checker_line(settings: Settings, names: list[str]) -> str:
    """Say whether the checker is on, so a tester knows which mode this call runs in."""
    if not settings.checker_enabled:
        return "Checker: off. The guards use their word lists alone."
    if not settings.checker_backends:
        return (
            "Checker: on, but CHECKER_BACKENDS is empty, so the guards use their word lists alone."
        )
    if not names:
        return "Checker: on, but no backend has a key, so the guards use their word lists alone."
    return f"Checker: on ({', then '.join(names)})."


def _prepare_data(deps: Deps) -> None:
    """Seed the data on the first run, and warn if the slots are running out."""
    calendar = deps.graph.calendar
    if calendar.slot_count() == 0:
        seed(calendar, deps.graph.clock, deps.graph.clinic)
    warning = seed_warning(calendar, deps.graph.clock)
    if warning:
        print(warning, file=sys.stderr)


def seed_warning(calendar: Calendar, clock: Clock) -> str | None:
    """Return a warning if under 14 days of future slots remain."""
    last = calendar.last_slot_start()
    if last is None or last - clock.now() < timedelta(days=MIN_FUTURE_DAYS):
        return "Few future slots remain. Run `dental-agent seed --reset`."
    return None


def _caller(args: argparse.Namespace, calendar: Calendar, first_sample: str) -> str:
    """Choose the number this call appears to come from."""
    if args.caller_phone:
        return args.caller_phone
    if args.new_caller:
        # Keep drawing until the number has no record.
        while True:
            phone = f"+34600{random.randrange(10**6):06d}"
            if not calendar.has_patient(phone):
                return phone
    return first_sample
