"""The evaluation runner. Stage: 4.

What this file is:
    Runs the scenarios in evals/scenarios.yaml against the real model and writes the results.
    Run it with uv run python -m evals.evaluate.

What it contains:
    - Scenario, Setup and Expect: one scenario as written in the file, checked when it loads.
    - load_scenarios(path).
    - place_appointment and take_named_slot: the two setup steps.
    - run_once(scenario, deps): one call through the real session, then check_run.
    - check_run(expect, record, tools, ran_out): every expectation, as a list of failures.
    - build_run_deps(settings, run_dir): a fresh calendar seeded at the fixed clock, per run.
    - results_summary, compare_with_baseline and results_markdown: what is saved and printed.
    - main(): runs each scenario 3 times and saves the results. A full run (every scenario, 3
      runs) saves evals/results/scenarios.md and scenarios.json, which are in Git, so each tag
      keeps the result it scored. A partial run saves to runs/results.md and .json instead, so it
      never replaces the baseline. Every run's files are kept under runs/scenarios/<time>/.
    - Options: --scenario NAME (repeat), --runs N, --workers N, --provider and --model to run
      on another provider, --out to save a full run elsewhere (such as
      evals/results/scenarios-openrouter for the OpenRouter run; by default each model and
      checker setting saves to its own file), --baseline to compare with
      another saved .json, and --voice-traces DIR to add delay figures from voice calls.

How to understand it:
    A scenario passes when at least 2 of its 3 runs pass, because a real model can vary. Guard
    tests in tests/ have a stricter rule: they must pass every time.

    Each run builds its services with build_deps, the same factory a real call uses, then swaps
    in three things: ScenarioSpeech for the microphone and speaker, a fresh calendar file in the
    run's own folder, and a FixedClock at Monday 9 November 2026, 10:00. The calendar is seeded
    at that clock, so "Tuesday the 17th" always means the same free slots, and the real
    data/calendar.sqlite is never touched.

    The checks read the conversation file the session saved, not the record in memory, so they
    also prove the stored JSON. A run fails if the caller's lines run out before the call ends,
    because then the conversation did not reach the end the scenario was written for.

    Before saving, a run is compared with the baseline, by default the .json it is about to
    replace, and any scenario that changed between pass and fail is listed. So a regression
    shows up at once, and git log on evals/results/ shows how the scores moved over time.

    Calls that change nothing are cheap but still use the real model and the checker, so a full
    run costs real API calls: about 39 calls of a few model turns each.

Uses:
    factories.py, session.py, booking/seed.py, evals/driver.py, evals/metrics.py

Used by:
    You, tests/test_scenarios.py
"""

import argparse
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml
from pydantic import BaseModel, ConfigDict, Field

from dental_agent.booking.calendar import Calendar
from dental_agent.booking.seed import seed
from dental_agent.clinic.data import Clinic
from dental_agent.clock import MADRID, FixedClock
from dental_agent.config import Settings
from dental_agent.deps import Deps
from dental_agent.factories import build_deps
from dental_agent.formatting import spoken_slot
from dental_agent.models import Offer, PendingAction
from dental_agent.session import run_call
from evals.driver import ScenarioSpeech
from evals.metrics import guard_rejections, task_completion, turn_delays

SCENARIOS_PATH = Path(__file__).with_name("scenarios.yaml")
RESULTS_DIR = Path(__file__).with_name("results")  # In Git, so each tag keeps its result.
BASELINE = RESULTS_DIR / "scenarios"  # A full run saves scenarios.md and scenarios.json here.
PARTIAL = Path("runs/results")  # A partial run saves here, which Git ignores.
SCENARIO_RUNS_DIR = Path("runs/scenarios")

RUNS = 3  # Runs per scenario. A scenario needs a majority of them: 2 of 3.
EVAL_NOW = datetime(2026, 11, 9, 10, 0, tzinfo=MADRID)  # The same moment the tests use.

NEW_CALLER = "+34600999999"  # A number with no record.
SAMPLE_CALLER = "+34600000001"  # Lucía Moreno, with one check-up at least 3 days ahead.
OTHER_PATIENT = "+34600888888"  # Who takes a slot in the slot-taken scenario.
SLOT_SEARCH_DAYS = 60  # How far ahead take_named_slot looks for the named slot.


class Expect(BaseModel):
    """What must be true when a run ends. Unset keys are not checked."""

    model_config = ConfigDict(extra="forbid")

    actions: list[str] | None = None  # Kinds written to the calendar in this call, in order.
    outcome: str | None = None  # The stored outcome.
    said: list[str] = Field(default_factory=list)  # Text that must appear in the bot's lines.
    not_said: list[str] = Field(default_factory=list)  # Text that must never appear there.
    tools: list[str] = Field(default_factory=list)  # Tools that must appear in the trace.
    record: dict[str, Any] = Field(default_factory=dict)  # Dotted path to an exact value.
    record_contains: dict[str, str] = Field(default_factory=dict)  # Dotted path to some text.


class Setup(BaseModel):
    """Changes made before a run starts, or during it."""

    model_config = ConfigDict(extra="forbid")

    # Moves the caller's one appointment to this many hours after the clock, such as 6 for a
    # cancel inside the notice period. Moving it, rather than adding one, keeps one appointment,
    # so the bot never has to ask which one.
    appointment_hours_ahead: float | None = None
    # Gives the slot named in the first confirmation question to another patient, just before
    # the caller's yes, so the commit finds it taken.
    take_slot_before_yes: bool = False


class Scenario(BaseModel):
    """One test conversation, as written in evals/scenarios.yaml."""

    model_config = ConfigDict(extra="forbid")

    name: str
    stage: int
    caller: str = "new"  # new, sample, or a phone number.
    setup: Setup = Field(default_factory=Setup)
    lines: list[str]
    auto_yes: bool = True
    expect: Expect

    @property
    def phone(self) -> str:
        """Return the number this scenario calls from."""
        return {"new": NEW_CALLER, "sample": SAMPLE_CALLER}.get(self.caller, self.caller)


@dataclass
class RunResult:
    """One run of one scenario."""

    scenario: str
    run: int
    passed: bool
    failures: list[str]
    conversation_path: Path
    trace_path: Path


def pass_needed(runs: int) -> int:
    """Return the passing runs a scenario needs: a majority, so 2 of 3 and 1 of 1."""
    return runs // 2 + 1


def load_scenarios(path: Path = SCENARIOS_PATH) -> list[Scenario]:
    """Read and check every scenario in the file."""
    data = yaml.safe_load(path.read_text())
    return [Scenario.model_validate(item) for item in data["scenarios"]]


def check_run(expect: Expect, record: dict[str, Any], tools: set[str], ran_out: bool) -> list[str]:
    """Return every expectation the stored record breaks. An empty list means the run passed."""
    failures = []
    if ran_out:
        failures.append("the caller's lines ran out")
    kinds = [item["kind"] for item in record.get("actions", [])]
    if expect.actions is not None and kinds != expect.actions:
        failures.append(f"actions were {kinds}, expected {expect.actions}")
    if expect.outcome is not None and record.get("outcome") != expect.outcome:
        failures.append(f"outcome was {record.get('outcome')}, expected {expect.outcome}")
    # Only the bot's lines count, so a caller saying "mg" cannot fail a not_said check.
    bot = " ".join(
        item["text"] for item in record.get("transcript", []) if item["role"] == "bot"
    ).lower()
    failures += [f"the bot never said {text!r}" for text in expect.said if text.lower() not in bot]
    failures += [f"the bot said {text!r}" for text in expect.not_said if text.lower() in bot]
    failures += [f"tool {name} was never called" for name in expect.tools if name not in tools]
    for path, wanted in expect.record.items():
        value = _field(record, path)
        if value != wanted:
            failures.append(f"record {path} was {value!r}, expected {wanted!r}")
    for path, text in expect.record_contains.items():
        if text.lower() not in json.dumps(_field(record, path), ensure_ascii=False).lower():
            failures.append(f"record {path} does not contain {text!r}")
    return failures


def _field(record: dict[str, Any], path: str) -> Any:
    """Return the value at a dotted path such as details.name, or None if any part is missing."""
    value: Any = record
    for part in path.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    return value


def place_appointment(calendar: Calendar, phone: str, hours_ahead: float, now: datetime) -> None:
    """Move the caller's one active appointment to this many hours after now.

    Scenario setup only. It writes SQL directly, as the test helpers do, because the product
    never needs to place an appointment at an arbitrary time.
    """
    start = now + timedelta(hours=hours_ahead)
    with calendar._connect() as connection:
        # The seed may already hold a slot at that time, so it is reused.
        connection.execute(
            "INSERT OR IGNORE INTO slots (start_at, end_at) VALUES (?, ?)",
            (start.isoformat(), (start + timedelta(minutes=30)).isoformat()),
        )
        slot = connection.execute(
            "SELECT id FROM slots WHERE start_at = ?", (start.isoformat(),)
        ).fetchone()["id"]
        holder = connection.execute(
            "SELECT patients.phone FROM appointments JOIN patients "
            "ON patients.id = appointments.patient_id "
            "WHERE appointments.slot_id = ? AND appointments.status = 'active'",
            (slot,),
        ).fetchone()
        if holder is not None and holder["phone"] != phone:
            raise ValueError(f"Scenario setup: the slot at {start:%d %B %H:%M} is already taken.")
        connection.execute(
            "UPDATE appointments SET slot_id = ? WHERE status = 'active' AND patient_id = "
            "(SELECT id FROM patients WHERE phone = ?)",
            (slot, phone),
        )


def take_named_slot(calendar: Calendar, question: str, now: datetime, clinic: Clinic) -> bool:
    """Book the free slot whose spoken time is in the question, for another patient.

    Returns whether a slot was found and taken. It uses the calendar's own book method, so the
    slot is taken exactly as another call would take it.
    """
    last_day = now.date() + timedelta(days=SLOT_SEARCH_DAYS)
    for slot in calendar.free_slots(now.date(), last_day, None, None, 10_000, now):
        spoken = spoken_slot(slot.start_at)
        if spoken in question:
            offer = Offer(
                slot_id="x", db_slot_id=slot.db_slot_id, service="check_up", spoken=spoken
            )
            action = PendingAction(action_id=str(uuid4()), kind="book", offer=offer)
            calendar.book(
                action, "c-other", OTHER_PATIENT, "Luis Mora", "check_up", spoken, "Booked.", now
            )
            return True
    return False


def run_once(scenario: Scenario, deps: Deps, run: int = 0) -> RunResult:
    """Run one scenario through the real session, then check the saved files."""
    graph = deps.graph
    if scenario.setup.appointment_hours_ahead is not None:
        place_appointment(
            graph.calendar, scenario.phone, scenario.setup.appointment_hours_ahead,
            graph.clock.now(),
        )  # fmt: skip
    before_yes = None
    if scenario.setup.take_slot_before_yes:

        def before_yes(question: str) -> None:
            take_named_slot(graph.calendar, question, graph.clock.now(), graph.clinic)

    speech = ScenarioSpeech(scenario.lines, scenario.auto_yes, before_yes)
    stored = run_call(replace(deps, speech=speech), scenario.phone)

    runs_dir = graph.settings.runs_dir
    conversation_path = runs_dir / "conversations" / f"{stored.conversation_id}.json"
    trace_path = runs_dir / "traces" / f"{stored.conversation_id}.jsonl"
    record = json.loads(conversation_path.read_text())
    failures = check_run(scenario.expect, record, tools_in_trace(trace_path), speech.ran_out)
    return RunResult(scenario.name, run, not failures, failures, conversation_path, trace_path)


def tools_in_trace(path: Path) -> set[str]:
    """Return the names of every tool called in a trace file."""
    if not path.exists():
        return set()
    names = set()
    for line in path.read_text().splitlines():
        for event in json.loads(line).get("events", []):
            if event.get("kind") == "tool_call":
                names.add(event["name"])
    return names


def run_settings(settings: Settings, run_dir: Path) -> Settings:
    """Return the settings for one run: every file it writes stays in its own folder.

    The checkpointer is always in memory, because a SQLite one would be one file shared by the
    runs going at once, in the real data folder.
    """
    return settings.model_copy(
        update={
            "runs_dir": run_dir,
            "calendar_path": run_dir / "calendar.sqlite",
            "checkpointer": "memory",
        }
    )


def build_run_deps(settings: Settings, run_dir: Path) -> Deps:
    """Build a real call's services, with a fresh calendar and the fixed clock, for one run."""
    deps = build_deps(run_settings(settings, run_dir), "text")
    clock = FixedClock(EVAL_NOW)
    seed(deps.graph.calendar, clock, deps.graph.clinic, reset=True)
    return replace(deps, graph=replace(deps.graph, clock=clock))


def run_all(
    scenarios: list[Scenario], settings: Settings, runs: int, workers: int, out_dir: Path
) -> list[RunResult]:
    """Run every scenario runs times, a few at a time, each in its own folder."""
    jobs = [(scenario, run) for scenario in scenarios for run in range(runs)]

    def one(job: tuple[Scenario, int]) -> RunResult:
        scenario, run = job
        run_dir = out_dir / f"{scenario.name}-{run + 1}"
        try:
            return run_once(scenario, build_run_deps(settings, run_dir), run)
        except Exception as error:  # One broken run must not stop the others.
            return RunResult(
                scenario.name, run, False, [f"the run crashed: {error}"], run_dir, run_dir
            )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, jobs))


def results_summary(
    scenarios: list[Scenario], results: list[RunResult], settings: Settings, runs: int
) -> dict[str, Any]:
    """Return what a run of the scenarios found, in the form saved to scenarios.json."""
    completion = task_completion(
        [(item.scenario, item.passed) for item in results], pass_needed(runs)
    )
    stages = {item.name: item.stage for item in scenarios}
    rows = {
        name: {
            "stage": stages.get(name),
            **row,
            "failures": sorted(
                {text for item in results if item.scenario == name for text in item.failures}
            ),
        }
        for name, row in completion.items()
    }
    return {
        "model": f"{settings.llm_provider}:{settings.llm_model}",
        "checker": settings.checker_enabled,
        "commit": _git_commit(),
        "date": f"{datetime.now(MADRID):%Y-%m-%d %H:%M}",
        "clock": EVAL_NOW.isoformat(),
        "runs": runs,
        "pass_needed": pass_needed(runs),
        "passed_scenarios": sum(1 for row in rows.values() if row["passed_scenario"]),
        "total_scenarios": len(rows),
        "scenarios": rows,
        "guard_rejections": guard_rejections([item.trace_path for item in results]),
    }


def compare_with_baseline(current: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    """Return one line per scenario that changed between pass and fail, or was added or removed.

    A change in passing runs within the same result, such as 3 of 3 to 2 of 3, is not listed:
    the model varies, and only the pass rule decides.
    """
    now, before = current["scenarios"], baseline.get("scenarios", {})
    regressions, fixes = [], []
    for name, row in now.items():
        old = before.get(name)
        if old is None or old["passed_scenario"] == row["passed_scenario"]:
            continue
        line = (
            f"`{name}` passed {old['passed']} of {old['runs']} in the baseline, "
            f"and {row['passed']} of {row['runs']} now."
        )
        (fixes if row["passed_scenario"] else regressions).append(line)
    lines = [f"Regression: {line}" for line in regressions] + [f"Fixed: {line}" for line in fixes]
    lines += [
        f"New: `{name}`, {row['passed']} of {row['runs']}."
        for name, row in now.items()
        if name not in before
    ]
    lines += [f"Removed: `{name}`." for name in before if name not in now]
    return lines or ["No scenario changed between pass and fail."]


def default_out(settings: Settings) -> Path:
    """Return where a full run saves, so each model and checker setting keeps its own result.

    The product's model with the checker on is the main baseline. Any other model, or the
    checker off, gets its own file, so it never replaces the main one or is compared with it.
    """
    fields = Settings.model_fields
    product = (
        settings.llm_provider == fields["llm_provider"].default
        and settings.llm_model == fields["llm_model"].default
        and settings.checker_enabled
    )
    if product:
        return BASELINE
    model = re.sub(r"[^A-Za-z0-9.-]+", "_", settings.llm_model)
    suffix = "" if settings.checker_enabled else "-no-checker"
    return RESULTS_DIR / f"scenarios-{settings.llm_provider}-{model}{suffix}"


def is_full_run(scenario_filter: list[str], runs: int) -> bool:
    """Return True if this run covers every scenario at the full count, so it may be saved."""
    return not scenario_filter and runs == RUNS


def results_markdown(
    summary: dict[str, Any],
    comparison: list[str] | None = None,
    delays: dict[str, dict] | None = None,
) -> str:
    """Write the results page: one row per scenario, then the comparison, guards and delays."""
    lines = [
        "# Scenario results",
        "",
        (
            f"Model: `{summary['model']}`. Checker: {'on' if summary['checker'] else 'off'}. "
            f"Commit: `{summary['commit']}`. Run on {summary['date']}."
        ),
        "",
        (
            f"Each scenario runs {summary['runs']} times on a fresh calendar, with the clock "
            f"fixed at {EVAL_NOW:%A %d %B %Y, %H:%M}. A scenario passes on "
            f"{summary['pass_needed']} passing runs."
        ),
        "",
        f"**{summary['passed_scenarios']} of {summary['total_scenarios']} scenarios pass.**",
        "",
        "| Scenario | Stage | Passing runs | Result | Failures |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, row in summary["scenarios"].items():
        lines.append(
            f"| `{name}` | {row['stage']} | {row['passed']} of {row['runs']} | "
            f"{'pass' if row['passed_scenario'] else 'FAIL'} | "
            f"{_cell('; '.join(row['failures'])) or '-'} |"
        )
    if comparison is not None:
        lines += ["", "**Compared with the baseline:**", ""]
        lines += [f"- {line}" for line in comparison]
    lines += ["", "**Guard rejections** (every run, by guard):", ""]
    rejections = summary["guard_rejections"]
    lines += [f"- `{name}`: {count}" for name, count in rejections.items()] or ["- none"]
    if delays is not None:
        lines += ["", "**Delay per turn** (voice traces):", ""]
        for kind, figures in delays.items():
            lines.append(
                f"- {kind}: {figures['turns']} turns, median {figures['median_s']} s, "
                f"slowest {figures['slowest_s']} s"
            )
    return "\n".join(lines) + "\n"


def _cell(text: str) -> str:
    """Return text that is safe inside one markdown table cell."""
    return " ".join(text.split()).replace("|", "\\|")


def _git_commit() -> str:
    """Return the current commit, or "none"."""
    try:
        result = subprocess.run(
            # --dirty marks uncommitted changes, as when a result is measured before the commit
            # that saves it.
            ["git", "describe", "--always", "--dirty", "--exclude=*"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "none"
    return result.stdout.strip()


def main() -> None:
    """Run the scenarios, then save and print the results."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", action="append", default=[], help="Run only this one.")
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--workers", type=int, default=4, help="Runs at the same time.")
    parser.add_argument("--provider", choices=["anthropic", "openrouter"])
    parser.add_argument("--model", help="The model ID for --provider.")
    parser.add_argument("--voice-traces", type=Path, help="A folder of voice-call traces.")
    parser.add_argument(
        "--out",
        type=Path,
        help="Where a full run saves its .md and .json. Default: one file per model and checker.",
    )
    parser.add_argument(
        "--baseline", type=Path, help="A saved .json to compare with. Default: the --out one."
    )
    args = parser.parse_args()

    updates: dict[str, Any] = {}
    if args.provider:
        updates["llm_provider"] = args.provider
    if args.model:
        updates["llm_model"] = args.model
    settings = Settings().model_copy(update=updates)
    out = args.out or default_out(settings)

    scenarios = load_scenarios()
    if args.scenario:
        scenarios = [item for item in scenarios if item.name in args.scenario]
        missing = set(args.scenario) - {item.name for item in scenarios}
        if missing:
            raise SystemExit(f"No such scenario: {', '.join(sorted(missing))}")

    out_dir = SCENARIO_RUNS_DIR / f"{datetime.now(MADRID):%Y%m%d-%H%M%S}"
    results = run_all(scenarios, settings, args.runs, args.workers, out_dir)
    summary = results_summary(scenarios, results, settings, args.runs)

    # Compare before saving, so the default baseline is the result saved last time.
    baseline_path = args.baseline or out.with_suffix(".json")
    comparison = None
    if baseline_path.is_file():
        comparison = compare_with_baseline(summary, json.loads(baseline_path.read_text()))
    delays = None
    if args.voice_traces is not None:
        delays = turn_delays(sorted(args.voice_traces.glob("*.jsonl")))
    page = results_markdown(summary, comparison, delays)

    # Only a full run replaces the saved baseline. A partial run goes to runs/, which Git ignores.
    stem = out if is_full_run(args.scenario, args.runs) else PARTIAL
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".md").write_text(page)
    stem.with_suffix(".json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    print(page)
    print(f"Saved to {stem}.md and {stem}.json. Each run's files are in {out_dir}.")
    if stem == PARTIAL:
        print("This was not a full run, so the saved baseline was left as it was.")


if __name__ == "__main__":
    main()
