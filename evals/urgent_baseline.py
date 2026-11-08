"""Counts how often the main model flags urgency on its own. Stage: checker design, Step 1.

What this file is:
    A measurement for the checker design. It runs every emergency case, true and false, as one
    graph turn with the live main model, with the emergency phrases switched off and no checker,
    and counts the main model's flag_urgent calls. Run it with
    uv run python -m evals.urgent_baseline. evals/system_gate.py reads its result to gate the
    whole system, checker and main model together.

What it contains:
    - build_baseline_deps(settings, workdir): the graph's services, built by hand so no checker
      setting can switch a checker on, with an empty phrase list.
    - run_case(graph, deps, text): one graph turn, returning the urgency levels flagged.
    - main(): runs the cases, prints one line per case and the three numbers, and saves them.

How to understand it:
    The phrase check runs before the main model, so a phrase match would hide what the model
    does. Emptying urgent_phrases makes detect pass every case, and only detect reads that list.
    Each run gets a fresh thread_id, the first-turn fields of a new caller from start_fields,
    and a new TraceSink, so no state or events carry over. The flags are read from the run's
    tool_call events, from arguments["level"].

    Three numbers come out:
    - Life-threatening: the share of runs on life-threatening cases that flagged
      life_threatening.
    - False alarms: the cases that are not emergencies where any run flagged life_threatening.
    - The dental ratio: the share of runs on dental cases that flagged urgency at either level.
      Below 80%, the checker design brings dental urgency back as its own change.

    Results go to runs/urgent_baseline/, which Git ignores. The calendar is a temporary copy,
    so data/calendar.sqlite is never touched.

Uses:
    factories.py, graph/build.py, session.py (start_fields), booking/seed.py, deps.py,
    evals/intent_eval.py (load_cases)

Used by:
    You, evals/system_gate.py
"""

import argparse
import json
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

from dental_agent.booking.calendar import Calendar
from dental_agent.booking.seed import seed
from dental_agent.checker import CheckerChain
from dental_agent.clinic.data import load_clinic
from dental_agent.clock import Clock
from dental_agent.config import Settings
from dental_agent.deps import GraphDeps
from dental_agent.factories import create_checkpointer, create_llm, create_retriever
from dental_agent.graph.build import build_graph
from dental_agent.session import start_fields
from dental_agent.storage.trace import TraceSink
from evals.intent_eval import CASES_PATH, ROOT, load_cases

OUT_DIR = ROOT / "runs" / "urgent_baseline"
NEW_CALLER = "+34600999999"  # A number with no record, so every run is a new caller.
DENTAL_TRIGGER = 0.80  # Below this dental ratio, dental urgency comes back as its own design.


def build_baseline_deps(settings: Settings, workdir: Path) -> GraphDeps:
    """Build the graph's services by hand, with no emergency phrases and no checker."""
    clinic = load_clinic(settings.clinic_path)
    # Only detect reads urgent_phrases, so an empty list hands every case to the main model.
    clinic = clinic.model_copy(update={"urgent_phrases": []})
    clock = Clock()
    calendar = Calendar(workdir / "calendar.sqlite")
    seed(calendar, clock, clinic)
    return GraphDeps(
        model=create_llm(settings),
        checkpointer=create_checkpointer(settings),
        calendar=calendar,
        clock=clock,
        clinic=clinic,
        retriever=create_retriever(settings, None),
        trace=TraceSink(),
        settings=settings,
        checker=CheckerChain([], {}),  # No checker, so this measures the main model alone.
    )


def run_case(graph, deps: GraphDeps, text: str) -> list[str]:
    """Run one graph turn as a new caller, and return each urgency level the model flagged."""
    deps.trace = TraceSink()  # A new, unbound sink, so no events carry over between runs.
    conversation_id = f"baseline-{uuid.uuid4().hex[:8]}"
    turn_input = start_fields(
        deps.calendar, deps.clinic, deps.clock.now(), conversation_id, NEW_CALLER
    )
    turn_input |= {"last_caller_text": text, "awaiting_yes": False}
    graph.invoke(turn_input, {"configurable": {"thread_id": conversation_id}})
    return [
        event["arguments"]["level"]
        for event in deps.trace.events
        if event.get("kind") == "tool_call" and event.get("name") == "flag_urgent"
    ]


def main() -> None:
    """Run the cases, print one line per case and the three numbers, and save them."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", action="append", type=Path, default=[])
    parser.add_argument("--runs", type=int, default=3, help="Runs per case.")
    args = parser.parse_args()
    cases = load_cases(args.cases or [CASES_PATH, ROOT / "evals" / "intent_holdout.yaml"])
    chosen = cases["emergency"]  # Every case, so false alarms are measured too.

    settings = Settings()
    results = []
    with tempfile.TemporaryDirectory() as workdir:
        deps = build_baseline_deps(settings, Path(workdir))
        graph = build_graph(deps)
        for case in chosen:
            levels = [run_case(graph, deps, case["text"]) for _ in range(args.runs)]
            kind = (
                "life_threatening" if case["label"] else "dental" if case.get("dental") else "none"
            )
            results.append({"id": case["id"], "kind": kind, "label": case["label"], "runs": levels})
            print(f"{case['id']:7} {kind:16} {levels}", flush=True)

    def share(kind: str, hit) -> tuple[int, int]:
        runs = [levels for r in results if r["kind"] == kind for levels in r["runs"]]
        return sum(1 for levels in runs if hit(levels)), len(runs)

    life, life_total = share("life_threatening", lambda levels: "life_threatening" in levels)
    dental, dental_total = share("dental", lambda levels: bool(levels))
    false_alarms = [
        r["id"]
        for r in results
        if r["label"] is False and any("life_threatening" in levels for levels in r["runs"])
    ]
    ratio = dental / dental_total if dental_total else 0.0
    print(f"\nLife-threatening runs flagged life_threatening: {life} of {life_total}")
    print(f"Cases that are not emergencies, flagged life_threatening in any run: {false_alarms}")
    verdict = "at or above" if ratio >= DENTAL_TRIGGER else "BELOW"
    print(
        f"Dental ratio: {dental} of {dental_total} = {ratio:.0%}, "
        f"{verdict} the {DENTAL_TRIGGER:.0%} trigger"
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (OUT_DIR / f"{stamp}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
