"""Shared test fixtures. Stage: 1.

What this file is:
    Setup that every test can use.

What it contains:
    - A FixedClock at Monday 9 November 2026, 10:00.
    - A temporary seeded calendar.
    - clinic.yaml as sample data.
    - build_test_deps(model=None, speech=None, calendar=None, clock=None), which fills unset
      values with fakes.

How to understand it:
    Pytest loads this file automatically. Read build_test_deps first, because most integration
    tests start there.

Used by:
    pytest
"""

from datetime import datetime
from pathlib import Path

import pytest

from dental_agent.booking.calendar import Calendar
from dental_agent.booking.seed import seed
from dental_agent.checker import CheckerChain
from dental_agent.clinic.data import Clinic, load_clinic
from dental_agent.clinic.retriever import NoRetriever
from dental_agent.clock import MADRID, FixedClock
from dental_agent.config import Settings
from dental_agent.deps import Deps, GraphDeps
from dental_agent.factories import create_checkpointer
from dental_agent.state import CallState
from dental_agent.storage.trace import TraceSink

# Every test starts at this moment, so results never depend on the real date.
TEST_NOW = datetime(2026, 11, 9, 10, 0, tzinfo=MADRID)

# Found from this file, so tests work from any folder.
PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture
def clock() -> FixedClock:
    """A clock fixed at Monday 9 November 2026, 10:00."""
    return FixedClock(TEST_NOW)


@pytest.fixture
def clinic() -> Clinic:
    """The real clinic.yaml, used as sample data."""
    return load_clinic(PROJECT_ROOT / "data" / "clinic.yaml")


@pytest.fixture
def empty_calendar(tmp_path) -> Calendar:
    """A calendar with tables and no rows, in a temporary file."""
    return Calendar(tmp_path / "calendar.sqlite")


@pytest.fixture
def calendar(empty_calendar, clock, clinic) -> Calendar:
    """A calendar filled with the sample data."""
    seed(empty_calendar, clock, clinic)
    return empty_calendar


def build_test_deps(
    tmp_path, clinic, clock, calendar, model=None, speech=None, checker=None
) -> Deps:
    """Build Deps from test objects, with no key, network or audio.

    With no checker given, the chain is empty, so the fixed rules decide alone, as on main.
    """
    # _env_file=None stops a developer's own .env from changing test results.
    settings = Settings(_env_file=None, runs_dir=tmp_path / "runs")
    graph = GraphDeps(
        model=model,
        checkpointer=create_checkpointer(settings),
        calendar=calendar,
        clock=clock,
        clinic=clinic,
        retriever=NoRetriever(),
        trace=TraceSink(),
        settings=settings,
        checker=checker if checker is not None else CheckerChain([], {}),
    )
    return Deps(graph=graph, speech=speech)


@pytest.fixture
def deps(tmp_path, clinic, clock, calendar) -> Deps:
    """Deps built from the fixtures above, with no model or speech."""
    return build_test_deps(tmp_path, clinic, clock, calendar)


@pytest.fixture
def state() -> CallState:
    """A new caller at the start of a call."""
    return CallState(conversation_id="c-test", caller_phone="+34600999999")
