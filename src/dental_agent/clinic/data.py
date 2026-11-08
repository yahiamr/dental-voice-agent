"""Loads clinic.yaml. Stage: 1 and 3.

What this file is:
    Reads data/clinic.yaml into a typed Clinic object at startup.

What it contains:
    - Clinic, with hours, services, rules, lines, phrases and word lists.
    - Clinic.line(key, **values), which fills placeholders such as {clinic_line}.
    - Clinic.service_name(service_id), the spoken name of a service, even one no longer offered.

How to understand it:
    Prices and hours live in one file, so the tools answer from data and the model never has to
    remember them. A bad file stops the program at startup with the file and field named, so a
    mistake appears before any call begins.

Uses:
    models.py

Used by:
    guards/, tools/, booking/seed.py, session.py
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError, model_validator

# The Fixed Lines keys. Every key must exist, because code asks for them by name.
REQUIRED_LINES = {
    "greeting",
    "still_there",
    "not_caught",
    "trouble",
    "commit_unknown",
    "cannot_finish",
    "call_clinic",
    "medical",
    "emergency",
    "urgent_message",
    "no_search",
    "unsupported",
}

# The Refusal Lines keys. Each one is an ErrorCode that write_to_calendar can refuse with.
REQUIRED_REFUSALS = {
    "slot_taken",
    "slot_not_offered",
    "not_owner",
    "policy_blocked",
    "first_visit_required",
    "existing_patient",
    "first_visit_booked",
    "missing_field",
}


class Service(BaseModel):
    """One treatment the clinic offers."""

    id: str  # The ID tools use, such as check_up.
    name: str  # The name said aloud, such as check-up.
    price_eur: int
    first_visit: bool = False  # True for the only service a new patient may book.


class Holidays(BaseModel):
    """Which public holidays close the clinic."""

    country: str
    subdivision: str


class Hours(BaseModel):
    """Opening hours as [start, end] pairs of "HH:MM" text."""

    morning: tuple[str, str]
    afternoon: tuple[str, str]


class BookingWindow(BaseModel):
    """How close and how far ahead a booking can be."""

    min_hours_ahead: int
    max_days_ahead: int


class Clinic(BaseModel):
    """Everything in clinic.yaml, checked and typed."""

    name: str
    clinic_line: str
    urgent_line: str
    timezone: str
    holidays: Holidays
    hours: Hours
    open_days: list[str]
    slot_minutes: int
    booking_window: BookingWindow
    notice_hours: int
    services: list[Service]
    lines: dict[str, str]
    refusal_lines: dict[str, str]
    urgent_phrases: list[str]
    negation_tokens: list[str]
    claim_phrases: dict[str, list[str]]
    dosing_words: list[str]

    @model_validator(mode="after")
    def check_required_keys(self) -> "Clinic":
        """Fail at startup if a sentence that code relies on is missing."""
        for field, required in (("lines", REQUIRED_LINES), ("refusal_lines", REQUIRED_REFUSALS)):
            missing = required - set(getattr(self, field))
            if missing:
                raise ValueError(f"{field} is missing: {', '.join(sorted(missing))}")
        return self

    def service(self, service_id: str) -> Service | None:
        """Return the service with this ID, or None if the clinic does not offer it."""
        return next((item for item in self.services if item.id == service_id), None)

    def service_name(self, service_id: str) -> str:
        """Return the service's name. An ID no longer in clinic.yaml is spoken as its words.

        A move or cancel takes its service from the calendar row, which may predate a change to
        clinic.yaml, so the name must never be missing.
        """
        service = self.service(service_id)
        return service.name if service else service_id.replace("_", " ")

    def line(self, key: str, **values: object) -> str:
        """Return a fixed or refusal line with its placeholders filled in."""
        # Fixed lines and refusal lines never share a key, so both can be searched here.
        text = self.lines.get(key) or self.refusal_lines[key]
        # These three are always available, and callers can add more through values.
        return text.format(
            clinic_line=self.clinic_line,
            urgent_line=self.urgent_line,
            notice_hours=self.notice_hours,
            **values,
        )


def load_clinic(path: Path) -> Clinic:
    """Load clinic.yaml, or stop the program with a message naming the file and the field."""
    try:
        raw = yaml.safe_load(Path(path).read_text())
        return Clinic.model_validate(raw)
    except (OSError, yaml.YAMLError) as error:
        # The file is missing or is not valid YAML.
        raise SystemExit(f"Could not read {path}: {error}") from error
    except ValidationError as error:
        # List every bad field, so one run shows all the problems at once.
        problems = "; ".join(
            f"{'.'.join(str(part) for part in item['loc']) or 'file'}: {item['msg']}"
            for item in error.errors()
        )
        raise SystemExit(f"Bad clinic file {path}: {problems}") from error
