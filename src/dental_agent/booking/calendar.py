"""All calendar reads and writes. Stage: 1 and 4.

What this file is:
    The only code that touches the calendar database.

What it contains:
    - Calendar(path), which opens the file and applies the schema.
    - Reads: free_slots, get_slot, slot_is_free, patient_name, patient_status, has_patient,
      patient_count, sample_patients, appointments_for, get_appointment, has_active_service,
      get_action, actions_for_call.
    - Writes: book, move, cancel, mark_attended, seed_rows.
    - Errors: SlotTaken, NotActive.

How to understand it:
    Each write changes the appointment and adds its actions row in one transaction, so both
    happen or neither does. Writes check get_action first and return the stored record if the ID
    exists, so a repeated commit cannot book twice. All SQL uses bound parameters, so caller
    text never becomes part of a query. One number is one patient: book finds the patient by
    number, so a known number always books for the patient on record. clear drops and rebuilds
    the tables, so seed --reset also applies schema changes.

Uses:
    schema.sql, models.py

Used by:
    tools/read_tools.py, guards/validate.py, graph/nodes.py, booking/seed.py
"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path

from dental_agent.models import ActionRecord, Appointment, PendingAction, Slot

# The schema sits next to this file, so it ships with the package.
SCHEMA_FILE = Path(__file__).with_name("schema.sql")

# SQLite's strftime('%w') gives 0 for Sunday, so weekday names map to these digits.
WEEKDAY_NUMBERS = {"mon": "1", "tue": "2", "wed": "3", "thu": "4", "fri": "5"}


class CalendarError(Exception):
    """A database problem. write_to_calendar then checks whether the write happened anyway."""


class SlotTaken(CalendarError):
    """Another appointment already holds this slot."""


class NotActive(CalendarError):
    """The appointment to move or cancel is no longer active, or no longer in the slot the call
    last saw. write_to_calendar maps it to not_owner."""


class Calendar:
    """The SQLite calendar. All SQL uses bound parameters, so text never becomes SQL."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        # Create the folder on first use, because a fresh checkout has no database yet.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            # Every statement uses IF NOT EXISTS, so this is safe on an existing file.
            connection.executescript(SCHEMA_FILE.read_text())

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open one connection for one piece of work, and close it afterwards.

        A new connection each time avoids SQLite's rule that a connection stays on one
        thread, because LangGraph may run a node on another thread.
        """
        connection = sqlite3.connect(self.path)
        try:
            # Rows can be read by column name.
            connection.row_factory = sqlite3.Row
            # SQLite ignores foreign keys unless this is set on each connection.
            connection.execute("PRAGMA foreign_keys = ON")
            # This block is one transaction: it commits at the end, or rolls back on an error.
            with connection:
                yield connection
        finally:
            connection.close()

    def free_slots(
        self,
        date_from: date,
        date_to: date,
        weekdays: list[str] | None,
        time_window: tuple[str, str] | None,
        limit: int,
        earliest: datetime,
    ) -> list[Slot]:
        """Return the earliest free slots.

        time_window is a ("HH:MM", "HH:MM") pair, and earliest is the first start time a
        caller can book. The calendar does not read clinic.yaml, so the tool works both out.
        """
        # The first ten characters of the stored text are the date, and the next part the time.
        query = """
            SELECT slots.* FROM slots
            WHERE substr(start_at, 1, 10) BETWEEN ? AND ?
              AND start_at >= ?
              AND id NOT IN (SELECT slot_id FROM appointments WHERE status = 'active')
        """
        values: list[object] = [date_from.isoformat(), date_to.isoformat(), earliest.isoformat()]
        if weekdays:
            # One placeholder per weekday, with the values still bound separately.
            marks = ", ".join("?" for _ in weekdays)
            query += f" AND strftime('%w', substr(start_at, 1, 10)) IN ({marks})"
            values += [WEEKDAY_NUMBERS[day] for day in weekdays]
        if time_window:
            query += " AND substr(start_at, 12, 5) >= ? AND substr(start_at, 12, 5) < ?"
            values += list(time_window)
        query += " ORDER BY start_at LIMIT ?"
        values.append(limit)
        with self._connect() as connection:
            return [_slot(row) for row in connection.execute(query, values)]

    def get_slot(self, db_slot_id: int) -> Slot | None:
        """Return one slot, or None if the ID does not exist."""
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM slots WHERE id = ?", (db_slot_id,)).fetchone()
        return _slot(row) if row else None

    def slot_is_free(self, db_slot_id: int) -> bool:
        """Return True if no active appointment holds this slot."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM appointments WHERE slot_id = ? AND status = 'active'",
                (db_slot_id,),
            ).fetchone()
        return row is None

    def patient_name(self, phone: str) -> str | None:
        """Return the name on record for this number, or None for a new caller."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT name FROM patients WHERE phone = ?", (phone,)
            ).fetchone()
        return row["name"] if row else None

    def patient_status(self, phone: str) -> str | None:
        """Return "new" or "existing" for a number on record, or None for a new caller."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT status FROM patients WHERE phone = ?", (phone,)
            ).fetchone()
        return row["status"] if row else None

    def mark_attended(self, phone: str, now: datetime, first_visit: str) -> bool:
        """Make a new patient existing once a first visit's time has passed. Return True if so.

        There is no front desk to record who turned up, so a first visit that was not
        cancelled counts as attended once its time has passed. A no-show therefore counts as
        attended, which the README lists as a limit. The flag never goes back to new.
        """
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT slots.start_at FROM appointments
                JOIN patients ON patients.id = appointments.patient_id
                JOIN slots ON slots.id = appointments.slot_id
                WHERE patients.phone = ? AND patients.status = 'new'
                  AND appointments.service = ? AND appointments.status = 'active'
                """,
                (phone, first_visit),
            ).fetchall()
            # Times are compared as datetimes, because text comparison breaks across a
            # daylight saving change.
            if not any(datetime.fromisoformat(row["start_at"]) <= now for row in rows):
                return False
            connection.execute("UPDATE patients SET status = 'existing' WHERE phone = ?", (phone,))
        return True

    def has_patient(self, phone: str) -> bool:
        """Return True if this number has any patient record."""
        return self.patient_count(phone) > 0

    def patient_count(self, phone: str) -> int:
        """Return how many patients have this number: 0 or 1, as one number is one patient."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS total FROM patients WHERE phone = ?", (phone,)
            ).fetchone()
        return row["total"]

    def sample_patients(self, limit: int) -> list[tuple[str, str]]:
        """Return seeded (phone, name) pairs, so the CLI can print numbers to try."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT phone, name FROM patients WHERE phone LIKE '+3460000000%' "
                "ORDER BY phone LIMIT ?",
                (limit,),
            ).fetchall()
        return [(row["phone"], row["name"]) for row in rows]

    def appointments_for(
        self, phone: str, limit: int, after: datetime | None = None
    ) -> list[Appointment]:
        """Return this number's active appointments, earliest first.

        With after, only appointments starting after that moment: one that has already started
        cannot be moved or cancelled, so the call never lists it as one to change. The result
        has no per-call IDs or spoken text. The lookup tool adds those.
        """
        # With no moment given, every active appointment is listed.
        after_at = after.isoformat() if after else ""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT appointments.id AS db_id, appointments.service, appointments.slot_id,
                       appointments.status, patients.name, slots.start_at
                FROM appointments
                JOIN patients ON patients.id = appointments.patient_id
                JOIN slots ON slots.id = appointments.slot_id
                WHERE patients.phone = ? AND appointments.status = 'active'
                  AND slots.start_at > ?
                ORDER BY slots.start_at LIMIT ?
                """,
                (phone, after_at, limit),
            ).fetchall()
        return [_appointment(row) for row in rows]

    def get_appointment(self, db_id: int) -> Appointment | None:
        """Return one appointment in any status, or None if no row has this ID."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT appointments.id AS db_id, appointments.service, appointments.slot_id,
                       appointments.status, patients.name, slots.start_at
                FROM appointments
                JOIN patients ON patients.id = appointments.patient_id
                JOIN slots ON slots.id = appointments.slot_id
                WHERE appointments.id = ?
                """,
                (db_id,),
            ).fetchone()
        return _appointment(row) if row else None

    def has_active_service(self, phone: str, service: str) -> bool:
        """Return True if this number already has that service booked."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM appointments
                JOIN patients ON patients.id = appointments.patient_id
                WHERE patients.phone = ? AND appointments.service = ?
                  AND appointments.status = 'active'
                """,
                (phone, service),
            ).fetchone()
        return row is not None

    def book(
        self,
        action: PendingAction,
        conversation_id: str,
        phone: str,
        name: str,
        service: str,
        spoken: str,
        result_text: str,
        now: datetime,
    ) -> ActionRecord:
        """Write a booking. The patient, appointment and action rows land together or not at all.

        Raises SlotTaken if the slot has gone, and CalendarError on any other database problem.
        """
        # A repeated commit returns the first result and writes nothing.
        existing = self.get_action(action.action_id)
        if existing:
            return existing
        slot_id = action.offer.db_slot_id
        created_at = now.isoformat()
        try:
            with self._connect() as connection:
                # INSERT OR IGNORE keeps the row a returning caller already has, so a known
                # number always books for the patient on record, whatever name is passed.
                connection.execute(
                    "INSERT OR IGNORE INTO patients (phone, name, created_at) VALUES (?, ?, ?)",
                    (phone, name, created_at),
                )
                patient_id = connection.execute(
                    "SELECT id FROM patients WHERE phone = ?", (phone,)
                ).fetchone()["id"]
                cursor = connection.execute(
                    "INSERT INTO appointments (patient_id, slot_id, service, status, created_at) "
                    "VALUES (?, ?, ?, 'active', ?)",
                    (patient_id, slot_id, service, created_at),
                )
                appointment_id = cursor.lastrowid
                connection.execute(
                    "INSERT INTO actions (id, conversation_id, kind, appointment_id, new_slot_id, "
                    "spoken, result_text, created_at) VALUES (?, ?, 'book', ?, ?, ?, ?, ?)",
                    (
                        action.action_id,
                        conversation_id,
                        appointment_id,
                        slot_id,
                        spoken,
                        result_text,
                        created_at,
                    ),
                )
        except sqlite3.IntegrityError as error:
            # Only the one_active_per_slot index means the slot is taken.
            if "appointments.slot_id" in str(error):
                raise SlotTaken(str(slot_id)) from error
            raise CalendarError(str(error)) from error
        except sqlite3.Error as error:
            raise CalendarError(str(error)) from error
        return ActionRecord(
            action_id=action.action_id,
            conversation_id=conversation_id,
            kind="book",
            appointment_id=appointment_id,
            new_slot_id=slot_id,
            spoken=spoken,
            result_text=result_text,
            created_at=created_at,
        )

    def move(
        self,
        action: PendingAction,
        conversation_id: str,
        spoken: str,
        result_text: str,
        now: datetime,
    ) -> ActionRecord:
        """Move an appointment to the offered slot. The update and its action row land together.

        It is one UPDATE of slot_id on the same row, so a failure leaves the old appointment as
        it was. Raises NotActive if the appointment is no longer active or has moved since the
        call looked it up, SlotTaken if the new slot has gone, and CalendarError on any other
        database problem.
        """
        existing = self.get_action(action.action_id)
        if existing:
            return existing
        appointment = action.appointment
        old_slot_id = appointment.db_slot_id
        new_slot_id = action.offer.db_slot_id
        return self._change(
            action,
            conversation_id,
            "UPDATE appointments SET slot_id = ? "
            "WHERE id = ? AND slot_id = ? AND status = 'active'",
            (new_slot_id, appointment.db_id, old_slot_id),
            old_slot_id,
            new_slot_id,
            spoken,
            result_text,
            now,
        )

    def cancel(
        self,
        action: PendingAction,
        conversation_id: str,
        spoken: str,
        result_text: str,
        now: datetime,
    ) -> ActionRecord:
        """Cancel an appointment. The row stays, marked cancelled, and its slot is free again.

        Raises NotActive if the appointment is no longer active or has moved since the call
        looked it up, and CalendarError on any other database problem.
        """
        existing = self.get_action(action.action_id)
        if existing:
            return existing
        appointment = action.appointment
        return self._change(
            action,
            conversation_id,
            "UPDATE appointments SET status = 'cancelled' "
            "WHERE id = ? AND slot_id = ? AND status = 'active'",
            (appointment.db_id, appointment.db_slot_id),
            appointment.db_slot_id,
            None,
            spoken,
            result_text,
            now,
        )

    def _change(
        self,
        action: PendingAction,
        conversation_id: str,
        update_sql: str,
        update_values: tuple,
        old_slot_id: int,
        new_slot_id: int | None,
        spoken: str,
        result_text: str,
        now: datetime,
    ) -> ActionRecord:
        """Run one update on an active appointment and add its action row, in one transaction."""
        created_at = now.isoformat()
        appointment_id = action.appointment.db_id
        try:
            with self._connect() as connection:
                # The update matches only an active row still in the slot the call saw, so a
                # cancelled or moved row changes nothing, and the caller's yes is not misused.
                if connection.execute(update_sql, update_values).rowcount == 0:
                    raise NotActive(str(appointment_id))
                connection.execute(
                    "INSERT INTO actions (id, conversation_id, kind, appointment_id, old_slot_id, "
                    "new_slot_id, spoken, result_text, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        action.action_id,
                        conversation_id,
                        action.kind,
                        appointment_id,
                        old_slot_id,
                        new_slot_id,
                        spoken,
                        result_text,
                        created_at,
                    ),
                )
        except sqlite3.IntegrityError as error:
            # Only the one_active_per_slot index means the new slot is taken.
            if "appointments.slot_id" in str(error):
                raise SlotTaken(str(new_slot_id)) from error
            raise CalendarError(str(error)) from error
        except sqlite3.Error as error:
            raise CalendarError(str(error)) from error
        return ActionRecord(
            action_id=action.action_id,
            conversation_id=conversation_id,
            kind=action.kind,
            appointment_id=appointment_id,
            old_slot_id=old_slot_id,
            new_slot_id=new_slot_id,
            spoken=spoken,
            result_text=result_text,
            created_at=created_at,
        )

    def get_action(self, action_id: str) -> ActionRecord | None:
        """Return the stored record for this action ID, or None if it was never written."""
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM actions WHERE id = ?", (action_id,)).fetchone()
        return _action(row) if row else None

    def actions_for_call(self, conversation_id: str) -> list[ActionRecord]:
        """Return every change made in this call, oldest first."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM actions WHERE conversation_id = ? ORDER BY created_at, rowid",
                (conversation_id,),
            ).fetchall()
        return [_action(row) for row in rows]

    def seed_rows(
        self,
        patients: list[tuple[int, str, str, str, str]],
        slots: list[tuple[int, str, str]],
        appointments: list[tuple[int, int, str, str, str]],
    ) -> None:
        """Insert sample rows with set IDs. Only seeding uses this."""
        with self._connect() as connection:
            connection.executemany(
                "INSERT INTO patients (id, phone, name, status, created_at) VALUES (?, ?, ?, ?, ?)",
                patients,
            )
            connection.executemany(
                "INSERT INTO slots (id, start_at, end_at) VALUES (?, ?, ?)", slots
            )
            connection.executemany(
                "INSERT INTO appointments (patient_id, slot_id, service, status, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                appointments,
            )

    def clear(self) -> None:
        """Drop every table and build them again from the schema. Used by seed --reset.

        Dropping, not just deleting rows, means a file made with an older schema gets the
        current one, such as one patient per number.
        """
        with self._connect() as connection:
            for table in ("actions", "appointments", "patients", "slots"):
                # The table name comes from the fixed list above, never from input.
                connection.execute(f"DROP TABLE IF EXISTS {table}")
            connection.executescript(SCHEMA_FILE.read_text())

    def slot_count(self) -> int:
        """Return how many slots exist. Zero means the data has not been seeded."""
        with self._connect() as connection:
            return connection.execute("SELECT COUNT(*) AS total FROM slots").fetchone()["total"]

    def last_slot_start(self) -> datetime | None:
        """Return the latest slot's start, so startup can warn when slots are running out."""
        with self._connect() as connection:
            row = connection.execute("SELECT MAX(start_at) AS last FROM slots").fetchone()
        return datetime.fromisoformat(row["last"]) if row["last"] else None


def _slot(row: sqlite3.Row) -> Slot:
    """Turn a slots row into a Slot."""
    return Slot(
        db_slot_id=row["id"],
        start_at=datetime.fromisoformat(row["start_at"]),
        end_at=datetime.fromisoformat(row["end_at"]),
    )


def _appointment(row: sqlite3.Row) -> Appointment:
    """Turn an appointments row, joined to its patient and slot, into an Appointment."""
    return Appointment(
        db_id=row["db_id"],
        patient_name=row["name"],
        service=row["service"],
        db_slot_id=row["slot_id"],
        start_at=datetime.fromisoformat(row["start_at"]),
        status=row["status"],
    )


def _action(row: sqlite3.Row) -> ActionRecord:
    """Turn an actions row into an ActionRecord."""
    return ActionRecord(
        action_id=row["id"],
        conversation_id=row["conversation_id"],
        kind=row["kind"],
        appointment_id=row["appointment_id"],
        old_slot_id=row["old_slot_id"],
        new_slot_id=row["new_slot_id"],
        spoken=row["spoken"],
        result_text=row["result_text"],
        created_at=row["created_at"],
    )
