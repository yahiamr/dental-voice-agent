-- Calendar database schema. Stage: 1.
--
-- What this file is:
--     The tables that hold the calendar.
--
-- What it will contain:
--     - patients: one row per phone number, because one number is one patient. status is
--       new until the first visit is attended, then existing for good.
--     - slots: bookable times.
--     - appointments: who has which slot, marked active or cancelled.
--     - actions: one row per change, with spoken and result_text.
--     - one_active_per_slot: the partial unique index.
--
-- How to understand it:
--     The index is the most important line. It allows only one active appointment per slot, so the
--     database itself refuses a double booking, even if two commits run at once. It covers active
--     rows only, so a cancelled slot can be booked again. Every statement uses IF NOT EXISTS, so
--     applying the file twice is safe.
--
-- Used by:
--     booking/calendar.py

-- Dates and times are ISO 8601 text with the Madrid offset. All statements use IF NOT EXISTS,
-- so this file can run every time the calendar opens.

-- One row per phone number. The number is how a caller is recognised, so one number is one
-- patient, and their name is kept as first confirmed.
CREATE TABLE IF NOT EXISTS patients (
  id INTEGER PRIMARY KEY,
  phone TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL,
  -- new until a first visit is attended, then existing, even with no appointments left.
  status TEXT NOT NULL DEFAULT 'new' CHECK (status IN ('new', 'existing')),
  created_at TEXT NOT NULL
);

-- Every bookable time. One slot holds one patient.
CREATE TABLE IF NOT EXISTS slots (
  id INTEGER PRIMARY KEY,
  start_at TEXT NOT NULL UNIQUE,
  end_at TEXT NOT NULL
);

-- Who has which slot. A cancelled row stays, as a record, and frees the slot.
CREATE TABLE IF NOT EXISTS appointments (
  id INTEGER PRIMARY KEY,
  patient_id INTEGER NOT NULL REFERENCES patients(id),
  slot_id INTEGER NOT NULL REFERENCES slots(id),
  service TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('active', 'cancelled')),
  created_at TEXT NOT NULL
);

-- The rule that makes a double booking impossible: at most one active appointment per slot.
CREATE UNIQUE INDEX IF NOT EXISTS one_active_per_slot ON appointments(slot_id) WHERE status = 'active';

-- One row per committed change, written in the same transaction as that change.
-- id is the action ID, so a repeated commit finds this row and returns the first result.
CREATE TABLE IF NOT EXISTS actions (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  kind TEXT NOT NULL CHECK (kind IN ('book', 'move', 'cancel')),
  appointment_id INTEGER NOT NULL REFERENCES appointments(id),
  old_slot_id INTEGER REFERENCES slots(id),
  new_slot_id INTEGER REFERENCES slots(id),
  spoken TEXT NOT NULL,
  result_text TEXT NOT NULL,
  created_at TEXT NOT NULL
);
