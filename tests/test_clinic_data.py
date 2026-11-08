"""Tests for clinic.yaml: the real file loads, and a broken file stops the program early."""

from pathlib import Path

import pytest
import yaml

from dental_agent.clinic.data import load_clinic

CLINIC_FILE = Path("data/clinic.yaml")


def test_the_real_file_loads():
    clinic = load_clinic(CLINIC_FILE)
    assert clinic.name == "Mirador Dental"
    assert clinic.service("check_up").price_eur == 40


def test_no_stays_a_string_in_negation_tokens():
    # Plain YAML can read an unquoted no as false, which would silently break negation.
    clinic = load_clinic(CLINIC_FILE)
    assert "no" in clinic.negation_tokens
    assert all(isinstance(token, str) for token in clinic.negation_tokens)


def test_every_fixed_line_key_exists():
    clinic = load_clinic(CLINIC_FILE)
    expected = {
        "greeting",
        "still_there",
        "not_caught",
        "trouble",
        "commit_unknown",
        "cannot_finish",
        "call_clinic",
        "medical",
        "dentist_referral",
        "emergency",
        "urgent_message",
        "no_search",
        "unsupported",
    }
    assert set(clinic.lines) == expected


def test_every_refusal_line_key_exists():
    clinic = load_clinic(CLINIC_FILE)
    expected = {
        "slot_taken",
        "slot_not_offered",
        "not_owner",
        "policy_blocked",
        "first_visit_required",
        "existing_patient",
        "first_visit_booked",
        "missing_field",
    }
    assert set(clinic.refusal_lines) == expected


def test_line_fills_placeholders():
    clinic = load_clinic(CLINIC_FILE)
    assert clinic.line("call_clinic") == "Please call the clinic on 910 000 001."
    assert "24-hour" in clinic.line("policy_blocked")


def test_a_bad_file_exits_naming_the_file_and_field(tmp_path):
    good = yaml.safe_load(CLINIC_FILE.read_text())
    del good["notice_hours"]
    bad_file = tmp_path / "clinic.yaml"
    bad_file.write_text(yaml.safe_dump(good))

    with pytest.raises(SystemExit) as caught:
        load_clinic(bad_file)

    message = str(caught.value)
    assert "clinic.yaml" in message
    assert "notice_hours" in message


def test_a_missing_line_key_exits_naming_the_key(tmp_path):
    good = yaml.safe_load(CLINIC_FILE.read_text())
    del good["lines"]["greeting"]
    bad_file = tmp_path / "clinic.yaml"
    bad_file.write_text(yaml.safe_dump(good))

    with pytest.raises(SystemExit) as caught:
        load_clinic(bad_file)

    assert "greeting" in str(caught.value)
