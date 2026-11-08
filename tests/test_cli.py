"""Tests for the command line. Stage: 1.

What this file is:
    Proves the commands and flags behave.

What it contains:
    - Phone checks.
    - Exclusive flags.
    - First-run seeding.
    - Sample numbers.

Used by:
    pytest
"""

import builtins
import json

import pytest
from tests.conftest import PROJECT_ROOT, TEST_NOW
from tests.fakes import ScriptedModel, text_reply

from dental_agent import cli, factories
from dental_agent.booking.calendar import Calendar
from dental_agent.booking.seed import seed
from dental_agent.clock import Clock, FixedClock
from dental_agent.speech.base import SpeakResult, SpeechResult
from dental_agent.speech.text import TextSpeech


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Run the CLI in an empty folder, with a fake model and no keyboard."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLINIC_PATH", str(PROJECT_ROOT / "data" / "clinic.yaml"))
    model = ScriptedModel(script=[text_reply("Hello"), text_reply("Summary.")])
    monkeypatch.setattr(factories, "create_llm", lambda settings: model)
    # No more keyboard input means the caller has hung up.
    monkeypatch.setattr(builtins, "input", _hang_up)
    return tmp_path


def _hang_up(prompt=""):
    """Behave like a closed keyboard."""
    raise EOFError


@pytest.mark.parametrize("phone", ["600000001", "+3460000000", "+346000000011", "+1 600000001"])
def test_a_bad_phone_number_is_refused(project, phone, capsys):
    with pytest.raises(SystemExit):
        cli.main(["chat", "--caller-phone", phone])
    assert "+34" in capsys.readouterr().err


def test_the_two_caller_flags_cannot_be_combined(project):
    with pytest.raises(SystemExit):
        cli.main(["chat", "--new-caller", "--caller-phone", "+34600000001"])


def test_the_first_run_seeds_the_data(project):
    cli.main(["chat"])
    assert Calendar(project / "data" / "calendar.sqlite").slot_count() > 0


def test_a_later_run_keeps_the_data(project):
    cli.main(["seed"])
    calendar = Calendar(project / "data" / "calendar.sqlite")
    before = calendar.slot_count()
    cli.main(["chat"])
    assert calendar.slot_count() == before


def test_chat_prints_sample_numbers(project, capsys):
    cli.main(["chat"])
    assert "+34600000001" in capsys.readouterr().out


def test_chat_defaults_to_the_first_sample_patient(project):
    cli.main(["chat"])
    stored = next((project / "runs" / "conversations").glob("*.json"))
    assert json.loads(stored.read_text())["caller_phone"] == "+34600000001"


def test_new_caller_picks_a_number_with_no_record(project):
    cli.main(["chat", "--new-caller"])
    stored = next((project / "runs" / "conversations").glob("*.json"))
    phone = json.loads(stored.read_text())["caller_phone"]
    assert phone.startswith("+34600")
    assert not Calendar(project / "data" / "calendar.sqlite").has_patient(phone)


def test_seed_refuses_to_overwrite(project, capsys):
    cli.main(["seed"])
    with pytest.raises(SystemExit):
        cli.main(["seed"])


def test_seed_reset_replaces_the_data(project):
    cli.main(["seed"])
    cli.main(["seed", "--reset"])
    assert Calendar(project / "data" / "calendar.sqlite").slot_count() > 0


def test_index_says_it_is_not_available_yet(project, capsys):
    cli.main(["index"])
    assert "not available" in capsys.readouterr().out


def test_old_data_gets_a_warning(project, capsys, clinic):
    calendar = Calendar(project / "data" / "calendar.sqlite")
    # Data seeded long ago has few future slots left.
    seed(calendar, FixedClock(TEST_NOW.replace(year=2020)), clinic)
    assert cli.seed_warning(calendar, Clock()) is not None


def test_new_data_gets_no_warning(project, clinic):
    calendar = Calendar(project / "data" / "calendar.sqlite")
    clock = Clock()
    seed(calendar, clock, clinic)
    assert cli.seed_warning(calendar, clock) is None


def test_text_speech_returns_what_was_typed(monkeypatch):
    monkeypatch.setattr(builtins, "input", lambda prompt="": "  Hello  ")
    result = TextSpeech().transcribe(expect_yes=False)
    assert result.kind == "speech"
    assert result.text == "Hello"


def test_text_speech_treats_an_empty_line_as_no_speech(monkeypatch):
    monkeypatch.setattr(builtins, "input", lambda prompt="": "   ")
    assert TextSpeech().transcribe(expect_yes=False).kind == "no_speech"


def test_text_speech_treats_a_closed_keyboard_as_closed(monkeypatch):
    monkeypatch.setattr(builtins, "input", _hang_up)
    assert TextSpeech().transcribe(expect_yes=False).kind == "closed"


def test_text_speech_prints_the_reply(capsys):
    assert TextSpeech().speak("Hello").ok
    assert "Hello" in capsys.readouterr().out


@pytest.fixture
def checker_setting(monkeypatch):
    """Record whether the call was built with the checker on."""
    seen = []
    real = factories.create_checker

    def record(settings):
        seen.append(settings.checker_enabled)
        return real(settings)

    monkeypatch.setattr(factories, "create_checker", record)
    return seen


def test_the_checker_flag_turns_the_checker_off_for_one_call(project, checker_setting, capsys):
    cli.main(["chat", "--checker", "off"])
    assert checker_setting == [False]
    assert "Checker: off" in capsys.readouterr().out


def test_the_checker_flag_can_turn_the_checker_on(project, checker_setting, monkeypatch, capsys):
    monkeypatch.setenv("CHECKER_ENABLED", "false")
    cli.main(["chat", "--checker", "on"])
    assert checker_setting == [True]
    # The tests set no keys, so the chain is on but has no backend to call.
    assert "Checker: on, but no backend has a key" in capsys.readouterr().out


def test_an_empty_backend_list_is_named_as_the_reason(project, monkeypatch, capsys):
    monkeypatch.setenv("CHECKER_BACKENDS", "")
    cli.main(["chat"])
    assert "Checker: on, but CHECKER_BACKENDS is empty" in capsys.readouterr().out


def test_without_the_flag_the_setting_decides(project, checker_setting, monkeypatch):
    monkeypatch.setenv("CHECKER_ENABLED", "false")
    cli.main(["chat"])
    assert checker_setting == [False]


def test_the_checker_flag_takes_only_on_or_off(project):
    with pytest.raises(SystemExit):
        cli.main(["chat", "--checker", "maybe"])


class HangUpSpeech(TextSpeech):
    """A voice backend stand-in that hangs up at once and records being closed."""

    uses_tone = True

    def __init__(self):
        self.closed = False

    def transcribe(self, expect_yes):
        return SpeechResult(kind="closed", provider="fake")

    def speak(self, text, asks_confirmation=False):
        return SpeakResult(ok=True)

    def close(self):
        self.closed = True


def test_voice_passes_the_fixture_folder_and_releases_the_mic(project, monkeypatch, capsys):
    made = {}

    def fake_speech(settings, mode, keyterms=None, fixture_dir=None):
        made.update(mode=mode, fixture_dir=fixture_dir, speech=HangUpSpeech())
        return made["speech"]

    monkeypatch.setattr(factories, "create_speech", fake_speech)
    cli.main(["voice", "--save-fixtures", "fixtures"])
    assert made["mode"] == "voice"
    assert made["fixture_dir"] == "fixtures"
    assert made["speech"].closed
    assert "Ctrl-C" in capsys.readouterr().out


def test_chat_still_hangs_up_with_ctrl_d(project, capsys):
    cli.main(["chat"])
    assert "Ctrl-D" in capsys.readouterr().out
