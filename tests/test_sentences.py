"""Tests for the fixed sentences. Stage: 1.

What this file is:
    Proves each template gives the right sentence.

What it contains:
    - Each template case under Sentences.

Used by:
    pytest
"""

from dental_agent.sentences import closing_text, confirmation_text, result_text, status_text

SPOKEN = "Tuesday 10 November at 10:00"


def test_book_confirmation_includes_the_name():
    text = confirmation_text("book", "check-up", SPOKEN, name="Ana Ruiz")
    assert text == (
        "Just to confirm: a check-up for Ana Ruiz on Tuesday 10 November at 10:00. "
        "Shall I book that?"
    )


def test_book_confirmation_warns_about_an_existing_appointment():
    text = confirmation_text("book", "check-up", SPOKEN, name="Ana Ruiz", in_addition=True)
    assert ", in addition to your existing appointment. Shall I book that?" in text


def test_result_text_for_a_booking():
    assert result_text("book", "check-up", SPOKEN) == (
        "You're booked for a check-up on Tuesday 10 November at 10:00."
    )


def test_status_text_with_no_actions():
    assert status_text([]) == "Nothing has been changed on this call so far."


def test_status_text_lists_each_result():
    assert status_text(["You're booked."]) == "So far on this call: You're booked."


def test_closing_text_with_no_actions():
    assert closing_text([]) == "Nothing was changed on this call. Goodbye."


def test_closing_text_repeats_each_result():
    assert closing_text(["You're booked."]) == "To confirm: You're booked. Goodbye."


def test_closing_text_when_the_calendar_could_not_be_read():
    text = closing_text([], read_failed_line="910 000 001")
    assert text == (
        "I couldn't check the result, so please call the clinic on 910 000 001. "
        "If this is an emergency, call 112. Goodbye."
    )
