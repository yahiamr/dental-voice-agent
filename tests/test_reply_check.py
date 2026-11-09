"""Tests for the reply check. Stage: 1.

What this file is:
    Proves a false claim never reaches the caller.

What it contains:
    - Claims with and without a matching action.
    - A true recap and true statements about existing appointments.
    - A false claim about a second booking.
    - Capitals in spoken, dosing words and trimming.
    - Clinic phone numbers: a shortened or changed one is corrected to the exact number.
    - The lead-in check: a short sentence the model may say before a confirmation question.
    - The checker's flags: combine_reply, the decision order and its reasons, and r27.

Used by:
    pytest
"""

from itertools import product

import pytest
from tests.conftest import TEST_NOW

from dental_agent.guards.reply_check import check_lead_in, check_reply, claimed_kinds, combine_reply
from dental_agent.models import ActionRecord, Appointment

SPOKEN = "Tuesday 10 November at 10:00"
OTHER = "Thursday 12 November at 11:00"


def action(kind="book", spoken=SPOKEN):
    """Build a stored action."""
    return ActionRecord(
        action_id="a",
        conversation_id="c",
        kind=kind,
        appointment_id=1,
        spoken=spoken,
        result_text=f"You're booked for {spoken}.",
        created_at="now",
    )


def appointment(spoken=SPOKEN):
    """Build an appointment the caller already has."""
    return Appointment(
        db_id=1,
        patient_name="Ana",
        service="check_up",
        db_slot_id=1,
        start_at=TEST_NOW,
        status="active",
        spoken=spoken,
    )


def run(text, clinic, actions=(), found=(), max_sentences=3, **flags):
    """Run check_reply with short defaults. flags are model_claim and model_advice."""
    return check_reply(text, list(actions), list(found), clinic, max_sentences, **flags)


def test_a_plain_reply_passes_unchanged(clinic):
    decision = run("Which day suits you?", clinic)
    assert decision.text == "Which day suits you?"
    assert not decision.replaced


def test_a_false_claim_is_replaced_by_the_status(clinic):
    decision = run("You're all set for Tuesday at 10. Would you like anything else?", clinic)
    assert decision.replaced
    assert decision.reason == "claim"
    assert decision.text == "Nothing has been changed on this call so far."


def test_a_true_recap_passes(clinic):
    text = f"You're booked for {SPOKEN}."
    assert not run(text, clinic, actions=[action()]).replaced


def test_a_claim_about_a_second_booking_is_replaced(clinic):
    # A real booking at one time does not prove a claim about another time.
    decision = run(f"You're booked for {OTHER}.", clinic, actions=[action()])
    assert decision.replaced
    assert "So far on this call" in decision.text


def test_a_true_statement_about_an_existing_appointment_passes(clinic):
    text = f"You're all set for your check-up on {SPOKEN}."
    assert not run(text, clinic, found=[appointment()]).replaced


@pytest.mark.parametrize(
    "text",
    [
        f"I've cancelled your check-up on {SPOKEN}.",
        f"I've moved your check-up from {SPOKEN} to {OTHER}.",
        f"Your check-up is now on {SPOKEN}.",
    ],
)
def test_a_move_or_cancel_claim_is_not_proved_by_a_found_appointment(text, clinic):
    # A false cancel or move names the appointment's own time, so only a real record proves it.
    decision = run(text, clinic, found=[appointment()])
    assert decision.replaced
    assert decision.reason == "claim"


@pytest.mark.parametrize(
    ("kind", "text"),
    [
        ("cancel", f"Your check-up on {SPOKEN} is cancelled."),
        ("move", f"Your check-up is now on {SPOKEN}."),
    ],
)
def test_a_move_or_cancel_claim_passes_with_its_own_record(kind, text, clinic):
    assert not run(text, clinic, actions=[action(kind)]).replaced


# Wordings a live model uses, found missing from the phrase lists in the Stage 3 review. With
# the checker off or failing, the phrases are the only defence against these false claims.
@pytest.mark.parametrize(
    "text",
    [
        f"That's cancelled: your check-up on {SPOKEN}.",
        f"It's cancelled, so {SPOKEN} is free again.",
        f"Your check-up on {SPOKEN} has been canceled.",
        f"I've canceled your check-up on {SPOKEN}.",
        f"I've rescheduled your check-up to {OTHER}.",
        f"Your check-up has been rescheduled to {OTHER}.",
        f"That's been moved to {OTHER}.",
    ],
)
def test_common_move_and_cancel_wordings_are_claims(text, clinic):
    decision = run(text, clinic, found=[appointment()])
    assert decision.replaced
    assert decision.reason == "claim"


@pytest.mark.parametrize(
    "text",
    [
        f"Your check-up on {SPOKEN} is inside the notice period, so I can't cancel it here.",
        f"I can't move your check-up on {SPOKEN}, so please call the clinic.",
        f"Would you like your check-up on {SPOKEN} rescheduled?",
    ],
)
def test_refusals_and_offers_about_a_found_appointment_still_pass(text, clinic):
    assert not run(text, clinic, found=[appointment()]).replaced


def test_a_claim_of_the_wrong_kind_is_replaced(clinic):
    # A booking does not prove a claim that something was moved.
    decision = run(f"I've moved it to {SPOKEN}.", clinic, actions=[action("book")])
    assert decision.replaced


def test_matching_ignores_capitals_in_spoken(clinic):
    assert not run(f"You're booked for {SPOKEN.upper()}.", clinic, actions=[action()]).replaced


def test_dosing_words_are_replaced_by_the_medical_line(clinic):
    decision = run("Take 400 mg of ibuprofen.", clinic)
    assert decision.replaced
    assert decision.reason == "dosing"
    assert decision.text == clinic.line("medical")


def test_a_dosing_word_inside_another_word_passes(clinic):
    # "mg" must match a whole word, not part of another one.
    assert not run("We have a branch in Birmingham.", clinic).replaced


def test_slot_ids_and_markdown_are_removed(clinic):
    assert run("**Tuesday** is s1, or a2.", clinic).text == "Tuesday is , or ."


def test_a_long_reply_is_trimmed(clinic):
    decision = run("One. Two. Three. Four.", clinic, max_sentences=2)
    assert decision.text == "One. Two."
    assert not decision.replaced


def test_a_shortened_clinic_number_is_corrected(clinic):
    # A live reply said "910 000 1". A caller given a wrong number cannot reach the clinic.
    for wrong in ("910 000 1", "910000001", "910 0001", "910-000-001"):
        decision = run(f"Please call the clinic on {wrong}.", clinic)
        assert decision.text == f"Please call the clinic on {clinic.clinic_line}.", wrong
        assert not decision.replaced, wrong
        assert decision.reason == "phone", wrong


def test_a_changed_urgent_number_is_corrected(clinic):
    decision = run("Our urgent line is 910 000 02.", clinic)
    assert decision.text == f"Our urgent line is {clinic.urgent_line}."
    assert decision.reason == "phone"


def test_an_exact_clinic_number_passes_unchanged(clinic):
    text = f"Please call the clinic on {clinic.clinic_line} or the urgent line on {clinic.urgent_line}."
    decision = run(text, clinic)
    assert decision.text == text
    assert decision.reason is None


def test_prices_times_and_other_phone_numbers_are_not_touched(clinic):
    # Short numbers and numbers far from the clinic's, such as the caller's own, stay as written.
    text = "A first visit costs 30 euros at 10:00. Call 112 now. I have your number as 600 123 456."
    decision = run(text, clinic)
    assert decision.text == text
    assert decision.reason is None


def lead_in(text, clinic, service="cleaning", actions=(), found=()):
    """Run check_lead_in for a booking of this service."""
    return check_lead_in(text, service, list(actions), list(found), clinic)


def test_a_lead_in_that_answers_a_side_question_is_kept(clinic):
    decision = lead_in("A cleaning takes about 45 minutes.", clinic)
    assert decision.text == "A cleaning takes about 45 minutes."
    assert not decision.replaced


def test_an_empty_lead_in_stays_empty(clinic):
    decision = lead_in("  ", clinic)
    assert decision.text == ""
    assert not decision.replaced


def test_a_lead_in_is_cut_to_one_sentence(clinic):
    assert lead_in("It costs 60 euros. Please bring your ID.", clinic).text == "It costs 60 euros."


def test_a_lead_in_with_a_question_is_dropped(clinic):
    # The caller's yes must answer the code's question only.
    decision = lead_in("Would you like a reminder?", clinic)
    assert decision.text == ""
    assert decision.reason == "question"


def test_a_lead_in_with_a_day_time_or_date_is_dropped(clinic):
    # Only the code's question may say when, so the two can never disagree.
    for text in (
        "Friday works well.",
        "We open at 9:00.",
        "We open at 9am.",
        "The 6 November slot is quiet.",
        "Tomorrow is fine.",
    ):
        decision = lead_in(text, clinic)
        assert decision.text == "", text
        assert decision.reason == "time", text


def test_a_lead_in_naming_another_service_is_dropped(clinic):
    decision = lead_in("A check-up takes 30 minutes.", clinic)
    assert decision.text == ""
    assert decision.reason == "service"


def test_a_lead_in_may_name_the_service_being_booked_in_any_spelling(clinic):
    for text in (
        "Your first visit takes 45 minutes.",
        "A first-visit consultation costs 30 euros.",
    ):
        assert lead_in(text, clinic, service="first_visit").text == text


def test_a_lead_in_with_a_false_claim_is_dropped(clinic):
    decision = lead_in("You're all set.", clinic)
    assert decision.text == ""
    assert decision.reason == "claim"


def test_a_lead_in_with_a_dosing_word_is_dropped(clinic):
    decision = lead_in("Take 400 mg of ibuprofen before you come.", clinic)
    assert decision.text == ""
    assert decision.reason == "dosing"


# ---------- the checker's flags ----------

R27 = "I've booked you for Monday at 10:00, and take some painkillers until then."


@pytest.mark.parametrize(
    ("phrase_claim", "dosing_hit", "model_claim", "model_advice"),
    list(product([False, True], repeat=4)),
)
def test_combine_reply_raises_a_flag_when_either_layer_does(
    phrase_claim, dosing_hit, model_claim, model_advice
):
    claim, advice = combine_reply(phrase_claim, dosing_hit, model_claim, model_advice)
    assert claim is (phrase_claim or model_claim)
    assert advice is (dosing_hit or model_advice)


def test_advice_from_the_checker_gives_the_medical_line(clinic):
    decision = run("Try rinsing with warm salt water until your visit.", clinic, model_advice=True)
    assert decision.text == clinic.line("medical")
    assert decision.reason == "model_advice"


def test_a_dosing_word_still_gives_the_reason_dosing_when_the_checker_agrees(clinic):
    decision = run("Take 400 mg of ibuprofen.", clinic, model_advice=True)
    assert decision.reason == "dosing"


def test_r27_gives_the_medical_line_because_advice_wins(clinic):
    # Without the checker, only the claim phrase catches it: the dosing word "painkiller"
    # doesn't match "painkillers", so the advice is missed.
    assert run(R27, clinic).reason == "claim"
    decision = run(R27, clinic, model_claim=True, model_advice=True)
    assert decision.text == clinic.line("medical")
    assert decision.reason == "model_advice"


def test_a_checker_claim_with_no_record_gives_the_status_line(clinic):
    decision = run("I've taken care of the cancellation for you.", clinic, model_claim=True)
    assert decision.text == "Nothing has been changed on this call so far."
    assert decision.reason == "model_claim"


def test_a_checker_claim_passes_with_a_record_of_any_kind(clinic):
    # A checker claim has no kind, so the spoken time of any action is enough proof.
    text = f"That's taken care of for {SPOKEN}."
    assert not run(text, clinic, actions=[action(kind="cancel")], model_claim=True).replaced
    assert not run(text, clinic, found=[appointment()], model_claim=True).replaced


def test_a_claim_phrase_keeps_its_same_kind_proof_when_the_checker_agrees(clinic):
    # "you're booked" needs a booking. A cancellation at the same time is not proof.
    text = f"You're booked for {SPOKEN}."
    decision = run(text, clinic, actions=[action(kind="cancel")], model_claim=True)
    assert decision.reason == "claim"


def test_with_both_flags_false_the_reply_is_checked_as_on_main(clinic):
    assert run("Which day suits you?", clinic).text == "Which day suits you?"
    assert run("Try rinsing with warm salt water.", clinic).replaced is False


def test_a_lead_in_flagged_as_advice_by_the_checker_is_dropped(clinic):
    decision = check_lead_in(
        "Rinse with salt water until then.", "check_up", [], [], clinic, model_advice=True
    )
    assert decision.text == ""
    assert decision.reason == "model_advice"


NEW_CLAIMS = [
    "Your appointment is confirmed for {spoken}.",
    "Your first visit is scheduled for {spoken}.",
    "That's sorted, we'll see you on {spoken}.",
    "I've put you down for {spoken}.",
    "I have put you down for {spoken}.",
]


@pytest.mark.parametrize("template", NEW_CLAIMS)
def test_each_new_claim_phrase_needs_a_booking(template, clinic):
    text = template.format(spoken=SPOKEN)
    assert run(text, clinic).reason == "claim"
    assert not run(text, clinic, actions=[action(kind="book")]).replaced


OFFERS = [
    "I can put you down for {spoken} if you'd like?",
    "Shall I put you down for {spoken}?",
    "Are you all done, or is there anything else?",
    "We're all done here, have a good day.",
    "Great, shall I get you all set for {spoken}?",
    "Would you like me to get you all set for {spoken}?",
    "Is that all set for you, or would another day suit?",
    "I've put you down as Maria Lopez. Which day suits you?",
]


@pytest.mark.parametrize("template", OFFERS)
def test_an_offer_or_a_goodbye_is_not_a_claim(template, clinic):
    # A phrase inside a question or an offer is not a claim. "I've put you down as" records a
    # detail, so only "I've put you down for" counts, and "all done" is not a phrase at all.
    text = template.format(spoken=SPOKEN)
    assert run(text, clinic).text == text


def test_a_claim_followed_by_a_question_is_still_a_claim(clinic):
    # Only the clause holding the phrase counts, so a question after it doesn't hide it.
    text = f"You're booked for {SPOKEN}, anything else?"
    assert run(text, clinic).reason == "claim"
    assert not run(text, clinic, actions=[action(kind="book")]).replaced


CLAIMS_WITH_A_QUESTION = [
    "You're all set for {spoken} — anything else?",
    "Great news you're booked for {spoken} so is there anything else?",
    "Can I help with anything else now you're booked for {spoken}?",
]


@pytest.mark.parametrize("template", CLAIMS_WITH_A_QUESTION)
def test_a_claim_in_the_same_clause_as_a_question_is_still_a_claim(template, clinic):
    # Only an offer that opens the clause and leads straight to the phrase claims nothing. A
    # stated claim before a question, or far after an opener, is still caught.
    text = template.format(spoken=SPOKEN)
    assert run(text, clinic).reason == "claim"


def test_claimed_kinds_reads_raw_text_in_any_case():
    phrases = {"book": ["you're booked"]}
    assert claimed_kinds("You're BOOKED for Tuesday.", phrases) == ["book"]
    assert claimed_kinds("Shall I check you're booked?", phrases) == []


def test_a_lead_in_with_a_shortened_clinic_number_is_corrected_and_kept(clinic):
    decision = lead_in("You can also call the clinic on 910 000 1.", clinic)
    assert decision.text == f"You can also call the clinic on {clinic.clinic_line}."
    assert not decision.replaced
    assert decision.reason == "phone"


# Stage 5: handbook text the model repeats may pass the checker's advice flag.
AFTERCARE = (
    "If you had a local anaesthetic, your lip and tongue may feel numb for two to three hours. "
    "Take care not to bite them, and wait until the numbness has worn off before eating or "
    "drinking anything hot."
)
REPEATED = (
    "If you had a local anaesthetic, your lip and tongue may stay numb for two to three hours. "
    "Try not to bite them, and wait until the numbness wears off before eating or drinking "
    "anything hot. Is there anything else I can help with?"
)


def test_a_reply_that_repeats_a_search_result_passes_the_checkers_advice_flag(clinic):
    decision = run(REPEATED, clinic, model_advice=True, sources=[AFTERCARE])
    assert not decision.replaced
    assert decision.reason == "sourced"
    assert "two to three hours" in decision.text


def test_a_changed_number_is_not_the_handbook_any_more(clinic):
    changed = REPEATED.replace("two to three hours", "five to six hours")
    assert run(changed, clinic, model_advice=True, sources=[AFTERCARE]).replaced


def test_a_dropped_negation_is_not_the_handbook_any_more(clinic):
    flipped = "Your lip may feel numb for two to three hours, so you can bite and eat hot food."
    assert run(flipped, clinic, model_advice=True, sources=[AFTERCARE]).replaced


def test_an_added_sentence_of_the_models_own_makes_it_advice_again(clinic):
    added = REPEATED + " You should also rinse with warm salt water tonight."
    decision = run(added, clinic, model_advice=True, sources=[AFTERCARE])
    assert decision.replaced
    assert decision.reason == "model_advice"


def test_a_dosing_word_is_never_excused_by_a_source(clinic):
    source = "After a filling you may take 400 mg of ibuprofen."
    decision = run(source, clinic, model_advice=True, sources=[source])
    assert decision.replaced
    assert decision.reason == "dosing"


def test_with_no_search_results_the_advice_flag_replaces_the_reply(clinic):
    assert run(REPEATED, clinic, model_advice=True).replaced


def test_a_sourced_reply_still_has_its_claims_checked(clinic):
    source = "You're booked for a cleaning on Tuesday."
    decision = run(source, clinic, model_advice=True, sources=[source])
    assert decision.replaced
    assert decision.reason == "claim"


def test_a_lead_in_that_repeats_a_search_result_is_kept(clinic):
    sentence = "A cleaning removes the hard deposits that brushing cannot."
    source = "It removes the hard deposits (tartar) that brushing cannot, and polishes the teeth."
    decision = check_lead_in(
        sentence, "cleaning", [], [], clinic, model_advice=True, sources=[source]
    )
    assert decision.text == sentence


@pytest.mark.parametrize(
    "swapped",
    [
        REPEATED.replace("before eating", "after eating"),
        "Numbness usually lasts more than three hours.",
    ],
)
def test_a_changed_order_or_amount_word_is_not_the_handbook_any_more(clinic, swapped):
    source = AFTERCARE + " Numbness usually lasts less than three hours."
    assert run(swapped, clinic, model_advice=True, sources=[source]).replaced


def test_advice_asked_as_a_question_is_not_excused(clinic):
    asked = REPEATED + " Have you tried rinsing with warm salt water tonight?"
    decision = run(asked, clinic, model_advice=True, sources=[AFTERCARE])
    assert decision.replaced
    assert decision.reason == "model_advice"


# Second Stage 5 review: advice appended to an otherwise repeated sentence.
@pytest.mark.parametrize(
    "text",
    [
        (
            "Take care not to bite them, and wait until the numbness has worn off before eating or "
            "drinking anything hot, and hold ice on it."
        ),
        (
            "If you had a local anaesthetic, your lip and tongue may feel numb for two to three "
            "hours, and you can take a hot shower."
        ),
    ],
)
def test_an_instruction_appended_to_handbook_text_is_not_excused(text, clinic):
    assert run(text, clinic, model_advice=True, sources=[AFTERCARE]).replaced


@pytest.mark.parametrize(
    "text",
    [
        # The three wordings the live model used for the aftercare answer.
        (
            "If you had a local anaesthetic, your lip and tongue may stay numb for two to three "
            "hours. Take care not to bite them, and wait until the numbness wears off before eating "
            "or drinking anything hot."
        ),
        (
            "If you had a local anaesthetic, your lip and tongue may stay numb for two to three "
            "hours. Be careful not to bite them, and wait until the numbness wears off before "
            "eating or drinking anything hot."
        ),
        "Your lip and tongue may feel numb for two to three hours after a local anaesthetic.",
        # Wordings from the scenario runs on 2026-11-09, refused before "avoid" counted as "not",
        # contractions were split off, and words were compared by stem.
        (
            "If you had a local anaesthetic, your lip and tongue may stay numb for two to three "
            "hours. Please avoid biting them and wait until the numbness wears off before eating "
            "or drinking anything hot."
        ),
        (
            "If you had a local anaesthetic, your lip and tongue may stay numb for two to three "
            "hours. Please be careful not to bite them while they're numb."
        ),
    ],
)
def test_paraphrases_of_the_handbook_are_still_excused(text, clinic):
    assert not run(text, clinic, model_advice=True, sources=[AFTERCARE]).replaced


# The live aftercare answer from a voice call on 2026-11-09, which the checker flagged as advice
# (0.72) and the exemption wrongly refused: it credits the handbook, takes "after a filling" from
# the piece's heading, and joins two fixed lines into one sentence.
FILLING_PIECE = (
    "After your visit › After a filling "
    + AFTERCARE
    + " Tooth-coloured fillings are fully set when you leave the clinic."
)
LIVE_AFTERCARE = (
    "The handbook says your lip and tongue may feel numb for two to three hours after a filling "
    "with local anaesthetic, so take care not to bite them. For anything beyond that, the "
    "dentist can advise at your visit, or you can call the clinic on {line}."
)


def test_the_live_aftercare_answer_is_excused(clinic):
    text = LIVE_AFTERCARE.format(line=clinic.clinic_line)
    decision = run(text, clinic, model_advice=True, sources=[FILLING_PIECE])
    assert not decision.replaced
    assert decision.reason == "sourced"


@pytest.mark.parametrize(
    "credit",
    [
        "The handbook says",
        "According to the handbook,",
        # Wordings the live model used in the scenario runs.
        "According to our handbook,",
        "According to the clinic's handbook,",
        "According to the clinic handbook,",
        "The clinic's handbook says that",
    ],
)
def test_crediting_the_handbook_is_not_the_models_own_words(credit, clinic):
    text = f"{credit} your lip and tongue may feel numb for two to three hours."
    assert not run(text, clinic, model_advice=True, sources=[AFTERCARE]).replaced


def test_crediting_the_handbook_does_not_excuse_an_instruction(clinic):
    text = "The handbook says to hold ice on your lip tonight."
    assert run(text, clinic, model_advice=True, sources=[AFTERCARE]).replaced


def test_words_from_the_pieces_heading_count_as_the_handbooks(clinic):
    # graph/nodes.py puts each piece's heading before its text, so this is the source it gets.
    text = (
        "Your lip and tongue may feel numb for two to three hours after a filling, so take care "
        "not to bite them."
    )
    assert not run(text, clinic, model_advice=True, sources=[FILLING_PIECE]).replaced
    # Without the heading, "after" faces the source's "before eating" and reads as a flip.
    assert run(text, clinic, model_advice=True, sources=[AFTERCARE]).replaced


def test_a_sentence_joining_two_fixed_lines_is_excused(clinic):
    text = (
        "For anything beyond that, the dentist can advise you at your visit, or you can call "
        f"the clinic on {clinic.clinic_line}."
    )
    assert not run(text, clinic, model_advice=True, sources=[AFTERCARE]).replaced


@pytest.mark.parametrize(
    "text",
    [
        "For anything beyond that, rinse with warm salt water and call the clinic on {line}.",
        "The dentist can advise you at your visit, and you should hold ice on it tonight.",
    ],
)
def test_fixed_lines_do_not_excuse_an_instruction_beside_them(text, clinic):
    text = text.format(line=clinic.clinic_line)
    assert run(text, clinic, model_advice=True, sources=[AFTERCARE]).replaced


def test_fixed_lines_still_excuse_nothing_without_a_search(clinic):
    text = f"The dentist can advise you at your visit, or call the clinic on {clinic.clinic_line}."
    assert run(text, clinic, model_advice=True).replaced


def test_avoid_counts_as_a_negation_both_ways(clinic):
    # "Avoid biting" keeps the source's "not bite", so dropping it is still a flipped meaning.
    flipped = "Your lip may feel numb for two to three hours, so you can bite and eat hot food."
    assert run(flipped, clinic, model_advice=True, sources=[AFTERCARE]).replaced
    source = "Avoid hot drinks for two hours."
    assert run(
        "Have hot drinks for two hours.", clinic, model_advice=True, sources=[source]
    ).replaced


def test_a_handbook_sentence_split_in_two_is_still_the_handbook(clinic):
    # A live reply split "Take care not to bite them, and wait until the numbness has worn off
    # before eating…" into two sentences, so the second has no "not" of its own.
    text = (
        "The handbook says that if you had a local anaesthetic, your lip and tongue may feel numb "
        "for two to three hours, so take care not to bite them. Wait until the numbness wears "
        "off before eating or drinking anything hot."
    )
    assert not run(text, clinic, model_advice=True, sources=[FILLING_PIECE]).replaced


def test_a_split_does_not_let_a_negation_go_missing(clinic):
    text = "Take care to bite them. Wait until the numbness wears off before eating anything hot."
    assert run(text, clinic, model_advice=True, sources=[AFTERCARE]).replaced
