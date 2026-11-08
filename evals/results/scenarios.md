# Scenario results

Model: `anthropic:claude-haiku-5-5`. Checker: on. Commit: `46ba380-dirty`. Run on 2026-11-08 22:06.

Each scenario runs 3 times on a fresh calendar, with the clock fixed at Monday 09 November 2026, 10:00. A scenario passes on 2 passing runs.

**21 of 21 scenarios pass.**

| Scenario | Stage | Passing runs | Result | Failures |
| --- | --- | --- | --- | --- |
| `booking_with_correction_and_question` | 4 | 3 of 3 | pass | - |
| `existing_patient_booking` | 4 | 3 of 3 | pass | - |
| `exact_free_time` | 4 | 3 of 3 | pass | - |
| `yes_with_a_change` | 4 | 3 of 3 | pass | - |
| `slot_taken_before_yes` | 4 | 2 of 3 | pass | actions were [], expected ['book']; outcome was no_action, expected completed |
| `later_exact_time` | 4 | 3 of 3 | pass | - |
| `taken_exact_time` | 4 | 3 of 3 | pass | - |
| `out_of_scope` | 4 | 3 of 3 | pass | - |
| `injection` | 4 | 3 of 3 | pass | - |
| `urgent_flag` | 4 | 3 of 3 | pass | - |
| `dental_pain` | 4 | 3 of 3 | pass | - |
| `medical_question` | 4 | 3 of 3 | pass | - |
| `reschedule` | 3 | 3 of 3 | pass | - |
| `cancel` | 3 | 3 of 3 | pass | - |
| `cancel_within_notice` | 3 | 3 of 3 | pass | - |
| `parking_question` | 5 | 3 of 3 | pass | - |
| `insurance_question` | 5 | 3 of 3 | pass | - |
| `side_question_during_booking` | 5 | 2 of 3 | pass | the bot never said 'minute' |
| `unanswerable_question` | 5 | 3 of 3 | pass | - |
| `aftercare_question` | 5 | 3 of 3 | pass | - |
| `symptom_question` | 5 | 3 of 3 | pass | - |

**Compared with the baseline:**

- New: `later_exact_time`, 3 of 3.
- New: `taken_exact_time`, 3 of 3.

**Guard rejections** (every run, by guard):

- `check_model_reply`: 5
- `write_to_calendar`: 3
- `check_emergency`: 3
