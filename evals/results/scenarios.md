# Scenario results

Model: `anthropic:claude-haiku-5-5`. Checker: on. Commit: `55c4556-dirty`. Run on 2026-11-08 21:46.

Each scenario runs 3 times on a fresh calendar, with the clock fixed at Monday 09 November 2026, 10:00. A scenario passes on 2 passing runs.

**18 of 19 scenarios pass.**

| Scenario | Stage | Passing runs | Result | Failures |
| --- | --- | --- | --- | --- |
| `booking_with_correction_and_question` | 4 | 3 of 3 | pass | - |
| `existing_patient_booking` | 4 | 3 of 3 | pass | - |
| `exact_free_time` | 4 | 3 of 3 | pass | - |
| `yes_with_a_change` | 4 | 3 of 3 | pass | - |
| `slot_taken_before_yes` | 4 | 3 of 3 | pass | - |
| `out_of_scope` | 4 | 3 of 3 | pass | - |
| `injection` | 4 | 3 of 3 | pass | - |
| `urgent_flag` | 4 | 3 of 3 | pass | - |
| `dental_pain` | 4 | 3 of 3 | pass | - |
| `medical_question` | 4 | 3 of 3 | pass | - |
| `reschedule` | 3 | 3 of 3 | pass | - |
| `cancel` | 3 | 3 of 3 | pass | - |
| `cancel_within_notice` | 3 | 3 of 3 | pass | - |
| `parking_question` | 5 | 3 of 3 | pass | - |
| `insurance_question` | 5 | 2 of 3 | pass | the caller's lines ran out |
| `side_question_during_booking` | 5 | 3 of 3 | pass | - |
| `unanswerable_question` | 5 | 3 of 3 | pass | - |
| `aftercare_question` | 5 | 1 of 3 | FAIL | the bot never said 'hours'; the caller's lines ran out |
| `symptom_question` | 5 | 3 of 3 | pass | - |

**Compared with the baseline:**

- Regression: `aftercare_question` passed 3 of 3 in the baseline, and 1 of 3 now.

**Guard rejections** (every run, by guard):

- `check_model_reply`: 5
- `write_to_calendar`: 3
- `check_emergency`: 3
