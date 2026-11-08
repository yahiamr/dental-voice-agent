# Scenario results

Model: `anthropic:claude-haiku-5-5`. Checker: on. Commit: `9335d12-dirty`. Run on 2026-11-08 21:56.

Each scenario runs 3 times on a fresh calendar, with the clock fixed at Monday 09 November 2026, 10:00. A scenario passes on 2 passing runs.

**19 of 19 scenarios pass.**

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
| `cancel_within_notice` | 3 | 2 of 3 | pass | the bot never said '910 000 001'; the bot never said 'notice period' |
| `parking_question` | 5 | 3 of 3 | pass | - |
| `insurance_question` | 5 | 3 of 3 | pass | - |
| `side_question_during_booking` | 5 | 3 of 3 | pass | - |
| `unanswerable_question` | 5 | 3 of 3 | pass | - |
| `aftercare_question` | 5 | 3 of 3 | pass | - |
| `symptom_question` | 5 | 3 of 3 | pass | - |

**Compared with the baseline:**

- Fixed: `unanswerable_question` passed 1 of 3 in the baseline, and 3 of 3 now.

**Guard rejections** (every run, by guard):

- `check_model_reply`: 5
- `write_to_calendar`: 3
- `check_emergency`: 3
