# Test coverage

Commit: `46ba380-dirty`. Run on 2026-11-08 22:14. Tests: 918 passed.

Branch coverage, which counts each side of every `if` as well as each line.

| Scope | Coverage | Target |
| --- | --- | --- |
| Guards and booking (`dental_agent.guards`, `dental_agent.booking`), the code that protects a booking | 98.36% | 85%, met |
| The whole app (`dental_agent`) | 97% | none |

The design sets a target only for the guards and booking, because they decide what is written
to the calendar and what a caller is told. The eval scripts in `evals/` are left out: their
entry points call live APIs, and their pure functions are tested in `tests/`.

## By file

| Name                                      |    Stmts |     Miss |   Branch |   BrPart |   Cover |
|------------------------------------------ | -------: | -------: | -------: | -------: | ------: |
| src/dental\_agent/booking/calendar.py     |      166 |        5 |       20 |        2 |     96% |
| src/dental\_agent/booking/seed.py         |       62 |        0 |       18 |        0 |    100% |
| src/dental\_agent/checker/\_\_init\_\_.py |        4 |        0 |        0 |        0 |    100% |
| src/dental\_agent/checker/anthropic.py    |       33 |        0 |        6 |        0 |    100% |
| src/dental\_agent/checker/base.py         |       17 |        0 |        0 |        0 |    100% |
| src/dental\_agent/checker/chain.py        |       46 |        0 |       10 |        0 |    100% |
| src/dental\_agent/checker/jev.py          |       51 |        0 |       14 |        0 |    100% |
| src/dental\_agent/checker/questions.py    |        7 |        0 |        0 |        0 |    100% |
| src/dental\_agent/cli.py                  |      116 |        3 |       28 |        4 |     95% |
| src/dental\_agent/clinic/data.py          |       62 |        1 |        4 |        0 |     98% |
| src/dental\_agent/clinic/embedder.py      |       16 |        1 |        2 |        1 |     89% |
| src/dental\_agent/clinic/index.py         |      110 |        2 |       16 |        0 |     98% |
| src/dental\_agent/clinic/retriever.py     |       89 |        3 |        8 |        1 |     96% |
| src/dental\_agent/clock.py                |       13 |        0 |        0 |        0 |    100% |
| src/dental\_agent/config.py               |       52 |        1 |        4 |        1 |     96% |
| src/dental\_agent/deps.py                 |       33 |        0 |        0 |        0 |    100% |
| src/dental\_agent/factories.py            |      113 |        6 |       24 |        0 |     96% |
| src/dental\_agent/formatting.py           |        5 |        0 |        0 |        0 |    100% |
| src/dental\_agent/graph/build.py          |       26 |        0 |        2 |        0 |    100% |
| src/dental\_agent/graph/nodes.py          |      253 |       15 |       68 |        2 |     95% |
| src/dental\_agent/graph/routing.py        |       23 |        1 |        8 |        1 |     94% |
| src/dental\_agent/guards/reply\_check.py  |      159 |        1 |       48 |        1 |     99% |
| src/dental\_agent/guards/urgent.py        |       25 |        0 |       12 |        0 |    100% |
| src/dental\_agent/guards/validate.py      |       69 |        1 |       34 |        1 |     98% |
| src/dental\_agent/guards/yes\_rule.py     |       39 |        0 |       18 |        0 |    100% |
| src/dental\_agent/models.py               |      134 |        0 |        0 |        0 |    100% |
| src/dental\_agent/prompts/build.py        |       68 |        1 |       24 |        1 |     98% |
| src/dental\_agent/sentences.py            |       30 |        2 |       20 |        2 |     92% |
| src/dental\_agent/session.py              |      265 |       11 |       54 |        3 |     96% |
| src/dental\_agent/speech/audio.py         |      143 |        9 |       30 |        4 |     92% |
| src/dental\_agent/speech/base.py          |       15 |        0 |        0 |        0 |    100% |
| src/dental\_agent/speech/deepgram.py      |      244 |       11 |       60 |        3 |     95% |
| src/dental\_agent/speech/text.py          |       15 |        0 |        2 |        0 |    100% |
| src/dental\_agent/state.py                |       40 |        0 |        0 |        0 |    100% |
| src/dental\_agent/storage/conversation.py |       33 |        0 |        0 |        0 |    100% |
| src/dental\_agent/storage/summary.py      |       37 |        3 |       14 |        2 |     90% |
| src/dental\_agent/storage/trace.py        |       34 |        2 |        4 |        2 |     89% |
| src/dental\_agent/tools/\_\_init\_\_.py   |       12 |        0 |        0 |        0 |    100% |
| src/dental\_agent/tools/action\_tools.py  |       35 |        1 |       10 |        1 |     96% |
| src/dental\_agent/tools/dispatcher.py     |       74 |        0 |       30 |        0 |    100% |
| src/dental\_agent/tools/read\_tools.py    |      111 |        1 |       30 |        1 |     99% |
| src/dental\_agent/tools/schemas.py        |       48 |        1 |        4 |        1 |     96% |
| src/dental\_agent/tools/session\_tools.py |       25 |        0 |        6 |        0 |    100% |
| **TOTAL**                                 | **2952** |   **82** |  **632** |   **34** | **97%** |

9 empty files skipped.
