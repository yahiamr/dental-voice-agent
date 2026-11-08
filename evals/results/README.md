# Evaluation results

This folder holds the saved evaluation results and the coverage report. It is in Git, so every
commit and tag keeps the scores it had when it was measured. The raw files behind each result (full
conversations, traces, calendars and raw model answers) stay in `runs/`, which Git ignores, and
can be made again at any time by rerunning the commands below.

| File | What it holds | Made by |
| --- | --- | --- |
| `scenarios.md` | Whole calls on the real model: one row per scenario, with passing runs and any failures | `evals/evaluate.py` |
| `scenarios.json` | The same numbers in a form a script can compare. The next full run is compared with it | `evals/evaluate.py` |
| `intent.md` | The golden set: how well each layer reads meaning in single messages | `evals/intent_eval.py --save` |
| `retrieval.md` | Document search: how often keyword, vector and hybrid search find the piece that answers a labelled question | `evals/retrieval_eval.py --save` |
| `coverage.md` | Test coverage: the guards and booking against the design's 85% target, then every file | `pytest --cov` and `coverage report` |
| `scenarios-<provider>-<model>.md` and `.json` | A full run on another model, or with `-no-checker` at the end, with the checker off. Each setting keeps its own files, so it never replaces the main result | `evals/evaluate.py --provider ... --model ...` |

## The golden set and the scenarios

They test different layers, and both are needed.

- **The golden set** (`intent.md`) checks the parts. Each case is one message with a label written
  by a person: is "my lips are turning blue" an emergency, does "You're all set for Tuesday" claim
  a booking, is "yeah go ahead but Thursday" a plain yes. It measures how often each layer is
  right, over 120 cases.
- **The scenarios** (`scenarios.md`) check the whole bot. Each scenario is a scripted call through
  the same session, graph, guards and calendar as a live call, and passes only if the call ends the
  right way: what was written to the calendar, what was said, and what was stored.

A part can be right while the call still fails, for example when a tool round runs out. And a
whole call that passes three times says little about rare wordings. So each covers what the other
cannot.

The rules decided by code alone, such as the yes rule, the notice period and double booking, are
not measured here. The tests in `tests/` prove them exactly, every time, with no API key:
`uv run pytest`.

## Reading `scenarios.md`

The header names the model, whether the checker was on, the commit, and when it ran. A commit
ending in `-dirty` means the code had changes not yet committed, which is normal: a result is
measured before the commit that saves it.

Each scenario runs 3 times, every time on a fresh calendar with the clock fixed at Monday 9
November 2026, 10:00, so the same free times are always on offer. A scenario passes when at least
2 of its 3 runs pass, because a real model varies a little between runs.

| Column | Meaning |
| --- | --- |
| Scenario | Its name in `evals/scenarios.yaml`, where its lines and checks are written |
| Stage | The stage that added it: 3 for reschedule and cancel, 4 for the core journeys, 5 for handbook questions |
| Passing runs | How many of its runs met every check |
| Result | `pass` or `FAIL`, by the 2-of-3 rule |
| Failures | Every reason a run failed, such as "actions were [], expected ['book']" or "the caller's lines ran out" |

A run fails when its script runs out before the call ends. That usually means the bot asked
something the script did not expect, such as its own "Would you like me to book that?".

**Compared with the baseline** lists any scenario that changed between pass and fail since the
saved result: a regression, a fix, or a scenario added or removed. A change such as 3 of 3 to 2 of
3 is not listed, because only the pass rule decides.

**Guard rejections** count how often each guard stepped in, over every run. In a clean run these
come from the scenarios built to trigger them: `write_to_calendar` refuses the commit in the
slot-taken scenario, and `check_emergency` stops the urgent call. `check_confirmation` is left out,
because it logs a rejection on every turn that is not a yes.

`scenarios.json` holds the same numbers: `model`, `checker`, `commit`, `date`, `clock`, `runs`,
`pass_needed`, `passed_scenarios`, `total_scenarios`, `guard_rejections`, and under `scenarios`,
for each name, `stage`, `passed`, `runs`, `passed_scenario` and `failures`.

## Reading `intent.md`

There is one section per layer measured:

- `fixed`: the phrase lists and rules alone.
- `jev:...`: the Jev checker.
- `anthropic:...`: the Haiku checker.

Each checker's section has three rows:

| Row | Meaning |
| --- | --- |
| `fixed` | The rules alone, repeated for comparison |
| `model` | The checker alone |
| `product` | What the bot really does, combining the two by its own rules: for an emergency, Jev decides alone and the Haiku checker raises it with either layer; for a claim or advice, either layer flags it |

For each check:

| Column | Meaning |
| --- | --- |
| Caught | Cases labelled true that were flagged |
| Missed | Cases labelled true that were not. For an emergency, this is the number that matters most |
| False flags | Cases labelled false that were flagged anyway |
| Missed ids, False flag ids | The case ids, to look up in `evals/intent_cases.yaml` |

The **yes** check is reported as right and wrong, because both mistakes matter: a missed yes costs
the caller a turn, and a false yes could write a booking. Only the fixed yes rule can write, so
the "Combined" lines show what would be written.

The id prefixes are `e` for emergency cases, `n` for non-emergencies, `r` for replies, `y` for
plain yeses and `x` for replies that are not a plain yes.

## Reading `retrieval.md`

Each labelled question in `evals/retrieval_questions.yaml` is a caller's own words with the
handbook pieces that answer it. A hit means one of those pieces is among the 3 the model would
see.

| Column | Meaning |
| --- | --- |
| Hits at 3, Hit rate | Questions with an answering piece in the top 3 |
| MRR at 5 | Mean reciprocal rank: 1 when the answer is always first, 0.5 when it is usually second |
| direct, reworded, lookalike, odd_place | The same hits by kind of question: sharing words, sharing none, sharing words with a wrong piece, or answered somewhere unexpected |

**Misses** lists the questions each search got wrong. **Unanswerable questions** shows what
hybrid search returned for questions the handbook does not answer: it always returns its closest
piece, so whether the bot admits it does not know is checked by the scenarios, not here.

## Making the results again

Both need the keys in `.env`. They make real API calls: a full scenario run is 63 calls (21
scenarios, 3 runs each) of a few model turns each, and takes a minute or two.

Scenarios, all of them, 3 runs each. This replaces `scenarios.md` and `scenarios.json`:

```bash
uv run python -m evals.evaluate
```

Useful options:

- `--scenario NAME` (repeat for several) or `--runs 1`: a quick check. A run that is not full
  saves to `runs/results.md` instead, so it never replaces the saved result here.
- `--provider openrouter --model <id>`: the same scenarios on another model. It saves to its own
  `scenarios-openrouter-<id>` files.
- `--baseline FILE.json`: compare with another saved result instead of the one being replaced.
- `--workers N`: runs at the same time, 4 by default.
- `--voice-traces DIR`: add delay per turn, from the traces of real voice calls.

Each run's conversation, trace and calendar are kept in `runs/scenarios/<time>/`, to look at a
failure in full.

The golden set, for the fixed rules and both checkers. This replaces `intent.md`:

```bash
uv run python -m evals.intent_eval --backend fixed --backend jev:typesafe/jev-1.14 --backend anthropic:claude-haiku-5-5 --save evals/results/intent.md
```

Document search, which needs no API key, only the embedding model. This replaces `retrieval.md`:

```bash
uv run python -m evals.retrieval_eval --save evals/results/retrieval.md
```

Test coverage, which needs no key. The first command fails if the guards and booking fall
below 85%. The second writes the per-file table that `coverage.md` holds:

```bash
uv run pytest --cov=dental_agent.guards --cov=dental_agent.booking --cov-branch --cov-fail-under=85
```

```bash
uv run pytest -q --cov=dental_agent --cov-branch --cov-report= && uv run coverage report --format=markdown --skip-empty
```

The held-out sets, `evals/intent_holdout.yaml` and `evals/intent_holdout_2.yaml`, are not part of
this page. They hold cases nobody looked at while the checker was tuned, so they are run only to
re-measure after the checker changes, with `--cases`, and are never used to change a prompt.

## Comparing over time

Because these files are in Git, any two points can be compared. The results were first saved
after `stage-3-reschedule-cancel`, so the examples use later points; `stage-4-submittable` exists
once Stage 4 is tagged.

```bash
git log --oneline -- evals/results/
```

```bash
git show stage-4-submittable:evals/results/scenarios.md
```

```bash
git diff stage-4-submittable main -- evals/results/
```

After changing the prompt, a tool or a guard, rerun the scenarios before committing. The
comparison at the end of the page shows at once whether anything that passed now fails.
