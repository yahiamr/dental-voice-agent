"""One formula per metric. Stage: 4 and 5.

What this file is:
    Turns stored runs into the numbers reported in the README.

What it will contain:
    - Task completion: passing runs per scenario.
    - Guard rail rejections: per call, listed by guard.
    - Retrieval hit rate: expected source in the top 3.
    - Delay per turn: median and slowest, split by tool and no-tool turns.

How to understand it:
    Each metric is read from trace files, so numbers can be recomputed without another run.

Used by:
    evals/evaluate.py, tests/test_metrics.py
"""
