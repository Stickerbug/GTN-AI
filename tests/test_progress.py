from __future__ import annotations

import io

from gtn_ai.progress import ProgressReporter


def test_progress_suppresses_identical_final_summary() -> None:
    stream = io.StringIO()
    progress = ProgressReporter("work", total=2, interval=60.0, stream=stream)

    progress.update(2, force=True, games=4)
    progress.finish(2, games=4)

    lines = stream.getvalue().splitlines()
    assert len(lines) == 1
    assert "100.0%" in lines[0]


def test_progress_keeps_distinct_final_summary() -> None:
    stream = io.StringIO()
    progress = ProgressReporter("work", total=2, interval=60.0, stream=stream)

    progress.update(2, force=True, loss="last-batch")
    progress.finish(2, loss="epoch-average")

    lines = stream.getvalue().splitlines()
    assert len(lines) == 2
    assert "last-batch" in lines[0]
    assert "epoch-average" in lines[1]
