from __future__ import annotations

import sys
import time
from typing import Any, TextIO


class ProgressReporter:
    """Low-frequency progress output that keeps hot loops effectively untouched."""

    def __init__(
        self,
        label: str,
        *,
        total: int | None = None,
        interval: float = 10.0,
        enabled: bool = True,
        stream: TextIO | None = None,
    ) -> None:
        self.label = str(label)
        self.total = max(0, int(total)) if total is not None else None
        self.interval = max(0.25, float(interval))
        self.enabled = bool(enabled)
        self.stream = stream or sys.stderr
        self.started = time.perf_counter()
        self.last_report = self.started
        self.last_length = 0
        self.last_signature: tuple[int, tuple[tuple[str, str], ...]] | None = None

    def update(
        self,
        completed: int,
        *,
        force: bool = False,
        **fields: Any,
    ) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        done = max(0, int(completed))
        complete = self.total is not None and done >= self.total
        if not force and not complete and now - self.last_report < self.interval:
            return
        self.last_report = now
        elapsed = max(1e-9, now - self.started)
        rate = done / elapsed
        prefix = self._prefix(done)
        details = [f"{rate:.1f}/s", f"elapsed {_duration(elapsed)}"]
        if self.total and rate > 0 and done < self.total:
            details.append(f"eta {_duration((self.total - done) / rate)}")
        details.extend(
            f"{key}={value}" for key, value in fields.items() if value is not None
        )
        signature = (
            done,
            tuple(
                (str(key), str(value))
                for key, value in fields.items()
                if value is not None
            ),
        )
        if force and signature == self.last_signature:
            return
        message = f"{self.label} {prefix} | " + " | ".join(details)
        if self._interactive():
            padding = " " * max(0, self.last_length - len(message))
            print(f"\r{message}{padding}", end="", file=self.stream, flush=True)
            self.last_length = len(message)
            if complete or force and self.total is not None and done >= self.total:
                print(file=self.stream, flush=True)
                self.last_length = 0
        else:
            print(message, file=self.stream, flush=True)
        self.last_signature = signature

    def finish(self, completed: int, **fields: Any) -> None:
        self.update(completed, force=True, **fields)
        if self._interactive() and self.last_length:
            print(file=self.stream, flush=True)
            self.last_length = 0

    def _prefix(self, completed: int) -> str:
        if not self.total:
            return f"{completed:,}"
        ratio = min(1.0, completed / max(1, self.total))
        width = 24
        filled = min(width, int(ratio * width))
        bar = "=" * filled + "." * (width - filled)
        return f"[{bar}] {ratio * 100:5.1f}% ({completed:,}/{self.total:,})"

    def _interactive(self) -> bool:
        try:
            return bool(self.stream.isatty())
        except (AttributeError, OSError):
            return False


def _duration(seconds: float) -> str:
    value = max(0, int(round(seconds)))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
