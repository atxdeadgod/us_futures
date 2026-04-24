"""Continuous front-month contract resolution for Algoseek futures data.

Front-month for a given (date, root) is defined operationally as the file with
the LARGEST bytesize in the day's folder. This matches CME's lead-contract flow:
the most-active contract is the most-liquid one, which Algoseek writes the most
events for.

Rules of the road:
  - Empty files (<1000 bytes) are rejected (placeholders with no real data)
  - If no valid file exists, return None (likely a market holiday)
  - Caller typically asks for 'taq' dataset since that's the highest-resolution
    signal for liquidity ranking; depth files follow the same roll logic
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from .ingest import day_dir

MIN_REAL_FILE_BYTES = 1000


@dataclass(frozen=True)
class FrontContract:
    day: date
    root: str
    expiry: str  # e.g., "ESH4"
    bytes: int

    @property
    def is_empty(self) -> bool:
        return self.bytes < MIN_REAL_FILE_BYTES


def front_month(
    root: str,
    day: date,
    dataset: str = "taq",
    algoseek_root: Path | str | None = None,
) -> FrontContract | None:
    """Return the front-month FrontContract for (root, day) under `dataset`.

    Returns None if the day's folder doesn't exist or has no files > MIN_REAL_FILE_BYTES.
    """
    folder = day_dir(dataset, root, day, algoseek_root)
    if not folder.exists():
        return None
    candidates = []
    for f in folder.glob("*.csv.gz"):
        try:
            sz = f.stat().st_size
        except OSError:
            continue
        if sz >= MIN_REAL_FILE_BYTES:
            candidates.append((f.stem.replace(".csv", ""), sz))
    if not candidates:
        return None
    candidates.sort(key=lambda kv: kv[1], reverse=True)
    expiry, sz = candidates[0]
    return FrontContract(day=day, root=root, expiry=expiry, bytes=sz)


def iter_front_series(
    root: str,
    start: date,
    end: date,
    dataset: str = "taq",
    algoseek_root: Path | str | None = None,
    weekdays_only: bool = True,
):
    """Yield FrontContract for each trading day in [start, end], skipping days with no data.

    When the front-month expiry changes between consecutive days, that's a ROLL event.
    Callers that need bar continuity should back-adjust prices across the roll.
    """
    d = start
    while d <= end:
        if weekdays_only and d.weekday() >= 5:
            d += timedelta(days=1)
            continue
        fc = front_month(root, d, dataset=dataset, algoseek_root=algoseek_root)
        if fc is not None:
            yield fc
        d += timedelta(days=1)


def detect_rolls(series: list[FrontContract]) -> list[tuple[date, str, str]]:
    """Given an ordered list of daily FrontContracts, return list of roll events.

    Each roll event is (date, from_expiry, to_expiry) — the first day the new expiry
    took over.
    """
    events = []
    prev_expiry: str | None = None
    for fc in series:
        if prev_expiry is not None and fc.expiry != prev_expiry:
            events.append((fc.day, prev_expiry, fc.expiry))
        prev_expiry = fc.expiry
    return events
