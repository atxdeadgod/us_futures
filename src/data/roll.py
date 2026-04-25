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


def _list_day_candidates(folder: Path) -> list[tuple[str, int]]:
    candidates: list[tuple[str, int]] = []
    for f in folder.glob("*.csv.gz"):
        try:
            sz = f.stat().st_size
        except OSError:
            continue
        if sz >= MIN_REAL_FILE_BYTES:
            candidates.append((f.stem.replace(".csv", ""), sz))
    candidates.sort(key=lambda kv: kv[1], reverse=True)
    return candidates


def front_month(
    root: str,
    day: date,
    dataset: str = "taq",
    algoseek_root: Path | str | None = None,
) -> FrontContract | None:
    """Return the front-month FrontContract for (root, day) under `dataset`."""
    folder = day_dir(dataset, root, day, algoseek_root)
    if not folder.exists():
        return None
    candidates = _list_day_candidates(folder)
    if not candidates:
        return None
    expiry, sz = candidates[0]
    return FrontContract(day=day, root=root, expiry=expiry, bytes=sz)


_VX_MONTH_CODES = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
    "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12,
}


def parse_vx_expiry(expiry: str, ref_year: int) -> tuple[int, int] | None:
    """Parse 'VX<MonCode><Y>' → (calendar_year, month) using `ref_year` for decade.

    Algoseek uses a single-digit year (VXH4 = Mar 2024 when ref_year is in 2020s).
    The decade is inferred from `ref_year` (the day's calendar year): the contract
    year is in the same decade unless that puts it more than 6 months in the past
    relative to `ref_year`, in which case bump to next decade.
    """
    if not expiry.startswith("VX") or len(expiry) < 4:
        return None
    mon_code = expiry[2]
    if mon_code not in _VX_MONTH_CODES:
        return None
    try:
        last_digit = int(expiry[3])
    except ValueError:
        return None
    decade = (ref_year // 10) * 10
    cand = decade + last_digit
    # If candidate is way in the past, bump to next decade.
    if cand < ref_year - 6:
        cand += 10
    return cand, _VX_MONTH_CODES[mon_code]


def front_n(
    root: str,
    day: date,
    n: int = 3,
    dataset: str = "taq",
    algoseek_root: Path | str | None = None,
) -> list[FrontContract]:
    """Return up to `n` front contracts for (root, day) ordered by ascending expiry.

    For VX (root='VX'/'VIX'-style with monthly cycle), sort by parsed calendar
    expiry so that front-of-curve order is robust to liquidity inversions near
    roll. For other roots, fall back to size-rank ordering (most-active first).
    """
    folder = day_dir(dataset, root, day, algoseek_root)
    if not folder.exists():
        return []
    candidates = _list_day_candidates(folder)
    if not candidates:
        return []

    if all(e.startswith("VX") for e, _ in candidates):
        # Sort by parsed calendar expiry; drop unparsable.
        by_calendar: list[tuple[tuple[int, int], str, int]] = []
        for expiry, sz in candidates:
            parsed = parse_vx_expiry(expiry, day.year)
            if parsed is None:
                continue
            by_calendar.append((parsed, expiry, sz))
        # Filter out contracts whose expiry month is strictly before the day's month
        # in the same year, AND whose front-month is no longer the largest (i.e.,
        # already expired). Algoseek typically keeps the file around until end-of-day
        # of expiry; safe heuristic: only drop if ((exp_year, exp_month) < (day.year, day.month)).
        by_calendar = [
            t for t in by_calendar
            if (t[0][0], t[0][1]) >= (day.year, day.month)
        ]
        by_calendar.sort(key=lambda t: t[0])
        return [FrontContract(day=day, root=root, expiry=e, bytes=s)
                for (_p, e, s) in by_calendar[:n]]

    return [FrontContract(day=day, root=root, expiry=e, bytes=s)
            for e, s in candidates[:n]]


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
