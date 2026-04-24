"""Depth (MBP-10) book-snapshot-at-bar-close enrichment.

Consumes the depth event stream from ingest.read_depth() and attaches the most-
recent-regular bid-side + ask-side L1..L10 snapshot (book state) to each 5-sec
bar-close timestamp.

Output adds 60 columns to the bars frame:
    bid_px_L1..L10, bid_sz_L1..L10, bid_ord_L1..L10   (30)
    ask_px_L1..L10, ask_sz_L1..L10, ask_ord_L1..L10   (30)

Plus a `book_ts_close` column recording the actual timestamp of the most-recent
snapshot (useful for staleness diagnostics).

Policy choices:
  * ONLY_REGULAR = True by default → filter Flags=0, drop implied (Flags=1) snaps.
    Rationale: implied book state derives from spread markets; features that
    assume direct-orderbook semantics (deep OFI, size imbalance) are polluted by
    implied book state. Can opt in by passing only_regular=False if desired.
  * Asof-join uses strict `<` via the engines.asof_strict_backward helper (§8.E)
    — depth events at exactly the bar-close timestamp are NOT used.
"""
from __future__ import annotations

import polars as pl

LEVEL_SOURCE_COLS = [f"L{k}{suf}" for k in range(1, 11) for suf in ("Price", "Size", "Orders")]


def _rename_map(side: str) -> dict[str, str]:
    """Build rename map L{k}Price -> {side}_px_L{k}, etc."""
    assert side in ("bid", "ask")
    m: dict[str, str] = {}
    for k in range(1, 11):
        m[f"L{k}Price"] = f"{side}_px_L{k}"
        m[f"L{k}Size"] = f"{side}_sz_L{k}"
        m[f"L{k}Orders"] = f"{side}_ord_L{k}"
    return m


def split_sides(depth: pl.DataFrame, only_regular: bool = True) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split depth frame into (bid_side, ask_side), each column-renamed to our
    bid_px_L1..ask_ord_L10 convention + sorted by ts.

    If `only_regular`, keeps only Flags=0 (regular book) rows. Implied book
    snapshots (Flags=1) are noisy for our features and are filtered by default.
    """
    df = depth
    if only_regular and "Flags" in df.columns:
        df = df.filter(pl.col("Flags") == 0)

    have_cols = [c for c in LEVEL_SOURCE_COLS if c in df.columns]
    bid = (
        df.filter(pl.col("Side") == "B")
        .select(["ts"] + have_cols)
        .rename({c: _rename_map("bid")[c] for c in have_cols})
        .sort("ts")
    )
    ask = (
        df.filter(pl.col("Side") == "S")
        .select(["ts"] + have_cols)
        .rename({c: _rename_map("ask")[c] for c in have_cols})
        .sort("ts")
    )
    return bid, ask


def _asof_strict_backward(
    left: pl.DataFrame,
    right: pl.DataFrame,
    on: str,
    carry_right_ts_as: str | None = None,
) -> pl.DataFrame:
    """Asof-join right onto left with strict `<` predicate (no exact-match leak).

    Shifts right's join key by +1 ns, then standard backward join allows exact
    equality on left.ts to still match the shifted value — net effect: a right
    event at time T matches a left row with left.ts >= T+1ns, i.e. left.ts > T.
    """
    right_work = right.with_columns(
        (pl.col(on) + pl.duration(nanoseconds=1)).alias(on)
    )
    if carry_right_ts_as is not None:
        # Also carry the ORIGINAL (pre-shift) right timestamp for staleness diagnostics.
        right_work = right_work.with_columns(
            (pl.col(on) - pl.duration(nanoseconds=1)).alias(carry_right_ts_as)
        )
    return left.sort(on).join_asof(right_work, on=on, strategy="backward")


def attach_book_snapshot(
    bars: pl.DataFrame,
    depth: pl.DataFrame,
    only_regular: bool = True,
) -> pl.DataFrame:
    """Attach the latest regular L1..L10 book snapshot (per side) to each bar row.

    Adds 60 level columns + `book_ts_close` (actual ts of the snapshot).
    Bar rows whose bar-close precedes the first depth event keep nulls.
    """
    bid, ask = split_sides(depth, only_regular=only_regular)

    # Attach bid side (and record the original bid-snap timestamp as book_ts_close).
    out = _asof_strict_backward(bars, bid, on="ts", carry_right_ts_as="book_ts_close_bid")
    # Attach ask side (no separate carry — book_ts_close will reflect the
    # MORE RECENT of the two sides; computed below)
    out = _asof_strict_backward(out, ask, on="ts", carry_right_ts_as="book_ts_close_ask")

    out = out.with_columns(
        pl.max_horizontal(
            pl.col("book_ts_close_bid"), pl.col("book_ts_close_ask")
        ).alias("book_ts_close")
    ).drop(["book_ts_close_bid", "book_ts_close_ask"])

    return out
