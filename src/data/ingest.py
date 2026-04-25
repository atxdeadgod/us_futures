"""Algoseek US futures ingestion (TAQ v2 + multiple-depth) → polars DataFrames.

Path layout (default = HPC sync destination):
    $ALGOSEEK_ROOT/{taq,depth,vix}/{root}/{YYYY}/{YYYYMMDD}/{EXPIRY}.csv.gz

Overridable via env var `ALGOSEEK_ROOT`, or explicit `algoseek_root` kwarg.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl

# Default HPC location (where the selective sync writes). Override via $ALGOSEEK_ROOT.
DEFAULT_ALGOSEEK_ROOT = Path(
    os.environ.get(
        "ALGOSEEK_ROOT",
        "/N/project/ksb-finance-backtesting/data/algoseek_futures",
    )
)

TRADE_TYPES = {"TRADE", "TRADE AGRESSOR ON BUY", "TRADE AGRESSOR ON SELL"}
QUOTE_TYPES = {"QUOTE BID", "QUOTE SELL"}

# Algoseek Flag values (Table 5 of the TAQ guide).
FLAG_REGULAR = 0
FLAG_IMPLIED = 1
FLAG_SHFLAG = 2  # session-high marker (Quantity=0)
FLAG_SLFLAG = 4  # session-low marker (Quantity=0)
FLAG_CALCULATED = 8  # CalculatedPrice — doc says exclude from bar aggregation
FLAG_OPENING = 16


@dataclass(frozen=True)
class ContractFile:
    dataset: str  # "taq" or "depth"
    root: str  # contract root, e.g., "ES"
    expiry: str  # expiry code, e.g., "ESH4"
    day: date
    path: Path

    @property
    def exists(self) -> bool:
        return self.path.exists()


def _day_folder(day: date) -> str:
    return day.strftime("%Y%m%d")


def locate(
    dataset: str,
    root: str,
    expiry: str,
    day: date,
    algoseek_root: Path | str | None = None,
) -> ContractFile:
    """Return the ContractFile descriptor under the new HPC layout:

        <root>/<dataset>/<root>/<YYYY>/<YYYYMMDD>/<EXPIRY>.csv.gz

    `dataset` must be in {"taq","depth"}.
    """
    if dataset not in ("taq", "depth"):
        raise ValueError(f"dataset must be 'taq' or 'depth', got {dataset!r}")
    root_dir = Path(algoseek_root) if algoseek_root else DEFAULT_ALGOSEEK_ROOT
    path = root_dir / dataset / root / f"{day.year}" / _day_folder(day) / f"{expiry}.csv.gz"
    return ContractFile(dataset=dataset, root=root, expiry=expiry, day=day, path=path)


def day_dir(
    dataset: str,
    root: str,
    day: date,
    algoseek_root: Path | str | None = None,
) -> Path:
    """Directory holding all expiry files for (dataset, root, day)."""
    if dataset not in ("taq", "depth"):
        raise ValueError(f"dataset must be 'taq' or 'depth', got {dataset!r}")
    root_dir = Path(algoseek_root) if algoseek_root else DEFAULT_ALGOSEEK_ROOT
    return root_dir / dataset / root / f"{day.year}" / _day_folder(day)


def _parse_ts(df: pl.DataFrame, date_col: str = "UTCDate", time_col: str = "UTCTime") -> pl.DataFrame:
    """Build a UTC nanosecond timestamp column `ts` from Algoseek YYYYMMDD + nanoseconds-of-day.

    TAQ UTCTime is 15 chars HHMMSSnnnnnnnnn. We right-pad to 15 and slice positionally.
    """
    return (
        df.with_columns(
            [
                pl.col(date_col).cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d").alias("_date"),
                pl.col(time_col).cast(pl.Utf8).str.zfill(15).alias("_tstr"),
            ]
        )
        .with_columns(
            ts=pl.col("_date").cast(pl.Datetime("ns", "UTC"))
            + pl.duration(
                hours=pl.col("_tstr").str.slice(0, 2).cast(pl.Int64),
                minutes=pl.col("_tstr").str.slice(2, 2).cast(pl.Int64),
                seconds=pl.col("_tstr").str.slice(4, 2).cast(pl.Int64),
                nanoseconds=pl.col("_tstr").str.slice(6, 9).cast(pl.Int64),
            )
        )
        .drop(["_date", "_tstr"])
    )


def read_taq(cf: ContractFile) -> pl.DataFrame:
    """Read a single Algoseek TAQ file. Returns polars DataFrame with parsed `ts`."""
    if cf.dataset != "taq":
        raise ValueError("read_taq expects a taq ContractFile")
    if not cf.exists:
        raise FileNotFoundError(cf.path)
    df = pl.read_csv(
        cf.path,
        schema_overrides={
            "UTCDate": pl.Int64,
            "UTCTime": pl.Utf8,
            "LocalDate": pl.Int64,
            "LocalTime": pl.Utf8,
            "Ticker": pl.Utf8,
            "SecurityID": pl.Int64,
            "TypeMask": pl.Int64,
            "Type": pl.Utf8,
            "Price": pl.Float64,
            "Quantity": pl.Int64,
            "Orders": pl.Int64,
            "Flags": pl.Int64,
        },
    )
    return _parse_ts(df)


def read_depth(cf: ContractFile) -> pl.DataFrame:
    """Read a single Algoseek multiple-depth file. Parsed `ts`; rows are per-side (B/S) L1..L10 snapshots.

    Schema notes: depth files have L1..L10 Price/Size/Orders columns. Levels
    deeper than the active book are often null/zero in the early rows of a
    file, so polars' default schema inference (first ~100 rows) sees integer-
    only data and infers `i64`. Later rows have fractional prices (e.g.
    `4390.250000`) that don't fit i64 → ComputeError mid-parse.

    Fix: explicitly force all L{k}Price columns to Float64 via schema_overrides.
    Sizes and Orders are kept as Int64 (counts/quantities). Other columns are
    left to inference.
    """
    if cf.dataset != "depth":
        raise ValueError("read_depth expects a depth ContractFile")
    if not cf.exists:
        raise FileNotFoundError(cf.path)
    price_cols_float = {f"L{k}Price": pl.Float64 for k in range(1, 11)}
    df = pl.read_csv(cf.path, schema_overrides=price_cols_float)
    return _parse_ts(df)


def split_trades_quotes(taq: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split a raw TAQ frame into (trades, quotes).

    Filters per Algoseek doc:
      - Trades: Type in {TRADE, TRADE AGRESSOR ON BUY, TRADE AGRESSOR ON SELL}
                AND Quantity > 0                  # excludes SH/SL session markers
                AND (Flags & 8) == 0              # excludes CalculatedPrice
      - Quotes: Type in {QUOTE BID, QUOTE SELL}

    Aggressor sign:
      +1 = TRADE AGRESSOR ON BUY  (initiator buying, lifted the ask)
      -1 = TRADE AGRESSOR ON SELL (initiator selling, hit the bid)
       0 = plain TRADE (Algoseek couldn't classify the initiator)
    """
    trades = (
        taq.filter(
            pl.col("Type").is_in(list(TRADE_TYPES))
            & (pl.col("Quantity") > 0)
            & ((pl.col("Flags").cast(pl.Int64) & FLAG_CALCULATED) == 0)
        )
        .with_columns(
            aggressor_sign=pl.when(pl.col("Type") == "TRADE AGRESSOR ON BUY")
            .then(1)
            .when(pl.col("Type") == "TRADE AGRESSOR ON SELL")
            .then(-1)
            .otherwise(0)
            .cast(pl.Int8),
            is_implied=(pl.col("Flags").cast(pl.Int64) & FLAG_IMPLIED).cast(pl.Boolean),
        )
        .select(["ts", "Price", "Quantity", "aggressor_sign", "is_implied"])
        .rename({"Price": "price", "Quantity": "quantity"})
    )

    quotes = (
        taq.filter(pl.col("Type").is_in(list(QUOTE_TYPES)))
        .with_columns(
            side=pl.when(pl.col("Type") == "QUOTE BID").then(pl.lit("bid")).otherwise(pl.lit("ask")),
            is_implied=(pl.col("Flags").cast(pl.Int64) & FLAG_IMPLIED).cast(pl.Boolean),
        )
        .select(["ts", "side", "Price", "Quantity", "Orders", "is_implied"])
        .rename({"Price": "price", "Quantity": "size", "Orders": "orders"})
    )

    return trades, quotes
