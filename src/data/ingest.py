"""Algoseek US futures ingestion (TAQ v2 + multiple-depth) → polars DataFrames.

Read-only streaming from the Expansion drive. No writes to HDD. Callers are
responsible for materializing outputs to local parquet if needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl

ALGOSEEK_ROOT = Path("/Volumes/Expansion/algoseek_data")
TAQ_TEMPLATE = "us-futures-taq-v2-{year}"
DEPTH_TEMPLATE = "us-futures-muliple-depth-{year}"

TRADE_TYPES = {"TRADE", "TRADE AGRESSOR ON BUY", "TRADE AGRESSOR ON SELL"}
QUOTE_TYPES = {"QUOTE BID", "QUOTE SELL"}


@dataclass(frozen=True)
class ContractFile:
    dataset: str  # "taq" or "depth"
    root: str  # contract root, e.g., "ES"
    expiry: str  # expiry code, e.g., "ESM4"
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
    algoseek_root: Path = ALGOSEEK_ROOT,
) -> ContractFile:
    """Return the ContractFile descriptor for a given (dataset, root, expiry, day).

    dataset must be "taq" or "depth".
    """
    if dataset == "taq":
        template = TAQ_TEMPLATE
    elif dataset == "depth":
        template = DEPTH_TEMPLATE
    else:
        raise ValueError(f"dataset must be 'taq' or 'depth', got {dataset!r}")
    year_dir = algoseek_root / template.format(year=day.year)
    path = year_dir / _day_folder(day) / root / f"{expiry}.csv.gz"
    return ContractFile(dataset=dataset, root=root, expiry=expiry, day=day, path=path)


def _parse_ts(df: pl.DataFrame, date_col: str = "UTCDate", time_col: str = "UTCTime") -> pl.DataFrame:
    """Build a UTC nanosecond timestamp column `ts` from Algoseek YYYYMMDD + nanoseconds-of-day.

    Algoseek TAQ UTCTime is 15 chars: HHMMSSnnnnnnnnn.
    Depth UTCTime is 9 chars: microseconds-of-day-ish (narrower precision).
    We right-pad to 15 and pull fields positionally.
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
    """Read a single Algoseek TAQ file. Returns polars DataFrame with parsed ts."""
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
    """Read a single Algoseek multiple-depth file. Parsed ts; rows are per-side (B/S) L1-L10 book snapshots."""
    if cf.dataset != "depth":
        raise ValueError("read_depth expects a depth ContractFile")
    if not cf.exists:
        raise FileNotFoundError(cf.path)
    df = pl.read_csv(cf.path)
    return _parse_ts(df)


def split_trades_quotes(taq: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split a raw TAQ frame into (trades, quotes), signed for trades via Algoseek aggressor flag.

    Trades carry `aggressor_sign`: +1 for TRADE AGRESSOR ON BUY, -1 for AGRESSOR ON SELL,
    0 for plain TRADE (no aggressor classification).

    Quotes carry `side`: 'bid' for QUOTE BID, 'ask' for QUOTE SELL.
    """
    trades = (
        taq.filter(pl.col("Type").is_in(list(TRADE_TYPES)))
        .with_columns(
            aggressor_sign=pl.when(pl.col("Type") == "TRADE AGRESSOR ON BUY")
            .then(1)
            .when(pl.col("Type") == "TRADE AGRESSOR ON SELL")
            .then(-1)
            .otherwise(0)
            .cast(pl.Int8)
        )
        .select(["ts", "Price", "Quantity", "aggressor_sign"])
        .rename({"Price": "price", "Quantity": "quantity"})
    )

    quotes = (
        taq.filter(pl.col("Type").is_in(list(QUOTE_TYPES)))
        .with_columns(
            side=pl.when(pl.col("Type") == "QUOTE BID").then(pl.lit("bid")).otherwise(pl.lit("ask"))
        )
        .select(["ts", "side", "Price", "Quantity", "Orders"])
        .rename({"Price": "price", "Quantity": "size", "Orders": "orders"})
    )

    return trades, quotes
