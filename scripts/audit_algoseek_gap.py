"""Audit local-vs-S3 Algoseek coverage per (date, symbol_root) for futures TAQ.

For each trading day in the requested range and each symbol root requested,
list the contract files on S3 and locally, sum bytes, report the gap.

Output: CSV to stdout (date,root,s3_files,s3_bytes,local_files,local_bytes,missing_bytes).
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

S3_BUCKET = "s3://plt-de-dev/lake/US/raw/Algoseek/s3"
LOCAL_ROOT = Path("/Volumes/Expansion/algoseek_data")
DATASET_TAQ = "us-futures-taq-v2"
PROFILE = "plt-de-dev"


def _ls_s3(dataset: str, day: date, root: str) -> list[tuple[str, int]]:
    """Return [(filename, bytes), ...] for contracts in s3 path. Empty list if path missing."""
    prefix = f"{S3_BUCKET}/{dataset}-{day.year}/{day:%Y%m%d}/{root}/"
    try:
        out = subprocess.run(
            ["aws", "s3", "ls", prefix, "--profile", PROFILE],
            capture_output=True, text=True, check=False, timeout=30,
        )
    except subprocess.TimeoutExpired:
        return []
    files = []
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[-1].endswith(".csv.gz"):
            try:
                files.append((parts[-1], int(parts[2])))
            except ValueError:
                pass
    return files


def _ls_local(dataset: str, day: date, root: str) -> list[tuple[str, int]]:
    folder = LOCAL_ROOT / f"{dataset}-{day.year}" / f"{day:%Y%m%d}" / root
    if not folder.exists():
        return []
    return [(p.name, p.stat().st_size) for p in folder.glob("*.csv.gz")]


def audit_day_root(dataset: str, day: date, root: str) -> dict:
    s3_files = _ls_s3(dataset, day, root)
    local_files = _ls_local(dataset, day, root)
    s3_by = {f: b for f, b in s3_files}
    loc_by = {f: b for f, b in local_files}
    s3_bytes = sum(s3_by.values())
    loc_bytes = sum(loc_by.values())
    # Missing = S3 has bytes that local does not (per file: max(0, s3 - local))
    missing = sum(max(0, s3_by.get(f, 0) - loc_by.get(f, 0)) for f in set(s3_by) | set(loc_by))
    return {
        "date": day.isoformat(),
        "root": root,
        "s3_files": len(s3_files),
        "s3_bytes": s3_bytes,
        "local_files": len(local_files),
        "local_bytes": loc_bytes,
        "missing_bytes": missing,
    }


def daterange(start: date, end: date, weekdays_only: bool = True):
    d = start
    while d <= end:
        if not weekdays_only or d.weekday() < 5:
            yield d
        d += timedelta(days=1)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--roots", default="ES", help="comma-separated symbol roots")
    p.add_argument("--dataset", default=DATASET_TAQ, choices=[DATASET_TAQ, "us-futures-muliple-depth"])
    p.add_argument("--threads", type=int, default=24)
    args = p.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    roots = [r.strip() for r in args.roots.split(",") if r.strip()]
    days = list(daterange(start, end, weekdays_only=True))

    tasks = [(d, r) for d in days for r in roots]
    print(f"# auditing {len(tasks)} (day, root) pairs across {len(roots)} roots, {len(days)} days; {args.threads} threads", file=sys.stderr)

    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=["date", "root", "s3_files", "s3_bytes", "local_files", "local_bytes", "missing_bytes"],
    )
    writer.writeheader()

    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futures = {ex.submit(audit_day_root, args.dataset, d, r): (d, r) for d, r in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            writer.writerow(row)
            if i % 50 == 0:
                print(f"# progress: {i}/{len(tasks)}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
