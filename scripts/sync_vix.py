"""Focused VIX-only sync from the plt-de-dev mirror.

We sync from the private mirror at:
    s3://plt-de-dev/lake/US/raw/Algoseek/s3/vix-futures/taq/{YYYY}/{YYYYMMDD}/{VXxN}.csv.gz

NOT the original `s3://vix-futures/` bucket — that one returns 403 Forbidden
on `GetObject` with our credentials (list works, get doesn't).

Storage-class handling: many of the larger (front-month) files in the mirror
are in S3 GLACIER storage (lifecycle-archived). `aws s3 cp` skips those with
a clear warning; we count them as glacier-skip and log the keys so a later
restore-then-resync run can pick them up.

Layout:
    s3://plt-de-dev/lake/US/raw/Algoseek/s3/vix-futures/taq/{YYYY}/{YYYYMMDD}/{VXxN}.csv.gz
    →
    /N/project/.../algoseek_futures/vix/{YYYY}/{YYYYMMDD}/{VXxN}.csv.gz

Idempotent: skip if dest exists with matching size.

Usage:
    python scripts/sync_vix.py --year 2024
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

DEST_ROOT = Path("/N/project/ksb-finance-backtesting/data/algoseek_futures/vix")
PROFILE = "plt-de-dev"
BUCKET = "plt-de-dev"
KEY_PREFIX = "lake/US/raw/Algoseek/s3/vix-futures/taq"
MIN_FILE_BYTES = 200


def daterange(start: date, end: date):
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)


def s3_ls(prefix: str) -> list[tuple[str, int]]:
    """List objects under s3://{BUCKET}/{prefix}. Returns (key, size) pairs.

    Note: the mirror is not requester-pays (it's our own bucket), so no
    --request-payer flag.
    """
    cmd = [
        "aws", "s3api", "list-objects-v2",
        "--bucket", BUCKET, "--prefix", prefix,
        "--profile", PROFILE,
        "--query", "Contents[].[Key,Size]", "--output", "text",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=120)
    if out.returncode != 0:
        return []
    files = []
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            try:
                files.append((parts[0], int(parts[1])))
            except ValueError:
                pass
    return files


GLACIER_MARKER = "glacier-skip"


def s3_cp(key: str, dest: Path) -> tuple[bool, str]:
    """Download s3://{BUCKET}/{key} → dest.

    Uses `aws s3 cp` which gracefully handles GLACIER-archived objects by
    skipping them with a warning (not an error). Returns the special string
    `glacier-skip` in the failure-message field when the file is in Glacier
    so the caller can count it separately from real failures.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "aws", "s3", "cp",
        f"s3://{BUCKET}/{key}", str(dest),
        "--profile", PROFILE, "--only-show-errors",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=600)
    combined = (out.stdout + "\n" + out.stderr).lower()
    if "storage class glacier" in combined or "object's storage class" in combined:
        return False, GLACIER_MARKER
    if out.returncode != 0:
        return False, out.stderr.strip()[:300]
    if not dest.exists() or dest.stat().st_size == 0:
        # `aws s3 cp` may print warnings (e.g., glacier skip) and still exit 0;
        # if no file landed, treat as failure.
        return False, "cp exit 0 but no file landed (likely glacier-skip)"
    return True, ""


def process_day(day: date, log: logging.Logger) -> dict:
    prefix = f"{KEY_PREFIX}/{day.year}/{day:%Y%m%d}/"
    files = s3_ls(prefix)
    if not files:
        return {"day": str(day), "status": "no-s3", "n_files": 0, "bytes": 0,
                "downloaded": 0, "skipped": 0, "failed": 0, "glacier_skip": 0}
    dest_dir = DEST_ROOT / f"{day.year}" / f"{day:%Y%m%d}"
    downloaded = skipped = failed = glacier_skip = 0
    total_bytes = 0
    glacier_keys: list[str] = []
    for key, sz in files:
        if sz < MIN_FILE_BYTES:
            continue
        expiry = Path(key).name
        dest_path = dest_dir / expiry
        if dest_path.exists() and dest_path.stat().st_size == sz:
            skipped += 1
            continue
        ok, msg = s3_cp(key, dest_path)
        if ok:
            downloaded += 1
            total_bytes += sz
        elif msg == GLACIER_MARKER:
            glacier_skip += 1
            glacier_keys.append(key)
        else:
            failed += 1
            log.warning(f"FAIL {day} {expiry}: {msg}")
    if glacier_keys:
        # Log glacier keys to a per-year list for later restore-then-resync
        glacier_log = DEST_ROOT / f"glacier_keys_{day.year}.txt"
        glacier_log.parent.mkdir(parents=True, exist_ok=True)
        with open(glacier_log, "a") as fh:
            for k in glacier_keys:
                fh.write(k + "\n")
    return {
        "day": str(day),
        "status": "ok" if (failed == 0) else "partial",
        "n_files": len(files),
        "downloaded": downloaded, "skipped": skipped,
        "failed": failed, "glacier_skip": glacier_skip,
        "bytes": total_bytes,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, required=True, help="single year to sync, e.g. 2024")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--log", default=None, help="log file path (default: stderr only)")
    args = p.parse_args()

    log = logging.getLogger("sync_vix")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    if args.log:
        fh = logging.FileHandler(args.log)
        fh.setFormatter(fmt)
        log.addHandler(fh)

    start = date(args.year, 1, 1)
    end = date(args.year, 12, 31)
    days = list(daterange(start, end))
    log.info(f"VIX sync year={args.year}: {len(days)} weekdays, {args.threads} threads")

    counts = {"ok": 0, "partial": 0, "no-s3": 0, "exception": 0}
    total_files = total_bytes = total_glacier = 0
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futures = [ex.submit(process_day, d, log) for d in days]
        for i, f in enumerate(as_completed(futures), 1):
            try:
                r = f.result()
                counts[r["status"]] = counts.get(r["status"], 0) + 1
                total_files += r.get("downloaded", 0)
                total_bytes += r.get("bytes", 0)
                total_glacier += r.get("glacier_skip", 0)
            except Exception as e:
                counts["exception"] += 1
                log.warning(f"exception in worker: {type(e).__name__}: {e}")
            if i % 25 == 0:
                log.info(f"[{i}/{len(days)}] dl_files={total_files}  dl_bytes={total_bytes/1e6:.1f}MB "
                         f"glacier_skip={total_glacier}  counts={counts}")

    log.info(f"DONE year={args.year}: dl_files={total_files} dl_bytes={total_bytes/1e6:.1f}MB "
             f"glacier_skip={total_glacier} counts={counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
