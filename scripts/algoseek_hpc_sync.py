"""Selective Algoseek sync from original buckets to HPC project space.

For each (date, root, dataset): list S3, identify the FRONT-MONTH file
by max bytesize, download only that file. Skip if local already has matching size.
Restartable, parallel, logs progress.

Targets (front-month only):
  TAQ:   us-futures-taq-v2-{year}            for ES, NQ, RTY, YM, ZN, 6E (2020-2025)
  Depth: us-futures-multiple-depth-{year}    for ES, NQ, RTY, YM, ZN, 6E (2020-2024 — depth not yet for 2025)
  VIX:   vix-futures/taq/{year}/{date}/      all expiries (whole curve, small)

Destination layout:
  /N/project/ksb-finance-backtesting/data/algoseek_futures/{taq,depth,vix}/{root}/{YYYY}/{YYYYMMDD}/{expiry}.csv.gz
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

DEST_ROOT = Path("/N/project/ksb-finance-backtesting/data/algoseek_futures")
PROFILE = "plt-de-dev"

DEFAULT_ROOTS = ["ES", "NQ", "RTY", "YM", "ZN", "6E"]


@dataclass(frozen=True)
class Job:
    dataset: str  # "taq" | "depth" | "vix"
    bucket: str  # full s3 bucket name
    s3_prefix: str  # path under bucket (no leading slash, ending with /)
    dest_dir: Path  # local directory to write into
    day: date
    root: str | None  # None for VIX (we keep all expiries)


def daterange(start: date, end: date, weekdays_only: bool = True):
    d = start
    while d <= end:
        if not weekdays_only or d.weekday() < 5:
            yield d
        d += timedelta(days=1)


def s3_ls(bucket: str, prefix: str, request_pays: bool = True) -> list[tuple[str, int]]:
    cmd = ["aws", "s3api", "list-objects-v2", "--bucket", bucket, "--prefix", prefix,
           "--profile", PROFILE, "--query", "Contents[].[Key,Size]", "--output", "text"]
    if request_pays:
        cmd += ["--request-payer", "requester"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=120)
    except subprocess.TimeoutExpired:
        return []
    except Exception:
        return []
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


def s3_cp(bucket: str, key: str, dest: Path, request_pays: bool = True) -> tuple[bool, str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["aws", "s3", "cp", f"s3://{bucket}/{key}", str(dest), "--profile", PROFILE, "--only-show-errors"]
    if request_pays:
        cmd += ["--request-payer", "requester"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=600)
    return (out.returncode == 0, (out.stdout + out.stderr).strip()[:200])


def process_taq_or_depth(job: Job, log: logging.Logger) -> dict:
    """Identify front-month file (max size) under prefix; download if missing or wrong size."""
    files = s3_ls(job.bucket, job.s3_prefix)
    if not files:
        return {"status": "no-s3-files", "key": None, "bytes": 0, "skipped": False}
    files.sort(key=lambda kv: kv[1], reverse=True)
    front_key, front_bytes = files[0]
    if front_bytes < 1000:  # all empty placeholders
        return {"status": "all-empty", "key": front_key, "bytes": front_bytes, "skipped": False}
    expiry = Path(front_key).name  # e.g. ESH4.csv.gz
    dest_path = job.dest_dir / expiry
    if dest_path.exists() and dest_path.stat().st_size == front_bytes:
        return {"status": "skip-already-have", "key": front_key, "bytes": front_bytes, "skipped": True}
    ok, msg = s3_cp(job.bucket, front_key, dest_path)
    return {"status": "downloaded" if ok else f"failed: {msg}", "key": front_key, "bytes": front_bytes, "skipped": False}


def process_vix(job: Job, log: logging.Logger) -> dict:
    """For VIX, sync the whole curve (all expiries) for that day."""
    files = s3_ls(job.bucket, job.s3_prefix)
    if not files:
        return {"status": "no-s3-files", "n_files": 0, "bytes": 0}
    total_bytes = 0
    downloaded = 0
    skipped = 0
    failed = 0
    for key, sz in files:
        if sz < 200:
            continue
        expiry = Path(key).name
        dest_path = job.dest_dir / expiry
        if dest_path.exists() and dest_path.stat().st_size == sz:
            skipped += 1
            continue
        ok, _ = s3_cp(job.bucket, key, dest_path)
        if ok:
            downloaded += 1
            total_bytes += sz
        else:
            failed += 1
    return {"status": f"vix d={downloaded}/s={skipped}/f={failed}", "n_files": len(files), "bytes": total_bytes}


def build_jobs(start: date, end: date, roots: list[str], include_vix: bool) -> list[Job]:
    jobs: list[Job] = []
    for d in daterange(start, end):
        # TAQ — original bucket per year
        for root in roots:
            jobs.append(Job(
                dataset="taq",
                bucket=f"us-futures-taq-v2-{d.year}",
                s3_prefix=f"{d:%Y%m%d}/{root}/",
                dest_dir=DEST_ROOT / "taq" / root / f"{d.year}" / f"{d:%Y%m%d}",
                day=d, root=root,
            ))
        # Depth — original bucket per year (correct spelling)
        for root in roots:
            jobs.append(Job(
                dataset="depth",
                bucket=f"us-futures-multiple-depth-{d.year}",
                s3_prefix=f"{d:%Y%m%d}/{root}/",
                dest_dir=DEST_ROOT / "depth" / root / f"{d.year}" / f"{d:%Y%m%d}",
                day=d, root=root,
            ))
        # VIX — separate bucket structure; pull whole curve (small)
        if include_vix:
            jobs.append(Job(
                dataset="vix",
                bucket="vix-futures",
                s3_prefix=f"taq/{d.year}/{d:%Y%m%d}/",
                dest_dir=DEST_ROOT / "vix" / f"{d.year}" / f"{d:%Y%m%d}",
                day=d, root=None,
            ))
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--roots", default=",".join(DEFAULT_ROOTS),
                    help="comma-separated symbol roots (default: ES,NQ,RTY,YM,ZN,6E)")
    ap.add_argument("--no-vix", action="store_true", help="skip VIX curve sync")
    ap.add_argument("--log", default="/N/project/ksb-finance-backtesting/data/algoseek_futures/sync.log")
    args = ap.parse_args()

    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(args.log), logging.StreamHandler()],
    )
    log = logging.getLogger("sync")

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    roots = [r.strip() for r in args.roots.split(",") if r.strip()]
    include_vix = not args.no_vix
    jobs = build_jobs(start, end, roots, include_vix)
    log.info(f"START sync: {len(jobs)} jobs across {(end-start).days+1} calendar days; "
             f"{args.threads} threads; roots={roots}; vix={include_vix}")
    log.info(f"DEST_ROOT={DEST_ROOT}")

    t0 = time.time()
    counts = {"downloaded": 0, "skipped": 0, "no-s3": 0, "all-empty": 0, "failed": 0, "vix": 0}
    bytes_dl = 0

    def run_one(job: Job):
        """Per-job work wrapped so NO exception can kill the pool. A failing job
        is counted as 'failed' and logged; the rest continue."""
        try:
            if job.dataset == "vix":
                return job, process_vix(job, log)
            return job, process_taq_or_depth(job, log)
        except Exception as exc:
            return job, {"status": f"exception: {type(exc).__name__}: {str(exc)[:200]}", "bytes": 0}

    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futures = [ex.submit(run_one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                job, result = fut.result()
            except Exception as exc:
                log.warning(f"worker exception leaked past run_one: {exc}")
                counts["failed"] += 1
                continue
            if result["status"] == "downloaded":
                counts["downloaded"] += 1
                bytes_dl += result["bytes"]
            elif result["status"].startswith("skip"):
                counts["skipped"] += 1
            elif result["status"] == "no-s3-files":
                counts["no-s3"] += 1
            elif result["status"] == "all-empty":
                counts["all-empty"] += 1
            elif result["status"].startswith("vix"):
                counts["vix"] += 1
            else:
                counts["failed"] += 1
                log.warning(f"FAIL {job.dataset} {job.root} {job.day}: {result['status']}")
            if i % 100 == 0:
                elapsed = time.time() - t0
                rate = bytes_dl / elapsed / 1e6 if elapsed else 0
                log.info(f"[{i}/{len(jobs)}] dl={counts['downloaded']} skip={counts['skipped']} fail={counts['failed']} no-s3={counts['no-s3']} empty={counts['all-empty']} vix={counts['vix']}  total_dl={bytes_dl/1e9:.2f}GB  rate={rate:.1f}MB/s  elapsed={elapsed/60:.1f}min")

    log.info(f"DONE in {(time.time()-t0)/60:.1f} min. counts={counts}  total_dl={bytes_dl/1e9:.2f}GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
