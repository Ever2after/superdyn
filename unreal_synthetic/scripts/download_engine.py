"""Resumable parallel download. Read the signed URL from stdin; never save it."""

import argparse
from concurrent.futures import ThreadPoolExecutor, wait
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    url = sys.stdin.readline().strip()
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("An HTTPS download URL is required")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = ROOT / "reports/engine_download_progress.json"
    started = time.monotonic()
    request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        content_range = response.headers.get("Content-Range", "")
        if response.status != 206 or not content_range.startswith("bytes 0-0/"):
            raise RuntimeError("The download server must support HTTP byte ranges")
        total = int(content_range.rsplit("/", 1)[1])
        response.read()
    workers = max(1, min(args.workers, 16))
    part_size = math.ceil(total / workers)
    parts = output.parent / (output.stem + ".parts")
    parts.mkdir(exist_ok=True)
    paths = [parts / f"{i:03d}.part" for i in range(workers)]
    prefix = output.with_suffix(output.suffix + ".part")
    if prefix.exists():
        if any(path.exists() for path in paths):
            raise RuntimeError("Both a prefix and partitioned download exist")
        if prefix.stat().st_size <= part_size:
            prefix.replace(paths[0])
        else:
            with prefix.open("rb") as source:
                for path in paths:
                    remaining = min(part_size, prefix.stat().st_size - source.tell())
                    if remaining <= 0:
                        break
                    with path.open("wb") as target:
                        while remaining:
                            data = source.read(min(8 * 1024 * 1024, remaining))
                            target.write(data)
                            remaining -= len(data)
            prefix.unlink()

    def progress(status, **extra):
        received = sum(path.stat().st_size for path in paths if path.exists())
        state = dict(
            status=status,
            archive=str(output),
            bytes_received=received,
            bytes_total=total,
            percent=round(100 * received / total, 2),
            elapsed_seconds=round(time.monotonic() - started),
            workers=workers,
            signed_url_persisted=False,
            **extra,
        )
        save(report, state)
        print(json.dumps(state), flush=True)

    def download_part(index):
        path = paths[index]
        start = index * part_size
        end = min(total, start + part_size) - 1
        expected = end - start + 1
        for attempt in range(4):
            existing = path.stat().st_size if path.exists() else 0
            if existing == expected:
                return
            if existing > expected:
                raise RuntimeError("A download partition is oversized")
            offset = start + existing
            request = urllib.request.Request(
                url, headers={"Range": f"bytes={offset}-{end}"}
            )
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    if (
                        response.status != 206
                        or response.headers.get("Content-Range")
                        != f"bytes {offset}-{end}/{total}"
                    ):
                        raise RuntimeError("Unexpected download byte range")
                    with path.open("ab") as target:
                        while True:
                            chunk = response.read(8 * 1024 * 1024)
                            if not chunk:
                                break
                            target.write(chunk)
                if path.stat().st_size != expected:
                    raise RuntimeError("Incomplete partition")
                return
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2)

    progress("downloading")
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(download_part, i) for i in range(workers)}
            while pending:
                done, pending = wait(pending, timeout=10)
                for future in done:
                    future.result()
                progress("downloading")
        progress("assembling_archive")
        digest = hashlib.sha256()
        assembled = output.with_suffix(output.suffix + ".assembling")
        with assembled.open("wb") as target:
            for path in paths:
                with path.open("rb") as source:
                    for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
                        target.write(chunk)
                        digest.update(chunk)
        if assembled.stat().st_size != total:
            raise RuntimeError("Assembled archive size differs")
        assembled.replace(output)
        receipt = dict(
            status="downloaded",
            archive=str(output),
            bytes=total,
            sha256=digest.hexdigest(),
            source_host=parsed.hostname,
            signed_url_persisted=False,
            elapsed_seconds=round(time.monotonic() - started),
        )
        save(ROOT / "reports/engine_download.json", receipt)
        progress("downloaded", sha256=digest.hexdigest())
        shutil.rmtree(parts)
    except Exception as error:
        progress(
            "download_failed",
            error_type=type(error).__name__,
            http_status=getattr(error, "code", None),
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
