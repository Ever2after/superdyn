"""Generic resumable ZIP preprocessing, preserving source archives."""

import concurrent.futures, fcntl, os, shutil, stat, time, zipfile, zlib
from pathlib import Path, PurePosixPath
from .config import DATASETS
from .storage import read, save


def extract_one(archive, state):
    archive = Path(archive)
    destination = archive.parent.resolve()
    started = time.time()
    status = dict(
        archive=str(archive), status="running", completed_files=0, completed_bytes=0
    )
    try:
        with zipfile.ZipFile(archive) as z:
            entries = [i for i in z.infolist() if not i.is_dir()]
            assert len({i.filename for i in entries}) == len(
                entries
            ), "Duplicate ZIP paths"
            total = sum(i.file_size for i in entries)
            assert (
                shutil.disk_usage(destination).free > total + 1024**3
            ), "Insufficient disk space"
            status.update(total_files=len(entries), total_bytes=total)
            save(state, status)
            last = 0
            for entry in entries:
                path = PurePosixPath(entry.filename)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or "\\" in entry.filename
                    or stat.S_ISLNK(entry.external_attr >> 16)
                ):
                    raise ValueError("Unsafe ZIP path " + entry.filename)
                target = destination.joinpath(*path.parts)
                target.resolve().relative_to(destination)
                if target.exists():
                    crc = 0
                    with target.open("rb") as f:
                        for chunk in iter(lambda: f.read(1024**2), b""):
                            crc = zlib.crc32(chunk, crc)
                    if (
                        target.stat().st_size != entry.file_size
                        or crc & 0xFFFFFFFF != entry.CRC
                    ):
                        raise ValueError(
                            "Existing file conflicts with archive: " + str(target)
                        )
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    tmp = target.with_name(target.name + ".qa_extract.partial")
                    with z.open(entry) as src, tmp.open("wb") as dst:
                        shutil.copyfileobj(src, dst, 1024**2)
                    os.link(tmp, target)
                    tmp.unlink()
                status["completed_files"] += 1
                status["completed_bytes"] += entry.file_size
                now = time.time()
                if now - last >= 2:
                    elapsed = now - started
                    status.update(
                        elapsed_seconds=elapsed,
                        remaining_seconds=(total - status["completed_bytes"])
                        * elapsed
                        / max(status["completed_bytes"], 1),
                    )
                    save(state, status)
                    last = now
            status.update(
                status="completed",
                elapsed_seconds=time.time() - started,
                remaining_seconds=0,
                crc_verified=True,
                archives_preserved=True,
            )
            save(state, status)
    except Exception as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save(state, status)
        raise


def run(root, out, datasets, jobs=3):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    archives = [
        p for d in datasets if d in DATASETS for p in sorted((root / d).glob("*.zip"))
    ]
    if not archives:
        raise ValueError("No ZIP archives in selected active datasets")
    with (root / ".qa_pipeline_extract.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = {
                pool.submit(extract_one, p, out / f"{p.parent.name}_{p.stem}.json"): p
                for p in archives
            }
            pending, failed = set(futures), []
            while pending:
                finished, pending = concurrent.futures.wait(
                    pending, timeout=2, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in finished:
                    try:
                        future.result()
                    except Exception as exc:
                        failed.append(
                            dict(archive=str(futures[future]), error=str(exc))
                        )
                states = {}
                for archive in archives:
                    state = out / f"{archive.parent.name}_{archive.stem}.json"
                    if state.exists():
                        states[str(archive)] = read(state)
                save(
                    out / "run_status.json",
                    dict(
                        status=(
                            "running"
                            if pending
                            else "failed" if failed else "completed"
                        ),
                        archives=len(archives),
                        failures=failed,
                        datasets=states,
                        archives_preserved=True,
                        completed_files=sum(
                            s.get("completed_files", 0) for s in states.values()
                        ),
                        total_files=sum(
                            s.get("total_files", 0) for s in states.values()
                        ),
                        completed_bytes=sum(
                            s.get("completed_bytes", 0) for s in states.values()
                        ),
                        total_bytes=sum(
                            s.get("total_bytes", 0) for s in states.values()
                        ),
                        remaining_seconds=max(
                            (s.get("remaining_seconds") or 0 for s in states.values()),
                            default=0,
                        ),
                    ),
                )
        if failed:
            raise RuntimeError("ZIP extraction failed; see run_status.json")
