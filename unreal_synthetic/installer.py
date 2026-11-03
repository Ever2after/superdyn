"""Extract a user-provided official engine ZIP while preserving Unix modes/links."""

import os, shutil, stat, zipfile, zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def install_archive(archive, destination, progress=None, workers=1, resume=False):
    dest = Path(destination).resolve()
    if dest.exists() and any(dest.iterdir()) and not resume:
        raise RuntimeError("Use an empty engine destination")
    with zipfile.ZipFile(archive) as z:
        names = set()
        for item in z.infolist():
            target = (dest / item.filename).resolve()
            if not target.is_relative_to(dest) or item.filename in names:
                raise RuntimeError("Unsafe/duplicate archive path")
            names.add(item.filename)
            if stat.S_ISLNK(item.external_attr >> 16):
                link = z.read(item).decode()
                if Path(link).is_absolute() or not (
                    target.parent / link
                ).resolve().is_relative_to(dest):
                    raise RuntimeError("Unsafe archive link")
        total_bytes = sum(item.file_size for item in z.infolist())
        total_files = len(z.infolist())
        completed_bytes = completed_files = 0
        if progress:
            progress(completed_files, total_files, completed_bytes, total_bytes)
        dest.mkdir(parents=True, exist_ok=True)
        links = [
            item for item in z.infolist() if stat.S_ISLNK(item.external_attr >> 16)
        ]
        regular = [
            item for item in z.infolist() if not stat.S_ISLNK(item.external_attr >> 16)
        ]

        def extract(item):
            mode = item.external_attr >> 16
            path = dest / item.filename
            if path.is_symlink():
                raise RuntimeError("Unexpected symlink at archive file destination")
            if item.is_dir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                valid_existing = False
                if resume and path.is_file() and path.stat().st_size == item.file_size:
                    checksum = 0
                    with path.open("rb") as stream:
                        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                            checksum = zlib.crc32(chunk, checksum)
                    valid_existing = (checksum & 0xFFFFFFFF) == item.CRC
                if not valid_existing:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(item) as source, path.open("wb") as target:
                        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
            if mode:
                os.chmod(path, mode & 0o777)
            return item.file_size

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for future in as_completed(
                [pool.submit(extract, item) for item in regular]
            ):
                completed_files += 1
                completed_bytes += future.result()
                if progress:
                    progress(completed_files, total_files, completed_bytes, total_bytes)
        for item in links:
            path = dest / item.filename
            link = z.read(item).decode()
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.parent.resolve().is_relative_to(dest) or not (
                path.parent / link
            ).resolve().is_relative_to(dest):
                raise RuntimeError("Unsafe archive link chain")
            if path.is_symlink() and os.readlink(path) == link:
                pass
            else:
                path.symlink_to(link)
            completed_files += 1
            completed_bytes += item.file_size
            if progress:
                progress(completed_files, total_files, completed_bytes, total_bytes)
    return dest
