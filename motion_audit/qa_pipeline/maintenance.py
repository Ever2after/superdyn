"""Conservative cleanup of task-owned, inactive runtime artifacts."""

from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path

from .config import ROOT
from .storage import save


SKIP = {"vendor", "checkpoints", "videos", "cache", "cases", "source", "media",
        "predictions", "raw", "exclude", "review", ".git"}


def runtime_files(root):
    """Never follow links or walk model environments, datasets or evidence."""
    root = Path(root)
    candidates = []
    for folder, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if d not in SKIP and not d.startswith(".venv")
                   and not (Path(folder) / d).is_symlink()]
        for name in files:
            p = Path(folder) / name
            if p.is_symlink():
                continue
            if "__pycache__" in p.parts or name.endswith((".pyc", ".log", ".pid", ".tmp", ".part")) or ".tmp." in name or name == "controller.lock":
                candidates.append(p)
    return sorted(candidates)


def active_controllers(root):
    active = []
    root = Path(root).resolve()
    for folder in Path("/proc").iterdir():
        if not folder.name.isdigit() or int(folder.name) == os.getpid():
            continue
        try:
            args = (folder / "cmdline").read_bytes().decode().split("\0")
            cwd = (folder / "cwd").resolve()
        except (OSError, UnicodeError):
            continue
        scripts = [(cwd / a).resolve() for a in args if a.endswith(".py")]
        owned_script = any(p == root / "qa.py" or p.is_relative_to(root / "legacy") for p in scripts)
        owned_module = cwd == root and any(a.startswith("qa_pipeline.") for a in args)
        if owned_script or owned_module:
            active.append(dict(pid=int(folder.name), command=args))
    return active


def cleanup(root=ROOT, *, apply=False):
    root = Path(root).resolve()
    files = runtime_files(root)
    active = active_controllers(root)
    if active:
        raise RuntimeError(f"Motion-audit jobs are still active; cleanup deferred: {active}")
    ledger = dict(status="cleaned" if apply else "dry_run", checked_at=datetime.now(timezone.utc).isoformat(),
                  files=len(files), bytes=sum(p.stat().st_size for p in files),
                  removed=[str(p.relative_to(root)) for p in files],
                  preserved="QA, model responses, references, marked media, geometry/GMOS caches, models and environments")
    if apply:
        # Check every lock before deleting anything.
        for path in files:
            if path.name == "controller.lock":
                with path.open("a+") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for path in files:
            path.unlink()
        empty_candidates = []
        for folder, dirs, names in os.walk(root, followlinks=False):
            dirs[:] = [d for d in dirs if d not in SKIP and not d.startswith(".venv")
                       and not (Path(folder) / d).is_symlink()]
            p = Path(folder)
            if p.name in {"__pycache__", "logs"}:
                empty_candidates.append(p)
        for p in reversed(empty_candidates):
            if not any(p.iterdir()):
                p.rmdir()
        save(root / "maintenance.json", ledger)
    return ledger
