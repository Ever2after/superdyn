"""Active-scope discovery and normalized manifests for cached geometry."""

import csv, os, hashlib
from pathlib import Path
from .config import ROOT, DATA_ROOT, DATASETS, EXCLUDED, group
from .storage import read, save, write_csv
from .metrics import interval_statistics, summarize_amplitudes
import numpy as np


def cache_source(result, cache):
    """Resolve an extracted ZIP source only when its cached frames still match."""
    source = Path(result["source_path"])
    if result.get("source_kind") != "zip":
        return source if (source / "rgb").is_dir() else None
    source = source.parent / result["zip_prefix"]
    if not (source / "rgb").is_dir():
        return None
    files = sorted(p for p in (source / "rgb").iterdir()
                   if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(files) < 9 or len(files) != result.get("total_frames"):
        return None
    indices = np.rint(np.linspace(0, len(files) - 1, 33)).astype(int)[::4]
    if (indices.tolist() != result.get("selected_frame_indices") or
            [files[i].name for i in indices] != result.get("selected_frame_names")):
        return None
    with np.load(cache) as archive:
        if not np.allclose(archive["times"], indices / (len(files) - 1)):
            return None
    from PIL import Image
    for frame, index in enumerate(indices):
        cached = Path(cache).parent / f"{frame * 4:04d}.png"
        if not cached.is_file():
            return None
        with Image.open(cached) as a, Image.open(files[index]) as b:
            if not np.array_equal(np.asarray(a.convert("RGB")), np.asarray(b.convert("RGB"))):
                return None
    return source


def discover(root=DATA_ROOT, datasets=DATASETS):
    rows = []
    for dataset in sorted(datasets):
        if dataset in EXCLUDED:
            raise ValueError("Dataset excluded from current scope: " + dataset)
        folder = Path(root) / dataset
        if not folder.is_dir():
            continue
        for entry in sorted(os.scandir(folder), key=lambda e: e.name):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            source = Path(entry.path)
            rows.append(
                dict(
                    video_id=f"{dataset}__{entry.name}",
                    stratum=dataset,
                    scene_id=entry.name,
                    source_kind="directory",
                    source_path=str(source),
                    rgb_directory_ready=(source / "rgb").is_dir(),
                )
            )
    return rows


def manifest(path):
    value = read(Path(path))
    rows = (
        value if isinstance(value, list) else value.get("videos", value.get("scenes"))
    )
    if rows is None:
        raise ValueError("Manifest must be a list or contain videos/scenes")
    return rows


def cache_catalog(roots=None):
    roots = (
        [Path(r).resolve() for r in roots]
        if roots
        else [ROOT / "outputs/temporal_geometry_fullscope/geometry",
              *sorted((ROOT / "outputs").glob("dynamicverse*"))]
    )
    catalog = {}
    for root in roots:
        raw = {}
        p = root / "per_video_unfiltered.csv"
        if p.exists():
            with p.open() as f:
                raw = {r["video_id"]: r for r in csv.DictReader(f)}
        if not (root / "results").is_dir():
            continue
        for p in sorted((root / "results").glob("*.json")):
            result = read(p)
            vid = result.get("video_id")
            cache = root / "cache" / str(vid) / "predictions.npz"
            if (
                result.get("stratum") not in DATASETS
                or result.get("status") != "ok"
                or not cache.is_file()
            ):
                continue
            if vid in catalog:
                continue
            source = cache_source(result, cache)
            if source is None:
                continue
            metrics = result.get("unfiltered_amplitude_summary") or raw.get(vid)
            if not metrics:
                with np.load(cache) as archive:
                    prediction = {k: archive[k] for k in archive.files}
                intervals, _ = interval_statistics(
                    **prediction, pixel_quality_filter=False
                )
                metrics = summarize_amplitudes(
                    [i | {"quality_valid": True} for i in intervals]
                )
            catalog[vid] = dict(
                video_id=vid,
                stratum=result["stratum"],
                scene_id=result["scene_id"],
                source_path=str(source),
                source_kind="directory",
                group=group(metrics),
                vggt_cache=str(cache),
                cache_origin=str(root),
                result_path=str(p),
                eligible=result["amplitude_summary"]["eligible"],
                **{
                    k: metrics[k]
                    for k in ["translation_high", "rotation_high", "dynamic_high"]
                },
            )
    return catalog


def prepare_generation(
    out,
    *,
    manifest_path=None,
    cache_roots=None,
    gpus=(0,),
    base_port=18620,
    qwen_model=None,
    quality_filter=False,
    skip_missing=False,
):
    from .config import TH, TASKS, SEED, QWEN_MODEL

    if qwen_model and Path(qwen_model).exists():
        qwen_model = str(Path(qwen_model).resolve())

    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "selection.json").exists():
        return read(out / "selection.json")
    if not gpus or len(set(gpus)) != len(gpus) or any(g < 0 for g in gpus):
        raise ValueError("Provide distinct nonnegative GPU indices")
    catalog = cache_catalog(cache_roots)
    requested = manifest(manifest_path) if manifest_path else list(catalog.values())
    requested = [r for r in requested if r["stratum"] in DATASETS]
    missing = [r["video_id"] for r in requested if r["video_id"] not in catalog]
    if missing and not skip_missing:
        raise ValueError(
            f"{len(missing)} requested videos lack a usable cache. Run preprocess first, or explicitly use --skip-missing-cache. First: {missing[:3]}"
        )
    videos = [
        dict(catalog[r["video_id"]]) for r in requested if r["video_id"] in catalog
    ]
    if quality_filter:
        videos = [r for r in videos if r["eligible"]]
    if not videos:
        raise ValueError("No cached videos selected")
    if len({r["video_id"] for r in videos}) != len(videos):
        raise ValueError("Duplicate video IDs in manifest")
    source = out / "source"
    (source / "cache").mkdir(parents=True, exist_ok=True)
    (source / "results").mkdir(exist_ok=True)
    for index, row in enumerate(videos):
        row.update(worker_gpu=gpus[index % len(gpus)], qa_split="development")
        target = source / "cache" / row["video_id"]
        if not target.exists():
            target.symlink_to(Path(row["vggt_cache"]).parent, target_is_directory=True)
        target = source / "results" / f'{row["video_id"]}.json'
        if not target.exists():
            original = read(Path(row["result_path"]))
            if original["source_path"] == row["source_path"]:
                target.symlink_to(Path(row["result_path"]))
            else:
                # Keep historical metadata untouched; QA consumes extracted RGB.
                save(target, original | dict(
                    source_path=row["source_path"], source_kind="directory",
                    original_source_path=original["source_path"],
                    source_rebound_after_exact_frame_validation=True,
                ))
    groups = {
        g: sum(r["group"] == g for r in videos)
        for g in sorted({r["group"] for r in videos})
    }
    selection = dict(
        planned=len(videos),
        videos=videos,
        selected_groups=groups,
        group_order="T/R/D",
        seed=SEED,
        policy="All requested cached videos; optional clip-quality filter only when explicitly selected. Unfiltered group metrics, unchanged local QA validity and margin gates.",
        manual_review=False,
    )
    save(
        out / "config.json",
        dict(
            planned=len(videos),
            source=str(source),
            base_port=base_port,
            gpus=list(gpus),
            qa_thresholds=TH,
            tasks=TASKS,
            excluded_sources=sorted(EXCLUDED),
            video_selection_quality_filter=quality_filter,
            new_vggt_inference=False,
            qwen=dict(
                model=qwen_model or QWEN_MODEL,
                gpu_memory_utilization=0.30,
                concurrency_per_replica=2,
            ),
            input_manifest=str(manifest_path) if manifest_path else None,
            cache_roots=[str(p) for p in cache_roots] if cache_roots else None,
            skipped_missing_cache=missing if skip_missing else [],
            manual_review=False,
            code_sha256={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (ROOT / "qa_pipeline").glob("*.py")
            },
        ),
    )
    save(out / "selection.json", selection)
    write_csv(out / "selected_videos.csv", videos)
    (out / "cases").mkdir(exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    for gpu in gpus:
        work = out / "workers" / f"gpu{gpu}"
        work.mkdir(parents=True, exist_ok=True)
        if not (work / "cases").exists():
            (work / "cases").symlink_to("../../cases", target_is_directory=True)
        shard = [r for r in videos if r["worker_gpu"] == gpu]
        save(
            work / "selection.json",
            dict(videos=shard, video_ids=[r["video_id"] for r in shard]),
        )
    return selection
