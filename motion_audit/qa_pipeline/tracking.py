"""Persistent surface-core tracks, bootstrap uncertainty and motion gates."""

import collections, itertools
from pathlib import Path
import numpy as np
import cv2
from . import storage as base
from .config import ROOT, SEED, TH, TASKS

OUT = ROOT / "outputs/qa"


def sample(array, xy):
    return cv2.remap(
        array.astype("float32"),
        xy[:, 0].astype("float32").reshape(-1, 1),
        xy[:, 1].astype("float32").reshape(-1, 1),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).reshape((len(xy),) + array.shape[2:])


def direction(v, minimum):
    v = np.asarray(v)
    idx = np.argsort(abs(v))
    a, b = idx[-1], idx[-2]
    if (
        abs(v[a]) < minimum
        or abs(v[a]) < TH["direction_dominance"] * abs(v[b])
        or abs(v[a]) - abs(v[b]) < TH["direction_gap"]
    ):
        return None
    return [("left", "right"), ("up", "down"), ("backward", "forward")][a][
        int(v[a] > 0)
    ]


def seed_points(mask, box):
    h, w = mask.shape
    y0, x0 = np.where(mask)
    if len(y0) < TH["min_mask_pixels"]:
        return np.empty((0, 2), float)
    if box is not None:
        x1, y1, x2, y2 = np.array(box) / 1000 * [w, h, w, h]
        y, x = np.mgrid[:h, :w]
        core = (
            mask
            & (x >= x1 + 0.2 * (x2 - x1))
            & (x <= x1 + 0.8 * (x2 - x1))
            & (y >= y1 + 0.15 * (y2 - y1))
            & (y <= y1 + 0.8 * (y2 - y1))
        )
        if core.sum() >= TH["min_mask_pixels"]:
            mask = core
    yy, xx = np.where(mask)
    step = max(1, int(np.sqrt(len(xx) / TH["seed_count"])))
    use = (xx % step == 0) & (yy % step == 0)
    pts = np.stack([xx[use], yy[use]], 1).astype(float)
    if len(pts) > TH["seed_count"]:
        pts = pts[np.linspace(0, len(pts) - 1, TH["seed_count"]).astype(int)]
    return pts


def persistent_track(p, masks, world, good, obj, a, b, score, min_score=None):
    if min_score is None:
        min_score = TH["sam_score"]
    if any(s < min_score for s in score[a : b + 1]):
        return None
    h, w = p["depth"].shape[1:]
    regions = []
    for i in range(a, b + 1):
        region = (
            cv2.erode(masks[i].astype("uint8"), np.ones((3, 3), np.uint8)).astype(bool)
            & good[i]
        )
        if region.sum() < TH["min_mask_pixels"]:
            return None
        regions.append(region)
    xy = seed_points(regions[0], obj["frames"][a]["bbox"])
    initial = len(xy)
    if initial < TH["min_tracks"]:
        return None
    locations = [xy.copy()]
    valid = np.ones(initial, bool)
    cover = []
    for i in range(a, b):
        dense = "dense_forward" in p
        steps = range(i * 4, (i + 1) * 4) if dense else [i]
        for j in steps:
            previous = int(valid.sum())
            flow = sample(p["dense_forward" if dense else "forward"][j], xy)
            next_xy = xy + flow
            backward = sample(p["dense_backward" if dense else "backward"][j], next_xy)
            fb = np.linalg.norm(flow + backward, axis=1)
            inside = (
                (next_xy[:, 0] >= 1)
                & (next_xy[:, 0] < w - 2)
                & (next_xy[:, 1] >= 1)
                & (next_xy[:, 1] < h - 2)
            )
            valid &= (
                inside
                & np.isfinite(flow).all(1)
                & (
                    fb
                    <= np.minimum(
                        TH["max_fb_error_px"], 1.0 + 0.05 * np.linalg.norm(flow, axis=1)
                    )
                )
            )
            if int(valid.sum()) < TH["min_forward_coverage"] * previous:
                return None
            xy = next_xy
        hit = sample(regions[i + 1 - a].astype(float), next_xy) > 0.99
        valid &= hit
        locations.append(xy.copy())
        cover.append(int(valid.sum()))
    n = int(valid.sum())
    if n < TH["min_tracks"] or n / initial < TH["min_track_survival"]:
        return None
    xyz = np.stack([sample(world[i], locations[i - a][valid]) for i in range(a, b + 1)])
    if not np.isfinite(xyz).all():
        return None
    seeds = locations[0][valid]
    span = np.maximum(np.ptp(seeds, axis=0), 1)
    cell = np.minimum(3, ((seeds - seeds.min(0)) / span * 4).astype(int))
    blocks = cell[:, 1] * 4 + cell[:, 0]
    if len(np.unique(blocks)) < 4:
        return None
    return dict(
        points=xyz,
        anchor=np.mean(xyz, axis=1),
        blocks=blocks,
        seeds=seeds,
        xy=np.stack([x[valid] for x in locations]),
        survived=n,
        initial=initial,
        survival=n / initial,
    )


def bootstrap(track, rng):
    blocks = np.unique(track["blocks"])
    samples = []
    for _ in range(TH["bootstrap_replicates"]):
        chosen = rng.choice(blocks, len(blocks), replace=True)
        ids = np.concatenate([np.flatnonzero(track["blocks"] == k) for k in chosen])
        samples.append(np.mean(track["points"][:, ids], axis=1))
    return np.array(samples)


def pair_metrics(a, b, boot_a=None, boot_b=None):
    distance = np.linalg.norm(a - b, axis=-1)
    length_a = float(np.linalg.norm(np.diff(a, axis=0), axis=-1).sum())
    length_b = float(np.linalg.norm(np.diff(b, axis=0), axis=-1).sum())
    delta = float(distance[-1] - distance[0])
    gap = abs(length_a - length_b)
    ratio = max(length_a, length_b) / max(min(length_a, length_b), 1e-8)
    separation = (
        abs(delta) >= TH["pair_range_abs"]
        and abs(delta) / max(distance[0], 1e-8) >= TH["pair_range_relative"]
        and distance[0] >= TH["pair_initial_separation"]
    )
    travel = (
        gap >= TH["travel_gap"]
        and ratio >= TH["travel_ratio"]
        and max(length_a, length_b) >= TH["travel_max_min"]
    )
    info = dict(
        distance_depths=distance.tolist(),
        distance_change_depths=delta,
        relative_distance_change=abs(delta) / max(float(distance[0]), 1e-8),
        travel_a_depths=length_a,
        travel_b_depths=length_b,
        travel_gap_depths=gap,
        travel_ratio=ratio,
    )
    if boot_a is not None:
        db = np.linalg.norm(boot_a - boot_b, axis=-1)
        dchange = db[:, -1] - db[:, 0]
        la = np.linalg.norm(np.diff(boot_a, axis=1), axis=-1).sum(1)
        lb = np.linalg.norm(np.diff(boot_b, axis=1), axis=-1).sum(1)
        dc = np.quantile(dchange, [0.025, 0.975])
        lc = np.quantile(la - lb, [0.025, 0.975])
        separation = separation and (
            dc[0] >= TH["bootstrap_gap"] if delta > 0 else dc[1] <= -TH["bootstrap_gap"]
        )
        travel = travel and (
            lc[0] >= TH["bootstrap_gap"]
            if length_a > length_b
            else lc[1] <= -TH["bootstrap_gap"]
        )
        info.update(
            distance_change_bootstrap95=dc.tolist(),
            travel_a_minus_b_bootstrap95=lc.tolist(),
        )
    return bool(separation), bool(travel), info


def scan_video(vid, *, include_nonpassing=False, save_single_tracks=False):
    target = OUT / "cases" / vid
    objects = base.read(target / "objects.json")
    p, r = base.load(vid)
    scale = r["depth_scale"]
    e = p["extrinsics"].astype(float)
    flowpath = target / "dense_flow.npz"
    if flowpath.exists():
        with np.load(flowpath) as f:
            p.update(
                {k: f[k].astype("float32") for k in ["dense_forward", "dense_backward"]}
            )
    world = np.array(
        [
            base.world_grid(p["depth"][i], p["intrinsics"][i], e[i]) / scale
            for i in range(9)
        ]
    )
    centers = -np.einsum("tji,tj->ti", e[:, :, :3], e[:, :, 3]) / scale
    good = np.isfinite(p["depth"]) & (p["depth"] > 0) & np.isfinite(p["conf"])
    for i in range(9):
        good[i] &= p["conf"][i] >= np.quantile(
            p["conf"][i][np.isfinite(p["conf"][i])], 0.2
        )
    with np.load(target / "instance_masks.npz") as f:
        masks = {k: f[k].astype(bool) for k in f.files}
    # Pixels claimed by more than one instance have ambiguous identity. They
    # cannot seed or support a persistent target track, including third objects.
    ambiguity = np.sum(np.stack(list(masks.values())), axis=0) > 1 if masks else None
    tracking_masks = {k: v & ~ambiguity for k, v in masks.items()}
    segmentation = base.read(target / "segmentation.json")
    scores = segmentation["scores"]
    min_score = segmentation.get("score_threshold", TH["sam_score"])
    windows = [
        (a, b)
        for a in range(9)
        for b in range(a + TH["min_intervals"], 9)
        if p["times"][b] - p["times"][a] >= TH["min_time_span"]
        and all(x["quality_valid"] for x in r["intervals"][a:b])
    ]
    windows.sort(key=lambda ab: (-(p["times"][ab[1]] - p["times"][ab[0]]), ab))
    tracks = {}
    records = []
    rejections = collections.Counter()
    fallback = None
    singles = []
    for obj in objects:
        for a, b in windows:
            tr = persistent_track(
                p,
                tracking_masks[obj["object_id"]],
                world,
                good,
                obj,
                a,
                b,
                scores[obj["object_id"]],
                min_score=min_score,
            )
            if tr:
                tracks[(obj["object_id"], a, b)] = tr
        if save_single_tracks:
            valid = [
                (a, b, tracks[(obj["object_id"], a, b)])
                for a, b in windows
                if (obj["object_id"], a, b) in tracks
            ]
            if valid:
                a, b, tr = valid[0]
                singles.append(
                    dict(
                        video_id=vid,
                        object_ids=[obj["object_id"]],
                        descriptions=[obj["description"]],
                        start=a,
                        end=b,
                        normalized_span=float(p["times"][b] - p["times"][a]),
                        pair_distance_pass=False,
                        travel_comparison_pass=False,
                        metrics={},
                        anchors_a=tr["anchor"].tolist(),
                        anchors_b=[],
                        bootstrap_a=bootstrap(
                            tr, np.random.default_rng(SEED + a * 100 + b)
                        ).tolist(),
                        bootstrap_b=[],
                        camera_centers=centers[a : b + 1].tolist(),
                        track_counts=[tr["survived"]],
                        initial_track_counts=[tr["initial"]],
                        track_survival=[tr["survival"]],
                        mask_overlap=0.0,
                        ambiguous_instance_pixels_excluded=True,
                        tracking=(
                            "RAFT-large 33 context frames"
                            if "dense_forward" in p
                            else "RAFT-small 9 measurement frames"
                        ),
                    )
                )
    for oa, ob in itertools.combinations(objects, 2):
        for a, b in windows:
            ta, tb = tracks.get((oa["object_id"], a, b)), tracks.get(
                (ob["object_id"], a, b)
            )
            if ta is None or tb is None:
                rejections["insufficient_persistent_tracks"] += 1
                continue
            overlap = max(
                float(
                    (masks[oa["object_id"]][i] & masks[ob["object_id"]][i]).sum()
                    / max(
                        min(
                            masks[oa["object_id"]][i].sum(),
                            masks[ob["object_id"]][i].sum(),
                        ),
                        1,
                    )
                )
                for i in range(a, b + 1)
            )
            if overlap > TH["max_pair_mask_overlap"]:
                rejections["instance_mask_overlap"] += 1
                continue
            rng = np.random.default_rng(SEED + a * 100 + b)
            ba, bb = bootstrap(ta, rng), bootstrap(tb, rng)
            distance, travel, metrics = pair_metrics(ta["anchor"], tb["anchor"], ba, bb)
            if not distance and not travel:
                rejections["insufficient_effect_or_bootstrap_margin"] += 1
                if not include_nonpassing:
                    continue
            record = dict(
                video_id=vid,
                object_ids=[oa["object_id"], ob["object_id"]],
                descriptions=[oa["description"], ob["description"]],
                start=a,
                end=b,
                normalized_span=float(p["times"][b] - p["times"][a]),
                pair_distance_pass=distance,
                travel_comparison_pass=travel,
                metrics=metrics,
                anchors_a=ta["anchor"].tolist(),
                anchors_b=tb["anchor"].tolist(),
                camera_centers=centers[a : b + 1].tolist(),
                track_counts=[ta["survived"], tb["survived"]],
                initial_track_counts=[ta["initial"], tb["initial"]],
                track_survival=[ta["survival"], tb["survival"]],
                mask_overlap=overlap,
            )
            record.update(
                bootstrap_a=ba.tolist(),
                bootstrap_b=bb.tolist(),
                tracking=(
                    "RAFT-large 33 context frames"
                    if "dense_forward" in p
                    else "RAFT-small 9 measurement frames"
                ),
                ambiguous_instance_pixels_excluded=True,
            )
            if distance or travel:
                records.append(record)
            elif fallback is None or (
                record["normalized_span"],
                min(record["track_counts"]),
            ) > (fallback["normalized_span"], min(fallback["track_counts"])):
                fallback = record
    if not records and fallback is not None:
        records.append(fallback)
    if save_single_tracks:
        base.save(target / "single_object_tracks.json", singles)
    base.save(target / "candidates.json", records)
    base.save(target / "automatic_rejections.json", dict(rejections))
    print(
        "scanned",
        vid,
        "distance",
        sum(c["pair_distance_pass"] for c in records),
        "travel",
        sum(c["travel_comparison_pass"] for c in records),
        flush=True,
    )
    return records
