"""Exact-GT temporal geometry QA for pilot clips (real-split rules and format).

Answers use motion_audit's thresholds (TH) and direction rule on rendered
camera poses, depths and instance masks. Each designed QA is kept only when
both anchor definitions agree, targets stay visible and GT equals the design.
Output mirrors motion_audit's final QA folder so its evaluator runs unchanged.
"""

import gzip
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2
import numpy as np

from bridge import ROOT
from superdyn_ue.io_utils import save_json
from pilot_design import FPS, FRAMES, MEASURE

AUDIT = ROOT.parent / "motion_audit"
sys.path.insert(0, str(AUDIT))
from qa_pipeline.config import TH, GROUPS  # noqa: E402

SEED = 20261102
AXES = ["left", "right", "up", "down", "backward", "forward"]
S = np.diag([1.0, -1.0, 1.0])
SKY_M = 200.0


def direction(v, minimum):
    """Identical to motion_audit qa_pipeline.tracking.direction."""
    v = np.asarray(v)
    idx = np.argsort(abs(v))
    a, b = idx[-1], idx[-2]
    if abs(v[a]) < minimum or abs(v[a]) < TH["direction_dominance"] * abs(v[b]) or abs(v[a]) - abs(v[b]) < TH["direction_gap"]:
        return None
    return [("left", "right"), ("up", "down"), ("backward", "forward")][a][int(v[a] > 0)]


def camera_rh(m):
    """Unreal actor matrix (cm, columns = fwd/right/up) -> OpenCV R (world->cam) and centre (m)."""
    m = np.asarray(m, float)
    axes = S @ m[:3, :3]
    axes = axes / np.linalg.norm(axes, axis=0)
    fwd, right, up = axes[:, 0], axes[:, 1], axes[:, 2]
    R = np.stack([right, -up, fwd])
    if abs(np.linalg.det(R) - 1) > 1e-4:
        raise RuntimeError("Camera conversion is not a rotation")
    return R, S @ m[:3, 3] / 100.0


def point_rh(p_cm):
    return S @ np.asarray(p_cm, float) / 100.0


def intrinsics(fov_deg, width, height):
    fx = width / 2 / math.tan(math.radians(fov_deg) / 2)
    return np.array([[fx, 0, width / 2], [0, fx, height / 2], [0, 0, 1.0]])


def project(K, R, C, X):
    x = R @ (np.asarray(X) - C)
    return np.array([K[0, 0] * x[0] / x[2] + K[0, 2], K[1, 1] * x[1] / x[2] + K[1, 2]]), x[2]


def read_gt(folder, frame):
    """Multilayer MRQ EXR: (instance ids, depth cm) for one rendered frame.

    Depth comes from MRQ's WorldDepth pass; ids from the pilot GT material,
    whose G channel (scene depth) is kept as a cross-check.
    """
    import OpenEXR
    path = folder / ("frame_%06d.exr" % frame)
    if not path.exists():
        raise RuntimeError("Missing GT EXR " + str(path))
    channels = OpenEXR.File(str(path)).parts[0].channels
    depth_key = next(k for k in channels if "WorldDepth" in k)
    id_key = next(k for k in channels if "GroundTruth" in k)
    depth = channels[depth_key].pixels[..., 0].astype(np.float64)
    ids = channels[id_key].pixels[..., 0].astype(np.float64)
    check = channels[id_key].pixels[..., 1].astype(np.float64)
    sky = ~(np.abs(depth) < 1e7)
    depth[sky] = np.inf
    finite = ~sky & np.isfinite(check)
    READ_CHECK.append(float(np.median(np.abs(depth[finite] - check[finite]))) if finite.any() else 0.0)
    return ids, depth


READ_CHECK = []


def unproject(K, R, C, depth_m, mask):
    v, u = np.nonzero(mask)
    z = depth_m[v, u]
    x = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0] * z, (v + 0.5 - K[1, 2]) / K[1, 1] * z, z])
    return (R.T @ x).T + C


class Clip:
    def __init__(self, output, plan):
        self.plan = plan
        self.id = plan["id"]
        self.folder = output / "clips" / self.id
        ev = json.loads((self.folder / "evaluated.json").read_text())
        self.ev = ev
        self.width, self.height = 1280, 720
        self.K = intrinsics(plan["camera"]["fov_deg"], self.width, self.height)
        cams = [camera_rh(m) for m in ev["camera"]]
        self.R = np.array([c[0] for c in cams])
        self.C = np.array([c[1] for c in cams])
        self.actors = {a["id"]: a for a in plan["actors"]}
        self.stencil = {a["id"]: a["stencil"] for a in plan["actors"]}
        self.center = {}
        for key, rec in ev["actors"].items():
            pts = [c if c is not None else b[0] for c, b in zip(rec["center"], rec["bounds"])]
            self.center[key] = np.array([point_rh(p) for p in pts])
        self.ids, self.depth = [], []
        for k in MEASURE:
            ids, depth = read_gt(self.folder, k)
            if np.abs(ids - np.round(ids)).max() > 1e-3:
                raise RuntimeError("Stencil pass is not integral in %s frame %d" % (self.id, k))
            self.ids.append(np.round(ids).astype(int))
            self.depth.append(depth / 100.0)
        ours = np.isin(np.array(self.ids), list(self.stencil.values()))
        background = np.array(self.depth)[~ours]
        background = background[(background > 0) & (background < SKY_M)]
        self.scale = float(np.median(background))
        kernel = np.ones((3, 3), np.uint8)
        self.masks = {key: np.array([ids == sid for ids in self.ids]) for key, sid in self.stencil.items()}
        self.eroded = {key: np.array([cv2.erode(m.astype(np.uint8), kernel) > 0 for m in masks]) for key, masks in self.masks.items()}
        self.visible = {}
        for key, masks in self.eroded.items():
            pts = []
            for i, k in enumerate(MEASURE):
                pts.append(unproject(self.K, self.R[k], self.C[k], self.depth[i], masks[i]).mean(0) if masks[i].sum() >= 20 else np.full(3, np.nan))
            self.visible[key] = np.array(pts)

    def anchors(self, key, kind):
        return self.center[key][MEASURE] if kind == "center" else self.visible[key]


def answer(clip, task, window, objects, kind):
    """GT answer and evidence for one task/window/anchor kind (None if below threshold)."""
    a, b = window
    fa, fb = MEASURE[a], MEASURE[b]
    s = clip.scale
    R, C = clip.R, clip.C
    if task == "camera_translation_direction":
        delta = R[fa] @ (C[fb] - C[fa]) / s
        return direction(delta, TH["camera_translation"]), dict(displacement_start_camera_depths=delta.tolist(), displacement_m=(delta * s).tolist())
    if task == "camera_view_direction_change":
        optical = R[fb][2] @ R[fa].T
        yaw, pitch = np.degrees([math.atan2(optical[0], optical[2]), math.atan2(optical[1], math.hypot(optical[0], optical[2]))])
        big, small = sorted([abs(yaw), abs(pitch)], reverse=True)
        ok = big >= TH["camera_turn_deg"] and big >= TH["direction_dominance"] * small and big - small >= TH["camera_turn_gap_deg"]
        value = (("right" if yaw > 0 else "left") if abs(yaw) > abs(pitch) else ("down" if pitch > 0 else "up")) if ok else None
        return value, dict(yaw_deg=float(yaw), pitch_deg=float(pitch))
    X = {o: clip.anchors(o, kind)[a:b + 1] for o in objects}
    if any(np.isnan(x).any() for x in X.values()):
        return None, dict(error="anchor unavailable")
    cams = C[MEASURE[a:b + 1]]
    if task == "object_world_motion_direction":
        delta = R[fa] @ (X[objects[0]][-1] - X[objects[0]][0]) / s
        return direction(delta, TH["object_translation"]), dict(displacement_start_camera_depths=delta.tolist(), displacement_m=(delta * s).tolist())
    if task == "camera_object_distance_change":
        ranges = np.linalg.norm(X[objects[0]] - cams, axis=1)
        diff = ranges[-1] - ranges[0]
        ok = abs(diff) / s >= TH["camera_object_range_abs"] and abs(diff) / ranges[0] >= TH["camera_object_range_relative"]
        return ("farther" if diff > 0 else "closer") if ok else None, dict(ranges_m=ranges.tolist(), delta_depths=diff / s, relative_change=abs(diff) / ranges[0])
    A, B = X[objects[0]], X[objects[1]]
    if task == "object_pair_distance_change":
        dist = np.linalg.norm(A - B, axis=1)
        delta = dist[-1] - dist[0]
        ok = abs(delta) / s >= TH["pair_range_abs"] and abs(delta) / dist[0] >= TH["pair_range_relative"] and dist[0] / s >= TH["pair_initial_separation"]
        return ("increased" if delta > 0 else "decreased") if ok else None, dict(distance_m=dist.tolist(), distance_change_depths=delta / s)
    if task == "object_travel_distance_comparison":
        la, lb = (float(np.linalg.norm(np.diff(x, axis=0), axis=1).sum()) for x in (A, B))
        gap, ratio = abs(la - lb) / s, max(la, lb) / max(min(la, lb), 1e-8)
        ok = gap >= TH["travel_gap"] and ratio >= TH["travel_ratio"] and max(la, lb) / s >= TH["travel_max_min"]
        return ("A" if la > lb else "B") if ok else None, dict(travel_a_m=la, travel_b_m=lb, travel_gap_depths=gap, travel_ratio=ratio)
    raise ValueError(task)


def cue(clip, task, window, objects, value):
    """Whether the targets' image-space change agrees with the 3D answer."""
    a, b = window
    frames = MEASURE[a:b + 1]
    proj = {}
    for o in objects:
        pts = [project(clip.K, clip.R[k], clip.C[k], clip.center[o][k])[0] for k in frames]
        proj[o] = np.array(pts)
    area = {o: clip.masks[o][[a, b]].sum(axis=(1, 2)).astype(float) for o in objects}
    if task == "object_world_motion_direction":
        du, dv = proj[objects[0]][-1] - proj[objects[0]][0]
        ratio = area[objects[0]][1] / max(area[objects[0]][0], 1)
        image = {"left": -du, "right": du, "up": -dv, "down": dv, "forward": 1 - ratio, "backward": ratio - 1}[value]
        return ("consistent" if image > 0 else "conflict"), dict(image_du_px=float(du), image_dv_px=float(dv), mask_area_ratio=float(ratio))
    if task == "object_travel_distance_comparison":
        lengths = {o: float(np.linalg.norm(np.diff(proj[o], axis=0), axis=1).sum()) for o in objects}
        winner = "A" if lengths[objects[0]] > lengths[objects[1]] else "B"
        return ("consistent" if winner == value else "conflict"), dict(image_path_px=lengths)
    if task == "camera_object_distance_change":
        ratio = area[objects[0]][1] / max(area[objects[0]][0], 1)
        return ("consistent" if (ratio > 1) == (value == "closer") else "conflict"), dict(mask_area_ratio=float(ratio))
    if task == "object_pair_distance_change":
        d = [np.linalg.norm(proj[objects[0]][i] - proj[objects[1]][i]) for i in (0, -1)]
        return ("consistent" if (d[1] > d[0]) == (value == "increased") else "conflict"), dict(image_separation_px=[float(x) for x in d])
    return None, {}


def motion_group(clip):
    """Real-split group bits from exact GT over the nine measurement frames."""
    C, R, s = clip.C[MEASURE], clip.R[MEASURE], clip.scale
    translation = float(np.linalg.norm(np.diff(C, axis=0), axis=1).sum() / s)
    rotation = float(sum(math.degrees(math.acos(np.clip((np.trace(R[i + 1] @ R[i].T) - 1) / 2, -1, 1))) for i in range(len(R) - 1)))
    diag = math.hypot(clip.width, clip.height)
    dynamics = 0.0
    for i in range(len(MEASURE) - 1):
        k0, k1 = MEASURE[i], MEASURE[i + 1]
        moves = [np.linalg.norm(project(clip.K, clip.R[k1], clip.C[k1], clip.center[o][k1])[0] - project(clip.K, clip.R[k1], clip.C[k1], clip.center[o][k0])[0])
                 for o in clip.stencil if clip.masks[o][i].sum() > 150]
        dynamics += max(moves, default=0.0) / diag
    bits = dict(translation_high=bool(translation >= 0.20), rotation_high=bool(rotation >= 40.0), dynamic_high=bool(dynamics >= 0.12))
    group = "".join("H" if bits[k] else "L" for k in ["translation_high", "rotation_high", "dynamic_high"])
    return group, dict(translation_path_depths=translation, rotation_path_deg=rotation, object_image_motion_diagonals=dynamics, **bits)


def question(task, window, labels, desc):
    a, b = window
    if task == "camera_translation_direction":
        return f"From F{a} to F{b}, in which dominant direction did the camera center move? Use the camera axes frozen at F{a}: right/down/forward are +x/+y/+z.", AXES
    if task == "camera_view_direction_change":
        return f"From F{a} to F{b}, toward which direction did the camera optical axis turn, using the camera axes frozen at F{a}?", AXES[:4]
    if task == "object_world_motion_direction":
        return f"After compensating for camera motion from F{a} to F{b}, in which dominant direction did object {labels[0]} ({desc[0]}) move? Use the camera axes frozen at F{a}: right/down/forward are +x/+y/+z.", AXES
    if task == "camera_object_distance_change":
        return f"From F{a} to F{b}, did object {labels[0]} ({desc[0]}) become closer to or farther from the camera? Compare 3D Euclidean range.", ["closer", "farther"]
    if task == "object_pair_distance_change":
        return f"From F{a} to F{b}, did the 3D distance between object A ({desc[0]}) and object B ({desc[1]}) increase or decrease? Compare their separation at the two specified frames.", ["increased", "decreased"]
    return (f"Over the sampled trajectory from F{a} through F{b}, which object traveled a greater cumulative 3D distance after compensating for camera motion: "
            f"A ({desc[0]}) or B ({desc[1]})? Sum consecutive sampled 3D displacements, rather than using only the start-to-end displacement."), ["A", "B"]


def build(output, only=None, qa_name="qa"):
    output = Path(output)
    plan = json.loads((output / "plan.json").read_text())
    if only:
        plan["clips"] = [c for c in plan["clips"] if c["id"] in only]
    qa_dir = output / qa_name
    qa_dir.mkdir(exist_ok=True)
    records, audit, clips_info = [], [], []
    for spec in plan["clips"]:
        clip = Clip(output, spec)
        vid = "synth_pilot__" + clip.id
        case = qa_dir / "cases" / vid
        case.mkdir(parents=True, exist_ok=True)
        for i, k in enumerate(MEASURE):
            image = cv2.imread(str(clip.folder / "rgb" / ("frame_%06d.png" % k)))
            if image is None or image.shape[:2] != (clip.height, clip.width):
                raise RuntimeError("Missing RGB frame %d for %s" % (k, clip.id))
            cv2.imwrite(str(case / f"rgb_{i}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        oid = {key: f"{clip.id}__{key}" for key in clip.stencil}
        np.savez_compressed(case / "instance_masks.npz", **{oid[k]: m.astype(np.uint8) for k, m in clip.masks.items()})
        group, group_evidence = motion_group(clip)
        gt_file = qa_dir / "gt" / (vid + ".npz")
        gt_file.parent.mkdir(exist_ok=True)
        extrinsics = np.concatenate([clip.R, (-np.einsum("nij,nj->ni", clip.R, clip.C))[:, :, None]], axis=2)
        np.savez_compressed(gt_file, extrinsics_w2c=extrinsics, intrinsics=clip.K, depth_m=np.array(clip.depth, np.float16),
                            instance_ids=np.array(clip.ids, np.uint8), measure_frames=np.array(MEASURE),
                            **{"center_" + k: v for k, v in clip.center.items()})
        # GT consistency: projected 3D centre should fall inside the object's own mask.
        inside, total = 0, 0
        for key in clip.stencil:
            for i, k in enumerate(MEASURE):
                if clip.masks[key][i].sum() < 150:
                    continue
                (u, v), z = project(clip.K, clip.R[k], clip.C[k], clip.center[key][k])
                total += 1
                if z > 0 and 0 <= int(v) < clip.height and 0 <= int(u) < clip.width:
                    ys, xs = np.nonzero(clip.masks[key][i])
                    inside += int(np.min(np.hypot(xs + 0.5 - u, ys + 0.5 - v)) <= 3.0)
        info = dict(id=clip.id, video_id=vid, env=spec["env"], split=spec["split"], pair=spec.get("pair"), scale_m=clip.scale, group=group,
                    group_evidence=group_evidence, center_in_mask=[inside, total],
                    mask_pixels={k: clip.masks[k].sum(axis=(1, 2)).tolist() for k in clip.stencil},
                    collisions=len(clip.ev["collisions"]), line_of_sight_blocked=len(clip.ev["line_of_sight_blocked"]))
        clips_info.append(info)
        for n, d in enumerate(spec["design"]):
            objects = d["objects"]
            labels = objects
            result = dict(clip=clip.id, task=d["task"], window=d["window"], objects=objects, design_answer=d["answer"], design_cue=d["cue"])
            values = {}
            for kind in (["center", "visible"] if objects else ["center"]):
                values[kind] = answer(clip, d["task"], d["window"], objects, kind)
            gt_value, evidence = values["center"]
            a, b = d["window"]
            visible = all((clip.eroded[o][a:b + 1].sum(axis=(1, 2)) >= TH["min_mask_pixels"]).all() for o in objects)
            agree = all(v[0] == gt_value for v in values.values())
            measured_cue, cue_evidence = cue(clip, d["task"], d["window"], objects, gt_value) if gt_value and objects else (None, {})
            gates = dict(effect=gt_value is not None, visible=bool(visible), anchors_agree=bool(agree), matches_design=gt_value == d["answer"])
            # The cue is the point of a counterfactual pair; elsewhere it is only recorded.
            if spec["split"] == "counterfactual":
                gates["cue_matches_design"] = d["cue"] is None or measured_cue == d["cue"]
            result.update(gt_answer=gt_value, visible_anchor_answer=values.get("visible", (None,))[0], gates=gates, measured_cue=measured_cue)
            audit.append(result)
            if not all(gates.values()):
                continue
            desc = [clip.actors[o]["description"] for o in objects]
            text, options = question(d["task"], d["window"], labels, desc)
            opts = list(options)
            random.Random(SEED + len(records) + sum(map(ord, vid))).shuffle(opts)
            frames = list(range(a, b + 1)) if d["task"] == "object_travel_distance_comparison" else [a, b]
            records.append(dict(
                qa_id=f"{vid}__{d['task']}__{n:02d}", video_id=vid, task=d["task"], question=text, answer=gt_value, options=opts,
                answer_index=opts.index(gt_value), frame_indices=frames,
                frame_references=[dict(frame_index=i, rendered_frame_index=MEASURE[i], normalized_time=MEASURE[i] / (FRAMES - 1),
                                       image_path=f"cases/{vid}/rgb_{i}.jpg") for i in frames],
                object_ids=[oid[o] for o in objects], target_mapping={l: oid[o] for l, o in zip("AB", objects)} if objects else {},
                target_descriptions={l: clip.actors[o]["description"] for l, o in zip("AB", objects)},
                evidence=dict(evidence, anchor_answers={k: v[0] for k, v in values.items()}, scale_m=clip.scale, cue=cue_evidence),
                thresholds=TH, answer_source="exact_Unreal_render_ground_truth", units="sequence background depths (also metres in evidence)",
                position_definition="evaluated body-centre bone; visible-surface centroid from GT depth+mask must give the same answer",
                split="synthetic_pilot", sub_split=spec["split"], pair_id=spec.get("pair"), cue=measured_cue, environment=spec["env"],
                annotation_status="exact_render_ground_truth_automated_gates", quality_status="provisional_automated_pass",
                group=group, manual_review=False, localization_source="Unreal custom-stencil visibility pass", geometry_thresholds_unchanged=True))
    save_json(qa_dir / "qa.json", records)
    save_json(qa_dir / "design_audit.json", audit)
    save_json(qa_dir / "clips.json", clips_info)
    return records, audit, clips_info


def prepare_eval(output):
    """Render marked F0-F8 model inputs with motion_audit's own evaluator."""
    env = dict(os.environ, TEMPORAL_VLM_OUT=str(output / "eval"), TEMPORAL_VLM_SOURCES=json.dumps([str(output / "qa")]))
    subprocess.run([str(AUDIT / ".venv/bin/python"), "-c", "from qa_pipeline.evaluation.runner import prepare; print(len(prepare()))"],
                   cwd=AUDIT, env=env, check=True)


def run(output, action):
    output = Path(output)
    if action == "qa":
        records, audit, clips = build(output)
        prepare_eval(output)
        return dict(status="built", qa=len(records), designed=len(audit), clips=len(clips))
    import pilot_report
    return pilot_report.write(output)


class DryClip:
    """Planned trajectories only (stage-local frame), for checking designs before rendering."""

    def __init__(self, design, scale_m):
        from superdyn_ue.math3d import compose
        cam = design["cam"]
        mats = [compose(dict(location_cm=[cam["f"][k] * 100, cam["r"][k] * 100, cam["h"][k] * 100],
                             rotation_rpy_deg=[cam["roll"][k], cam["pitch"][k], cam["yaw"][k]], scale=[1, 1, 1])) for k in range(FRAMES)]
        cams = [camera_rh(m) for m in mats]
        self.R = np.array([c[0] for c in cams])
        self.C = np.array([c[1] for c in cams])
        self.scale = scale_m
        self.center = {}
        from pilot_stage import target_height
        for spec in design["actors"]:
            if not spec["label"]:
                continue
            h = spec["h"] if spec["h"] is not None else np.full(FRAMES, target_height(spec))
            self.center[spec["label"]] = np.array([point_rh([f * 100, r * 100, z * 100]) for f, r, z in zip(spec["f"], spec["r"], h)])

    def anchors(self, key, kind):
        return self.center[key][MEASURE]


def framing(design, width=1280, height=720, margin=0.03):
    """Planned in-frame fraction and minimum projected height (px) of each labelled target."""
    from pilot_design import ASSETS
    clip = DryClip(design, 10.0)
    K = intrinsics(design["cam"]["fov_deg"], width, height)
    out = {}
    for spec in design["actors"]:
        if not spec["label"]:
            continue
        a = ASSETS[spec["asset"]]
        size = a.get("height_m", a.get("length_m", 1.0) * 0.6)
        inside, heights = 0, []
        for k in MEASURE:
            (u, v), z = project(K, clip.R[k], clip.C[k], clip.center[spec["label"]][k])
            ok = z > 0.3 and margin * width <= u <= (1 - margin) * width and margin * height <= v <= (1 - margin) * height
            inside += ok
            heights.append(K[1, 1] * size / max(z, 1e-3))
        out[spec["label"]] = dict(in_frame=inside / len(MEASURE), min_height_px=float(min(heights)))
    return out


def dry_run(scales=(8.0, 15.0, 25.0)):
    from pilot_design import designs
    rows = []
    for design in designs():
        for scale in scales:
            clip = DryClip(design, scale)
            for d in design["design"]:
                value, evidence = answer(clip, d["task"], d["window"], d["objects"], "center")
                rows.append(dict(clip=design["id"], scale=scale, task=d["task"], objects=d["objects"], design=d["answer"], planned=value, ok=value == d["answer"]))
    return rows
