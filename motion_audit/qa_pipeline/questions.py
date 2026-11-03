"""Six deterministic QA tasks; semantic labels never determine an answer."""

import random
import numpy as np
from . import tracking as t


def record_key(r):
    return f"{r['video_id']}__{'_'.join(r['object_ids'])}__F{r['start']}-F{r['end']}"


def rank(r):
    return (
        int(r["pair_distance_pass"]) + int(r["travel_comparison_pass"]),
        r["normalized_span"],
        min(r["track_counts"]),
    )


def make_qas(r):
    vid = r["video_id"]
    p, source = t.base.load(vid)
    e = p["extrinsics"].astype(float)
    scale = source["depth_scale"]
    centers = -np.einsum("tji,tj->ti", e[:, :, :3], e[:, :, 3]) / scale
    a, b = r["start"], r["end"]
    qas = []

    def add(task, question, answer, options, frames, ids, evidence):
        opts = list(options)
        random.Random(t.SEED + len(qas) + sum(map(ord, vid))).shuffle(opts)
        qas.append(
            dict(
                qa_id=f"{vid}__{task}__{len(qas):02d}",
                video_id=vid,
                task=task,
                question=question,
                answer=answer,
                options=opts,
                answer_index=opts.index(answer),
                frame_indices=frames,
                frame_references=[
                    dict(
                        t.base.reference(i, source, p),
                        image_path=f"cases/{vid}/rgb_{i}.jpg",
                    )
                    for i in frames
                ],
                object_ids=ids,
                target_mapping=dict(zip("AB", r["object_ids"])),
                target_descriptions=dict(zip("AB", r["descriptions"])),
                evidence=evidence,
                thresholds=t.TH,
                answer_source="deterministic_VGGT_Omega_geometry_with_RAFT_large_tracks",
                position_definition="centroid of the same surviving core surface points; model estimate, not object center of mass",
                units="sequence background depths, not meters",
                split="development",
                annotation_status="model_estimated_numeric_checks_only",
            )
        )

    windows = [
        (i, j)
        for i in range(9)
        for j in range(i + 2, 9)
        if p["times"][j] - p["times"][i] >= t.TH["min_time_span"]
        and all(x["quality_valid"] for x in source["intervals"][i:j])
    ]
    windows.sort(key=lambda ab: -(p["times"][ab[1]] - p["times"][ab[0]]))
    axes = ["left", "right", "up", "down", "backward", "forward"]
    for i, j in windows:
        delta = (centers[j] - centers[i]) @ e[i, :, :3].T
        answer = t.direction(delta, t.TH["camera_translation"])
        if answer:
            add(
                t.TASKS[0],
                f"From F{i} to F{j}, in which dominant direction did the camera center move? Use the camera axes frozen at F{i}: right/down/forward are +x/+y/+z.",
                answer,
                axes,
                [i, j],
                [],
                dict(
                    displacement_start_camera_depths=delta.tolist(),
                    formula="(C_end-C_start) @ R_start.T / background_depth",
                ),
            )
            break
    for i, j in windows:
        optical = e[j, 2, :3] @ e[i, :, :3].T
        yaw, pitch = np.degrees(
            [
                np.arctan2(optical[0], optical[2]),
                np.arctan2(optical[1], np.hypot(optical[0], optical[2])),
            ]
        )
        big, small = sorted([abs(yaw), abs(pitch)], reverse=True)
        if (
            big >= t.TH["camera_turn_deg"]
            and big >= t.TH["direction_dominance"] * small
            and big - small >= t.TH["camera_turn_gap_deg"]
        ):
            answer = (
                ("right" if yaw > 0 else "left")
                if abs(yaw) > abs(pitch)
                else ("down" if pitch > 0 else "up")
            )
            add(
                t.TASKS[1],
                f"From F{i} to F{j}, toward which direction did the camera optical axis turn, using the camera axes frozen at F{i}?",
                answer,
                axes[:4],
                [i, j],
                [],
                dict(
                    yaw_deg=float(yaw),
                    pitch_deg=float(pitch),
                    formula="R_end[2,:] @ R_start.T; optical-axis yaw and pitch, excluding roll",
                ),
            )
            break
    for label, key, bootkey, oid, desc in zip(
        "AB",
        ["anchors_a", "anchors_b"],
        ["bootstrap_a", "bootstrap_b"],
        r["object_ids"],
        r["descriptions"],
    ):
        anchor = np.array(r[key])
        boot = np.array(r[bootkey])
        delta = (anchor[-1] - anchor[0]) @ e[a, :, :3].T
        answer = t.direction(delta, t.TH["object_translation"])
        db = (boot[:, -1] - boot[:, 0]) @ e[a, :, :3].T
        axis = int(np.argmax(abs(delta)))
        ci = np.quantile(db[:, axis], [0.025, 0.975])
        signed = (
            ci[0] >= t.TH["bootstrap_gap"]
            if delta[axis] > 0
            else ci[1] <= -t.TH["bootstrap_gap"]
        )
        agreement = (
            float(
                np.mean(
                    [t.direction(v, t.TH["object_translation"]) == answer for v in db]
                )
            )
            if answer
            else 0.0
        )
        if answer and signed and agreement >= t.TH["bootstrap_direction_agreement"]:
            add(
                t.TASKS[2],
                f"After compensating for camera motion from F{a} to F{b}, in which dominant direction did object {label} ({desc}) move? Use the camera axes frozen at F{a}: right/down/forward are +x/+y/+z.",
                answer,
                axes,
                [a, b],
                [oid],
                dict(
                    displacement_start_camera_depths=delta.tolist(),
                    dominant_component_bootstrap95=ci.tolist(),
                    bootstrap_direction_agreement=agreement,
                    min_bootstrap_direction_agreement=0.95,
                ),
            )
        cam = np.array(r["camera_centers"])
        ranges = np.linalg.norm(anchor - cam, axis=-1)
        diff = float(ranges[-1] - ranges[0])
        br = np.linalg.norm(boot - cam, axis=-1)
        ci = np.quantile(br[:, -1] - br[:, 0], [0.025, 0.975])
        signed = (
            ci[0] >= t.TH["bootstrap_gap"]
            if diff > 0
            else ci[1] <= -t.TH["bootstrap_gap"]
        )
        if (
            abs(diff) >= t.TH["camera_object_range_abs"]
            and abs(diff) / ranges[0] >= t.TH["camera_object_range_relative"]
            and signed
        ):
            add(
                t.TASKS[3],
                f"From F{a} to F{b}, did object {label} ({desc}) become closer to or farther from the camera? Compare 3D Euclidean range.",
                "farther" if diff > 0 else "closer",
                ["closer", "farther"],
                [a, b],
                [oid],
                dict(
                    ranges_depths=ranges.tolist(),
                    delta_depths=diff,
                    relative_change=abs(diff) / float(ranges[0]),
                    range_change_bootstrap95=ci.tolist(),
                ),
            )
    m = r["metrics"]
    shared = dict(
        metrics=m,
        record_key=record_key(r),
        anchors_a=r["anchors_a"],
        anchors_b=r["anchors_b"],
        track_counts=r["track_counts"],
        initial_track_counts=r["initial_track_counts"],
        track_survival=r["track_survival"],
        mask_overlap=r["mask_overlap"],
        tracking=r["tracking"],
        position_definition="arithmetic mean of the same persistent surface-core points",
    )
    if r["pair_distance_pass"]:
        add(
            t.TASKS[4],
            f'From F{a} to F{b}, did the 3D distance between object A ({r["descriptions"][0]}) and object B ({r["descriptions"][1]}) increase or decrease? Compare their separation at the two specified frames.',
            "increased" if m["distance_change_depths"] > 0 else "decreased",
            ["increased", "decreased"],
            [a, b],
            r["object_ids"],
            shared,
        )
    if r["travel_comparison_pass"]:
        add(
            t.TASKS[5],
            f'Over the sampled trajectory from F{a} through F{b}, which object traveled a greater cumulative 3D distance after compensating for camera motion: A ({r["descriptions"][0]}) or B ({r["descriptions"][1]})? Sum consecutive sampled 3D displacements, rather than using only the start-to-end displacement.',
            "A" if m["travel_a_depths"] > m["travel_b_depths"] else "B",
            ["A", "B"],
            list(range(a, b + 1)),
            r["object_ids"],
            shared,
        )
    return qas
