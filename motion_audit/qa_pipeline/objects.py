"""Full GMOS masks -> local geometry gates -> semantic-only Qwen labels."""

from concurrent.futures import ThreadPoolExecutor
import base64, functools, hashlib, io, json, os, sys, time, urllib.request
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from . import tracking as t
from .config import ROOT

GMOS_CHECKPOINT = ROOT / "checkpoints/gmos/gmos.pth"


@functools.lru_cache(maxsize=1)
def checkpoint_digest():
    return hashlib.sha256(GMOS_CHECKPOINT.read_bytes()).hexdigest()


def configure(out):
    t.OUT = Path(out)
    t.OUT.mkdir(parents=True, exist_ok=True)
    t.base.OUT = t.OUT


def sync():
    import torch

    torch.cuda.synchronize()


def bbox_from_mask(mask):
    """Exclusive xmax/ymax; all coordinates are normalized to the image grid."""
    yy, xx = np.where(mask)
    if not len(xx):
        return None
    h, w = mask.shape
    return [
        int(np.floor(xx.min() * 1000 / w)),
        int(np.floor(yy.min() * 1000 / h)),
        int(np.ceil((xx.max() + 1) * 1000 / w)),
        int(np.ceil((yy.max() + 1) * 1000 / h)),
    ]


def context_rgb(vid):
    p, r = t.base.load(vid)
    h, w = p["depth"].shape[1:]
    paths = sorted(
        x
        for x in (Path(r["source_path"]) / "rgb").iterdir()
        if x.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    ids = np.rint(np.linspace(0, len(paths) - 1, 33)).astype(int)
    assert ids[::4].tolist() == r["selected_frame_indices"]
    frames = []
    for idx in ids:
        im = Image.open(paths[idx]).convert("RGB")
        ow, oh = im.size
        if oh / ow < 0.5:
            cw = min(ow, round(oh / 0.5))
            left = (ow - cw) // 2
            im = im.crop((left, 0, left + cw, oh))
        elif oh / ow > 2:
            ch = min(oh, round(ow * 2))
            top = (oh - ch) // 2
            im = im.crop((0, top, ow, top + ch))
        frames.append(np.array(im.resize((w, h), Image.Resampling.BICUBIC)))
    return np.stack(frames), ids


class GMOSRunner:
    def __init__(self, batch_size=4):
        import torch
        from safetensors.torch import load_file

        torch.set_num_threads(2)
        sys.path.insert(0, str(ROOT / "vendor/gmos"))
        sys.path.insert(0, str(ROOT / "vendor/gmos/core/sam2"))
        import inference_gmos as official
        from core.pi3_encoder import Pi3Encoder
        from core.gmos_proposer import GMOSProposer
        from sam2.build_sam import build_sam2_video_predictor

        self.torch = torch
        self.official = official
        self.batch_size = batch_size
        self.pi3 = Pi3Encoder().eval().cuda()
        state = self.pi3.load_state_dict(
            load_file(str(ROOT / "checkpoints/gmos/pi3/model.safetensors")),
            strict=False,
        )
        assert not state.missing_keys, state.missing_keys
        self.gmos = GMOSProposer().eval().cuda()
        w = torch.load(GMOS_CHECKPOINT, map_location="cpu", weights_only=False)[
            "model_state_dict"
        ]
        state = self.gmos.load_state_dict(
            {k.removeprefix("module."): v for k, v in w.items()}, strict=False
        )
        assert not state.missing_keys and set(state.unexpected_keys) <= {
            "pe_layer.positional_encoding_gaussian_matrix"
        }, state
        self.predictor = build_sam2_video_predictor(
            "configs/sam2.1/sam2.1_hiera_l.yaml",
            str(ROOT / "checkpoints/gmos/sam2.1_hiera_large.pt"),
            device="cuda",
        )
        # Match the existing installation's behavior without the optional CUDA
        # connected-components extension; no implicit warning-driven fallback.
        self.predictor.fill_hole_area = 0
        official.patch_predictor_cache_additive(self.predictor)

    def infer(self, vid, frames):
        import gc

        official = self.official
        torch = self.torch
        cache = {}
        timing = {}
        sync()
        start = time.perf_counter()
        with torch.inference_mode():
            result = official.run_proposer(
                None,
                self.pi3,
                self.predictor,
                self.gmos,
                torch.device("cuda"),
                batch_size=self.batch_size,
                sam_feat_cache=cache,
                stride=1,
                preloaded_video=frames,
                seq_name=vid,
            )
        sync()
        timing["proposer_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        prop = official.build_propagator_inputs(
            result["seq_annos"], result["pred_motions"], result["pred_ious"]
        )
        timing["proposal_handoff_seconds"] = time.perf_counter() - start
        sync()
        start = time.perf_counter()
        prev = torch.backends.cuda.matmul.allow_tf32
        prevcu = torch.backends.cudnn.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                on, off = official.run_propagator(
                    self.predictor,
                    None,
                    frames.shape,
                    prop,
                    torch.device("cuda"),
                    sam_feat_cache=cache,
                )
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev
            torch.backends.cudnn.allow_tf32 = prevcu
        sync()
        timing["propagator_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        mids = np.arange(0, 33, 4)
        h, w = frames.shape[1:3]
        full = off["full_segments"] if off is not None else {}
        object_ids = sorted({oid for frame in full.values() for oid in frame})
        masks = {}
        scores = {}
        motions = {}
        for oid in object_ids:
            key = f"g{int(oid)+1}"
            m = np.stack(
                [
                    np.asarray(full.get(int(i), {}).get(oid, np.zeros((1, h, w), bool)))
                    .reshape(h, w)
                    .astype(bool)
                    for i in mids
                ]
            )
            if not m.any():
                continue
            masks[key] = m
            values = []
            moving = []
            for i in mids:
                value = np.asarray(
                    on.get("object_ious", {}).get(int(i), {}).get(oid, [0.0])
                ).reshape(-1)
                values.append(
                    float(value[0]) if len(value) and np.isfinite(value[0]) else 0.0
                )
                value = np.asarray(
                    off.get("motions_alter", {}).get(int(i), {}).get(oid, [0.0])
                ).reshape(-1)
                moving.append(
                    float(value[0]) if len(value) and np.isfinite(value[0]) else 0.0
                )
            scores[key] = values
            motions[key] = moving
        # Never sum overlapping IDs into a fictitious new label. Ambiguous pixels
        # are excluded from every instance and hence from geometry/point tracks.
        if masks:
            overlap = np.sum(np.stack(list(masks.values())), axis=0) > 1
            masks = {k: (m & ~overlap).astype("uint8") for k, m in masks.items()}
        timing["mask_pack_seconds"] = time.perf_counter() - start
        del cache, result, prop, on, off
        gc.collect()
        return masks, scores, motions, timing


def mask_stage(ids, batch_size=4, runner=None):
    import torch

    load = 0.0
    if runner is None:
        start = time.perf_counter()
        runner = GMOSRunner(batch_size)
        sync()
        load = time.perf_counter() - start
    t.base.save(
        t.OUT / "thresholds.json",
        dict(
            thresholds=t.TH,
            tasks=t.TASKS,
            geometry_thresholds_unchanged=True,
            mask_quality="stage-1 SAM2 predicted mask IoU within GMOS, cutoff 0.8; not calibrated to legacy image-predictor score",
        ),
    )
    rows = []
    for vid in ids:
        target = t.OUT / "cases" / vid
        target.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        frames, source_indices = context_rgb(vid)
        prepare = time.perf_counter() - start
        sync()
        start = time.perf_counter()
        masks, scores, motions, timing = runner.infer(vid, frames)
        sync()
        elapsed = time.perf_counter() - start
        start = time.perf_counter()
        temporary = target / "instance_masks.tmp.npz"
        np.savez_compressed(temporary, **masks)
        temporary.replace(target / "instance_masks.npz")
        objects = []
        for key, mask in masks.items():
            objects.append(
                dict(
                    object_id=key,
                    description=f"tracked object [{key}]",
                    category="unlabeled",
                    identity_source="GMOS video track ID",
                    frames=[
                        dict(frame_index=i, bbox=bbox_from_mask(mask[i]))
                        for i in range(9)
                    ],
                )
            )
        t.base.save(target / "objects.json", objects)
        t.base.save(
            target / "segmentation.json",
            dict(
                model="GMOS multi-object proposer + two-stage SAM2 video propagator",
                checkpoint=str(GMOS_CHECKPOINT),
                output="MOS/full_segments",
                scores=scores,
                score_type="stage1 SAM2 predicted mask IoU",
                score_threshold=t.TH["sam_score"],
                motion_scores=motions,
                bbox_source="mask min/max, no VLM localization",
                instance_id_source="GMOS propagator",
                context_frames=33,
                measurement_context_indices=list(range(0, 33, 4)),
                source_indices=source_indices.tolist(),
                standalone_sam2_refinement=False,
                qwen_localization=False,
                mask_is_ground_truth=False,
            ),
        )
        for i, frame in enumerate(frames[::4]):
            Image.fromarray(frame).save(target / f"rgb_{i}.jpg", quality=95)
        write = time.perf_counter() - start
        row = dict(
            video_id=vid,
            prepare_seconds=prepare,
            mask_seconds=elapsed,
            mask_write_seconds=write,
            objects=len(objects),
            **timing,
        )
        rows.append(row)
        t.base.save(
            t.OUT / "mask_timings.json",
            dict(
                status="running",
                model_load_seconds=load,
                per_video=rows,
                batch_size=batch_size,
            ),
        )
        print("GMOS_MASK", row, flush=True)
    t.base.save(
        t.OUT / "mask_timings.json",
        dict(
            status="completed",
            model_load_seconds=load,
            per_video=rows,
            batch_size=batch_size,
            timing="CUDA synchronized; includes proposer, internal propagator, CPU handoff and packing; no Qwen or separate SAM2 pass",
            gmos_checkpoint_sha256=checkpoint_digest(),
        ),
    )


def camera_record(vid):
    return dict(
        video_id=vid,
        start=0,
        end=8,
        object_ids=[],
        descriptions=[],
        metrics={},
        anchors_a=[],
        anchors_b=[],
        bootstrap_a=[],
        bootstrap_b=[],
        camera_centers=[],
        track_counts=[],
        initial_track_counts=[],
        track_survival=[],
        mask_overlap=0.0,
        tracking="none",
        pair_distance_pass=False,
        travel_comparison_pass=False,
    )


def geometry_stage(ids):
    from .questions import make_qas, rank

    rows = []
    qas = []
    selected = []
    for vid in ids:
        folder = t.OUT / "cases" / vid
        if not (folder / "dense_flow.npz").is_file():
            raise FileNotFoundError("Run RAFT preprocessing before geometry: " + vid)
        start = time.perf_counter()
        records = t.scan_video(vid, include_nonpassing=True, save_single_tracks=True)
        geometry = time.perf_counter() - start
        start = time.perf_counter()
        candidates = records or t.base.read(folder / "single_object_tracks.json")
        r = max(candidates, key=rank) if candidates else camera_record(vid)
        selected.append(r)
        qq = make_qas(r)
        for q in qq:
            q["annotation_status"] = "model_estimated_numeric_pass_needs_visual_review"
        qas.extend(qq)
        rows.append(
            dict(
                video_id=vid,
                geometry_seconds=geometry,
                qa_seconds=time.perf_counter() - start,
                qa_count=len(qq),
                objects_for_semantics=sorted(
                    {oid for q in qq for oid in q["object_ids"]}
                ),
            )
        )
        print("GMOS_GEOMETRY", rows[-1], flush=True)
    t.base.save(t.OUT / "selected_records.json", selected)
    t.base.save(t.OUT / "qa_before_semantics.json", qas)
    t.base.save(
        t.OUT / "geometry_timings.json", dict(status="completed", per_video=rows)
    )


def semantic_payload(vid, r, object_ids):
    a, b = r["start"], r["end"]
    frames = sorted({a, (a + b) // 2, b})
    folder = t.OUT / "cases" / vid
    content = [
        dict(
            type="text",
            text=SEMANTIC_PROMPT + "\nRequested IDs: " + ", ".join(object_ids),
        )
    ]
    with np.load(folder / "instance_masks.npz") as f:
        masks = {k: f[k] for k in object_ids}
    palette = [(255, 112, 30), (10, 230, 130), (255, 80, 220), (70, 170, 255)]
    for i in frames:
        im = Image.open(folder / f"rgb_{i}.jpg").convert("RGB")
        d = ImageDraw.Draw(im)
        for j, oid in enumerate(object_ids):
            mask = masks[oid][i].astype("uint8")
            color = palette[j % len(palette)]
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                xy = [tuple(xy) for xy in contour[:, 0].tolist()]
                if len(xy) > 1:
                    d.line(xy + [xy[0]], fill=color, width=2)
            yy, xx = np.where(mask)
            if len(xx):
                x, y = int(np.median(xx)), int(np.median(yy))
                d.rectangle((x, y, x + 28, y + 16), fill=color)
                d.text((x + 2, y + 2), oid, fill="black")
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=92)
        content.extend(
            [
                dict(type="text", text=f"Frame {i}"),
                dict(
                    type="image_url",
                    image_url=dict(
                        url="data:image/jpeg;base64,"
                        + base64.b64encode(buf.getvalue()).decode()
                    ),
                ),
            ]
        )
    return content, frames


def semantic_stage(ids, port, *, continue_on_error=False, on_case_complete=None):
    from .questions import make_qas

    selected = {r["video_id"]: r for r in t.base.read(t.OUT / "selected_records.json")}
    before = t.base.read(t.OUT / "qa_before_semantics.json")
    rows = []
    qa = []
    (t.OUT / "semantic_prompt.txt").write_text(SEMANTIC_PROMPT)
    for _ in range(120):
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{port}/v1/models", timeout=2
            ).close()
            break
        except Exception:
            time.sleep(5)
    else:
        raise RuntimeError("Qwen startup timeout")

    def one(vid):
        r = selected[vid]
        folder = t.OUT / "cases" / vid
        requested = sorted(
            {oid for q in before if q["video_id"] == vid for oid in q["object_ids"]}
        )
        if not requested:
            return (
                dict(
                    video_id=vid,
                    qwen_seconds=0.0,
                    prepare_seconds=0.0,
                    objects=0,
                    frames=[],
                    skipped="No object QA survived geometry gates",
                ),
                {},
            )
        start = time.perf_counter()
        content, frames = semantic_payload(vid, r, requested)
        pre = time.perf_counter() - start
        payload = dict(
            model="temporal-object-qwen",
            messages=[dict(role="user", content=content)],
            temperature=0,
            max_tokens=600,
            seed=t.SEED,
            chat_template_kwargs=dict(enable_thinking=False),
            response_format=dict(type="json_object"),
        )
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=600) as response:
            raw = json.load(response)
        elapsed = time.perf_counter() - start
        t.base.save(folder / "semantic_qwen_raw.json", raw)
        result = json.loads(raw["choices"][0]["message"]["content"])
        labels = {o["object_id"]: o for o in result["objects"]}
        assert set(labels) == set(
            requested
        ), "Semantic model must preserve the supplied IDs exactly"
        assert all(
            set(o)
            == {
                "object_id",
                "category",
                "description",
                "identity_confidence",
                "single_entity",
            }
            for o in labels.values()
        ), "Localization/extra fields are prohibited"
        for o in labels.values():
            assert (
                o["category"]
                and o["description"]
                and o["identity_confidence"] in ["high", "medium", "low"]
                and isinstance(o["single_entity"], bool)
            )
        t.base.save(
            folder / "objects_semantics.json",
            dict(
                annotation=result,
                frames=frames,
                usage=raw.get("usage"),
                localization_source="GMOS only; Qwen has no localization fields",
                prompt_sha256=hashlib.sha256(SEMANTIC_PROMPT.encode()).hexdigest(),
                elapsed_seconds=elapsed,
            ),
        )
        return (
            dict(
                video_id=vid,
                qwen_seconds=elapsed,
                prepare_seconds=pre,
                frames=frames,
                objects=len(requested),
                usage=raw.get("usage"),
            ),
            labels,
        )

    def safe_one(vid):
        start = time.perf_counter()
        try:
            return one(vid)
        except Exception as exc:
            if not continue_on_error:
                raise
            row = dict(
                video_id=vid,
                error=f"{type(exc).__name__}: {exc}",
                failed_request_elapsed_seconds=time.perf_counter() - start,
                qwen_seconds=0.0,
                prepare_seconds=0.0,
                objects=0,
                frames=[],
            )
            t.base.save(t.OUT / "cases" / vid / "semantic_error.json", row)
            return row, {}

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(safe_one, ids))
    wall = time.perf_counter() - start
    assembly_start = time.perf_counter()
    for row, labels in results:
        vid = row["video_id"]
        r = dict(selected[vid])
        objects = t.base.read(t.OUT / "cases" / vid / "objects.json")
        for o in objects:
            if o["object_id"] in labels:
                o["semantic_annotation"] = labels[o["object_id"]]
                o["description"] = labels[o["object_id"]]["description"]
                o["category"] = labels[o["object_id"]]["category"]
        t.base.save(t.OUT / "cases" / vid / "objects.json", objects)
        descriptions = {o["object_id"]: o["description"] for o in objects}
        r["descriptions"] = [descriptions[k] for k in r["object_ids"]]
        selected[vid] = r
        qs = make_qas(r)
        for q in qs:
            valid = all(
                oid in labels
                and labels[oid]["identity_confidence"] == "high"
                and labels[oid]["single_entity"]
                for oid in q["object_ids"]
            )
            q.update(
                annotation_status="model_estimated_automated_checks_only_needs_visual_review",
                quality_status=(
                    "provisional_automated_pass"
                    if valid
                    else "semantic_review_required"
                ),
                localization_source="GMOS MOS tracks",
                geometry_thresholds_unchanged=True,
            )
            if row.get("error") and q["object_ids"]:
                q["semantic_error"] = row["error"]
        t.base.save(t.OUT / "cases" / vid / "qa_generated.json", qs)
        passed = [q for q in qs if q["quality_status"] == "provisional_automated_pass"]
        t.base.save(t.OUT / "cases" / vid / "qa.json", passed)
        if on_case_complete:
            on_case_complete(row, qs, passed)
        qa.extend(qs)
        rows.append(row)
        print("GMOS_SEMANTIC", row, flush=True)
    t.base.save(t.OUT / "qa_generated.json", qa)
    t.base.save(t.OUT / "selected_records.json", list(selected.values()))
    t.base.save(
        t.OUT / "qa.json",
        [q for q in qa if q["quality_status"] == "provisional_automated_pass"],
    )
    (t.OUT / "qa.jsonl").write_text(
        "".join(
            json.dumps(q, ensure_ascii=False) + "\n"
            for q in qa
            if q["quality_status"] == "provisional_automated_pass"
        )
    )
    t.base.save(
        t.OUT / "semantic_timings.json",
        dict(
            status="completed",
            wall_seconds=wall,
            final_qa_assembly_seconds=time.perf_counter() - assembly_start,
            concurrency=2,
            per_video=rows,
            policy="Only objects with numeric QA after geometry filtering; 2-3 annotated keyframes; max_tokens=600; no bbox or identity tracking generation",
        ),
    )


SEMANTIC_PROMPT = """Identify only the objects marked by the supplied track IDs in these ordered frames.
Colored contours and ID labels are supplied by a video segmentation model.
Do not generate bounding boxes, masks, tracking IDs, 3D geometry, or QA answers.
For each requested ID, return category, a short uniquely identifying visible
description, identity_confidence (high/medium/low), and single_entity (true/false).
Use low confidence or single_entity=false for merged/different entities.
Do not infer an identity for an invisible target. Return only JSON:
{"objects":[{"object_id":"g1","category":"person","description":"person in red shirt",
"identity_confidence":"high","single_entity":true}]}"""
