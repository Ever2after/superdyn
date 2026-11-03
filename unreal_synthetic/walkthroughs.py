"""Verify final walkthroughs without launching Unreal or rewriting render evidence."""

import json
from pathlib import Path

from bridge import ROOT, sha256

DEFAULT_OUTPUT = ROOT / "outputs/ab_walkthroughs"
CASE_IDS = ("01_east_to_west", "02_west_to_east", "03_full_loop")


def read_json(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_case(folder, scene_path, *, decode=True):
    import cv2

    cv2.setNumThreads(1)
    folder = Path(folder)
    receipt = read_json(folder / "verification.json")
    route = read_json(folder / "route.json")
    native = read_json(folder / "native_validation.json")
    packet = route["packet"]
    require(receipt["status"] == native["status"] == "passed", f"Failed case: {folder.name}")
    require(receipt["case"] == packet["case_id"] == folder.name, "Case ID mismatch")
    require(receipt["frames"] == packet["frames"] == 960, "Expected 960 frames")
    require(receipt["fps"] == packet["fps"] == 24 and receipt["seconds"] == 40, "Timing mismatch")
    require(len(packet["camera"]) == 960, "Camera trajectory length mismatch")
    require(sha256(folder / "sample.mp4") == receipt["movie_sha256"], "Movie hash mismatch")
    require(sha256(folder / "route.json") == receipt["route_sha256"], "Route hash mismatch")
    require(sha256(scene_path) == receipt["scene_manifest_sha256"], "Scene manifest hash mismatch")
    require(receipt["input_sha256"] == sha256(ROOT / "inputs/03_low_orbit/scene.json.gz"), "Base terrain input hash mismatch")
    expected = [f"frame_{i:06d}.png" for i in range(960)]
    frames = sorted((folder / "rgb").glob("*.png"))
    require([p.name for p in frames] == expected, "Missing or unexpected RGB frames")
    require(set(receipt["image_sha256"]) == set(expected), "Frame receipt mismatch")
    for frame in frames:
        require(sha256(frame) == receipt["image_sha256"][frame.name], f"Frame hash mismatch: {frame}")
    poses = packet["camera"]
    vertical_step = max(abs(a["location_cm"][2] - b["location_cm"][2]) for a, b in zip(poses, poses[1:]))
    require(abs(vertical_step - route["max_vertical_step_cm"]) < 1e-6 and vertical_step <= 12, "Camera crosses a height discontinuity")
    require(native["terrain_support_error_cm"] <= 2 and native["camera_evaluation_error_cm"] <= .1, "Native camera checks failed")
    require(native["eye_height_cm"] == route["eye_height_cm"] == 165, "Eye height mismatch")
    cap = cv2.VideoCapture(str(folder / "sample.mp4"))
    require(cap.isOpened(), "Movie cannot be opened")
    try:
        dimensions = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        require(dimensions == [packet["width"], packet["height"]] == [1280, 720], "Movie dimensions mismatch")
        require(abs(cap.get(cv2.CAP_PROP_FPS) - 24) < 1e-6, "Movie FPS mismatch")
        count = 0
        if decode:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                require(list(frame.shape[:2]) == [720, 1280], "Decoded frame dimensions mismatch")
                count += 1
            require(count == 960, "Movie decode frame count mismatch")
        else:
            require(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 960, "Movie frame metadata mismatch")
    finally:
        cap.release()
    return dict(id=folder.name, frames=960, seconds=40, resolution=dimensions,
                path_length_m=route["path_length_cm"] / 100, eye_height_cm=165,
                max_vertical_step_cm=vertical_step, movie_sha256=receipt["movie_sha256"],
                decoded_frames=count if decode else None)


def verify_delivery(output=DEFAULT_OUTPUT, *, decode=True):
    output = Path(output).resolve()
    manifest = read_json(output / "manifest.json")
    scene = read_json(output / "scene_manifest.json")
    require(manifest["status"] == "passed", "Final manifest is not passed")
    require([c["id"] for c in manifest["cases"]] == list(CASE_IDS), "Unexpected tour set")
    require(scene["logical_objects"] == len(scene["objects"]) == 51, "Placement count mismatch")
    require(len(scene["source_packs"]) == 14 and "unfinished_building" not in scene["source_packs"], "Source selection mismatch")
    results = [verify_case(output / case, output / "scene_manifest.json", decode=decode) for case in CASE_IDS]
    for case, result in zip(manifest["cases"], results):
        require(case["movie_sha256"] == result["movie_sha256"], "Manifest movie hash mismatch")
    for relative, expected in manifest["source_hashes"].items():
        require(sha256(output / "source_snapshot" / relative) == expected, f"Original source snapshot changed: {relative}")
    for relative, expected in manifest["saved_scene_assets_sha256"].items():
        require(sha256(ROOT / relative) == expected, f"Saved Unreal asset changed: {relative}")
    return dict(status="passed", movies=3, frames=2880, decoded_frames=sum(r["decoded_frames"] or 0 for r in results),
                image_hashes_checked=2880, source_snapshot_files=len(manifest["source_hashes"]),
                saved_scene_assets=len(manifest["saved_scene_assets_sha256"]), cases=results)
