"""Launch native A/B walkthroughs; completed deliveries are verified before reuse."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bridge import command, engine_binary, runtime_environment, verify_frame_sequence, encode_and_verify
from superdyn_ue.io_utils import save_json as save
from walkthroughs import DEFAULT_OUTPUT, read_json, verify_case
import plan_ab_walkthroughs as planner

OUT = DEFAULT_OUTPUT


def start(path, logpath, gpu):
    entry = ROOT / "Content/Python/run_ab_walkthroughs.py"
    executable = engine_binary()
    if executable is None:
        raise RuntimeError("UnrealEditor is unavailable; run manage.py doctor")
    cmd = ["-ExecCmds=r.RDG.AsyncCompute 0,py '" + str(entry) + "'" if value.startswith("-ExecCmds=")
           else value for value in command(executable, path, gpu)] + ["-norhithread"]
    env = runtime_environment()
    env["SUPERDYN_UE_REQUEST"] = str(path)
    save(logpath.with_suffix(".launch.json"), dict(command=cmd, entry_sha256=hashlib.sha256(entry.read_bytes()).hexdigest()))
    log = logpath.open("w")
    try:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    except BaseException:
        log.close()
        raise
    return proc, log


def stop(proc):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def wait(proc, status_path, stages, timeout=900):
    began = time.monotonic()
    while time.monotonic() - began < timeout:
        if status_path.exists():
            try:
                result = read_json(status_path)
            except ValueError:
                result = {}
            if result.get("stage") == "failed":
                raise RuntimeError(str(result))
            if result.get("stage") in stages:
                return result
        if proc.poll() is not None:
            raise RuntimeError("Native editor exited early: " + str(proc.returncode))
        time.sleep(2)
    raise RuntimeError("Native stage timeout")


def preview(gpu, reuse=False):
    if (OUT / "manifest.json").exists() or any(OUT.glob("*/sample.mp4")):
        raise RuntimeError("Completed delivery is preserved. Use --output with a fresh experiment directory.")
    if reuse:
        request = read_json(OUT / "request.json")
        request["action"] = "reload_preview"
        save(OUT / "request.json", request)
    else:
        if OUT.exists() and any(OUT.iterdir()):
            raise RuntimeError("Use a fresh output directory, or retry_preview for a prepared preview")
        planner.initial_request()
    state = OUT / "native_status.json"
    state.unlink(missing_ok=True)
    proc, log = start(OUT / "request.json", OUT / "preview_editor.log", gpu)
    try:
        if not reuse:
            result = wait(proc, state, {"scene_built"})
            manifest = read_json(OUT / "scene_manifest.json")
            print(json.dumps(result), flush=True)
            cases = planner.make_routes(manifest)
        else:
            manifest = read_json(OUT / "scene_manifest.json")
            cases = read_json(OUT / "request.json")["cases"]
        wait(proc, state, {"preview_complete"}, 1800)
        proc.wait(timeout=60)
        if proc.returncode:
            raise RuntimeError("Preview process exit " + str(proc.returncode))
        for case in cases:
            folder = Path(case["output"])
            packet = read_json(folder / "route.json")["packet"]
            frames = verify_frame_sequence(folder / "preview", packet, preview=True)
            save(folder / "preview_verification.json", dict(status="passed", frames=len(frames)))
        save(OUT / "preview_verification.json", dict(status="passed", cases=len(cases), objects=manifest["logical_objects"], source_packs=len(manifest["source_packs"])))
    finally:
        stop(proc)
        log.close()


def render(gpu, only=None):
    request = read_json(OUT / "request.json")
    if only and only not in {case["id"] for case in request["cases"]}:
        raise ValueError("Unknown case: " + only)
    for case in request["cases"]:
        if only and case["id"] != only:
            continue
        folder = OUT / case["id"]
        packet = read_json(folder / "route.json")["packet"]
        if (folder / "verification.json").exists():
            verify_case(folder, OUT / "scene_manifest.json")
            print("Verified existing " + case["id"], flush=True)
            continue
        if any((folder / "rgb").glob("*.png")):
            raise RuntimeError("Partial render exists; use a fresh directory after reviewing it")
        state = folder / "native_status.json"
        state.unlink(missing_ok=True)
        native_case = dict(case, output=str(folder))
        save(folder / "request.json", dict(ready=True, action="render", output=str(folder), map_path=request["map_path"], cases=[native_case]))
        proc, log = start(folder / "request.json", folder / "editor.log", gpu)
        try:
            wait(proc, state, {"frames_complete"}, 1800)
            proc.wait(timeout=60)
            if proc.returncode:
                raise RuntimeError("Render process exit " + str(proc.returncode))
        finally:
            stop(proc)
            log.close()
        encode_and_verify(folder, packet, input_path=ROOT / "inputs/03_low_orbit/scene.json.gz")
        receipt = read_json(folder / "verification.json")
        receipt.update(route_sha256=hashlib.sha256((folder / "route.json").read_bytes()).hexdigest(),
                       scene_manifest_sha256=hashlib.sha256((OUT / "scene_manifest.json").read_bytes()).hexdigest())
        save(folder / "verification.json", receipt)
        print("Completed " + case["id"], flush=True)
    results = []
    for case in request["cases"]:
        receipt = OUT / case["id"] / "verification.json"
        if receipt.exists():
            value = read_json(receipt)
            results.append(dict(id=case["id"], **{key: value[key] for key in ["status", "frames", "seconds", "movie_sha256"]}))
    save(OUT / "render_manifest.json", dict(status="passed" if len(results) == len(request["cases"]) else "in_progress", expected_cases=len(request["cases"]), cases=results))


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["preview", "retry_preview", "render"])
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--case")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    OUT = args.output.resolve()
    planner.OUT = OUT
    if args.action in ["preview", "retry_preview"]:
        preview(args.gpu, args.action == "retry_preview")
    else:
        render(args.gpu, args.case)


if __name__ == "__main__":
    main()
