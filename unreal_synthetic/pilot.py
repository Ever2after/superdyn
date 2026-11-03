#!/usr/bin/env python3
"""Synthetic-split pilot: survey curated maps, plan clips, render RGB+GT, build QA.

Run with .venv/bin/python from the workspace root or this folder.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from bridge import ROOT, command, engine_binary, runtime_environment, sha256
from superdyn_ue.io_utils import save_json

PILOT_CONFIG = ROOT / "configs/pilot_split.json"
DEFAULT_OUTPUT = ROOT / "outputs/pilot_v1"


def latest_progress(output):
    """Newest status/output file time, so long renders count as progress."""
    newest = 0.0
    for folder in [output / "survey", output / "clips"]:
        if folder.exists():
            for path in folder.rglob("*"):
                if path.is_file():
                    newest = max(newest, path.stat().st_mtime)
    states = list((output / "native").glob("*/native_status.json"))
    return max([newest] + [p.stat().st_mtime for p in states])


def protected_files():
    """Original pack maps used as environments; sessions must never modify them."""
    config = json.loads(PILOT_CONFIG.read_text())
    files = []
    for env in config["environments"] + config.get("excluded_environments", []):
        base = ROOT / "Content" / env["map"].removeprefix("/Game/")
        files += [p for p in [base.with_suffix(".umap"), base.parent / (base.name + "_BuiltData.uasset")] if p.exists()]
    return files


def launch(request, gpu, timeout_s=7200, stall_s=1200, nullrhi=False, attempt_name=None):
    """Run Content/Python/run_pilot.py for one request; never reuse an attempt.

    nullrhi sessions (map loading, traces, actor/sequence building) never touch a
    shared GPU; only render-stage sessions use Vulkan for Movie Render Queue.
    """
    output = Path(request["output"]).resolve()
    attempt = output / "native" / (attempt_name or request["stage"])
    if attempt.exists() and any(attempt.iterdir()):
        raise RuntimeError("A native attempt already exists at " + str(attempt) + "; use a fresh output")
    attempt.mkdir(parents=True, exist_ok=True)
    request = dict(request, output=str(output), status_dir=str(attempt))
    save_json(attempt / "request.json", request)
    executable = engine_binary()
    if executable is None:
        raise RuntimeError("UnrealEditor is unavailable")
    entry = ROOT / "Content/Python/run_pilot.py"
    args = ["-ExecCmds=r.RDG.AsyncCompute 0,py '" + str(entry) + "'" if a.startswith("-ExecCmds=") else a
            for a in command(executable, attempt / "request.json", gpu)] + ["-norhithread"]
    if nullrhi:
        args = [a for a in args if a != "-vulkan" and not a.startswith("-graphicsadapter=")] + ["-nullrhi"]
    env = runtime_environment()
    env["SUPERDYN_UE_REQUEST"] = str(attempt / "request.json")
    save_json(attempt / "launch.json", dict(command=args, gpu=gpu, entry_sha256=sha256(entry)))
    state = attempt / "native_status.json"
    guarded = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in protected_files()}
    started = time.monotonic()
    with (attempt / "editor.log").open("w") as log:
        process = subprocess.Popen(args, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        previous = None
        try:
            while process.poll() is None:
                if state.exists():
                    try:
                        value = json.loads(state.read_text())
                    except ValueError:
                        value = {}
                    marker = (value.get("stage"), value.get("step"), value.get("current"))
                    if marker != previous:
                        print(json.dumps(dict(value, elapsed_s=round(time.monotonic() - started))), flush=True)
                        previous = marker
                    if value.get("stage") in ("failed", "render_error"):
                        raise RuntimeError(json.dumps(value)[:3000])
                if time.monotonic() - started > timeout_s:
                    raise RuntimeError("Pilot editor session exceeded " + str(timeout_s) + " s")
                if time.monotonic() - started > stall_s and time.time() - latest_progress(output) > stall_s:
                    raise RuntimeError("No pilot progress for " + str(stall_s) + " s (shared GPU stall?); see " + str(attempt / "editor.log"))
                time.sleep(2)
            if process.returncode:
                raise RuntimeError("UnrealEditor exited " + str(process.returncode))
            value = json.loads(state.read_text())
            if value["stage"] != "frames_complete":
                raise RuntimeError(json.dumps(value)[:3000])
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    changed = [str(p) for p, sig in guarded.items() if (p.stat().st_size, p.stat().st_mtime_ns) != sig]
    if changed:
        raise RuntimeError("Original environment content was modified: " + str(changed))
    failures = [l for l in (attempt / "editor.log").read_text(errors="replace").splitlines() if "Failed to compile Material /Game/SuperDyn/" in l]
    if failures:
        raise RuntimeError("Generated material failed to compile: " + str(failures[:3]))
    return dict(status="completed", stage=request["stage"], seconds=round(time.monotonic() - started))


def pick_gpu(samples=10):
    """Least-utilised GPU over a short window (shared machine), with memory headroom."""
    totals = {}
    for _ in range(samples):
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout
        for line in out.strip().splitlines():
            index, util, used, total = (float(x) for x in line.split(","))
            if total - used > 30000:
                totals[int(index)] = totals.get(int(index), 0.0) + util
        time.sleep(0.5)
    return min(totals, key=totals.get)


def pick_gpus(count, samples=10):
    """Distinct GPUs ordered by recent utilisation (memory headroom > 30 GB)."""
    totals = {}
    for _ in range(samples):
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout
        for line in out.strip().splitlines():
            index, util, used, total = (float(x) for x in line.split(","))
            if total - used > 30000:
                totals[int(index)] = totals.get(int(index), 0.0) + util
        time.sleep(0.5)
    return sorted(totals, key=totals.get)[:count]


def render_parallel(output, jobs, tag, sessions=4):
    """Split MRQ jobs over several GPU sessions (shared machine: per-GPU throughput is low)."""
    from concurrent.futures import ThreadPoolExecutor
    output = Path(output).resolve()
    gpus = pick_gpus(sessions)
    chunks = [jobs[i::len(gpus)] for i in range(len(gpus))]
    plan = [dict(gpu=g, jobs=[j["name"] for j in c]) for g, c in zip(gpus, chunks) if c]
    print(json.dumps(dict(render_parallel=plan)), flush=True)
    with ThreadPoolExecutor(len(plan)) as pool:
        futures = [pool.submit(launch, dict(stage="render", output=str(output), jobs=c), g, 6 * 3600, 1800, False, "render_%s_%d" % (tag, i))
                   for i, (g, c) in enumerate(zip(gpus, chunks)) if c]
        return [f.result() for f in futures]


def render_deferred(output, gpu, name):
    """GPU session that only runs Movie Render Queue jobs saved by a nullrhi stage."""
    output = Path(output).resolve()
    if gpu < 0:
        gpu = pick_gpu()
        print(json.dumps(dict(render_gpu=gpu)), flush=True)
    jobs = json.loads((output / "native" / name / "render_jobs.json").read_text())
    return launch(dict(stage="render", output=str(output), jobs=jobs), gpu, timeout_s=6 * 3600, attempt_name="render_" + name)


def survey(output, gpu, only=None):
    config = json.loads(PILOT_CONFIG.read_text())
    envs = [e for e in config["environments"] if not only or e["id"] in only]
    request = dict(stage="survey", namespace="/Game/SuperDyn/Pilot/" + Path(output).name, environments=envs, defer_render=True)
    built = launch(dict(request, output=str(output)), gpu, nullrhi=True)
    return dict(survey=built, render=render_deferred(output, gpu, "survey"))


def detail(output, only=None):
    """Fine (0.5 m) rasters over windows chosen from the coarse survey (nullrhi)."""
    from pilot_stage import Survey, windows
    output = Path(output).resolve()
    config = json.loads(PILOT_CONFIG.read_text())
    envs = []
    for env in config["environments"]:
        if only and env["id"] not in only:
            continue
        coarse = json.loads((output / "survey" / (env["id"] + "_summary.json")).read_text())
        survey = Survey(output / "survey" / (env["id"] + ".json.gz"), env.get("exclude_surfaces", ()), env.get("paved_surfaces", ()))
        size = env.get("window_m", 70.0)
        b = coarse["bounds_cm"]
        whole = (b[1] - b[0]) <= 100 * size * 1.6 and (b[3] - b[2]) <= 100 * size * 1.6
        chosen = [b] if whole else windows(survey, size=size, count=env.get("window_count", 4))
        import gzip
        info = json.load(gzip.open(output / "survey" / (env["id"] + ".json.gz"), "rt"))
        envs.append(dict(env, windows=chosen, trace_top_cm=info["trace_top_cm"], trace_bottom_cm=info["trace_bottom_cm"]))
    save_json(output / "survey" / "detail_windows.json", {e["id"]: e["windows"] for e in envs})
    request = dict(stage="survey", namespace="/Game/SuperDyn/Pilot/" + output.name, environments=envs, defer_render=True)
    return launch(dict(request, output=str(output)), -1, nullrhi=True, attempt_name="survey_detail")


def look(output, gpu, only=None, exposure=None):
    """Eye-level look test at each detail window before rendering clips."""
    import gzip
    from pilot_stage import Survey
    output = Path(output).resolve()
    config = json.loads(PILOT_CONFIG.read_text())
    envs = []
    for env in config["environments"]:
        if only and env["id"] not in only:
            continue
        views = []
        for path in sorted((output / "survey").glob(env["id"] + "_detail_*.json.gz")):
            survey = Survey(path, env.get("exclude_surfaces", ()), env.get("paved_surfaces", ()))
            import numpy as np
            j, i = np.unravel_index(np.argmax(survey.clearance), survey.clearance.shape)
            x, y, z = survey.x0 + i * survey.step, survey.y0 + j * survey.step, survey.top[j, i]
            for yaw in [0, 90, 180, 270]:
                views.append(dict(location_cm=[100 * x, 100 * y, 100 * (z + 1.6)], rotation_rpy_deg=[0, -3, yaw], scale=[1, 1, 1]))
            views.append(dict(location_cm=[100 * (x - 18), 100 * y, 100 * (z + 14)], rotation_rpy_deg=[0, -32, 0], scale=[1, 1, 1]))
        envs.append(dict(env, views=views, exposure=exposure if exposure is not None else env.get("exposure")))
    name = "look%d" % len(list((output / "native").glob("look*")))
    request = dict(stage="survey", namespace="/Game/SuperDyn/Pilot/" + output.name + "_" + name, environments=envs, defer_render=True)
    built = launch(dict(request, output=str(output)), gpu, nullrhi=True, attempt_name=name)
    return dict(build=built, render=render_deferred(output, gpu, name))


def exposure_sweep(output, gpu, env_id, settings):
    """Same eye-level views under several camera exposure overrides (one level each)."""
    output = Path(output).resolve()
    config = json.loads(PILOT_CONFIG.read_text())
    env = next(e for e in config["environments"] if e["id"] == env_id)
    views = json.loads((output / "native" / "look2" / "request.json").read_text())
    views = next(e for e in views["environments"] if e["id"] == env_id)["views"][:2]
    envs = [dict(env, id="%s_x%d" % (env_id, k), views=views, exposure=x) for k, x in enumerate(settings)]
    name = "sweep%d" % len(list((output / "native").glob("sweep*")))
    request = dict(stage="survey", namespace="/Game/SuperDyn/Pilot/" + output.name + "_" + name, environments=envs, defer_render=True)
    built = launch(dict(request, output=str(output)), gpu, nullrhi=True, attempt_name=name)
    save_json(output / "native" / name / "settings.json", settings)
    return dict(build=built, render=render_deferred(output, gpu, name))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["survey", "detail", "look", "plan", "render", "qa", "report"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpu", type=int, default=-1, help="-1 picks the least-utilised GPU at render time")
    parser.add_argument("--env", action="append")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.action == "survey":
        result = survey(output, args.gpu, args.env)
    elif args.action == "detail":
        result = detail(output, args.env)
    elif args.action == "look":
        result = look(output, args.gpu, args.env)
    else:
        import pilot_clips
        result = pilot_clips.run(args.action, output, args.gpu, args.env)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
