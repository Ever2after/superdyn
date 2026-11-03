"""Launch scene-family inspection rendering and verify the native delivery."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bridge import command, engine_binary, runtime_environment, sha256
from scene_families import DEFAULT_OUTPUT, plan, finalize, verify
from superdyn_ue.io_utils import save_json


def render(output, gpu):
    output = Path(output).resolve()
    request = json.loads((output / "request.json").read_text())
    if (output / "manifest.json").exists():
        return verify(output)
    if (output / "editor.log").exists():
        raise RuntimeError("A native attempt already exists. Preserve it and plan to a fresh output directory.")
    executable = engine_binary()
    if executable is None:
        raise RuntimeError("UnrealEditor is unavailable")
    for relative, expected in request["source_hashes"].items():
        if sha256(ROOT.parent / relative) != expected:
            raise RuntimeError("Source changed since planning: " + relative + "; use a fresh plan")
    entry = ROOT / "Content/Python/run_scene_families.py"
    args = ["-ExecCmds=r.RDG.AsyncCompute 0,py '" + str(entry) + "'" if value.startswith("-ExecCmds=") else value for value in command(executable, output / "request.json", gpu)] + ["-norhithread"]
    env = runtime_environment()
    env["SUPERDYN_UE_REQUEST"] = str(output / "request.json")
    save_json(output / "launch.json", dict(command=args, gpu=gpu, entry_sha256=sha256(entry)))
    with (output / "editor.log").open("w") as log:
        process = subprocess.Popen(args, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        save_json(output / "run_status.json", dict(status="running", pid=process.pid, gpu=gpu))
        try:
            started = time.monotonic()
            previous = None
            while process.poll() is None:
                state = output / "native_status.json"
                if state.exists():
                    try:
                        value = json.loads(state.read_text())
                    except ValueError:
                        value = {}
                    marker = (value.get("stage"), value.get("family"), value.get("completed_builds"))
                    if marker != previous:
                        print(json.dumps(value), flush=True)
                        previous = marker
                    if value.get("stage") in ("failed", "render_error"):
                        raise RuntimeError(str(value))
                if time.monotonic() - started > 5400:
                    raise RuntimeError("Native scene-family render exceeded 90 minutes")
                time.sleep(2)
            if process.returncode:
                raise RuntimeError("UnrealEditor exited " + str(process.returncode))
            native = json.loads((output / "native_status.json").read_text())
            if native["stage"] != "frames_complete":
                raise RuntimeError(str(native))
            failures = [line for line in (output / "editor.log").read_text(errors="replace").splitlines() if "Failed to compile Material /Game/SuperDyn/" in line]
            if failures:
                raise RuntimeError("Generated material shader failed: " + str(failures[:3]))
        except BaseException as error:
            save_json(output / "run_status.json", dict(status="failed", error=str(error)))
            raise
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    return finalize(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "render", "verify", "report"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--family", action="append", choices=["industrial", "urban", "park"])
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=5)
    args = parser.parse_args()
    if args.action == "plan":
        result = plan(args.output, args.family, args.seed_offset)
        result = dict(status="planned", scenes=len(result["jobs"]), output=str(args.output))
    elif args.action == "render":
        if not (args.output / "request.json").exists():
            plan(args.output, args.family, args.seed_offset)
        result = render(args.output, args.gpu)
    elif args.action == "verify":
        result = verify(args.output)
    else:
        result = finalize(args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
