"""Host-side input validation, engine discovery and reproducible process launch."""

import ctypes.util, gzip, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Content/Python"))
from superdyn_ue.math3d import compose
from superdyn_ue.io_utils import save_json, load_scene, sha256


def engine_binary(value=None):
    values = [
        value,
        os.environ.get("SUPERDYN_UE_ENGINE"),
        os.environ.get("UE_ROOT"),
        str(ROOT / ".engine"),
    ]
    for raw in values:
        if not raw:
            continue
        path = Path(raw).expanduser().resolve()
        for p in [
            path,
            path / "Engine/Binaries/Linux/UnrealEditor",
            path / "Engine/Binaries/Linux/UnrealEditor-Cmd",
        ]:
            if (
                p.is_file()
                and p.name in ["UnrealEditor", "UnrealEditor-Cmd"]
                and os.access(p, os.X_OK)
            ):
                return p
        if path == ROOT / ".engine" and path.exists():
            for p in path.glob("*/Engine/Binaries/Linux/UnrealEditor"):
                if os.access(p, os.X_OK):
                    return p
    for name in ["UnrealEditor", "UnrealEditor-Cmd"]:
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


def runtime_environment():
    env = os.environ.copy()
    local_paths = [
        ROOT / ".deps/sysroot/usr/lib/x86_64-linux-gnu",
        ROOT / ".deps/sysroot/lib/x86_64-linux-gnu",
    ]
    available = [str(path) for path in local_paths if path.exists()]
    if available:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(available) + (
            os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
        )
    # Keep Unreal's native per-user Zen/DDC so compiled shaders survive runs.
    return env


def doctor(value=None):
    executable = engine_binary(value)
    missing_libraries = []
    native_evidence = []
    if executable:
        dependencies = subprocess.run(
            ["ldd", str(executable)],
            env=runtime_environment(),
            capture_output=True,
            text=True,
        )
        missing_libraries = sorted(
            set(
                line.strip()
                for line in dependencies.stdout.splitlines()
                if "not found" in line
            )
        )
        build_file = executable.parents[2] / "Build/Build.version"
        if build_file.exists():
            version = json.loads(build_file.read_text())
            prefix = (
                ".".join(
                    str(version[key])
                    for key in ["MajorVersion", "MinorVersion", "PatchVersion"]
                )
                + "-"
            )
            for native_path in (ROOT / "outputs").glob(
                "*/*/native_validation.json"
            ):
                folder = native_path.parent
                validation = json.loads(native_path.read_text())
                receipt_path = folder / "verification.json"
                if receipt_path.exists() and (folder / "sample.mp4").exists():
                    receipt = json.loads(receipt_path.read_text())
                    if (
                        validation.get("status") == "passed"
                        and validation.get("unreal_engine_version", validation.get("engine", "")).startswith(
                            prefix
                        )
                        and receipt.get("status") == "passed"
                        and receipt.get("movie_sha256") == sha256(folder / "sample.mp4")
                    ):
                        native_evidence.append(str(folder.relative_to(ROOT)))
    probe = subprocess.run(
        [sys.executable, str(ROOT / "scripts/vulkan_probe.py")],
        env=runtime_environment(),
        capture_output=True,
        text=True,
    )
    try:
        vulkan = json.loads(probe.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        vulkan = dict(status="unavailable", error=probe.stderr[-1000:])
    result = dict(
        status=(
            "native_render_verified"
            if native_evidence and not missing_libraries
            else (
                "ready_for_engine_smoke_test"
                if executable
                and vulkan.get("status") == "passed"
                and not missing_libraries
                else (
                    "engine_required"
                    if not executable
                    else "runtime_dependencies_missing"
                )
            )
        ),
        engine=str(executable) if executable else None,
        engine_installed=executable is not None,
        unreal_execution_verified=bool(native_evidence),
        native_render_evidence=native_evidence,
        missing_libraries=missing_libraries,
        vulkan=vulkan,
        official_download="https://www.unrealengine.com/en-US/linux",
        next_step=(
            "View the final movies and report in outputs/ab_walkthroughs/."
            if native_evidence
            else (
                "Provide an existing UnrealEditor path or an official Linux engine ZIP; Epic login is required for the official download."
                if not executable
                else "Run a native build/preview to verify the editor and API version."
            )
        ),
    )
    save_json(ROOT / "reports/environment_status.json", result)
    return result


def validate_inputs():
    import numpy as np

    manifest = json.loads((ROOT / "inputs/manifest.json").read_text())
    records = []
    for name, expected in manifest["files"].items():
        if sha256(ROOT / "inputs" / name) != expected:
            raise ValueError("Prepared input hash mismatch: " + name)
    for case in manifest["cases"]:
        packet = load_scene(ROOT / "inputs" / case["case"] / "scene.json.gz")
        n = packet["frames"]
        keys = packet["actor_keys"]
        count = 0
        assert (
            packet["schema"] == 1
            and n == 192
            and packet["fps"] == 24
            and len(packet["camera"]) == n
        )
        assert keys == ["person", "ground", "air"]
        normal_agreement = []
        for mesh in packet["meshes"]:
            assert mesh["parent"] is None or mesh["parent"] in keys
            if mesh["key"] != "terrain":
                assert len(packet["tracks"][mesh["key"]]) == n
            for section in mesh["sections"]:
                v = np.array(section["vertices_cm"])
                normals = np.array(section["normals"])
                uv = np.array(section["uv0"])
                tris = np.array(section["triangles"], int)
                assert (
                    np.isfinite(v).all()
                    and np.isfinite(normals).all()
                    and v.shape == normals.shape
                    and uv.shape == (len(v), 2)
                )
                assert tris.shape[1] == 3 and tris.min() >= 0 and tris.max() < len(v)
                cross = np.cross(
                    v[tris[:, 1]] - v[tris[:, 0]], v[tris[:, 2]] - v[tris[:, 0]]
                )
                sizes = np.linalg.norm(cross, axis=1)
                mask = sizes > 1e-5
                dots = (
                    np.sum(cross[mask] * normals[tris[mask, 0]], axis=1) / sizes[mask]
                )
                normal_agreement.extend(dots.tolist())
                count += len(tris)
        # Smooth bevel normals can differ from a geometric face normal; they must
        # still be on the outward side after the handedness conversion.
        assert min(normal_agreement) > -0.01
        for poses in [packet["camera"], *packet["tracks"].values()]:
            for pose in poses:
                assert (
                    np.isfinite(np.array(compose(pose))).all()
                    and min(pose["scale"]) > 0
                )
        assert packet["conversion_validation"]["max_projection_error_px"] < 0.01
        records.append(
            dict(
                case=case["case"],
                frames=n,
                meshes=len(packet["meshes"]),
                triangles=count,
                max_projection_error_px=packet["conversion_validation"][
                    "max_projection_error_px"
                ],
            )
        )
    report = dict(
        status="passed",
        scope="offline geometry/coordinate/schema verification; not an Unreal engine execution test",
        cases=len(records),
        frames=sum(r["frames"] for r in records),
        unreal_execution_verified=False,
        results=records,
    )
    save_json(ROOT / "reports/input_validation.json", report)
    return report


def command(executable, request_path, gpu):
    entry = ROOT / "Content/Python/run_superdyn.py"
    return [
        str(executable),
        str(ROOT / "SuperDynUnreal.uproject"),
        "-unattended",
        "-nosplash",
        "-nosound",
        "-RenderOffscreen",
        "-vulkan",
        "-graphicsadapter=" + str(gpu),
        "-corelimit=16",
        "-ExecCmds=py '" + str(entry) + "'",
        "-stdout",
        "-FullStdOutLogOutput",
    ]


def launch(
    case,
    output,
    engine=None,
    gpu=6,
    preview=False,
    action="render",
    exr=False,
    rebuild=False,
    spatial_samples=1,
):
    if not 1 <= spatial_samples <= 64:
        raise ValueError("Spatial samples must be between 1 and 64")
    executable = engine_binary(engine)
    if executable is None:
        doctor(engine)
        raise RuntimeError(
            "Unreal Editor is not installed. Use --engine /path/to/UnrealEditor or install an official Linux ZIP. No Unreal render was executed."
        )
    packet = ROOT / "inputs" / case / "scene.json.gz"
    if not packet.exists():
        raise ValueError("Unknown/unprepared case: " + case)
    validate_inputs()
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Use an empty output folder")
    output.mkdir(parents=True, exist_ok=True)
    request_path = output / "request.json"
    input_sha = sha256(packet)
    request = dict(
        input=str(packet),
        output=str(output),
        action=action,
        preview=preview,
        exr=exr,
        rebuild=rebuild,
        spatial_samples=spatial_samples,
        input_sha256=input_sha,
    )
    save_json(request_path, request)
    source_files = [
        ROOT / "manage.py",
        ROOT / "bridge.py",
        ROOT / "SuperDynUnreal.uproject",
        *(ROOT / "Config").glob("*.ini"),
        *(ROOT / "Content/Python").rglob("*.py"),
        ROOT / "inputs/official_assets/license_and_source.json",
    ]
    source_hashes = {}
    for source in source_files:
        relative = source.relative_to(ROOT)
        destination = output / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        source_hashes[str(relative)] = sha256(source)
    save_json(
        output / "source_receipt.json",
        dict(input_sha256=input_sha, source_files=source_hashes),
    )
    env = runtime_environment()
    env["SUPERDYN_UE_REQUEST"] = str(request_path)
    cmd = command(executable, request_path, gpu)
    save_json(output / "launch.json", dict(command=cmd, engine=str(executable)))
    with (output / "editor.log").open("w") as log:
        result = subprocess.run(
            cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
        )
    status_path = output / "render_status.json"
    if result.returncode or not status_path.exists():
        raise RuntimeError(
            "Editor did not complete successfully; inspect "
            + str(output / "editor.log")
        )
    status = json.loads(status_path.read_text())
    if status["status"] not in ["built", "frames_complete"]:
        raise RuntimeError("Native build/render failed: " + str(status))
    if action == "render":
        if sha256(packet) != input_sha:
            raise RuntimeError("Input package changed during rendering")
        shader_failures = [
            line
            for line in (output / "editor.log").read_text(errors="replace").splitlines()
            if "Failed to compile Material /Game/SuperDyn/" in line
        ]
        if shader_failures:
            save_json(
                status_path,
                dict(
                    status="failed",
                    unreal_render_verified=False,
                    error="Generated material fell back to the default shader",
                    shader_failures=shader_failures,
                ),
            )
            raise RuntimeError(
                "Generated material shader compilation failed; inspect "
                + str(output / "editor.log")
            )
        native = json.loads((output / "native_validation.json").read_text())
        if native.get("status") != "passed":
            raise RuntimeError("Native validation did not pass")
        if preview:
            frames = verify_frame_sequence(output, load_scene(packet), preview=True)
            status = dict(
                status="preview_passed",
                unreal_execution_verified=True,
                unreal_render_verified=True,
                frames=len(frames),
                video_created=False,
            )
            save_json(output / "render_status.json", status)
        else:
            encode_and_verify(output, load_scene(packet))
            status = json.loads(status_path.read_text())
    return status


def verify_frame_sequence(output, packet, preview=False):
    import cv2

    output = Path(output)
    paths = sorted((output / "rgb").glob("*.png"))
    step = max(1, packet["frames"] // 3) if preview else 1
    expected = [f"frame_{i:06d}.png" for i in range(0, packet["frames"], step)]
    if [p.name for p in paths] != expected:
        raise RuntimeError(
            "Unreal image sequence is incomplete or numbered unexpectedly"
        )
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.shape[:2] != (packet["height"], packet["width"]):
            raise RuntimeError(
                "Unreal image is unreadable or has the wrong dimensions: " + str(path)
            )
    return paths


def encode_and_verify(output, packet, *, input_path=None):
    import cv2, imageio_ffmpeg

    output = Path(output)
    provenance = Path(input_path) if input_path is not None else ROOT / "inputs" / packet["case_id"] / "scene.json.gz"
    input_hash = sha256(provenance)
    paths = verify_frame_sequence(output, packet)
    movie = output / "sample.mp4"
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-framerate",
        str(packet["fps"]),
        "-start_number",
        "0",
        "-i",
        str(output / "rgb/frame_%06d.png"),
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(movie),
    ]
    with (output / "encode.log").open("w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    cap = cv2.VideoCapture(str(movie))
    count = 0
    while cap.read()[0]:
        count += 1
    cap.release()
    if count != packet["frames"]:
        raise RuntimeError("Encoded video has an unexpected frame count")
    save_json(
        output / "verification.json",
        dict(
            status="passed",
            case=packet["case_id"],
            input_sha256=input_hash,
            renderer="Unreal Engine Movie Render Queue",
            unreal_execution_verified=True,
            frames=count,
            fps=packet["fps"],
            seconds=count / packet["fps"],
            movie_sha256=sha256(movie),
            image_sha256={p.name: sha256(p) for p in paths},
        ),
    )
    save_json(
        output / "render_status.json",
        dict(
            status="passed",
            unreal_render_verified=True,
            frames=count,
            video="sample.mp4",
        ),
    )
