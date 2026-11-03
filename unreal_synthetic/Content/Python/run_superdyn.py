"""Stage build, validation and rendering across editor ticks for async asset loading."""

import importlib
import json
import os
from pathlib import Path
import traceback
import unreal
from superdyn_ue.io_utils import load_scene, save_json, sha256
from superdyn_ue.scene_builder import build, initialize_terrain_collision
from superdyn_ue.audit import audit_scene
from superdyn_ue.render_queue import render

_ROOT = Path(__file__).resolve().parents[2]
_REQUEST = Path(os.environ.get("SUPERDYN_UE_REQUEST", str(_ROOT / "run_request.json")))
_HANDLE = None
_RETRY_MTIME = None
_PENDING = None
_TICKS_LEFT = 0


def unregister():
    global _HANDLE
    if _HANDLE is not None:
        unreal.unregister_slate_post_tick_callback(_HANDLE)
        _HANDLE = None


def fail(request, output, error):
    global _HANDLE, _RETRY_MTIME, _PENDING
    _PENDING = None
    save_json(
        output / "render_status.json",
        dict(
            status="failed",
            error=str(error),
            traceback=traceback.format_exc(),
            unreal_render_verified=False,
        ),
    )
    unreal.log_error(traceback.format_exc())
    if request.get("development_retry", False):
        _RETRY_MTIME = _REQUEST.stat().st_mtime_ns
        unreal.log_warning("SuperDyn development mode: update request.json to retry.")
        _HANDLE = unreal.register_slate_post_tick_callback(wait_for_retry)
    else:
        unreal.SystemLibrary.quit_editor()


def wait_for_retry(_delta=0.0):
    global _HANDLE, build, audit_scene, render, initialize_terrain_collision
    if _REQUEST.stat().st_mtime_ns == _RETRY_MTIME:
        return
    unregister()
    for name in ["materials", "asset_layout", "official_assets", "scene_builder", "audit", "render_queue"]:
        importlib.reload(importlib.import_module("superdyn_ue." + name))
    builder = importlib.import_module("superdyn_ue.scene_builder")
    build = builder.build
    initialize_terrain_collision = builder.initialize_terrain_collision
    audit_scene = importlib.import_module("superdyn_ue.audit").audit_scene
    render = importlib.import_module("superdyn_ue.render_queue").render
    _HANDLE = unreal.register_slate_post_tick_callback(run)


def run(_delta=0.0):
    global _HANDLE, _PENDING, _TICKS_LEFT
    if unreal.AssetRegistryHelpers.get_asset_registry().is_loading_assets():
        return
    unregister()
    request = json.loads(_REQUEST.read_text())
    output = Path(request["output"])
    output.mkdir(parents=True, exist_ok=True)
    try:
        version = unreal.SystemLibrary.get_engine_version()
        major, minor = map(int, version.split("-")[0].split(".")[:2])
        if major != 5 or minor < 6:
            raise RuntimeError("Target is UE 5.6 or newer; engine is " + version)
        for name in [
            "GeometryScript_MeshEdits",
            "GeometryScriptSimpleMeshBuffers",
            "GeometryScript_NewAssetUtils",
            "MoviePipelineQueueSubsystem",
        ]:
            if not hasattr(unreal, name):
                raise RuntimeError("Required built-in plugin API is missing: " + name)
        input_sha = sha256(request["input"])
        if request.get("input_sha256", input_sha) != input_sha:
            raise RuntimeError("Input package changed after launch")
        save_json(
            output / "render_status.json",
            dict(status="building", unreal_render_verified=False),
        )
        save_json(
            output / "job_status.json",
            dict(stage="building_assets", engine_version=version),
        )
        scene = build(
            load_scene(request["input"]),
            _ROOT,
            input_sha,
            rebuild=request.get("rebuild", False),
        )
        _PENDING = (scene, request, output, version)
        _TICKS_LEFT = 3
        save_json(
            output / "job_status.json",
            dict(stage="waiting_for_asset_initialization", engine_version=version),
        )
        _HANDLE = unreal.register_slate_post_tick_callback(validate_ready_scene)
    except Exception as error:
        fail(request, output, error)


def validate_ready_scene(_delta=0.0):
    global _HANDLE, _TICKS_LEFT
    _TICKS_LEFT -= 1
    if (
        _TICKS_LEFT > 0
        or unreal.AssetRegistryHelpers.get_asset_registry().is_loading_assets()
    ):
        return
    unregister()
    scene, request, output, version = _PENDING
    try:
        initialize_terrain_collision(scene["actors"])
        save_json(
            output / "job_status.json",
            dict(stage="native_validation", engine_version=version),
        )
        audit_scene(scene, output)
        unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).save_current_level()
        if request["action"] == "build":
            save_json(
                output / "render_status.json",
                dict(
                    status="built",
                    unreal_build_verified=True,
                    unreal_render_verified=False,
                ),
            )
            unreal.SystemLibrary.quit_editor()
        else:
            # Let Sequencer window teardown and rendering-resource updates settle.
            _HANDLE = unreal.register_slate_post_tick_callback(begin_render)
    except Exception as error:
        fail(request, output, error)


def begin_render(_delta=0.0):
    unregister()
    scene, request, output, version = _PENDING
    try:
        save_json(
            output / "job_status.json", dict(stage="rendering", engine_version=version)
        )
        render(
            scene,
            output,
            request.get("preview", False),
            request.get("exr", False),
            request.get("spatial_samples", 1),
        )
    except Exception as error:
        fail(request, output, error)


_HANDLE = unreal.register_slate_post_tick_callback(run)
