"""Build/audit all theme maps, then render them in one native editor session."""

import importlib
import json
import os
from pathlib import Path
import traceback
import unreal
from superdyn_ue.io_utils import load_scene, save_json, sha256
from superdyn_ue.scene_builder import build, initialize_terrain_collision
from superdyn_ue.audit import audit_scene
from superdyn_ue.render_queue import add_job
from superdyn_ue.themes import signature
from superdyn_ue.enclosure_audit import validate as validate_enclosure

ROOT = Path(__file__).resolve().parents[2]
REQUEST_PATH = Path(os.environ["SUPERDYN_UE_REQUEST"])
REQUEST = json.loads(REQUEST_PATH.read_text())
HANDLE = None
INDEX = 0
PENDING = None
JOBS = []
TICKS = 0
EXECUTOR = None
FAILED = False
RETRY_MTIME = None


def schedule(function):
    global HANDLE
    if HANDLE is not None:
        unreal.unregister_slate_post_tick_callback(HANDLE)
    HANDLE = unreal.register_slate_post_tick_callback(function)


def unschedule():
    global HANDLE
    if HANDLE is not None:
        unreal.unregister_slate_post_tick_callback(HANDLE)
        HANDLE = None


def progress(stage, **extra):
    save_json(REQUEST["batch_status"], dict(stage=stage, completed_builds=INDEX, total=len(REQUEST["jobs"]), **extra))


def fail(error):
    global RETRY_MTIME
    unschedule()
    progress("failed", error=str(error), traceback=traceback.format_exc())
    unreal.log_error(traceback.format_exc())
    if REQUEST.get("development_retry", False):
        RETRY_MTIME = REQUEST_PATH.stat().st_mtime_ns
        schedule(retry)
    else:
        unreal.SystemLibrary.quit_editor()


def retry(_delta):
    global REQUEST, INDEX, PENDING, JOBS, build, initialize_terrain_collision, audit_scene, add_job, signature, validate_enclosure
    if REQUEST_PATH.stat().st_mtime_ns == RETRY_MTIME:
        return
    unschedule()
    REQUEST = json.loads(REQUEST_PATH.read_text())
    for name in ["materials", "theme_design", "theme_materials", "enclosure", "hdri", "themes", "enclosure_audit", "official_assets", "scene_builder", "audit", "render_queue"]:
        importlib.reload(importlib.import_module("superdyn_ue." + name))
    builder = importlib.import_module("superdyn_ue.scene_builder")
    build, initialize_terrain_collision = builder.build, builder.initialize_terrain_collision
    audit_scene = importlib.import_module("superdyn_ue.audit").audit_scene
    add_job = importlib.import_module("superdyn_ue.render_queue").add_job
    signature = importlib.import_module("superdyn_ue.themes").signature
    validate_enclosure = importlib.import_module("superdyn_ue.enclosure_audit").validate
    INDEX, PENDING, JOBS = 0, None, []
    schedule(build_next)


def build_next(_delta):
    global PENDING, TICKS
    if unreal.AssetRegistryHelpers.get_asset_registry().is_loading_assets():
        return
    unschedule()
    try:
        if INDEX == len(REQUEST["jobs"]):
            begin_render()
            return
        job = REQUEST["jobs"][INDEX]
        progress("building", theme=job["theme"])
        packet = load_scene(REQUEST["input"])
        if sha256(REQUEST["input"]) != REQUEST["input_sha256"]:
            raise RuntimeError("Base input changed")
        PENDING = build(packet, ROOT, REQUEST["input_sha256"], rebuild=REQUEST.get("rebuild", True), theme=job["theme"])
        TICKS = 3
        schedule(validate)
    except Exception as error:
        fail(error)


def validate(_delta):
    global INDEX, PENDING, TICKS
    TICKS -= 1
    if TICKS > 0 or unreal.AssetRegistryHelpers.get_asset_registry().is_loading_assets():
        return
    unschedule()
    try:
        job = REQUEST["jobs"][INDEX]
        output = Path(job["output"])
        initialize_terrain_collision(PENDING["actors"])
        report = audit_scene(PENDING, output)
        settings = PENDING["theme_settings"]
        decorations = [a for a in PENDING["ignored"] if a.get_actor_label().startswith("SD_Theme_")]
        enclosure=validate_enclosure(PENDING,unreal.get_editor_subsystem(unreal.EditorActorSubsystem))
        save_json(output / "theme_scene.json", dict(
            status="passed", theme_signature=signature(job["theme"], ROOT),
            decorations=len(decorations), settings=settings,
            enclosure_validation=enclosure,
            scope="Four closed boundary walls, no background props, actual HDRI bindings and original camera/object GT",
        ))
        report["theme"] = job["theme"]
        report["theme_decorations"] = len(decorations)
        report["enclosure_validation"]=enclosure
        save_json(output / "native_validation.json", report)
        unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).save_current_level()
        JOBS.append(dict(packet=PENDING["packet"], map_path=PENDING["map_path"], sequence_path=PENDING["sequence_path"], output=str(output)))
        PENDING = None
        INDEX += 1
        progress("validated", theme=job["theme"])
        schedule(build_next)
    except Exception as error:
        fail(error)


def render_error(executor, pipeline, fatal, message):
    global FAILED
    FAILED = True
    progress("render_error", error=str(message))


def render_finished(executor, success):
    success = bool(success and not FAILED)
    for job in JOBS:
        save_json(Path(job["output"]) / "render_status.json", dict(
            status="frames_complete" if success else "failed", unreal_execution_verified=True, unreal_render_verified=False))
    progress("frames_complete" if success else "failed")
    unreal.SystemLibrary.quit_editor()


def begin_render():
    global EXECUTOR
    subsystem = unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem)
    queue = subsystem.get_queue()
    queue.delete_all_jobs()
    for record in JOBS:
        scene = dict(packet=record["packet"], map_path=record["map_path"], sequence=unreal.load_asset(record["sequence_path"]))
        add_job(queue, scene, record["output"], preview=REQUEST["preview"], spatial_samples=1)
    progress("rendering")
    EXECUTOR = unreal.MoviePipelinePIEExecutor()
    EXECUTOR.on_executor_errored_delegate.add_callable_unique(render_error)
    EXECUTOR.on_executor_finished_delegate.add_callable_unique(render_finished)
    subsystem.render_queue_with_executor_instance(EXECUTOR)


schedule(build_next)
