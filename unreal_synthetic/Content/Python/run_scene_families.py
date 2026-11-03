"""Build, audit and render the three families in one native editor session."""

import json
import os
from pathlib import Path
import traceback

import unreal

from superdyn_ue.family_builder import build, audit
from superdyn_ue.io_utils import save_json
from superdyn_ue.render_queue import add_job

ROOT = Path(__file__).resolve().parents[2]
REQUEST = json.loads(Path(os.environ["SUPERDYN_UE_REQUEST"]).read_text())
INDEX = 0
PENDING = None
RENDER_JOBS = []
HANDLE = None
TICKS = 0
EXECUTOR = None
FAILED = False


def status(stage, **values):
    save_json(Path(REQUEST["output"]) / "native_status.json", dict(stage=stage, completed_builds=INDEX, expected=len(REQUEST["jobs"]), **values))


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


def fail(error):
    unschedule()
    status("failed", error=str(error), traceback=traceback.format_exc())
    unreal.log_error(traceback.format_exc())
    unreal.SystemLibrary.quit_editor()


def next_scene(_delta):
    global PENDING, TICKS
    if unreal.AssetRegistryHelpers.get_asset_registry().is_loading_assets():
        return
    unschedule()
    try:
        if INDEX == len(REQUEST["jobs"]):
            render()
            return
        job = REQUEST["jobs"][INDEX]
        status("building", family=job["id"])
        PENDING = build(job, ROOT)
        TICKS = 4
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
        report = audit(PENDING)
        folder = Path(PENDING["job"]["output"])
        for label, sequence in PENDING["sequences"].items():
            packet = dict(PENDING["packet"], case_id=PENDING["job"]["id"] + "_" + label,
                          frames=PENDING["packet"]["frames"] if label == "walk" else len(PENDING["packet"]["review_camera"]))
            RENDER_JOBS.append(dict(packet=packet, map_path=PENDING["job"]["map_path"], sequence_path=sequence.get_path_name(), output=str(folder / label)))
        INDEX += 1
        status("validated", family=PENDING["job"]["id"], report=report)
        PENDING = None
        schedule(next_scene)
    except Exception as error:
        fail(error)


def error(executor, pipeline, fatal, message):
    global FAILED
    FAILED = True
    status("render_error", error=str(message))


def finished(executor, success):
    status("frames_complete" if success and not FAILED else "failed", rendered_jobs=len(RENDER_JOBS))
    unreal.SystemLibrary.quit_editor()


def render():
    global EXECUTOR
    subsystem = unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem)
    queue = subsystem.get_queue()
    queue.delete_all_jobs()
    for job in RENDER_JOBS:
        scene = dict(packet=job["packet"], map_path=job["map_path"], sequence=unreal.load_asset(job["sequence_path"]))
        add_job(queue, scene, job["output"], spatial_samples=REQUEST["spatial_samples"])
    status("rendering", jobs=len(RENDER_JOBS))
    EXECUTOR = unreal.MoviePipelinePIEExecutor()
    EXECUTOR.on_executor_errored_delegate.add_callable_unique(error)
    EXECUTOR.on_executor_finished_delegate.add_callable_unique(finished)
    subsystem.render_queue_with_executor_instance(EXECUTOR)


status("starting")
schedule(next_scene)
