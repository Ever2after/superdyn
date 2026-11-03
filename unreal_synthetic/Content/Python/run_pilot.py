"""Synthetic-split pilot inside one native editor session.

Stages are queued callables run on Slate ticks once asset loading settles,
followed by one Movie Render Queue pass over every collected job.
"""

import json
import os
from pathlib import Path
import traceback

import unreal

from superdyn_ue.io_utils import save_json

ROOT = Path(__file__).resolve().parents[2]
REQUEST = json.loads(Path(os.environ["SUPERDYN_UE_REQUEST"]).read_text())
OUTPUT = Path(REQUEST["output"])
STEPS = []
RENDER_JOBS = []
# Map loads pump Slate ticks (progress UI); `busy` blocks re-entrant steps.
STATE = dict(handle=None, wait=30, failed=False, executor=None, index=0, busy=False)


def status(stage, **values):
    # One status file per attempt so concurrent nullrhi/render sessions never collide.
    save_json(Path(REQUEST["status_dir"]) / "native_status.json", dict(stage=stage, step=STATE["index"], steps=len(STEPS), **values))


def fail(error):
    if STATE["handle"] is not None:
        unreal.unregister_slate_post_tick_callback(STATE["handle"])
    status("failed", error=str(error), traceback=traceback.format_exc())
    unreal.log_error(traceback.format_exc())
    unreal.SystemLibrary.quit_editor()


def tick(_delta):
    if STATE["busy"]:
        return
    if STATE["wait"] > 0:
        STATE["wait"] -= 1
        return
    if unreal.AssetRegistryHelpers.get_asset_registry().is_loading_assets():
        return
    try:
        if STATE["index"] == len(STEPS):
            unreal.unregister_slate_post_tick_callback(STATE["handle"])
            STATE["handle"] = None
            render()
            return
        name, function = STEPS[STATE["index"]]
        status("running", current=name)
        STATE["busy"] = True
        try:
            result = function()
        finally:
            STATE["busy"] = False
        STATE["index"] += 1
        STATE["wait"] = result if isinstance(result, int) else 4
    except Exception as error:
        fail(error)


def error(executor, pipeline, fatal, message):
    STATE["failed"] = True
    status("render_error", error=str(message))


def finished(executor, success):
    status("frames_complete" if success and not STATE["failed"] else "failed", rendered_jobs=len(RENDER_JOBS))
    unreal.SystemLibrary.quit_editor()


def throttle(enabled):
    """Shared GPUs: keep the editor viewport cheap until Movie Render Queue runs."""
    unreal.SystemLibrary.execute_console_command(None, "t.MaxFPS " + ("4" if enabled else "0"))
    subsystem = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    for name in ["editor_set_viewport_realtime", "set_viewport_realtime"]:
        if hasattr(subsystem, name):
            try:
                getattr(subsystem, name)(not enabled)
            except Exception as error:
                unreal.log_warning("viewport realtime toggle failed: " + str(error))


def render():
    throttle(False)
    if REQUEST.get("defer_render"):
        save_json(Path(REQUEST["status_dir"]) / "render_jobs.json", RENDER_JOBS)
        status("frames_complete", rendered_jobs=0, deferred_jobs=len(RENDER_JOBS))
        unreal.SystemLibrary.quit_editor()
        return
    if not RENDER_JOBS:
        status("frames_complete", rendered_jobs=0)
        unreal.SystemLibrary.quit_editor()
        return
    from superdyn_ue.pilot_render import add_job
    subsystem = unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem)
    queue = subsystem.get_queue()
    queue.delete_all_jobs()
    for job in RENDER_JOBS:
        add_job(queue, job)
    status("rendering", jobs=len(RENDER_JOBS))
    STATE["executor"] = unreal.MoviePipelinePIEExecutor()
    STATE["executor"].on_executor_errored_delegate.add_callable_unique(error)
    STATE["executor"].on_executor_finished_delegate.add_callable_unique(finished)
    subsystem.render_queue_with_executor_instance(STATE["executor"])


def main():
    stage = REQUEST["stage"]
    if stage == "render":
        RENDER_JOBS.extend(REQUEST["jobs"])
    else:
        if stage == "survey":
            from superdyn_ue.pilot_survey import steps
        elif stage == "clips":
            from superdyn_ue.pilot_builder import steps
        elif stage == "probe":
            from superdyn_ue.pilot_probe import steps
        else:
            raise ValueError("Unknown pilot stage " + stage)
        STEPS.extend(steps(REQUEST, ROOT, RENDER_JOBS))
    throttle(True)
    status("starting")
    STATE["handle"] = unreal.register_slate_post_tick_callback(tick)


try:
    main()
except Exception as exc:
    fail(exc)
