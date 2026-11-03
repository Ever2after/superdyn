"""Movie Render Queue: native deferred/Lumen PNG frames, optional beauty EXRs."""

from pathlib import Path
import unreal
from .io_utils import save_json

_EXECUTOR = None
_FAILED = False
_OUTPUT = None


def failed(executor, pipeline, is_fatal, error_text):
    global _FAILED
    _FAILED = True
    save_json(
        _OUTPUT / "render_status.json",
        dict(status="failed", unreal_render_verified=False, error=str(error_text)),
    )


def finished(executor, success):
    success = bool(success and not _FAILED)
    save_json(
        _OUTPUT / "render_status.json",
        dict(
            status="frames_complete" if success else "failed",
            unreal_execution_verified=True,
            unreal_render_verified=False,
        ),
    )
    unreal.SystemLibrary.quit_editor()


def add_job(queue, scene, output, preview=False, exr=False, spatial_samples=1):
    """Configure one saved scene without starting or clearing the queue."""
    output = Path(output).resolve()
    packet = scene["packet"]
    job = queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
    job.set_editor_property("job_name", packet["case_id"])
    job.set_editor_property(
        "map",
        unreal.SoftObjectPath(
            scene["map_path"] + "." + scene["map_path"].rsplit("/", 1)[-1]
        ),
    )
    job.set_editor_property(
        "sequence", unreal.SoftObjectPath(scene["sequence"].get_path_name())
    )
    config = job.get_configuration()
    setting = config.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
    setting.set_editor_property("output_directory", unreal.DirectoryPath(str(output)))
    setting.set_editor_property("file_name_format", "rgb/frame_{frame_number}")
    setting.set_editor_property(
        "output_resolution", unreal.IntPoint(packet["width"], packet["height"])
    )
    setting.set_editor_property("use_custom_frame_rate", True)
    setting.set_editor_property("output_frame_rate", unreal.FrameRate(packet["fps"], 1))
    setting.set_editor_property("use_custom_playback_range", True)
    setting.set_editor_property("custom_start_frame", 0)
    setting.set_editor_property("custom_end_frame", packet["frames"])
    setting.set_editor_property("zero_pad_frame_numbers", 6)
    setting.set_editor_property("frame_number_offset", 0)
    setting.set_editor_property(
        "output_frame_step", max(1, packet["frames"] // 3) if preview else 1
    )
    setting.set_editor_property("flush_disk_writes_per_shot", True)
    setting.set_editor_property("override_existing_output", False)
    config.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)
    config.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_PNG)
    game = config.find_or_add_setting_by_class(unreal.MoviePipelineGameOverrideSetting)
    game.set_editor_property(
        "soft_game_mode_override",
        unreal.load_class(
            None, "/Script/MovieRenderPipelineCore.MoviePipelineGameMode"
        ),
    )
    game.set_editor_property("cinematic_quality_settings", True)
    if exr:
        config.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_EXR)
    aa = config.find_or_add_setting_by_class(unreal.MoviePipelineAntiAliasingSetting)
    aa.set_editor_property("spatial_sample_count", spatial_samples)
    aa.set_editor_property("temporal_sample_count", 1)
    aa.set_editor_property("override_anti_aliasing", True)
    aa.set_editor_property("anti_aliasing_method", unreal.AntiAliasingMethod.AAM_NONE)
    aa.set_editor_property("engine_warm_up_count", 32)
    aa.set_editor_property("render_warm_up_count", 8 if spatial_samples > 1 else 1)
    camera = config.find_or_add_setting_by_class(unreal.MoviePipelineCameraSetting)
    camera.set_editor_property(
        "shutter_timing", unreal.MoviePipelineShutterTiming.FRAME_OPEN
    )
    unreal.EditorAssetLibrary.save_loaded_asset(scene["sequence"])
    save_json(
        output / "render_status.json",
        dict(
            status="rendering",
            renderer="Unreal Movie Render Queue / Deferred + Lumen",
            expected_frames=packet["frames"],
            preview=preview,
            spatial_samples=spatial_samples,
            unreal_render_verified=False,
        ),
    )
    return job


def render(scene, output, preview=False, exr=False, spatial_samples=1):
    global _EXECUTOR, _OUTPUT, _FAILED
    _FAILED = False
    _OUTPUT = Path(output).resolve()
    output = _OUTPUT
    subsystem = unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem)
    queue = subsystem.get_queue()
    queue.delete_all_jobs()
    add_job(queue, scene, output, preview, exr, spatial_samples)
    _EXECUTOR = unreal.MoviePipelinePIEExecutor()
    _EXECUTOR.on_executor_errored_delegate.add_callable_unique(failed)
    _EXECUTOR.on_executor_finished_delegate.add_callable_unique(finished)
    subsystem.render_queue_with_executor_instance(_EXECUTOR)
    save_json(
        output / "job_status.json",
        dict(stage="queue_submitted", spatial_samples=spatial_samples),
    )
