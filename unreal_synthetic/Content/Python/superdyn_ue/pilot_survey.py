"""Read-only survey of curated environment maps plus review renders."""

import gzip
import json
import math
from pathlib import Path

import unreal

from .io_utils import save_json
from .pilot_env import inventory, raster, compose_level, look_at, save_generated
from .scene_builder import add_transform_track


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))]


def camera_actor(editor, fov, width, height):
    camera = editor.spawn_actor_from_class(unreal.CineCameraActor, unreal.Vector())
    camera.set_actor_label("SD_Pilot_Camera")
    cc = camera.get_cine_camera_component()
    cc.set_editor_property("filmback", unreal.CameraFilmbackSettings(sensor_width=36, sensor_height=36 * height / width))
    cc.set_editor_property("current_focal_length", 36 / (2 * math.tan(math.radians(fov / 2))))
    focus = cc.get_editor_property("focus_settings")
    focus.set_editor_property("focus_method", unreal.CameraFocusMethod.DISABLE)
    cc.set_editor_property("focus_settings", focus)
    post = cc.get_editor_property("post_process_settings")
    post.set_editor_property("override_motion_blur_amount", True)
    post.set_editor_property("motion_blur_amount", 0.0)
    cc.set_editor_property("post_process_settings", post)
    return camera


def sequence_for(camera, poses, fps, folder, name):
    sequence = unreal.AssetToolsHelpers.get_asset_tools().create_asset(name, folder, unreal.LevelSequence, unreal.LevelSequenceFactoryNew())
    if sequence is None:
        raise RuntimeError("Sequence exists: " + folder + "/" + name)
    sequence.set_display_rate(unreal.FrameRate(fps, 1))
    sequence.set_tick_resolution_directly(unreal.FrameRate(fps * 1000, 1))
    sequence.set_playback_start(0)
    sequence.set_playback_end(len(poses))
    binding = add_transform_track(sequence, camera, poses)
    cut = sequence.add_track(unreal.MovieSceneCameraCutTrack).add_section()
    cut.set_range(0, len(poses))
    bid = unreal.MovieSceneObjectBindingID()
    bid.set_editor_property("guid", binding.get_id())
    cut.set_camera_binding_id(bid)
    return sequence


def steps(request, root, render_jobs):
    output = Path(request["output"])
    levels = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    editor = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    namespace = request["namespace"]
    found = {}
    result = []
    for env in request["environments"]:
        key = env["id"]

        def load(env=env):
            if not levels.load_level(env["map"]):
                raise RuntimeError("Cannot load " + env["map"])
            return 60

        def survey(env=env, key=key):
            info = inventory()
            pts = info.pop("mesh_centers_cm")
            if len(pts) < 10:
                raise RuntimeError("Too few environment meshes in " + env["map"])
            xs, ys, zs = zip(*pts)
            bounds = [percentile(xs, 0.04), percentile(xs, 0.96), percentile(ys, 0.04), percentile(ys, 0.96)]
            area = (bounds[1] - bounds[0]) * (bounds[3] - bounds[2])
            step = max(100.0, math.ceil(math.sqrt(area / 30000.0) / 25.0) * 25.0)
            top, bottom = percentile(zs, 0.98) + 3000, percentile(zs, 0.02) - 3000
            grid = raster(bounds, step, top, bottom)
            info.update(map=env["map"], bounds_cm=bounds, trace_top_cm=top, trace_bottom_cm=bottom, raster=grid)
            path = output / "survey" / (key + ".json.gz")
            path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(path, "wt") as stream:
                json.dump(info, stream)
            ground = sorted(c[-1][0] for row in grid["columns"] for c in row if c)
            found[key] = dict(bounds=bounds, ground=percentile(ground, 0.5) if ground else 0.0, cameras=info["cameras"])
            save_json(output / "survey" / (key + "_summary.json"), dict(map=env["map"], class_counts=info["class_counts"],
                      cameras=info["cameras"], lights=info["lights"], post_process=info["post_process"], bounds_cm=bounds,
                      raster_step_cm=step, raster_shape=grid["shape"]))
            return 4

        def compose(env=env, key=key):
            data = found[key]
            map_path = namespace + "/Maps/Survey_" + key
            persistent = compose_level(map_path, env["map"])
            a, b, c, d = data["bounds"]
            center = [(a + b) / 2, (c + d) / 2, data["ground"]]
            span = max(b - a, d - c)
            poses = []
            height = 0.5 * span / math.tan(math.radians(40)) + 2000
            poses.append(dict(location_cm=[center[0], center[1], center[2] + height], rotation_rpy_deg=[0, -89.9, 0], scale=[1, 1, 1]))
            for azimuth in [0, 90, 180, 270]:
                t = math.radians(azimuth)
                eye = [center[0] + 0.55 * span * math.cos(t), center[1] + 0.55 * span * math.sin(t), center[2] + 0.35 * span]
                poses.append(dict(location_cm=eye, rotation_rpy_deg=look_at(eye, center), scale=[1, 1, 1]))
            for cam in data["cameras"][:7]:
                poses.append(dict(location_cm=cam["location_cm"], rotation_rpy_deg=cam["rotation_rpy_deg"], scale=[1, 1, 1]))
            camera = camera_actor(editor, 80, 1280, 720)
            sequence = sequence_for(camera, poses, 24, namespace + "/Sequences", "LS_Survey_" + key)
            unreal.EditorAssetLibrary.save_loaded_asset(sequence)
            save_generated(map_path, persistent)
            save_json(output / "survey" / (key + "_views.json"), dict(poses=poses, map_path=map_path))
            render_jobs.append(dict(name="survey_" + key, map_path=map_path, sequence_path=sequence.get_path_name(),
                                    output=str(output / "survey" / key), frames=len(poses), width=1280, height=720, fps=24,
                                    kind="rgb", spatial_samples=1, warm_up=16))
            return 30

        if env.get("views"):
            def look(env=env, key=key):
                map_path = namespace + "/Maps/Look_" + key
                persistent = compose_level(map_path, env["map"])
                camera = camera_actor(editor, 75, 1280, 720)
                if env.get("exposure"):
                    from .pilot_builder import apply_exposure
                    apply_exposure(camera, env["exposure"])
                sequence = sequence_for(camera, env["views"], 24, namespace + "/Sequences", "LS_Look_" + key)
                unreal.EditorAssetLibrary.save_loaded_asset(sequence)
                save_generated(map_path, persistent)
                render_jobs.append(dict(name="look_" + key, map_path=map_path, sequence_path=sequence.get_path_name(),
                                        output=str(output / "look" / key), frames=len(env["views"]), width=1280, height=720, fps=24,
                                        kind="rgb", spatial_samples=2, warm_up=24))
                return 30
            result += [("look:" + key, look)]
            continue
        if env.get("windows"):
            def detail(env=env, key=key):
                for n, bounds in enumerate(env["windows"]):
                    pts_top = env.get("trace_top_cm", 10000.0)
                    grid = raster(bounds, env.get("detail_step_cm", 50.0), pts_top, env.get("trace_bottom_cm", -5000.0))
                    path = output / "survey" / ("%s_detail_%d.json.gz" % (key, n))
                    with gzip.open(path, "wt") as stream:
                        json.dump(dict(map=env["map"], bounds_cm=bounds, raster=grid), stream)
                return 4
            result += [("load:" + key, load), ("detail:" + key, detail)]
        else:
            result += [("load:" + key, load), ("survey:" + key, survey), ("compose:" + key, compose)]
    return result
