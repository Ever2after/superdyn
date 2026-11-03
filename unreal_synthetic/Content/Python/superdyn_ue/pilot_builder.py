"""Build pilot clips on curated maps: actors, cameras, sequences, evaluation.

One generated persistent level per environment streams the unmodified map.
Each clip owns possessable actors parked far below the map; only that clip's
sequence keys them onto the stage, so clips sharing a level never meet.
"""

import json
import math
from pathlib import Path

import unreal

from .audit import matrix_from_actor
from .io_utils import save_json
from .pilot_env import compose_level, world, vec, save_generated
from .scene_builder import transform, add_transform_track

PARK_Z_CM = -60000.0
FEET = ("foot", "ball", "toe", "hoof", "paw")
CENTER_BONES = ("pelvis", "hips", "spine_01", "spine1", "bip01_pelvis", "spine")


def editor():
    return unreal.get_editor_subsystem(unreal.EditorActorSubsystem)


def set_frame(sequence, frame):
    unreal.LevelSequenceEditorBlueprintLibrary.open_level_sequence(sequence)
    params = unreal.MovieSceneSequencePlaybackParams(frame=unreal.FrameTime(unreal.FrameNumber(frame)),
                                                     position_type=unreal.MovieScenePositionType.FRAME,
                                                     update_method=unreal.UpdatePositionMethod.JUMP)
    unreal.LevelSequenceEditorBlueprintLibrary.set_global_position(params, unreal.MovieSceneTimeUnit.DISPLAY_RATE)
    unreal.LevelSequenceEditorBlueprintLibrary.force_update()


def new_sequence(folder, name, frames, fps):
    sequence = unreal.AssetToolsHelpers.get_asset_tools().create_asset(name, folder, unreal.LevelSequence, unreal.LevelSequenceFactoryNew())
    if sequence is None:
        raise RuntimeError("Sequence exists: " + folder + "/" + name)
    sequence.set_display_rate(unreal.FrameRate(fps, 1))
    sequence.set_tick_resolution_directly(unreal.FrameRate(fps * 1000, 1))
    sequence.set_playback_start(0)
    sequence.set_playback_end(frames)
    return sequence


def anim_section(binding, clip, start, end, rate):
    track = None
    for existing in binding.get_tracks():
        if isinstance(existing, unreal.MovieSceneSkeletalAnimationTrack):
            track = existing
    track = track or binding.add_track(unreal.MovieSceneSkeletalAnimationTrack)
    section = track.add_section()
    section.set_range(start, end)
    params = section.get_editor_property("params")
    params.set_editor_property("animation", clip)
    params.set_editor_property("play_rate", unreal.MovieSceneTimeWarpExtensions.make_time_warp(float(rate)))
    params.set_editor_property("force_custom_mode", True)
    params.set_editor_property("skip_anim_notifiers", True)
    section.set_editor_property("params", params)
    return section


def skeletal(mesh_path, label, stencil):
    mesh = unreal.load_asset(mesh_path)
    if not isinstance(mesh, unreal.SkeletalMesh):
        raise RuntimeError("Cannot load skeletal mesh " + mesh_path)
    actor = editor().spawn_actor_from_class(unreal.SkeletalMeshActor, unreal.Vector(0, 0, PARK_Z_CM))
    actor.set_actor_label(label)
    component = actor.get_component_by_class(unreal.SkeletalMeshComponent)
    component.set_skeletal_mesh_asset(mesh)
    component.set_mobility(unreal.ComponentMobility.MOVABLE)
    component.set_simulate_physics(False)
    component.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
    component.set_editor_property("render_custom_depth", True)
    component.set_editor_property("custom_depth_stencil_value", stencil)
    component.set_editor_property("visibility_based_anim_tick_option", unreal.VisibilityBasedAnimTickOption.ALWAYS_TICK_POSE_AND_REFRESH_BONES)
    component.set_update_animation_in_editor(True)
    component.set_animation_mode(unreal.AnimationMode.ANIMATION_SINGLE_NODE)
    if any(component.get_material(i) is None for i in range(component.get_num_materials())):
        raise RuntimeError("Missing material on " + mesh_path)
    return actor, component, mesh


def bone_names(component, keys, exclude_ik=True):
    names = [str(component.get_bone_name(i)) for i in range(component.get_num_bones())]
    return [n for n in names if any(k in n.lower() for k in keys) and not (exclude_ik and n.lower().startswith("ik_"))]


def center_bone(component):
    names = [str(component.get_bone_name(i)) for i in range(component.get_num_bones())]
    lower = {n.lower().replace(" ", "_"): n for n in names}
    for key in CENTER_BONES:
        for low, name in lower.items():
            if low == key or low.endswith("_" + key) or low.startswith(key):
                return name
    return None


def calibrate(mesh_path, clip_path, folder, key, fps=30):
    """Native speed (cm/s at scale 1), mesh-forward yaw and root drift from planted feet."""
    actor, component, mesh = skeletal(mesh_path, "SD_Pilot_Calib_" + key, 0)
    clip = unreal.load_asset(clip_path)
    if not isinstance(clip, unreal.AnimSequence) or clip.get_editor_property("skeleton") != mesh.get_editor_property("skeleton"):
        raise RuntimeError("Clip/skeleton mismatch: " + clip_path)
    length = float(clip.get_play_length()) if hasattr(clip, "get_play_length") else float(clip.get_editor_property("sequence_length"))
    frames = max(8, int(length * fps))
    sequence = new_sequence(folder, "LS_Calib_" + key, frames + 1, fps)
    binding = sequence.add_possessable(actor)
    anim_section(binding, clip, 0, frames + 1, 1.0)
    feet = bone_names(component, FEET)
    middle = center_bone(component)
    samples = {n: [] for n in feet}
    centre = []
    for k in range(frames + 1):
        set_frame(sequence, k)
        for n in feet:
            samples[n].append(vec(component.get_socket_location(n)))
        if middle:
            centre.append(vec(component.get_socket_location(middle)))
    unreal.LevelSequenceEditorBlueprintLibrary.close_level_sequence()
    base = actor.get_actor_location()
    editor().destroy_actor(actor)
    result = dict(mesh=mesh_path, clip=clip_path, clip_seconds=length, feet=feet, center_bone=middle, frames=frames)
    lows = [min(p[2] for p in pts) for pts in samples.values() if pts]
    result["sole_below_root_cm"] = float(base.z - min(lows)) if lows else 0.0
    if centre:
        drift = [centre[-1][0] - centre[0][0], centre[-1][1] - centre[0][1]]
        result["root_drift_cm"] = math.hypot(*drift)
        result["center_height_cm"] = float(sum(c[2] for c in centre) / len(centre) - base.z)
        if result["root_drift_cm"] > 20:
            result.update(native_speed_cm_s=result["root_drift_cm"] / length, forward_yaw_deg=math.degrees(math.atan2(drift[1], drift[0])), speed_source="root_drift")
            return result
    velocities = []
    for pts in samples.values():
        floor = min(p[2] for p in pts)
        for a, b in zip(pts, pts[1:]):
            if a[2] < floor + 2.5 and b[2] < floor + 2.5:
                velocities.append(((b[0] - a[0]) * fps, (b[1] - a[1]) * fps))
    if len(velocities) >= 4:
        vx = sorted(v[0] for v in velocities)[len(velocities) // 2]
        vy = sorted(v[1] for v in velocities)[len(velocities) // 2]
        result.update(native_speed_cm_s=math.hypot(vx, vy), forward_yaw_deg=math.degrees(math.atan2(-vy, -vx)), planted_samples=len(velocities), speed_source="planted_feet")
    return result


def root_locked(clip_path, folder):
    """Duplicate a root-motion clip with the root locked to its first frame."""
    name = clip_path.rsplit("/", 1)[-1] + "_RootLocked"
    target = folder + "/" + name
    if not unreal.EditorAssetLibrary.does_asset_exist(target):
        clip = unreal.EditorAssetLibrary.duplicate_asset(clip_path, target)
        clip.set_editor_property("enable_root_motion", True)
        clip.set_editor_property("force_root_lock", True)
        clip.set_editor_property("root_motion_root_lock", unreal.RootMotionRootLock.ANIM_FIRST_FRAME)
        unreal.EditorAssetLibrary.save_loaded_asset(clip)
    return target


def mesh_size(mesh):
    box = mesh.get_imported_bounds()
    e = box.box_extent
    return [2 * e.x, 2 * e.y, 2 * e.z]


def mesh_floor(mesh):
    """Bind-pose bottom relative to the mesh origin (cm, scale 1)."""
    box = mesh.get_imported_bounds()
    return float(box.origin.z - box.box_extent.z)


def trace_ground(x, y, z_hint, ignored):
    hit = unreal.SystemLibrary.line_trace_single(world(), unreal.Vector(x, y, z_hint + 180), unreal.Vector(x, y, z_hint - 400),
                                                 unreal.TraceTypeQuery.TRACE_TYPE_QUERY1, True, ignored, unreal.DrawDebugTrace.NONE, True)
    values = hit.to_tuple() if hit is not None else None
    if not values or not values[0]:
        raise RuntimeError("Ground trace missed at " + str([x, y]))
    return float(values[4].z), values[9].get_actor_label() if values[9] else ""


def blocked(a, b, ignored, radius=0.0):
    if radius > 0:
        hit = unreal.SystemLibrary.sphere_trace_single(world(), unreal.Vector(*a), unreal.Vector(*b), radius,
                                                       unreal.TraceTypeQuery.TRACE_TYPE_QUERY1, True, ignored, unreal.DrawDebugTrace.NONE, True)
    else:
        hit = unreal.SystemLibrary.line_trace_single(world(), unreal.Vector(*a), unreal.Vector(*b),
                                                     unreal.TraceTypeQuery.TRACE_TYPE_QUERY1, True, ignored, unreal.DrawDebugTrace.NONE, True)
    values = hit.to_tuple() if hit is not None else None
    if values and values[0]:
        return values[9].get_actor_label() if values[9] else "?"
    return None


def cine_camera(label, fov, width, height):
    camera = editor().spawn_actor_from_class(unreal.CineCameraActor, unreal.Vector(0, 0, PARK_Z_CM))
    camera.set_actor_label(label)
    cc = camera.get_cine_camera_component()
    cc.set_editor_property("filmback", unreal.CameraFilmbackSettings(sensor_width=36, sensor_height=36 * height / width))
    cc.set_editor_property("current_focal_length", 36 / (2 * math.tan(math.radians(fov / 2))))
    focus = cc.get_editor_property("focus_settings")
    focus.set_editor_property("focus_method", unreal.CameraFocusMethod.DISABLE)
    cc.set_editor_property("focus_settings", focus)
    post = cc.get_editor_property("post_process_settings")
    post.set_editor_property("override_motion_blur_amount", True)
    post.set_editor_property("motion_blur_amount", 0.0)
    return camera


def apply_exposure(camera, exposure):
    if not exposure:
        return
    cc = camera.get_cine_camera_component()
    post = cc.get_editor_property("post_process_settings")
    method = dict(manual=unreal.AutoExposureMethod.AEM_MANUAL, histogram=unreal.AutoExposureMethod.AEM_HISTOGRAM)[exposure["method"]]
    values = dict(override_auto_exposure_method=True, auto_exposure_method=method,
                  override_auto_exposure_bias=True, auto_exposure_bias=float(exposure.get("bias", 0.0)),
                  override_auto_exposure_apply_physical_camera_exposure=True, auto_exposure_apply_physical_camera_exposure=False)
    if exposure["method"] == "histogram":
        values.update(override_auto_exposure_min_brightness=True, auto_exposure_min_brightness=float(exposure.get("min", 0.03)),
                      override_auto_exposure_max_brightness=True, auto_exposure_max_brightness=float(exposure.get("max", 20.0)),
                      override_auto_exposure_speed_up=True, auto_exposure_speed_up=float(exposure.get("speed", 2.0)),
                      override_auto_exposure_speed_down=True, auto_exposure_speed_down=float(exposure.get("speed", 2.0)))
    for key, value in values.items():
        post.set_editor_property(key, value)
    cc.set_editor_property("post_process_settings", post)


def build_clip(clip, request, calibration, folder):
    """Spawn, pose (ground-traced) and sequence one clip; returns runtime handles."""
    fps, frames = request["fps"], request["frames"]
    ignored = []
    handles = dict(actors=[], clip=clip)
    stage_z = clip["ground_z_cm"]
    for index, spec in enumerate(clip["actors"]):
        cal = calibration[spec["asset"]]
        actor, component, mesh = skeletal(spec["mesh"], "SD_Pilot_" + clip["id"] + "_" + spec["id"], spec["stencil"])
        size = mesh_size(mesh)
        scale = spec["size_cm"] / (size[2] if spec["size_basis"] == "height" else max(size[0], size[1]))
        forward = cal.get("forward_yaw_deg", 90.0)
        poses, grounds = [], []
        for k in range(frames):
            x, y = spec["x_cm"][k], spec["y_cm"][k]
            if spec["z_mode"] == "ground":
                gz, _ = trace_ground(x, y, stage_z, ignored)
                z = gz - mesh_floor(mesh) * scale + 0.5
            else:
                gz = stage_z
                z = stage_z + spec["h_cm"][k]
            grounds.append(gz)
            poses.append(dict(location_cm=[x, y, z], rotation_rpy_deg=[0.0, 0.0, spec["yaw_deg"][k] - forward], scale=[scale] * 3))
        handles["actors"].append(dict(spec=spec, actor=actor, component=component, scale=scale, poses=poses,
                                      ground_cm=grounds, forward_yaw_deg=forward, mesh_size_cm=size, mesh_floor_cm=mesh_floor(mesh)))
    cam = clip["camera"]
    camera = cine_camera("SD_Pilot_" + clip["id"] + "_Camera", cam["fov_deg"], request["width"], request["height"])
    apply_exposure(camera, clip.get("exposure"))
    cam_poses = []
    for k in range(frames):
        x, y = cam["x_cm"][k], cam["y_cm"][k]
        base = trace_ground(x, y, stage_z, ignored)[0] if cam["z_mode"] == "ground" else stage_z
        cam_poses.append(dict(location_cm=[x, y, base + cam["h_cm"][k]], rotation_rpy_deg=[cam["roll_deg"][k], cam["pitch_deg"][k], cam["yaw_deg"][k]], scale=[1, 1, 1]))
    sequence = new_sequence(folder, "LS_" + clip["id"], frames, fps)
    for item in handles["actors"]:
        spec = item["spec"]
        binding = add_transform_track(sequence, item["actor"], item["poses"])
        for start, end, gait in spec["segments"]:
            clip_path = spec["clips"].get(gait) or spec["clips"].get("walk") or next(iter(spec["clips"].values()))
            anim = unreal.load_asset(clip_path)
            native = None
            if gait in ("walk", "run") and spec.get("speed_mps"):
                measured = calibration.get(spec["asset"] + ":" + gait, {}).get("native_speed_cm_s")
                # Fallback only when planted feet were not detected; recorded per section.
                native = (measured or (150.0 if gait == "walk" else 380.0) / item["scale"]) * item["scale"]
                rate = max(0.2, min(3.0, spec["speed_mps"] * 100 / native))
            elif gait == "hold":
                rate = 0.01
            else:
                rate = 1.0
            anim_section(binding, anim, start, end, rate)
            item.setdefault("sections", []).append(dict(start=start, end=end, gait=gait, clip=clip_path, play_rate=rate, native_speed_cm_s=native))
    camera_binding = add_transform_track(sequence, camera, cam_poses)
    cut = sequence.add_track(unreal.MovieSceneCameraCutTrack).add_section()
    cut.set_range(0, frames)
    bid = unreal.MovieSceneObjectBindingID()
    bid.set_editor_property("guid", camera_binding.get_id())
    cut.set_camera_binding_id(bid)
    unreal.EditorAssetLibrary.save_loaded_asset(sequence)
    handles.update(camera=camera, camera_poses=cam_poses, sequence=sequence)
    return handles


def evaluate_clip(handles, request, output):
    """Evaluated camera/actor state per frame plus native collision and LOS checks."""
    clip, sequence, frames = handles["clip"], handles["sequence"], request["frames"]
    ours = [a["actor"] for a in handles["actors"]] + [handles["camera"]]
    record = dict(id=clip["id"], camera=[], actors={a["spec"]["id"]: dict(root=[], center=[], bounds=[]) for a in handles["actors"]})
    centers = {a["spec"]["id"]: center_bone(a["component"]) for a in handles["actors"]}
    camera_error = 0.0
    for k in range(frames):
        set_frame(sequence, k)
        m = matrix_from_actor(handles["camera"])
        camera_error = max(camera_error, max(abs(m[i][3] - handles["camera_poses"][k]["location_cm"][i]) for i in range(3)))
        record["camera"].append(m)
        for item in handles["actors"]:
            key = item["spec"]["id"]
            rec = record["actors"][key]
            rec["root"].append(matrix_from_actor(item["actor"]))
            bone = centers[key]
            rec["center"].append(vec(item["component"].get_socket_location(bone)) if bone else None)
            origin, extent = item["actor"].get_actor_bounds(False)
            rec["bounds"].append([vec(origin), vec(extent)])
    unreal.LevelSequenceEditorBlueprintLibrary.close_level_sequence()
    collisions, occlusions = [], []
    for item in handles["actors"]:
        spec = item["spec"]
        if spec["z_mode"] != "ground":
            continue
        for k in range(0, frames - 4, 4):
            for lift in (35.0, 110.0):
                a = list(item["poses"][k]["location_cm"])
                b = list(item["poses"][k + 4]["location_cm"])
                a[2] = item["ground_cm"][k] + lift
                b[2] = item["ground_cm"][k + 4] + lift
                hit = blocked(a, b, ours, radius=18.0) if math.dist(a, b) > 1 else None
                if hit:
                    collisions.append(dict(actor=spec["id"], frame=k, lift_cm=lift, hit=hit))
    poses = handles["camera_poses"]
    for k in range(0, frames - 4, 4):
        moved = math.dist(poses[k]["location_cm"], poses[k + 4]["location_cm"]) > 1
        hit = blocked(poses[k]["location_cm"], poses[k + 4]["location_cm"], ours, radius=20.0) if moved else None
        if hit:
            collisions.append(dict(actor="camera", frame=k, hit=hit))
    for k in request["measure"]:
        for item in handles["actors"]:
            if not item["spec"].get("label"):
                continue
            center = record["actors"][item["spec"]["id"]]["center"][k] or record["actors"][item["spec"]["id"]]["bounds"][k][0]
            hit = blocked(poses[k]["location_cm"], center, ours)
            if hit:
                occlusions.append(dict(actor=item["spec"]["id"], frame=k, hit=hit))
    record.update(camera_max_error_cm=camera_error, collisions=collisions, line_of_sight_blocked=occlusions,
                  center_bones=centers, map_path=handles["map_path"], sequence_path=sequence.get_path_name(),
                  actors_meta={a["spec"]["id"]: dict(scale=a["scale"], forward_yaw_deg=a["forward_yaw_deg"], mesh_size_cm=a["mesh_size_cm"],
                                                     sections=a.get("sections", []), stencil=a["spec"]["stencil"], ground_cm=a["ground_cm"])
                               for a in handles["actors"]},
                  camera_poses=poses)
    save_json(Path(output) / "clips" / clip["id"] / "evaluated.json", record)
    return dict(id=clip["id"], camera_max_error_cm=camera_error, collisions=len(collisions), occlusions=len(occlusions))


def steps(request, root, render_jobs):
    output = Path(request["output"])
    namespace = request["namespace"]
    levels = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    calibration = {}
    state = {}
    result = []
    by_env = {}
    for clip in request["clips"]:
        by_env.setdefault(clip["env"], []).append(clip)
    for env, clips in by_env.items():
        map_path = namespace + "/Maps/Pilot_" + env

        def compose(env=env, clips=clips, map_path=map_path):
            state["persistent:" + env] = compose_level(map_path, clips[0]["env_map"])
            return 90

        def calibrate_all(clips=clips):
            folder = namespace + "/Calibration"
            for clip in clips:
                for spec in clip["actors"]:
                    for gait, path in list(spec["clips"].items()):
                        key = spec["asset"] + ":" + gait
                        if gait not in ("walk", "run"):
                            continue
                        if key not in calibration:
                            result = calibrate(spec["mesh"], path, folder, key.replace(":", "_"))
                            if result.get("speed_source") == "root_drift":
                                locked = root_locked(path, namespace + "/Clips")
                                check = calibrate(spec["mesh"], locked, folder, key.replace(":", "_") + "_locked")
                                result.update(locked_clip=locked, locked_drift_cm=check.get("root_drift_cm"))
                            calibration[key] = result
                        if calibration[key].get("locked_clip"):
                            spec["clips"][gait] = calibration[key]["locked_clip"]
                    if spec["asset"] not in calibration:
                        moving = [calibration[spec["asset"] + ":" + g] for g in ("walk", "run") if spec["asset"] + ":" + g in calibration]
                        base = dict(moving[0]) if moving else calibrate(spec["mesh"], next(iter(spec["clips"].values())), folder, spec["asset"] + "_any")
                        if "forward_yaw_deg" not in base:
                            base["forward_yaw_deg"] = 90.0
                        calibration[spec["asset"]] = base
            save_json(output / "calibration.json", calibration)
            return 4

        def build(clips=clips, map_path=map_path, env=env):
            folder = namespace + "/Sequences"
            for clip in clips:
                handles = build_clip(clip, request, calibration, folder)
                handles["map_path"] = map_path
                state[clip["id"]] = handles
            save_generated(map_path, state["persistent:" + env])
            return 10

        def validate(clips=clips, map_path=map_path, env=env):
            reports = []
            for clip in clips:
                reports.append(evaluate_clip(state[clip["id"]], request, output))
                handles = state[clip["id"]]
                for kind in request["passes"]:
                    render_jobs.append(dict(name=clip["id"] + "_" + kind, map_path=map_path, sequence_path=handles["sequence"].get_path_name(),
                                            output=str(output / "clips" / clip["id"]), frames=request["frames"], width=request["width"],
                                            height=request["height"], fps=request["fps"], kind=kind, spatial_samples=request["spatial_samples"],
                                            frame_step=request["gt_frame_step"] if kind == "gt" else 1, warm_up=32))
            save_generated(map_path, state["persistent:" + env])
            save_json(output / "native" / ("validation_" + clips[0]["env"] + ".json"), reports)
            return 4

        result += [("compose:" + env, compose), ("calibrate:" + env, calibrate_all), ("build:" + env, build), ("validate:" + env, validate)]
    return result
