"""Audit evaluated Unreal transforms and collision, not just imported source keys."""

import unreal
from .math3d import compose
from .io_utils import save_json
from .asset_layout import person_mesh, OFFROAD_MESH


def matrix_from_actor(actor, relative=False):
    t = (
        actor.get_actor_transform()
        if not relative
        else actor.get_editor_property("root_component").get_relative_transform()
    )
    r = t.rotation.rotator()
    return compose(
        dict(
            location_cm=[t.translation.x, t.translation.y, t.translation.z],
            rotation_rpy_deg=[r.roll, r.pitch, r.yaw],
            scale=[t.scale3d.x, t.scale3d.y, t.scale3d.z],
        )
    )


def errors(actual, expected):
    return (
        max(abs(actual[i][3] - expected[i][3]) for i in range(3)),
        max(abs(actual[i][j] - expected[i][j]) for i in range(3) for j in range(3)),
    )


def audit_scene(scene, output):
    try:
        return _audit_scene(scene, output)
    finally:
        unreal.LevelSequenceEditorBlueprintLibrary.close_level_sequence()


def _audit_scene(scene, output):
    packet = scene["packet"]
    sequence = scene["sequence"]
    visuals = scene["visuals"]
    by_label = {a.get_actor_label(): a for a in unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()}
    vehicle_parts = scene["visual_settings"]["ground"].get("parts", [])
    if len(vehicle_parts) != 5:
        raise RuntimeError("OffroadCar requires its body and four tires")
    for part in vehicle_parts:
        actor = by_label.get(part["label"])
        if actor is None or actor.get_attach_parent_actor() != visuals["ground"]:
            raise RuntimeError("Missing or detached OffroadCar part: " + part["label"])
        component = actor.get_component_by_class(unreal.StaticMeshComponent)
        mesh = component.get_editor_property("static_mesh")
        if mesh is None or mesh.get_path_name().split(".")[0] != part["mesh"]:
            raise RuntimeError("Incorrect OffroadCar part: " + part["label"])
        if component.get_editor_property("custom_depth_stencil_value") != packet["instance_ids"]["ground"]:
            raise RuntimeError("Incorrect vehicle part instance ID")
        if any(component.get_material(i) is None for i in range(component.get_num_materials())):
            raise RuntimeError("Missing vehicle part material")
    visual_components = {}
    for key, expected_mesh in {"person": person_mesh(packet["case_id"]), "ground": OFFROAD_MESH}.items():
        actor = visuals[key]
        component = actor.get_component_by_class(unreal.SkeletalMeshComponent)
        mesh = component.get_editor_property("skeletal_mesh_asset")
        if mesh is None or mesh.get_path_name().split(".")[0] != expected_mesh:
            raise RuntimeError("Incorrect official visual for " + key)
        if actor.get_attach_parent_actor() != scene["actors"][key]:
            raise RuntimeError("Official visual is not attached to its GT anchor: " + key)
        position, linear = errors(matrix_from_actor(actor, True), compose(scene["visual_settings"][key]["local_pose"]))
        if position > 0.01 or linear > 1e-5:
            raise RuntimeError("Official visual alignment differs for " + key)
        if component.get_editor_property("custom_depth_stencil_value") != packet["instance_ids"][key]:
            raise RuntimeError("Incorrect visual instance ID for " + key)
        if any(component.get_material(i) is None for i in range(component.get_num_materials())):
            raise RuntimeError("Missing official visual material for " + key)
        visual_components[key] = component
    unreal.LevelSequenceEditorBlueprintLibrary.open_level_sequence(sequence)
    maximum = 0.0
    linear_maximum = 0.0
    evaluated = []
    for frame in range(packet["frames"]):
        params = unreal.MovieSceneSequencePlaybackParams(
            frame=unreal.FrameTime(unreal.FrameNumber(frame)),
            position_type=unreal.MovieScenePositionType.FRAME,
            update_method=unreal.UpdatePositionMethod.JUMP,
        )
        unreal.LevelSequenceEditorBlueprintLibrary.set_global_position(
            params, unreal.MovieSceneTimeUnit.DISPLAY_RATE
        )
        unreal.LevelSequenceEditorBlueprintLibrary.force_update()
        record = dict(frame=frame, camera=matrix_from_actor(scene["camera"]), actors={})
        for key, poses in packet["tracks"].items():
            expected = compose(poses[frame])
            actual = matrix_from_actor(
                scene["actors"][key],
                relative=next(m["parent"] for m in packet["meshes"] if m["key"] == key)
                is not None,
            )
            position, linear = errors(actual, expected)
            maximum = max(maximum, position)
            linear_maximum = max(linear_maximum, linear)
            record["actors"][key] = actual
        expected = compose(packet["camera"][frame])
        position, linear = errors(record["camera"], expected)
        maximum = max(maximum, position)
        linear_maximum = max(linear_maximum, linear)
        person = visual_components["person"]
        record["person_bones_world_cm"] = {}
        for bone in ["pelvis", "head", "hand_l", "hand_r", "foot_l", "foot_r", "ball_l", "ball_r"]:
            point = person.get_socket_location(bone)
            record["person_bones_world_cm"][bone] = [point.x, point.y, point.z]
        evaluated.append(record)
    # Compare bones in the mannequin's local frame so root movement alone
    # cannot make a frozen reference pose appear to pass animation validation.
    foot_poses = []
    for frame in [0, 8, 16, 24]:
        params = unreal.MovieSceneSequencePlaybackParams(
            frame=unreal.FrameTime(unreal.FrameNumber(frame)),
            position_type=unreal.MovieScenePositionType.FRAME,
            update_method=unreal.UpdatePositionMethod.JUMP,
        )
        unreal.LevelSequenceEditorBlueprintLibrary.set_global_position(params, unreal.MovieSceneTimeUnit.DISPLAY_RATE)
        unreal.LevelSequenceEditorBlueprintLibrary.force_update()
        point = visual_components["person"].get_socket_transform("foot_l", unreal.RelativeTransformSpace.RTS_COMPONENT).translation
        foot_poses.append([point.x, point.y, point.z])
    animation_span = max(max(p[i] for p in foot_poses) - min(p[i] for p in foot_poses) for i in range(3))
    if animation_span < 1.0:
        raise RuntimeError("Person animation appears frozen: foot span cm=" + str(animation_span))
    if maximum > 0.1 or linear_maximum > 1e-4:
        raise RuntimeError(
            "Sequencer evaluated transform differs: position cm="
            + str(maximum)
            + " rotation/scale matrix="
            + str(linear_maximum)
        )
    ignored = (
        [a for key, a in scene["actors"].items() if key != "terrain"]
        + list(visuals.values())
        + [by_label[part["label"]] for part in vehicle_parts]
        + scene["ignored"]
        + [scene["camera"]]
    )
    support_error = 0.0
    misses = []
    world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
    for point in packet["terrain_support_probes_cm"]:
        start = unreal.Vector(point[0], point[1], 3500.0)
        end = unreal.Vector(point[0], point[1], -1000.0)
        # Python wraps the C++ bool + out-HitResult as HitResult or None.
        hit = unreal.SystemLibrary.line_trace_single(
            world,
            start,
            end,
            unreal.TraceTypeQuery.TRACE_TYPE_QUERY1,
            True,
            ignored,
            unreal.DrawDebugTrace.NONE,
            True,
        )
        if hit is None:
            misses.append(point)
            continue
        # Native make/break functions are folded into the Python struct API.
        try:
            location = hit.get_editor_property("location")
        except Exception:
            location = hit.to_tuple()[4]
        support_error = max(support_error, abs(location.z - point[2]))
    if misses or support_error > 0.1:
        raise RuntimeError(
            "Native terrain collision differs: misses="
            + str(len(misses))
            + " error cm="
            + str(support_error)
        )
    report = dict(
        status="passed",
        unreal_engine_version=unreal.SystemLibrary.get_engine_version(),
        frames=packet["frames"],
        evaluated_position_max_error_cm=maximum,
        evaluated_rotation_scale_matrix_max_error=linear_maximum,
        terrain_support_probes=len(packet["terrain_support_probes_cm"]),
        terrain_collision_max_height_error_cm=support_error,
        official_asset_signature=scene["official_asset_signature"],
        visual_assets=scene["visual_settings"],
        person_animation_local_foot_span_cm=animation_span,
        visual_validation="Native mesh/material/attachment/stencil checks and evaluated skeletal motion; no foot IK or wheel/Chaos simulation",
        scope="actual Unreal Sequencer evaluation and native complex collision; RGB image calibration still requires render validation",
    )
    save_json(
        output / "unreal_evaluated_gt.json",
        dict(
            coordinate_system="Unreal left-handed cm; camera local X forward/Y right/Z up",
            actor_transform_spaces={
                m["key"]: ("relative to " + m["parent"]) if m["parent"] else "world"
                for m in packet["meshes"]
            },
            K=packet["K"],
            fps=packet["fps"],
            visual_assets=scene["visual_settings"],
            frames=evaluated,
        ),
    )
    save_json(output / "native_validation.json", report)
    save_json(output / "visual_assets.json", dict(
        status="passed", official_asset_signature=scene["official_asset_signature"],
        assets=scene["visual_settings"],
        gt_scope="Original root trajectories unchanged; visual mesh alignment and evaluated person bones included. Geometry-dependent masks/depth must be regenerated.",
    ))
    unreal.LevelSequenceEditorBlueprintLibrary.close_level_sequence()
    return report
