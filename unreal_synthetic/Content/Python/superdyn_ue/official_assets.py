"""Native Epic template visuals attached to unchanged ground-truth anchors."""

import hashlib
import json
from pathlib import Path
import unreal
from .asset_layout import (
    ASSET_REVISION, WALK_CLIP, OFFROAD_MESH, OFFROAD_BODY, OFFROAD_TIRE, person_mesh, record_bounds,
    fit_visual, walking_play_rate,
)
from .io_utils import sha256


def bounds_points(bounds):
    center, extent = bounds.origin, bounds.box_extent
    return [unreal.Vector(center.x + x * extent.x, center.y + y * extent.y, center.z + z * extent.z)
            for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]]


def add_vehicle_parts(editor, suspension, component, iid):
    """Body uses the mesh component frame; tires use VisWheel sockets."""
    parts = []
    points = []
    for name, socket, path in [
        ("body", "", OFFROAD_BODY),
        *[("tire_" + side, "VisWheel_" + side, OFFROAD_TIRE) for side in ["FL", "FR", "BL", "BR"]],
    ]:
        if socket and component.get_bone_index(socket) < 0:
            raise RuntimeError("Official vehicle socket is missing: " + socket)
        mesh = unreal.load_asset(path)
        if not isinstance(mesh, unreal.StaticMesh):
            raise RuntimeError("Official vehicle part could not be loaded: " + path)
        mesh_editor = unreal.get_editor_subsystem(unreal.StaticMeshEditorSubsystem)
        if mesh_editor.get_number_verts(mesh, 0) <= 0:
            raise RuntimeError("Vehicle part render data is not ready: " + path)
        actor = editor.spawn_actor_from_class(unreal.StaticMeshActor, unreal.Vector())
        label = "SD_ground_" + name
        actor.set_actor_label(label)
        part = actor.get_component_by_class(unreal.StaticMeshComponent)
        part.set_mobility(unreal.ComponentMobility.MOVABLE)
        part.set_static_mesh(mesh)
        part.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
        part.set_editor_property("render_custom_depth", True)
        part.set_editor_property("custom_depth_stencil_value", iid)
        actor.attach_to_actor(suspension, socket or "None", unreal.AttachmentRule.KEEP_RELATIVE,
                              unreal.AttachmentRule.KEEP_RELATIVE, unreal.AttachmentRule.KEEP_RELATIVE, False)
        actor.set_actor_relative_transform(unreal.Transform(), False, True)
        # The skeleton's bone named Root is rotated -90 degrees around X.
        # Body geometry is already authored in component space, so attaching
        # it to that bone would tip the body onto its side.
        socket_transform = component.get_socket_transform(socket, unreal.RelativeTransformSpace.RTS_COMPONENT) if socket else unreal.Transform()
        points.extend(unreal.MathLibrary.transform_location(socket_transform, v) for v in bounds_points(mesh.get_bounds()))
        parts.append(dict(label=label, mesh=path, socket=socket))
    return parts, points


def asset_signature(project_root):
    receipt = Path(project_root) / "inputs/official_assets/license_and_source.json"
    if not receipt.is_file():
        raise RuntimeError("Run scripts/import_official_assets.py before building scenes")
    data = json.loads(receipt.read_text())
    for name, expected in data["files"].items():
        file = Path(project_root) / name
        if not file.is_file() or sha256(file) != expected:
            raise RuntimeError("Official content differs from its receipt: " + name)
    return hashlib.sha256((ASSET_REVISION + receipt.read_text()).encode()).hexdigest()


def spawn_visual(editor, anchor, record, case, iid, make_transform):
    key = record["key"]
    path = person_mesh(case) if key == "person" else OFFROAD_MESH
    mesh = unreal.load_asset(path)
    if not isinstance(mesh, unreal.SkeletalMesh):
        raise RuntimeError("Official Skeletal Mesh could not be loaded: " + path)
    actor = editor.spawn_actor_from_class(unreal.SkeletalMeshActor, unreal.Vector())
    actor.set_actor_label("SD_" + key + "_visual")
    component = actor.get_component_by_class(unreal.SkeletalMeshComponent)
    component.set_mobility(unreal.ComponentMobility.MOVABLE)
    component.set_skeletal_mesh_asset(mesh)
    component.set_simulate_physics(False)
    component.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
    component.set_editor_property("render_custom_depth", True)
    component.set_editor_property("custom_depth_stencil_value", iid)
    component.set_editor_property(
        "visibility_based_anim_tick_option",
        unreal.VisibilityBasedAnimTickOption.ALWAYS_TICK_POSE_AND_REFRESH_BONES,
    )
    component.set_update_animation_in_editor(True)
    component.set_animation_mode(unreal.AnimationMode.ANIMATION_SINGLE_NODE)
    points = bounds_points(mesh.get_imported_bounds())
    parts = []
    if key == "ground":
        parts, part_points = add_vehicle_parts(editor, actor, component, iid)
        points.extend(part_points)
    xyz = [[v.x, v.y, v.z] for v in points]
    low = [min(v[i] for v in xyz) for i in range(3)]
    high = [max(v[i] for v in xyz) for i in range(3)]
    target_min, target_max = record_bounds(record)
    pose = fit_visual(low, high, target_min, target_max, person=key == "person")
    actor.attach_to_actor(
        anchor, "None", unreal.AttachmentRule.KEEP_RELATIVE,
        unreal.AttachmentRule.KEEP_RELATIVE, unreal.AttachmentRule.KEEP_RELATIVE, False,
    )
    actor.set_actor_relative_transform(make_transform(pose), False, True)
    if any(component.get_material(i) is None for i in range(component.get_num_materials())):
        raise RuntimeError("Official visual has missing materials: " + path)
    return actor, dict(
        mesh=path, local_pose=pose, source_bounds_cm=[low, high],
        original_bounds_cm=[target_min, target_max], instance_id=iid,
        parts=parts,
    )


def add_walk_track(sequence, actor, packet):
    clip = unreal.load_asset(WALK_CLIP)
    if not isinstance(clip, unreal.AnimSequence):
        raise RuntimeError("Official forward walk animation could not be loaded")
    binding = sequence.add_possessable(actor)
    section = binding.add_track(unreal.MovieSceneSkeletalAnimationTrack).add_section()
    section.set_range(0, packet["frames"])
    params = section.get_editor_property("params")
    params.set_editor_property("animation", clip)
    rate = walking_play_rate(packet["tracks"]["person"], packet["fps"])
    params.set_editor_property("play_rate", unreal.MovieSceneTimeWarpExtensions.make_time_warp(rate))
    params.set_editor_property("force_custom_mode", True)
    params.set_editor_property("skip_anim_notifiers", True)
    section.set_editor_property("params", params)
    return dict(animation=WALK_CLIP, play_rate=rate)


def load_visuals(by_label):
    return {key: by_label["SD_" + key + "_visual"] for key in ["person", "ground"]}
