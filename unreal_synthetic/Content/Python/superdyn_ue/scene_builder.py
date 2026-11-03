"""Native Static Mesh assets, moving actors, CineCamera and Level Sequence."""

import json, math, re
from pathlib import Path
import unreal
from .materials import create_material, create_backdrop_material
from .math3d import compose
from .official_assets import asset_signature, spawn_visual, add_walk_track, load_visuals


def safe_name(value):
    return re.sub("[^A-Za-z0-9_]", "_", value)


def transform(pose):
    roll, pitch, yaw = pose["rotation_rpy_deg"]
    return unreal.Transform(
        location=unreal.Vector(*pose["location_cm"]),
        rotation=unreal.Rotator(pitch=pitch, yaw=yaw, roll=roll),
        scale=unreal.Vector(*pose["scale"]),
    )


def make_mesh(case, record, materials, asset_root="/Game/SuperDyn"):
    name = "SM_" + safe_name(case + "_" + record["key"])
    path = asset_root + "/Meshes/" + name
    existing = unreal.load_asset(path)
    if existing:
        return existing
    dynamic = unreal.DynamicMesh()
    for material_id, section in enumerate(record["sections"]):
        buffers = unreal.GeometryScriptSimpleMeshBuffers(
            vertices=[unreal.Vector(*v) for v in section["vertices_cm"]],
            normals=[unreal.Vector(*v) for v in section["normals"]],
            uv0=[unreal.Vector2D(*v) for v in section["uv0"]],
            # Match Unreal raster front faces while retaining outward normals.
            triangles=[
                unreal.IntVector(v[0], v[2], v[1]) for v in section["triangles"]
            ],
        )
        unreal.GeometryScript_MeshEdits.append_buffers_to_mesh(
            dynamic, buffers, material_id=material_id
        )
    options = unreal.GeometryScriptCreateNewStaticMeshAssetOptions(
        enable_recompute_normals=False,
        enable_recompute_tangents=True,
        enable_nanite=False,
        enable_collision=True,
        collision_mode=unreal.CollisionTraceFlag.CTF_USE_COMPLEX_AS_SIMPLE,
    )
    result = unreal.GeometryScript_NewAssetUtils.create_new_static_mesh_asset_from_mesh(
        dynamic, path, options
    )
    mesh = result[0] if isinstance(result, tuple) else result
    if not isinstance(mesh, unreal.StaticMesh):
        raise RuntimeError(
            "Geometry Script failed to create " + path + ": " + str(result)
        )
    for i, section in enumerate(record["sections"]):
        mesh.set_material(i, materials[section["material"]])
    body = mesh.get_editor_property("body_setup")
    if body:
        body.set_editor_property("double_sided_geometry", True)
    unreal.EditorAssetLibrary.save_loaded_asset(mesh)
    return mesh


def add_transform_track(sequence, actor, poses):
    binding = sequence.add_possessable(actor)
    track = binding.add_track(unreal.MovieScene3DTransformTrack)
    section = track.add_section()
    section.set_range(0, len(poses))
    # Transform sections also expose a float Weight channel; key only the nine
    # double channels for translation, rotation and scale.
    channels = [
        channel
        for channel in unreal.MovieSceneSectionExtensions.get_all_channels(section)
        if isinstance(channel, unreal.MovieSceneScriptingDoubleChannel)
    ]
    if len(channels) != 9:
        raise RuntimeError(
            "Expected nine transform channels, got " + str(len(channels))
        )
    for frame, pose in enumerate(poses):
        values = pose["location_cm"] + pose["rotation_rpy_deg"] + pose["scale"]
        for channel, value in zip(channels, values):
            channel.add_key(
                unreal.FrameNumber(frame),
                float(value),
                time_unit=unreal.MovieSceneTimeUnit.DISPLAY_RATE,
                interpolation=unreal.MovieSceneKeyInterpolation.LINEAR,
            )
    return binding


def initialize_terrain_collision(actors):
    mesh_editor = unreal.get_editor_subsystem(unreal.StaticMeshEditorSubsystem)
    for actor in actors.values():
        component = actor.get_component_by_class(unreal.StaticMeshComponent)
        if component is None:
            continue
        mesh = component.get_editor_property("static_mesh")
        if mesh is None:  # Empty transform anchors for official skeletal visuals.
            continue
        # GetRenderData waits for asynchronous static mesh compilation.
        if mesh_editor.get_number_verts(mesh, 0) <= 0:
            raise RuntimeError("Static mesh render data is not ready")
    component = actors["terrain"].get_component_by_class(unreal.StaticMeshComponent)
    component.set_collision_profile_name("BlockAll")
    # Recreate query state after loading a world in an editor session.
    component.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
    component.set_collision_enabled(unreal.CollisionEnabled.QUERY_AND_PHYSICS)


def add_lighting(editor, project_root):
    task = unreal.AssetImportTask()
    task.set_editor_property(
        "filename", str(project_root / "inputs/lighting/environment.hdr")
    )
    task.set_editor_property("destination_path", "/Game/SuperDyn/Lighting")
    task.set_editor_property("destination_name", "Environment")
    task.set_editor_property("automated", True)
    task.set_editor_property("replace_existing", False)
    task.set_editor_property("save", True)
    hdr = unreal.load_asset("/Game/SuperDyn/Lighting/Environment")
    if not hdr:
        unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
        paths = task.get_editor_property("imported_object_paths")
        if not paths:
            raise RuntimeError("HDRI import failed")
        hdr = unreal.load_asset(paths[0])
    if not isinstance(hdr, unreal.TextureCube):
        raise RuntimeError("HDRI must import as a TextureCube")
    sky = editor.spawn_actor_from_class(unreal.SkyLight, unreal.Vector(0, 0, 1000))
    sky.set_actor_label("SD_SkyLight")
    component = sky.get_component_by_class(unreal.SkyLightComponent)
    component.set_editor_property(
        "source_type", unreal.SkyLightSourceType.SLS_SPECIFIED_CUBEMAP
    )
    component.set_editor_property("cubemap", hdr)
    component.set_editor_property("intensity", 0.8)
    component.set_mobility(unreal.ComponentMobility.MOVABLE)
    sun = editor.spawn_actor_from_class(
        unreal.DirectionalLight, unreal.Vector(0, 0, 1500)
    )
    sun.set_actor_rotation(unreal.Rotator(pitch=-42, yaw=-35, roll=0), False)
    sun.set_actor_label("SD_Sun")
    sun_component = sun.get_component_by_class(unreal.DirectionalLightComponent)
    sun_component.set_mobility(unreal.ComponentMobility.MOVABLE)
    sun_component.set_editor_property("intensity", 3.0)
    backdrop = editor.spawn_actor_from_class(
        unreal.StaticMeshActor, unreal.Vector(0, 0, 0)
    )
    backdrop.set_actor_label("SD_Backdrop")
    comp = backdrop.get_component_by_class(unreal.StaticMeshComponent)
    comp.set_static_mesh(unreal.load_asset("/Engine/BasicShapes/Sphere.Sphere"))
    comp.set_material(0, create_backdrop_material())
    comp.set_editor_property("cast_shadow", False)
    comp.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
    backdrop.set_actor_scale3d(unreal.Vector(1000, 1000, 1000))
    return [backdrop]


def build(packet, project_root, input_sha, rebuild=False, theme=None):
    unreal.LevelSequenceEditorBlueprintLibrary.close_level_sequence()
    official_signature = asset_signature(project_root)
    case = packet["case_id"]
    if safe_name(case) != case:
        raise ValueError("Case ID must contain only letters, digits and underscores")
    asset_root = "/Game/SuperDyn" if theme is None else "/Game/SuperDyn/Themes/" + safe_name(theme)
    theme_signature = ""
    if theme is not None:
        from .themes import signature
        theme_signature = signature(theme, project_root)
    map_path = asset_root + "/Maps/" + case
    sequence_path = asset_root + "/Sequences/LS_" + case
    levels = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    reuse_map = False
    if rebuild:
        unreal.LevelSequenceEditorBlueprintLibrary.close_level_sequence()
        # World assets cannot be reset using the regular asset deletion helper.
        # Reuse the map and remove only our generated actors instead.
        if unreal.EditorAssetLibrary.does_asset_exist(map_path):
            if not levels.load_level(map_path):
                raise RuntimeError("Cannot open generated map for rebuilding")
            editor = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
            for actor in editor.get_all_level_actors():
                if actor.get_actor_label().startswith("SD_"):
                    if not editor.destroy_actor(actor):
                        raise RuntimeError("Cannot reset generated actor")
            reuse_map = True
        # Only exact generated assets belonging to the requested case are reset.
        assets = [sequence_path] + [
            asset_root + "/Meshes/SM_" + safe_name(case + "_" + record["key"])
            for record in packet["meshes"]
        ]
        for path in assets:
            if unreal.EditorAssetLibrary.does_asset_exist(path):
                if not unreal.EditorAssetLibrary.delete_asset(path):
                    raise RuntimeError("Cannot reset generated case asset " + path)
    # Registry removal notifications can lag behind a successful force delete.
    exists_map = (
        False if rebuild else unreal.EditorAssetLibrary.does_asset_exist(map_path)
    )
    exists_sequence = (
        False if rebuild else unreal.EditorAssetLibrary.does_asset_exist(sequence_path)
    )
    if exists_map != exists_sequence:
        raise RuntimeError(
            "Partial generated case exists; resolve it in the editor before rebuilding"
        )
    if exists_map:
        if theme is None:
            for key, value in packet["materials"].items():
                create_material("M_" + safe_name(key), value)
        sequence = unreal.load_asset(sequence_path)
        if (
            unreal.EditorAssetLibrary.get_metadata_tag(sequence, "SuperDynInputSHA")
            != input_sha
        ):
            raise RuntimeError(
                "Existing case was built from different input; use a new case asset name"
            )
        if unreal.EditorAssetLibrary.get_metadata_tag(sequence, "SuperDynOfficialAssets") != official_signature:
            raise RuntimeError("Existing scene uses different visual assets; run with --rebuild")
        if unreal.EditorAssetLibrary.get_metadata_tag(sequence, "SuperDynTheme") != theme_signature:
            raise RuntimeError("Existing theme settings differ; run with --rebuild")
        levels = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
        if not levels.load_level(map_path):
            raise RuntimeError("Cannot load existing map")
        editor = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        all_actors = editor.get_all_level_actors()
        by_label = {a.get_actor_label(): a for a in all_actors}
        actors = {m["key"]: by_label["SD_" + m["key"]] for m in packet["meshes"]}
        initialize_terrain_collision(actors)
        return dict(
            packet=packet,
            actors=actors,
            camera=by_label["SD_Camera"],
            sequence=sequence,
            map_path=map_path,
            sequence_path=sequence_path,
            ignored=[a for a in all_actors if a.get_actor_label().startswith("SD_Theme_")] + [by_label["SD_Backdrop"]],
            visuals=load_visuals(by_label),
            visual_settings=json.loads(unreal.EditorAssetLibrary.get_metadata_tag(sequence, "SuperDynVisualSettings")),
            official_asset_signature=official_signature,
            theme_settings=json.loads(unreal.EditorAssetLibrary.get_metadata_tag(sequence, "SuperDynThemeSettings") or "null"),
        )
    for folder in ["Maps", "Sequences", "Meshes", "Materials", "Lighting"]:
        unreal.EditorAssetLibrary.make_directory(asset_root + "/" + folder)
    levels = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    if not reuse_map and not levels.new_level(map_path):
        raise RuntimeError("Cannot create map " + map_path)
    editor = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    materials = {
        key: create_material("M_" + safe_name(key), value)
        for key, value in packet["materials"].items()
    }
    if theme is not None:
        from .themes import terrain_materials
        themed = terrain_materials(theme, project_root)
        for key in materials:
            if key.startswith("terrain"):
                materials[key] = themed["wall" if "retaining" in key else "ground"]
    actors = {}
    visuals = {}
    visual_settings = {}
    for record in packet["meshes"]:
        replacement = record["key"] in {"person", "ground"}
        mesh = None if replacement else make_mesh(case, record, materials, asset_root)
        actor = editor.spawn_actor_from_class(
            unreal.StaticMeshActor, unreal.Vector(0, 0, 0)
        )
        actor.set_actor_label("SD_" + record["key"])
        component = actor.get_component_by_class(unreal.StaticMeshComponent)
        component.set_mobility(
            unreal.ComponentMobility.STATIC
            if record["key"] == "terrain"
            else unreal.ComponentMobility.MOVABLE
        )
        if mesh is not None:
            component.set_static_mesh(mesh)
        parent = record.get("parent")
        iid = packet["instance_ids"].get(parent or record["key"], 100)
        component.set_editor_property("render_custom_depth", True)
        component.set_editor_property("custom_depth_stencil_value", iid)
        if record["key"] != "terrain":
            component.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
        if parent:
            actor.attach_to_actor(
                actors[parent],
                "None",
                unreal.AttachmentRule.KEEP_RELATIVE,
                unreal.AttachmentRule.KEEP_RELATIVE,
                unreal.AttachmentRule.KEEP_RELATIVE,
                False,
            )
            actor.set_actor_relative_transform(
                transform(packet["tracks"][record["key"]][0]), False, True
            )
        elif record["key"] in packet["tracks"]:
            actor.set_actor_transform(
                transform(packet["tracks"][record["key"]][0]), False, True
            )
        actors[record["key"]] = actor
        if replacement:
            visuals[record["key"]], visual_settings[record["key"]] = spawn_visual(
                editor, actor, record, case, iid, transform
            )
    initialize_terrain_collision(actors)
    ignored = add_lighting(editor, project_root)
    theme_settings = None
    if theme is not None:
        from .themes import apply
        props, theme_settings = apply(theme, dict(packet=packet, actors=actors, visuals=visuals), editor, project_root)
        ignored.extend(props)
    camera = editor.spawn_actor_from_class(
        unreal.CineCameraActor, unreal.Vector(0, 0, 0)
    )
    camera.set_actor_label("SD_Camera")
    component = camera.get_cine_camera_component()
    component.set_editor_property(
        "filmback",
        unreal.CameraFilmbackSettings(
            sensor_width=36.0, sensor_height=36.0 * packet["height"] / packet["width"]
        ),
    )
    component.set_editor_property(
        "current_focal_length",
        36.0 / (2 * math.tan(math.radians(packet["horizontal_fov_degrees"] / 2))),
    )
    focus = component.get_editor_property("focus_settings")
    focus.set_editor_property("focus_method", unreal.CameraFocusMethod.DISABLE)
    component.set_editor_property("focus_settings", focus)
    post = component.get_editor_property("post_process_settings")
    post.set_editor_property("override_motion_blur_amount", True)
    post.set_editor_property("motion_blur_amount", 0.0)
    post.set_editor_property("override_auto_exposure_method", True)
    post.set_editor_property(
        "auto_exposure_method", unreal.AutoExposureMethod.AEM_MANUAL
    )
    post.set_editor_property(
        "override_auto_exposure_apply_physical_camera_exposure", True
    )
    post.set_editor_property("auto_exposure_apply_physical_camera_exposure", False)
    component.set_editor_property("post_process_settings", post)
    camera.set_actor_transform(transform(packet["camera"][0]), False, True)
    sequence = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
        "LS_" + case,
        asset_root + "/Sequences",
        unreal.LevelSequence,
        unreal.LevelSequenceFactoryNew(),
    )
    sequence.set_display_rate(unreal.FrameRate(packet["fps"], 1))
    sequence.set_tick_resolution_directly(unreal.FrameRate(packet["fps"] * 1000, 1))
    sequence.set_playback_start(0)
    sequence.set_playback_end(packet["frames"])
    for key, poses in packet["tracks"].items():
        add_transform_track(sequence, actors[key], poses)
    visual_settings["person"].update(add_walk_track(sequence, visuals["person"], packet))
    binding = add_transform_track(sequence, camera, packet["camera"])
    cut = sequence.add_track(unreal.MovieSceneCameraCutTrack).add_section()
    cut.set_range(0, packet["frames"])
    binding_id = unreal.MovieSceneObjectBindingID()
    binding_id.set_editor_property("guid", binding.get_id())
    cut.set_camera_binding_id(binding_id)
    playback = editor.spawn_actor_from_class(
        unreal.LevelSequenceActor, unreal.Vector(0, 0, 0)
    )
    playback.set_actor_label("SD_Sequence")
    playback.set_sequence(sequence)
    # MRQ owns playback in PIE; a second autoplay player would fight its keys.
    playback_settings = playback.get_editor_property("playback_settings")
    playback_settings.set_editor_property("auto_play", False)
    playback_settings.set_editor_property(
        "loop_count", unreal.MovieSceneSequenceLoopCount(value=-1)
    )
    playback.set_editor_property("playback_settings", playback_settings)
    unreal.EditorAssetLibrary.set_metadata_tag(sequence, "SuperDynInputSHA", input_sha)
    unreal.EditorAssetLibrary.set_metadata_tag(sequence, "SuperDynOfficialAssets", official_signature)
    unreal.EditorAssetLibrary.set_metadata_tag(sequence, "SuperDynVisualSettings", json.dumps(visual_settings))
    unreal.EditorAssetLibrary.set_metadata_tag(sequence, "SuperDynTheme", theme_signature)
    unreal.EditorAssetLibrary.set_metadata_tag(sequence, "SuperDynThemeSettings", json.dumps(theme_settings))
    unreal.EditorAssetLibrary.save_loaded_asset(sequence)
    levels.save_current_level()
    return dict(
        packet=packet,
        actors=actors,
        camera=camera,
        sequence=sequence,
        map_path=map_path,
        sequence_path=sequence_path,
        ignored=ignored,
        visuals=visuals,
        visual_settings=visual_settings,
        official_asset_signature=official_signature,
        theme_settings=theme_settings,
    )
