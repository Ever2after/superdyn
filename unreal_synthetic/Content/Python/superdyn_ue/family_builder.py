"""Build coherent scene families from validated terrain and installed assets."""

import json
import math
from pathlib import Path
import random

import unreal

from .family_geometry import Ground, fits
from .io_utils import load_scene, save_json, sha256
from .scene_builder import make_mesh, transform, add_transform_track, initialize_terrain_collision, add_lighting
from .theme_materials import pbr_material, solid
from .hdri import receipt as hdri_receipt
from .audit import matrix_from_actor


def bounds(mesh):
    box = mesh.get_imported_bounds() if isinstance(mesh, unreal.SkeletalMesh) else mesh.get_bounds()
    low = [getattr(box.origin, k) - getattr(box.box_extent, k) for k in "xyz"]
    high = [getattr(box.origin, k) + getattr(box.box_extent, k) for k in "xyz"]
    return low, high


def configure(component, iid, static=False):
    component.set_mobility(unreal.ComponentMobility.STATIC if static else unreal.ComponentMobility.MOVABLE)
    component.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
    component.set_editor_property("render_custom_depth", True)
    component.set_editor_property("custom_depth_stencil_value", iid)


def cube(editor, name, center, size, material, iid=3):
    actor = editor.spawn_actor_from_class(unreal.StaticMeshActor, unreal.Vector())
    actor.set_actor_label("SD_Main_" + name)
    component = actor.get_component_by_class(unreal.StaticMeshComponent)
    component.set_static_mesh(unreal.load_asset("/Engine/BasicShapes/Cube.Cube"))
    configure(component, iid, static=True)
    component.set_material(0, material)
    actor.set_actor_transform(transform(dict(location_cm=[v * 100 for v in center], rotation_rpy_deg=[0, 0, 0], scale=list(size))), False, True)
    return actor


def facade_panel(editor, name, path, center, width, height):
    mesh = unreal.load_asset(path)
    if not isinstance(mesh, unreal.StaticMesh):
        raise RuntimeError("Building module is missing: " + path)
    low, high = bounds(mesh)
    dims = [b - a for a, b in zip(low, high)]
    horizontal = 0 if dims[0] >= dims[1] else 1
    scale = [1, 1, height * 100 / dims[2]]
    scale[horizontal] = width * 100 / dims[horizontal]
    # These modules' authored exterior faces point along +Y. The plaza views
    # the south facade, so turn the authored face outward toward -Y.
    yaw = 180 if horizontal == 0 else -90
    local = [(a + b) / 2 * s for a, b, s in zip(low, high, scale)]
    angle = math.radians(yaw)
    offset = [local[0] * math.cos(angle) - local[1] * math.sin(angle), local[0] * math.sin(angle) + local[1] * math.cos(angle), local[2]]
    actor = editor.spawn_actor_from_class(unreal.StaticMeshActor, unreal.Vector())
    actor.set_actor_label("SD_Main_" + name)
    comp = actor.get_component_by_class(unreal.StaticMeshComponent)
    comp.set_static_mesh(mesh)
    configure(comp, 3, static=True)
    actor.set_actor_transform(transform(dict(location_cm=[v * 100 - o for v, o in zip(center, offset)], rotation_rpy_deg=[0, 0, yaw], scale=scale)), False, True)
    return actor


def lighting(editor, root):
    ignored = add_lighting(editor, root)
    source = hdri_receipt(root)
    cube_path = "/Game/SuperDyn/Lighting/PolyHaven_" + source["asset"] + "_4k"
    sky_cube = unreal.load_asset(cube_path)
    sky_material = unreal.load_asset("/Game/SuperDyn/ThemeMaterials/M_PolyHavenPureSky")
    if sky_cube is None or sky_material is None:
        raise RuntimeError("Prepared Poly Haven lighting assets are missing")
    actors = {a.get_actor_label(): a for a in editor.get_all_level_actors()}
    actors["SD_Backdrop"].get_component_by_class(unreal.StaticMeshComponent).set_material(0, sky_material)
    sky = actors["SD_SkyLight"].get_component_by_class(unreal.SkyLightComponent)
    sky.set_editor_property("cubemap", sky_cube)
    sky.set_editor_property("intensity", float(source["skylight_intensity"]) * 3)
    actors["SD_Sun"].set_actor_rotation(unreal.Rotator(**source["directional_light_rotation"]), False)
    sun = actors["SD_Sun"].get_component_by_class(unreal.DirectionalLightComponent)
    sun.set_editor_property("intensity", 3.0)
    sun.set_editor_property("light_source_angle", 0.55)
    return ignored, source


def architecture(editor, family, ground, prefix):
    """Complete volumes at the scene edge; no isolated single-sided facades."""
    a, b, c, d = ground.bounds
    actors = []
    if family == "industrial":
        metal = solid(prefix + "Warehouse", [0.12, 0.19, 0.23], metallic=0.3)
        door = solid(prefix + "LoadingDoor", [0.22, 0.24, 0.23], metallic=0.5)
        trim = solid(prefix + "WarehouseTrim", [0.48, 0.43, 0.31], metallic=0.4)
        floor = ground.z((a + b) / 2, d - 0.01)
        top = floor + 5.5
        actors.append(cube(editor, "warehouse", [0, d + 4.0, (top - 0.5) / 2], [17, 6.5, top + 0.5], metal))
        actors.append(cube(editor, "warehouse_roof", [0, d + 4, top + 0.12], [17.6, 7, 0.24], trim))
        for i, x in enumerate([-5, 0, 5]):
            actors.append(cube(editor, f"loading_door_{i}", [x, d + 0.72, floor + 1.7], [3.6, 0.08, 3.4], door))
            for row in range(7):
                actors.append(cube(editor, f"door_rib_{i}_{row}", [x, d + 0.66, floor + 0.3 + row * 0.45], [3.5, 0.08, 0.035], trim))
    elif family == "urban":
        colors = [[0.48, 0.32, 0.23], [0.35, 0.39, 0.40], [0.60, 0.52, 0.36]]
        roof = solid(prefix + "Roof", [0.16, 0.17, 0.17])
        brick = "/Game/ModularBuildingSet/Meshes/Upper_Brick_Wndws/upper_brick_wall_a_l_2x2"
        storefront = "/Game/ModularBuildingSet/Meshes/Store_fronts/store_front_windows_intact_4x1_a"
        upper_windows = "/Game/ModularBuildingSet/Meshes/Store_Upper_Wndws/windows_building_upper_4x1_a"
        for i, x in enumerate([-8, 0, 8]):
            base = ground.z(min(b - 0.1, max(a + 0.1, x)), d - 0.01)
            top = base + [8.3, 10.3, 7.3][i]
            body = solid(prefix + f"Facade_{i}", colors[i])
            actors.append(cube(editor, f"building_{i}", [x, d + 3.4, (top - 0.5) / 2], [7.4, 6, top + 0.5], body))
            actors.append(cube(editor, f"roof_{i}", [x, d + 3.4, top], [7.8, 6.4, 0.25], roof))
            for col in [-1, 1]:
                actors.append(facade_panel(editor, f"shopfront_{i}_{col}", storefront, [x + col * 1.8, d + 0.22, base + 1.35], 3.6, 2.7))
            for row in range(1, 3):
                for col in range(3):
                    xx, zz = x + (col - 1) * 2.4, base + 1.35 + row * 2.5
                    if zz + 1.25 > top:
                        continue
                    actors.append(facade_panel(editor, f"brick_facade_{i}_{row}_{col}", brick, [xx, d + 0.24, zz], 2.4, 2.5))
                zz = base + 1.35 + row * 2.5
                if zz + 0.65 <= top:
                    for col in [-1, 1]:
                        actors.append(facade_panel(editor, f"upper_window_{i}_{row}_{col}", upper_windows, [x + col * 1.8, d + 0.04, zz], 3.6, 1.3))
    if family == "park":
        paths = ["/Game/Modular_Rural_Cabin/Meshes/Foliage/SM_Pine_Tree_01", "/Game/CityPark/Meshes/Flora/Trees/SM_AmurCork01"]
        for i in range(20):
            angle = 2 * math.pi * i / 20 + 0.17
            x = (b - a) * 0.72 * math.cos(angle)
            y = (d - c) * 0.72 * math.sin(angle)
            mesh = unreal.load_asset(paths[i % 2])
            low, high = bounds(mesh)
            scale = (700 + 90 * (i % 3)) / (high[2] - low[2])
            actor = editor.spawn_actor_from_class(unreal.StaticMeshActor, unreal.Vector())
            actor.set_actor_label(f"SD_Main_BackgroundTree_{i:02d}")
            component = actor.get_component_by_class(unreal.StaticMeshComponent)
            component.set_static_mesh(mesh)
            configure(component, 3, static=True)
            actor.set_actor_transform(transform(dict(location_cm=[x * 100, y * 100, ground.context_z(x, y) * 100 - low[2] * scale], rotation_rpy_deg=[0, 0, (i * 137) % 360], scale=[scale] * 3)), False, True)
            actors.append(actor)
    return actors


def make_sequence(camera, poses, packet, name, path):
    sequence = unreal.AssetToolsHelpers.get_asset_tools().create_asset(name, path, unreal.LevelSequence, unreal.LevelSequenceFactoryNew())
    if sequence is None:
        raise RuntimeError("Sequence already exists: " + path + "/" + name)
    sequence.set_display_rate(unreal.FrameRate(packet["fps"], 1))
    sequence.set_tick_resolution_directly(unreal.FrameRate(packet["fps"] * 1000, 1))
    sequence.set_playback_start(0)
    sequence.set_playback_end(len(poses))
    binding = add_transform_track(sequence, camera, poses)
    cut = sequence.add_track(unreal.MovieSceneCameraCutTrack).add_section()
    cut.set_range(0, len(poses))
    bid = unreal.MovieSceneObjectBindingID()
    bid.set_editor_property("guid", binding.get_id())
    cut.set_camera_binding_id(bid)
    return sequence


def set_frame(sequence, frame):
    unreal.LevelSequenceEditorBlueprintLibrary.open_level_sequence(sequence)
    params = unreal.MovieSceneSequencePlaybackParams(frame=unreal.FrameTime(unreal.FrameNumber(frame)), position_type=unreal.MovieScenePositionType.FRAME, update_method=unreal.UpdatePositionMethod.JUMP)
    unreal.LevelSequenceEditorBlueprintLibrary.set_global_position(params, unreal.MovieSceneTimeUnit.DISPLAY_RATE)
    unreal.LevelSequenceEditorBlueprintLibrary.force_update()


def build(job, root):
    folder = Path(job["output"])
    packet = load_scene(folder / "scene.json.gz")
    if sha256(folder / "scene.json.gz") != job["input_sha256"]:
        raise RuntimeError("Terrain packet changed")
    recipe = json.loads((folder / "recipe.json").read_text())
    layout = json.loads((folder / "layout.json").read_text())
    route = json.loads((folder / "route.json").read_text())
    family = recipe["family"]
    ground = Ground(layout)
    levels = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    editor = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    unreal.LevelSequenceEditorBlueprintLibrary.close_level_sequence()
    if unreal.EditorAssetLibrary.does_asset_exist(job["map_path"]):
        raise RuntimeError("Generated map already exists; use a fresh output namespace")
    if not levels.new_level(job["map_path"]):
        raise RuntimeError("Cannot create scene map")
    for directory in ["Meshes", "Sequences", "Motion"]:
        unreal.EditorAssetLibrary.make_directory(job["asset_root"] + "/" + directory)
    prefix = "Main_" + job["asset_root"].split("/")[-2] + "_" + family["id"] + "_"
    mats = dict(
        ground=pbr_material(root, prefix + "Ground", family["ground_texture"], family["ground_tile_cm"]),
        wall=pbr_material(root, prefix + "Retaining", family["wall_texture"], 220, tint=family["wall_tint"]),
        path=pbr_material(root, prefix + "Path", family["path_texture"], 180),
    )
    terrain_actors = {}
    for record in packet["meshes"]:
        mesh = make_mesh(family["id"], record, mats, job["asset_root"])
        actor = editor.spawn_actor_from_class(unreal.StaticMeshActor, unreal.Vector())
        actor.set_actor_label("SD_Main_" + record["key"])
        component = actor.get_component_by_class(unreal.StaticMeshComponent)
        component.set_static_mesh(mesh)
        configure(component, 1 if record["key"] == "terrain" else 2, static=True)
        terrain_actors[record["key"]] = actor
    initialize_terrain_collision({"terrain": terrain_actors["terrain"]})
    ignored, sky = lighting(editor, root)
    architectural = architecture(editor, family["id"], ground, prefix)
    rng = random.Random(family["seed"])
    rects = sorted(layout["generation"]["flat_rectangles"], key=lambda x: -x["area_m2"])
    a, b, c, d = ground.bounds
    candidates = [(a + 0.5 * ix, c + 0.5 * iy) for ix in range(1, math.floor((b - a) / 0.5)) for iy in range(1, math.floor((d - c) / 0.5))]
    placed, records, skeletal, object_actors = [], [], [], []
    specs = sorted(recipe["assets"], key=lambda s: (-s["size_cm"] if s["asset"] not in ("tree", "pine", "lamp") else -100))
    for index, spec in enumerate(specs):
        mesh = unreal.load_asset(spec["mesh"])
        if not isinstance(mesh, (unreal.StaticMesh, unreal.SkeletalMesh)):
            raise RuntimeError("Cannot load mesh " + spec["mesh"])
        skinned = isinstance(mesh, unreal.SkeletalMesh)
        low, high = bounds(mesh)
        dimensions = [hi - lo for lo, hi in zip(low, high)]
        divisor = dimensions[2] if spec.get("height_size") else max(dimensions)
        scale = spec["size_cm"] / divisor
        yaw = [0, 90, 180, 270][rng.randrange(4)]
        cosine, sine = abs(math.cos(math.radians(yaw))), abs(math.sin(math.radians(yaw)))
        half = [(dimensions[0] * cosine + dimensions[1] * sine) * scale / 200, (dimensions[0] * sine + dimensions[1] * cosine) * scale / 200]
        tree = spec["asset"] in ("tree", "pine")
        rectangle = rects[(index // 3) % len(rects)]
        x0, x1, y0, y1 = rectangle["bounds"]
        preferred = [(x0 + x1) / 2 + rng.uniform(-1, 1), (y0 + y1) / 2 + rng.uniform(-1, 1)]
        ordered = sorted(candidates, key=lambda p: math.dist(p, preferred))
        found = None
        for xy in ordered:
            z = fits(ground, xy, half, placed, route["reserved"], tree=tree)
            if z is not None:
                found = (xy, z)
                break
        if found is None:
            raise RuntimeError("No flat, collision-free position for " + spec["id"])
        xy, z = found
        actor = editor.spawn_actor_from_class(unreal.SkeletalMeshActor if skinned else unreal.StaticMeshActor, unreal.Vector())
        actor.set_actor_label("SD_Main_" + spec["id"])
        component = actor.get_component_by_class(unreal.SkeletalMeshComponent if skinned else unreal.StaticMeshComponent)
        configure(component, index + 10)
        if skinned:
            component.set_skeletal_mesh_asset(mesh)
            component.set_simulate_physics(False)
            component.set_update_animation_in_editor(True)
            component.set_editor_property("visibility_based_anim_tick_option", unreal.VisibilityBasedAnimTickOption.ALWAYS_TICK_POSE_AND_REFRESH_BONES)
        else:
            component.set_static_mesh(mesh)
            unreal.get_editor_subsystem(unreal.StaticMeshEditorSubsystem).get_number_verts(mesh, 0)
        center = [(lo + hi) / 2 * scale for lo, hi in zip(low, high)]
        angle = math.radians(yaw)
        offset = [center[0] * math.cos(angle) - center[1] * math.sin(angle), center[0] * math.sin(angle) + center[1] * math.cos(angle)]
        raised = 240 if spec["asset"] == "shelter" else 0
        pose = dict(location_cm=[xy[0] * 100 - offset[0], xy[1] * 100 - offset[1], z * 100 - low[2] * scale + raised + 0.5], rotation_rpy_deg=[0, 0, yaw], scale=[scale] * 3)
        actor.set_actor_transform(transform(pose), False, True)
        if any(component.get_material(i) is None for i in range(component.get_num_materials())):
            raise RuntimeError("Missing material: " + spec["id"])
        record = dict(spec, actor=actor.get_actor_label(), anchor_m=list(xy), ground_z_m=z, half_xy_m=half,
                      collision_half_xy_m=[0.4, 0.4] if tree else half, transform=pose, instance_id=index + 10)
        records.append(record)
        placed.append((xy[0], xy[1], *record["collision_half_xy_m"]))
        object_actors.append(actor)
        if spec.get("animation"):
            clip = unreal.load_asset(spec["animation"])
            if not isinstance(clip, unreal.AnimSequence) or clip.get_editor_property("skeleton") != mesh.get_editor_property("skeleton"):
                raise RuntimeError("Animation skeleton mismatch: " + spec["id"])
            local_path = job["asset_root"] + "/Motion/" + spec["id"]
            clip = unreal.EditorAssetLibrary.duplicate_asset(spec["animation"], local_path)
            clip.set_editor_property("enable_root_motion", True)
            clip.set_editor_property("force_root_lock", True)
            clip.set_editor_property("root_motion_root_lock", unreal.RootMotionRootLock.ANIM_FIRST_FRAME)
            unreal.EditorAssetLibrary.save_loaded_asset(clip)
            component.set_animation_mode(unreal.AnimationMode.ANIMATION_SINGLE_NODE)
            component.play_animation(clip, True)
            skeletal.append((actor, component, clip, record))
        if spec["asset"] == "shelter":
            wood = solid(prefix + "ShelterPosts", [0.16, 0.09, 0.045])
            for ix in [-1, 1]:
                for iy in [-1, 1]:
                    architectural.append(cube(editor, f"shelter_post_{ix}_{iy}", [xy[0] + ix * half[0] * 0.76, xy[1] + iy * half[1] * 0.76, z + 1.2], [0.16, 0.16, 2.4], wood, index + 10))
    camera = editor.spawn_actor_from_class(unreal.CineCameraActor, unreal.Vector())
    camera.set_actor_label("SD_Main_Camera")
    cc = camera.get_cine_camera_component()
    cc.set_editor_property("filmback", unreal.CameraFilmbackSettings(sensor_width=36, sensor_height=36 * packet["height"] / packet["width"]))
    cc.set_editor_property("current_focal_length", 36 / (2 * math.tan(math.radians(packet["horizontal_fov_degrees"] / 2))))
    focus = cc.get_editor_property("focus_settings")
    focus.set_editor_property("focus_method", unreal.CameraFocusMethod.DISABLE)
    cc.set_editor_property("focus_settings", focus)
    post = cc.get_editor_property("post_process_settings")
    for key, value in dict(override_motion_blur_amount=True, motion_blur_amount=0.0, override_auto_exposure_method=True, auto_exposure_method=unreal.AutoExposureMethod.AEM_MANUAL, override_auto_exposure_apply_physical_camera_exposure=True, auto_exposure_apply_physical_camera_exposure=False, override_auto_exposure_bias=True, auto_exposure_bias=packet["exposure_bias"]).items():
        post.set_editor_property(key, value)
    cc.set_editor_property("post_process_settings", post)
    sequences = {}
    for label, poses in [("walk", packet["camera"]), ("overview", packet["review_camera"])]:
        seq = make_sequence(camera, poses, packet, "LS_" + label, job["asset_root"] + "/Sequences")
        for actor, component, clip, record in skeletal:
            section = seq.add_possessable(actor).add_track(unreal.MovieSceneSkeletalAnimationTrack).add_section()
            section.set_range(0, len(poses))
            params = section.get_editor_property("params")
            params.set_editor_property("animation", clip)
            params.set_editor_property("force_custom_mode", True)
            params.set_editor_property("skip_anim_notifiers", True)
            section.set_editor_property("params", params)
        sequences[label] = seq
    set_frame(sequences["walk"], 0)
    for actor, component, clip, record in skeletal:
        names = [str(component.get_bone_name(i)) for i in range(component.get_num_bones())]
        feet = [n for n in names if any(s in n.lower() for s in ["foot", "ball", "toe", "hoof", "paw"]) and not n.lower().startswith("ik_")]
        if feet:
            low = min(component.get_socket_location(name).z for name in feet)
            delta = record["ground_z_m"] * 100 + 1 - low
            if abs(delta) > 100:
                raise RuntimeError("Invalid foot correction: " + record["id"])
            location = actor.get_actor_location()
            location.z += delta
            actor.set_actor_location(location, False, True)
            record["foot_adjustment_cm"] = delta
    for record, actor in zip(records, object_actors):
        record["evaluated_world_matrix_cm"] = matrix_from_actor(actor)
    for sequence in sequences.values():
        unreal.EditorAssetLibrary.save_loaded_asset(sequence)
    unreal.LevelSequenceEditorBlueprintLibrary.close_level_sequence()
    levels.save_current_level()
    save_json(folder / "scene_manifest.json", dict(status="built", family=family, map_path=job["map_path"],
              objects=records, logical_objects=len(records), architecture_actors=[a.get_actor_label() for a in architectural],
              source_packs=sorted({r["source_pack"] for r in records} | ({"modular_building_set"} if family["id"] == "urban" else set())), lighting_source=sky,
              architecture_materials="Native solid-volume construction with original Modular Building Set shopfront/brick modules in the urban family",
              scope="Independent seeded terrain and coherent family dressing; root-static scene inspection with idle clips; benchmark dynamics and QA are a later stage"))
    return dict(job=job, ground=ground, layout=layout, packet=packet, route=route, terrain=terrain_actors["terrain"], camera=camera, sequences=sequences, records=records)


def audit(scene):
    editor = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
    ignored = [actor for actor in editor.get_all_level_actors() if actor != scene["terrain"]]
    ground = scene["ground"]
    def actual(x, y):
        hit = unreal.SystemLibrary.line_trace_single(world, unreal.Vector(x * 100, y * 100, 3500), unreal.Vector(x * 100, y * 100, -500), unreal.TraceTypeQuery.TRACE_TYPE_QUERY1, True, ignored, unreal.DrawDebugTrace.NONE, True)
        if hit is None:
            raise RuntimeError("Native terrain ray missed at " + str([x, y]))
        try:
            return hit.get_editor_property("location").z
        except Exception:
            return hit.to_tuple()[4].z
    a, b, c, d = ground.bounds
    probes = [(a + (b - a) * ix / 22 + 0.013, c + (d - c) * iy / 20 + 0.017) for ix in range(1, 22) for iy in range(1, 20)]
    maximum = max(abs(actual(x, y) - ground.z(x, y) * 100) for x, y in probes)
    evaluated = []
    camera_error = 0
    eye_error = 0
    for frame, pose in enumerate(scene["packet"]["camera"]):
        set_frame(scene["sequences"]["walk"], frame)
        matrix = matrix_from_actor(scene["camera"])
        camera_error = max(camera_error, max(abs(matrix[k][3] - pose["location_cm"][k]) for k in range(3)))
        x, y, z = pose["location_cm"]
        eye_error = max(eye_error, abs(z - actual(x / 100, y / 100) - 165))
        evaluated.append(dict(frame=frame, camera_world_matrix_cm=matrix))
    if maximum > 0.1 or camera_error > 0.1 or eye_error > 0.1:
        raise RuntimeError("Native terrain/camera validation failed: " + str([maximum, camera_error, eye_error]))
    report = dict(status="passed", engine=unreal.SystemLibrary.get_engine_version(), terrain_probes=len(probes),
                  terrain_support_error_cm=maximum, camera_evaluation_error_cm=camera_error, eye_height_error_cm=eye_error,
                  camera_frames=len(evaluated), terrain_levels_m=scene["layout"]["levels"],
                  ramps=len(scene["layout"]["ramps"]), route_vertical_span_m=scene["route"]["vertical_span_m"],
                  logical_objects=len(scene["records"]),
                  scope="Actual terrain collision and every walking camera pose; footprint placement and source material checks; no dense pixel GT certification")
    folder = Path(scene["job"]["output"])
    save_json(folder / "native_validation.json", report)
    save_json(folder / "evaluated_scene.json", dict(coordinate_system="Unreal centimetres; X forward/Y right/Z up camera basis", objects=scene["records"], camera=evaluated))
    unreal.LevelSequenceEditorBlueprintLibrary.close_level_sequence()
    unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).save_current_level()
    return report
