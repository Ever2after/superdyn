"""Curated environment maps: read-only survey and sublevel composition.

Original pack maps are never saved. Generated persistent levels stream them as
always-loaded sublevels and own every SuperDyn actor, camera and sequence.
"""

import math
from collections import Counter

import unreal


def world():
    return unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()


def actors():
    return unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()


def vec(v):
    return [float(v.x), float(v.y), float(v.z)]


def inventory():
    """Class counts, lighting/exposure settings and authored showcase cameras."""
    counts = Counter()
    cameras, lights, volumes, points = [], [], [], []
    for actor in actors():
        name = actor.get_class().get_name()
        counts[name] += 1
        if isinstance(actor, unreal.CameraActor):
            t = actor.get_actor_transform()
            r = t.rotation.rotator()
            cameras.append(dict(label=actor.get_actor_label(), location_cm=vec(t.translation), rotation_rpy_deg=[r.roll, r.pitch, r.yaw]))
        elif isinstance(actor, (unreal.DirectionalLight, unreal.SkyLight)):
            r = actor.get_actor_rotation()
            component = actor.get_component_by_class(unreal.LightComponentBase)
            lights.append(dict(label=actor.get_actor_label(), cls=name, rotation=[r.roll, r.pitch, r.yaw],
                               intensity=float(component.get_editor_property("intensity")) if component else None))
        elif isinstance(actor, unreal.PostProcessVolume):
            settings = actor.get_editor_property("settings")
            values = {}
            for key in ["auto_exposure_method", "auto_exposure_bias", "auto_exposure_min_brightness", "auto_exposure_max_brightness",
                        "auto_exposure_apply_physical_camera_exposure"]:
                try:
                    values[key] = str(settings.get_editor_property(key))
                    values["override_" + key] = bool(settings.get_editor_property("override_" + key))
                except Exception:
                    pass
            volumes.append(dict(label=actor.get_actor_label(), unbound=bool(actor.get_editor_property("unbound")), exposure=values))
        if isinstance(actor, (unreal.StaticMeshActor, unreal.Landscape, unreal.InstancedFoliageActor)) or name.startswith("Landscape"):
            origin, extent = actor.get_actor_bounds(False)
            if max(extent.x, extent.y) < 50000:
                points.append(vec(origin))
    return dict(class_counts=dict(counts.most_common()), cameras=cameras, lights=lights, post_process=volumes, mesh_centers_cm=points)


def trace_down(x, y, top, bottom, ignored=()):
    hit = unreal.SystemLibrary.line_trace_single(world(), unreal.Vector(x, y, top), unreal.Vector(x, y, bottom),
                                                 unreal.TraceTypeQuery.TRACE_TYPE_QUERY1, True, list(ignored),
                                                 unreal.DrawDebugTrace.NONE, True)
    if hit is None:
        return None
    values = hit.to_tuple()
    blocking, location, normal, actor = values[0], values[4], values[7], values[9]
    if not blocking and location.z == 0 and normal.z == 0:
        return None
    return float(location.z), float(normal.z), actor.get_actor_label() if actor else ""


def column(x, y, top, bottom, hits=4):
    """Successive surfaces below (x, y), highest first, in centimetres."""
    result, start = [], top
    for _ in range(hits):
        hit = trace_down(x, y, start, bottom)
        if hit is None:
            break
        result.append(hit)
        start = hit[0] - 3.0
    return result


def raster(bounds_cm, step_cm, top_cm, bottom_cm):
    x0, x1, y0, y1 = bounds_cm
    nx, ny = int((x1 - x0) // step_cm) + 1, int((y1 - y0) // step_cm) + 1
    cells = []
    for j in range(ny):
        row = []
        for i in range(nx):
            row.append(column(x0 + i * step_cm, y0 + j * step_cm, top_cm, bottom_cm))
        cells.append(row)
    return dict(origin_cm=[x0, y0], step_cm=step_cm, shape=[ny, nx], columns=cells)


def compose_level(map_path, environment_map):
    """New persistent level that streams an unmodified environment map.

    add_level_to_world makes the streamed level current; the persistent level
    is restored as current so every spawned actor belongs to the generated map.
    """
    levels = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    if unreal.EditorAssetLibrary.does_asset_exist(map_path):
        raise RuntimeError("Generated map already exists; use a fresh output namespace: " + map_path)
    if not levels.new_level(map_path):
        raise RuntimeError("Cannot create " + map_path)
    persistent = levels.get_current_level()
    streaming = unreal.EditorLevelUtils.add_level_to_world(world(), environment_map, unreal.LevelStreamingAlwaysLoaded)
    if streaming is None:
        raise RuntimeError("Cannot stream environment " + environment_map)
    unreal.EditorLevelUtils.set_level_visibility(streaming.get_loaded_level(), True, False)
    levels.set_current_level_by_name(map_path.rsplit("/", 1)[-1])
    assert_persistent(persistent)
    return persistent


def assert_persistent(persistent):
    current = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).get_current_level()
    if current != persistent:
        raise RuntimeError("Current level is not the generated persistent level: " + current.get_path_name())


def save_generated(map_path, persistent):
    """Save only the generated persistent map; streamed originals are never saved."""
    assert_persistent(persistent)
    if not unreal.EditorLoadingAndSavingUtils.save_map(world(), map_path):
        raise RuntimeError("Cannot save " + map_path)


def look_at(eye, target):
    d = [t - e for t, e in zip(target, eye)]
    yaw = math.degrees(math.atan2(d[1], d[0]))
    pitch = math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))
    return [0.0, pitch, yaw]
