"""Verify closed boundary walls and the real HDRI bindings in the native world."""

import unreal
from .enclosure import ground_elevation_range


def validate(scene,editor):
    settings=scene["theme_settings"]
    expected=set(settings["enclosure"]["wall_actor_labels"])
    actors=[a for a in editor.get_all_level_actors() if a.get_actor_label().startswith("SD_Theme_")]
    if len(actors)!=4 or {a.get_actor_label() for a in actors}!=expected:
        raise RuntimeError("Expected exactly four boundary walls and zero background props")
    if settings.get("background_object_count")!=0:
        raise RuntimeError("Background props must be absent")
    enclosure=settings["enclosure"]
    terrain=next(m for m in scene["packet"]["meshes"] if m["key"]=="terrain")
    ground_low,ground_high=ground_elevation_range(terrain)
    expected_top=ground_high+settings["config"]["wall_clearance_above_highest_cm"]
    if (not enclosure["uniform_top"] or abs(enclosure["top_z_cm"]-expected_top)>.001
            or abs(ground_low+enclosure["height_above_lowest_ground_cm"]-expected_top)>.001):
        raise RuntimeError("Boundary tops must be level above the highest ground, referenced to lowest ground")
    by_label={a.get_actor_label():a for a in actors}
    ignored=[a for a in editor.get_all_level_actors() if a.get_actor_label() not in expected]
    world=unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
    probes=[]
    for record in settings["decorations"]:
        if abs(record["top_z_cm"]-expected_top)>.001:
            raise RuntimeError("All four walls must share the same top elevation")
        component=by_label[record["label"]].get_component_by_class(unreal.StaticMeshComponent)
        mesh=component.get_editor_property("static_mesh")
        if mesh is None or mesh.get_path_name()!=record["mesh"]:
            raise RuntimeError("Incorrect boundary mesh")
        unreal.get_editor_subsystem(unreal.StaticMeshEditorSubsystem).get_number_verts(mesh,0)
        component.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
        component.set_collision_profile_name("BlockAll")
        component.set_collision_enabled(unreal.CollisionEnabled.QUERY_AND_PHYSICS)
        if component.get_editor_property("custom_depth_stencil_value")!=100:
            raise RuntimeError("Boundary walls must use stencil 100")
        if any(component.get_material(i) is None for i in range(component.get_num_materials())):
            raise RuntimeError("Missing boundary material")
        profile=record["profile"]
        for (ta,_),(tb,_) in zip(profile["floor_profile_cm"],profile["floor_profile_cm"][1:]):
            for fraction in [.1,.5,.9]:
                xy=[0.0,0.0]
                xy[profile["axis"]]=profile["fixed"]+profile["outward_sign"]*record["thickness_cm"]/2
                xy[1-profile["axis"]]=ta+(tb-ta)*fraction
                hit=unreal.SystemLibrary.line_trace_single(world,unreal.Vector(*xy,expected_top+100),
                        unreal.Vector(*xy,record["bottom_z_cm"]-100),unreal.TraceTypeQuery.TRACE_TYPE_QUERY1,
                        True,ignored,unreal.DrawDebugTrace.NONE,True)
                if hit is None:
                    raise RuntimeError("Boundary collision ray missed "+profile["side"])
                try:
                    point=hit.get_editor_property("location")
                except Exception:
                    point=hit.to_tuple()[4]
                error=abs(point.z-expected_top)
                if error>.1:
                    raise RuntimeError("Boundary wall top differs by "+str(error)+" cm")
                probes.append(dict(side=profile["side"],expected_top_cm=expected_top,actual_top_cm=point.z,error_cm=error))
    all_actors={a.get_actor_label():a for a in editor.get_all_level_actors()}
    sky=all_actors["SD_SkyLight"].get_component_by_class(unreal.SkyLightComponent)
    cube=sky.get_editor_property("cubemap")
    if cube is None or cube.get_path_name()!=settings["hdri"]["cube_path"]:
        raise RuntimeError("SkyLight must use the downloaded Poly Haven cube")
    dome=all_actors["SD_Backdrop"].get_component_by_class(unreal.StaticMeshComponent)
    if dome.get_material(0).get_path_name()!=settings["hdri"]["material"]:
        raise RuntimeError("Visible sky must use the HDRI material")
    return dict(walls=4,background_objects=0,uniform_top_z_cm=expected_top,
                height_above_lowest_ground_cm=expected_top-ground_low,
                clearance_above_highest_ground_cm=expected_top-ground_high,
                ground_elevation_range_cm=[ground_low,ground_high],
                actual_wall_top_span_cm=max(x["actual_top_cm"] for x in probes)-min(x["actual_top_cm"] for x in probes),
                wall_top_collision_max_error_cm=max(x["error_cm"] for x in probes),
                wall_top_probes=probes,hdri_cube=cube.get_path_name(),visible_hdri_verified=True)
