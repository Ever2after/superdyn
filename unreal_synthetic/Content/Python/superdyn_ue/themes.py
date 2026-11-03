"""Material variants inside terrain-edge walls, with a Pure Sky HDRI outside."""

import hashlib
import json
from pathlib import Path
import unreal
from .theme_design import preset
from .theme_materials import pbr_material
from .enclosure import wall_records
from .hdri import receipt as hdri_receipt, apply as apply_hdri
from .io_utils import sha256


def signature(name,project_root):
    config=preset(name)
    source=Path(project_root)/"inputs/themes/asset_sources.json"
    data=json.loads(source.read_text())
    for record in data["textures"]:
        for path,expected in record["files"].items():
            if sha256(Path(project_root)/path)!=expected:
                raise RuntimeError("Theme texture differs from its receipt: "+path)
    hdri=hdri_receipt(project_root)
    code="".join(Path(__file__).with_name(x).read_text() for x in
                 ["themes.py","theme_materials.py","theme_design.py","enclosure.py","hdri.py"])
    return hashlib.sha256((json.dumps(config,sort_keys=True)+source.read_text()+json.dumps(hdri,sort_keys=True)+code).encode()).hexdigest()


def terrain_materials(name,project_root):
    config=preset(name)
    return dict(
        ground=pbr_material(project_root,"M_"+name+"_Ground",config["surface"],config["tile_cm"]),
        wall=pbr_material(project_root,"M_"+name+"_Wall",config["wall"],250,
                          tint=[.85,.9,.88] if name=="park" else None),
    )


def apply(name,scene,editor,project_root):
    from .scene_builder import make_mesh

    config=preset(name)
    packet=scene["packet"]
    terrain=next(m for m in packet["meshes"] if m["key"]=="terrain")
    records=wall_records(terrain,config["wall_clearance_above_highest_cm"],config["wall_thickness_cm"])
    material=pbr_material(project_root,"M_"+name+"_Boundary",config["wall"],250)
    root="/Game/SuperDyn/Themes/"+name+"/Enclosure"
    unreal.EditorAssetLibrary.make_directory(root+"/Meshes")
    actors=[]
    settings=[]
    for record in records:
        mesh_name="SM_"+packet["case_id"]+"_"+record["key"]
        mesh_path=root+"/Meshes/"+mesh_name
        if unreal.EditorAssetLibrary.does_asset_exist(mesh_path):
            if not unreal.EditorAssetLibrary.delete_asset(mesh_path):
                raise RuntimeError("Cannot rebuild boundary mesh: "+mesh_path)
        mesh=make_mesh(packet["case_id"],record,{"boundary":material},root)
        actor=editor.spawn_actor_from_class(unreal.StaticMeshActor,unreal.Vector())
        actor.set_actor_label("SD_Theme_"+name+"_Boundary_"+record["profile"]["side"])
        component=actor.get_component_by_class(unreal.StaticMeshComponent)
        component.set_static_mesh(mesh)
        component.set_collision_profile_name("BlockAll")
        component.set_collision_enabled(unreal.CollisionEnabled.QUERY_AND_PHYSICS)
        component.set_editor_property("render_custom_depth",True)
        component.set_editor_property("custom_depth_stencil_value",100)
        actors.append(actor)
        settings.append(dict(label=actor.get_actor_label(),mesh=mesh.get_path_name(),
                             profile=record["profile"],top_z_cm=record["top_z_cm"],
                             height_above_lowest_ground_cm=record["height_above_lowest_ground_cm"],
                             clearance_above_highest_ground_cm=record["clearance_above_highest_ground_cm"],
                             thickness_cm=record["thickness_cm"],bottom_z_cm=record["bottom_z_cm"],
                             material=material.get_path_name(),stencil=100))
    if name=="industrial":
        hazard=pbr_material(project_root,"M_industrial_Hazard","Concrete033",180,safety=True)
        component=scene["actors"]["terrain"].get_component_by_class(unreal.StaticMeshComponent)
        for i,section in enumerate(terrain["sections"]):
            if "retaining" in section["material"]:
                component.set_material(i,hazard)
    hdri=apply_hdri(editor,project_root)
    return actors,dict(
        name=name,config=config,decorations=settings,
        enclosure=dict(wall_actor_labels=[a.get_actor_label() for a in actors],
                       uniform_top=True,top_z_cm=records[0]["top_z_cm"],
                       height_above_lowest_ground_cm=records[0]["height_above_lowest_ground_cm"],
                       clearance_above_highest_ground_cm=records[0]["clearance_above_highest_ground_cm"],
                       ground_elevation_range_cm=records[0]["ground_elevation_range_cm"],
                       thickness_cm=config["wall_thickness_cm"],
                       terrain_bounds_cm=records[0]["terrain_bounds_cm"],
                       geometry="Four closed wall meshes; one level top above highest ground, height measured from lowest ground; buried lower skirt closes all gaps"),
        background_object_count=0,hdri=hdri,
        asset_paths=[r["mesh"] for r in settings],source_receipt="inputs/lighting/polyhaven/license_and_source.json",
        sky="Poly Haven Pure Sky TextureCube, visible background plus SkyLight illumination/reflections",
        geometry_scope="Original terrain and actor/camera tracks unchanged. No trees, buildings, containers, benches, rocks, surrounding ground or other background props.",
    )
