"""One Poly Haven texture cube shared by visible sky, illumination and reflections."""

import json
from pathlib import Path
import unreal
from .io_utils import sha256
from .theme_materials import new_material

SKYLIGHT_FILL_MULTIPLIER = 3.0

def receipt(project_root):
    path = Path(project_root)/"inputs/lighting/polyhaven/license_and_source.json"
    data = json.loads(path.read_text())
    if sha256(Path(project_root)/data["relative_path"]) != data["sha256"]:
        raise RuntimeError("Poly Haven HDRI differs from its source receipt")
    return data


def apply(editor,project_root):
    source = receipt(project_root)
    name = "PolyHaven_"+source["asset"]+"_4k"
    path = "/Game/SuperDyn/Lighting/"+name
    cube = unreal.load_asset(path)
    if cube is None:
        task = unreal.AssetImportTask()
        for key,value in dict(filename=str(Path(project_root)/source["relative_path"]),
                              destination_path="/Game/SuperDyn/Lighting",destination_name=name,
                              automated=True,replace_existing=False,save=True).items():
            task.set_editor_property(key,value)
        unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
        cube = unreal.load_asset(path)
    if not isinstance(cube,unreal.TextureCube):
        raise RuntimeError("Pure Sky HDRI must import as a TextureCube")
    material,edit = new_material("M_PolyHavenPureSky")
    material.set_editor_property("shading_model",unreal.MaterialShadingModel.MSM_UNLIT)
    material.set_editor_property("is_sky",True)
    view = edit.create_material_expression(material,unreal.MaterialExpressionCameraVectorWS,-650,0)
    negate = edit.create_material_expression(material,unreal.MaterialExpressionMultiply,-450,0)
    negate.set_editor_property("const_b",-1.0)
    edit.connect_material_expressions(view,"",negate,"A")
    sample = edit.create_material_expression(material,unreal.MaterialExpressionTextureSample,-250,0)
    sample.set_editor_property("texture",cube)
    sample.set_editor_property("sampler_type",unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR)
    edit.connect_material_expressions(negate,"",sample,"UVs")
    scale = edit.create_material_expression(material,unreal.MaterialExpressionMultiply,0,0)
    scale.set_editor_property("const_b",float(source["background_multiplier"]))
    edit.connect_material_expressions(sample,"RGB",scale,"A")
    edit.connect_material_property(scale,"",unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    edit.recompile_material(material)
    unreal.EditorAssetLibrary.save_loaded_asset(material)
    by_label={a.get_actor_label():a for a in editor.get_all_level_actors()}
    backdrop=by_label["SD_Backdrop"].get_component_by_class(unreal.StaticMeshComponent)
    backdrop.set_material(0,material)
    skylight=by_label["SD_SkyLight"].get_component_by_class(unreal.SkyLightComponent)
    skylight.set_editor_property("source_type",unreal.SkyLightSourceType.SLS_SPECIFIED_CUBEMAP)
    skylight.set_editor_property("cubemap",cube)
    skylight.set_editor_property("source_cubemap_angle",0.0)
    fill=float(source["skylight_intensity"])*SKYLIGHT_FILL_MULTIPLIER
    skylight.set_editor_property("intensity",fill)
    rotation=source["directional_light_rotation"]
    by_label["SD_Sun"].set_actor_rotation(unreal.Rotator(**rotation),False)
    sun=by_label["SD_Sun"].get_component_by_class(unreal.DirectionalLightComponent)
    sun.set_editor_property("intensity",3.0)
    sun.set_editor_property("use_temperature",True)
    sun.set_editor_property("temperature",float(source["whitebalance_kelvin"]))
    sun.set_editor_property("light_source_angle",0.55)
    return dict(asset=source["asset"],source=source["source"],license=source["license"],
                sha256=source["sha256"],cube_path=cube.get_path_name(),material=material.get_path_name(),
                background_multiplier=source["background_multiplier"],skylight_intensity=fill,
                skylight_fill_multiplier=SKYLIGHT_FILL_MULTIPLIER,
                directional_light_rotation=rotation,credit=source["credit"],rotation_degrees=0.0)
