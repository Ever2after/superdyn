"""CC0 PBR materials with world-projected UVs, plus native set-dressing materials."""

from pathlib import Path
import unreal
from .materials import custom_input, create_material

FOLDER = "/Game/SuperDyn/ThemeMaterials"


def texture(project_root, asset, kind):
    name = asset + "_" + kind
    path = "/Game/SuperDyn/ThemeTextures/" + name
    old = unreal.load_asset(path)
    if old:
        return old
    source = Path(project_root) / "inputs/themes/textures" / asset / (asset + "_1K-JPG_" + kind + ".jpg")
    task = unreal.AssetImportTask()
    for key, value in dict(filename=str(source), destination_path="/Game/SuperDyn/ThemeTextures", destination_name=name,
                           automated=True, save=True, replace_existing=False).items():
        task.set_editor_property(key, value)
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    paths = task.get_editor_property("imported_object_paths")
    if not paths:
        raise RuntimeError("Theme texture import failed: " + str(source))
    result = unreal.load_asset(paths[0])
    result.set_editor_property("srgb", kind == "Color")
    if kind == "NormalDX":
        result.set_editor_property("compression_settings", unreal.TextureCompressionSettings.TC_NORMALMAP)
    unreal.EditorAssetLibrary.save_loaded_asset(result)
    return result


def new_material(name):
    path = FOLDER + "/" + name
    material = unreal.load_asset(path)
    edit = unreal.MaterialEditingLibrary
    if material:
        edit.delete_all_material_expressions(material)
    else:
        material = unreal.AssetToolsHelpers.get_asset_tools().create_asset(name, FOLDER, unreal.Material, unreal.MaterialFactoryNew())
    material.set_editor_property("two_sided", True)
    return material, edit


def pbr_material(project_root, name, asset, tile_cm, tint=None, safety=False):
    material, edit = new_material(name)
    position = edit.create_material_expression(material, unreal.MaterialExpressionWorldPosition, -900, -300)
    normal = edit.create_material_expression(material, unreal.MaterialExpressionVertexNormalWS, -900, -100)
    uv = edit.create_material_expression(material, unreal.MaterialExpressionCustom, -650, -300)
    uv.set_editor_property("output_type", unreal.CustomMaterialOutputType.CMOT_FLOAT2)
    uv.set_editor_property("inputs", [custom_input("Position"), custom_input("Normal")])
    uv.set_editor_property("code", "float3 n=abs(Normal); return (n.z>0.7 ? Position.xy : (n.x>n.y ? Position.yz : Position.xz)) / " + str(float(tile_cm)) + ";")
    edit.connect_material_expressions(position, "", uv, "Position")
    edit.connect_material_expressions(normal, "", uv, "Normal")
    for row, kind, property_ in [
        (0, "Color", unreal.MaterialProperty.MP_BASE_COLOR),
        (1, "Roughness", unreal.MaterialProperty.MP_ROUGHNESS),
    ]:
        sample = edit.create_material_expression(material, unreal.MaterialExpressionTextureSample, -400, row * 220)
        sample.set_editor_property("texture", texture(project_root, asset, kind))
        sample.set_editor_property("sampler_type", unreal.MaterialSamplerType.SAMPLERTYPE_COLOR if kind == "Color" else unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR)
        edit.connect_material_expressions(uv, "", sample, "UVs")
        if kind == "Color" and (tint or safety):
            custom = edit.create_material_expression(material, unreal.MaterialExpressionCustom, -100, 0)
            custom.set_editor_property("output_type", unreal.CustomMaterialOutputType.CMOT_FLOAT3)
            custom.set_editor_property("inputs", [custom_input("Color"), custom_input("Position")])
            if safety:
                code = "float stripe=step(0.5,frac((Position.x+Position.y+Position.z)/45.0)); float3 paint=lerp(float3(0.025,0.03,0.035),float3(0.9,0.56,0.025),stripe); float band=step(122.0,fmod(abs(Position.z)+0.1,150.0)); return lerp(Color*0.72,paint,band*0.92);"
            else:
                code = "return Color * float3(" + ",".join(str(float(x)) for x in tint) + ");"
            custom.set_editor_property("code", code)
            edit.connect_material_expressions(sample, "RGB", custom, "Color")
            edit.connect_material_expressions(position, "", custom, "Position")
            edit.connect_material_property(custom, "", property_)
        elif kind == "Roughness":
            clamp = edit.create_material_expression(material, unreal.MaterialExpressionCustom, -100, 240)
            clamp.set_editor_property("output_type", unreal.CustomMaterialOutputType.CMOT_FLOAT1)
            clamp.set_editor_property("inputs", [custom_input("Value")])
            clamp.set_editor_property("code", "return max(Value," + ("0.9" if asset == "Grass004" else "0.65") + ");")
            edit.connect_material_expressions(sample, "R", clamp, "Value")
            edit.connect_material_property(clamp, "", property_)
        else:
            edit.connect_material_property(sample, "RGB" if kind == "Color" else "R", property_)
    # Use a tangent normal only on the source's upward surfaces.
    # Retaining walls keep their exact geometric normal to avoid tangent seams.
    if name.endswith("_Ground"):
        sample = edit.create_material_expression(material, unreal.MaterialExpressionTextureSample, -400, 470)
        sample.set_editor_property("texture", texture(project_root, asset, "NormalDX"))
        sample.set_editor_property("sampler_type", unreal.MaterialSamplerType.SAMPLERTYPE_NORMAL)
        edit.connect_material_expressions(uv, "", sample, "UVs")
        strength = edit.create_material_expression(material, unreal.MaterialExpressionCustom, -100, 480)
        strength.set_editor_property("output_type", unreal.CustomMaterialOutputType.CMOT_FLOAT3)
        strength.set_editor_property("inputs", [custom_input("MapNormal")])
        strength.set_editor_property("code", "return normalize(float3(MapNormal.xy*0.25,max(MapNormal.z,0.5)));")
        edit.connect_material_expressions(sample, "RGB", strength, "MapNormal")
        edit.connect_material_property(strength, "", unreal.MaterialProperty.MP_NORMAL)
    edit.recompile_material(material)
    unreal.EditorAssetLibrary.save_loaded_asset(material)
    return material


def solid(name, color, metallic=0, roughness=0.75):
    return create_material("Theme_" + name, dict(color=color, roughness=roughness, metallic=metallic))


def sky_material(theme, config):
    material, edit = new_material("M_" + theme + "_Sky")
    material.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_UNLIT)
    material.set_editor_property("is_sky", True)
    position = edit.create_material_expression(material, unreal.MaterialExpressionWorldPosition, -500, -200)
    custom = edit.create_material_expression(material, unreal.MaterialExpressionCustom, -150, -100)
    custom.set_editor_property("output_type", unreal.CustomMaterialOutputType.CMOT_FLOAT3)
    custom.set_editor_property("inputs", [custom_input("Position")])
    low = ",".join(str(x) for x in config["horizon"])
    high = ",".join(str(x) for x in config["zenith"])
    custom.set_editor_property("code", "float h=saturate(normalize(Position).z);return lerp(float3(" + low + "),float3(" + high + "),pow(h,0.55));")
    edit.connect_material_expressions(position, "", custom, "Position")
    edit.connect_material_property(custom, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    edit.recompile_material(material)
    unreal.EditorAssetLibrary.save_loaded_asset(material)
    return material
