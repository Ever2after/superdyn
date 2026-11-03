"""Native Unreal materials: grayscale metric grid and actor surface colors."""

import hashlib, json
import unreal

MATERIAL_REVISION = "ue58-v3"


def custom_input(name):
    value = unreal.CustomInput()
    value.set_editor_property("input_name", name)
    return value


def create_material(name, record):
    folder = "/Game/SuperDyn/Materials"
    path = folder + "/" + name
    old = unreal.load_asset(path)
    signature = (
        MATERIAL_REVISION
        + hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    )
    if (
        old
        and unreal.EditorAssetLibrary.get_metadata_tag(old, "SuperDynMaterial")
        == signature
    ):
        return old
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    mat = old or tools.create_asset(
        name, folder, unreal.Material, unreal.MaterialFactoryNew()
    )
    edit = unreal.MaterialEditingLibrary
    if old:
        edit.delete_all_material_expressions(mat)
    mat.set_editor_property("two_sided", True)
    color = edit.create_material_expression(
        mat, unreal.MaterialExpressionConstant3Vector, -600, -60
    )
    color.set_editor_property("constant", unreal.LinearColor(*record["color"], 1.0))
    if record.get("grid"):
        custom = edit.create_material_expression(
            mat, unreal.MaterialExpressionCustom, -100, -60
        )
        custom.set_editor_property(
            "output_type", unreal.CustomMaterialOutputType.CMOT_FLOAT3
        )
        custom.set_editor_property(
            "inputs",
            [
                custom_input("Position"),
                custom_input("Normal"),
                custom_input("Tint"),
            ],
        )
        custom.set_editor_property(
            "code",
            """float3 n = abs(Normal);
float2 q = (n.z > 0.7) ? Position.xy : ((n.x > n.y) ? Position.yz : Position.xz);
q = q / 100.0;
float2 distanceToLine = abs(frac(q + 0.5) - 0.5);
float2 aa = max(fwidth(q), 0.001);
float2 gridLines = 1.0 - smoothstep(0.009, 0.009 + aa, distanceToLine);
return lerp(Tint, float3(0.19,0.215,0.24), max(gridLines.x,gridLines.y));""",
        )
        position = edit.create_material_expression(
            mat, unreal.MaterialExpressionWorldPosition, -600, -200
        )
        normal = edit.create_material_expression(
            mat, unreal.MaterialExpressionPixelNormalWS, -600, -340
        )
        edit.connect_material_expressions(position, "", custom, "Position")
        edit.connect_material_expressions(normal, "", custom, "Normal")
        edit.connect_material_expressions(color, "", custom, "Tint")
        edit.connect_material_property(
            custom, "", unreal.MaterialProperty.MP_BASE_COLOR
        )
    else:
        edit.connect_material_property(color, "", unreal.MaterialProperty.MP_BASE_COLOR)
    for offset, property_, value in [
        (100, unreal.MaterialProperty.MP_ROUGHNESS, record["roughness"]),
        (200, unreal.MaterialProperty.MP_METALLIC, record["metallic"]),
    ]:
        node = edit.create_material_expression(
            mat, unreal.MaterialExpressionConstant, -100, offset
        )
        node.set_editor_property("r", value)
        edit.connect_material_property(node, "", property_)
    edit.recompile_material(mat)
    unreal.EditorAssetLibrary.set_metadata_tag(mat, "SuperDynMaterial", signature)
    unreal.EditorAssetLibrary.save_loaded_asset(mat)
    return mat


def create_backdrop_material():
    path = "/Game/SuperDyn/Materials/M_Backdrop"
    mat = unreal.load_asset(path)
    if mat:
        return mat
    mat = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
        "M_Backdrop",
        "/Game/SuperDyn/Materials",
        unreal.Material,
        unreal.MaterialFactoryNew(),
    )
    mat.set_editor_property("two_sided", True)
    mat.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_UNLIT)
    node = unreal.MaterialEditingLibrary.create_material_expression(
        mat, unreal.MaterialExpressionConstant3Vector, 0, 0
    )
    node.set_editor_property("constant", unreal.LinearColor(0.075, 0.09, 0.11, 1.0))
    unreal.MaterialEditingLibrary.connect_material_property(
        node, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR
    )
    unreal.MaterialEditingLibrary.recompile_material(mat)
    unreal.EditorAssetLibrary.save_loaded_asset(mat)
    return mat
