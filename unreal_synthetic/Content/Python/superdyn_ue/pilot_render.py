"""Movie Render Queue jobs for pilot RGB and exact ground-truth passes.

RGB uses jittered spatial supersampling. The GT job renders one unjittered
sample through a post-process material that writes, per pixel,
R = visible instance stencil, G = scene depth (cm), B = custom depth (cm), into
32-bit EXR, so IDs and depths are never blended across edges.
"""

from pathlib import Path

import unreal

# V1 (Replacing the Tonemapper) never reached MRQ's post-process capture; V2 is
# placed after tonemapping, next to MRQ's own WorldDepth pass.
GT_MATERIAL = "/Game/SuperDyn/Pilot/Materials/M_PilotGroundTruthV2"


def enum_member(enum, *needles):
    for name in dir(enum):
        if name.isupper() and all(n in name for n in needles):
            return getattr(enum, name)
    raise RuntimeError("Missing enum member " + str(needles) + " in " + str(enum))


def gt_material():
    material = unreal.load_asset(GT_MATERIAL)
    if material:
        return material
    folder, name = GT_MATERIAL.rsplit("/", 1)
    unreal.EditorAssetLibrary.make_directory(folder)
    material = unreal.AssetToolsHelpers.get_asset_tools().create_asset(name, folder, unreal.Material, unreal.MaterialFactoryNew())
    edit = unreal.MaterialEditingLibrary
    material.set_editor_property("material_domain", unreal.MaterialDomain.MD_POST_PROCESS)
    material.set_editor_property("blendable_location", enum_member(unreal.BlendableLocation, "AFTER", "TONEMAPPING"))
    nodes = {}
    for row, (key, texture_id) in enumerate([("scene", "SCENE_DEPTH"), ("custom", "CUSTOM_DEPTH"), ("stencil", "CUSTOM_STENCIL")]):
        node = edit.create_material_expression(material, unreal.MaterialExpressionSceneTexture, -700, row * 200)
        node.set_editor_property("scene_texture_id", enum_member(unreal.SceneTextureId, "PPI_" + texture_id))
        nodes[key] = node
    custom = edit.create_material_expression(material, unreal.MaterialExpressionCustom, -300, 100)
    custom.set_editor_property("output_type", unreal.CustomMaterialOutputType.CMOT_FLOAT3)
    inputs = []
    for key in ["Scene", "Custom", "Stencil"]:
        item = unreal.CustomInput()
        item.set_editor_property("input_name", key)
        inputs.append(item)
    custom.set_editor_property("inputs", inputs)
    # A custom-depth surface is visible when it is the nearest scene surface.
    custom.set_editor_property("code", "float s=Scene.r; float c=Custom.r; float id=(c<=s*1.0005+1.0)?floor(Stencil.r+0.5):0.0; return float3(id,s,c);")
    for key, pin in [("scene", "Scene"), ("custom", "Custom"), ("stencil", "Stencil")]:
        edit.connect_material_expressions(nodes[key], "Color", custom, pin)
    edit.connect_material_property(custom, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    edit.recompile_material(material)
    unreal.EditorAssetLibrary.save_loaded_asset(material)
    return material


def set_if(obj, key, value):
    try:
        obj.set_editor_property(key, value)
        return True
    except Exception:
        return False


def add_job(queue, spec):
    """spec: map_path, sequence_path, output, frames, width, height, fps, kind (rgb|gt), spatial_samples."""
    output = Path(spec["output"]).resolve()
    gt = spec["kind"] == "gt"
    job = queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
    job.set_editor_property("job_name", spec["name"])
    job.set_editor_property("map", unreal.SoftObjectPath(spec["map_path"] + "." + spec["map_path"].rsplit("/", 1)[-1]))
    job.set_editor_property("sequence", unreal.SoftObjectPath(spec["sequence_path"]))
    config = job.get_configuration()
    out = config.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
    out.set_editor_property("output_directory", unreal.DirectoryPath(str(output)))
    out.set_editor_property("file_name_format", "{render_pass}/frame_{frame_number}" if gt else "rgb/frame_{frame_number}")
    out.set_editor_property("output_resolution", unreal.IntPoint(spec["width"], spec["height"]))
    out.set_editor_property("use_custom_frame_rate", True)
    out.set_editor_property("output_frame_rate", unreal.FrameRate(spec["fps"], 1))
    out.set_editor_property("use_custom_playback_range", True)
    out.set_editor_property("custom_start_frame", 0)
    out.set_editor_property("custom_end_frame", spec["frames"])
    out.set_editor_property("zero_pad_frame_numbers", 6)
    out.set_editor_property("frame_number_offset", 0)
    out.set_editor_property("output_frame_step", spec.get("frame_step", 1))
    out.set_editor_property("flush_disk_writes_per_shot", True)
    out.set_editor_property("override_existing_output", False)
    deferred = config.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)
    if gt:
        defaults = list(deferred.get_editor_property("additional_post_process_materials"))
        if spec.get("probe_dir"):
            rows = []
            for item in defaults:
                row = {}
                for key in [n for n in dir(item) if not n.startswith("_")]:
                    try:
                        row[key] = str(item.get_editor_property(key))
                    except Exception:
                        pass
                rows.append(row)
            Path(spec["probe_dir"]).mkdir(parents=True, exist_ok=True)
            (Path(spec["probe_dir"]) / ("mrq_defaults_" + spec["name"] + ".txt")).write_text(repr(rows) + "\n" + repr([n for n in dir(deferred) if not n.startswith("_")]))
        mode = spec.get("pp_mode", "depth_and_ids")
        pass_ = unreal.MoviePipelinePostProcessPass()
        pass_.set_editor_property("enabled", True)
        pass_.set_editor_property("material", gt_material())
        if spec.get("high_precision", True):
            set_if(pass_, "high_precision_output", True)
        if mode == "builtin_depth":
            for item in defaults:
                item.set_editor_property("enabled", "WorldDepth" in str(item.get_editor_property("material")))
            passes = defaults
        elif mode in ("append", "depth_and_ids"):
            for item in defaults:
                item.set_editor_property("enabled", mode == "depth_and_ids" and "WorldDepth" in str(item.get_editor_property("material")))
            passes = defaults + [pass_]
        else:
            passes = [pass_]
        deferred.set_editor_property("additional_post_process_materials", passes)
        if spec.get("high_precision", True):
            set_if(deferred, "use32_bit_post_process_materials", True)
        if not spec.get("main_pass", True):
            set_if(deferred, "render_main_pass", False)
        set_if(deferred, "disable_multisample_effects", True)
        exr = config.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_EXR)
        set_if(exr, "compression", enum_member(unreal.EXRCompressionFormat, "ZIP"))
    else:
        config.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_PNG)
    game = config.find_or_add_setting_by_class(unreal.MoviePipelineGameOverrideSetting)
    game.set_editor_property("soft_game_mode_override", unreal.load_class(None, "/Script/MovieRenderPipelineCore.MoviePipelineGameMode"))
    game.set_editor_property("cinematic_quality_settings", True)
    aa = config.find_or_add_setting_by_class(unreal.MoviePipelineAntiAliasingSetting)
    aa.set_editor_property("spatial_sample_count", 1 if gt else spec.get("spatial_samples", 4))
    aa.set_editor_property("temporal_sample_count", 1)
    aa.set_editor_property("override_anti_aliasing", True)
    aa.set_editor_property("anti_aliasing_method", unreal.AntiAliasingMethod.AAM_NONE)
    aa.set_editor_property("engine_warm_up_count", spec.get("warm_up", 32))
    aa.set_editor_property("render_warm_up_count", 8)
    camera = config.find_or_add_setting_by_class(unreal.MoviePipelineCameraSetting)
    camera.set_editor_property("shutter_timing", unreal.MoviePipelineShutterTiming.FRAME_OPEN)
    if spec.get("console"):
        cvars = config.find_or_add_setting_by_class(unreal.MoviePipelineConsoleVariableSetting)
        for key, value in spec["console"].items():
            cvars.add_or_update_console_variable(key, float(value))
    return job
