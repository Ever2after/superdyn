"""Plan, render and verify the three reusable synthetic-main scene families.

All generated files live in this project. The retained synthetic terrain
generator is imported read-only; its source hashes accompany each recipe.
"""

from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from bridge import ROOT, sha256
from superdyn_ue.family_geometry import Ground
from superdyn_ue.io_utils import save_json

DEFAULT_OUTPUT = ROOT / "outputs/main_scene_families_v3"
CONFIG = ROOT / "configs/main_scene_families.json"
from terrain.sampling import generate, load_settings
from terrain.layout import terrain_mesh
from terrain.validation import validate, raster, clearance_mask
from terrain.paths import grid_path, shortcut, resample_path


ASSETS = {
    "container": ("construction_site_vol1", "/Game/Construction_VOL1/Meshes/SM_FreightContainer_01"),
    "pallet": ("construction_site_vol1", "/Game/Construction_VOL1/Meshes/SM_BrickPaletteStack_01a"),
    "crate": ("construction_site_vol1", "/Game/Construction_VOL1/Meshes/SM_Crate_01a"),
    "reel": ("construction_site_vol1", "/Game/Construction_VOL1/Meshes/SM_CableReel"),
    "barrel": ("industry_props_6", "/Game/IndustryPropsPack6/Meshes/SM_Barrel01"),
    "pickup": ("vehicle_variety_pack", "/Game/VehicleVarietyPack/Skeletons/SK_Pickup"),
    "sedan": ("vehicle_variety_pack_2", "/Game/VehicleVarietyVol2/Skeletons/SK_Sedan_01a"),
    "worker": ("game_animation_sample", "/Game/Characters/UEFN_Mannequin/Meshes/SKM_UEFN_Mannequin", "/Game/Characters/UEFN_Mannequin/Animations/Idle/M_Neutral_Stand_Idle_Loop"),
    "cafe": ("city_park_environment", "/Game/CityPark/Meshes/Buildings/SM_CafeBuilding01"),
    "bench": ("city_park_environment", "/Game/CityPark/Meshes/Props/SM_Bench01_1"),
    "lamp": ("city_park_environment", "/Game/CityPark/Meshes/Props/SM_LampPost01"),
    "trash": ("city_park_environment", "/Game/CityPark/Meshes/Props/SM_TrashCan01"),
    "tree": ("city_park_environment", "/Game/CityPark/Meshes/Flora/Trees/SM_AmurCork01"),
    "pine": ("modular_rural_cabins", "/Game/Modular_Rural_Cabin/Meshes/Foliage/SM_Pine_Tree_01"),
    "bush": ("city_park_environment", "/Game/CityPark/Meshes/Flora/Bushs/SM_Bush1_01"),
    "rock": ("city_park_environment", "/Game/CityPark/Meshes/Ground/SM_Rock01"),
    "shelter": ("modular_rural_cabins", "/Game/Modular_Rural_Cabin/Meshes/Modular/Porch_Roof_4x4m"),
    "deer": ("animal_variety_pack", "/Game/AnimalVarietyPack/DeerStagAndDoe/Meshes/SK_DeerStag", "/Game/AnimalVarietyPack/DeerStagAndDoe/Animations/ANIM_DeerStag_IdleBreathe"),
    "fox": ("animal_variety_pack", "/Game/AnimalVarietyPack/Fox/Meshes/SK_Fox", "/Game/AnimalVarietyPack/Fox/Animations/ANIM_Fox_IdleBreathe"),
}


def source_paths():
    paths = [Path(__file__), CONFIG, ROOT / "manage.py", ROOT / "bridge.py"]
    paths += sorted((ROOT / "Content/Python/superdyn_ue").glob("*.py"))
    paths += [ROOT / "Content/Python/run_scene_families.py"]
    paths += [ROOT / "scripts/render_scene_families.py"]
    paths += sorted((ROOT / "terrain").glob("*.py")) + sorted((ROOT / "configs/terrain").glob("*.json"))
    return sorted(set(p for p in paths if p.is_file()))


def scaled_layout(seed, scale):
    """Scale geometry and corresponding validation distances together."""
    layout = deepcopy(generate(seed))
    layout["bounds"] = [x * scale for x in layout["bounds"]]
    layout["grid_step"] *= scale
    for patch in layout["plateaus"] + layout["ramps"]:
        patch["bounds"] = [x * scale for x in patch["bounds"]]
    settings = layout["generation"]["settings"]
    for name in ["dimension_snap", "walk_grid_step", "mesh_grid_step", "minimum_region_side", "landing_length", "edge_margin", "ramp_clearance", "minimum_flat_rectangle_side", "navigation_clearance_radius"]:
        settings[name] *= scale
    settings["minimum_flat_rectangle_area"] *= scale ** 2
    for name in ["length_range", "width_range", "ramp_width_range", "ramp_length_range"]:
        settings[name] = [x * scale for x in settings[name]]
    layout["generation"]["xy_scale"] = scale
    geometry = dict(bounds=layout["bounds"], plateaus=layout["plateaus"], ramps=layout["ramps"])
    layout["generation"]["layout_sha256"] = hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest()
    report = validate(layout, raise_on_failure=True)
    layout["generation"]["flat_rectangles"] = report["flat_rectangles"]
    # Context figures from the original terrain preview are not scene objects.
    for key in ["people", "rovers", "props", "level_labels", "camera_entry"]:
        layout.pop(key, None)
    return layout, report


def inspection_route(layout, frames):
    ground = Ground(layout)
    step = layout["generation"]["settings"]["walk_grid_step"]
    xs, ys, z, _, _ = raster(layout)
    grade = layout["generation"]["settings"]["maximum_grade"]
    safe = clearance_mask(z, step, grade, 0.85)
    rects = layout["generation"]["flat_rectangles"]
    points = []
    for r in rects:
        a, b, c, d = r["bounds"]
        x, y = (a + b) / 2, (c + d) / 2
        ix, iy = int((x - layout["bounds"][0]) / step), int((y - layout["bounds"][2]) / step)
        if safe[iy, ix]:
            points.append(((iy, ix), r["height"]))
    candidates = []
    for i, (a, za) in enumerate(points):
        for b, zb in points[i + 1:]:
            if abs(za - zb) < 1.4:
                continue
            route = grid_path(z, safe, a, b, step, grade)
            if route is None:
                continue
            xy = shortcut(np.array([[xs[x], ys[y]] for y, x in route]), layout, safe, step, grade)
            sampled, length = resample_path(xy, layout, frames, amplitude=0.12)
            if 9 <= length <= 28:
                candidates.append((abs(za - zb) * 4 - abs(length - 17) * 0.1, sampled, xy, length))
    if not candidates:
        raise RuntimeError("No safe inspection route crossing different terrain levels")
    _, sampled, route, length = max(candidates, key=lambda x: x[0])
    # The terrain helper and independent runtime query must agree before export.
    error = max(abs(ground.z(*p[:2]) - p[2]) for p in sampled)
    if error > 1e-6:
        raise RuntimeError("Runtime terrain query differs from the source generator")
    poses = []
    yaw_values = []
    for i, here in enumerate(sampled):
        j = min(len(sampled) - 1, i + 22)
        if j == i:
            target = here + (here - sampled[max(0, i - 22)])
        else:
            target = sampled[j]
        delta = target - here
        yaw_values.append(math.atan2(delta[1], delta[0]))
        pitch = math.degrees(math.atan2(delta[2], max(0.5, np.linalg.norm(delta[:2])))) - 4
        poses.append(dict(location_cm=[float(here[0] * 100), float(here[1] * 100), float((here[2] + 1.65) * 100)], rotation_rpy_deg=[0, float(np.clip(pitch, -22, 22)), 0], scale=[1, 1, 1]))
    yaw_values = np.unwrap(yaw_values)
    for pose, yaw in zip(poses, yaw_values):
        pose["rotation_rpy_deg"][2] = float(math.degrees(yaw))
    reserved = sampled[::2, :2].tolist() + [sampled[-1, :2].tolist()]
    return dict(poses=poses, reserved=reserved, route_xy_m=route.tolist(), length_m=length,
                vertical_span_m=float(np.ptp(sampled[:, 2])), support_query_error_m=error)


def mesh_record(vertices, faces, roles, family, path_xy=()):
    groups = {}
    indexes = {}
    path = np.asarray(path_xy)
    for face, role in zip(faces, roles):
        for k in range(1, len(face) - 1):
            triangle = np.array([vertices[face[0]], vertices[face[k]], vertices[face[k + 1]]], float)
            normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            size = np.linalg.norm(normal)
            if size < 1e-10:
                continue
            normal /= size
            material = "wall" if role in ("wall", "bottom") else "ground"
            if family == "park" and material == "ground" and path.size and np.linalg.norm(path - triangle.mean(0)[:2], axis=1).min() < 1.1:
                material = "path"
            section = groups.setdefault(material, dict(material=material, vertices_cm=[], normals=[], uv0=[], triangles=[]))
            lookup = indexes.setdefault(material, {})
            ids = []
            for v in triangle:
                key = tuple(np.round(np.r_[v, normal], 8))
                if key not in lookup:
                    lookup[key] = len(section["vertices_cm"])
                    section["vertices_cm"].append((v * 100).tolist())
                    section["normals"].append(normal.tolist())
                    section["uv0"].append((v[:2] / 2).tolist())
                ids.append(lookup[key])
            section["triangles"].append(ids)
    return dict(key="terrain", parent=None, animated=False, sections=list(groups.values()))


def context_mesh(layout):
    ground = Ground(layout)
    a, b, c, d = ground.bounds
    # Match every core boundary vertex; a coarse skirt can bridge across a
    # discontinuous terrace edge and leave visible triangular gaps.
    step = layout["grid_step"]
    xs = np.r_[np.arange(a - 10, a, 1), np.linspace(a, b, round((b - a) / step) + 1), np.arange(b + 1, b + 11, 1)]
    ys = np.r_[np.arange(c - 10, c, 1), np.linspace(c, d, round((d - c) / step) + 1), np.arange(d + 1, d + 11, 1)]
    vertices, faces = [], []
    for x0, x1 in zip(xs, xs[1:]):
        for y0, y1 in zip(ys, ys[1:]):
            if a <= (x0 + x1) / 2 <= b and c <= (y0 + y1) / 2 <= d:
                continue
            start = len(vertices)
            vertices += [(x, y, ground.context_z(x, y) - 0.03) for x, y in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]]
            faces.append([start, start + 1, start + 2, start + 3])
    # An exterior ground ring reaches beyond inspection views without placing
    # a hidden plane underneath any of the core ground or its depressions.
    for x0, x1, y0, y1 in [(-180, 180, -180, c - 10), (-180, 180, d + 10, 180), (-180, a - 10, c - 10, d + 10), (b + 10, 180, c - 10, d + 10)]:
        start = len(vertices)
        vertices += [(x0, y0, -0.03), (x1, y0, -0.03), (x1, y1, -0.03), (x0, y1, -0.03)]
        faces.append([start, start + 1, start + 2, start + 3])
    roles = ["flat"] * len(faces)
    # Each outside edge needs its own vertical closure. At a discontinuous
    # boundary the skirt can be higher than the adjacent lower core terrace;
    # the core's existing wall only reaches that lower terrace's height.
    for line, fixed, axis in [(xs, c, 1), (xs, d, 1), (ys, a, 0), (ys, b, 0)]:
        for u, v in zip(line, line[1:]):
            lo, hi = (a, b) if axis == 1 else (c, d)
            if u < lo - 1e-8 or v > hi + 1e-8:
                continue
            p = (u, fixed) if axis == 1 else (fixed, u)
            q = (v, fixed) if axis == 1 else (fixed, v)
            bottom = layout["bottom_z"] - 0.05
            start = len(vertices)
            vertices += [(*p, bottom), (*q, bottom), (*q, ground.context_z(*q) - 0.03), (*p, ground.context_z(*p) - 0.03)]
            faces.append([start, start + 1, start + 2, start + 3])
            roles.append("wall")
    result = mesh_record(vertices, faces, roles, "context")
    result["key"] = "context"
    return result


def review_camera(layout):
    a, b, c, d = layout["bounds"]
    target = np.array([(a + b) / 2, (c + d) / 2, max(layout["levels"]) * 0.45])
    radius = max(b - a, d - c) * 0.88
    result = []
    for angle in [-65, 25, 115, 205]:
        theta = math.radians(angle)
        eye = target + np.array([radius * math.cos(theta), radius * math.sin(theta), radius * 0.65])
        delta = target - eye
        result.append(dict(location_cm=(eye * 100).tolist(), rotation_rpy_deg=[0, math.degrees(math.atan2(delta[2], np.linalg.norm(delta[:2]))), math.degrees(math.atan2(delta[1], delta[0]))], scale=[1, 1, 1]))
    return result


def plan(output=DEFAULT_OUTPUT, families=None, seed_offset=0):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Use an empty output directory; existing scene deliveries are preserved")
    config = json.loads(CONFIG.read_text())
    selected = [f for f in config["families"] if not families or f["id"] in families]
    if families and {f["id"] for f in selected} != set(families):
        raise ValueError("Unknown scene family")
    output.mkdir(parents=True, exist_ok=True)
    namespace = "/Game/SuperDyn/SyntheticMain/" + output.name
    jobs = []
    for original in selected:
        spec = deepcopy(original)
        spec["seed"] += seed_offset
        folder = output / spec["id"]
        folder.mkdir()
        layout, report = scaled_layout(spec["seed"], spec["xy_scale"])
        route = inspection_route(layout, config["render"]["frames"])
        vertices, faces, roles, _ = terrain_mesh(layout)
        terrain = mesh_record(vertices, faces, roles, spec["id"], route["reserved"])
        packet = dict(config["render"], schema=1, case_id=spec["id"], title=spec["title"],
                      coordinate_contract="Source layout X/Y maps directly to Unreal X/Y; metres to centimetres; Unreal camera X forward/Y right/Z up",
                      meshes=[terrain, context_mesh(layout)], camera=route["poses"], review_camera=review_camera(layout))
        with gzip.GzipFile(filename=str(folder / "scene.json.gz"), mode="wb", mtime=0) as stream:
            stream.write(json.dumps(packet, separators=(",", ":")).encode())
        specs = []
        for group in spec["asset_groups"]:
            asset = ASSETS[group["asset"]]
            for i in range(group["count"]):
                item = dict(group, id=f'{group["asset"]}_{i:02d}', source_pack=asset[0], mesh=asset[1])
                if len(asset) > 2:
                    item["animation"] = asset[2]
                for key in ["mesh", "animation"]:
                    if key in item and not (ROOT / "Content" / (item[key].removeprefix("/Game/") + ".uasset")).is_file():
                        raise RuntimeError("Installed source asset is missing: " + item[key])
                specs.append(item)
        save_json(folder / "layout.json", layout)
        save_json(folder / "terrain_validation.json", report)
        save_json(folder / "route.json", route)
        save_json(folder / "recipe.json", dict(family=spec, assets=specs, stage="Scene construction and inspection; no benchmark QA or dynamic root trajectories yet"))
        jobs.append(dict(id=spec["id"], output=str(folder), map_path=namespace + "/Maps/" + spec["id"], asset_root=namespace + "/" + spec["id"], input_sha256=sha256(folder / "scene.json.gz")))
        print(json.dumps(dict(family=spec["id"], seed=spec["seed"], levels=layout["levels"], ramps=len(layout["ramps"]), flat_fraction=report["flat_fraction"], route_vertical_span_m=route["vertical_span_m"])), flush=True)
    source_hashes = {str(p.relative_to(ROOT.parent)): sha256(p) for p in source_paths()}
    for relative in source_hashes:
        target = output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT.parent / relative).read_bytes())
    request = dict(schema=1, output=str(output), jobs=jobs, source_hashes=source_hashes,
                   spatial_samples=config["render"]["spatial_samples"], stage="scene_families")
    save_json(output / "request.json", request)
    save_json(output / "run_status.json", dict(status="planned", scenes=len(jobs)))
    return request


def verify(output=DEFAULT_OUTPUT):
    """Check immutable recipes, evaluated scene evidence and all delivered media."""
    import cv2
    from superdyn_ue.family_geometry import fits, overlaps

    output = Path(output).resolve()
    request = json.loads((output / "request.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest.get("status") != "passed":
        raise RuntimeError("Scene family delivery is not passed")
    for relative, expected in manifest["artifact_sha256"].items():
        if sha256(output / relative) != expected:
            raise RuntimeError("Scene artifact hash mismatch: " + relative)
    for relative, expected in request["source_hashes"].items():
        if sha256(output / "source_snapshot" / relative) != expected:
            raise RuntimeError("Source snapshot hash mismatch: " + relative)
    for relative, expected in manifest["saved_scene_assets_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError("Saved Unreal scene asset hash mismatch: " + relative)
    results = []
    cv2.setNumThreads(1)
    for job in request["jobs"]:
        folder = output / job["id"]
        layout = json.loads((folder / "layout.json").read_text())
        terrain = validate(layout, raise_on_failure=True)
        ground = Ground(layout)
        route = json.loads((folder / "route.json").read_text())
        scene = json.loads((folder / "scene_manifest.json").read_text())
        native = json.loads((folder / "native_validation.json").read_text())
        if native["status"] != "passed" or max(native[k] for k in ["terrain_support_error_cm", "camera_evaluation_error_cm", "eye_height_error_cm"]) > 0.1:
            raise RuntimeError("Native validation failed: " + job["id"])
        placed = []
        expected = sum(g["count"] for g in scene["family"]["asset_groups"])
        if scene["logical_objects"] != expected or len(scene["objects"]) != expected:
            raise RuntimeError("Scene object count differs from its recipe")
        for record in scene["objects"]:
            tree = record["asset"] in ("tree", "pine")
            support = fits(ground, record["anchor_m"], record["half_xy_m"], placed, route["reserved"], tree=tree)
            if support is None or abs(support - record["ground_z_m"]) > 1e-6:
                raise RuntimeError("Invalid object footprint or terrain support: " + record["id"])
            placed.append((*record["anchor_m"], *record["collision_half_xy_m"]))
        packet = json.loads(gzip.decompress((folder / "scene.json.gz").read_bytes()))
        for label, count in [("walk", packet["frames"]), ("overview", 4)]:
            frames = sorted((folder / label / "rgb").glob("*.png"))
            if [p.name for p in frames] != [f"frame_{i:06d}.png" for i in range(count)]:
                raise RuntimeError("Incomplete RGB sequence: " + str(folder / label))
            for frame in frames:
                value = cv2.imread(str(frame))
                if value is None or value.shape[:2] != (packet["height"], packet["width"]):
                    raise RuntimeError("Unreadable frame: " + str(frame))
        movie = cv2.VideoCapture(str(folder / "walk/sample.mp4"))
        try:
            if not movie.isOpened() or abs(movie.get(cv2.CAP_PROP_FPS) - packet["fps"]) > 1e-6:
                raise RuntimeError("Invalid inspection movie")
            decoded = 0
            while True:
                ok, frame = movie.read()
                if not ok:
                    break
                if frame.shape[:2] != (packet["height"], packet["width"]):
                    raise RuntimeError("Movie dimensions differ from RGB")
                decoded += 1
            if decoded != packet["frames"]:
                raise RuntimeError("Incomplete inspection movie")
        finally:
            movie.release()
        results.append(dict(id=job["id"], seed=scene["family"]["seed"], dimensions_m=[layout["bounds"][1] - layout["bounds"][0], layout["bounds"][3] - layout["bounds"][2]],
                            height_span_m=max(layout["levels"]) - min(layout["levels"]), levels=layout["levels"], ramps=len(layout["ramps"]),
                            ramp_area_fraction=1 - terrain["flat_fraction"], objects=len(placed), decoded_frames=decoded,
                            route_vertical_span_m=route["vertical_span_m"], native_support_error_cm=native["terrain_support_error_cm"]))
    return dict(status="passed", scenes=len(results), decoded_frames=sum(r["decoded_frames"] for r in results), overview_frames=4 * len(results), results=results)


def final_images(folder, packet):
    from PIL import Image, ImageDraw
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    scene = json.loads((folder / "scene_manifest.json").read_text())
    layout = json.loads((folder / "layout.json").read_text())
    route = json.loads((folder / "route.json").read_text())
    xs, ys, z, _, _ = raster(layout)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.pcolormesh(xs, ys, z, cmap="terrain", shading="nearest", vmin=0, vmax=max(layout["levels"]) + 0.5)
    xy = np.array(route["reserved"])
    ax.plot(xy[:, 0], xy[:, 1], color="crimson", linewidth=2, label="Reserved inspection / motion corridor")
    for record in scene["objects"]:
        x, y = record["anchor_m"]
        hx, hy = record["collision_half_xy_m"]
        ax.add_patch(Rectangle((x - hx, y - hy), hx * 2, hy * 2, fill=False, edgecolor="black", linewidth=0.7))
        ax.text(x, y, record["id"], fontsize=5, ha="center", clip_on=True)
    for ramp in layout["ramps"]:
        a, b, c, d = ramp["bounds"]
        ax.add_patch(Rectangle((a, c), b - a, d - c, fill=False, edgecolor="orange", linewidth=1.2))
    fig.colorbar(im, ax=ax, label="Ground elevation (m)")
    ax.set(title=scene["family"]["title"], xlabel="Unreal X (m)", ylabel="Unreal Y (m)", aspect="equal")
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(folder / "layout.png", dpi=150)
    plt.close(fig)
    tiles = [(folder / "overview/rgb" / f"frame_{i:06d}.png", f"Overview {i + 1}") for i in range(4)]
    tiles += [(folder / "walk/rgb" / f"frame_{i:06d}.png", f"Eye-level F{i}") for i in [0, 64, 128, packet["frames"] - 1]]
    sheet = Image.new("RGB", (640 * 4, 390 * 2), "#182028")
    draw = ImageDraw.Draw(sheet)
    for index, (path, title) in enumerate(tiles):
        x, y = (index % 4) * 640, (index // 4) * 390
        with Image.open(path) as source:
            sheet.paste(source.resize((640, 360)), (x, y + 30))
        draw.text((x + 12, y + 8), title, fill="white")
    sheet.save(folder / "contact_sheet.jpg", quality=92)


def finalize(output=DEFAULT_OUTPUT):
    from bridge import encode_and_verify, verify_frame_sequence
    from PIL import Image, ImageDraw

    output = Path(output).resolve()
    if (output / "manifest.json").exists():
        return verify(output)
    request = json.loads((output / "request.json").read_text())
    native = json.loads((output / "native_status.json").read_text())
    if native["stage"] != "frames_complete":
        raise RuntimeError("Native scene rendering is incomplete")
    summaries = []
    artifacts = [output / "request.json", output / "native_status.json", output / "launch.json"]
    saved_assets = {}
    for job in request["jobs"]:
        folder = output / job["id"]
        packet = json.loads(gzip.decompress((folder / "scene.json.gz").read_bytes()))
        validation = json.loads((folder / "native_validation.json").read_text())
        if validation["status"] != "passed":
            raise RuntimeError("Native scene validation did not pass")
        verify_frame_sequence(folder / "walk", packet)
        verify_frame_sequence(folder / "overview", dict(packet, frames=4))
        encode_and_verify(folder / "walk", packet, input_path=folder / "scene.json.gz")
        final_images(folder, packet)
        layout = json.loads((folder / "layout.json").read_text())
        scene = json.loads((folder / "scene_manifest.json").read_text())
        terrain = json.loads((folder / "terrain_validation.json").read_text())
        summaries.append(dict(id=job["id"], title=scene["family"]["title_ko"], seed=scene["family"]["seed"], objects=scene["logical_objects"],
                              dimensions_m=[round(layout["bounds"][1] - layout["bounds"][0], 2), round(layout["bounds"][3] - layout["bounds"][2], 2)],
                              levels=layout["levels"], ramps=len(layout["ramps"]), ramp_area_fraction=1 - terrain["flat_fraction"], map_path=job["map_path"]))
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix in (".json", ".gz", ".png", ".jpg", ".mp4"):
                artifacts.append(path)
        prefix = job["asset_root"].removeprefix("/Game/")
        for path in (ROOT / "Content" / prefix).rglob("*.uasset"):
            saved_assets[str(path.relative_to(ROOT))] = sha256(path)
        map_file = ROOT / "Content" / (job["map_path"].removeprefix("/Game/") + ".umap")
        if not map_file.is_file():
            raise RuntimeError("Saved scene map is missing")
        saved_assets[str(map_file.relative_to(ROOT))] = sha256(map_file)
    overview = Image.new("RGB", (1920, len(summaries) * 390), "#182028")
    draw = ImageDraw.Draw(overview)
    for row, item in enumerate(summaries):
        folder = output / item["id"]
        for col, path in enumerate([folder / "overview/rgb/frame_000000.png", folder / "overview/rgb/frame_000001.png", folder / "walk/rgb/frame_000096.png"]):
            with Image.open(path) as source:
                overview.paste(source.resize((640, 360)), (col * 640, row * 390 + 30))
        draw.text((12, row * 390 + 8), item["id"] + " | independent terrain / family dressing / eye-level view", fill="white")
    overview.save(output / "overview.jpg", quality=93)
    lines = ["# Synthetic main: 세 장면군 제작 결과", "", "산업 야드·도시 광장·공원은 서로 다른 seed의 연결 단차 지형과 장면별 에셋 배치로 구성했다.",
             "각 장면은 1280×720, 24fps, 8초 눈높이 검사 영상과 네 방향의 전체 조망 렌더를 포함한다.", "", "![세 장면 비교](overview.jpg)", "",
             "| 장면 | 지형 크기 | 높이 레벨 | 경사로 / 면적 비율 | 배치 객체 | 확인 |", "|---|---|---|---|---:|---|"]
    for item in summaries:
        key = item["id"]
        lines.append(f'| {item["title"]} | {item["dimensions_m"][0]} × {item["dimensions_m"][1]}m | {item["levels"]}m | {item["ramps"]}개 / {item["ramp_area_fraction"]:.1%} | {item["objects"]} | [영상]({key}/walk/sample.mp4) · [8장]({key}/contact_sheet.jpg) · [배치도]({key}/layout.png) |')
    lines += ["", "검증 범위: 기존 지형 생성기의 연결성·경사·평지 면적 검사, 실제 메시 바운드의 배치/통로 회피, Unreal 지면 ray query와 전 프레임 카메라 높이·시퀀서 평가, 전체 RGB·MP4와 저장 맵/시퀀스 해시.",
              "", "이 산출물은 장면군 구현 단계다. 차량은 정차하고 사람·동물은 대기 동작을 재생한다. 독립적인 객체 이동, 네 카메라 조건, dense mask/depth·표면점 GT와 T1–T6 QA는 아직 생성하지 않았다.",
              "나무 배치 충돌은 줄기 footprint를 사용하며 전체 수관과 모든 idle 동작의 접촉을 인증하지 않는다.",
              "", "재현: `unreal_synthetic/.venv/bin/python unreal_synthetic/manage.py scenes verify`.",
              "새 장면: `scenes render --output unreal_synthetic/outputs/new_scene_families --seed-offset 10 --gpu 5`.",
              "각 장면의 `layout.json`, `recipe.json`, `scene_manifest.json`, `native_validation.json`, `evaluated_scene.json`과 소스 스냅샷을 보존한다."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    artifacts += [output / "overview.jpg", output / "REPORT.md"]
    manifest = dict(status="passed", stage="scene_families", scenes=summaries,
                    artifact_sha256={str(path.relative_to(output)): sha256(path) for path in sorted(set(artifacts))},
                    saved_scene_assets_sha256=saved_assets)
    save_json(output / "manifest.json", manifest)
    result = verify(output)
    save_json(output / "verification.json", result)
    save_json(output / "run_status.json", result)
    return result
