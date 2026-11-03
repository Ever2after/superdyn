"""Pilot orchestration: place designs on surveyed maps, render, then build QA."""

import json
from pathlib import Path

import numpy as np

from bridge import ROOT
from superdyn_ue.io_utils import save_json
from pilot_design import ASSETS, FPS, FRAMES, MEASURE, designs
from pilot_stage import Survey, place_any, to_world

CONFIG = ROOT / "configs/pilot_split.json"


def segments(spec):
    """Contiguous animation sections: moving frames use the gait clip."""
    if spec["gait"] == "fly":
        return [[0, FRAMES, "fly"]]
    clips = ASSETS[spec["asset"]]["clips"]
    still = "idle" if "idle" in clips else "hold"
    out = []
    for k, moving in enumerate(spec["moving"]):
        gait = spec["gait"] if moving and spec["gait"] != "idle" else still
        if out and out[-1][2] == gait:
            out[-1][1] = k + 1
        else:
            out.append([k, k + 1, gait])
    return out


def world_actor(spec, index, origin, yaw, ground_z_cm):
    a = ASSETS[spec["asset"]]
    x, y = to_world(origin, yaw, spec["f"], spec["r"])
    person = a["category"] == "person"
    return dict(id=spec["label"] or "D%d" % index, label=spec["label"], role=spec["role"], asset=spec["asset"],
                category=a["category"], description=a["description"], mesh=a["mesh"], clips=a["clips"],
                size_cm=100 * (a["height_m"] if person else a["length_m"]), size_basis="height" if person else "length",
                stencil=10 + index, x_cm=(x * 100).tolist(), y_cm=(y * 100).tolist(),
                yaw_deg=(yaw + spec["heading"]).tolist(), z_mode=spec["z_mode"],
                h_cm=None if spec["h"] is None else (spec["h"] * 100).tolist(), speed_mps=spec["speed_mps"],
                gait=spec["gait"], segments=segments(spec))


def world_camera(cam, origin, yaw):
    x, y = to_world(origin, yaw, cam["f"], cam["r"])
    return dict(primitive=cam["primitive"], z_mode=cam["z_mode"], fov_deg=cam["fov_deg"], x_cm=(x * 100).tolist(), y_cm=(y * 100).tolist(),
                h_cm=(cam["h"] * 100).tolist(), yaw_deg=(yaw + cam["yaw"]).tolist(), pitch_deg=cam["pitch"].tolist(), roll_deg=cam["roll"].tolist())


def plan(output, only_env=None):
    config = json.loads(CONFIG.read_text())
    envs = {e["id"]: e for e in config["environments"]}
    clips = {c["id"]: c for c in designs()}
    surveys = {}
    placed, records, used = [], [], {}
    groups = {}
    for cid, env in config["assignments"].items():
        clip = clips[cid]
        groups.setdefault(clip.get("pair", cid), []).append((clip, env))
    for key, members in groups.items():
        preferred = members[0][1]
        stage = None
        for env in [preferred] + [e for e in envs if e != preferred]:
            if only_env and env not in only_env:
                continue
            if env not in surveys:
                files = sorted((output / "survey").glob(env + "_detail_*.json.gz")) or [output / "survey" / (env + ".json.gz")]
                surveys[env] = [Survey(f, envs[env].get("exclude_surfaces", ()), envs[env].get("paved_surfaces", ())) for f in files]
            try:
                stage = place_any(surveys[env], [m[0] for m in members], avoid=used.get(env, []), seed=len(placed))
                break
            except RuntimeError as error:
                print(json.dumps(dict(clip=key, env=env, placement_failed=str(error))), flush=True)
        if stage is None:
            raise RuntimeError("No environment can host " + key)
        used.setdefault(env, []).append(stage["origin_m"])
        for clip, _ in members:
            origin, yaw = stage["origin_m"], stage["yaw_deg"]
            ground = stage["ground_z_m"] * 100
            actors = [world_actor(s, i, origin, yaw, ground) for i, s in enumerate(clip["actors"])]
            placed.append(dict(id=clip["id"], split=clip["split"], pair=clip.get("pair"), env=env, env_map=envs[env]["map"],
                               exposure=envs[env].get("exposure"), stage=stage, ground_z_cm=ground,
                               camera=world_camera(clip["cam"], origin, yaw), actors=actors, design=clip["design"]))
            records.append(dict(id=clip["id"], env=env, **{k: stage[k] for k in ["origin_m", "yaw_deg", "score", "view_structure", "view_void", "stage_paved", "window"]}))
    request = dict(stage="clips", namespace="/Game/SuperDyn/Pilot/" + output.name, clips=placed, fps=FPS, frames=FRAMES,
                   measure=MEASURE, width=1280, height=720, spatial_samples=config["render"]["spatial_samples"],
                   gt_frame_step=4, passes=["rgb", "gt"])
    save_json(output / "plan.json", dict(placements=records, clips=placed))
    preview(output, surveys, placed)
    return request


def preview(output, surveys, placed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for env, windows in surveys.items():
      for n, survey in enumerate(windows):
        if not any(c["env"] == env and c["stage"].get("window", 0) == n for c in placed):
            continue
        fig, ax = plt.subplots(figsize=(11, 10))
        top = np.where(np.isfinite(survey.top), survey.top, np.nan)
        extent = [survey.x0, survey.x0 + survey.step * top.shape[1], survey.y0, survey.y0 + survey.step * top.shape[0]]
        ax.imshow(top, origin="lower", extent=extent, cmap="terrain")
        ax.imshow(np.where(survey.walkable, np.nan, 1.0), origin="lower", extent=extent, cmap="gray", alpha=0.35, vmin=0, vmax=1.5)
        for clip in placed:
            if clip["env"] != env or clip["stage"].get("window", 0) != n:
                continue
            cam = clip["camera"]
            ax.plot(np.array(cam["x_cm"]) / 100, np.array(cam["y_cm"]) / 100, "k-", lw=2)
            ax.text(cam["x_cm"][0] / 100, cam["y_cm"][0] / 100, clip["id"], fontsize=7)
            for a in clip["actors"]:
                ax.plot(np.array(a["x_cm"]) / 100, np.array(a["y_cm"]) / 100, "-", lw=1.5, color="orange" if a["label"] == "A" else "lime" if a["label"] == "B" else "red")
        ax.set(title="%s window %d (gray = not walkable)" % (env, n), xlabel="Unreal X (m)", ylabel="Unreal Y (m)")
        fig.savefig(output / ("plan_%s_%d.png" % (env, n)), dpi=110)
        plt.close(fig)


def render_timing(output, request, attempt):
    """Per-job MRQ durations from the editor log, in queue order."""
    import re
    order, envs = [], []
    for clip in request["clips"]:
        if clip["env"] not in envs:
            envs.append(clip["env"])
    for env in envs:
        for clip in request["clips"]:
            if clip["env"] == env:
                order += [clip["id"] + "_" + kind for kind in request["passes"]]
    text = (output / "native" / attempt / "editor.log").read_text(errors="replace")
    durations = re.findall(r"Movie Pipeline completed\. Duration: \+(\d+):(\d+):([\d.]+)", text)
    seconds = [int(h) * 3600 + int(m) * 60 + float(s) for h, m, s in durations]
    jobs = dict(zip(order, seconds))
    frames = {"rgb": request["frames"], "gt": len(range(0, request["frames"], request["gt_frame_step"]))}
    per_frame = {kind: sum(v for k, v in jobs.items() if k.endswith("_" + kind)) / max(1, frames[kind] * len(request["clips"])) for kind in request["passes"]}
    return dict(jobs_s=jobs, seconds_per_frame=per_frame, spatial_samples_rgb=request["spatial_samples"], gpu_shared=True)


def run(action, output, gpu, env=None):
    output = Path(output)
    if action == "plan":
        request = plan(output, env)
        save_json(output / "clips_request.json", request)
        return dict(status="planned", clips=len(request["clips"]))
    if action == "render":
        from pilot import launch, render_deferred
        request = json.loads((output / "clips_request.json").read_text())
        if env:
            request["clips"] = [c for c in request["clips"] if c["env"] in env]
        # Each attempt gets its own content namespace: failed sessions may leave saved maps.
        name = "clips%d" % len([p for p in (output / "native").glob("clips*")])
        request["namespace"] = "/Game/SuperDyn/Pilot/" + output.name + "_" + name
        built = launch(dict(request, output=str(output.resolve()), defer_render=True), gpu, timeout_s=4 * 3600, nullrhi=True, attempt_name=name)
        rendered = render_deferred(output, gpu, name)
        save_json(output / "render_timing.json", render_timing(output, request, "render_" + name))
        return dict(build=built, render=rendered, namespace=request["namespace"])
    if action in ("qa", "report"):
        import pilot_qa
        return pilot_qa.run(output, action)
    raise ValueError(action)
