"""Answer-first clip designs in a stage-local frame.

Stage-local metres: f forward (initial view direction), r right, h up. A clip
is placed on a surveyed map by choosing an origin and yaw (pilot_stage.py).
Every design lists the QA it is built to produce; GT later decides whether
each intended answer actually holds.
"""

import math

import numpy as np

FPS = 24
FRAMES = 129
MEASURE = list(range(0, FRAMES, 16))
T = (FRAMES - 1) / FPS
TIMES = np.arange(FRAMES) / FPS

# Mesh/clip paths are resolved and calibrated natively; sizes are targets.
ASSETS = {
    "rocketbox_man": dict(category="person", mesh="/Game/SuperDyn/SourceModels/microsoft_rocketbox/Demo",
                          clips=dict(walk="/Game/SuperDyn/SourceModels/microsoft_rocketbox/Demo_Anim"), height_m=1.78,
                          description="man in casual clothes"),
    "twinblast": dict(category="person", mesh="/Game/ParagonTwinblast/Characters/Heroes/TwinBlast/Meshes/TwinBlast",
                      clips=dict(walk="/Game/ParagonTwinblast/Characters/Heroes/TwinBlast/Animations/Walk_Fwd",
                                 run="/Game/ParagonTwinblast/Characters/Heroes/TwinBlast/Animations/Jog_Fwd",
                                 idle="/Game/ParagonTwinblast/Characters/Heroes/TwinBlast/Animations/Idle_Relaxed"),
                      height_m=1.82, description="man in a long coat"),
    "sparrow": dict(category="person", mesh="/Game/ParagonSparrow/Characters/Heroes/Sparrow/Meshes/Sparrow",
                    clips=dict(run="/Game/ParagonSparrow/Characters/Heroes/Sparrow/Animations/Jog_Fwd",
                               idle="/Game/ParagonSparrow/Characters/Heroes/Sparrow/Animations/Idle_FrontEnd"),
                    height_m=1.70, description="woman with a bow"),
    "gadget": dict(category="person", mesh="/Game/ParagonGadget/Characters/Heroes/Gadget/Meshes/Gadget",
                   clips=dict(run="/Game/ParagonGadget/Characters/Heroes/Gadget/Animations/Run_Fwd",
                              idle="/Game/ParagonGadget/Characters/Heroes/Gadget/Animations/Idle_Relaxed"),
                   height_m=1.66, description="woman with goggles"),
    "wolf": dict(category="animal", mesh="/Game/AnimalVarietyPack/Wolf/Meshes/SK_Wolf",
                 clips=dict(walk="/Game/AnimalVarietyPack/Wolf/Animations/ANIM_Wolf_Walk",
                            run="/Game/AnimalVarietyPack/Wolf/Animations/ANIM_Wolf_Run",
                            idle="/Game/AnimalVarietyPack/Wolf/Animations/ANIM_Wolf_IdleBreathe"),
                 length_m=1.45, description="gray wolf"),
    "deer": dict(category="animal", mesh="/Game/AnimalVarietyPack/DeerStagAndDoe/Meshes/SK_DeerStag",
                 clips=dict(walk="/Game/AnimalVarietyPack/DeerStagAndDoe/Animations/ANIM_DeerStag_Walk",
                            run="/Game/AnimalVarietyPack/DeerStagAndDoe/Animations/ANIM_DeerStag_Run",
                            idle="/Game/AnimalVarietyPack/DeerStagAndDoe/Animations/ANIM_DeerStag_IdleBreathe"),
                 length_m=2.0, description="stag deer with antlers"),
    "doe": dict(category="animal", mesh="/Game/AnimalVarietyPack/DeerStagAndDoe/Meshes/SK_DeerDoe",
                clips=dict(walk="/Game/AnimalVarietyPack/DeerStagAndDoe/Animations/ANIM_DeerDoe_Walk",
                           run="/Game/AnimalVarietyPack/DeerStagAndDoe/Animations/ANIM_DeerDoe_Run",
                           idle="/Game/AnimalVarietyPack/DeerStagAndDoe/Animations/ANIM_DeerDoe_IdleBreathe"),
                length_m=1.8, description="female deer"),
    "pig": dict(category="animal", mesh="/Game/AnimalVarietyPack/Pig/Meshes/SK_Pig",
                clips=dict(walk="/Game/AnimalVarietyPack/Pig/Animations/ANIM_Pig_Walk",
                           run="/Game/AnimalVarietyPack/Pig/Animations/ANIM_Pig_Run",
                           idle="/Game/AnimalVarietyPack/Pig/Animations/ANIM_Pig_IdleBreathe"),
                length_m=1.2, description="pink pig"),
    "fox": dict(category="animal", mesh="/Game/AnimalVarietyPack/Fox/Meshes/SK_Fox",
                clips=dict(walk="/Game/AnimalVarietyPack/Fox/Animations/ANIM_Fox_Walk",
                           run="/Game/AnimalVarietyPack/Fox/Animations/ANIM_Fox_Run",
                           idle="/Game/AnimalVarietyPack/Fox/Animations/ANIM_Fox_IdleBreathe"),
                length_m=1.0, description="red fox"),
    "crow": dict(category="animal", mesh="/Game/AnimalVarietyPack/Crow/Meshes/SK_Crow",
                 clips=dict(walk="/Game/AnimalVarietyPack/Crow/Animations/ANIM_Crow_Walk",
                            fly="/Game/AnimalVarietyPack/Crow/Animations/ANIM_Crow_Fly",
                            idle="/Game/AnimalVarietyPack/Crow/Animations/ANIM_Crow_IdleLookAround"),
                 length_m=0.75, description="black crow"),
}


def ease(u):
    u = np.clip(u, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(math.pi * u)


def window_fraction(t0=0.0, t1=T):
    return ease((TIMES - t0) / max(1e-6, t1 - t0))


def look(cf, cr, ch, tf, tr, th):
    """Local yaw/pitch (degrees) from camera point to target point."""
    df, dr, dh = np.asarray(tf) - cf, np.asarray(tr) - cr, np.asarray(th) - ch
    return np.degrees(np.arctan2(dr, df)), np.degrees(np.arctan2(dh, np.hypot(df, dr)))


def jitter(seed, pos_cm=1.2, rot_deg=0.35):
    """Smooth deterministic handheld shake: (df, dr, dh) metres and (yaw, pitch, roll) degrees."""
    rng = np.random.default_rng(seed)
    out = []
    for amplitude in [pos_cm / 100] * 3 + [rot_deg] * 3:
        value = np.zeros(FRAMES)
        for hz in [0.35, 0.8, 1.7]:
            value += math.sin(rng.uniform(0, 2 * math.pi)) * 0 + np.sin(2 * math.pi * hz * TIMES + rng.uniform(0, 2 * math.pi)) / hz
        out.append(amplitude * value / np.abs(value).max())
    return out


def camera(f, r, h, yaw, pitch, z_mode="stage", fov=75.0, shake=None, primitive=""):
    f, r, h, yaw, pitch = [np.broadcast_to(np.asarray(v, float), (FRAMES,)).copy() for v in [f, r, h, yaw, pitch]]
    roll = np.zeros(FRAMES)
    if shake is not None:
        j = jitter(shake)
        f, r, h, yaw, pitch, roll = f + j[0], r + j[1], h + j[2], yaw + j[3], pitch + j[4], roll + j[5] * 0.5
    return dict(primitive=primitive, z_mode=z_mode, fov_deg=fov, f=f, r=r, h=h, yaw=yaw, pitch=pitch, roll=roll)


def path(points, speed, start_s=0.0, ramp_s=0.5):
    """Constant-speed polyline with eased start/stop; returns f, r, heading, moving flags."""
    pts = np.asarray(points, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(seg.sum())
    duration = total / speed + ramp_s
    u = (TIMES - start_s) / duration
    # Trapezoid-like profile: eased ends, linear middle, normalised to total length.
    profile = np.where(u <= 0, 0.0, np.where(u >= 1, 1.0, u - (ramp_s / duration) * np.sin(2 * math.pi * u) / (2 * math.pi)))
    s = np.clip(profile, 0, 1) * total
    cum = np.r_[0, np.cumsum(seg)]
    f = np.interp(s, cum, pts[:, 0])
    r = np.interp(s, cum, pts[:, 1])
    k = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1)
    d = pts[k + 1] - pts[k]
    heading = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    moving = (u > 0) & (u < 1)
    return f, r, heading, moving


def actor(asset, label, f, r, heading, moving, gait="walk", h=None, z_mode="ground", speed=None, role="target"):
    return dict(asset=asset, label=label, role=role, f=np.asarray(f, float), r=np.asarray(r, float),
                heading=np.asarray(heading, float), moving=np.asarray(moving, bool), gait=gait,
                h=None if h is None else np.asarray(h, float), z_mode=z_mode, speed_mps=speed)


def standing(asset, label, f, r, heading, role="target", gait="idle"):
    z = np.zeros(FRAMES)
    return actor(asset, label, z + f, z + r, z + heading, np.zeros(FRAMES, bool), gait=gait, role=role)


def walker(asset, label, points, speed, gait=None, start_s=0.3, role="target"):
    f, r, heading, moving = path(points, speed, start_s)
    gait = gait or ("run" if speed > 2.2 else "walk")
    return actor(asset, label, f, r, heading, moving, gait=gait, speed=speed, role=role)


def flyer(asset, label, start, end, start_s=0.4, role="target"):
    """Straight 3D flight (f, r, h) with an eased profile."""
    u = window_fraction(start_s, T - 0.2)
    p = np.asarray(start, float)[None] + u[:, None] * (np.asarray(end, float) - np.asarray(start, float))[None]
    heading = np.full(FRAMES, math.degrees(math.atan2(end[1] - start[1], end[0] - start[0])))
    return actor(asset, label, p[:, 0], p[:, 1], heading, u > 0, gait="fly", h=p[:, 2], z_mode="stage")


def q(task, answer, window=(0, 8), objects=(), cue=None):
    return dict(task=task, window=list(window), answer=answer, objects=list(objects), cue=cue)


T1, T2, T3, T4, T5, T6 = ("camera_translation_direction", "camera_view_direction_change", "object_world_motion_direction",
                          "camera_object_distance_change", "object_pair_distance_change", "object_travel_distance_comparison")


def designs():
    """Pilot clip set: S-Matched clips and two S-Counterfactual pairs."""
    clips = []
    u = window_fraction(0.0, T)

    # 1. Truck right while a man crosses left and a wolf trots away.
    cam = camera(0, -2.5 + 5.0 * u, 1.6, 0, -4, shake=11, primitive="truck_right")
    clips.append(dict(id="m01_truck_right", split="matched", cam=cam, actors=[
        walker("twinblast", "A", [[9, 3.5], [9, -3.5]], 1.4),
        walker("wolf", "B", [[6, 1.5], [13, 2.0]], 1.5),
        standing("doe", None, 15, -6, 120, role="distractor")],
        design=[q(T1, "right"), q(T3, "left", objects=["A"], cue="consistent"), q(T3, "forward", objects=["B"]),
                q(T4, "farther", objects=["B"]), q(T5, "increased", objects=["A", "B"])]))

    # 2. Pan left following a running fox that crosses perpendicular to the F0 view; a pig crosses right.
    yaw0 = math.degrees(math.atan2(4.0, 10.0))
    left = np.array([math.cos(math.radians(yaw0 - 90)), math.sin(math.radians(yaw0 - 90))])
    right = -left
    fox = walker("fox", "A", [[10.0, 4.0], list(np.array([10.0, 4.0]) + 12.0 * left)], 2.6)
    yaw, pitch = look(0, 0, 1.5, fox["f"], fox["r"], 0.3)
    cam = camera(0, 0, 1.5, yaw, pitch, shake=12, primitive="pan_left")
    clips.append(dict(id="m02_pan_left", split="matched", cam=cam, actors=[
        fox, walker("pig", "B", [[8.0, -2.5], list(np.array([8.0, -2.5]) + 4.0 * right)], 0.85),
        standing("sparrow", None, 13, 2, 200, role="distractor")],
        design=[q(T2, "left"), q(T3, "left", objects=["A"]), q(T3, "right", objects=["B"]),
                q(T6, "A", objects=["A", "B"])]))

    # 3. Handheld walk forward; a man walks away slower, a stag walks toward camera.
    cam = camera(0.0 + 8.0 * u, 0, 1.62, 0, -3, shake=13, primitive="walk_forward")
    clips.append(dict(id="m03_walk_forward", split="matched", cam=cam, actors=[
        walker("rocketbox_man", "A", [[8.0, -0.7], [11.5, -0.7]], 0.72),
        walker("deer", "B", [[17, 2.5], [11, 2.0]], 1.2),
        standing("pig", None, 10, -4, 90, role="distractor")],
        design=[q(T1, "forward"), q(T3, "forward", objects=["A"]), q(T3, "backward", objects=["B"]),
                q(T4, "closer", objects=["A"], cue="conflict"), q(T4, "closer", objects=["B"]),
                q(T5, "decreased", objects=["A", "B"])]))

    # 4. Pull back while two runners approach.
    cam = camera(0.0 - 4.5 * u, 0, 1.6, 0, -2, shake=14, primitive="pull_back")
    clips.append(dict(id="m04_pull_back", split="matched", cam=cam, actors=[
        walker("sparrow", "A", [[15, -1.5], [4, -1.0]], 2.4),
        walker("wolf", "B", [[13, 3.0], [9, 1.0]], 1.0),
        standing("deer", None, 18, 6, 220, role="distractor")],
        design=[q(T1, "backward"), q(T3, "backward", objects=["A"]), q(T4, "closer", objects=["A"]),
                q(T6, "A", objects=["A", "B"])]))

    # 5. Crane up with tilt down; pig and fox diverge.
    h = 1.4 + 3.6 * u
    _, pitch = look(0, 0, h, 8.0, 0.0, 0.0)
    cam = camera(0, 0, h, 0, pitch, shake=15, primitive="crane_up_tilt_down")
    clips.append(dict(id="m05_crane_up", split="matched", cam=cam, actors=[
        walker("pig", "A", [[8, -1.5], [8.5, -6.0]], 0.9),
        walker("fox", "B", [[8, 1.5], [9.0, 9.5]], 1.7),
        standing("twinblast", None, 13, 1, 180, role="distractor")],
        design=[q(T1, "up"), q(T2, "down"), q(T3, "left", objects=["A"]), q(T3, "right", objects=["B"]),
                q(T5, "increased", objects=["A", "B"]), q(T6, "B", objects=["A", "B"])]))

    # 6. Crane down while a crow takes off; a deer walks right.
    h = 4.2 - 3.0 * u
    cam = camera(0, 0, h, 0, -10 + 6 * u, shake=16, primitive="crane_down")
    clips.append(dict(id="m06_crane_down_crow_up", split="matched", cam=cam, actors=[
        flyer("crow", "A", [6.0, -1.0, 0.25], [6.8, -0.5, 4.6]),
        walker("doe", "B", [[11, -4.0], [11.5, 3.0]], 1.3),
        standing("wolf", None, 9, 4, 250, role="distractor")],
        design=[q(T1, "down"), q(T3, "up", objects=["A"]), q(T3, "right", objects=["B"])]))

    # 7. Static tilt up following a rising crow; then a landing crow (down) in the same take.
    crow = flyer("crow", "A", [7.0, 0.5, 0.3], [7.6, 1.0, 4.4], start_s=0.2)
    _, pitch = look(0, 0, 1.5, crow["f"], crow["r"], crow["h"])
    cam = camera(0, 0, 1.5, 4, pitch * 0.9, shake=None, primitive="tilt_up")
    clips.append(dict(id="m07_tilt_up", split="matched", cam=cam, actors=[
        crow, flyer("crow", "B", [9.5, -3.0, 3.8], [9.0, -2.2, 0.25], start_s=0.6),
        standing("gadget", None, 11, -1, 160, role="distractor")],
        design=[q(T2, "up"), q(T3, "up", objects=["A"]), q(T3, "down", objects=["B"])]))

    # 8. Orbit left around two people walking toward each other.
    angle = np.radians(-35 * u)
    cf, cr = 9.0 - 9.0 * np.cos(angle), -9.0 * np.sin(angle)
    yaw, pitch = look(cf, cr, 1.6, 9.0, 0.0, 0.9)
    cam = camera(cf, cr, 1.6, yaw, pitch, shake=18, primitive="orbit")
    clips.append(dict(id="m08_orbit", split="matched", cam=cam, actors=[
        walker("twinblast", "A", [[9.5, -4.0], [9.5, -0.7]], 0.75),
        walker("gadget", "B", [[8.5, 4.5], [8.5, 0.8]], 0.8, gait="run"),
        standing("pig", None, 14, 3, 300, role="distractor")],
        design=[q(T1, "right"), q(T2, "left"), q(T3, "right", objects=["A"]), q(T3, "left", objects=["B"]),
                q(T5, "decreased", objects=["A", "B"])]))

    # Counterfactual pair 1 (T3): same screen motion of A (leftward), opposite world direction.
    clips.append(dict(id="c1a_static_left", split="counterfactual", pair="cf1", cam=camera(0, 0, 1.6, 0, -4, primitive="static"),
                      actors=[walker("rocketbox_man", "A", [[9, 2.0], [9, -2.0]], 0.8),
                              standing("wolf", None, 12, 4, 200, role="distractor")],
                      design=[q(T3, "left", objects=["A"], cue="consistent")]))
    cam = camera(0, -3.5 + 7.5 * u, 1.6, 0, -4, shake=21, primitive="truck_right")
    clips.append(dict(id="c1b_truck_right", split="counterfactual", pair="cf1", cam=cam,
                      actors=[walker("rocketbox_man", "A", [[9, -1.0], [9, 2.0]], 0.62),
                              standing("wolf", None, 12, 4, 200, role="distractor")],
                      design=[q(T1, "right"), q(T3, "right", objects=["A"], cue="conflict")]))

    # Counterfactual pair 2 (T6): A nearly fixed on screen in both; who actually travelled more flips.
    cam = camera(0, -3.5 + 7.0 * u, 1.6, 0, -4, shake=22, primitive="follow_truck")
    clips.append(dict(id="c2a_follow", split="counterfactual", pair="cf2", cam=cam, actors=[
        walker("sparrow", "A", [[8, -3.5], [8, 3.5]], 1.45, gait="run", start_s=0.0),
        walker("deer", "B", [[12, 3.0], [12, 0.0]], 0.7)],
        design=[q(T6, "A", objects=["A", "B"], cue="conflict")]))
    clips.append(dict(id="c2b_static", split="counterfactual", pair="cf2", cam=camera(0, 0, 1.6, 0, -4, primitive="static"), actors=[
        standing("sparrow", "A", 8, 0.0, 90),
        walker("deer", "B", [[12, 6.5], [12, -3.5]], 2.0, gait="walk")],
        design=[q(T6, "B", objects=["A", "B"], cue="consistent")]))
    return clips
