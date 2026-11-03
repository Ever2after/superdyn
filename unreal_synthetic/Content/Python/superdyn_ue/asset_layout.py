"""Pure layout math for fitting official visuals to the original GT anchors."""

import math

ASSET_REVISION = "epic-templates-v3"
PERSON_FOLDER = "/Game/Characters/Mannequins"
WALK_CLIP = PERSON_FOLDER + "/Anims/Unarmed/Walk/MF_Unarmed_Walk_Fwd"
OFFROAD_MESH = "/Game/Vehicles/OffroadCar/SKM_Offroad"
OFFROAD_BODY = "/Game/Vehicles/OffroadCar/SM_Offroad_Body"
OFFROAD_TIRE = "/Game/Vehicles/OffroadCar/SM_Offroad_Tire"


def person_mesh(case):
    variant = "Manny" if int(case.split("_", 1)[0]) % 2 else "Quinn"
    return PERSON_FOLDER + "/Meshes/SKM_" + variant + "_Simple"


def record_bounds(record):
    points = [v for section in record["sections"] for v in section["vertices_cm"]]
    return (
        [min(p[i] for p in points) for i in range(3)],
        [max(p[i] for p in points) for i in range(3)],
    )


def fit_visual(source_min, source_max, target_min, target_max, person=False):
    """Keep proportions, fit height/person or footprint/car, and align ground Z."""
    size = [source_max[i] - source_min[i] for i in range(3)]
    target = [target_max[i] - target_min[i] for i in range(3)]
    if min(size + target) <= 0:
        raise ValueError("Visual and target bounds must have positive dimensions")
    scale = target[2] / size[2] if person else min(target[i] / size[i] for i in [0, 1])
    yaw = -90.0 if person else 0.0  # UE mannequin +Y forward -> logical +X.
    angle = math.radians(yaw)
    center = [(source_min[i] + source_max[i]) * 0.5 for i in [0, 1]]
    offset = [
        -scale * (math.cos(angle) * center[0] - math.sin(angle) * center[1]),
        -scale * (math.sin(angle) * center[0] + math.cos(angle) * center[1]),
        target_min[2] - scale * source_min[2],
    ]
    return dict(location_cm=offset, rotation_rpy_deg=[0.0, 0.0, yaw], scale=[scale] * 3)


def walking_play_rate(poses, fps):
    distance = sum(
        math.dist(a["location_cm"][:2], b["location_cm"][:2])
        for a, b in zip(poses, poses[1:])
    )
    seconds = (len(poses) - 1) / fps
    scale = poses[0]["scale"][0]
    # Template forward walk is approximately 150 cm/s at original mesh scale.
    return max(0.1, min(3.0, distance / seconds / (150.0 * scale)))
