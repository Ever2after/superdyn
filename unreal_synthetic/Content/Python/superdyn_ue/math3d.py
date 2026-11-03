"""Explicit Blender metres/right-handed to Unreal centimetres/left-handed bridge."""

import math

S = (1.0, -1.0, 1.0)
CAMERA_AXES = ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (-1.0, 0.0, 0.0))


def transpose(m):
    return [list(row) for row in zip(*m)]


def multiply(a, b):
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def point_to_unreal(point):
    return [100.0 * S[i] * point[i] for i in range(3)]


def point_to_blender(point):
    return [S[i] * point[i] / 100.0 for i in range(3)]


def actor_matrix_to_unreal(m):
    return [
        [S[i] * m[i][j] * S[j] if j < 3 else 100.0 * S[i] * m[i][j] for j in range(4)]
        for i in range(3)
    ] + [[0.0, 0.0, 0.0, 1.0]]


def camera_matrix_to_unreal(m):
    r = multiply([[S[i] * m[i][j] for j in range(3)] for i in range(3)], CAMERA_AXES)
    return [r[i] + [100.0 * S[i] * m[i][3]] for i in range(3)] + [[0.0, 0.0, 0.0, 1.0]]


def rotation_matrix(roll, pitch, yaw):
    """Unreal Rotator: Rz(yaw) @ Ry(-pitch) @ Rx(-roll), column vectors."""
    r, p, y = map(math.radians, [roll, pitch, yaw])
    sr, cr = math.sin(r), math.cos(r)
    sp, cp = math.sin(p), math.cos(p)
    sy, cy = math.sin(y), math.cos(y)
    return [
        [cp * cy, sr * sp * cy - cr * sy, -cr * sp * cy - sr * sy],
        [cp * sy, sr * sp * sy + cr * cy, sr * cy - cr * sp * sy],
        [sp, -sr * cp, cr * cp],
    ]


def decompose(m):
    scale = [math.sqrt(sum(m[i][j] ** 2 for i in range(3))) for j in range(3)]
    if min(scale) < 1e-9:
        raise ValueError("Degenerate actor scale")
    r = [[m[i][j] / scale[j] for j in range(3)] for i in range(3)]
    pitch = math.asin(max(-1.0, min(1.0, r[2][0])))
    if abs(math.cos(pitch)) > 1e-7:
        roll = math.atan2(-r[2][1], r[2][2])
        yaw = math.atan2(r[1][0], r[0][0])
    else:
        roll = 0.0
        yaw = math.atan2(-r[0][1], r[1][1])
    return dict(
        location_cm=[m[i][3] for i in range(3)],
        rotation_rpy_deg=list(map(math.degrees, [roll, pitch, yaw])),
        scale=scale,
    )


def compose(pose):
    r = rotation_matrix(*pose["rotation_rpy_deg"])
    s = pose["scale"]
    p = pose["location_cm"]
    return [[r[i][j] * s[j] for j in range(3)] + [p[i]] for i in range(3)] + [
        [0.0, 0.0, 0.0, 1.0]
    ]


def unwrap_poses(poses):
    previous = None
    for p in poses:
        values = p["rotation_rpy_deg"]
        if previous is not None:
            for i in range(3):
                values[i] += 360 * round((previous[i] - values[i]) / 360)
        previous = values[:]
    return poses


def project_unreal(point, camera_matrix, K):
    delta = [point[i] - camera_matrix[i][3] for i in range(3)]
    local = [sum(camera_matrix[i][j] * delta[i] for i in range(3)) for j in range(3)]
    if local[0] <= 1e-6:
        return None
    return [
        K[0][0] * local[1] / local[0] + K[0][2],
        K[1][2] - K[1][1] * local[2] / local[0],
    ]
