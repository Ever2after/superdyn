"""Renderer-independent terrain queries and placement checks, in metres.

The source layout axes are used directly as Unreal X/Y; the packet explicitly
scales metres to centimetres. No Blender camera convention is involved here.
"""

import math


def ramp_blend(value):
    t = min(1.0, max(0.0, value))
    shoulder = 0.16
    if t < shoulder:
        value = t / 2 - shoulder / (2 * math.pi) * math.sin(math.pi * t / shoulder)
    elif t > 1 - shoulder:
        q = 1 - t
        value = 1 - shoulder - (q / 2 - shoulder / (2 * math.pi) * math.sin(math.pi * q / shoulder))
    else:
        value = t - shoulder / 2
    return value / (1 - shoulder)


class Ground:
    """Query the exact triangle diagonal used by terrain.layout."""

    def __init__(self, layout):
        self.layout = layout
        self.bounds = layout["bounds"]
        self.step = layout["grid_step"]

    def patch(self, x, y):
        selected = {"height": 0}
        for patch in self.layout["plateaus"] + self.layout["ramps"]:
            a, b, c, d = patch["bounds"]
            if a <= x < b and c <= y < d:
                selected = patch
        return selected

    @staticmethod
    def patch_height(patch, x, y):
        if "axis" not in patch:
            return float(patch["height"])
        a, b, c, d = patch["bounds"]
        u = (x - a) / (b - a) if patch["axis"] == "x" else (y - c) / (d - c)
        return patch["z0"] + (patch["z1"] - patch["z0"]) * ramp_blend(u)

    def z(self, x, y):
        x0, x1, y0, y1 = self.bounds
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            raise ValueError("Outside family terrain")
        step = self.step
        i = min(round((x1 - x0) / step) - 1, max(0, math.floor((x - x0) / step)))
        j = min(round((y1 - y0) / step) - 1, max(0, math.floor((y - y0) / step)))
        a, b = x0 + i * step, y0 + j * step
        patch = self.patch(a + step / 2, b + step / 2)
        u, v = (x - a) / step, (y - b) / step
        z00 = self.patch_height(patch, a, b)
        z10 = self.patch_height(patch, a + step, b)
        z11 = self.patch_height(patch, a + step, b + step)
        z01 = self.patch_height(patch, a, b + step)
        return z00 * (1 - u) + z10 * (u - v) + z11 * v if u >= v else z00 * (1 - v) + z11 * u + z01 * (v - u)

    def context_z(self, x, y):
        a, b, c, d = self.bounds
        xx, yy = min(b - 1e-6, max(a + 1e-6, x)), min(d - 1e-6, max(c + 1e-6, y))
        distance = max(abs(x - xx), abs(y - yy))
        return self.z(xx, yy) * max(0, 1 - distance / 8) ** 2


def overlaps(a, b, margin=0.2):
    """Axis-aligned footprints: center X/Y, half width/depth in metres."""
    return abs(a[0] - b[0]) < a[2] + b[2] + margin and abs(a[1] - b[1]) < a[3] + b[3] + margin


def fits(ground, xy, half, placed, reserved, *, tree=False):
    x, y = xy
    hx, hy = (0.4, 0.4) if tree else half
    a, b, c, d = ground.bounds
    if x - hx < a + 0.2 or x + hx > b - 0.2 or y - hy < c + 0.2 or y + hy > d - 0.2:
        return None
    box = (x, y, hx, hy)
    if any(overlaps(box, p) for p in placed):
        return None
    if any(abs(x - p[0]) < hx + 0.8 and abs(y - p[1]) < hy + 0.8 for p in reserved):
        return None
    heights = [ground.z(x + ox, y + oy) for ox in (-hx, 0, hx) for oy in (-hy, 0, hy)]
    if max(heights) - min(heights) > (0.16 if tree else 0.08):
        return None
    return max(heights)
