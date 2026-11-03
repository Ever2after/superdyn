"""Place stage-local clip designs on a surveyed map (2.5D raster checks).

The raster stores, per column, the successive collision surfaces below a
vertical ray. Placement requires walkable clearance for every actor/camera
sample, flat support, and an unblocked 2.5D line of sight to each target at
every measurement frame. Final occlusion is decided by the rendered GT masks.
"""

import gzip
import json
import math

import numpy as np
from scipy import ndimage

from pilot_design import ASSETS, MEASURE


class Survey:
    def __init__(self, path, exclude=(), paved=()):
        info = json.load(gzip.open(path, "rt"))
        grid = info["raster"]
        ny, nx = grid["shape"]
        self.path = str(path)
        self.step = grid["step_cm"] / 100.0
        self.x0, self.y0 = (v / 100.0 for v in grid["origin_cm"])
        self.top = np.full((ny, nx), np.nan)
        self.normal = np.zeros((ny, nx))
        self.label = np.full((ny, nx), "", dtype=object)
        for j, row in enumerate(grid["columns"]):
            for i, col in enumerate(row):
                if col:
                    self.top[j, i] = col[0][0] / 100.0
                    self.normal[j, i] = col[0][1]
                    self.label[j, i] = col[0][2]
        starts = lambda prefixes: np.vectorize(lambda v: any(v.startswith(x) for x in prefixes), otypes=[bool])(self.label) if prefixes else np.zeros((ny, nx), bool)
        self.excluded = starts(exclude)
        self.paved = starts(paved)
        self.top[self.excluded] = np.nan
        finite = np.isfinite(self.top)
        filled = np.where(finite, self.top, np.nanmax(self.top) + 50)
        floor = ndimage.minimum_filter(filled, size=3)
        # A cell is ground if it is not resting on something; distance to
        # obstacles is handled separately by the clearance transform.
        self.walkable = finite & (self.normal > 0.85) & (self.top - floor < 0.3)
        # Structure: anything standing well above the local floor (trees, buildings, props).
        local_floor = ndimage.minimum_filter(filled, size=9)
        self.structure = finite & (self.top - local_floor > 1.5)
        self.clearance = ndimage.distance_transform_edt(self.walkable) * self.step
        self.info = {k: v for k, v in info.items() if k != "raster"}

    def index(self, x, y):
        i = np.floor((np.asarray(x) - self.x0) / self.step + 0.5).astype(int)
        j = np.floor((np.asarray(y) - self.y0) / self.step + 0.5).astype(int)
        inside = (i >= 0) & (j >= 0) & (i < self.top.shape[1]) & (j < self.top.shape[0])
        return np.clip(j, 0, self.top.shape[0] - 1), np.clip(i, 0, self.top.shape[1] - 1), inside


def to_world(origin, yaw_deg, f, r):
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return origin[0] + np.asarray(f) * c - np.asarray(r) * s, origin[1] + np.asarray(f) * s + np.asarray(r) * c


def radius(spec):
    a = ASSETS[spec["asset"]]
    return 0.45 if a["category"] == "person" else max(0.35, a.get("length_m", 1.0) * 0.5)


def target_height(spec):
    a = ASSETS[spec["asset"]]
    return a.get("height_m", a.get("length_m", 1.0) * 0.55) * 0.55


def evaluate(survey, clip, origin, yaw):
    """None if infeasible, else (score, details)."""
    cam = clip["cam"]
    frames = list(range(0, len(cam["f"]), 4))
    checks = [(cam["f"][frames], cam["r"][frames], 0.6)]
    for spec in clip["actors"]:
        if spec["z_mode"] == "ground":
            checks.append((spec["f"][frames], spec["r"][frames], radius(spec)))
    z0 = None
    for f, r, rad in checks:
        x, y = to_world(origin, yaw, f, r)
        j, i, inside = survey.index(x, y)
        if not inside.all() or (survey.clearance[j, i] < rad).any():
            return None
        z = survey.top[j, i]
        z0 = z[0] if z0 is None else z0
        if np.abs(z - z0).max() > 0.6:
            return None
    # 2.5D line of sight from the camera to each labelled target.
    cx, cy = to_world(origin, yaw, cam["f"], cam["r"])
    for spec in clip["actors"]:
        if not spec["label"]:
            continue
        tx, ty = to_world(origin, yaw, spec["f"], spec["r"])
        for k in MEASURE:
            ch = z0 + cam["h"][k]
            th = z0 + (spec["h"][k] if spec["h"] is not None else target_height(spec))
            dist = math.hypot(tx[k] - cx[k], ty[k] - cy[k])
            n = max(2, int(dist / (survey.step * 0.5)))
            u = np.linspace(0, 1, n)[1:-1]
            u = u[(u * dist > 1.0) & ((1 - u) * dist > 1.0)]
            j, i, inside = survey.index(cx[k] + u * (tx[k] - cx[k]), cy[k] + u * (ty[k] - cy[k]))
            line = ch + u * (th - ch)
            top = np.where(inside, survey.top[j, i], -1e9)
            if (np.nan_to_num(top, nan=-1e9) > line + 0.1).any():
                return None
    # Background richness inside the initial view: standing structure 8-45 m away.
    c, s = math.cos(math.radians(yaw + cam["yaw"][0])), math.sin(math.radians(yaw + cam["yaw"][0]))
    px, py = cx[0], cy[0]
    jj, ii = np.mgrid[0:survey.top.shape[0], 0:survey.top.shape[1]]
    dx, dy = survey.x0 + ii * survey.step - px, survey.y0 + jj * survey.step - py
    along = dx * c + dy * s
    across = -dx * s + dy * c
    half = math.tan(math.radians(cam["fov_deg"] / 2))
    view = (along > 8) & (along < 45) & (np.abs(across) < along * half)
    if view.sum() == 0:
        return None
    void = (~np.isfinite(survey.top) & ~survey.excluded & view).sum() / view.sum()
    rich = (survey.structure & view).sum() / view.sum()
    j, i, _ = survey.index(*to_world(origin, yaw, clip["cam"]["f"][::8], clip["cam"]["r"][::8]))
    paved = float(survey.paved[j, i].mean())
    return rich - 2.0 * void + 0.3 * paved, dict(ground_z_m=float(z0), view_structure=float(rich), view_void=float(void), stage_paved=paved)


def windows(survey, size=70.0, count=4, separation=80.0):
    """Detail-survey windows (cm bounds) centred on paved, structured ground."""
    finite = np.isfinite(survey.top)
    radius = max(1, int(round(size / 2 / survey.step)))
    kernel = np.ones((2 * radius + 1, 2 * radius + 1))
    score = ndimage.convolve(survey.paved.astype(float), kernel, mode="constant") * 2 + ndimage.convolve(survey.structure.astype(float), kernel, mode="constant")
    score[~(finite & survey.walkable)] = -1
    chosen = []
    for flat in np.argsort(score, axis=None)[::-1]:
        j, i = np.unravel_index(flat, score.shape)
        if score[j, i] <= 0 or len(chosen) == count:
            break
        x, y = survey.x0 + i * survey.step, survey.y0 + j * survey.step
        if all(math.dist((x, y), c) >= separation for c in chosen):
            chosen.append((x, y))
    return [[100 * (x - size / 2), 100 * (x + size / 2), 100 * (y - size / 2), 100 * (y + size / 2)] for x, y in chosen]


def place_any(surveys, clips, avoid=(), seed=0):
    """Best placement over several detail windows of one environment."""
    best, errors = None, []
    for k, survey in enumerate(surveys):
        try:
            stage = place(survey, clips, avoid=avoid, seed=seed)
        except RuntimeError as error:
            errors.append(str(error))
            continue
        if best is None or stage["score"] > best["score"]:
            best = dict(stage, window=k, survey=survey.path)
    if best is None:
        raise RuntimeError("; ".join(errors[:2]))
    return best


def place(survey, clips, avoid=(), spacing=12.0, step=2.0, yaw_step=15, seed=0):
    """Joint placement: counterfactual pairs share one origin and yaw."""
    rng = np.random.default_rng(seed)
    js, is_ = np.nonzero(survey.walkable & (survey.clearance >= 1.0))
    if len(js) == 0:
        raise RuntimeError("Survey has no walkable cells")
    pts = np.c_[survey.x0 + is_ * survey.step, survey.y0 + js * survey.step]
    keep = np.unique(np.floor(pts / step).astype(int), axis=0, return_index=True)[1]
    pts = pts[keep]
    rng.shuffle(pts)
    best = None
    for origin in pts[:4000]:
        if any(math.dist(origin, a) < spacing for a in avoid):
            continue
        for yaw in range(0, 360, yaw_step):
            results = [evaluate(survey, clip, origin, yaw) for clip in clips]
            if all(results):
                score = min(r[0] for r in results)
                if best is None or score > best[0]:
                    best = (score, [float(origin[0]), float(origin[1])], float(yaw), results[0][1])
    if best is None:
        raise RuntimeError("No feasible stage for " + ",".join(c["id"] for c in clips))
    return dict(origin_m=best[1], yaw_deg=best[2], score=best[0], **best[3])
