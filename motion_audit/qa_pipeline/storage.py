"""Atomic JSON/CSV I/O and shared cached-geometry access per worker."""

import json, csv
from pathlib import Path
from .config import ROOT
import numpy as np
from PIL import Image

SOURCE = ROOT / "outputs/cache"
OUT = ROOT / "outputs/qa"


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def read(path):
    return json.loads(path.read_text())


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def load(vid):
    with np.load(SOURCE / "cache" / vid / "predictions.npz") as f:
        p = {k: f[k] for k in f.files}
    r = read(SOURCE / "results" / (vid + ".json"))
    assert p["depth"].shape == p["conf"].shape == p["masks"].shape
    assert len(p["times"]) == 9 and np.all(np.diff(p["times"]) > 0)
    return p, r


def aligned_rgb(vid, i, h, w):
    im = Image.open(SOURCE / "cache" / vid / f"{i * 4:04d}.png").convert("RGB")
    ow, oh = im.size
    if oh / ow < 0.5:
        cw = min(ow, round(oh / 0.5))
        left = (ow - cw) // 2
        im = im.crop((left, 0, left + cw, oh))
    elif oh / ow > 2:
        ch = min(oh, round(ow * 2))
        top = (oh - ch) // 2
        im = im.crop((0, top, ow, top + ch))
    return np.array(im.resize((w, h), Image.Resampling.BICUBIC))


def world_grid(depth, k, e):
    h, w = depth.shape
    y, x = np.mgrid[:h, :w]
    pix = np.stack([x, y, np.ones_like(x)], -1)
    cam = (pix @ np.linalg.inv(k).T) * depth[..., None]
    return (cam - e[:3, 3]) @ e[:3, :3]


def box_mask(box, h, w):
    mask = np.zeros((h, w), dtype=bool)
    if box is not None:
        x0, y0 = np.floor(np.array(box[:2]) / 1000 * [w, h]).astype(int)
        x1, y1 = np.ceil(np.array(box[2:]) / 1000 * [w, h]).astype(int)
        mask[max(0, y0) : min(h, y1), max(0, x0) : min(w, x1)] = True
    return mask


def reference(i, r, p):
    return dict(
        frame_index=i,
        released_frame_index=r["selected_frame_indices"][i],
        released_frame_name=r["selected_frame_names"][i],
        normalized_time=float(p["times"][i]),
    )
