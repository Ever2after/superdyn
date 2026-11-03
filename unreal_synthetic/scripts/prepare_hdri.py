"""Download a Poly Haven Pure Sky HDRI with integrity and lighting metadata."""

import hashlib
import json
import math
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Content/Python"))
from superdyn_ue.io_utils import save_json, sha256

ASSET = "kloofendal_48d_partly_cloudy_puresky"


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "SuperDyn-Unreal/1.0 (HDRI scene setup)"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def main():
    import cv2
    import numpy as np

    directory = ROOT / "inputs/lighting/polyhaven"
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for endpoint in ["info", "files"]:
        path = directory / (ASSET + "_" + endpoint + ".json")
        if not path.exists():
            path.write_bytes(fetch("https://api.polyhaven.com/" + endpoint + "/" + ASSET))
        metadata[endpoint] = json.loads(path.read_text())
    entry = metadata["files"]["hdri"]["4k"]["hdr"]
    target = directory / (ASSET + "_4k.hdr")
    if not target.exists():
        data = fetch(entry["url"])
        if len(data) != entry["size"] or hashlib.md5(data).hexdigest() != entry["md5"]:
            raise RuntimeError("Poly Haven HDRI integrity check failed")
        temporary = target.with_suffix(".tmp")
        temporary.write_bytes(data)
        temporary.rename(target)
    raw = target.read_bytes()
    if hashlib.md5(raw).hexdigest() != entry["md5"]:
        raise RuntimeError("Existing HDRI differs from official metadata")
    bgr = cv2.imread(str(target), cv2.IMREAD_UNCHANGED)
    if bgr is None or bgr.dtype != np.float32 or bgr.shape[1] != 4096:
        raise RuntimeError("Expected a readable 4K floating-point HDR image")
    luminance = bgr @ np.array([.0722,.7152,.2126], dtype=np.float32)
    height,width = luminance.shape
    upper = luminance[:height//2]
    peak_y,peak_x = np.unravel_index(upper.argmax(), upper.shape)
    theta = ((peak_x+.5)/width*2-1)*math.pi
    phi = (peak_y+.5)/height*math.pi
    direction = [math.sin(phi)*math.sin(theta), -math.sin(phi)*math.cos(theta), math.cos(phi)]
    weights = np.sin((np.arange(height)+.5)/height*math.pi)[:,None]
    mean = float((luminance*weights).sum()/(weights.sum()*width))
    median = float(np.median(upper))
    sun_yaw = math.degrees(math.atan2(direction[1],direction[0]))
    elevation = math.degrees(math.asin(direction[2]))
    receipt = dict(
        asset=ASSET, title=metadata["info"]["name"], authors=metadata["info"]["authors"],
        publisher="Poly Haven", source="https://polyhaven.com/a/"+ASSET,
        license="CC0-1.0", license_url="https://polyhaven.com/license",
        api_terms_url="https://polyhaven.com/our-api", credit="Powered by Poly Haven",
        download_url=entry["url"], relative_path=str(target.relative_to(ROOT)),
        bytes=len(raw), md5=entry["md5"], sha256=sha256(target), resolution=[width,height],
        purpose="Pure sky for visible sky dome, skylight illumination and reflections; no exterior scenery",
        luminance=dict(upper_median=median, spherical_mean=mean, maximum=float(upper.max())),
        background_multiplier=.28/max(median,1e-6), skylight_intensity=.55/max(mean,1e-6),
        sun_direction_world=direction,
        directional_light_rotation=dict(pitch=-elevation,yaw=(sun_yaw+360)%360-180,roll=0),
        whitebalance_kelvin=metadata["info"].get("whitebalance",5400),
        basis="Unreal long-lat cubemap import: sample-space direction=(sin(phi)*sin(theta),-sin(phi)*cos(theta),cos(phi))",
    )
    save_json(directory / "license_and_source.json",receipt)
    print(json.dumps(receipt,indent=2))


if __name__ == "__main__":
    main()
