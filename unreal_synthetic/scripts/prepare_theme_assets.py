"""Fetch four CC0 PBR sets and copy official environment template content."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bridge import engine_binary
from superdyn_ue.io_utils import sha256, save_json

ASSETS = ["Concrete033", "Asphalt012", "Grass004", "PavingStones046"]


def fetch(name):
    directory = ROOT / "inputs/themes/textures" / name
    directory.mkdir(parents=True, exist_ok=True)
    archive = ROOT / ".downloads" / (name + "_1K-JPG.zip")
    url = "https://ambientcg.com/get?file=" + archive.name
    if not archive.exists():
        request = urllib.request.Request(url, headers={"User-Agent": "SuperDyn-Unreal-asset-preparation/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            data = response.read()
        temporary = archive.with_suffix(".tmp")
        temporary.write_bytes(data)
        with zipfile.ZipFile(temporary) as package:
            if package.testzip() is not None:
                raise RuntimeError("Corrupt texture archive: " + name)
        temporary.rename(archive)
    files = {}
    with zipfile.ZipFile(archive) as package:
        for item in package.infolist():
            filename = Path(item.filename).name
            if filename.endswith(("_Color.jpg", "_Roughness.jpg", "_NormalDX.jpg")) or filename.lower().endswith(".txt"):
                destination = directory / filename
                destination.write_bytes(package.read(item))
                files[str(destination.relative_to(ROOT))] = sha256(destination)
    return dict(id=name, publisher="ambientCG / Lennart Demes", license="CC0-1.0",
                source="https://ambientcg.com/view?id=" + name, download=url,
                license_url="https://docs.ambientcg.com/license/", archive_sha256=sha256(archive), files=files)


def main():
    (ROOT / ".downloads").mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as workers:
        textures = list(workers.map(fetch, ASSETS))
    executable = engine_binary()
    if executable is None:
        raise RuntimeError("Official Unreal installation required")
    resources = executable.parents[3] / "Templates/TemplateResources"
    template_files = {}
    for source_name, destination_name in [
        ("Standard/ArchVis/Content", "ArchVis"),
        ("Standard/SimSandbox/Content", "SimSandbox"),
    ]:
        source = resources / source_name
        for file in source.rglob("*.uasset"):
            destination = ROOT / "Content" / destination_name / file.relative_to(source)
            digest = sha256(file)
            if destination.exists() and sha256(destination) != digest:
                raise RuntimeError("Modified template asset: " + str(destination))
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copyfile(file, destination)
            template_files[str(destination.relative_to(ROOT))] = digest
    save_json(ROOT / "inputs/themes/asset_sources.json", dict(
        textures=textures,
        templates=dict(publisher="Epic Games", source="Official UE 5.8.4 Templates/TemplateResources",
                       license_url="https://www.unrealengine.com/eula/unreal", files=template_files),
    ))
    print(json.dumps(dict(texture_sets=len(textures), template_assets=len(template_files))))


if __name__ == "__main__":
    main()
