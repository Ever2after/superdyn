"""Copy Epic template content from a locally licensed UE installation, with hashes."""

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bridge import engine_binary
from superdyn_ue.io_utils import sha256, save_json


def import_assets(engine=None):
    executable = engine_binary(engine)
    if executable is None:
        raise RuntimeError("A locally installed, licensed Unreal Engine is required")
    engine_root = executable.parents[3]
    resources = engine_root / "Templates/TemplateResources"
    packs = [
        ("High/Characters/Content/Mannequins", "Characters/Mannequins"),
        ("Standard/Vehicles/Content/OffroadCar", "Vehicles/OffroadCar"),
    ]
    files = {}
    pending = []
    for source_name, destination_name in packs:
        source = resources / source_name
        if not source.is_dir():
            raise RuntimeError("Official template content is missing: " + str(source))
        for item in sorted(source.rglob("*")):
            # The engine's source FBXs are unnecessary for native rendering.
            if not item.is_file() or item.suffix not in {".uasset", ".ubulk", ".uexp"}:
                continue
            relative = Path("Content") / destination_name / item.relative_to(source)
            destination = ROOT / relative
            digest = sha256(item)
            if destination.exists() and sha256(destination) != digest:
                raise RuntimeError("Refusing to replace modified content: " + str(relative))
            files[str(relative)] = digest
            pending.append((item, destination))
    for source, destination in pending:
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    receipt = dict(
        schema=1,
        publisher="Epic Games",
        source="Official Unreal Engine installation / Templates/TemplateResources",
        engine_version=json.loads((engine_root / "Engine/Build/Build.version").read_text()),
        source_packs=[dict(source=a, destination=b) for a, b in packs],
        license="Unreal Engine EULA; template content classified as Examples under section 1",
        license_url="https://www.unrealengine.com/eula/unreal",
        documentation=[
            "https://dev.epicgames.com/documentation/unreal-engine/third-person-template-in-unreal-engine",
            "https://dev.epicgames.com/documentation/unreal-engine/vehicle-template-in-unreal-engine",
        ],
        intended_use="Rendered synthetic images for non-generative object tracking/detection training and evaluation",
        files=files,
    )
    save_json(ROOT / "inputs/official_assets/license_and_source.json", receipt)
    return dict(status="imported", files=len(files), publisher=receipt["publisher"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine")
    print(json.dumps(import_assets(parser.parse_args().engine), indent=2))
