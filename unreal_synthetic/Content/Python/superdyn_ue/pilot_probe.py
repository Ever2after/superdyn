"""Introspect editor APIs (names only) for pilot development."""

import json
from pathlib import Path

import unreal


def steps(request, root, render_jobs):
    def probe():
        names = {}
        for cls in [unreal.EditorLevelUtils, unreal.LevelEditorSubsystem, unreal.EditorLoadingAndSavingUtils, unreal.EditorAssetLibrary, unreal.World]:
            names[cls.__name__] = sorted(n for n in dir(cls) if any(k in n.lower() for k in ["current", "save", "level", "persistent"]))
        Path(request["status_dir"], "probe.json").write_text(json.dumps(names, indent=1))
        return 2
    return [("probe", probe)]
