"""Deterministic theme presets; all distances are Unreal centimetres."""

PRESETS = {
    "industrial": dict(
        title="Industrial test yard", surface="Asphalt012", wall="Concrete033", tile_cm=220,
        wall_clearance_above_highest_cm=100.0, wall_thickness_cm=30.0,
        description="Asphalt terraces; level wall tops 1 m above the highest ground. Poly Haven Pure Sky HDRI; no background props.",
    ),
    "park": dict(
        title="Terraced garden", surface="Grass004", wall="Concrete033", tile_cm=140,
        wall_clearance_above_highest_cm=100.0, wall_thickness_cm=30.0,
        description="Grass terraces; level wall tops 1 m above the highest ground. Poly Haven Pure Sky HDRI; no trees or background props.",
    ),
    "urban": dict(
        title="Urban pedestrian plaza", surface="PavingStones046", wall="Concrete033", tile_cm=240,
        wall_clearance_above_highest_cm=100.0, wall_thickness_cm=30.0,
        description="Stone paving; level wall tops 1 m above the highest ground. Poly Haven Pure Sky HDRI; no buildings or background props.",
    ),
}


def preset(name):
    if name not in PRESETS:
        raise ValueError("Unknown theme: " + str(name))
    return dict(PRESETS[name])
